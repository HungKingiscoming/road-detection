import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from modeling.bridge import compute_topo_loss
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.coanet import CoANet
# Import kiến trúc CoANet + TopoNet và Loss
from modeling.coanet_topo import CoANetWithTopo, TopoConfig
from utils.loss import CoANetLoss, TopoLoss
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def compute_grad_norm(model, norm_type: float = 2.0) -> float:
    """Tính tổng grad-norm của toàn bộ parameters có requires_grad và grad != None.
    Dùng để phát hiện gradient exploding (norm tăng đột biến) hoặc vanishing (norm ~0)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def positive_ratio(x) -> float:
    """Tỉ lệ pixel/giá trị > 0 (hoặc == 1) trong tensor/array. Dùng để theo dõi
    mất cân bằng lớp và phát hiện hiện tượng model 'collapse' (dự đoán toàn 0 hoặc toàn 1)."""
    if hasattr(x, 'float'):
        return x.float().mean().item()
    return float(np.mean(x))


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 1. Khởi tạo Saver & Tensorboard
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # 2. Khởi tạo Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        topo_config = TopoConfig(
            max_points=args.max_points,
            k_neighbors=args.k_neighbors,
            graph_mask_score_threshold=args.topo_score_thresh,
            coord_format='pixel'
        )
        base_coanet = CoANet(
            backbone=args.backbone,
            output_stride=args.out_stride,
            num_classes=self.nclass,
            freeze_bn=args.freeze_bn
        )
        model = CoANetWithTopo(
            backbone=args.backbone,
            num_classes=1,
            num_neighbor=9,
            sync_bn=args.sync_bn,
            convnext_model_name='convnextv2_tiny.fcmae_ft_in22k_in1k' if 'convnext' in args.backbone else None,
            pretrained=True
        )
        
        # 4. Cấu hình Optimizer
        train_params = [
            {'params': model.coanet.parameters(), 'lr': args.lr},
            {'params': model.topo_head.parameters(), 'lr': args.lr}
        ]
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.05
        )

        # 5. Khởi tạo Loss Function
        self.coanet_loss_fn = CoANetLoss(
            seg_loss_weight=1.0,
            connect_loss_weight=0.12,     
            connect_d1_loss_weight=0.08   
        )
        self.topo_loss_fn = TopoLoss(pos_weight=2.0)

        self.model, self.optimizer = model, optimizer

        # 6. Khởi tạo Evaluator & Scheduler
        self.evaluator = Evaluator(num_class=2)  # Binary Segmentation (Background / Road)
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, 
            args.lr, 
            args.epochs, 
            len(self.train_loader)
        )

        # 7. Cấu hình GPU (CUDA & DataParallel)
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # 8. Load Checkpoint (Resume)
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> Không tìm thấy checkpoint tại '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            # Restore state_dict
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(f"=> Đã load checkpoint '{args.resume}' (Epoch {checkpoint['epoch']})")

        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        train_coanet_loss = 0.0
        train_topo_loss = 0.0

        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            # Lấy dữ liệu từ DataLoader
            image = sample['image']               # [B, 3, H, W]
            gt_mask = sample['gt_mask']           # [B, 1, H, W]
            gt_connect = sample['gt_connect_d1']  # [B, 9, H, W]
            gt_connect_d1 = sample['gt_connect_d3']# [B, 9, H, W]

            if self.args.cuda:
                image = image.cuda(non_blocking=True)
                gt_mask = gt_mask.cuda(non_blocking=True)
                gt_connect = gt_connect.cuda(non_blocking=True)
                gt_connect_d1 = gt_connect_d1.cuda(non_blocking=True)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            # Forward qua CoANetWithTopo
            out_dict = self.model(image, gt_mask=gt_mask, return_aux=True)

            fused_mask = out_dict['fused_mask']       # [B, 1, H, W]
            seg_logits = out_dict['seg_logits']       # [B, 1, H, W]
            connect = out_dict['connect']             # [B, 9, H, W]
            connect_d1 = out_dict['connect_d1']       # [B, 9, H, W]
            aux_seg = out_dict.get('aux_seg', None)   # [B, 1, H, W]
            topo_out = out_dict.get('topo', {})       # Dictionary chứa kết quả từ TopoNet
            
            # 1. Tính CoANet Segmentation & Connectivity Loss
            c_loss, c_loss_dict = self.coanet_loss_fn(
                seg=seg_logits,
                connect=connect,
                connect_d1=connect_d1,
                aux_seg=aux_seg,
                gt_seg=gt_mask,
                gt_connect=gt_connect,
                gt_connect_d1=gt_connect_d1
            )
            
            # 2. Tính TopoNet Loss từ dictionary topo_out
            t_loss = torch.tensor(0.0, device=image.device)
            if 'edge_gt' in topo_out and topo_out['edge_gt'] is not None:
                t_loss = compute_topo_loss(
                    logits=topo_out['logits'],
                    edge_gt=topo_out['edge_gt'],
                    pairs_valid=topo_out['pairs_valid']
                )
            
            # Tổng Loss
            total_loss = c_loss + self.args.topo_weight * t_loss

            total_loss.backward()

            # --- Debug: Grad-norm TRƯỚC khi optimizer.step() (norm bằng 0/NaN/Inf là dấu hiệu bug) ---
            grad_norm = compute_grad_norm(self.model)

            self.optimizer.step()

            # Tích lũy Log Loss
            train_loss += total_loss.item()
            train_coanet_loss += c_loss.item()
            train_topo_loss += t_loss.item()

            global_step = i + num_img_tr * epoch
            current_lr = self.optimizer.param_groups[0]['lr']

            tbar.set_description(
                f'Loss: {train_loss/(i+1):.4f} | CoANet: {train_coanet_loss/(i+1):.4f} | Topo: {train_topo_loss/(i+1):.4f}'
            )
            self.writer.add_scalar('train/total_loss_iter', total_loss.item(), global_step)
            self.writer.add_scalar('train/lr_iter', current_lr, global_step)
            self.writer.add_scalar('train/grad_norm_iter', grad_norm, global_step)

            # --- Debug: log định kỳ (không log mọi iter để đỡ nặng I/O) ---
            if global_step % self.args.log_interval == 0:
                # 2a. Log từng thành phần loss con của CoANet (seg / connect / connect_d1 / aux...)
                if isinstance(c_loss_dict, dict):
                    for k, v in c_loss_dict.items():
                        v_val = v.item() if torch.is_tensor(v) else v
                        self.writer.add_scalar(f'train/loss_{k}', v_val, global_step)

                # 2b. Cảnh báo sớm nếu loss/grad bất thường (NaN, Inf, hoặc grad "chết")
                if not np.isfinite(total_loss.item()):
                    print(f"[CẢNH BÁO] total_loss không hợp lệ (NaN/Inf) tại step {global_step}: {total_loss.item()}")
                if grad_norm == 0.0:
                    print(f"[CẢNH BÁO] grad_norm = 0 tại step {global_step} — có thể gradient không chảy qua model (vanishing/branch bị detach nhầm).")
                elif not np.isfinite(grad_norm):
                    print(f"[CẢNH BÁO] grad_norm không hợp lệ (NaN/Inf) tại step {global_step}.")

                # 2c. Thống kê tỉ lệ pixel "dương" (road) của seg_logits so với GT
                #     -> phát hiện model bị "collapse" (dự đoán gần như toàn 0 hoặc toàn 1)
                with torch.no_grad():
                    seg_pos_ratio = positive_ratio(torch.sigmoid(seg_logits) > 0.5)
                    gt_pos_ratio = positive_ratio(gt_mask > 0.5)
                self.writer.add_scalar('train/seg_pred_positive_ratio', seg_pos_ratio, global_step)
                self.writer.add_scalar('train/gt_positive_ratio', gt_pos_ratio, global_step)

                # 2d. Thống kê nhánh TopoNet: số cạnh candidate hợp lệ, tỉ lệ cạnh dương (edge_gt),
                #     điểm số trung bình -> phát hiện trường hợp không có cạnh nào hợp lệ
                #     (topo loss luôn = 0, hoặc mất cân bằng lớp cực đoan giữa cạnh dương/âm).
                pairs_valid = topo_out.get('pairs_valid', None)
                if pairs_valid is not None:
                    n_valid_pairs = pairs_valid.float().sum().item()
                    self.writer.add_scalar('train/topo_num_valid_pairs', n_valid_pairs, global_step)
                    edge_gt = topo_out.get('edge_gt', None)
                    if edge_gt is not None and n_valid_pairs > 0:
                        edge_pos_ratio = (edge_gt * pairs_valid.float()).sum().item() / n_valid_pairs
                        self.writer.add_scalar('train/topo_edge_positive_ratio', edge_pos_ratio, global_step)
                    if 'scores' in topo_out:
                        mean_score = topo_out['scores'].detach().mean().item()
                        self.writer.add_scalar('train/topo_mean_score', mean_score, global_step)

                # 2e. In gọn ra console để dễ theo dõi khi chạy trên Kaggle/Colab (log không lưu tensorboard)
                tqdm.write(
                    f"[step {global_step}] lr={current_lr:.6f} grad_norm={grad_norm:.4f} "
                    f"seg_pos_ratio={seg_pos_ratio:.4f} gt_pos_ratio={gt_pos_ratio:.4f}"
                )

            # --- Đánh giá Metric trên Fused Mask đầu ra ---
            # Align kích thước Spatial (H, W) của fused_mask khớp với gt_mask trước khi nhị phân hóa
            if fused_mask.shape[-2:] != gt_mask.shape[-2:]:
                fused_mask_eval = F.interpolate(fused_mask, size=gt_mask.shape[-2:], mode='bilinear', align_corners=True)
            else:
                fused_mask_eval = fused_mask

            pred = (fused_mask_eval > 0.5).detach().cpu().numpy().astype(np.int64)
            target_n = gt_mask.detach().cpu().numpy().astype(np.int64)
            self.evaluator.add_batch(target_n, pred)

        # Tính Metrics sau 1 Epoch
        Acc = self.evaluator.Pixel_Accuracy()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()

        self.writer.add_scalar('train/total_loss_epoch', train_loss / num_img_tr, epoch)
        self.writer.add_scalar('train/coanet_loss_epoch', train_coanet_loss / num_img_tr, epoch)
        self.writer.add_scalar('train/topo_loss_epoch', train_topo_loss / num_img_tr, epoch)
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/Acc', Acc, epoch)
        self.writer.add_scalar('train/Precision', Precision, epoch)
        self.writer.add_scalar('train/Recall', Recall, epoch)
        self.writer.add_scalar('train/F1', F1, epoch)
        self.writer.add_scalar('train/lr_epoch', self.optimizer.param_groups[0]['lr'], epoch)

        print(f'\n--- Train Epoch {epoch} Results ---')
        print(f'mIoU: {mIoU:.4f} | Precision: {Precision:.4f} | Recall: {Recall:.4f} | F1: {F1:.4f}')

        # --- Debug: cảnh báo sớm nếu Recall gần 1.0 nhưng Precision rất thấp
        # (dấu hiệu kinh điển của việc model/metric dự đoán gần như toàn bộ ảnh là "road")
        if Recall > 0.98 and Precision < 0.2:
            print(f"[CẢNH BÁO] Recall={Recall:.4f} rất cao trong khi Precision={Precision:.4f} rất thấp "
                  f"-> khả năng cao model đang dự đoán gần như toàn bộ pixel là positive. "
                  f"Kiểm tra lại: (1) có bị sigmoid 2 lần trên xác suất không, "
                  f"(2) class imbalance / pos_weight trong loss, (3) ngưỡng threshold khi nhị phân hoá.")

        if self.args.no_val:
            is_best = False
            model_state = self.model.module.state_dict() if self.args.cuda else self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_state,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
        val_loss = 0.0
        num_img_val = len(self.val_loader)

        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image = sample['image']
                gt_mask = sample['gt_mask']
                gt_connect = sample['gt_connect_d1']
                gt_connect_d1 = sample['gt_connect_d3']

                if self.args.cuda:
                    image = image.cuda(non_blocking=True)
                    gt_mask = gt_mask.cuda(non_blocking=True)
                    gt_connect = gt_connect.cuda(non_blocking=True)
                    gt_connect_d1 = gt_connect_d1.cuda(non_blocking=True)

                out_dict = self.model(image)
                fused_mask = out_dict['fused_mask']
                seg_logits = out_dict['seg_logits']
                connect = out_dict['connect']
                connect_d1 = out_dict['connect_d1']

                c_loss, _ = self.coanet_loss_fn(
                    seg=seg_logits,
                    connect=connect,
                    connect_d1=connect_d1,
                    aux_seg=None,
                    gt_seg=gt_mask,
                    gt_connect=gt_connect,
                    gt_connect_d1=gt_connect_d1
                )

                val_loss += c_loss.item()
                tbar.set_description(f'Val Loss: {val_loss / (i + 1):.4f}')

                # Align kích thước Spatial (H, W) của fused_mask khớp với gt_mask
                if fused_mask.shape[-2:] != gt_mask.shape[-2:]:
                    fused_mask_eval = F.interpolate(fused_mask, size=gt_mask.shape[-2:], mode='bilinear', align_corners=True)
                else:
                    fused_mask_eval = fused_mask

                pred = (fused_mask_eval > 0.5).cpu().numpy().astype(np.int64)
                target_n = gt_mask.cpu().numpy().astype(np.int64)
                self.evaluator.add_batch(target_n, pred)

                # --- Debug: theo dõi tỉ lệ pixel dương của val prediction so với GT ---
                # Nếu pred_pos_ratio lệch rất xa gt_pos_ratio (vd luôn gần 1.0) -> model đang
                # collapse / có bug tương tự lỗi sigmoid-kép trước đây.
                if i % max(1, num_img_val // 10) == 0:
                    val_pred_pos_ratio = positive_ratio(pred)
                    val_gt_pos_ratio = positive_ratio(target_n)
                    self.writer.add_scalar('val/pred_positive_ratio', val_pred_pos_ratio, epoch * num_img_val + i)
                    self.writer.add_scalar('val/gt_positive_ratio', val_gt_pos_ratio, epoch * num_img_val + i)

                if i % max(1, num_img_val // 5) == 0:
                    self.summary.visualize_image(
                        self.writer, self.args.dataset, image, gt_mask, fused_mask_eval, i, split='Val'
                    )

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()

        self.writer.add_scalar('val/total_loss_epoch', val_loss / num_img_val, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Precision', Precision, epoch)
        self.writer.add_scalar('val/Recall', Recall, epoch)
        self.writer.add_scalar('val/F1', F1, epoch)

        print(f'\n--- Validation Epoch {epoch} Results ---')
        print(f'mIoU: {mIoU:.4f} | Precision: {Precision:.4f} | Recall: {Recall:.4f} | F1: {F1:.4f}')

        # Lưu Checkpoint nếu đạt mIoU tốt nhất
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            model_state = self.model.module.state_dict() if self.args.cuda else self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_state,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
            print(f"==> Đã lưu Best Checkpoint mới với mIoU: {self.best_pred:.4f}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch CoANet + TopoNet Training")
    parser.add_argument('--backbone', type=str, default='convnextv2_tiny',
                    choices=['gcnet', 'convnextv2_tiny', 'convnextv2_nano'],
                    help='Backbone architecture')
    # Model Hyperparams
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='spacenet', choices=['spacenet', 'DeepGlobe'])
    parser.add_argument('--workers', type=int, default=8, help='dataloader threads')
    parser.add_argument('--freeze-bn', action='store_true', default=False, help='freeze BN parameters')
    parser.add_argument('--data-dir', type=str, default=None, help='path to dataset root')
    parser.add_argument('--loss-type', type=str, default='ce', help='loss type tag for logging')
    parser.add_argument('--base-size', type=int, default=1024, help='base image size for preprocessing')
    parser.add_argument('--crop-size', type=int, default=512, help='crop size for preprocessing')

    # TopoNet Params
    parser.add_argument('--max-points', type=int, default=128, help='Số điểm nút tối đa trích xuất từ Skeleton')
    parser.add_argument('--k-neighbors', type=int, default=5, help='Số lân cận k-NN nối cạnh đồ thị')
    parser.add_argument('--topo-score-thresh', type=float, default=0.3, help='Ngưỡng lọc cạnh Topology')
    parser.add_argument('--topo-weight', type=float, default=0.5, help='Trọng số Topology Loss')

    # Training Hyperparams
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (Khuyên dùng 1e-4 cho ConvNeXt-V2)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')

    # CUDA & System
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3', help='comma-separated gpu ids')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--checkname', type=str, default=None, help='set checkpoint name')
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning mode')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation')
    parser.add_argument('--log-interval', type=int, default=20,
                         help='số iteration giữa mỗi lần log chi tiết (grad-norm, sub-loss, topo stats,...) vào tensorboard')

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = os.environ.get('DATA_DIR')
    args.loss_type = getattr(args, 'loss_type', 'ce')
    args.base_size = getattr(args, 'base_size', 1024)
    args.crop_size = getattr(args, 'crop_size', 512)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids phải là danh sách số nguyên phân cách bằng dấu phẩy!')

    if args.checkname is None:
        args.checkname = f'CoANetTopo-{args.backbone}'

    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = Trainer(args)
    print('Bắt đầu Epoch:', trainer.args.start_epoch)
    print('Tổng số Epochs:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
