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
from modeling.coanet_topo import CoANetWithTopo, TopoConfig
from utils.loss import CoANetLoss, TopoLoss
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import cv2

# Khống chế CPU thread contention giữa DataLoader workers
cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def print_freeze_status(model, epoch: int):
    """In ra trạng thái freeze/unfreeze của các module trong mô hình."""
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    print(f"\n==================================================")
    print(f"🔒 TRẠNG THÁI FREEZE/UNFREEZE - EPOCH {epoch}")
    print(f"==================================================")
    
    modules_to_check = {
        'Backbone': getattr(raw_model.coanet, 'backbone', None),
        'ASPP': getattr(raw_model.coanet, 'aspp', None),
        'Decoder': getattr(raw_model.coanet, 'decoder', None),
        'Connect': getattr(raw_model.coanet, 'connect', None),
        'Topo Head': getattr(raw_model, 'topo_head', None),
    }

    total_trainable_params = 0
    total_frozen_params = 0

    for name, module in modules_to_check.items():
        if module is None:
            continue
        
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        
        total_trainable_params += trainable
        total_frozen_params += frozen

        if trainable > 0 and frozen == 0:
            status = "🔓 UNFROZEN (Đang Train)"
        elif trainable == 0 and frozen > 0:
            status = "🔒 FROZEN (Đã Đóng Băng)"
        elif trainable > 0 and frozen > 0:
            status = "🌗 PARTIAL (Một phần đang train)"
        else:
            status = "❓ EMPTY (Không có tham số)"

        print(f" • {name:<12}: {status:<28} | Trainable: {trainable:,} | Frozen: {frozen:,}")

    print(f"--------------------------------------------------")
    print(f"📊 TỔNG THAM SỐ TRAINABLE : {total_trainable_params:,}")
    print(f"📊 TỔNG THAM SỐ FROZEN    : {total_frozen_params:,}")
    print(f"==================================================\n")


def load_coanet_weights_safely(coanet_model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load weights từ checkpoint cho CoANet và kiểm tra độ khớp (Key & Shape).
    """
    print(f"\n==================================================")
    print(f"🔍 BẮT ĐẦU KIỂM TRA & LOAD WEIGHTS TỪ: {checkpoint_path}")
    print(f"==================================================")
    
    if not os.path.isfile(checkpoint_path):
        print(f"⚠️ [CẢNH BÁO] Không tìm thấy file weights tại '{checkpoint_path}'. Bỏ qua bước load weights.")
        return coanet_model

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        if name.startswith('coanet.'):
            name = name.replace('coanet.', '')
        cleaned_state_dict[name] = v

    model_state_dict = coanet_model.state_dict()
    
    matched_keys = []
    shape_mismatched_keys = []
    missing_in_ckpt_keys = []

    module_stats = {
        'backbone': {'total': 0, 'matched': 0},
        'aspp': {'total': 0, 'matched': 0},
        'decoder': {'total': 0, 'matched': 0},
        'connect': {'total': 0, 'matched': 0},
    }

    filtered_state_dict = {}

    for name, param in model_state_dict.items():
        sub_module = name.split('.')[0] if '.' in name else 'other'
        if sub_module in module_stats:
            module_stats[sub_module]['total'] += 1

        if name in cleaned_state_dict:
            ckpt_param = cleaned_state_dict[name]
            if param.shape == ckpt_param.shape:
                filtered_state_dict[name] = ckpt_param
                matched_keys.append(name)
                if sub_module in module_stats:
                    module_stats[sub_module]['matched'] += 1
            else:
                shape_mismatched_keys.append((name, tuple(param.shape), tuple(ckpt_param.shape)))
        else:
            missing_in_ckpt_keys.append(name)

    total_model_keys = len(model_state_dict)
    total_matched = len(matched_keys)
    match_percentage = (total_matched / total_model_keys) * 100 if total_model_keys > 0 else 0

    print(f"\n📊 BÁO CÁO TỔNG QUAN TƯƠNG THÍCH WEIGHTS:")
    print(f" - Tổng số tham số trong CoANet:       {total_model_keys}")
    print(f" - Số tham số KHỚP HOÀN HẢO:           {total_matched} ({match_percentage:.2f}%)")
    print(f" - Số tham số LỆCH SHAPE:               {len(shape_mismatched_keys)}")
    print(f" - Số tham số THIẾU trong Checkpoint:  {len(missing_in_ckpt_keys)}")
    
    print(f"\n🧩 TỶ LỆ KHỚP THEO TỪNG MODULE CON:")
    for mod_name, stats in module_stats.items():
        if stats['total'] > 0:
            pct = (stats['matched'] / stats['total']) * 100
            print(f"  • {mod_name.upper():<10}: {stats['matched']}/{stats['total']} keys ({pct:.1f}%)")

    coanet_model.load_state_dict(filtered_state_dict, strict=False)
    print(f"\n✅ Đã load thành công {len(filtered_state_dict)} weights vào CoANet!")
    print(f"==================================================\n")
    
    return coanet_model


def compute_grad_norm(model, norm_type: float = 2.0) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    return total_norm ** (1.0 / norm_type)


def positive_ratio(x) -> float:
    if hasattr(x, 'float'):
        return x.float().mean().item()
    return float(np.mean(x))


def get_road_iou(evaluator: Evaluator) -> float:
    confusion_matrix = evaluator.confusion_matrix
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    if union[1] == 0:
        return 0.0
    
    road_iou = intersection[1] / union[1]
    return float(road_iou)


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        kwargs = {
            'num_workers': args.workers, 
            'pin_memory': True,
            'persistent_workers': True if args.workers > 0 else False
        }
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        base_coanet = CoANet(
            backbone=args.backbone,
            output_stride=args.out_stride,
            num_classes=self.nclass,
            freeze_bn=args.freeze_bn
        )

        if hasattr(args, 'coanet_weights') and args.coanet_weights is not None:
            base_coanet = load_coanet_weights_safely(base_coanet, args.coanet_weights)

        topo_config = TopoConfig(
            max_points=args.max_points,
            k_neighbors=args.k_neighbors,
            graph_mask_score_threshold=args.topo_score_thresh,
            coord_format='pixel'
        )
        model = CoANetWithTopo(
            coanet=base_coanet,
            topo_cfg=topo_config,
            decoder_feature_dim=64
        )
        
        self.coanet_loss_fn = CoANetLoss(
            seg_loss_weight=1.0,
            connect_loss_weight=0.12,     
            connect_d1_loss_weight=0.08   
        )
        self.topo_loss_fn = TopoLoss(pos_weight=2.0)
        self.model = model

        self.current_stage = 1 if args.freeze_coanet_epochs > 0 else 2
        self.optimizer = self.build_optimizer(stage=self.current_stage)

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.args.cuda)

        self.evaluator = Evaluator(num_class=2)
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, 
            args.lr, 
            args.epochs, 
            len(self.train_loader)
        )

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> Không tìm thấy checkpoint tại '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            args.start_epoch = checkpoint['epoch']

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

    def build_optimizer(self, stage: int = 2):
        raw_model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )

        if stage == 1:
            print(
                f'🔒 [Stage 1] Đóng băng CoANet. Chỉ tập trung huấn luyện TopoHead'
                f' trong {self.args.freeze_coanet_epochs} epoch đầu.'
            )
            if hasattr(raw_model, 'set_freeze_coanet'):
                raw_model.set_freeze_coanet(True)
            else:
                for param in raw_model.coanet.parameters():
                    param.requires_grad = False
                for param in raw_model.topo_head.parameters():
                    param.requires_grad = True

            train_params = filter(
                lambda p: p.requires_grad, raw_model.parameters()
            )
            return torch.optim.AdamW(
                train_params, lr=self.args.lr, weight_decay=self.args.weight_decay
            )

        else:
            print(
                '🔓 [Stage 2] Mở khóa toàn bộ mô hình (Chạy Fine-tune End-to-End'
                ' với Differential LR).'
            )
            if hasattr(raw_model, 'set_freeze_coanet'):
                raw_model.set_freeze_coanet(False)
            else:
                for param in raw_model.parameters():
                    param.requires_grad = True

            train_params = [
                {'params': raw_model.coanet.backbone.parameters(), 'lr': self.args.lr * 0.1},
                {'params': raw_model.coanet.aspp.parameters(), 'lr': self.args.lr * 0.5},
                {'params': raw_model.coanet.decoder.parameters(), 'lr': self.args.lr * 0.5},
                {'params': raw_model.coanet.connect.parameters(), 'lr': self.args.lr * 0.5},
                {'params': raw_model.topo_head.parameters(), 'lr': self.args.lr * 1.0},
            ]
            return torch.optim.AdamW(
                train_params, lr=self.args.lr, weight_decay=self.args.weight_decay
            )

    def check_and_update_stage(self, epoch: int):
        stage_changed = False

        if epoch < self.args.freeze_coanet_epochs and self.current_stage != 1:
            self.current_stage = 1
            self.optimizer = self.build_optimizer(stage=1)
            stage_changed = True

        elif epoch >= self.args.freeze_coanet_epochs and self.current_stage == 1:
            self.current_stage = 2
            self.optimizer = self.build_optimizer(stage=2)
            stage_changed = True

        raw_model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        is_frozen = epoch < self.args.freeze_coanet_epochs
        if hasattr(raw_model, 'set_freeze_coanet'):
            raw_model.set_freeze_coanet(is_frozen)

        if epoch == self.args.start_epoch or stage_changed:
            print_freeze_status(self.model, epoch)

    def training(self, epoch):
        self.check_and_update_stage(epoch)

        train_loss = 0.0
        train_coanet_loss = 0.0
        train_topo_loss = 0.0

        self.model.train()

        # --- FIX: self.model.train() vừa set TẤT CẢ submodule (kể cả coanet đang
        # bị freeze) sang train-mode, khiến BatchNorm dùng batch statistics của
        # batch hiện tại thay vì running mean/var đã pretrain -> phá hỏng hoàn
        # toàn chất lượng CoANet dù trọng số Conv đã load đúng 100%.
        # Ép lại coanet về eval() để BN dùng đúng running stats đã học.
        raw_model = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        if getattr(raw_model, 'freeze_coanet', False):
            raw_model.coanet.eval()

        tbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image = sample['image']               
            gt_mask = sample['gt_mask']           
            gt_connect = sample['gt_connect_d1']  
            gt_connect_d1 = sample['gt_connect_d3']

            gt_mask = (gt_mask > 0.5).float()

            if self.args.cuda:
                image = image.cuda(non_blocking=True)
                gt_mask = gt_mask.cuda(non_blocking=True)
                gt_connect = gt_connect.cuda(non_blocking=True)
                gt_connect_d1 = gt_connect_d1.cuda(non_blocking=True)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.args.cuda):
                out_dict = self.model(image, gt_mask=gt_mask, return_aux=True)

                fused_mask = out_dict['fused_mask']       
                seg_logits = out_dict['seg_logits']       
                connect = out_dict['connect']             
                connect_d1 = out_dict['connect_d1']       
                aux_seg = out_dict.get('aux_seg', None)   
                topo_out = out_dict.get('topo', {})       

                c_loss, c_loss_dict = self.coanet_loss_fn(
                    seg=seg_logits,
                    connect=connect,
                    connect_d1=connect_d1,
                    aux_seg=aux_seg,
                    gt_seg=gt_mask,
                    gt_connect=gt_connect,
                    gt_connect_d1=gt_connect_d1
                )
                
                t_loss = torch.tensor(0.0, device=image.device)
                if 'edge_gt' in topo_out and topo_out['edge_gt'] is not None and 'logits' in topo_out:
                    t_loss = compute_topo_loss(
                        logits=topo_out['logits'],
                        edge_gt=topo_out['edge_gt'],
                        pairs_valid=topo_out['pairs_valid']
                    )
                
                total_loss = c_loss + self.args.topo_weight * t_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = compute_grad_norm(self.model)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += total_loss.item()
            train_coanet_loss += c_loss.item()
            train_topo_loss += t_loss.item()

            global_step = i + num_img_tr * epoch
            current_lr = self.optimizer.param_groups[0]['lr']

            tbar.set_description(
                f'Loss: {train_loss/(i+1):.4f} | CoANet: {train_coanet_loss/(i+1):.4f} | Topo: {train_topo_loss/(i+1):.4f}'
            )

            if global_step % self.args.log_interval == 0:
                self.writer.add_scalar('train/total_loss_iter', total_loss.item(), global_step)
                self.writer.add_scalar('train/lr_iter', current_lr, global_step)
                self.writer.add_scalar('train/grad_norm_iter', grad_norm, global_step)

                with torch.no_grad():
                    seg_pos_ratio = positive_ratio(torch.sigmoid(seg_logits) > 0.5)
                    gt_pos_ratio = positive_ratio(gt_mask > 0.5)
                self.writer.add_scalar('train/seg_pred_positive_ratio', seg_pos_ratio, global_step)
                self.writer.add_scalar('train/gt_positive_ratio', gt_pos_ratio, global_step)

        print(f'\n--- Train Epoch {epoch} Results ---')
        print(f'Train Loss Total: {train_loss / num_img_tr:.4f} | CoANet: {train_coanet_loss / num_img_tr:.4f} | Topo: {train_topo_loss / num_img_tr:.4f}')

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        # --- Evaluator riêng cho seg_logits THUẦN (chưa qua TopoNet/fusion) ---
        # Mục đích: cô lập xem CoANet pretrained tự nó tốt tới đâu, tách khỏi
        # ảnh hưởng (có thể đang làm nhiễu) của graph_mask từ TopoNet mới train.
        if not hasattr(self, 'evaluator_seg_only'):
            self.evaluator_seg_only = Evaluator(num_class=2)
        self.evaluator_seg_only.reset()

        tbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
        val_loss = 0.0
        val_coanet_loss = 0.0
        val_topo_loss = 0.0
        num_img_val = len(self.val_loader)

        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image = sample['image']
                gt_mask = sample['gt_mask']
                gt_connect = sample['gt_connect_d1']
                gt_connect_d1 = sample['gt_connect_d3']

                gt_mask = (gt_mask > 0.5).float()

                if self.args.cuda:
                    image = image.cuda(non_blocking=True)
                    gt_mask = gt_mask.cuda(non_blocking=True)
                    gt_connect = gt_connect.cuda(non_blocking=True)
                    gt_connect_d1 = gt_connect_d1.cuda(non_blocking=True)

                # ✅ Đã sửa API AMP tương thích mới
                with torch.amp.autocast('cuda', enabled=self.args.cuda):
                    out_dict = self.model(image, gt_mask=gt_mask, return_aux=True)
                    fused_mask = out_dict['fused_mask']
                    seg_logits = out_dict['seg_logits']
                    connect = out_dict['connect']
                    connect_d1 = out_dict['connect_d1']
                    topo_out = out_dict.get('topo', {})

                    c_loss, _ = self.coanet_loss_fn(
                        seg=seg_logits,
                        connect=connect,
                        connect_d1=connect_d1,
                        aux_seg=None,
                        gt_seg=gt_mask,
                        gt_connect=gt_connect,
                        gt_connect_d1=gt_connect_d1
                    )

                    t_loss = torch.tensor(0.0, device=image.device)
                    if 'edge_gt' in topo_out and topo_out['edge_gt'] is not None and 'logits' in topo_out:
                        t_loss = compute_topo_loss(
                            logits=topo_out['logits'],
                            edge_gt=topo_out['edge_gt'],
                            pairs_valid=topo_out['pairs_valid']
                        )

                    total_val_loss = c_loss + self.args.topo_weight * t_loss

                val_loss += total_val_loss.item()
                val_coanet_loss += c_loss.item()
                val_topo_loss += t_loss.item()

                tbar.set_description(f'Val Loss: {val_loss / (i + 1):.4f}')

                if fused_mask.shape[-2:] != gt_mask.shape[-2:]:
                    fused_mask_eval = F.interpolate(fused_mask, size=gt_mask.shape[-2:], mode='bilinear', align_corners=True)
                else:
                    fused_mask_eval = fused_mask

                # --- Đánh giá FUSED mask (CoANet + TopoNet) ---
                pred = (fused_mask_eval > 0.5).squeeze(1).cpu().numpy().astype(np.int64)
                target_n = gt_mask.squeeze(1).cpu().numpy().astype(np.int64)
                self.evaluator.add_batch(target_n, pred)

                # --- Đánh giá seg_logits THUẦN (bỏ qua TopoNet/fusion hoàn toàn) ---
                if seg_logits.shape[-2:] != gt_mask.shape[-2:]:
                    seg_logits_eval = F.interpolate(seg_logits, size=gt_mask.shape[-2:], mode='bilinear', align_corners=True)
                else:
                    seg_logits_eval = seg_logits
                pred_seg = (torch.sigmoid(seg_logits_eval) > 0.5).squeeze(1).cpu().numpy().astype(np.int64)
                self.evaluator_seg_only.add_batch(target_n, pred_seg)

        road_iou = get_road_iou(self.evaluator)
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()

        # --- Kết quả seg-only (thuần CoANet, không qua TopoNet/fusion) ---
        road_iou_seg = get_road_iou(self.evaluator_seg_only)
        Precision_seg = self.evaluator_seg_only.Pixel_Precision()
        Recall_seg = self.evaluator_seg_only.Pixel_Recall()
        F1_seg = self.evaluator_seg_only.Pixel_F1()

        self.writer.add_scalar('val/Road_IoU_seg_only', road_iou_seg, epoch)
        self.writer.add_scalar('val/Precision_seg_only', Precision_seg, epoch)
        self.writer.add_scalar('val/Recall_seg_only', Recall_seg, epoch)
        self.writer.add_scalar('val/F1_seg_only', F1_seg, epoch)

        self.writer.add_scalar('val/total_loss_epoch', val_loss / num_img_val, epoch)
        self.writer.add_scalar('val/Road_IoU', road_iou, epoch)
        self.writer.add_scalar('val/Precision', Precision, epoch)
        self.writer.add_scalar('val/Recall', Recall, epoch)
        self.writer.add_scalar('val/F1', F1, epoch)

        print(f'\n--- Validation Epoch {epoch} Results ---')
        print(f'[FUSED   ] Road IoU: {road_iou:.4f} | Precision: {Precision:.4f} | Recall: {Recall:.4f} | F1: {F1:.4f}')
                print(
            f'[SEG-ONLY] Road IoU: {road_iou_seg:.4f} | '
            f'Precision: {Precision_seg:.4f} | '
            f'Recall: {Recall_seg:.4f} | '
            f'F1: {F1_seg:.4f} '
            f'<-- Chỉ số này phản ánh đúng chất lượng CoANet pretrained'
        )

        new_pred = road_iou
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
            print(f"==> Đã lưu Best Checkpoint mới với Road IoU: {self.best_pred:.4f}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch CoANet + TopoNet Training")
    
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--out-stride', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='spacenet', choices=['spacenet', 'DeepGlobe'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--loss-type', type=str, default='ce')
    parser.add_argument('--base-size', type=int, default=1024)
    parser.add_argument('--crop-size', type=int, default=512)

    parser.add_argument('--coanet-weights', type=str, default=None)
    parser.add_argument('--freeze-coanet-epochs', type=int, default=0)

    parser.add_argument('--max-points', type=int, default=128)
    parser.add_argument('--k-neighbors', type=int, default=5)
    parser.add_argument('--topo-score-thresh', type=float, default=0.3)
    parser.add_argument('--topo-weight', type=float, default=0.5)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0,1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkname', type=str, default=None)
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--no-val', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=50)

    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = os.environ.get('DATA_DIR')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.checkname is None:
        args.checkname = f'CoANetTopo-{args.backbone}'

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = Trainer(args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
