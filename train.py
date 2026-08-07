import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders import make_data_loader
from modeling.coanet import CoANet
from modeling.sync_batchnorm.replicate import patch_replication_callback
from mypath import Path
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.loss import SegmentationLosses, dice_bce_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def load_coanet_weights_safely(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Nạp weights từ checkpoint cho CoANet và kiểm tra độ khớp (Key & Shape)."""
    print(f"\n==================================================")
    print(f"🔍 BẮT ĐẦU KIỂM TRA & LOAD WEIGHTS TỪ: {checkpoint_path}")
    print(f"==================================================")
    
    if not os.path.isfile(checkpoint_path):
        print(f"⚠️ [CẢNH BÁO] Không tìm thấy file weights tại '{checkpoint_path}'. Bỏ qua bước load weights.")
        return model

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Bóc tách state_dict từ các định dạng lưu trữ checkpoint khác nhau
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    else:
        state_dict = checkpoint

    # Chuẩn hóa tên Key (Loại bỏ prefix thừa do DataParallel/Wrapper)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k
        for prefix in ['module.', 'coanet.']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned_state_dict[name] = v

    model_state_dict = model.state_dict()
    
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

    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Kiểm tra giá trị tensor thực tế sau khi nạp
    sample_param = list(model.parameters())[0]
    print(f"\n🧪 Weights Tensor Check (Layer 0): Mean={sample_param.mean().item():.6f} | Std={sample_param.std().item():.6f}")
    print(f"✅ Đã load thành công {len(filtered_state_dict)} weights vào CoANet!")
    print(f"==================================================\n")
    
    return model


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define Network
        model = CoANet(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn
        )

        # Nạp Pretrained Weights nếu có đường dẫn
        if hasattr(args, 'coanet_weights') and args.coanet_weights is not None:
            model = load_coanet_weights_safely(model, args.coanet_weights)

        # Tách tham số cho Optimizer (1x LR cho Backbone, 2x LR cho các phần Head/Decoder)
        train_params = [
            {'params': model.get_1x_lr_params(), 'lr': args.lr},
            {'params': model.get_2x_lr_params(), 'lr': args.lr * 2}
        ]

        # Define Optimizer
        optimizer = torch.optim.SGD(
            train_params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )

        # Define Criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = dice_bce_loss()
        self.criterion_con = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator & Scheduler
        self.evaluator = Evaluator(2)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Multi-GPU DataParallel Setup
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming Checkpoint
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
            print(f"=> Đã nạp checkpoint '{args.resume}' (Epoch {checkpoint['epoch']})")

        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image = sample['image']

            # 1. Lấy GT Segmentation Mask
            target = sample.get('gt_mask', sample.get('label', sample.get('mask')))
            
            # 2. Lấy GT Connect
            if 'gt_connect' in sample:
              connect_label = sample['gt_connect']
            elif 'connect' in sample:
              connect_label = sample['connect']
            else:
              con0, con1, con2 = (
                  sample['connect0'],
                  sample['connect1'],
                  sample['connect2'],
              )
              connect_label = torch.cat((con0, con1, con2), 1)
            
            # 3. Lấy GT Connect D1
            if 'gt_connect_d1' in sample:
              connect_d1_label = sample['gt_connect_d1']
            elif 'connect_d1' in sample:
              connect_d1_label = sample['connect_d1']
            else:
              con_d1_0, con_d1_1, con_d1_2 = (
                  sample['connect_d1_0'],
                  sample['connect_d1_1'],
                  sample['connect_d1_2'],
              )
              connect_d1_label = torch.cat((con_d1_0, con_d1_1, con_d1_2), 1)

            if self.args.cuda:
                image = image.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                connect_label = connect_label.cuda(non_blocking=True)
                connect_d1_label = connect_d1_label.cuda(non_blocking=True)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            output, out_connect, out_connect_d1 = self.model(image)
            target = torch.unsqueeze(target, 1)

            loss1 = self.criterion(output, target)
            loss2 = self.criterion_con(out_connect, connect_label)
            loss3 = self.criterion_con(out_connect_d1, connect_d1_label)

            lad = 0.2
            loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)
            loss.backward()
            self.optimizer.step()

            train_loss1 += loss1.item()
            train_loss2 += lad * 0.6 * loss2.item()
            train_loss3 += lad * 0.4 * loss3.item()
            train_loss += loss.item()

            tbar.set_description(
                f'Loss: {train_loss/(i+1):.3f} | L1(Seg): {train_loss1/(i+1):.4f} | '
                f'L2(Con): {train_loss2/(i+1):.3f} | L3(ConD1): {train_loss3/(i+1):.3f}'
            )
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            pred[pred > 0.1] = 1
            pred[pred < 0.1] = 0
            self.evaluator.add_batch(target_n, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()

        self.writer.add_scalar('train/total_loss_epoch', train_loss / num_img_tr, epoch)
        self.writer.add_scalar('train/loss1_epoch', train_loss1 / num_img_tr, epoch)
        self.writer.add_scalar('train/loss2_epoch', train_loss2 / num_img_tr, epoch)
        self.writer.add_scalar('train/loss3_epoch', train_loss3 / num_img_tr, epoch)
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/Acc', Acc, epoch)
        self.writer.add_scalar('train/IoU', IoU, epoch)
        self.writer.add_scalar('train/Precision', Precision, epoch)
        self.writer.add_scalar('train/Recall', Recall, epoch)
        self.writer.add_scalar('train/F1', F1, epoch)

        print(f'\n--- Train Epoch {epoch} Summary ---')
        print(f"Acc: {Acc:.4f} | mIoU: {mIoU:.4f} | IoU: {IoU:.4f} | Precision: {Precision:.4f} | Recall: {Recall:.4f} | F1: {F1:.4f}")

        if self.args.no_val:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if self.args.cuda else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
        test_loss1 = 0.0
        test_loss2 = 0.0
        test_loss3 = 0.0
        test_loss = 0.0
        num_img_val = len(self.val_loader)

        with torch.no_grad():
            for i, sample in enumerate(tbar):
                val_sample = sample[0] if isinstance(sample, list) else sample

                image = val_sample['image']
                target = val_sample.get(
                    'gt_mask', val_sample.get('label', val_sample.get('mask'))
                )
                
                # 1. Lấy GT Connect
                if 'gt_connect' in val_sample:
                  connect_label = val_sample['gt_connect']
                elif 'connect' in val_sample:
                  connect_label = val_sample['connect']
                else:
                  con0, con1, con2 = (
                      val_sample['connect0'],
                      val_sample['connect1'],
                      val_sample['connect2'],
                  )
                  connect_label = torch.cat((con0, con1, con2), 1)
                
                # 2. Lấy GT Connect D1
                if 'gt_connect_d1' in val_sample:
                  connect_d1_label = val_sample['gt_connect_d1']
                elif 'connect_d1' in val_sample:
                  connect_d1_label = val_sample['connect_d1']
                else:
                  con_d1_0, con_d1_1, con_d1_2 = (
                      val_sample['connect_d1_0'],
                      val_sample['connect_d1_1'],
                      val_sample['connect_d1_2'],
                  )
                  connect_d1_label = torch.cat((con_d1_0, con_d1_1, con_d1_2), 1)

                if self.args.cuda:
                    image = image.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                    connect_label = connect_label.cuda(non_blocking=True)
                    connect_d1_label = connect_d1_label.cuda(non_blocking=True)

                output, out_connect, out_connect_d1 = self.model(image)
                target = torch.unsqueeze(target, 1)

                loss1 = self.criterion(output, target)
                loss2 = self.criterion_con(out_connect, connect_label)
                loss3 = self.criterion_con(out_connect_d1, connect_d1_label)

                lad = 0.2
                loss = loss1 + lad * (0.6 * loss2 + 0.4 * loss3)

                test_loss1 += loss1.item()
                test_loss2 += lad * 0.6 * loss2.item()
                test_loss3 += lad * 0.4 * loss3.item()
                test_loss += loss.item()

                tbar.set_description(f'Test loss: {test_loss / (i + 1):.3f}')

                pred = output.data.cpu().numpy()
                target_n = target.cpu().numpy()
                pred[pred > 0.1] = 1
                pred[pred < 0.1] = 0
                self.evaluator.add_batch(target_n, pred)

                if i % max(1, num_img_val // 5) == 0:
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i, split='Val')

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()

        self.writer.add_scalar('val/total_loss_epoch', test_loss / num_img_val, epoch)
        self.writer.add_scalar('val/loss1_epoch', test_loss1 / num_img_val, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/IoU', IoU, epoch)
        self.writer.add_scalar('val/Precision', Precision, epoch)
        self.writer.add_scalar('val/Recall', Recall, epoch)
        self.writer.add_scalar('val/F1', F1, epoch)

        print(f'\n--- Validation Epoch {epoch} Results ---')
        print(f"Acc: {Acc:.4f} | mIoU: {mIoU:.4f} | IoU: {IoU:.4f} | Precision: {Precision:.4f} | Recall: {Recall:.4f} | F1: {F1:.4f}")

        new_pred = IoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict() if self.args.cuda else self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
            print(f"==> Đã lưu Best Checkpoint mới với IoU: {self.best_pred:.4f}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch CoANet Training")
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='spacenet', choices=['spacenet', 'DeepGlobe'], help='dataset name')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--sync-bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--freeze-bn', action='store_true', default=False, help='whether to freeze bn parameters')
    parser.add_argument('--loss-type', type=str, default='con_ce', choices=['ce', 'con_ce', 'focal'], help='loss func type')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='path to dataset directory (optional)',
    )
    # Pretrained weights argument
    parser.add_argument('--coanet-weights', type=str, default=None, help='đường dẫn tới file weights (.pth hoặc .pth.tar)')

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False, help='whether to use balanced weights')
    
    # Optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov')
    
    # System
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1', help='GPU IDs list')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    
    # Checkpoints
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be comma-separated list of integers only')

    if args.checkname is None:
        args.checkname = f'CoANet-{args.backbone}'
        
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epochs:', trainer.args.epochs)
    
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
    main()
