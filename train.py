import argparse
import os, time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.coanet import *
from utils.loss import SegmentationLosses, dice_bce_loss
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def load_coanet_weights_safely(model, checkpoint_path):
    """Load CoANet weights and migrate the old SCM kernels into DSConv.

    At zero offset, a DSConv sampling kernel is equivalent to its source strip
    convolution. Spatial transposition is needed because sampled points are
    packed on the dimension that ``sample_conv`` subsequently collapses.
    """
    print(f"\n==================================================")
    print(f"🔍 BẮT ĐẦU KIỂM TRA & LOAD WEIGHTS TỪ: {checkpoint_path}")
    print(f"==================================================")
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k
        for prefix in ['module.', 'coanet.']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned_state_dict[name] = v

    model_state = model.state_dict()
    transferred = {}
    converted = []

    # First retain every tensor whose name and shape did not change.
    for name, value in cleaned_state_dict.items():
        if name in model_state and model_state[name].shape == value.shape:
            transferred[name] = value

    # Then migrate all four strip branches in all four DecoderBlocks.
    for block in ('decoder1', 'decoder2', 'decoder3', 'decoder4'):
        for branch in range(1, 5):
            old_prefix = f'decoder.{block}.deconv{branch}'
            new_prefix = f'decoder.{block}.dsconv{branch}.sample_conv'
            old_weight = f'{old_prefix}.weight'
            new_weight = f'{new_prefix}.weight'
            if old_weight in cleaned_state_dict and new_weight in model_state:
                candidate = cleaned_state_dict[old_weight].transpose(-1, -2).contiguous()
                if candidate.shape != model_state[new_weight].shape:
                    raise RuntimeError(
                        f"Không thể chuyển {old_weight}: {tuple(candidate.shape)} "
                        f"!= {tuple(model_state[new_weight].shape)}"
                    )
                transferred[new_weight] = candidate
                converted.append((old_weight, new_weight))

            old_bias = f'{old_prefix}.bias'
            new_bias = f'{new_prefix}.bias'
            if old_bias in cleaned_state_dict and new_bias in model_state:
                if cleaned_state_dict[old_bias].shape != model_state[new_bias].shape:
                    raise RuntimeError(f"Không thể chuyển bias {old_bias}")
                transferred[new_bias] = cleaned_state_dict[old_bias]
                converted.append((old_bias, new_bias))

    result = model.load_state_dict(transferred, strict=False)
    allowed_missing = [key for key in result.missing_keys if '.offset_conv.' in key]
    invalid_missing = [key for key in result.missing_keys if key not in allowed_missing]
    if invalid_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint không tương thích: missing={invalid_missing}, "
            f"unexpected={result.unexpected_keys}"
        )

    loaded_elements = sum(model_state[name].numel() for name in transferred)
    total_elements = sum(value.numel() for value in model_state.values())
    print(f"✅ Đã nạp {len(transferred)} tensors cũ, gồm {len(converted)} tensors SCM→DSConv.")
    print(f"✅ Tái sử dụng {loaded_elements:,}/{total_elements:,} phần tử "
          f"({100.0 * loaded_elements / total_elements:.3f}%).")
    if allowed_missing:
        print(f"🆕 {len(allowed_missing)} tensors offset mới được giữ ở khởi tạo zero.")
    print(f"==================================================\n")
    return model


def build_transfer_param_groups(model, base_lr):
    """Create persistent discriminative-LR groups for gradual unfreezing."""
    groups = {
        'backbone_early': (0.05, []),
        'backbone_layer3': (0.10, []),
        'backbone_layer4': (0.20, []),
        'aspp': (0.50, []),
        'decoder': (1.00, []),
        'connect': (1.00, []),
        'offset': (5.00, []),
    }
    for name, parameter in model.named_parameters():
        if '.offset_conv.' in name:
            key = 'offset'
        elif name.startswith('backbone.layer4.'):
            key = 'backbone_layer4'
        elif name.startswith('backbone.layer3.'):
            key = 'backbone_layer3'
        elif name.startswith('backbone.'):
            key = 'backbone_early'
        elif name.startswith('aspp.'):
            key = 'aspp'
        elif name.startswith('decoder.'):
            key = 'decoder'
        elif name.startswith('connect.'):
            key = 'connect'
        else:
            raise RuntimeError(f"Tham số chưa được phân nhóm: {name}")
        groups[key][1].append(parameter)

    return [
        {'params': parameters, 'lr': base_lr * multiplier,
         'lr_mult': multiplier, 'group_name': name}
        for name, (multiplier, parameters) in groups.items() if parameters
    ]


def transfer_stage_for_epoch(epoch, milestones):
    return sum(epoch >= milestone for milestone in milestones)


def model_state_dict(model):
    core_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    return core_model.state_dict()


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def parameter_count_report(model):
    """Return trainable/total parameter counts grouped by top-level module."""
    report = {}
    for name, parameter in unwrap_model(model).named_parameters():
        group = name.split('.', 1)[0]
        values = report.setdefault(group, {'trainable': 0, 'total': 0})
        values['total'] += parameter.numel()
        if parameter.requires_grad:
            values['trainable'] += parameter.numel()
    return report


def log_transfer_state(model, optimizer, epoch, stage):
    """Print exactly what is trainable and the discriminative LR of each group."""
    print(f"\n[DEBUG][epoch={epoch}] transfer_stage={stage}")
    for name, values in parameter_count_report(model).items():
        print(
            f"  params/{name}: trainable={values['trainable']:,} "
            f"total={values['total']:,}"
        )
    for group in optimizer.param_groups:
        trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
        print(
            f"  lr/{group.get('group_name', 'unnamed')}: "
            f"lr={group['lr']:.3e} trainable={trainable:,}"
        )


class OffsetDebugCapture:
    """Capture lightweight DSConv offset statistics only on selected batches."""
    def __init__(self, model, enabled):
        self.enabled = enabled
        self.handles = []
        self.values = []
        if not enabled:
            return
        for name, module in unwrap_model(model).named_modules():
            if name.endswith('offset_conv'):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def collect(_module, _inputs, output):
            offset = torch.tanh(output.detach())
            self.values.append((
                name,
                offset.abs().mean().item(),
                offset.abs().max().item(),
                offset.abs().gt(0.95).float().mean().item(),
            ))
        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self):
        if not self.values:
            return None
        count = len(self.values)
        return {
            'modules': count,
            'mean_abs': sum(value[1] for value in self.values) / count,
            'max_abs': max(value[2] for value in self.values),
            'saturated_fraction': sum(value[3] for value in self.values) / count,
        }


def optimizer_gradient_report(optimizer):
    report = []
    for group in optimizer.param_groups:
        grad_squared = 0.0
        parameter_squared = 0.0
        tensors_with_grad = 0
        for parameter in group['params']:
            if not parameter.requires_grad:
                continue
            parameter_squared += parameter.detach().float().norm().item() ** 2
            if parameter.grad is not None:
                grad_squared += parameter.grad.detach().float().norm().item() ** 2
                tensors_with_grad += 1
        report.append((
            group.get('group_name', 'unnamed'),
            grad_squared ** 0.5,
            parameter_squared ** 0.5,
            tensors_with_grad,
        ))
    return report


def log_debug_batch(epoch, iteration, output, target, optimizer, offset_summary,
                    thresholds):
    prediction = output.detach()
    target_detached = target.detach()
    positive_rates = ', '.join(
        f"p>={threshold:g}:{prediction.ge(threshold).float().mean().item():.4f}"
        for threshold in thresholds
    )
    print(
        f"\n[DEBUG][epoch={epoch} iter={iteration}] "
        f"pred(min/mean/max)={prediction.min().item():.4f}/"
        f"{prediction.mean().item():.4f}/{prediction.max().item():.4f} "
        f"target_positive={target_detached.mean().item():.4f} {positive_rates}"
    )
    if offset_summary is not None:
        print(
            "  offsets: modules={modules} mean_abs={mean_abs:.6f} "
            "max_abs={max_abs:.6f} saturated={saturated_fraction:.6f}".format(
                **offset_summary
            )
        )
    for name, grad_norm, parameter_norm, tensor_count in optimizer_gradient_report(optimizer):
        print(
            f"  grad/{name}: norm={grad_norm:.3e} param_norm={parameter_norm:.3e} "
            f"tensors={tensor_count}"
        )
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f"  cuda:{device}: allocated={allocated:.2f}GiB "
                f"reserved={reserved:.2f}GiB peak={peak:.2f}GiB"
            )

def get_sample_tensors(sample, split='train'):
    """Giải nén dữ liệu và In ra màn hình các keys đang có nếu bị lỗi."""
    if split == 'val' and isinstance(sample, list):
        sample = sample[0]
        
    required_keys = ['image', 'label', 'connect0', 'connect1', 'connect2', 'connect_d1_0', 'connect_d1_1', 'connect_d1_2']
    
    if not isinstance(sample, dict):
        raise TypeError(f"[LỖI] DataLoader trả về {type(sample)} thay vì dictionary. Hãy kiểm tra lại class Dataset.")

    missing_keys = [k for k in required_keys if k not in sample]
    if missing_keys:
        print(f"\n❌ [LỖI DATALOADER - {split.upper()}] Dữ liệu trả về không khớp định dạng của CoANet.")
        print(f"👉 Các keys tác giả yêu cầu : {required_keys}")
        print(f"👀 Các keys HIỆN CÓ của bạn: {list(sample.keys())}")
        print(f"-> GỢI Ý: Bạn cần mở file Dataset (VD: dataloaders/datasets/spacenet.py) và đảm bảo hàm __getitem__ return đầy đủ các biến connect0, connect1...!")
        raise KeyError(f"DataLoader thiếu keys: {missing_keys}")

    image = sample['image']
    target = sample['label']
    connect_label = torch.cat((sample['connect0'], sample['connect1'], sample['connect2']), 1)
    connect_d1_label = torch.cat((sample['connect_d1_0'], sample['connect_d1_1'], sample['connect_d1_2']), 1)
    
    return image, target, connect_label, connect_d1_label


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

        # 🟢 ÉP CỨNG SỐ CLASS VỀ 1 CHO BÀI TOÁN BINARY ROAD DETECTION
        self.nclass = 1

        # Checkpoint SpaceNet gốc dùng một lớp foreground (road).
        model = CoANet(num_classes= 1,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        scm_type=args.scm_type,
                        dsconv_kernel_size=args.dsconv_kernel_size,
                        dsconv_extend_scope=args.dsconv_extend_scope,
                        backbone_pretrained=not bool(args.coanet_weights))

        # Load weights
        if hasattr(args, 'coanet_weights') and args.coanet_weights:
            model = load_coanet_weights_safely(model, args.coanet_weights)

        use_progressive_transfer = (
            args.scm_type == 'dsconv'
            and args.coanet_weights is not None
            and not args.no_progressive_transfer
        )
        model.set_transfer_stage(0 if use_progressive_transfer else 4)
        train_params = build_transfer_param_groups(model, args.lr)

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
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
        self.use_progressive_transfer = use_progressive_transfer
        self.current_transfer_stage = model.transfer_stage
        
        # Define Evaluator
        self.evaluator = Evaluator(2)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss3 = 0.0
        train_loss = 0.0
        self.model.train()
        core_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        next_stage = (
            transfer_stage_for_epoch(epoch, self.args.unfreeze_epochs)
            if self.use_progressive_transfer else 4
        )
        if next_stage != self.current_transfer_stage:
            self.current_transfer_stage = next_stage
            print(f"\n🔓 Chuyển sang transfer-learning stage {next_stage} tại epoch {epoch}")
        # Calling model.train() re-enables every BN, so stage constraints must
        # be re-applied at the beginning of each training epoch.
        core_model.set_transfer_stage(next_stage)
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        
        for i, sample in enumerate(tbar):
            image, target, connect_label, connect_d1_label = get_sample_tensors(sample, split='train')
            
            if self.args.cuda:
                image, target, connect_label, connect_d1_label = image.cuda(), target.cuda(), connect_label.cuda(), connect_d1_label.cuda()
                
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            if i == 0 and self.args.debug_log_interval > 0:
                log_transfer_state(core_model, self.optimizer, epoch, next_stage)
            self.optimizer.zero_grad()

            debug_now = (
                self.args.debug_log_interval > 0
                and (i == 0 or (i + 1) % self.args.debug_log_interval == 0)
            )
            offset_capture = OffsetDebugCapture(core_model, enabled=debug_now)
            output, out_connect, out_connect_d1 = self.model(image)
            offset_capture.close()
            target = torch.unsqueeze(target, 1)
            loss1 = self.criterion(output, target)
            loss2 = self.criterion_con(out_connect, connect_label)
            loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
            
            lad = 0.2
            loss = loss1 + lad*(0.6*loss2 + 0.4*loss3)
            loss.backward()
            if debug_now:
                log_debug_batch(
                    epoch, i + 1, output, target, self.optimizer,
                    offset_capture.summary(), self.args.debug_thresholds,
                )
            self.optimizer.step()
            
            train_loss1 += loss1.item()
            train_loss2 += lad * 0.6 * loss2.item()
            train_loss3 += lad * 0.4 * loss3.item()
            train_loss += loss.item()
            
            current_loss = train_loss / (i + 1)
            l1 = train_loss1 / (i + 1)
            l2 = train_loss2 / (i + 1)
            l3 = train_loss3 / (i + 1)
            
            tbar.set_postfix(
                loss=f"{current_loss:.3f}",
                l1=f"{l1:.4f}",
                l2=f"{l2:.3f}",
                l3=f"{l3:.3f}"
            )
            
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            pred = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            pred = (pred >= self.args.eval_threshold).astype(np.uint8)
            self.evaluator.add_batch(target_n, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.writer.add_scalar('train/loss1_epoch', train_loss1, epoch)
        self.writer.add_scalar('train/loss2_epoch', train_loss2, epoch)
        self.writer.add_scalar('train/loss3_epoch', train_loss3, epoch)
        self.writer.add_scalar('train/mIoU', mIoU, epoch)
        self.writer.add_scalar('train/Acc', Acc, epoch)
        self.writer.add_scalar('train/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('train/IoU', IoU, epoch)
        self.writer.add_scalar('train/Precision', Precision, epoch)
        self.writer.add_scalar('train/Recall', Recall, epoch)
        self.writer.add_scalar('train/F1', F1, epoch)
        
        print('Train:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))
        print('Loss: %.3f, Loss1: %.6f, Loss2: %.3f, Loss3: %.3f' % (train_loss, train_loss1, train_loss2, train_loss3))

        if self.args.no_val:
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_state_dict(self.model),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        threshold_evaluators = {
            threshold: Evaluator(2) for threshold in self.args.debug_thresholds
        }
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss1 = 0.0
        test_loss2 = 0.0
        test_loss3 = 0.0
        test_loss = 0.0
        num_img_tr = len(self.val_loader)
        
        for i, sample in enumerate(tbar):
            image, target, connect_label, connect_d1_label = get_sample_tensors(sample, split='val')

            if self.args.cuda:
                image, target, connect_label, connect_d1_label = image.cuda(), target.cuda(), connect_label.cuda(), connect_d1_label.cuda()
                
            with torch.no_grad():
                output, out_connect, out_connect_d1 = self.model(image)
                
            target = torch.unsqueeze(target, 1)
            loss1 = self.criterion(output, target)
            loss2 = self.criterion_con(out_connect, connect_label)
            loss3 = self.criterion_con(out_connect_d1, connect_d1_label)
            
            lad = 0.2
            loss = loss1 + lad * (0.6*loss2 + 0.4*loss3)
            test_loss1 += loss1.item()
            test_loss2 += lad * 0.6 * loss2.item()
            test_loss3 += lad * 0.4 * loss3.item()
            test_loss += loss.item()
            
            tbar.set_description('Test loss: %.3f, loss1: %.6f, loss2: %.3f, loss3: %.3f' % (test_loss / (i + 1), test_loss1 / (i + 1), test_loss2 / (i + 1), test_loss3 / (i + 1)))
            
            prediction_probability = output.data.cpu().numpy()
            target_n = target.cpu().numpy()
            pred = (prediction_probability >= self.args.eval_threshold).astype(np.uint8)
            self.evaluator.add_batch(target_n, pred)
            for threshold, evaluator in threshold_evaluators.items():
                evaluator.add_batch(
                    target_n,
                    (prediction_probability >= threshold).astype(np.uint8),
                )

            if i % (num_img_tr // 1) == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, i, split='Val')

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        IoU = self.evaluator.Intersection_over_Union()
        Precision = self.evaluator.Pixel_Precision()
        Recall = self.evaluator.Pixel_Recall()
        F1 = self.evaluator.Pixel_F1()
        
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/loss1_epoch', test_loss1, epoch)
        self.writer.add_scalar('val/loss2_epoch', test_loss2, epoch)
        self.writer.add_scalar('val/loss3_epoch', test_loss3, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/IoU', IoU, epoch)
        self.writer.add_scalar('val/Precision', Precision, epoch)
        self.writer.add_scalar('val/Recall', Recall, epoch)
        self.writer.add_scalar('val/F1', F1, epoch)
        
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))
        print('Loss: %.3f, Loss1: %.3f, Loss2: %.3f, Loss3: %.3f' % (test_loss, test_loss1, test_loss2, test_loss3))
        if threshold_evaluators:
            print('Validation threshold sweep:')
            for threshold, evaluator in threshold_evaluators.items():
                threshold_iou = evaluator.Intersection_over_Union()
                threshold_f1 = evaluator.Pixel_F1()
                threshold_precision = evaluator.Pixel_Precision()
                threshold_recall = evaluator.Pixel_Recall()
                print(
                    f"  threshold={threshold:.2f} IoU={threshold_iou:.6f} "
                    f"F1={threshold_f1:.6f} Precision={threshold_precision:.6f} "
                    f"Recall={threshold_recall:.6f}"
                )
                self.writer.add_scalar(
                    f'val_threshold/IoU_{threshold:.2f}', threshold_iou, epoch
                )

        new_pred = IoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_state_dict(self.model),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch CoANet Training")
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8, help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='spacenet', choices=['spacenet', 'DeepGlobe'], help='dataset name (default: spacenet)')
    
    # 🟢 Đã thêm 2 tham số quan trọng bạn truyền vào qua dòng lệnh
    parser.add_argument('--data-dir', type=str, default=None, help='path to dataset directory')
    parser.add_argument('--coanet-weights', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--scm-type', type=str, default='dsconv', choices=['strip', 'dsconv'],
                        help='SCM implementation; dsconv enables the improved model')
    parser.add_argument('--dsconv-kernel-size', type=int, default=9,
                        help='odd Dynamic Snake kernel size; use 9 to transfer SCM weights')
    parser.add_argument('--dsconv-extend-scope', type=float, default=1.0,
                        help='maximum scale of accumulated snake offsets')
    parser.add_argument('--unfreeze-epochs', type=int, nargs=4, default=[5, 15, 30, 45],
                        metavar=('DECODER', 'LAYER4', 'LAYER3', 'ALL'),
                        help='epochs that activate transfer stages 1, 2, 3 and 4')
    parser.add_argument('--no-progressive-transfer', action='store_true',
                        help='train every layer immediately instead of gradual unfreezing')
    
    parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=False, help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False, help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='con_ce', choices=['ce', 'con_ce', 'focal'], help='loss func type')
    parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for DSConv training (default: 2)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False, help='whether to use balanced weights (default: False)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='decoder learning rate (default: 0.001)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1,2,3', help='use which gpu to train (default=0,1,2,3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    parser.add_argument('--ft', action='store_true', default=False, help='finetuning on a different dataset')
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--eval-threshold', type=float, default=0.1,
                        help='probability threshold used for primary IoU/F1 and best checkpoint')
    parser.add_argument('--debug-log-interval', type=int, default=50,
                        help='print gradients, offsets, predictions and VRAM every N batches; 0 disables')
    parser.add_argument('--debug-thresholds', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='thresholds included in batch diagnostics and validation sweep')

    args = parser.parse_args()
    if args.dsconv_kernel_size % 2 != 1:
        parser.error('--dsconv-kernel-size must be odd')
    if args.coanet_weights and args.scm_type == 'dsconv' and args.dsconv_kernel_size != 9:
        parser.error('SCM checkpoint transfer requires --dsconv-kernel-size 9')
    if args.unfreeze_epochs != sorted(args.unfreeze_epochs):
        parser.error('--unfreeze-epochs must be in non-decreasing order')
    if not 0.0 <= args.eval_threshold <= 1.0:
        parser.error('--eval-threshold must be between 0 and 1')
    if args.debug_log_interval < 0:
        parser.error('--debug-log-interval cannot be negative')
    if any(threshold < 0.0 or threshold > 1.0 for threshold in args.debug_thresholds):
        parser.error('--debug-thresholds values must be between 0 and 1')
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be comma-separated list of integers only')

    if args.checkname is None:
        args.checkname = 'CoANet-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
