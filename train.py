import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from collections import defaultdict
import math, gc, json, warnings
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except Exception:
    _TB = False

SEP = "=" * 70

from modeling.decoder import GCNetHead



IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _sample_key(path):
    """Normalize common image/mask suffixes so files can be paired safely."""
    key = path.stem.lower()
    suffixes = ('_image', '_images', '_img', '_sat', '_mask', '_masks',
                '_gt', '_label', '_labels')
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                changed = True
                break
    return key


def _index_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")
    files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No supported image files found in: {folder}")
    index = {}
    for path in files:
        key = _sample_key(path)
        if key in index:
            raise RuntimeError(f"Duplicate sample key '{key}': {index[key]} and {path}")
        index[key] = path
    return index


def build_image_mask_pairs(image_dir, mask_dir):
    images, masks = _index_files(image_dir), _index_files(mask_dir)
    common = sorted(images.keys() & masks.keys())
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise RuntimeError(
            f"Image/mask pairing failed: {len(missing_masks)} images have no mask and "
            f"{len(missing_images)} masks have no image. "
            f"Examples: missing masks={missing_masks[:5]}, missing images={missing_images[:5]}"
        )
    if len(common) < 2:
        raise RuntimeError("At least two paired samples are required for train/validation split.")
    return [(images[key], masks[key]) for key in common]


class RoadFolderDataset(Dataset):
    def __init__(self, pairs, img_size, num_classes=2, augment=False):
        self.pairs = list(pairs)
        self.img_size = tuple(img_size)  # (height, width)
        self.num_classes = num_classes
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def _read_mask(self, path):
        mask = np.asarray(Image.open(path))
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        if self.num_classes == 2:
            mask = (mask > 0).astype(np.uint8)
        elif mask.max() >= self.num_classes:
            raise ValueError(
                f"Mask {path} contains label {int(mask.max())}, but num_classes={self.num_classes}."
            )
        return mask

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert('RGB')
        mask = Image.fromarray(self._read_mask(mask_path))
        height, width = self.img_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)

        image = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())

        if self.augment:
            if torch.rand(()) < 0.5:
                image, mask = image.flip(-1), mask.flip(-1)
            if torch.rand(()) < 0.5:
                image, mask = image.flip(-2), mask.flip(-2)

        mean = image.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return (image - mean) / std, mask.long()


def _compute_class_weights(pairs, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, mask_path in tqdm(pairs, desc="Computing class weights"):
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        if num_classes == 2:
            mask = (mask > 0).astype(np.int64)
        valid = (mask >= 0) & (mask < num_classes)
        counts += np.bincount(mask[valid].astype(np.int64), minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def create_folder_dataloaders(image_dir, mask_dir, val_ratio, seed, batch_size,
                              num_workers, img_size, num_classes,
                              compute_class_weights=False):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    pairs = build_image_mask_pairs(image_dir, mask_dir)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    val_count = max(1, min(len(pairs) - 1, round(len(pairs) * val_ratio)))
    val_pairs = [pairs[i] for i in order[:val_count]]
    train_pairs = [pairs[i] for i in order[val_count:]]

    train_set = RoadFolderDataset(train_pairs, img_size, num_classes, augment=True)
    val_set = RoadFolderDataset(val_pairs, img_size, num_classes, augment=False)
    loader_args = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                       persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=len(train_set) > batch_size, **loader_args)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            drop_last=False, **loader_args)
    weights = _compute_class_weights(train_pairs, num_classes) if compute_class_weights else None
    print(f"Dataset: {len(pairs)} pairs | train={len(train_set)} | val={len(val_set)} "
          f"| split_seed={seed}")
    print(f"Example pair: {pairs[0][0].name} <-> {pairs[0][1].name}")
    return train_loader, val_loader, weights


# ============================================================
# LOGGING
# ============================================================

class _DummyWriter:
    def __init__(self, log_dir):
        import csv, pathlib
        p = pathlib.Path(log_dir); p.mkdir(parents=True, exist_ok=True)
        self._f   = open(p / "metrics.csv", 'w', newline='')
        self._csv = csv.writer(self._f)
        self._csv.writerow(['tag', 'step', 'value'])

    def add_scalar(self, tag, value, step):
        self._csv.writerow([tag, step, f"{value:.6f}"]); self._f.flush()

    def close(self): self._f.close()


def _make_writer(log_dir):
    if _TB:
        try: return SummaryWriter(log_dir=str(log_dir))
        except Exception: pass
    return _DummyWriter(log_dir)


class DiagnosticLogger:
    def __init__(self, save_dir, class_names):
        import csv
        self.save_dir    = Path(save_dir)
        self.class_names = class_names
        self.history     = defaultdict(list)
        self._f   = open(self.save_dir / "diagnostics.csv", 'w', newline='')
        self._csv = csv.writer(self._f)
        self._csv.writerow(['epoch', 'key', 'value'])

    def log(self, epoch, key, value):
        self.history[key].append((epoch, float(value)))
        self._csv.writerow([epoch, key, f"{float(value):.6f}"])
        self._f.flush()

    def log_dict(self, epoch, d, prefix=''):
        for k, v in d.items():
            self.log(epoch, f"{prefix}{k}" if prefix else k, v)

    def print_epoch_summary(self, epoch):
        print(f"\n{'─'*70}\n  EPOCH {epoch+1:>3} SUMMARY\n{'─'*70}")
        metrics = [
            ('val/miou',         'Val mIoU',        '.4f'),
            ('val/accuracy',     'Val Accuracy',    '.4f'),
            ('val/loss',         'Val Loss',        '.4f'),
            ('train/ohem',       'Train OHEM',      '.4f'),
            ('train/dice',       'Train Dice',      '.4f'),
            ('train/max_grad',   'Max Gradient',    '.3f'),
            ('dwsa/gamma4',      'DWSA gamma4',     '.4f'),
            ('dwsa/gamma5',      'DWSA gamma5',     '.4f'),
            ('dwsa/gamma6',      'DWSA gamma6',     '.4f'),
            ('fan/alpha1_mean',  'FAN alpha1',      '.4f'),
            ('fan/alpha2_mean',  'FAN alpha2',      '.4f'),
            ('train/hard_ratio', 'OHEM hard ratio', '.3f'),
        ]
        print(f"  {'Metric':<24}  {'Value':>10}  {'Trend':>12}")
        print(f"  {'─'*24}  {'─'*10}  {'─'*12}")
        for key, label, fmt in metrics:
            h = self.history.get(key, [])
            if not h: continue
            val = h[-1][1]
            if len(h) < 3:
                trend = '(new)'
            else:
                delta = h[-1][1] - h[-3][1]
                arrow = '↑' if delta > 1e-4 else ('↓' if delta < -1e-4 else '→')
                trend = f"{arrow} {abs(delta):.4f}"
            print(f"  {label:<24}  {val:>10{fmt}}  {trend:>12}")
        print(f"{'─'*70}\n")

    def print_full_history(self):
        print(f"\n{'='*70}\n  FULL TRAINING HISTORY\n{'='*70}")
        miou_hist = self.history.get('val/miou', [])
        if miou_hist:
            best_ep, best_val = max(miou_hist, key=lambda x: x[1])
            print(f"  Best mIoU: {best_val:.4f} at epoch {best_ep+1}")
            print(f"  Final mIoU: {miou_hist[-1][1]:.4f}")
            if len(miou_hist) >= 10:
                last10 = [v for _, v in miou_hist[-10:]]
                spread = max(last10) - min(last10)
                print(f"  Last-10 spread: {spread:.4f} {'← PLATEAU' if spread<0.003 else ''}")
        print(f"\n  Epoch │ mIoU   │ OHEM   │ Dice   │ gamma4 │ gamma5 │ hard%")
        print(f"  {'─'*65}")
        n = max((len(v) for v in self.history.values()), default=0)
        best_miou = max((v for _, v in self.history.get('val/miou', [(0,0)])), default=0)
        for i in range(n):
            def _g(k):
                h = self.history.get(k, [])
                return h[i][1] if i < len(h) else float('nan')
            miou = _g('val/miou')
            mark = ' ← BEST' if not math.isnan(miou) and miou == best_miou else ''
            print(f"  {i+1:>5} │ {miou:.4f} │ {_g('train/ohem'):.4f} │ "
                  f"{_g('train/dice'):.4f} │ {_g('dwsa/gamma4'):.4f} │ "
                  f"{_g('dwsa/gamma5'):.4f} │ {_g('train/hard_ratio'):.3f}{mark}")
        print(f"{'='*70}\n")

    def close(self): self._f.close()


# ============================================================
# BN RESET (K1)
# ============================================================

def reset_bn_stats(model, momentum=0.3):
    n = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = momentum
    print(f"  K1: Reset {n} BN layers, momentum={momentum}")

def restore_bn_momentum(model, momentum=0.1):
    n = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
    print(f"  K1: BN momentum restored to {momentum} ({n} layers)")


# ============================================================
# WEIGHT LOADING
# ============================================================

def _remap_stem_key(key, N2=4):
    import re
    for pref in ['backbone.', 'model.', 'module.']:
        if key.startswith(pref): key = key[len(pref):]
    m = re.match(r'stem\.(\d+)\.(.+)$', key)
    if not m: return key
    idx, rest = int(m.group(1)), m.group(2)
    def _cm(rest, pref):
        return f'{pref}.{rest[len("conv."):].lstrip(".")}' if rest.startswith('conv.') else None
    if idx == 0:   return _cm(rest, 'stem_conv1.0')
    elif idx == 1: return _cm(rest, 'stem_conv2.0')
    elif 2 <= idx <= 1+N2: return f'stem_stage2.{idx-2}.{rest}'
    else: return f'stem_stage3.{idx-(2+N2)}.{rest}'


def _strip_checkpoint_prefix(key):
    """Remove wrapper prefixes while preserving backbone/head ownership."""
    changed = True
    while changed:
        changed = False
        for prefix in ('module.', 'model.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _map_gcnet_key_to_coming(key):
    """Map only semantically compatible GCNet modules to CoMingNet."""
    prefix_map = (
        ('stem_conv1.0.', 'stem.0.'),
        ('stem_conv2.0.', 'stem.1.'),
        ('compression_1.', 'global_to_local1.'),
        ('compression_2.', 'global_to_local2.'),
        ('down_1.', 'local_to_global1.'),
        ('down_2.', 'local_to_global2.'),
        ('spp.', 'context.'),
    )
    for source, destination in prefix_map:
        if key.startswith(source):
            return destination + key[len(source):]
    return key


def load_pretrained_weights(
    model,
    ckpt_path,
    target_variant='coming',
    source='auto',
    strict_match=False,
):
    """Load same-architecture weights or transfer compatible GCNet weights.

    For GCNet -> CoMingNet, only compatible stem/fusion/context/head tensors
    are transferred. New CoMingBlock parameters remain randomly initialized.
    """
    print(f"Loading pretrained weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = (
        ckpt.get('model')
        or ckpt.get('model_state_dict')
        or ckpt.get('state_dict')
        or ckpt
    )

    backbone_state = model.backbone.state_dict()
    head_state = model.decode_head.state_dict()
    backbone_loaded, head_loaded = {}, {}
    skipped = []

    normalized = {}
    for source_key, value in state.items():
        key = _strip_checkpoint_prefix(source_key)
        normalized[key] = value

    if source == 'auto':
        source = 'coming' if any(
            ('branch_horizontal' in key or 'local_stage1' in key)
            for key in normalized
        ) else 'gcnet'
    print(f"Detected pretrained source: {source}")

    for source_key, value in normalized.items():
        if source_key.startswith('decode_head.'):
            head_key = source_key[len('decode_head.'):]
            if head_key.startswith('conv_seg.'):
                head_key = 'cls_seg.' + head_key[len('conv_seg.'):]
            if head_key in head_state and head_state[head_key].shape == value.shape:
                head_loaded[head_key] = value
            continue

        backbone_key = (
            source_key[len('backbone.') :]
            if source_key.startswith('backbone.')
            else source_key
        )

        if target_variant == 'coming' and source == 'gcnet':
            # Handle the original sequential GCNet stem first.
            if backbone_key.startswith('stem.'):
                mapped_key = backbone_key
            else:
                mapped_key = _map_gcnet_key_to_coming(backbone_key)
        elif target_variant != 'coming':
            mapped_key = _remap_stem_key(backbone_key)
        else:
            mapped_key = backbone_key

        matched = False
        if (
            mapped_key in backbone_state
            and backbone_state[mapped_key].shape == value.shape
        ):
            backbone_loaded[mapped_key] = value
            matched = True

        # Suffix matching is retained only for legacy GCNet variants. It is
        # intentionally disabled for GCNet -> CoMingNet to avoid accidental
        # transfer into an unrelated road block with the same tensor shape.
        if not matched and not strict_match and target_variant != 'coming':
            for candidate in backbone_state:
                if (
                    (candidate.endswith(mapped_key) or mapped_key.endswith(candidate))
                    and backbone_state[candidate].shape == value.shape
                ):
                    backbone_loaded[candidate] = value
                    matched = True
                    break

        if not matched:
            skipped.append(source_key)

    model.backbone.load_state_dict(backbone_loaded, strict=False)
    model.decode_head.load_state_dict(head_loaded, strict=False)

    loaded_backbone_params = sum(
        backbone_state[key].numel() for key in backbone_loaded
    )
    total_backbone_params = sum(t.numel() for t in backbone_state.values())
    backbone_ratio = 100.0 * loaded_backbone_params / max(total_backbone_params, 1)

    print(f"\n{SEP}\nTRANSFER-LEARNING SUMMARY\n{SEP}")
    print(f"Backbone tensors: {len(backbone_loaded):>5} / {len(backbone_state)}")
    print(f"Backbone params:  {loaded_backbone_params:>12,} / "
          f"{total_backbone_params:,} ({backbone_ratio:.2f}%)")
    print(f"Head tensors:     {len(head_loaded):>5} / {len(head_state)}")
    print(f"Skipped tensors:  {len(skipped):>5}")
    if target_variant == 'coming' and source == 'gcnet':
        print("NOTE: CoMingBlock weights are new and were not copied from GCNet.")
    print(f"{SEP}\n")
    return backbone_ratio


# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

def build_optimizer(model, args):
    STEM = {'stem','stem_conv1','stem_conv2','stem_stage2','stem_stage3'}
    dwsa, alpha, stem, backbone, head = [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'dwsa' in name:
            dwsa.append(p)
        elif 'alpha' in name:
            alpha.append(p)
        elif 'backbone' in name:
            part = name.split('.')[1] if len(name.split('.')) > 1 else ''
            if part in STEM:
                stem.append(p)
            else:
                backbone.append(p)
        else:
            head.append(p)

    slr = getattr(args, 'stem_lr_factor', 0.01)
    groups = []
    if head:     groups.append({'params': head,     'lr': args.lr,                          'name': 'head'})
    if backbone: groups.append({'params': backbone, 'lr': args.lr * args.backbone_lr_factor,'name': 'backbone'})
    if stem:     groups.append({'params': stem,     'lr': args.lr * slr,                    'name': 'stem'})
    if dwsa:     groups.append({'params': dwsa,     'lr': args.lr * args.dwsa_lr_factor,    'name': 'dwsa'})
    if alpha:    groups.append({'params': alpha,    'lr': args.lr * args.alpha_lr_factor,   'name': 'alpha'})

    for g in groups: g.setdefault('initial_lr', g['lr'])

    if getattr(args, 'optimizer', 'adamw').lower() == 'sgd':
        opt = torch.optim.SGD(groups, momentum=getattr(args,'sgd_momentum',0.9),
                              weight_decay=args.weight_decay, nesterov=True)
        print(f"Optimizer: SGD (momentum={getattr(args,'sgd_momentum',0.9)})")
    else:
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        print("Optimizer: AdamW")

    for g in groups:
        print(f"  '{g['name']}': lr={g['lr']:.2e}, params={len(g['params'])}")
    return opt


def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    use_cosine = (args.freeze_backbone and args.unfreeze_schedule) or args.scheduler == 'cosine'
    if use_cosine:
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs-start_epoch, eta_min=1e-6)
        print("CosineAnnealingLR")
    elif args.scheduler == 'onecycle':
        steps   = len(train_loader) * (args.epochs - start_epoch)
        max_lrs = [g['initial_lr'] for g in optimizer.param_groups]
        sch = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=steps,
            pct_start=0.05, anneal_strategy='cos',
            cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
            div_factor=25, final_div_factor=100000)
        print(f"OneCycleLR (steps={steps})")
    elif args.scheduler == 'cosine_wr':
        T0 = getattr(args, 'cosine_wr_t0', 10)
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T0, T_mult=1, eta_min=1e-7)
        print(f"CosineAnnealingWarmRestarts (T_0={T0})")
    else:
        sch = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: (1 - e/args.epochs)**0.9)
        print("Polynomial LR")
    return sch


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class OHEMLoss(nn.Module):
    def __init__(self, ignore_index=255, keep_ratio=0.3,
                 min_kept=100000, thresh=None, class_weights=None):
        super().__init__()
        self.ignore_index  = ignore_index
        self.keep_ratio    = keep_ratio
        self.min_kept      = min_kept
        self.thresh        = thresh
        self.class_weights = class_weights
        self.last_hard_ratio = 0.0

    def forward(self, logits, labels):
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_px = F.cross_entropy(logits.float(), labels,
                                  weight=w.float() if w is not None else None,
                                  ignore_index=self.ignore_index,
                                  reduction='none').view(-1)
        valid = labels.view(-1) != self.ignore_index
        loss_px = loss_px[valid]
        n = loss_px.numel()
        if n == 0:
            self.last_hard_ratio = 0.0
            return logits.sum() * 0

        if self.thresh is not None:
            with torch.no_grad():
                probs     = torch.softmax(logits.detach().float(), dim=1).max(1)[0].view(-1)[valid]
                hard_mask = probs < self.thresh
                if hard_mask.sum() < self.min_kept:
                    _, idx = torch.topk(probs, min(self.min_kept, n), largest=False)
                    hard_mask = torch.zeros(n, dtype=torch.bool, device=logits.device)
                    hard_mask[idx] = True
            self.last_hard_ratio = hard_mask.float().mean().item()
            loss_px = loss_px[hard_mask]
        else:
            n_keep = min(max(int(self.keep_ratio * n), min(self.min_kept, n)), n)
            self.last_hard_ratio = n_keep / n
            if n_keep < n:
                thr     = torch.sort(loss_px, descending=True)[0][n_keep-1].detach()
                loss_px = loss_px[loss_px >= thr]
        return loss_px.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255, class_weights=None):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        logits = logits.float()
        B, C, H, W = logits.shape
        valid   = targets != self.ignore_index
        tgt_oh  = F.one_hot(targets.clamp(0,C-1), C).permute(0,3,1,2).float() * valid.unsqueeze(1)
        probs   = F.softmax(logits, dim=1) * valid.unsqueeze(1)
        pf, tf  = probs.reshape(B,C,-1), tgt_oh.reshape(B,C,-1)
        inter   = (pf * tf).sum(2)
        dice    = (2*inter + self.smooth) / (pf.sum(2) + tf.sum(2) + self.smooth)
        loss    = 1.0 - dice
        if self.class_weights is not None:
            loss = loss * self.class_weights.float().unsqueeze(0)
        present = tf.sum(2) > 0
        return (loss * present.float()).sum(1).div(present.float().sum(1).clamp(1)).mean()


# ============================================================
# UTILITIES
# ============================================================

def check_gradients(model, threshold=10.0):
    max_g, max_n = 0.0, ""
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.norm().item()
            if g > max_g: max_g, max_n = g, name
    if max_g > threshold:
        print(f"Large gradient: {max_n[:60]}... = {max_g:.2f}")
    return max_g


def check_spp_bn_health(model, epoch):
    spp = getattr(model.backbone, 'spp', None)
    if spp is None:
        spp = getattr(model.backbone, 'context', None)
    if spp is None: return
    for name, m in spp.named_modules():
        if not isinstance(m, nn.BatchNorm2d): continue
        rv = m.running_var
        if rv is None: continue
        bad = torch.isnan(rv).any() or torch.isinf(rv).any() or rv.min() < 1e-6
        if bad:
            print(f"  ⚠️ SPP BN bad: spp.{name} — resetting")
            m.running_mean.zero_(); m.running_var.fill_(1.0)


def log_dwsa_health(model, epoch, diag):
    print(f"\n  DWSA Health (epoch {epoch+1}):")
    print(f"  {'Stage':<12} {'gamma':>8}  {'Δgamma':>8}  Status")
    print(f"  {'─'*48}")
    for name, tag in [('dwsa_stage4','gamma4'),('dwsa_stage5','gamma5'),('dwsa_stage6','gamma6')]:
        mod = getattr(model.backbone, name, None)
        if mod is None: continue
        g = mod.gamma.item()
        diag.log(epoch, f'dwsa/{tag}', g)
        h = diag.history.get(f'dwsa/{tag}', [])
        delta = f"{g-h[-2][1]:+.5f}" if len(h) >= 2 else '(first)'
        status = ('⚠️  NOT LEARNING' if g < 0.11 else '📈 Warming up' if g < 0.2 else '✅ Active' if g < 0.4 else '🔥 Highly active')
        print(f"  {name:<12} {g:>8.5f}  {delta:>8}  {status}")
    print()


def log_fan_health(model, epoch, diag):
    info = []
    for stem, tag in [('stem_conv1','1'),('stem_conv2','2')]:
        mod = getattr(model.backbone, stem, None)
        if mod is None or len(mod) < 2 or not hasattr(mod[1], 'alpha'): continue
        a   = torch.sigmoid(mod[1].alpha.data)
        info.append((stem, a.mean().item(), a.std().item(), a.min().item(), a.max().item()))
        diag.log(epoch, f'fan/alpha{tag}_mean', a.mean().item())
    if not info: return
    print(f"  FoggyAwareNorm alpha (epoch {epoch+1}):")
    print(f"  {'Layer':<12} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}  Blend")
    print(f"  {'─'*53}")
    for stem, mean, std, mn, mx in info:
        bias = '→ IN' if mean > 0.6 else ('→ BN' if mean < 0.4 else 'balanced')
        print(f"  {stem:<12} {mean:>7.4f} {std:>7.4f} {mn:>7.4f} {mx:>7.4f}  {bias} {'█'*int(mean*20)}")
    print()


def count_trainable_params(model):
    tot  = sum(p.numel() for p in model.parameters())
    tr   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bbt  = sum(p.numel() for p in model.backbone.parameters())
    bbtr = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    hdt  = sum(p.numel() for p in model.decode_head.parameters())
    hdtr = sum(p.numel() for p in model.decode_head.parameters() if p.requires_grad)
    print(f"\n{SEP}\nPARAMETER STATISTICS\n{SEP}")
    print(f"Total:      {tot:>15,} | 100%")
    print(f"Trainable:  {tr:>15,} | {100*tr/tot:.1f}%")
    print(f"Frozen:     {tot-tr:>15,} | {100*(tot-tr)/tot:.1f}%")
    print(f"{'─'*70}")
    print(f"Backbone:   {bbtr:>15,} / {bbt:,} | {100*bbtr/max(bbt,1):.1f}%")
    print(f"Head:       {hdtr:>15,} / {hdt:,} | {100*hdtr/max(hdt,1):.1f}%")
    print(f"{SEP}\n")


def freeze_backbone(model, variant='fan_dwsa'):
    has_dwsa = hasattr(model.backbone, 'dwsa_stage4')
    has_fan  = (hasattr(model.backbone, 'stem_conv1') and
                len(model.backbone.stem_conv1) > 1 and
                hasattr(model.backbone.stem_conv1[1], 'alpha'))
    print(f"Freezing backbone...")
    for p in model.backbone.parameters(): p.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None: m.weight.requires_grad = False
            if m.bias   is not None: m.bias.requires_grad   = False

    if has_dwsa:
        for name in ['dwsa_stage4','dwsa_stage5','dwsa_stage6']:
            mod = getattr(model.backbone, name, None)
            if mod is None: continue
            for p in mod.parameters(): p.requires_grad = True
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
                    if m.weight is not None: m.weight.requires_grad = True
                    if m.bias   is not None: m.bias.requires_grad   = True

    if has_fan:
        for name in ['stem_conv1','stem_conv2']:
            mod = getattr(model.backbone, name, None)
            if mod is None or len(mod) < 2 or not hasattr(mod[1], 'alpha'): continue
            for p in mod[1].parameters(): p.requires_grad = True
            mod[1].bn.train()
            if mod[1].bn.weight is not None: mod[1].bn.weight.requires_grad = True
            if mod[1].bn.bias   is not None: mod[1].bn.bias.requires_grad   = True
    print("Backbone frozen\n")


def unfreeze_backbone_progressive(model, stage_names):
    if isinstance(stage_names, str): stage_names = [stage_names]
    total = 0
    for name in stage_names:
        mod = getattr(model.backbone, name, None)
        if mod is None and '.' in name:
            parts = name.split('.', 1)
            base  = getattr(model.backbone, parts[0], None)
            if base is not None and parts[1].isdigit():
                try: mod = base[int(parts[1])]
                except: pass
        if mod is None: print(f"  [skip] '{name}' not found"); continue
        cnt = 0
        for p in mod.parameters():
            if not p.requires_grad: p.requires_grad = True; cnt += 1
        for m in mod.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
                if m.weight is not None: m.weight.requires_grad = True
                if m.bias   is not None: m.bias.requires_grad   = True
        total += cnt
        if cnt: print(f"  Unfrozen: backbone.{name} ({cnt:,} params)")
    print(f"  Total unfrozen: {total:,} params\n")


def print_backbone_structure(model):
    print(f"\n{SEP}\n BACKBONE STRUCTURE\n{SEP}")
    for name, mod in model.backbone.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if isinstance(mod, nn.ModuleList):
            print(f"  {name}: ModuleList[{len(mod)}]  ({n:,} params)")
            for i, sub in enumerate(mod):
                sp = sum(p.numel() for p in sub.parameters())
                print(f"    [{i}]: {type(sub).__name__}  ({sp:,} params)")
        else:
            print(f"  {name}: {type(mod).__name__}  ({n:,} params)")
    print(f"{SEP}\n")


def get_context_module(backbone):
    """Return GCNet SPP or CoMingNet context module."""
    module = getattr(backbone, 'spp', None)
    return module if module is not None else getattr(backbone, 'context', None)


def freeze_module_and_bn(module):
    if module is None:
        return 0
    frozen = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            frozen += parameter.numel()
        parameter.requires_grad = False
    for child in module.modules():
        if isinstance(child, nn.BatchNorm2d):
            child.eval()
    return frozen


def freeze_stem_only(model, variant, verbose=True):
    """Freeze the selected backbone stem without assuming GCNet key names."""
    if variant == 'coming':
        frozen = freeze_module_and_bn(getattr(model.backbone, 'stem', None))
        if verbose:
            print(f"CoMingNet stem frozen: {frozen:,} params")
        return frozen

    frozen = 0
    for stem_name in ('stem_conv1', 'stem_conv2'):
        module = getattr(model.backbone, stem_name, None)
        if module is None:
            continue
        for parameter_name, parameter in module.named_parameters():
            # Preserve trainable FAN parameters in legacy variants.
            if not any(key in parameter_name for key in ('alpha', 'bn.', 'in_.')):
                if parameter.requires_grad:
                    frozen += parameter.numel()
                parameter.requires_grad = False

    for stem_name in ('stem_stage2', 'stem_stage3'):
        frozen += freeze_module_and_bn(getattr(model.backbone, stem_name, None))
    if verbose:
        print(f"GCNet stem frozen: {frozen:,} params (FAN remains trainable)")
    return frozen


# ============================================================
# MODEL CONFIG
# ============================================================

class ModelConfig:
    @staticmethod
    def get_config(
        variant='fan_dwsa',
        coming_kernel_size=7,
        local_blocks=(2, 2, 2),
        global_blocks=(2, 3, 2),
    ):
        C = 32
        if variant == 'coming':
            return {
                "backbone": {
                    "in_channels": 3,
                    "channels": C,
                    "ppm_channels": 128,
                    "local_blocks": tuple(local_blocks),
                    "global_blocks": tuple(global_blocks),
                    "kernel_size": coming_kernel_size,
                    "align_corners": False,
                    "deploy": False,
                    "zero_init_residual": False,
                    "norm_cfg": dict(type='BN', requires_grad=True),
                    "act_cfg": dict(type='ReLU', inplace=True),
                },
                "head": {
                    "in_channels": C * 4,
                    "channels": 64,
                    "align_corners": False,
                    "dropout_ratio": 0.1,
                    "loss_weight_aux": 0.4,
                    "norm_cfg": dict(type='BN', requires_grad=True),
                    "act_cfg": dict(type='ReLU', inplace=True),
                },
                "loss": {
                    "ce_weight": 1.0,
                    "dice_weight": 0.5,
                    "dice_smooth": 1e-5,
                },
            }

        bb = {
            "in_channels": 3, "channels": C, "ppm_channels": 128,
            "num_blocks_per_stage": [4, 4, [5,4], [5,4], [2,2]],
            "align_corners": False, "deploy": False,
            "norm_cfg": dict(type='BN', requires_grad=True),
            "act_cfg":  dict(type='ReLU', inplace=True),
        }
        if variant in ('fan_dwsa', 'dwsa_only'):
            bb["dwsa_reduction"] = 8
        return {
            "backbone": bb,
            "head": {
                "in_channels": C*4, "channels": 64,
                "align_corners": False, "dropout_ratio": 0.1,
                "loss_weight_aux": 0.4,
                "norm_cfg": dict(type='BN', requires_grad=True),
                "act_cfg":  dict(type='ReLU', inplace=True),
            },
            "loss": {"ce_weight": 1.0, "dice_weight": 0.5, "dice_smooth": 1e-5},
        }


# ============================================================
# SEGMENTOR
# ============================================================

class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        return self.decode_head(self.backbone(x))

    def forward_train(self, x):
        return {"main": self.decode_head(self.backbone(x))}


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, args,
                 class_weights=None, diag=None):
        self.model       = model.to(device)
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.args        = args
        self.best_miou   = 0.0
        self.start_epoch = 0
        self.global_step = 0
        self.diag        = diag

        lcfg = args.loss_config
        self.ce_weight    = lcfg['ce_weight']
        self.dice_weight  = lcfg['dice_weight']
        self.base_loss_cfg = lcfg
        self.loss_phase   = 'full'

        cw = class_weights.to(device) if class_weights is not None else None
        self.ohem = OHEMLoss(
            ignore_index=args.ignore_index,
            keep_ratio=getattr(args,'ohem_keep_ratio',0.3),
            min_kept=getattr(args,'ohem_min_kept',100000),
            thresh=getattr(args,'ohem_thresh',None),
            class_weights=class_weights)

        self.dice = DiceLoss(smooth=lcfg['dice_smooth'],
                             ignore_index=args.ignore_index,
                             class_weights=class_weights)
        _ls = getattr(args, 'label_smoothing', 0.0)
        self.ce = nn.CrossEntropyLoss(weight=cw, ignore_index=args.ignore_index,
                                      label_smoothing=_ls)

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = _make_writer(self.save_dir / "tensorboard")
        self._save_config()
        self._print_config()

        ohem_mode = (f"threshold-based (thresh={getattr(args,'ohem_thresh',None)})"
                     if getattr(args,'ohem_thresh',None)
                     else f"ratio-based (keep_ratio={getattr(args,'ohem_keep_ratio',0.3)})")
        print(f"OHEM: {ohem_mode}")
        if _ls > 0: print(f"Label smoothing: {_ls}")

    def _save_config(self):
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2, default=str)

    def _print_config(self):
        print(f"\n{SEP}\nTRAINER CONFIGURATION\n{SEP}")
        print(f"Batch size:            {self.args.batch_size}")
        print(f"Gradient accumulation: {self.args.accumulation_steps}")
        print(f"Effective batch:       {self.args.batch_size * self.args.accumulation_steps}")
        print(f"Mixed precision:       {self.args.use_amp}")
        print(f"Gradient clipping:     {self.args.grad_clip}")
        print(f"Loss: CE({self.ce_weight}) + Dice({self.dice_weight})")
        print(f"{SEP}\n")

    def set_loss_phase(self, phase):
        if phase == self.loss_phase: return
        self.dice_weight = 0.0 if phase == 'ce_only' else self.base_loss_cfg['dice_weight']
        self.loss_phase  = phase
        print(f"Loss phase → {phase}  (CE={self.ce_weight}, Dice={self.dice_weight})")

    def train_epoch(self, loader, epoch):
        self.model.train()

        # Re-apply freezes each epoch (model.train() doesn't restore requires_grad)
        if getattr(self.args, "freeze_spp_bn", False):
            spp = get_context_module(self.model.backbone)
            if spp:
                for p in spp.parameters(): p.requires_grad = False
                for m in spp.modules():
                    if isinstance(m, nn.BatchNorm2d): m.eval()

        if getattr(self.args, "freeze_stem_conv", False):
            freeze_stem_only(self.model, self.args.model_variant, verbose=False)

        total_loss = total_ohem = total_dice = 0.0
        max_grad_epoch = hard_ratio_acc = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4: masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                c4_logit, c6_logit = self.model.forward_train(imgs)["main"]
                target_size = masks.shape[-2:]
                c4_full = F.interpolate(c4_logit, size=target_size,
                                        mode='bilinear', align_corners=False)
                c6_full = F.interpolate(c6_logit, size=target_size,
                                        mode='bilinear', align_corners=False)

                ohem_loss = self.ohem(c6_full, masks)

                if self.dice_weight > 0:
                    masks_small = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=c6_logit.shape[-2:], mode='nearest'
                    ).squeeze(1).long()
                    dice_loss = self.dice(c6_logit, masks_small)
                else:
                    dice_loss = torch.tensor(0.0, device=self.device)

                task_loss = self.ce_weight * ohem_loss + self.dice_weight * dice_loss

                if self.args.aux_weight > 0:
                    aux_decay = getattr(self.args, 'aux_decay_exp', 0.9)
                    aux_w     = self.args.aux_weight * (1 - epoch / self.args.epochs) ** aux_decay
                    task_loss = task_loss + aux_w * self.ohem(c4_full, masks)

                loss = task_loss / self.args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf loss at batch {batch_idx} — skipping")
                self.optimizer.zero_grad(set_to_none=True); continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                mg = check_gradients(self.model, threshold=10.0)
                max_grad_epoch = max(max_grad_epoch, mg)
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.scheduler and self.args.scheduler == 'onecycle':
                    self.scheduler.step()

            total_loss  += loss.item() * self.args.accumulation_steps
            total_ohem  += ohem_loss.item()
            total_dice  += dice_loss.item()
            hard_ratio_acc += self.ohem.last_hard_ratio

            pbar.set_postfix({
                'loss': f'{loss.item()*self.args.accumulation_steps:.4f}',
                'ohem': f'{ohem_loss.item():.4f}',
                'dice': f'{dice_loss.item():.4f}',
                'lr':   f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'hard%':f'{self.ohem.last_hard_ratio:.2f}',
                'mg':   f'{mg:.2f}',
            })
            if batch_idx % 200 == 0: torch.cuda.empty_cache()

        n = len(loader)
        avg_hr = hard_ratio_acc / n
        print(f"\nEpoch {epoch+1} — Max grad: {max_grad_epoch:.2f}  |  Hard%: {avg_hr:.3f}")
        print(f"  LR head={self.optimizer.param_groups[0]['lr']:.2e}")

        torch.cuda.empty_cache()
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        result = {'loss': total_loss/n, 'ohem': total_ohem/n,
                  'dice': total_dice/n, 'hard_ratio': avg_hr}
        if self.diag:
            self.diag.log_dict(epoch, result, prefix='train/')
            self.diag.log(epoch, 'train/max_grad', max_grad_epoch)
        return result

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.model.eval()
        total_loss = 0.0
        C  = self.args.num_classes
        cm = np.zeros((C, C), dtype=np.int64)
        pbar = tqdm(loader, desc="Validation")

        for batch_idx, (imgs, masks) in enumerate(pbar):
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            with autocast(device_type='cuda', enabled=self.args.use_amp):
                logits = self.model(imgs)
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode='bilinear', align_corners=False)
                loss   = self.ce(logits, masks)

            total_loss += loss.item()
            pred   = logits.argmax(1).cpu().numpy()
            target = masks.cpu().numpy()
            valid  = (target >= 0) & (target < C)
            lbl    = C * target[valid].astype(int) + pred[valid]
            cm    += np.bincount(lbl, minlength=C * C).reshape(C, C)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            if batch_idx % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        inter = np.diag(cm)
        union = cm.sum(1) + cm.sum(0) - inter
        iou   = inter / (union + 1e-10)
        result = {
            'loss'         : total_loss / len(loader),
            'miou'         : float(np.nanmean(iou)),
            'accuracy'     : float(inter.sum() / (cm.sum() + 1e-10)),
            'per_class_iou': iou,
        }
        if self.diag:
            self.diag.log(epoch, 'val/miou',     result['miou'])
            self.diag.log(epoch, 'val/loss',     result['loss'])
            self.diag.log(epoch, 'val/accuracy', result['accuracy'])
        return result

    def save_checkpoint(self, epoch, metrics, is_best=False):
        ckpt = {
            'epoch': epoch, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
            'best_miou': self.best_miou, 'metrics': metrics,
            'global_step': self.global_step,
        }
        torch.save(ckpt, self.save_dir / "last.pth")
        if is_best:
            torch.save(ckpt, self.save_dir / "best.pth")
            print(f"Best model saved! mIoU: {metrics['miou']:.4f}")
        if (epoch + 1) % self.args.save_interval == 0:
            torch.save(ckpt, self.save_dir / f"epoch_{epoch+1}.pth")

    def load_checkpoint(self, path, reset_epoch=True, load_optimizer=True, reset_best_metric=False):
        ckpt  = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get('model') or ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
        self.model.load_state_dict(state, strict=False)
        if load_optimizer and not reset_epoch:
            try: self.optimizer.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError) as e: print(f"Optimizer not loaded: {e}")
            if self.scheduler and ckpt.get('scheduler'):
                try: self.scheduler.load_state_dict(ckpt['scheduler'])
                except Exception as e: print(f"Scheduler not loaded: {e}")
            if 'scaler' in ckpt and ckpt['scaler']:
                try: self.scaler.load_state_dict(ckpt['scaler'])
                except Exception as e: print(f"Scaler not loaded: {e}")
        if reset_epoch:
            self.start_epoch = 0; self.global_step = 0
            self.best_miou   = 0.0 if reset_best_metric else ckpt.get('best_miou', 0.0)
            print(f"Weights loaded (epoch {ckpt.get('epoch','?')}), starting from 0")
        else:
            self.start_epoch = ckpt['epoch'] + 1
            self.best_miou   = ckpt.get('best_miou', 0.0)
            self.global_step = ckpt.get('global_step', 0)
            print(f"Resuming from epoch {self.start_epoch}")


# ============================================================
# CONSTANTS
# ============================================================

UNFREEZE_STAGES_GCNET = [
    ['stem_conv1','stem_conv2','stem_stage2','stem_stage3','compression_1','down_1'],
    ['semantic_branch_layers.0','detail_branch_layers.0','dwsa_stage4'],
    ['semantic_branch_layers.1','detail_branch_layers.1','dwsa_stage5','compression_2','down_2'],
    ['semantic_branch_layers.2','detail_branch_layers.2','dwsa_stage6','spp'],
]

UNFREEZE_STAGES_COMING = [
    ['stem'],
    ['local_stage1','global_stage1','global_to_local1','local_to_global1'],
    ['local_stage2','global_stage2','global_to_local2','local_to_global2'],
    ['local_transition','local_stage3','global_stage3','context',
     'local_projection','final_refine'],
]

CLASS_NAMES = ['road','sidewalk','building','wall','fence','pole',
               'traffic_light','traffic_sign','vegetation','terrain',
               'sky','person','rider','car','truck','bus',
               'train','motorcycle','bicycle']


# ============================================================
# MAIN
# ============================================================

def _parse_stage_blocks(value):
    blocks = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if len(blocks) != 3 or any(item < 1 for item in blocks):
        raise argparse.ArgumentTypeError(
            "Stage blocks must contain three positive integers, e.g. 2,2,2."
        )
    return blocks


def main():
    parser = argparse.ArgumentParser(description="GCNet / CoMingNet Training")
    # Model
    parser.add_argument("--model_variant",      type=str, default="fan_dwsa",
                        choices=["fan_dwsa","fan_only","dwsa_only","coming"])
    parser.add_argument("--pretrained_weights", type=str, default=None)
    parser.add_argument("--pretrained_source",  type=str, default="auto",
                        choices=["auto","gcnet","coming"])
    parser.add_argument("--coming_kernel_size", type=int, default=7)
    parser.add_argument("--local_blocks", type=_parse_stage_blocks, default=(2,2,2),
                        help="CoMingNet local blocks, e.g. 2,2,2")
    parser.add_argument("--global_blocks", type=_parse_stage_blocks, default=(2,3,2),
                        help="CoMingNet global blocks, e.g. 2,3,2")
    # Backbone freeze/unfreeze
    parser.add_argument("--freeze_backbone",    action="store_true")
    parser.add_argument("--unfreeze_schedule",  type=str, default="")
    parser.add_argument("--freeze_stem_conv",   action="store_true")
    parser.add_argument("--freeze_spp_bn",      action="store_true")
    # LR factors
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1)
    parser.add_argument("--dwsa_lr_factor",     type=float, default=0.5)
    parser.add_argument("--alpha_lr_factor",    type=float, default=0.1)
    parser.add_argument("--stem_lr_factor",     type=float, default=0.01)
    # Data
    parser.add_argument("--image_dir", type=str, default=(
        "/kaggle/input/datasets/balraj98/massachusetts-roads-dataset/"
        "tiff/train"))
    parser.add_argument("--mask_dir", type=str, default=(
        "/kaggle/input/datasets/balraj98/massachusetts-roads-dataset/"
        "tiff/train_labels"))
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Deterministic validation fraction split from the paired folders")
    parser.add_argument("--num_classes",        type=int, default=2)
    parser.add_argument("--ignore_index",       type=int, default=255)
    parser.add_argument("--use_class_weights",  action="store_true")
    parser.add_argument("--class_weights_file", type=str, default=None)
    # Training
    parser.add_argument("--epochs",             type=int,   default=100)
    parser.add_argument("--batch_size",         type=int,   default=4)
    parser.add_argument("--accumulation_steps", type=int,   default=2)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--weight_decay",       type=float, default=1e-4)
    parser.add_argument("--optimizer",          type=str,   default="adamw",
                        choices=["adamw","sgd"])
    parser.add_argument("--sgd_momentum",       type=float, default=0.9)
    parser.add_argument("--grad_clip",          type=float, default=5.0)
    parser.add_argument("--scheduler",          default="cosine",
                        choices=["onecycle","poly","cosine","cosine_wr"])
    parser.add_argument("--cosine_wr_t0",       type=int,   default=10)
    # Loss
    parser.add_argument("--aux_weight",         type=float, default=0.4)
    parser.add_argument("--aux_decay_exp",      type=float, default=0.9)
    parser.add_argument("--dice_weight",        type=float, default=None)
    parser.add_argument("--ce_weight",          type=float, default=None)
    parser.add_argument("--label_smoothing",    type=float, default=0.0)
    parser.add_argument("--ohem_keep_ratio",    type=float, default=0.3)
    parser.add_argument("--ohem_min_kept",      type=int,   default=100000)
    parser.add_argument("--ohem_thresh",        type=float, default=None)
    # Resolution
    parser.add_argument("--img_h",              type=int,   default=512)
    parser.add_argument("--img_w",              type=int,   default=1024)
    # BN warmup (K1)
    parser.add_argument("--reset_bn_stats",     action="store_true")
    parser.add_argument("--bn_warmup_epochs",   type=int,   default=3)
    parser.add_argument("--bn_warmup_momentum", type=float, default=0.3)
    # Misc
    parser.add_argument("--use_amp",            action="store_true", default=True)
    parser.add_argument("--num_workers",        type=int,   default=4)
    parser.add_argument("--save_dir",           default="./checkpoints")
    parser.add_argument("--resume",             type=str,   default=None)
    parser.add_argument("--resume_mode",        type=str,   default="transfer",
                        choices=["transfer","continue"])
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--save_interval",      type=int,   default=10)
    parser.add_argument("--reset_best_metric",  action="store_true")
    parser.add_argument("--diag_interval",      type=int,   default=1)
    parser.add_argument("--ce_only_epochs_after_unfreeze", type=int, default=3)
    args = parser.parse_args()

    # Validate unfreeze schedule
    unfreeze_list = []
    if args.freeze_backbone and args.unfreeze_schedule:
        unfreeze_list = sorted(int(e) for e in args.unfreeze_schedule.split(','))
        if max(unfreeze_list) >= args.epochs:
            raise ValueError("unfreeze_schedule epoch >= total epochs")
        if args.scheduler == 'onecycle':
            args.scheduler = 'cosine'
            print("[INFO] scheduler auto-switched: onecycle → cosine")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{SEP}\nSEGMENTATION TRAINING  |  {args.model_variant}\n{SEP}")
    print(f"Device: {device}  |  Image: {args.img_h}x{args.img_w}")
    print(f"Epochs: {args.epochs}  |  Scheduler: {args.scheduler}")
    print(f"Grad clip: {args.grad_clip}  |  AMP: {args.use_amp}")
    if args.reset_bn_stats:
        print(f"K1: BN Reset (warmup={args.bn_warmup_epochs} ep, mom={args.bn_warmup_momentum})")
    print(f"{SEP}\n")

    # Import backbone
    if args.model_variant == 'coming':
        from modeling.backbone import CoMingNet as Backbone

    cfg = ModelConfig.get_config(
        variant=args.model_variant,
        coming_kernel_size=args.coming_kernel_size,
        local_blocks=args.local_blocks,
        global_blocks=args.global_blocks,
    )
    args.loss_config = cfg["loss"]
    class_names = (
        CLASS_NAMES
        if args.num_classes == len(CLASS_NAMES)
        else ['background', 'road']
        if args.num_classes == 2
        else [f'class_{index}' for index in range(args.num_classes)]
    )

    # DataLoaders
    train_loader, val_loader, class_weights = create_folder_dataloaders(
        image_dir=args.image_dir, mask_dir=args.mask_dir,
        val_ratio=args.val_ratio, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w), num_classes=args.num_classes,
        compute_class_weights=args.use_class_weights)

    if getattr(args, "class_weights_file", None):
        cw_path = Path(args.class_weights_file)
        if cw_path.exists():
            class_weights = torch.load(cw_path, map_location="cpu")
            print(f"Class weights: {cw_path}  "
                  f"(min={class_weights.min():.3f}, max={class_weights.max():.3f})")
        else:
            print(f"WARNING: {cw_path} not found"); class_weights = None

    # Build model
    model = Segmentor(Backbone(**cfg["backbone"]),
                      GCNetHead(**cfg["head"], num_classes=args.num_classes,
                                ignore_index=args.ignore_index)).to(device)
    model.apply(init_weights)
    check_model_health(model)

    transfer_ratio = None
    if args.pretrained_weights:
        transfer_ratio = load_pretrained_weights(
            model,
            args.pretrained_weights,
            target_variant=args.model_variant,
            source=args.pretrained_source,
        )
        if (
            args.model_variant == 'coming'
            and args.freeze_backbone
            and transfer_ratio < 60.0
        ):
            print(
                "WARNING: less than 60% of CoMingNet backbone parameters were "
                "transferred. Freezing the whole backbone will also freeze "
                "randomly initialized CoMingBlocks. For GCNet -> CoMingNet, "
                "omit --freeze_backbone and use a small backbone LR instead."
            )
    if args.freeze_backbone:
        freeze_backbone(model, variant=args.model_variant)

    count_trainable_params(model)
    print_backbone_structure(model)

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, train_loader)

    save_path = Path(args.save_dir); save_path.mkdir(parents=True, exist_ok=True)
    diag    = DiagnosticLogger(save_dir=save_path, class_names=class_names)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler,
                      device=device, args=args,
                      class_weights=class_weights if args.use_class_weights else None,
                      diag=diag)

    if args.dice_weight is not None:
        trainer.dice_weight = args.dice_weight
        trainer.base_loss_cfg["dice_weight"] = args.dice_weight
    if args.ce_weight is not None:
        trainer.ce_weight = args.ce_weight

    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            reset_epoch=(args.resume_mode == "transfer"),
            load_optimizer=(args.resume_mode == "continue"),
            reset_best_metric=args.reset_best_metric)

    if args.reset_bn_stats:
        reset_bn_stats(model, momentum=args.bn_warmup_momentum)

    # Freeze stem after checkpoint load
    if args.freeze_stem_conv:
        freeze_stem_only(model, args.model_variant)
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=trainer.start_epoch)
        trainer.optimizer = optimizer; trainer.scheduler = scheduler

    # Freeze SPP
    if args.freeze_spp_bn:
        spp = get_context_module(model.backbone)
        if spp:
            frozen = sum(p.numel() for p in spp.parameters() if p.requires_grad)
            for p in spp.parameters(): p.requires_grad = False
            for m in spp.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
            print(f"Context/SPP frozen: {frozen:,} params")
            optimizer = build_optimizer(model, args)
            scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=trainer.start_epoch)
            trainer.optimizer = optimizer; trainer.scheduler = scheduler

    print(f"\n{SEP}\nSTARTING TRAINING\n{SEP}\n")
    applied_unfreeze = set()
    unfreeze_stages = (
        UNFREEZE_STAGES_COMING
        if args.model_variant == 'coming'
        else UNFREEZE_STAGES_GCNET
    )

    for epoch in range(trainer.start_epoch, args.epochs):

        # K1: Restore BN momentum after warmup
        if args.reset_bn_stats and epoch == trainer.start_epoch + args.bn_warmup_epochs:
            restore_bn_momentum(model)

        # Progressive unfreeze
        if epoch in unfreeze_list and epoch not in applied_unfreeze:
            idx = unfreeze_list.index(epoch)
            if idx < len(unfreeze_stages):
                print(f"[Epoch {epoch+1}] Unfreeze stage {idx+1}/{len(unfreeze_stages)}")
                unfreeze_backbone_progressive(model, unfreeze_stages[idx])
                applied_unfreeze.add(epoch)
                optimizer = build_optimizer(model, args)
                scheduler = build_scheduler(optimizer, args, train_loader, start_epoch=epoch)
                trainer.optimizer = optimizer; trainer.scheduler = scheduler
                trainer.set_loss_phase('ce_only')

        if unfreeze_list and trainer.loss_phase == 'ce_only':
            last_un = max((e for e in unfreeze_list if e in applied_unfreeze and e <= epoch), default=None)
            if last_un is not None and epoch >= last_un + args.ce_only_epochs_after_unfreeze:
                trainer.set_loss_phase('full')

        check_spp_bn_health(model, epoch)

        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics   = trainer.validate(val_loader, epoch)

        if epoch % args.diag_interval == 0:
            log_dwsa_health(model, epoch, diag)
            log_fan_health(model,  epoch, diag)

        # Per-class IoU
        iou_arr = val_metrics['per_class_iou']
        print(f"\n  Per-class IoU (epoch {epoch+1}):")
        print(f"  {'Class':<16} {'IoU':>6}  Bar")
        print(f"  {'─'*43}")
        for cname, ciou in zip(class_names, iou_arr):
            mark = ' ⚠️' if ciou < 0.4 else (' ★' if ciou > 0.75 else '')
            print(f"  {cname:<16} {ciou:>6.4f}  {'█'*int(ciou*20)}{mark}")
        low = [n for n, v in zip(class_names, iou_arr) if v < 0.4]
        if low: print(f"\n  ⚠️  LOW (<0.4): {low}")

        diag.log(epoch, 'iou/best',  float(max(iou_arr)))
        diag.log(epoch, 'iou/worst', float(min(iou_arr)))

        print(f"\n{SEP}\nEpoch {epoch+1}/{args.epochs}\n{SEP}")
        print(f"Train — Loss: {train_metrics['loss']:.4f} | "
              f"OHEM: {train_metrics['ohem']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | "
              f"Hard%: {train_metrics['hard_ratio']:.3f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | "
              f"mIoU: {val_metrics['miou']:.4f}  | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"{SEP}\n")

        diag.print_epoch_summary(epoch)

        is_best = val_metrics['miou'] > trainer.best_miou
        if is_best:
            trainer.best_miou = val_metrics['miou']
            print(f"  ★ NEW BEST mIoU: {trainer.best_miou:.4f}")
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)

    diag.print_full_history()
    diag.close()
    trainer.writer.close()

    print(f"\n{SEP}\nTRAINING COMPLETED!\nBest mIoU: {trainer.best_miou:.4f}\n{SEP}\n")


if __name__ == "__main__":
    main()
