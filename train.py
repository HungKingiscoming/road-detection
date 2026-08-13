import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from collections import defaultdict, deque
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
def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm.
    
    This is CRITICAL for training with batch_size < 16.
    BatchNorm becomes unreliable when batch size is small.
    GroupNorm works perfectly even with batch_size=1.
    
    Args:
        module: PyTorch module (model)
        num_groups: Number of groups for GroupNorm (default 32)
    
    Returns:
        Module with GroupNorm instead of BatchNorm
    
    Example:
        >>> model = replace_bn_with_gn(model)
        >>> print(model)  # Should not have BatchNorm2d anymore
    """
    # If the module itself is BatchNorm, replace it
    if isinstance(module, nn.BatchNorm2d):
        num_channels = module.num_features
        
        # Ensure num_groups divides num_channels evenly
        current_groups = num_groups
        while num_channels % current_groups != 0:
            current_groups //= 2
        
        # Create GroupNorm with same number of channels
        return nn.GroupNorm(current_groups, num_channels)
    
    # Otherwise, recursively iterate over children
    for name, child in module.named_children():
        module.add_module(name, replace_bn_with_gn(child, num_groups))
    
    return module



def init_weights(module):
    """
    Apply robust Kaiming (He) initialization for training from scratch.
    
    This is CRITICAL for from-scratch training without pretrained weights.
    Default initialization is too weak. Kaiming init jumpstarts learning.
    
    Args:
        module: PyTorch module (usually apply with model.apply(init_weights))
    
    Example:
        >>> model.apply(init_weights)
        >>> # Now model has proper Kaiming initialization
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        # Kaiming Normal (He Init) for ReLU/GeLU networks
        # Fan-out mode: good for conv layers
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize bias to 0
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        # Normalization layers: weight=1.0, bias=0.0
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def count_parameters(model):
    """
    Count total trainable parameters in the model.
    
    Args:
        model: PyTorch module
    
    Returns:
        int: Number of trainable parameters
    
    Example:
        >>> total = count_parameters(model)
        >>> print(f"Model has {total:,} parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    """Road segmentation dataset.

    [FIX] Resizing a square source tile (Massachusetts Roads tiles are
    1500x1500) to a non-square target (e.g. img_h=512, img_w=1024, as used
    in one run) stretches every road by a *different* factor horizontally
    vs. vertically. For a class defined almost entirely by shape (width,
    curvature, connectivity) that distortion actively hurts learning —
    worse than plain downsampling. When `crop_size` is given, we instead
    resize so the short side matches `crop_size` (aspect-ratio preserving)
    and crop a square patch, so road geometry is never stretched.

    [NEW] `road_oversample_tries`: plain random cropping on this dataset
    would draw mostly pure-background patches, since roads are a thin,
    sparse minority class. When training, we sample up to
    `road_oversample_tries` candidate crop locations and keep the one with
    the most road pixels, biasing training batches towards patches that
    actually contain road to learn from.
    """

    def __init__(self, pairs, img_size, num_classes=2, augment=False,
                 crop_size=None, road_oversample_tries=0):
        self.pairs = list(pairs)
        self.img_size = tuple(img_size)  # (height, width), used only if crop_size is None
        self.num_classes = num_classes
        self.augment = augment
        self.crop_size = crop_size
        self.road_oversample_tries = road_oversample_tries if augment else 0

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

    def _resize_short_side(self, image, mask, target):
        w, h = image.size
        scale = target / min(w, h)
        new_w, new_h = max(target, round(w * scale)), max(target, round(h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        return image, mask

    def _crop(self, image, mask, x, y, size):
        box = (x, y, x + size, y + size)
        return image.crop(box), mask.crop(box)

    def _road_aware_random_crop(self, image, mask, size):
        w, h = image.size
        max_x, max_y = max(0, w - size), max(0, h - size)
        best = None
        for _ in range(max(1, self.road_oversample_tries)):
            x = int(torch.randint(0, max_x + 1, (1,)).item())
            y = int(torch.randint(0, max_y + 1, (1,)).item())
            road_px = int((np.asarray(mask)[y:y + size, x:x + size] > 0).sum())
            if best is None or road_px > best[0]:
                best = (road_px, x, y)
            if road_px > 0:
                break
        _, x, y = best
        return self._crop(image, mask, x, y, size)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert('RGB')
        mask = Image.fromarray(self._read_mask(mask_path))

        if self.crop_size:
            image, mask = self._resize_short_side(image, mask, self.crop_size)
            if self.augment:
                image, mask = self._road_aware_random_crop(image, mask, self.crop_size)
            else:
                w, h = image.size
                x, y = (w - self.crop_size) // 2, (h - self.crop_size) // 2
                image, mask = self._crop(image, mask, x, y, self.crop_size)
        else:
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
            # [NEW] 90-degree rotations: roads run in every direction in
            # aerial imagery and this augmentation is "free" (no
            # interpolation blur/artifacts unlike arbitrary-angle rotation).
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                image = torch.rot90(image, k, dims=(-2, -1))
                mask = torch.rot90(mask, k, dims=(-2, -1))

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
                              compute_class_weights=False,
                              crop_size=None, road_oversample_tries=4):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    pairs = build_image_mask_pairs(image_dir, mask_dir)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    val_count = max(1, min(len(pairs) - 1, round(len(pairs) * val_ratio)))
    val_pairs = [pairs[i] for i in order[:val_count]]
    train_pairs = [pairs[i] for i in order[val_count:]]

    # [FIX] Warn loudly about aspect-ratio-distorting resize, which silently
    # stretches road width/curvature and is easy to miss (it trained
    # "successfully", just worse — no error, no crash).
    if not crop_size and img_size[0] != img_size[1]:
        print(f"  ⚠️  WARNING: img_size={img_size} is non-square while source "
              f"tiles are square. Resizing directly to a non-square target "
              f"stretches roads by different factors horizontally vs. "
              f"vertically, distorting the exact shape info the model needs "
              f"for a thin, curved class like 'road'. Prefer --crop_size "
              f"(aspect-preserving square crop) or a square img_h/img_w.")

    train_set = RoadFolderDataset(train_pairs, img_size, num_classes, augment=True,
                                  crop_size=crop_size, road_oversample_tries=road_oversample_tries)
    val_set = RoadFolderDataset(val_pairs, img_size, num_classes, augment=False,
                                crop_size=crop_size)
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
            ('val/miou_smooth',  'Val mIoU (smooth)','.4f'),
            ('val/accuracy',     'Val Accuracy',    '.4f'),
            ('val/loss',         'Val Loss',        '.4f'),
            ('train/bce',        'Train BCE',       '.4f'),
            ('train/dice',       'Train Dice',      '.4f'),
            ('train/max_grad',   'Max Gradient',    '.3f'),
            ('dwsa/gamma4',      'DWSA gamma4',     '.4f'),
            ('dwsa/gamma5',      'DWSA gamma5',     '.4f'),
            ('dwsa/gamma6',      'DWSA gamma6',     '.4f'),
            ('fan/alpha1_mean',  'FAN alpha1',      '.4f'),
            ('fan/alpha2_mean',  'FAN alpha2',      '.4f'),
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
        print(f"\n  Epoch │ mIoU   │ BCE    │ Dice   │ gamma4 │ gamma5")
        print(f"  {'─'*65}")
        n = max((len(v) for v in self.history.values()), default=0)
        best_miou = max((v for _, v in self.history.get('val/miou', [(0,0)])), default=0)
        for i in range(n):
            def _g(k):
                h = self.history.get(k, [])
                return h[i][1] if i < len(h) else float('nan')
            miou = _g('val/miou')
            mark = ' ← BEST' if not math.isnan(miou) and miou == best_miou else ''
            print(f"  {i+1:>5} │ {miou:.4f} │ {_g('train/bce'):.4f} │ "
                  f"{_g('train/dice'):.4f} │ {_g('dwsa/gamma4'):.4f} │ "
                  f"{_g('dwsa/gamma5'):.4f}{mark}")
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

    print(f"\n{SEP}\nOPTIMIZER PARAMETER GROUPS\n{SEP}")
    for g in groups:
        print(f"  '{g['name']}': lr={g['lr']:.2e}, params={len(g['params'])}")

    # [FIX] The right advice here depends on whether the backbone starts
    # from pretrained weights or from scratch — a fixed rule of thumb was
    # wrong for one of those two cases and gave bad guidance in an earlier
    # run. Discounting backbone/stem LR (small --*_lr_factor) is the right
    # move when finetuning a *pretrained* backbone (protects good features
    # from being overwritten by a freshly-initialized head). It is the
    # *wrong* move when training everything from scratch (as CoMingNet is
    # here, with no --pretrained_weights): the backbone then needs to learn
    # its features at a comparable rate to the head, or it lags badly and
    # caps achievable IoU (observed: --backbone_lr_factor 0.1 --stem_lr_factor
    # 0.01 from scratch capped road IoU well below a from-scratch run with
    # equal LR across groups).
    is_finetune = bool(getattr(args, 'pretrained_weights', None))
    head_lr = next((g['lr'] for g in groups if g['name'] == 'head'), args.lr)
    for g in groups:
        if g['name'] not in ('backbone', 'stem'):
            continue
        ratio = g['lr'] / head_lr if head_lr else 1.0
        if is_finetune and ratio >= 0.9:
            print(f"  ⚠️  WARNING: '{g['name']}' LR ({g['lr']:.2e}) is not meaningfully "
                  f"lower than head LR ({head_lr:.2e}) while finetuning pretrained "
                  f"weights (--pretrained_weights set). This risks overwriting good "
                  f"pretrained features with noisy head gradients early on. Consider "
                  f"--{g['name']}_lr_factor < 1.0.")
        elif not is_finetune and ratio <= 0.05:
            print(f"  ⚠️  WARNING: '{g['name']}' LR ({g['lr']:.2e}) is {ratio:.3f}x the "
                  f"head LR while training from scratch (no --pretrained_weights). A "
                  f"randomly-initialized backbone needs to learn features at a "
                  f"comparable rate to the head, or it becomes a bottleneck that caps "
                  f"achievable IoU. Consider raising --{g['name']}_lr_factor towards "
                  f"0.5-1.0 for from-scratch training.")
    print(f"{SEP}\n")
    return opt


class WarmupWrapper:
    """Linear LR warmup applied before handing control to the base scheduler.

    [FIX] Training from scratch with lr=3e-4 applied immediately to the whole
    backbone is a common source of the large early-epoch gradient spikes and
    mIoU oscillation seen in the logs (e.g. max grad 1.86 -> 0.54 -> 0.78 in
    the first few epochs, mIoU dropping from 0.50 to 0.37 at epoch 4). A short
    linear warmup smooths this out.
    """

    def __init__(self, optimizer, base_scheduler, warmup_epochs, warmup_start_factor=0.1):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = max(0, warmup_epochs)
        self.warmup_start_factor = warmup_start_factor
        self.target_lrs = [g['lr'] for g in optimizer.param_groups]
        self._epoch = 0
        if self.warmup_epochs > 0:
            self._set_lrs(self.warmup_start_factor)

    def _set_lrs(self, factor):
        for g, target in zip(self.optimizer.param_groups, self.target_lrs):
            g['lr'] = target * factor

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * (
                self._epoch / max(1, self.warmup_epochs)
            )
            self._set_lrs(factor)
        else:
            if self.base_scheduler is not None:
                self.base_scheduler.step()

    def state_dict(self):
        return {
            'epoch': self._epoch,
            'base': self.base_scheduler.state_dict() if self.base_scheduler else None,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self._epoch = state.get('epoch', 0)
        if self.base_scheduler is not None and state.get('base'):
            try:
                self.base_scheduler.load_state_dict(state['base'])
            except Exception as e:
                print(f"Warmup base scheduler not loaded: {e}")


def build_scheduler(optimizer, args, train_loader, start_epoch=0):
    use_cosine = (args.freeze_backbone and args.unfreeze_schedule) or args.scheduler == 'cosine'
    warmup_epochs = getattr(args, 'warmup_epochs', 0)
    remaining_epochs = max(1, args.epochs - start_epoch - warmup_epochs)

    if use_cosine:
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=1e-6)
        print(f"CosineAnnealingLR (T_max={remaining_epochs})")
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

    if warmup_epochs > 0 and args.scheduler != 'onecycle':
        print(f"Linear LR warmup: {warmup_epochs} epoch(s)")
        return WarmupWrapper(optimizer, sch, warmup_epochs,
                             warmup_start_factor=getattr(args, 'warmup_start_factor', 0.1))
    return sch


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class BCEDiceLoss(nn.Module):
    """Binary Cross-Entropy + Dice — simple, standard combo for binary
    (background/road) segmentation. No hard-example mining, no
    Tversky/clDice knobs: two terms, easy to reason about while the model
    architecture itself is being iterated on.

    Works directly on the model's 2-channel logits (road_logit =
    logits[:,1] - logits[:,0] is exactly the binary logit a 2-class softmax
    implies, so this is a drop-in replacement for CE on a 2-class head).

    `pos_weight` (optional) up-weights the positive/road class in the BCE
    term — pass `neg_count / pos_count` to counter class imbalance, e.g.
    derived from the same per-class counts used elsewhere for class_weights.
    """

    def __init__(self, dice_weight=0.5, smooth=1e-5, ignore_index=255,
                 pos_weight=None, road_class=1):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.road_class = road_class
        pw = torch.tensor(float(pos_weight)) if pos_weight is not None else None
        self.register_buffer('pos_weight', pw)

    def forward(self, logits, targets):
        logits = logits.float()
        valid = (targets != self.ignore_index).float()
        target_bin = (targets == self.road_class).float()

        road_logit = logits[:, self.road_class] - logits[:, 1 - self.road_class]

        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype) \
            if self.pos_weight is not None else None
        bce_px = F.binary_cross_entropy_with_logits(
            road_logit, target_bin, pos_weight=pos_weight, reduction='none')
        bce = (bce_px * valid).sum() / valid.sum().clamp(min=1)

        prob = torch.sigmoid(road_logit) * valid
        tgt = target_bin * valid
        inter = (prob * tgt).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        total = bce + self.dice_weight * dice_loss
        return total, bce.detach(), dice_loss.detach()




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
            "loss": {"dice_weight": 0.5, "dice_smooth": 1e-5},
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
        self.best_metric_value = 0.0
        self.start_epoch = 0
        self.global_step = 0
        self.diag        = diag

        # [NEW] Smoothed / alternative best-checkpoint tracking.
        # A single noisy epoch (val mIoU jumps of +/-0.08-0.15 are common in
        # the previous run) can look like "the best model" while actually
        # being a lucky/unlucky validation pass. Averaging over a small
        # window makes checkpoint selection more robust.
        self.miou_window = deque(maxlen=max(1, getattr(args, 'best_metric_window', 1)))
        self.best_metric_class = getattr(args, 'best_metric_class', None)

        lcfg = args.loss_config
        self.dice_weight  = lcfg['dice_weight']
        self.base_loss_cfg = lcfg
        self.loss_phase   = 'full'

        # [SIMPLIFIED] Single BCE + Dice loss, no OHEM/Tversky/clDice —
        # fewer moving parts while the model architecture is being
        # iterated on. pos_weight (neg/pos pixel ratio) reuses the same
        # per-class counts as class_weights to counter road being the
        # minority class, without a separate hard-mining mechanism.
        pos_weight = None
        if class_weights is not None and len(class_weights) == 2:
            pos_weight = (class_weights[1] / class_weights[0]).item()
        self.criterion = BCEDiceLoss(
            dice_weight=self.dice_weight,
            smooth=lcfg['dice_smooth'],
            ignore_index=args.ignore_index,
            pos_weight=pos_weight,
            road_class=getattr(args, 'road_class', 1),
        ).to(device)
        print(f"Loss: BCE + Dice({self.dice_weight})"
              + (f", pos_weight={pos_weight:.3f}" if pos_weight is not None else ""))

        self.scaler   = GradScaler(enabled=args.use_amp)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer   = _make_writer(self.save_dir / "tensorboard")
        self._save_config()
        self._print_config()

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
        print(f"Loss: BCE + Dice({self.dice_weight})")
        print(f"{SEP}\n")

    def set_loss_phase(self, phase):
        if phase == self.loss_phase: return
        self.dice_weight = 0.0 if phase == 'ce_only' else self.base_loss_cfg['dice_weight']
        self.criterion.dice_weight = self.dice_weight
        self.loss_phase  = phase
        print(f"Loss phase → {phase}  (BCE + Dice={self.dice_weight})")

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

        total_loss = total_bce = total_dice = 0.0
        max_grad_epoch = 0.0
        # [FIX] `mg` used to only be defined inside the
        # `if (batch_idx+1) % accumulation_steps == 0:` block below, but was
        # referenced unconditionally in `pbar.set_postfix` every iteration.
        # With accumulation_steps > 1 this raises a NameError on the very
        # first batch. Initialize it up-front so the progress bar always has
        # a valid (possibly stale, until the next optimizer step) value.
        mg = 0.0
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

                task_loss, bce_loss, dice_loss = self.criterion(c6_full, masks)

                if self.args.aux_weight > 0:
                    aux_decay = getattr(self.args, 'aux_decay_exp', 0.9)
                    aux_w     = self.args.aux_weight * (1 - epoch / self.args.epochs) ** aux_decay
                    aux_loss, _, _ = self.criterion(c4_full, masks)
                    task_loss = task_loss + aux_w * aux_loss

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

            total_loss += loss.item() * self.args.accumulation_steps
            total_bce  += bce_loss.item()
            total_dice += dice_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item()*self.args.accumulation_steps:.4f}',
                'bce':  f'{bce_loss.item():.4f}',
                'dice': f'{dice_loss.item():.4f}',
                'lr':   f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'mg':   f'{mg:.2f}',
            })
            if batch_idx % 200 == 0: torch.cuda.empty_cache()

        n = len(loader)
        print(f"\nEpoch {epoch+1} — Max grad: {max_grad_epoch:.2f}")
        print(f"  LR head={self.optimizer.param_groups[0]['lr']:.2e}")

        torch.cuda.empty_cache()
        if self.scheduler and self.args.scheduler != 'onecycle':
            self.scheduler.step()

        result = {'loss': total_loss/n, 'bce': total_bce/n, 'dice': total_dice/n}
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
                loss, _, _ = self.criterion(logits, masks)

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

    def compute_checkpoint_metric(self, val_metrics, epoch):
        """[NEW] Decide what value drives 'is this the best checkpoint?'.

        Two knobs, both optional and backward-compatible (default behaviour
        is identical to before: raw val mIoU, single epoch):
          --best_metric_window N : average the metric over the last N epochs
                                    before comparing to the running best.
          --best_metric_class I  : use per_class_iou[I] (e.g. the road class)
                                    instead of mean IoU.
        """
        if self.best_metric_class is not None:
            raw = float(val_metrics['per_class_iou'][self.best_metric_class])
        else:
            raw = val_metrics['miou']

        self.miou_window.append(raw)
        smoothed = float(np.mean(self.miou_window))
        if self.diag:
            self.diag.log(epoch, 'val/miou_smooth', smoothed)
        return smoothed

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
    # [FIX] Previous defaults (0.1 / 0.01) assumed a *pretrained* backbone
    # being finetuned. CoMingNet here trains from scratch by default (no
    # --pretrained_weights), where discounting backbone/stem LR this hard
    # starves the backbone and caps achievable road IoU (confirmed: a
    # from-scratch run using these old defaults topped out at road IoU
    # 0.23 vs. ~0.30 with equal LR across groups). New defaults keep
    # backbone/stem close to the head LR; explicitly lower them via these
    # flags only when passing --pretrained_weights.
    parser.add_argument("--backbone_lr_factor", type=float, default=1.0,
                        help="Backbone LR = lr * this. Use ~0.1 only when finetuning "
                             "--pretrained_weights; keep near 1.0 for from-scratch training.")
    parser.add_argument("--dwsa_lr_factor",     type=float, default=0.5)
    parser.add_argument("--alpha_lr_factor",    type=float, default=0.1)
    parser.add_argument("--stem_lr_factor",     type=float, default=1.0,
                        help="Stem LR = lr * this. Use ~0.01 only when finetuning "
                             "--pretrained_weights; keep near 1.0 for from-scratch training.")
    # [NEW] LR warmup — mitigates the large early-epoch gradient spikes /
    # mIoU oscillation seen when training the backbone from scratch.
    parser.add_argument("--warmup_epochs",      type=int,   default=3,
                        help="Linear LR warmup epochs before the main scheduler kicks in. 0 disables.")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="LR multiplier at the very start of warmup (fraction of target LR).")
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
    # [SIMPLIFIED] BCE + Dice only — OHEM/Tversky/clDice removed to keep the
    # loss surface simple while the model architecture is being iterated on.
    parser.add_argument("--aux_weight",         type=float, default=0.4)
    parser.add_argument("--aux_decay_exp",      type=float, default=0.9)
    parser.add_argument("--dice_weight",        type=float, default=0.5,
                        help="Weight of the Dice term added to BCE.")
    parser.add_argument("--label_smoothing",    type=float, default=0.0)
    # Resolution
    parser.add_argument("--img_h",              type=int,   default=512)
    parser.add_argument("--img_w",              type=int,   default=512,
                        help="[FIX] Was 1024. Source tiles are square (e.g. 1500x1500); "
                             "a non-square img_h/img_w stretches roads by different "
                             "factors horizontally vs. vertically. Prefer --crop_size "
                             "instead of relying on img_h/img_w resize.")
    # [NEW] Aspect-preserving square crop, recommended over plain resize.
    parser.add_argument("--crop_size",          type=int,   default=None,
                        help="If set, train/val on an aspect-ratio-preserving square crop "
                             "of this size instead of resizing to --img_h/--img_w (which "
                             "distorts road geometry on non-square targets). Recommended.")
    parser.add_argument("--road_oversample_tries", type=int, default=4,
                        help="During training with --crop_size, sample this many candidate "
                             "crop locations and keep the one with the most road pixels, "
                             "since plain random crops are mostly pure background.")
    # BN warmup (K1)
    parser.add_argument("--reset_bn_stats",     action="store_true")
    parser.add_argument("--bn_warmup_epochs",   type=int,   default=3)
    parser.add_argument("--bn_warmup_momentum", type=float, default=0.3)
    # [NEW] Checkpoint-selection robustness
    parser.add_argument("--best_metric_window", type=int,   default=1,
                        help="Average val metric over the last N epochs before "
                             "deciding a new best checkpoint. 1 = old behaviour "
                             "(single noisy epoch can trigger a save).")
    parser.add_argument("--best_metric_class",  type=int,   default=None,
                        help="If set, use per_class_iou[this_index] (e.g. the road "
                             "class index) instead of mean IoU to select the best "
                             "checkpoint.")
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

    # Import backbone.
    # [FIX] The original code only handled `model_variant == 'coming'` and
    # left `Backbone` undefined for every other choice, which would raise a
    # confusing `NameError` deep inside `Segmentor(Backbone(...), ...)`
    # instead of a clear, actionable error at startup.
    if args.model_variant == 'coming':
        from modeling.backbone import CoMingNet as Backbone
    else:
        try:
            from modeling.backbone import GCNet as Backbone
        except ImportError as e:
            raise ImportError(
                f"--model_variant='{args.model_variant}' requires a GCNet-style "
                f"backbone class (e.g. `GCNet`) in modeling/backbone.py, which "
                f"could not be imported ({e}). Either add that class, or pass "
                f"--model_variant coming to use CoMingNet."
            ) from e

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

    if args.best_metric_class is not None and not (0 <= args.best_metric_class < args.num_classes):
        raise ValueError(
            f"--best_metric_class={args.best_metric_class} is out of range for "
            f"num_classes={args.num_classes}."
        )

    # [NEW] BCEDiceLoss operates on a binary road/background logit derived
    # from the model's 2-channel output — it does not generalize to >2
    # classes, so fail fast with a clear message rather than a confusing
    # shape error deep inside the loss.
    if args.num_classes != 2:
        raise ValueError(
            f"--num_classes={args.num_classes}, but the simplified BCE+Dice loss "
            f"only supports binary segmentation (num_classes=2, e.g. "
            f"['background','road'])."
        )

    # [FIX] `--class_weights_file` was silently ignored unless
    # `--use_class_weights` was *also* passed, because Trainer only received
    # weights when `args.use_class_weights` was true. Track the two sources
    # separately and enable weighting if either is provided.
    weights_requested = args.use_class_weights or bool(args.class_weights_file)

    # DataLoaders
    train_loader, val_loader, class_weights = create_folder_dataloaders(
        image_dir=args.image_dir, mask_dir=args.mask_dir,
        val_ratio=args.val_ratio, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=(args.img_h, args.img_w), num_classes=args.num_classes,
        compute_class_weights=args.use_class_weights,
        crop_size=args.crop_size, road_oversample_tries=args.road_oversample_tries)

    if getattr(args, "class_weights_file", None):
        cw_path = Path(args.class_weights_file)
        if cw_path.exists():
            class_weights = torch.load(cw_path, map_location="cpu")
            print(f"Class weights: {cw_path}  "
                  f"(min={class_weights.min():.3f}, max={class_weights.max():.3f})")
        else:
            print(f"WARNING: {cw_path} not found"); class_weights = None
            weights_requested = args.use_class_weights and class_weights is not None

    # Build model
    model = Segmentor(Backbone(**cfg["backbone"]),
                      GCNetHead(**cfg["head"], num_classes=args.num_classes,
                                ignore_index=args.ignore_index)).to(device)
    model.apply(init_weights)

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
                      class_weights=class_weights if weights_requested else None,
                      diag=diag)

    if args.dice_weight is not None:
        trainer.dice_weight = args.dice_weight
        trainer.criterion.dice_weight = args.dice_weight
        trainer.base_loss_cfg["dice_weight"] = args.dice_weight

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
              f"BCE: {train_metrics['bce']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f}  | "
              f"mIoU: {val_metrics['miou']:.4f}  | "
              f"Acc: {val_metrics['accuracy']:.4f}")
        print(f"{SEP}\n")

        diag.print_epoch_summary(epoch)

        # [FIX] Checkpoint selection now optionally uses a smoothed metric
        # and/or a specific class's IoU (e.g. road) instead of a single raw
        # mIoU value, which was prone to picking an unlucky/lucky epoch —
        # the earlier run's "best" epoch (33) beat epoch 20 by only 0.0017,
        # well within normal epoch-to-epoch noise.
        checkpoint_metric = trainer.compute_checkpoint_metric(val_metrics, epoch)
        is_best = checkpoint_metric > trainer.best_metric_value
        if is_best:
            trainer.best_metric_value = checkpoint_metric
            trainer.best_miou = val_metrics['miou']
            metric_name = (
                f"{class_names[args.best_metric_class]} IoU"
                if args.best_metric_class is not None else "mIoU"
            )
            window_note = f" (avg of last {len(trainer.miou_window)})" if args.best_metric_window > 1 else ""
            print(f"  ★ NEW BEST {metric_name}{window_note}: {checkpoint_metric:.4f} "
                  f"(raw epoch mIoU: {val_metrics['miou']:.4f})")
        trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)

    diag.print_full_history()
    diag.close()
    trainer.writer.close()

    print(f"\n{SEP}\nTRAINING COMPLETED!\nBest mIoU: {trainer.best_miou:.4f}\n{SEP}\n")


if __name__ == "__main__":
    main()
