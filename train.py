"""Train the balanced CoMingNet on Massachusetts Roads or DeepGlobe Roads.

Key choices:
  * 512x512 crops for training; no multi-scale augmentation.
  * Official train/val/test folders when available.
  * Sliding-window validation on complete 1500x1500 images.
  * BCE + Dice baseline; every extra objective is explicitly opt-in.
  * Training-only optional deep supervision, centerline head and distillation.
  * Threshold calibration and both micro/macro road IoU.

Place this file at ``road-detection/train.py`` and the accompanying backbone
and decoder files under ``road-detection/modeling/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from modeling.backbone import CoMingBlock, CoMingNet
from modeling.decoder import GCNetHead


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_masks", "_gt", "_label", "_labels")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_key(path: Path) -> str:
    key = path.stem.lower()
    for suffix in ("_image", "_images", "_img", "_sat", "_mask", "_masks", "_gt", "_label", "_labels"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


def index_files(folder: str | Path, role: Optional[str] = None) -> Dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")
    files = sorted(
        path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if role is not None:
        if role not in {"image", "mask"}:
            raise ValueError("role must be 'image', 'mask' or None")
        files = [
            path for path in files
            if any(path.stem.lower().endswith(suffix) for suffix in MASK_SUFFIXES)
            == (role == "mask")
        ]
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    result: Dict[str, Path] = {}
    for path in files:
        key = sample_key(path)
        if key in result:
            raise RuntimeError(f"Duplicate sample key {key}: {result[key]} and {path}")
        result[key] = path
    return result


def build_pairs(image_dir: str | Path, mask_dir: str | Path) -> List[Tuple[Path, Path]]:
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    same_folder = image_dir.resolve() == mask_dir.resolve()
    images = index_files(image_dir, role="image" if same_folder else None)
    masks = index_files(mask_dir, role="mask" if same_folder else None)
    keys = sorted(images.keys() & masks.keys())
    if len(keys) != len(images) or len(keys) != len(masks):
        raise RuntimeError(
            f"Pairing mismatch: images={len(images)}, masks={len(masks)}, pairs={len(keys)}"
        )
    return [(images[key], masks[key]) for key in keys]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return (mask > 0).astype(np.uint8)


def dump_alignment_check(
    pairs: Sequence[Tuple[Path, Path]],
    output_dir: str | Path,
    n_samples: int = 6,
) -> None:
    """Write red road-mask overlays for a one-time image/mask sanity check."""
    if n_samples <= 0 or not pairs:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(int(n_samples), len(pairs))
    # Spread samples over the sorted dataset instead of inspecting only one area.
    indices = np.linspace(0, len(pairs) - 1, sample_count, dtype=np.int64)
    for output_index, pair_index in enumerate(indices.tolist()):
        image_path, mask_path = pairs[pair_index]
        image = read_rgb(image_path)
        mask = read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"Image/mask shape mismatch for {image_path.name}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )
        overlay = image.copy()
        overlay[mask > 0] = np.asarray((255, 0, 0), dtype=np.uint8)
        blended = np.clip(
            0.60 * image.astype(np.float32)
            + 0.40 * overlay.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
        output_path = output_dir / (
            f"check_{output_index:02d}_{image_path.stem}.png"
        )
        Image.fromarray(blended).save(output_path)
    print(
        f"Saved {sample_count} alignment-check overlays to "
        f"{output_dir.resolve()}"
    )


class RoadDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        crop_size: int = 512,
        training: bool = False,
        full_image: bool = False,
        road_crop_probability: float = 0.5,
        road_oversample_tries: int = 4,
        use_multiscale: bool = False,
        multiscale_min: float = 0.75,
        multiscale_max: float = 1.50,
        multiscale_probability: float = 1.0,
        fixed_crop_seed: Optional[int] = None,
        fixed_crop_candidates: int = 32,
    ) -> None:
        self.pairs = list(pairs)
        self.crop_size = crop_size
        self.training = training
        self.full_image = full_image
        self.road_crop_probability = road_crop_probability
        self.road_oversample_tries = max(1, road_oversample_tries)
        self.use_multiscale = use_multiscale
        self.multiscale_min = multiscale_min
        self.multiscale_max = multiscale_max
        self.multiscale_probability = multiscale_probability
        self.fixed_crop_seed = fixed_crop_seed
        self.fixed_crop_candidates = max(1, int(fixed_crop_candidates))

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _pad(image: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        height, width = mask.shape
        pad_h, pad_w = max(0, size - height), max(0, size - width)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        return image, mask

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Crop a variable-size source window, then resize only that window back
        # to crop_size. This is equivalent to resize-then-crop augmentation but
        # avoids resizing the complete 1500x1500 Massachusetts image.
        scale = 1.0
        if (
            self.training
            and self.use_multiscale
            and random.random() < self.multiscale_probability
        ):
            scale = random.uniform(self.multiscale_min, self.multiscale_max)
        source_size = max(32, int(round(self.crop_size / scale)))

        image, mask = self._pad(image, mask, source_size)
        height, width = mask.shape
        max_y, max_x = height - source_size, width - source_size

        def uniform_position() -> Tuple[int, int]:
            return random.randint(0, max_y), random.randint(0, max_x)

        use_road_crop = random.random() < self.road_crop_probability and mask.any()
        if not use_road_crop:
            top, left = uniform_position()
        else:
            road_y, road_x = np.nonzero(mask)
            candidates: List[Tuple[int, int]] = []
            for _ in range(self.road_oversample_tries):
                index = random.randrange(len(road_y))
                jitter = source_size // 4
                center_y = int(road_y[index]) + random.randint(-jitter, jitter)
                center_x = int(road_x[index]) + random.randint(-jitter, jitter)
                top = min(max(center_y - source_size // 2, 0), max_y)
                left = min(max(center_x - source_size // 2, 0), max_x)
                candidates.append((top, left))
            top, left = max(
                candidates,
                key=lambda position: int(
                    mask[
                        position[0] : position[0] + source_size,
                        position[1] : position[1] + source_size,
                    ].sum()
                ),
            )

        image_crop = image[top : top + source_size, left : left + source_size]
        mask_crop = mask[top : top + source_size, left : left + source_size]
        if source_size != self.crop_size:
            image_crop = np.asarray(
                Image.fromarray(image_crop).resize(
                    (self.crop_size, self.crop_size), Image.Resampling.BILINEAR
                )
            )
            mask_crop = np.asarray(
                Image.fromarray(mask_crop.astype(np.uint8)).resize(
                    (self.crop_size, self.crop_size), Image.Resampling.NEAREST
                )
            )
            mask_crop = (mask_crop > 0).astype(np.uint8)
        return image_crop, mask_crop

    def _fixed_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the same road-rich crop every time for an overfit test."""
        image, mask = self._pad(image, mask, self.crop_size)
        height, width = mask.shape
        max_y = height - self.crop_size
        max_x = width - self.crop_size
        rng = np.random.default_rng(int(self.fixed_crop_seed) + int(index))

        if mask.any():
            road_y, road_x = np.nonzero(mask)
            count = min(self.fixed_crop_candidates, len(road_y))
            sampled = rng.integers(0, len(road_y), size=count)
            candidates: List[Tuple[int, int]] = []
            for road_index in sampled.tolist():
                center_y = int(road_y[road_index])
                center_x = int(road_x[road_index])
                top = min(max(center_y - self.crop_size // 2, 0), max_y)
                left = min(max(center_x - self.crop_size // 2, 0), max_x)
                candidates.append((top, left))
            top, left = max(
                candidates,
                key=lambda position: int(
                    mask[
                        position[0] : position[0] + self.crop_size,
                        position[1] : position[1] + self.crop_size,
                    ].sum()
                ),
            )
        else:
            top, left = max_y // 2, max_x // 2

        return (
            image[top : top + self.crop_size, left : left + self.crop_size],
            mask[top : top + self.crop_size, left : left + self.crop_size],
        )

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image[::-1, :], mask[::-1, :]
        rotations = random.randrange(4)
        if rotations:
            image = np.rot90(image, rotations)
            mask = np.rot90(mask, rotations)
        if random.random() < 0.6:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-12.0, 12.0)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return image, mask

    @staticmethod
    def _to_tensor(image: np.ndarray, mask: np.ndarray) -> Tuple[Tensor, Tensor]:
        image_float = image.astype(np.float32) / 255.0
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_float.transpose(2, 0, 1))
        )
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image_path, mask_path = self.pairs[index]
        image, mask = read_rgb(image_path), read_binary_mask(mask_path)
        if not self.full_image:
            if self.fixed_crop_seed is not None:
                image, mask = self._fixed_crop(image, mask, index)
            else:
                image, mask = self._crop(image, mask)
        if self.training:
            image, mask = self._augment(image, mask)
        return self._to_tensor(image, mask)


def split_or_official_pairs(args: argparse.Namespace) -> Tuple[List, List]:
    train_pairs = build_pairs(args.image_dir, args.mask_dir)
    val_image_dir = Path(args.val_image_dir) if args.val_image_dir else None
    val_mask_dir = Path(args.val_mask_dir) if args.val_mask_dir else None
    if (
        val_image_dir is not None
        and val_mask_dir is not None
        and val_image_dir.is_dir()
        and val_mask_dir.is_dir()
    ):
        val_pairs = build_pairs(val_image_dir, val_mask_dir)
        print(f"Official split: train={len(train_pairs)}, val={len(val_pairs)}")
        return train_pairs, val_pairs

    generator = np.random.default_rng(args.seed)
    indices = generator.permutation(len(train_pairs))
    val_count = max(1, round(len(train_pairs) * args.val_ratio))
    val_indices = set(indices[:val_count].tolist())
    val_pairs = [pair for index, pair in enumerate(train_pairs) if index in val_indices]
    train_pairs = [pair for index, pair in enumerate(train_pairs) if index not in val_indices]
    print(
        f"WARNING: official val folders not found; deterministic random split "
        f"train={len(train_pairs)}, val={len(val_pairs)}"
    )
    return train_pairs, val_pairs


def configure_dataset_paths(args: argparse.Namespace) -> None:
    """Fill dataset defaults without mixing DeepGlobe and Massachusetts."""
    if args.dataset == "massachusetts":
        root = Path("/kaggle/input/datasets/balraj98/massachusetts-roads-dataset/tiff")
        args.image_dir = args.image_dir or str(root / "train")
        args.mask_dir = args.mask_dir or str(root / "train_labels")
        args.val_image_dir = args.val_image_dir or str(root / "val")
        args.val_mask_dir = args.val_mask_dir or str(root / "val_labels")
        args.test_image_dir = args.test_image_dir or str(root / "test")
        args.test_mask_dir = args.test_mask_dir or str(root / "test_labels")
    else:
        root = Path(
            "/kaggle/input/datasets/balraj98/"
            "deepglobe-road-extraction-dataset/train"
        )
        # DeepGlobe *_sat.jpg and *_mask.png files share this directory.
        args.image_dir = args.image_dir or str(root)
        args.mask_dir = args.mask_dir or str(root)


def make_loaders(args: argparse.Namespace):
    train_pairs, val_pairs = split_or_official_pairs(args)
    if args.overfit_debug:
        if args.overfit_samples < 1:
            raise ValueError("overfit_samples must be >= 1")
        sample_count = min(args.overfit_samples, len(train_pairs))
        generator = np.random.default_rng(args.seed)
        selected_indices = sorted(
            generator.choice(
                len(train_pairs), size=sample_count, replace=False
            ).tolist()
        )
        train_pairs = [train_pairs[index] for index in selected_indices]
        val_pairs = list(train_pairs)
        print(
            "OVERFIT DEBUG: "
            f"samples={sample_count}, fixed road-rich {args.crop_size}x{args.crop_size} "
            "crops, augmentation=off, multiscale=off"
        )
        train_dataset = RoadDataset(
            train_pairs,
            crop_size=args.crop_size,
            training=False,
            full_image=False,
            fixed_crop_seed=args.seed,
            fixed_crop_candidates=args.overfit_crop_candidates,
        )
        val_dataset = RoadDataset(
            val_pairs,
            crop_size=args.crop_size,
            training=False,
            full_image=False,
            fixed_crop_seed=args.seed,
            fixed_crop_candidates=args.overfit_crop_candidates,
        )
    else:
        train_dataset = RoadDataset(
            train_pairs,
            crop_size=args.crop_size,
            training=True,
            road_crop_probability=args.road_crop_probability,
            road_oversample_tries=args.road_oversample_tries,
            use_multiscale=args.use_multiscale,
            multiscale_min=args.multiscale_min,
            multiscale_max=args.multiscale_max,
            multiscale_probability=args.multiscale_probability,
        )
        val_dataset = RoadDataset(
            val_pairs, crop_size=args.crop_size, training=False, full_image=True
        )
    common = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=not args.overfit_debug,
        **common,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **common)
    return train_loader, val_loader, train_pairs


def compute_pos_weight(pairs: Sequence[Tuple[Path, Path]], cap: float) -> Tuple[float, float]:
    positive, total = 0, 0
    for _, mask_path in tqdm(pairs, desc="Computing road pixel ratio"):
        mask = read_binary_mask(mask_path)
        positive += int(mask.sum())
        total += mask.size
    negative = total - positive
    raw = negative / max(positive, 1)
    used = min(math.sqrt(raw), cap)
    return raw, used


class Segmentor(nn.Module):
    def __init__(self, backbone: CoMingNet, decode_head: GCNetHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    def forward(self, x: Tensor):
        return self.decode_head(self.backbone(x))

    def switch_to_deploy(self) -> "Segmentor":
        self.backbone.switch_to_deploy()
        self.decode_head.switch_to_deploy()
        return self


def build_model(args: argparse.Namespace) -> Segmentor:
    if args.model_variant != "coming":
        raise ValueError("This balanced file intentionally supports --model_variant coming only")
    feature_channels = args.channels * 4
    backbone = CoMingNet(
        in_channels=3,
        channels=args.channels,
        local_blocks=args.local_blocks,
        global_blocks=args.global_blocks,
        highres_kernel_size=args.highres_kernel_size,
        context_kernel_size=args.coming_kernel_size,
        local_expansion=args.local_expansion,
        global_expansion=args.global_expansion,
        local_spatial_ratio=args.local_spatial_ratio,
        global_spatial_ratio=args.global_spatial_ratio,
    )
    head = GCNetHead(
        in_channels=feature_channels,
        channels=args.decoder_channels,
        num_classes=2,
        feature_channels=(feature_channels, feature_channels, feature_channels),
        stem_channels=args.channels,
        highres_kernel_size=args.highres_kernel_size,
        context_kernel_size=args.coming_kernel_size,
        local_expansion=args.local_expansion,
        global_expansion=args.global_expansion,
        local_spatial_ratio=args.local_spatial_ratio,
        global_spatial_ratio=args.global_spatial_ratio,
        enable_seg_aux=args.aux_weight > 0,
        enable_centerline_aux=args.centerline_weight > 0,
        enable_half_refine=args.use_half_refine,
        half_refine_channels=args.half_refine_channels,
        enable_fullres_head=args.use_fullres_head,
        fullres_channels=args.fullres_channels,
        dropout_ratio=args.dropout,
    )
    model = Segmentor(backbone, head)
    zeroed = 0
    for module in model.modules():
        if isinstance(module, CoMingBlock):
            module.zero_init_residual()
            zeroed += 1
    print(f"Zero-initialized residual branches: {zeroed} CoMingBlocks")
    return model


def road_logit(logits: Tensor) -> Tensor:
    if logits.shape[1] != 2:
        raise ValueError("The balanced loss expects two output channels")
    return logits[:, 1:2] - logits[:, 0:1]


def soft_dice_loss(probability: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def soft_boundary_map(value: Tensor, kernel_size: int = 3) -> Tensor:
    padding = kernel_size // 2
    maximum = F.max_pool2d(value, kernel_size, stride=1, padding=padding)
    minimum = -F.max_pool2d(-value, kernel_size, stride=1, padding=padding)
    return (maximum - minimum).clamp(0.0, 1.0)


def soft_erode(value: Tensor) -> Tensor:
    vertical = -F.max_pool2d(-value, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-value, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(value: Tensor) -> Tensor:
    return F.max_pool2d(value, 3, stride=1, padding=1)


def soft_skeleton(value: Tensor, iterations: int = 5) -> Tensor:
    opened = soft_dilate(soft_erode(value))
    skeleton = F.relu(value - opened)
    for _ in range(iterations):
        value = soft_erode(value)
        opened = soft_dilate(soft_erode(value))
        delta = F.relu(value - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(probability: Tensor, target: Tensor, iterations: int = 5) -> Tensor:
    pred_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    smooth = 1e-6
    topology_precision = (
        (pred_skeleton * target).sum(dim=(1, 2, 3)) + smooth
    ) / (pred_skeleton.sum(dim=(1, 2, 3)) + smooth)
    topology_sensitivity = (
        (target_skeleton * probability).sum(dim=(1, 2, 3)) + smooth
    ) / (target_skeleton.sum(dim=(1, 2, 3)) + smooth)
    cldice = (
        2.0 * topology_precision * topology_sensitivity
        / (topology_precision + topology_sensitivity + smooth)
    )
    return (1.0 - cldice).mean()


def centerline_auxiliary_loss(
    centerline_logits: Tensor,
    segmentation_logits: Tensor,
    target: Tensor,
    iterations: int = 5,
    pos_weight_cap: float = 8.0,
    dice_weight: float = 0.5,
    cldice_weight: float = 0.25,
    containment_weight: float = 0.1,
) -> Dict[str, Tensor]:
    """Supervise a training-only centerline head at output stride four.

    The skeleton target is generated online from the binary road mask.  This
    keeps Massachusetts and DeepGlobe compatible without extra label files.
    """
    size = centerline_logits.shape[-2:]
    with torch.no_grad():
        road_target = F.interpolate(
            target[:, None].float(), size=size, mode="nearest"
        )
        centerline_target = soft_skeleton(
            road_target.float(), iterations=max(1, iterations)
        ).clamp(0.0, 1.0)

    positive = centerline_target.sum()
    negative = centerline_target.numel() - positive
    dynamic_pos_weight = (negative / positive.clamp_min(1.0)).clamp(
        min=1.0, max=pos_weight_cap
    ).reshape(1)
    bce = F.binary_cross_entropy_with_logits(
        centerline_logits,
        centerline_target.to(centerline_logits.dtype),
        pos_weight=dynamic_pos_weight.to(centerline_logits.dtype),
    )
    probability = torch.sigmoid(centerline_logits)
    dice = soft_dice_loss(probability, centerline_target)
    topology = soft_cldice_loss(probability, centerline_target, iterations=3)

    road_probability = torch.sigmoid(road_logit(segmentation_logits))
    road_probability = F.interpolate(
        road_probability, size=size, mode="bilinear", align_corners=False
    )
    containment = (probability * (1.0 - road_probability)).mean()
    total = (
        bce
        + dice_weight * dice
        + cldice_weight * topology
        + containment_weight * containment
    )
    return {
        "total": total,
        "bce": bce,
        "dice": dice,
        "cldice": topology,
        "containment": containment,
    }


class RoadLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 1.0,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.1,
        cldice_weight: float = 0.05,
        cldice_start_epoch: int = 10,
        cldice_downsample: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.cldice_weight = cldice_weight
        self.cldice_start_epoch = cldice_start_epoch
        self.cldice_downsample = max(1, cldice_downsample)

    def base(self, logits: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        target_float = target[:, None].float()
        binary_logit = road_logit(logits)
        probability = torch.sigmoid(binary_logit)
        bce = F.binary_cross_entropy_with_logits(
            binary_logit, target_float, pos_weight=self.pos_weight
        )
        dice = soft_dice_loss(probability, target_float)
        return bce + self.dice_weight * dice, bce, dice, probability

    def forward(self, logits: Tensor, target: Tensor, epoch: int) -> Dict[str, Tensor]:
        total, bce, dice, probability = self.base(logits, target)
        target_float = target[:, None].float()

        boundary = logits.sum() * 0.0
        if self.boundary_weight > 0:
            # BCELoss on probabilities is explicitly disallowed inside AMP
            # autocast.  Boundary extraction is also more stable in FP32, so
            # disable autocast only for this small training-only term.  The
            # main network forward and all other losses still benefit from AMP.
            with torch.autocast(device_type=logits.device.type, enabled=False):
                pred_boundary = soft_boundary_map(probability.float())
                target_boundary = soft_boundary_map(target_float.float())
                boundary = F.binary_cross_entropy(
                    pred_boundary.clamp(1e-5, 1.0 - 1e-5),
                    target_boundary,
                )
            total = total + self.boundary_weight * boundary

        cldice = logits.sum() * 0.0
        if self.cldice_weight > 0 and epoch >= self.cldice_start_epoch:
            if self.cldice_downsample > 1:
                size = (
                    max(1, probability.shape[-2] // self.cldice_downsample),
                    max(1, probability.shape[-1] // self.cldice_downsample),
                )
                probability_topology = F.interpolate(
                    probability, size=size, mode="bilinear", align_corners=False
                )
                target_topology = F.interpolate(target_float, size=size, mode="nearest")
            else:
                probability_topology, target_topology = probability, target_float
            cldice = soft_cldice_loss(probability_topology, target_topology)
            total = total + self.cldice_weight * cldice

        return {
            "total": total,
            "bce": bce,
            "dice": dice,
            "boundary": boundary,
            "cldice": cldice,
        }


def distillation_loss(student: Tensor, teacher: Tensor, temperature: float) -> Tensor:
    student_distribution = F.log_softmax(student / temperature, dim=1)
    teacher_distribution = F.softmax(teacher / temperature, dim=1)
    return F.kl_div(
        student_distribution, teacher_distribution, reduction="batchmean"
    ) * (temperature**2) / (student.shape[-2] * student.shape[-1])


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a Kaggle dataset folder or a direct checkpoint file."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    preferred_names = (
        "best_road_iou.pt", "best_road_iou.pth", "best.pt", "best.pth",
        "last.pt", "last.pth", "checkpoint.pt", "checkpoint.pth",
    )
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    by_name = {candidate.name.lower(): candidate for candidate in files}
    for name in preferred_names:
        if name in by_name:
            resolved = by_name[name]
            print(f"Resolved checkpoint directory to: {resolved}")
            return resolved
    candidates = [
        candidate for candidate in files
        if candidate.suffix.lower() in {".pt", ".pth", ".ckpt"}
    ]
    if len(candidates) == 1:
        print(f"Resolved checkpoint directory to: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth/.ckpt checkpoint found below: {path}")
    names = "\n".join(f"  - {candidate}" for candidate in sorted(candidates))
    raise RuntimeError(
        "Checkpoint directory is ambiguous; pass the exact file path:\n" + names
    )


def safe_torch_load(path: str | Path):
    path = resolve_checkpoint_path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 has no weights_only argument.
        return torch.load(path, map_location="cpu")


def load_state(path: str | Path) -> Dict[str, Tensor]:
    checkpoint = safe_torch_load(path)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_transfer(model: nn.Module, path: str | Path) -> None:
    source = load_state(path)
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    ratio = 100.0 * sum(value.numel() for value in compatible.values()) / max(
        sum(value.numel() for value in target.values()), 1
    )
    print(f"Transferred {len(compatible)}/{len(target)} tensors ({ratio:.1f}% by elements)")


def build_teacher(args: argparse.Namespace, model: Segmentor, device: torch.device):
    if not args.teacher_checkpoint:
        return None
    teacher = copy.deepcopy(model)
    teacher.load_state_dict(load_state(args.teacher_checkpoint), strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(f"Knowledge distillation enabled: {args.teacher_checkpoint}")
    return teacher


def parameter_groups(model: Segmentor, args: argparse.Namespace):
    groups = {"head": [], "backbone": [], "stem": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.stem_"):
            groups["stem"].append(parameter)
        elif name.startswith("backbone."):
            groups["backbone"].append(parameter)
        else:
            groups["head"].append(parameter)
    learning_rates = {
        "head": args.lr,
        "backbone": args.lr * args.backbone_lr_factor,
        "stem": args.lr * args.stem_lr_factor,
    }
    result = []
    for name, parameters in groups.items():
        if parameters:
            result.append({"params": parameters, "lr": learning_rates[name], "name": name})
            count = sum(parameter.numel() for parameter in parameters)
            print(f"Optimizer group {name:>8}: lr={learning_rates[name]:.2e}, params={count:,}")
    return result


def build_scheduler(optimizer, steps_per_epoch: int, args: argparse.Namespace):
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / max(warmup_steps, 1)
            return args.warmup_start_factor + (1.0 - args.warmup_start_factor) * progress
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def train_one_epoch(
    model: Segmentor,
    teacher: Optional[Segmentor],
    loader: DataLoader,
    criterion: RoadLoss,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        key: 0.0
        for key in (
            "total", "bce", "dice", "boundary", "cldice", "aux",
            "centerline", "centerline_dice", "centerline_cldice", "kd",
        )
    }
    updates = 0
    max_gradient = 0.0
    road_fraction_sum = 0.0
    road_fraction_min = 1.0
    road_fraction_max = 0.0
    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}")

    for batch_index, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        road_fraction = float((masks > 0).float().mean())
        road_fraction_sum += road_fraction
        road_fraction_min = min(road_fraction_min, road_fraction)
        road_fraction_max = max(road_fraction_max, road_fraction)
        if (
            args.road_density_log_interval > 0
            and batch_index % args.road_density_log_interval == 0
        ):
            progress.write(
                f"  [road-density] epoch={epoch + 1} batch={batch_index:03d} "
                f"road_fraction={road_fraction:.4f}"
            )

        teacher_logits = None
        if teacher is not None:
            with torch.no_grad(), autocast_context(device, args.use_amp):
                teacher_logits = teacher(images)

        with autocast_context(device, args.use_amp):
            aux_logits, centerline_logits, main_logits = model(images)
            main_logits = F.interpolate(
                main_logits, masks.shape[-2:], mode="bilinear", align_corners=False
            )
            losses = criterion(main_logits, masks, epoch)
            aux_base = main_logits.sum() * 0.0
            aux_weight = 0.0
            if aux_logits is not None and args.aux_weight > 0:
                aux_logits = F.interpolate(
                    aux_logits,
                    masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                aux_base, _, _, _ = criterion.base(aux_logits, masks)
                aux_weight = (
                    args.aux_weight
                    * (1.0 - epoch / args.epochs) ** args.aux_decay_exp
                )
            total_loss = losses["total"] + aux_weight * aux_base

            centerline_losses = {
                "total": main_logits.sum() * 0.0,
                "dice": main_logits.sum() * 0.0,
                "cldice": main_logits.sum() * 0.0,
            }
            if (
                centerline_logits is not None
                and args.centerline_weight > 0
                and epoch >= args.centerline_start_epoch
            ):
                centerline_losses = centerline_auxiliary_loss(
                    centerline_logits,
                    main_logits,
                    masks,
                    iterations=args.centerline_iterations,
                    pos_weight_cap=args.centerline_pos_weight_cap,
                    dice_weight=args.centerline_dice_weight,
                    cldice_weight=args.centerline_cldice_weight,
                    containment_weight=args.centerline_containment_weight,
                )
                total_loss = total_loss + (
                    args.centerline_weight * centerline_losses["total"]
                )

            kd = main_logits.sum() * 0.0
            if teacher_logits is not None:
                teacher_logits = F.interpolate(
                    teacher_logits,
                    masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                kd = distillation_loss(main_logits, teacher_logits, args.kd_temperature)
                total_loss = total_loss + args.kd_weight * kd

            scaled_loss = total_loss / args.accumulation_steps

        if not torch.isfinite(scaled_loss):
            print(f"Non-finite loss at batch {batch_index}; batch skipped")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(scaled_loss).backward()
        should_update = (
            (batch_index + 1) % args.accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_update:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=False
            )
            if torch.isfinite(gradient_norm):
                max_gradient = max(max_gradient, float(gradient_norm))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                updates += 1
            else:
                print(f"Non-finite gradient at batch {batch_index}; optimizer step skipped")
                scaler.update()
            optimizer.zero_grad(set_to_none=True)

        totals["total"] += float(total_loss.detach())
        for key in ("bce", "dice", "boundary", "cldice"):
            totals[key] += float(losses[key].detach())
        totals["aux"] += float(aux_base.detach())
        totals["centerline"] += float(centerline_losses["total"].detach())
        totals["centerline_dice"] += float(centerline_losses["dice"].detach())
        totals["centerline_cldice"] += float(centerline_losses["cldice"].detach())
        totals["kd"] += float(kd.detach())
        progress.set_postfix(
            loss=f"{float(total_loss.detach()):.4f}",
            road_dice=f"{1.0 - float(losses['dice'].detach()):.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

    count = max(len(loader), 1)
    metrics = {key: value / count for key, value in totals.items()}
    metrics["aux_weight"] = (
        args.aux_weight
        * (1.0 - epoch / args.epochs) ** args.aux_decay_exp
        if args.aux_weight > 0
        else 0.0
    )
    metrics["max_gradient"] = max_gradient
    metrics["updates"] = float(updates)
    metrics["road_fraction_mean"] = road_fraction_sum / count
    metrics["road_fraction_min"] = road_fraction_min if len(loader) else 0.0
    metrics["road_fraction_max"] = road_fraction_max if len(loader) else 0.0
    return metrics


def sliding_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def sliding_window_inference(
    model: Segmentor,
    image: Tensor,
    tile_size: int,
    overlap: int,
    tile_batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> Tensor:
    if image.shape[0] != 1:
        raise ValueError("Full-image sliding validation requires batch_size=1")
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("val_overlap must be smaller than crop_size")

    _, _, original_h, original_w = image.shape
    pad_h, pad_w = max(0, tile_size - original_h), max(0, tile_size - original_w)
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    height, width = image.shape[-2:]
    y_positions = sliding_positions(height, tile_size, stride)
    x_positions = sliding_positions(width, tile_size, stride)

    one_dimensional = torch.hann_window(
        tile_size, periodic=False, device=device, dtype=torch.float32
    ).clamp_min(0.05)
    weight = (one_dimensional[:, None] * one_dimensional[None, :])[None, None]
    logits_sum = torch.zeros((1, 2, height, width), device=device, dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)

    patches: List[Tensor] = []
    coordinates: List[Tuple[int, int]] = []

    def flush() -> None:
        if not patches:
            return
        batch = torch.cat(patches, dim=0)
        with autocast_context(device, use_amp):
            batch_logits = model(batch)
            batch_logits = F.interpolate(
                batch_logits,
                (tile_size, tile_size),
                mode="bilinear",
                align_corners=False,
            )
        batch_logits = batch_logits.float()
        for index, (top, left) in enumerate(coordinates):
            logits_sum[:, :, top : top + tile_size, left : left + tile_size] += (
                batch_logits[index : index + 1] * weight
            )
            weight_sum[:, :, top : top + tile_size, left : left + tile_size] += weight
        patches.clear()
        coordinates.clear()

    for top in y_positions:
        for left in x_positions:
            patches.append(image[:, :, top : top + tile_size, left : left + tile_size])
            coordinates.append((top, left))
            if len(patches) == tile_batch_size:
                flush()
    flush()
    logits = logits_sum / weight_sum.clamp_min(1e-6)
    return logits[:, :, :original_h, :original_w]


def relaxed_overlap_components(
    prediction: Tensor,
    target: Tensor,
    buffer_px: int,
) -> Tuple[float, float, float, float]:
    """Return tolerance-based matched pixels for diagnostic use only.

    This is a symmetric relaxed overlap/Dice-style diagnostic, not strict
    Jaccard IoU. A predicted road pixel is accepted when it lies within
    ``buffer_px`` of ground truth, and vice versa.
    """
    prediction = prediction.bool()
    target = target.bool()
    if buffer_px > 0:
        kernel = 2 * int(buffer_px) + 1
        pred_dilated = F.max_pool2d(
            prediction[:, None].float(), kernel, stride=1, padding=buffer_px
        )[:, 0].bool()
        target_dilated = F.max_pool2d(
            target[:, None].float(), kernel, stride=1, padding=buffer_px
        )[:, 0].bool()
    else:
        pred_dilated, target_dilated = prediction, target
    prediction_hits = float((prediction & target_dilated).sum())
    target_hits = float((target & pred_dilated).sum())
    prediction_count = float(prediction.sum())
    target_count = float(target.sum())
    return prediction_hits, target_hits, prediction_count, target_count


@torch.no_grad()
def quick_train_iou(
    model: Segmentor,
    train_pairs: Sequence[Tuple[Path, Path]],
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """Measure strict full-image train IoU on a deterministic unaugmented subset."""
    sample_count = min(int(args.train_iou_samples), len(train_pairs))
    if sample_count <= 0:
        return float("nan")
    generator = np.random.default_rng(args.seed)
    indices = np.sort(
        generator.choice(len(train_pairs), size=sample_count, replace=False)
    )
    subset = [train_pairs[int(index)] for index in indices]
    dataset = RoadDataset(
        subset,
        crop_size=args.crop_size,
        training=False,
        full_image=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, args.num_workers),
        pin_memory=True,
    )
    was_training = model.training
    model.eval()
    intersection_sum = 0.0
    union_sum = 0.0
    progress = tqdm(loader, desc=f"Quick train IoU ({sample_count} full images)")
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = sliding_window_inference(
            model,
            images,
            args.crop_size,
            args.val_overlap,
            args.val_tile_batch_size,
            device,
            args.use_amp,
        )
        prediction = torch.sigmoid(road_logit(logits))[:, 0] >= 0.5
        target = masks > 0
        intersection_sum += float((prediction & target).sum())
        union_sum += float((prediction | target).sum())
    model.train(was_training)
    return intersection_sum / max(union_sum, 1.0)


class ProbabilityHistogram:
    """Threshold metrics without storing all 1500x1500 probability maps."""

    def __init__(self, bins: int = 1001) -> None:
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)
        self.per_image: List[Tuple[np.ndarray, np.ndarray]] = []

    def update(self, probability: Tensor, target: Tensor) -> None:
        probability_np = probability.detach().cpu().numpy().reshape(-1)
        target_np = target.detach().cpu().numpy().reshape(-1).astype(bool)
        indices = np.minimum(
            (probability_np * (self.bins - 1)).astype(np.int64), self.bins - 1
        )
        positive = np.bincount(indices[target_np], minlength=self.bins)
        negative = np.bincount(indices[~target_np], minlength=self.bins)
        self.positive += positive
        self.negative += negative
        self.per_image.append((positive, negative))

    @staticmethod
    def _counts(positive: np.ndarray, negative: np.ndarray, index: int):
        true_positive = int(positive[index:].sum())
        false_positive = int(negative[index:].sum())
        false_negative = int(positive[:index].sum())
        true_negative = int(negative[:index].sum())
        return true_positive, false_positive, false_negative, true_negative

    def at_threshold(self, threshold: float) -> Dict[str, float]:
        index = min(self.bins - 1, max(0, math.ceil(threshold * (self.bins - 1))))
        tp, fp, fn, tn = self._counts(self.positive, self.negative, index)
        road_iou = tp / max(tp + fp + fn, 1)
        background_iou = tn / max(tn + fp + fn, 1)
        macro_values = []
        for positive, negative in self.per_image:
            image_tp, image_fp, image_fn, _ = self._counts(positive, negative, index)
            denominator = image_tp + image_fp + image_fn
            if denominator > 0:
                macro_values.append(image_tp / denominator)
        return {
            "threshold": threshold,
            "road_iou_micro": road_iou,
            "road_iou_macro": float(np.mean(macro_values)) if macro_values else 0.0,
            "background_iou": background_iou,
            "miou": 0.5 * (road_iou + background_iou),
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }


@torch.no_grad()
def evaluate(
    model: Segmentor,
    loader: DataLoader,
    criterion: RoadLoss,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    forced_threshold: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    histogram = ProbabilityHistogram(args.threshold_bins)
    total_loss = 0.0
    samples = 0
    per_image: List[Dict[str, float | str]] = []
    relaxed_prediction_hits = 0.0
    relaxed_target_hits = 0.0
    relaxed_prediction_count = 0.0
    relaxed_target_count = 0.0
    dataset_pairs = getattr(loader.dataset, "pairs", [])
    progress = tqdm(loader, desc="Full-image sliding validation")
    for batch_index, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = sliding_window_inference(
            model,
            images,
            args.crop_size,
            args.val_overlap,
            args.val_tile_batch_size,
            device,
            args.use_amp,
        )
        losses = criterion(logits, masks, epoch)
        probability = torch.sigmoid(road_logit(logits))[:, 0]
        prediction = probability >= 0.5
        target = masks > 0
        intersection = float((prediction & target).sum())
        union = float((prediction | target).sum())
        image_iou = intersection / max(union, 1.0)
        image_name = (
            dataset_pairs[batch_index][0].name
            if batch_index < len(dataset_pairs)
            else f"image_{batch_index:03d}"
        )
        per_image.append(
            {
                "name": image_name,
                "iou": image_iou,
                "gt_fraction": float(target.float().mean()),
                "pred_fraction": float(prediction.float().mean()),
            }
        )
        relaxed = relaxed_overlap_components(
            prediction,
            target,
            args.relaxed_buffer_px,
        )
        relaxed_prediction_hits += relaxed[0]
        relaxed_target_hits += relaxed[1]
        relaxed_prediction_count += relaxed[2]
        relaxed_target_count += relaxed[3]
        histogram.update(probability, masks)
        total_loss += float(losses["total"])
        samples += 1
        progress.set_postfix(loss=f"{float(losses['total']):.4f}")

    diagnostic_epoch = (
        args.per_image_iou_interval > 0
        and (
            (epoch + 1) % args.per_image_iou_interval == 0
            or epoch + 1 == args.epochs
        )
    )
    if diagnostic_epoch and per_image:
        values = np.asarray([float(item["iou"]) for item in per_image])
        print("Per-image val IoU@0.50 (lowest first):")
        for item in sorted(per_image, key=lambda value: float(value["iou"])):
            print(
                f"  {str(item['name']):28s} IoU={float(item['iou']):.4f} "
                f"GT={100.0 * float(item['gt_fraction']):5.2f}% "
                f"Pred={100.0 * float(item['pred_fraction']):5.2f}%"
            )
        print(
            f"  distribution: min={values.min():.4f} "
            f"median={np.median(values):.4f} max={values.max():.4f} "
            f"std={values.std():.4f}"
        )

    fixed = histogram.at_threshold(0.5 if forced_threshold is None else forced_threshold)
    if forced_threshold is None:
        thresholds = np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step / 2,
            args.threshold_step,
        )
        candidates = [histogram.at_threshold(float(value)) for value in thresholds]
        selected = max(candidates, key=lambda item: item["road_iou_micro"])
    else:
        selected = fixed
    relaxed_precision = relaxed_prediction_hits / max(
        relaxed_prediction_count, 1.0
    )
    relaxed_recall = relaxed_target_hits / max(relaxed_target_count, 1.0)
    relaxed_f1 = (
        2.0 * relaxed_precision * relaxed_recall
        / max(relaxed_precision + relaxed_recall, 1e-12)
    )
    result = {
        "loss": total_loss / max(samples, 1),
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": relaxed_f1,
        "per_image_iou_min": min(
            (float(item["iou"]) for item in per_image), default=0.0
        ),
        "per_image_iou_max": max(
            (float(item["iou"]) for item in per_image), default=0.0
        ),
        "per_image_iou_std": float(
            np.std([float(item["iou"]) for item in per_image])
        ) if per_image else 0.0,
        **{f"fixed_{key}": value for key, value in fixed.items()},
        **{f"selected_{key}": value for key, value in selected.items()},
    }
    return result


def checkpoint_dict(
    model, optimizer, scheduler, scaler, epoch, best_iou, threshold, args
):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_road_iou": best_iou,
        "threshold": threshold,
        "args": vars(args),
    }


def resume_training(path, model, optimizer, scheduler, scaler):
    checkpoint = safe_torch_load(path)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Strict resume failed. If this checkpoint predates the centerline "
            "head or comes from another dataset, use --resume_mode transfer."
        ) from error
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("epoch", -1)) + 1,
        float(checkpoint.get("best_road_iou", 0.0)),
        float(checkpoint.get("threshold", 0.5)),
    )


def transfer_resume(path: str | Path, model: nn.Module) -> Tuple[int, float, float]:
    """Load compatible model tensors but restart optimization and metrics."""
    load_transfer(model, path)
    print(
        "Transfer resume: optimizer/scheduler/scaler reset; "
        "new auxiliary heads start from fresh initialization"
    )
    return 0, 0.0, 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="massachusetts",
        choices=["massachusetts", "deepglobe"],
    )
    parser.add_argument("--model_variant", default="coming", choices=["coming"])
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--decoder_channels", type=int, default=96)
    parser.add_argument(
        "--use_half_refine",
        action="store_true",
        help="Add an additive stem skip and one learned dense 3x3 refinement at H/2.",
    )
    parser.add_argument(
        "--half_refine_channels",
        type=int,
        default=32,
        help="Width of the optional H/2 refinement path.",
    )
    parser.add_argument(
        "--use_fullres_head",
        action="store_true",
        help="Add a learned PixelShuffle residual correction from H/2 to H.",
    )
    parser.add_argument(
        "--fullres_channels",
        type=int,
        default=16,
        help="Low width used by the learned full-resolution correction head.",
    )
    parser.add_argument("--local_blocks", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--global_blocks", type=int, nargs=2, default=(3, 4))
    parser.add_argument("--highres_kernel_size", type=int, default=5)
    parser.add_argument("--coming_kernel_size", type=int, default=7,
                        help="Context-stream kernel. High-resolution blocks use --highres_kernel_size=5.")
    parser.add_argument("--local_expansion", type=float, default=1.5,
                        help="Expansion in high-resolution stages and the final decoder block.")
    parser.add_argument("--global_expansion", type=float, default=2.0,
                        help="Expansion in context stages and low-resolution decoder blocks.")
    parser.add_argument("--local_spatial_ratio", type=float, default=0.25,
                        help="Fraction of local-block channels processed by dense spatial convs.")
    parser.add_argument("--global_spatial_ratio", type=float, default=0.5,
                        help="Fraction of global-block channels processed by dense spatial convs.")
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--val_mask_dir", default=None)
    parser.add_argument("--test_image_dir", default=None)
    parser.add_argument("--test_mask_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--road_crop_probability", type=float, default=0.5)
    parser.add_argument("--road_oversample_tries", type=int, default=4)
    parser.add_argument("--use_multiscale", action="store_true")
    parser.add_argument("--multiscale_min", type=float, default=0.75)
    parser.add_argument("--multiscale_max", type=float, default=1.50)
    parser.add_argument("--multiscale_probability", type=float, default=1.0)
    parser.add_argument(
        "--overfit_debug",
        action="store_true",
        help=(
            "Capacity test: train and validate on the same deterministic "
            "road-rich crops with all augmentation disabled."
        ),
    )
    parser.add_argument(
        "--overfit_samples",
        type=int,
        default=32,
        help="Number of fixed training images used by --overfit_debug.",
    )
    parser.add_argument(
        "--overfit_crop_candidates",
        type=int,
        default=32,
        help="Road-centred candidates searched for each deterministic crop.",
    )

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone_lr_factor", type=float, default=1.0)
    parser.add_argument("--stem_lr_factor", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--pos_weight_cap", type=float, default=3.0)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=0.0)
    parser.add_argument("--cldice_weight", type=float, default=0.0)
    parser.add_argument("--cldice_start_epoch", type=int, default=10)
    parser.add_argument("--cldice_downsample", type=int, default=2)
    parser.add_argument("--aux_weight", type=float, default=0.0)
    parser.add_argument("--aux_decay_exp", type=float, default=0.9)
    parser.add_argument("--centerline_weight", type=float, default=0.0)
    parser.add_argument("--centerline_start_epoch", type=int, default=0)
    parser.add_argument("--centerline_iterations", type=int, default=5)
    parser.add_argument("--centerline_pos_weight_cap", type=float, default=8.0)
    parser.add_argument("--centerline_dice_weight", type=float, default=0.5)
    parser.add_argument("--centerline_cldice_weight", type=float, default=0.25)
    parser.add_argument("--centerline_containment_weight", type=float, default=0.1)

    parser.add_argument("--teacher_checkpoint", default=None)
    parser.add_argument("--kd_weight", type=float, default=0.1)
    parser.add_argument("--kd_temperature", type=float, default=2.0)

    parser.add_argument("--val_overlap", type=int, default=128)
    parser.add_argument("--val_tile_batch_size", type=int, default=4)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--threshold_min", type=float, default=0.2)
    parser.add_argument("--threshold_max", type=float, default=0.8)
    parser.add_argument("--threshold_step", type=float, default=0.02)
    parser.add_argument("--threshold_bins", type=int, default=1001)
    parser.add_argument("--evaluate_test", action="store_true")
    parser.add_argument(
        "--alignment_samples", type=int, default=6,
        help="Number of one-time train image/mask overlays; use 0 to disable.",
    )
    parser.add_argument(
        "--per_image_iou_interval", type=int, default=5,
        help="Print named per-image val IoU every N epochs; use 0 to disable.",
    )
    parser.add_argument(
        "--train_iou_interval", type=int, default=10,
        help="Run full-image train-subset IoU every N epochs; use 0 to disable.",
    )
    parser.add_argument(
        "--train_iou_samples", type=int, default=64,
        help="Deterministic unaugmented train images used by quick train IoU.",
    )
    parser.add_argument(
        "--relaxed_buffer_px", type=int, default=3,
        help="Pixel tolerance for diagnostic relaxed precision/recall/F1.",
    )
    parser.add_argument(
        "--road_density_log_interval", type=int, default=20,
        help="Print road-pixel fraction every N training batches; use 0 to disable.",
    )

    parser.add_argument("--pretrained_weights", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--resume_mode", default="continue", choices=["continue", "transfer"],
        help="continue restores all states; transfer loads compatible model weights only.",
    )
    parser.add_argument("--save_dir", default="./checkpoints/coming_balanced")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    # Accepted only so old notebook commands do not crash.
    parser.add_argument("--ohem_keep_ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--loss_type", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_dataset_paths(args)
    if args.pretrained_weights and args.resume:
        raise ValueError("Use only one of --pretrained_weights and --resume")
    if not 0.0 <= args.road_crop_probability <= 1.0:
        raise ValueError("road_crop_probability must be in [0, 1]")
    if args.multiscale_min <= 0 or args.multiscale_max <= 0:
        raise ValueError("multiscale_min and multiscale_max must be positive")
    if args.multiscale_min > args.multiscale_max:
        raise ValueError("multiscale_min must be <= multiscale_max")
    if not 0.0 <= args.multiscale_probability <= 1.0:
        raise ValueError("multiscale_probability must be in [0, 1]")
    if args.centerline_weight < 0:
        raise ValueError("centerline_weight must be non-negative")
    if not 0.0 < args.local_spatial_ratio <= 1.0:
        raise ValueError("local_spatial_ratio must be in (0, 1]")
    if not 0.0 < args.global_spatial_ratio <= 1.0:
        raise ValueError("global_spatial_ratio must be in (0, 1]")
    if args.centerline_iterations < 1:
        raise ValueError("centerline_iterations must be >= 1")
    if args.ohem_keep_ratio is not None:
        print("INFO: --ohem_keep_ratio is ignored; this configuration does not use OHEM.")
    if args.loss_type is not None:
        print("INFO: --loss_type is deprecated; loss weights are controlled explicitly.")
    if args.overfit_debug:
        args.use_multiscale = False
        args.multiscale_probability = 0.0
        args.road_crop_probability = 0.0

    seed_everything(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = bool(args.use_amp and device.type == "cuda")

    train_loader, val_loader, train_pairs = make_loaders(args)
    if args.use_multiscale:
        print(
            "Multi-scale crop: enabled "
            f"scale=[{args.multiscale_min:.2f}, {args.multiscale_max:.2f}], "
            f"probability={args.multiscale_probability:.2f}"
        )
    else:
        print("Multi-scale crop: disabled")
    dump_alignment_check(
        train_pairs,
        Path(args.save_dir) / "sanity_check",
        args.alignment_samples,
    )
    if args.pos_weight is not None:
        pos_weight = args.pos_weight
        print(f"Using explicit pos_weight={pos_weight:.4f}")
    elif args.use_class_weights:
        raw_weight, pos_weight = compute_pos_weight(train_pairs, args.pos_weight_cap)
        print(f"Road imbalance raw={raw_weight:.4f}, sqrt/capped pos_weight={pos_weight:.4f}")
    else:
        pos_weight = 1.0

    model = build_model(args).to(device)
    if args.pretrained_weights:
        load_transfer(model, args.pretrained_weights)
    teacher = build_teacher(args, model, device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {total_parameters:,} | device={device} | AMP={args.use_amp}")
    optimizer = AdamW(
        parameter_groups(model, args),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    scheduler = build_scheduler(optimizer, updates_per_epoch, args)
    scaler = make_scaler(args.use_amp)
    criterion = RoadLoss(
        pos_weight=pos_weight,
        dice_weight=args.dice_weight,
        boundary_weight=args.boundary_weight,
        cldice_weight=args.cldice_weight,
        cldice_start_epoch=args.cldice_start_epoch,
        cldice_downsample=args.cldice_downsample,
    ).to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    start_epoch, best_road_iou, best_threshold = 0, 0.0, 0.5
    best_selected_iou = 0.0
    if args.resume:
        if args.resume_mode == "continue":
            start_epoch, best_road_iou, best_threshold = resume_training(
                args.resume, model, optimizer, scheduler, scaler
            )
            print(
                f"Continued at epoch {start_epoch + 1}; "
                f"best road IoU={best_road_iou:.4f}"
            )
        else:
            start_epoch, best_road_iou, best_threshold = transfer_resume(
                args.resume, model
            )

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model,
            teacher,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            args,
        )
        print(
            f"Epoch {epoch + 1}: train loss={train_metrics['total']:.4f}, "
            f"BCE={train_metrics['bce']:.4f}, DiceLoss={train_metrics['dice']:.4f}, "
            f"boundary={train_metrics['boundary']:.4f}, clDice={train_metrics['cldice']:.4f}, "
            f"aux={train_metrics['aux']:.4f} (w={train_metrics['aux_weight']:.4f}), "
            f"centerline={train_metrics['centerline']:.4f}, "
            f"centerlineDice={train_metrics['centerline_dice']:.4f}, "
            f"max_grad={train_metrics['max_gradient']:.2f}"
        )
        print(
            "  Train crop road fraction: "
            f"mean={train_metrics['road_fraction_mean']:.4f}, "
            f"min={train_metrics['road_fraction_min']:.4f}, "
            f"max={train_metrics['road_fraction_max']:.4f}"
        )

        should_validate = (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs
        if should_validate:
            train_iou = None
            if (
                args.train_iou_interval > 0
                and (
                    (epoch + 1) % args.train_iou_interval == 0
                    or epoch + 1 == args.epochs
                )
            ):
                train_iou = quick_train_iou(
                    model,
                    train_pairs,
                    device,
                    args,
                )
            validation = evaluate(model, val_loader, criterion, device, args, epoch)
            # Select checkpoints with the fixed operating point. Threshold
            # search remains a reporting/calibration metric only.
            checkpoint_iou = validation["fixed_road_iou_micro"]
            selected_iou = validation["selected_road_iou_micro"]
            selected_threshold = validation["selected_threshold"]
            print(
                f"Val full-image: fixed@0.50 road IoU={validation['fixed_road_iou_micro']:.4f}, "
                f"mIoU={validation['fixed_miou']:.4f} | "
                f"selected@{selected_threshold:.2f} road IoU micro={selected_iou:.4f}, "
                f"macro={validation['selected_road_iou_macro']:.4f}"
            )
            print(
                f"  Relaxed overlap ±{args.relaxed_buffer_px}px: "
                f"P={validation['relaxed_precision']:.4f}, "
                f"R={validation['relaxed_recall']:.4f}, "
                f"F1={validation['relaxed_f1']:.4f} "
                "(diagnostic; not strict IoU)"
            )
            if train_iou is not None:
                print(
                    f"  Generalization check: train IoU@0.50={train_iou:.4f} "
                    f"vs val IoU@0.50={checkpoint_iou:.4f}, "
                    f"gap={train_iou - checkpoint_iou:+.4f}"
                )
            if checkpoint_iou > best_road_iou:
                best_road_iou, best_threshold = checkpoint_iou, 0.5
                torch.save(
                    checkpoint_dict(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_road_iou,
                        best_threshold,
                        args,
                    ),
                    save_dir / "best_road_iou.pt",
                )
                print(
                    f"New best fixed@0.50 road IoU={best_road_iou:.4f}; "
                    "checkpoint saved"
                )
            if selected_iou > best_selected_iou:
                best_selected_iou = selected_iou
                torch.save(
                    checkpoint_dict(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_selected_iou,
                        selected_threshold,
                        args,
                    ),
                    save_dir / "best_selected_iou.pt",
                )
                print(
                    f"New best selected IoU={best_selected_iou:.4f} "
                    f"at threshold={selected_threshold:.2f}; checkpoint saved"
                )

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            torch.save(
                checkpoint_dict(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_road_iou,
                    best_threshold,
                    args,
                ),
                save_dir / "last.pt",
            )

    if args.evaluate_test:
        if not args.test_image_dir or not args.test_mask_dir:
            raise FileNotFoundError("Test directories were not configured")
        test_image_dir, test_mask_dir = Path(args.test_image_dir), Path(args.test_mask_dir)
        if not test_image_dir.is_dir() or not test_mask_dir.is_dir():
            raise FileNotFoundError("Official test folders were not found")
        best_checkpoint = safe_torch_load(save_dir / "best_road_iou.pt")
        model.load_state_dict(best_checkpoint["model"], strict=True)
        best_threshold = float(best_checkpoint["threshold"])
        test_dataset = RoadDataset(
            build_pairs(test_image_dir, test_mask_dir),
            crop_size=args.crop_size,
            full_image=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            args,
            args.epochs - 1,
            forced_threshold=best_threshold,
        )
        print(
            f"Official test @ val threshold {best_threshold:.2f}: "
            f"road IoU micro={test_metrics['selected_road_iou_micro']:.4f}, "
            f"macro={test_metrics['selected_road_iou_macro']:.4f}, "
            f"mIoU={test_metrics['selected_miou']:.4f}"
        )


if __name__ == "__main__":
    main()
