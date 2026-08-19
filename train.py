"""Modern training pipeline for ``modeling/model.py``.

Expected model file
-------------------
``modeling/model.py`` must export:

    RoadRepVGGNet
    RoadSegCenterlineLoss

These classes are provided by ``CoMingNet_RepVGG_RoadAux.py``.  Place that
file at ``modeling/model.py`` before running this script from the repository
root.

Design goals
------------
* Train native-resolution 1024x1024 crops without shrinking thin roads.
* Prefer road-containing crops while retaining some random background crops.
* Keep exactly one road-specific auxiliary: OS4 centerline prediction.
* Validate the native image with overlap-tiled inference and weighted blending.
* Native PyTorch DDP for one process per GPU; distributed train and validation.
* AMP, channels-last, gradient accumulation, gradient clipping, model EMA.
* AdamW with no weight decay on normalization and bias parameters.
* Per-update linear warmup followed by cosine decay.
* Fixed@0.50 checkpoint selection plus validation-only threshold calibration.
* Atomic resumable checkpoints and JSONL metric logging.

Recommended Kaggle P100 command
-------------------------------
torchrun --standalone --nproc_per_node=2 train_repvgg_roadaux.py \
    --dataset massachusetts \
    --crop_size 1024 \
    --road_crop_probability 0.70 \
    --val_tile_size 1024 \
    --val_overlap 256 \
    --batch_size 2 \
    --accumulation_steps 2 \
    --epochs 120 \
    --lr 2e-4 \
    --channels 40 \
    --decoder_channels 128 \
    --aux_weight 0.20 \
    --save_dir ./checkpoints/repvgg_roadaux_1024
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from modeling.model import RoadRepVGGNet, RoadSegCenterlineLoss


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_masks", "_gt", "_label", "_labels")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def init_distributed() -> Tuple[bool, int, int, int, torch.device]:
    """Initialize torchrun DDP, while preserving ordinary one-GPU execution."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA GPUs")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def distributed_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not distributed_active() or dist.get_rank() == 0


def rank_zero_print(*values, **kwargs) -> None:
    if is_main_process():
        print(*values, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


@torch.no_grad()
def broadcast_module(module: nn.Module, source: int = 0) -> None:
    """Make EMA parameters and buffers bit-identical before distributed eval."""
    if not distributed_active():
        return
    for tensor in module.state_dict().values():
        dist.broadcast(tensor, src=source)


def cleanup_distributed() -> None:
    if distributed_active():
        dist.destroy_process_group()


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding or duplicated evaluation samples."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, math.ceil(remaining / self.world_size))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sample_key(path: Path) -> str:
    key = path.stem.lower()
    for suffix in (
        "_image",
        "_images",
        "_img",
        "_sat",
        "_mask",
        "_masks",
        "_gt",
        "_label",
        "_labels",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def index_files(folder: str | Path, role: Optional[str] = None) -> Dict[str, Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if role is not None:
        if role not in {"image", "mask"}:
            raise ValueError("role must be image, mask, or None")
        files = [
            path
            for path in files
            if any(path.stem.lower().endswith(suffix) for suffix in MASK_SUFFIXES)
            == (role == "mask")
        ]
    if not files:
        raise RuntimeError(f"No supported image files found in {folder}")
    indexed: Dict[str, Path] = {}
    for path in files:
        key = sample_key(path)
        if key in indexed:
            raise RuntimeError(
                f"Duplicate sample key '{key}': {indexed[key]} and {path}"
            )
        indexed[key] = path
    return indexed


def build_pairs(image_dir: str | Path, mask_dir: str | Path) -> List[Tuple[Path, Path]]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    same_folder = image_dir.resolve() == mask_dir.resolve()
    images = index_files(image_dir, role="image" if same_folder else None)
    masks = index_files(mask_dir, role="mask" if same_folder else None)
    common_keys = sorted(images.keys() & masks.keys())
    if len(common_keys) != len(images) or len(common_keys) != len(masks):
        missing_masks = sorted(images.keys() - masks.keys())[:5]
        missing_images = sorted(masks.keys() - images.keys())[:5]
        raise RuntimeError(
            "Image/mask pairing mismatch: "
            f"images={len(images)}, masks={len(masks)}, pairs={len(common_keys)}, "
            f"missing_masks={missing_masks}, missing_images={missing_images}"
        )
    return [(images[key], masks[key]) for key in common_keys]


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return (mask > 0).astype(np.uint8)


def pad_pair_to_size(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad only when an image side is smaller than the requested crop."""
    height, width = mask.shape
    pad_height = max(0, size - height)
    pad_width = max(0, size - width)
    if pad_height == 0 and pad_width == 0:
        return image, mask

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    image_mode = "reflect" if height > 1 and width > 1 else "edge"
    image = np.pad(
        image,
        ((top, bottom), (left, right), (0, 0)),
        mode=image_mode,
    )
    mask = np.pad(
        mask,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0,
    )
    return image, mask


def coarse_max_mask(mask: np.ndarray, factor: int = 8) -> np.ndarray:
    """Cheap max-pooled mask used only to choose road-aware crop locations."""
    height, width = mask.shape
    pooled_height = math.ceil(height / factor)
    pooled_width = math.ceil(width / factor)
    pad_height = pooled_height * factor - height
    pad_width = pooled_width * factor - width
    padded = np.pad(
        mask,
        ((0, pad_height), (0, pad_width)),
        mode="constant",
        constant_values=0,
    )
    return padded.reshape(
        pooled_height, factor, pooled_width, factor
    ).max(axis=(1, 3))


def random_crop_pair(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    road_probability: float,
    min_road_fraction: float,
    tries: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a native-resolution crop, optionally centred around road pixels.

    Crop quality is estimated on an 8x max-pooled mask. This avoids repeatedly
    summing million-pixel candidate crops inside every DataLoader worker.
    """
    image, mask = pad_pair_to_size(image, mask, crop_size)
    height, width = mask.shape
    max_y = height - crop_size
    max_x = width - crop_size

    def random_origin() -> Tuple[int, int]:
        y = random.randint(0, max_y) if max_y > 0 else 0
        x = random.randint(0, max_x) if max_x > 0 else 0
        return y, x

    y0, x0 = random_origin()
    use_road_crop = road_probability > 0 and random.random() < road_probability
    if use_road_crop and mask.any():
        factor = 8
        coarse = coarse_max_mask(mask, factor=factor)
        road_cells = np.flatnonzero(coarse)
        coarse_crop = max(1, math.ceil(crop_size / factor))
        best_score = -1.0
        for _ in range(max(1, tries)):
            flat_index = int(road_cells[random.randrange(len(road_cells))])
            coarse_y, coarse_x = np.unravel_index(flat_index, coarse.shape)
            road_y = min(height - 1, coarse_y * factor + random.randrange(factor))
            road_x = min(width - 1, coarse_x * factor + random.randrange(factor))

            # Keep the selected road point away from a fixed crop position.
            # This prevents a centre bias while still guaranteeing road support.
            y0 = min(max(road_y - random.randrange(crop_size), 0), max_y)
            x0 = min(max(road_x - random.randrange(crop_size), 0), max_x)
            cy0 = y0 // factor
            cx0 = x0 // factor
            region = coarse[
                cy0 : min(coarse.shape[0], cy0 + coarse_crop),
                cx0 : min(coarse.shape[1], cx0 + coarse_crop),
            ]
            score = float(region.mean()) if region.size else 0.0
            if score > best_score:
                best_score = score
                best_origin = (y0, x0)
            if score >= min_road_fraction:
                break
        y0, x0 = best_origin

    image_crop = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
    mask_crop = mask[y0 : y0 + crop_size, x0 : x0 + crop_size]
    return np.ascontiguousarray(image_crop), np.ascontiguousarray(mask_crop)


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        image, mask = image[::-1, :], mask[::-1, :]
    rotations = random.randrange(4)
    if rotations:
        image = np.rot90(image, rotations)
        mask = np.rot90(mask, rotations)

    pil_image = Image.fromarray(np.ascontiguousarray(image))
    if random.random() < 0.60:
        pil_image = ImageEnhance.Brightness(pil_image).enhance(
            random.uniform(0.85, 1.15)
        )
    if random.random() < 0.60:
        pil_image = ImageEnhance.Contrast(pil_image).enhance(
            random.uniform(0.85, 1.15)
        )
    if random.random() < 0.35:
        pil_image = ImageEnhance.Color(pil_image).enhance(
            random.uniform(0.90, 1.10)
        )
    return np.asarray(pil_image, dtype=np.uint8), mask


def image_to_tensor(image: np.ndarray) -> Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


class RoadCropDataset(Dataset):
    """Native-resolution random crops for training."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        crop_size: int = 1024,
        road_crop_probability: float = 0.70,
        road_crop_min_fraction: float = 0.002,
        road_crop_tries: int = 8,
    ) -> None:
        self.pairs = list(pairs)
        self.crop_size = int(crop_size)
        self.road_crop_probability = float(road_crop_probability)
        self.road_crop_min_fraction = float(road_crop_min_fraction)
        self.road_crop_tries = int(road_crop_tries)
        if self.crop_size < 32 or self.crop_size % 32 != 0:
            raise ValueError("crop_size must be >= 32 and divisible by 32")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = read_rgb(image_path)
        mask = read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"Image/mask mismatch for {image_path.name}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )
        image, mask = random_crop_pair(
            image,
            mask,
            crop_size=self.crop_size,
            road_probability=self.road_crop_probability,
            min_road_fraction=self.road_crop_min_fraction,
            tries=self.road_crop_tries,
        )
        image, mask = augment_pair(image, mask)
        return image_to_tensor(image), torch.from_numpy(
            np.ascontiguousarray(mask)
        ).long()


class RoadNativeValidationDataset(Dataset):
    """Return untouched native-resolution images and masks for tiled validation."""

    def __init__(self, pairs: Sequence[Tuple[Path, Path]]) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        image = read_rgb(image_path)
        mask = read_binary_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"Image/mask mismatch for {image_path.name}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )
        return (
            image_to_tensor(image),
            torch.from_numpy(np.ascontiguousarray(mask)).long(),
            image_path.stem,
        )


def configure_dataset_paths(args: argparse.Namespace) -> None:
    if args.dataset == "massachusetts":
        root = Path(
            args.data_root
            or "/kaggle/input/datasets/balraj98/"
            "massachusetts-roads-dataset/tiff"
        )
        args.train_image_dir = args.train_image_dir or str(root / "train")
        args.train_mask_dir = args.train_mask_dir or str(root / "train_labels")
        args.val_image_dir = args.val_image_dir or str(root / "val")
        args.val_mask_dir = args.val_mask_dir or str(root / "val_labels")
        args.test_image_dir = args.test_image_dir or str(root / "test")
        args.test_mask_dir = args.test_mask_dir or str(root / "test_labels")
    else:
        root = Path(
            args.data_root
            or "/kaggle/input/datasets/balraj98/"
            "deepglobe-road-extraction-dataset/train"
        )
        args.train_image_dir = args.train_image_dir or str(root)
        args.train_mask_dir = args.train_mask_dir or str(root)


def resolve_splits(
    args: argparse.Namespace,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    all_train_pairs = build_pairs(args.train_image_dir, args.train_mask_dir)
    val_image_dir = Path(args.val_image_dir) if args.val_image_dir else None
    val_mask_dir = Path(args.val_mask_dir) if args.val_mask_dir else None
    if (
        val_image_dir is not None
        and val_mask_dir is not None
        and val_image_dir.is_dir()
        and val_mask_dir.is_dir()
    ):
        val_pairs = build_pairs(val_image_dir, val_mask_dir)
        rank_zero_print(
            f"Official split: train={len(all_train_pairs)}, val={len(val_pairs)}"
        )
        return all_train_pairs, val_pairs

    generator = np.random.default_rng(args.seed)
    indices = generator.permutation(len(all_train_pairs))
    val_count = max(1, round(len(all_train_pairs) * args.val_ratio))
    val_indices = set(indices[:val_count].tolist())
    train_pairs = [
        pair for index, pair in enumerate(all_train_pairs) if index not in val_indices
    ]
    val_pairs = [
        pair for index, pair in enumerate(all_train_pairs) if index in val_indices
    ]
    rank_zero_print(
        "WARNING: official validation folders not found; using deterministic "
        f"random split train={len(train_pairs)}, val={len(val_pairs)}"
    )
    return train_pairs, val_pairs


def make_loaders(
    args: argparse.Namespace,
) -> Tuple[
    DataLoader,
    DataLoader,
    List[Tuple[Path, Path]],
    Optional[DistributedSampler],
]:
    train_pairs, val_pairs = resolve_splits(args)
    train_dataset = RoadCropDataset(
        train_pairs,
        crop_size=args.crop_size,
        road_crop_probability=args.road_crop_probability,
        road_crop_min_fraction=args.road_crop_min_fraction,
        road_crop_tries=args.road_crop_tries,
    )
    val_dataset = RoadNativeValidationDataset(val_pairs)
    generator = torch.Generator()
    generator.manual_seed(args.seed + args.rank)
    train_sampler: Optional[DistributedSampler]
    val_sampler: Optional[DistributedEvalSampler]
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        val_sampler = DistributedEvalSampler(
            val_dataset,
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        train_sampler = None
        val_sampler = None
    common = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_pairs, train_sampler


def compute_road_weight(
    pairs: Sequence[Tuple[Path, Path]],
    cap: float,
) -> Tuple[float, float]:
    positive = 0
    total = 0
    for _, mask_path in tqdm(pairs, desc="Computing road imbalance"):
        mask = read_binary_mask(mask_path)
        positive += int(mask.sum())
        total += int(mask.size)
    raw_ratio = (total - positive) / max(positive, 1)
    return raw_ratio, min(math.sqrt(raw_ratio), cap)


def distributed_road_weight(
    pairs: Sequence[Tuple[Path, Path]],
    cap: float,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute dataset statistics once on rank 0 and broadcast the result."""
    if is_main_process():
        raw_ratio, class_weight = compute_road_weight(pairs, cap)
    else:
        raw_ratio, class_weight = 0.0, 0.0
    statistics = torch.tensor(
        [raw_ratio, class_weight], dtype=torch.float64, device=device
    )
    if distributed_active():
        dist.broadcast(statistics, src=0)
    return float(statistics[0].item()), float(statistics[1].item())


def parameter_groups(model: nn.Module, weight_decay: float):
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    updates_per_epoch: int,
    args: argparse.Namespace,
) -> LambdaLR:
    total_updates = max(1, args.epochs * updates_per_epoch)
    warmup_updates = min(
        total_updates - 1,
        max(0, args.warmup_epochs * updates_per_epoch),
    )

    def schedule(update: int) -> float:
        if warmup_updates > 0 and update < warmup_updates:
            progress = update / max(1, warmup_updates)
            return args.warmup_start_factor + (
                1.0 - args.warmup_start_factor
            ) * progress
        progress = (update - warmup_updates) / max(
            1, total_updates - warmup_updates
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    return LambdaLR(optimizer, schedule)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class ModelEMA:
    """Exponential moving average of parameters and floating-point buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # A short warm start prevents stale random initialization early on.
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)


def train_one_epoch(
    model: nn.Module,
    ema: ModelEMA,
    loader: DataLoader,
    criterion: RoadSegCenterlineLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    target_aux_weight = args.aux_weight
    if args.aux_warmup_epochs > 0:
        aux_scale = min(1.0, (epoch + 1) / args.aux_warmup_epochs)
    else:
        aux_scale = 1.0
    criterion.aux_weight = target_aux_weight * aux_scale

    meters = {
        "total": RunningAverage(),
        "main_ce": RunningAverage(),
        "main_dice": RunningAverage(),
        "aux": RunningAverage(),
        "centerline_bce": RunningAverage(),
        "centerline_dice": RunningAverage(),
        "road_fraction": RunningAverage(),
    }
    optimizer.zero_grad(set_to_none=True)
    successful_updates = 0
    skipped_nonfinite = 0
    max_gradient = 0.0
    progress = tqdm(
        loader,
        desc=f"Train {epoch + 1:03d}",
        leave=False,
        disable=not is_main_process(),
    )

    for step, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)

        group_start = (step // args.accumulation_steps) * args.accumulation_steps
        group_size = min(
            args.accumulation_steps,
            len(loader) - group_start,
        )
        do_update = (
            (step + 1) % args.accumulation_steps == 0
            or step + 1 == len(loader)
        )
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and not do_update
            else nullcontext()
        )
        with sync_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.use_amp,
            ):
                outputs = model(images)
                losses = criterion(outputs, masks)
                scaled_loss = losses["loss_total"] / group_size

            finite_flag = torch.tensor(
                int(bool(torch.isfinite(scaled_loss).item())),
                dtype=torch.int32,
                device=device,
            )
            if distributed_active():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if not bool(finite_flag.item()):
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(scaled_loss).backward()

        if do_update:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            grad_value = float(grad_norm)
            if math.isfinite(grad_value):
                max_gradient = max(max_gradient, grad_value)
            previous_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step_succeeded = float(scaler.get_scale()) >= previous_scale
            if step_succeeded:
                scheduler.step()
                ema.update(unwrap_model(model))
                successful_updates += 1

        batch_size = images.shape[0]
        meters["total"].update(float(losses["loss_total"].detach()), batch_size)
        meters["main_ce"].update(float(losses["loss_main_ce"]), batch_size)
        meters["main_dice"].update(float(losses["loss_main_dice"]), batch_size)
        meters["aux"].update(float(losses["loss_aux_centerline"]), batch_size)
        meters["centerline_bce"].update(
            float(losses["loss_centerline_bce"]), batch_size
        )
        meters["centerline_dice"].update(
            float(losses["loss_centerline_dice"]), batch_size
        )
        meters["road_fraction"].update(float((masks > 0).float().mean()), batch_size)
        progress.set_postfix(
            loss=f"{meters['total'].mean:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            aux=f"{criterion.aux_weight:.3f}",
        )

    if distributed_active():
        for meter in meters.values():
            statistics = torch.tensor(
                [meter.total, float(meter.count)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            meter.total = float(statistics[0].item())
            meter.count = int(statistics[1].item())
        counters = torch.tensor(
            [successful_updates, skipped_nonfinite],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(counters, op=dist.ReduceOp.MAX)
        successful_updates = int(counters[0].item())
        skipped_nonfinite = int(counters[1].item())
        maximum = torch.tensor(max_gradient, dtype=torch.float32, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        max_gradient = float(maximum.item())

    return {
        key: meter.mean for key, meter in meters.items()
    } | {
        "aux_weight": criterion.aux_weight,
        "lr": optimizer.param_groups[0]["lr"],
        "max_gradient": max_gradient,
        "successful_updates": float(successful_updates),
        "skipped_nonfinite": float(skipped_nonfinite),
    }


def histogram_counts(
    probability: np.ndarray,
    target: np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    positive_hist, _ = np.histogram(
        probability[target], bins=bins, range=(0.0, 1.0)
    )
    negative_hist, _ = np.histogram(
        probability[~target], bins=bins, range=(0.0, 1.0)
    )
    return positive_hist.astype(np.int64), negative_hist.astype(np.int64)


def counts_at_threshold(
    positive_hist: np.ndarray,
    negative_hist: np.ndarray,
    threshold: float,
) -> Tuple[int, int, int, int]:
    bins = positive_hist.shape[-1]
    index = min(bins - 1, max(0, int(math.floor(threshold * bins))))
    tp = int(positive_hist[index:].sum())
    fn = int(positive_hist[:index].sum())
    fp = int(negative_hist[index:].sum())
    tn = int(negative_hist[:index].sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    eps = 1e-9
    road_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "road_iou": road_iou,
        "background_iou": background_iou,
        "miou": 0.5 * (road_iou + background_iou),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def sliding_positions(length: int, tile_size: int, overlap: int) -> List[int]:
    """Cover a dimension completely and always anchor the final border tile."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def hann_blend_weight(tile_size: int, device: torch.device) -> Tensor:
    """Smooth overlap blending with non-zero image-border support."""
    window = torch.hann_window(
        tile_size,
        periodic=False,
        dtype=torch.float32,
        device=device,
    )
    weight = window[:, None] * window[None, :]
    weight = 0.05 + 0.95 * weight
    return weight[None, None]


@torch.inference_mode()
def sliding_window_logits(
    model: nn.Module,
    image: Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Tensor:
    """Infer one native-resolution image using batched overlapping tiles."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("sliding_window_logits expects a BCHW tensor with B=1")

    tile_size = args.val_tile_size
    _, _, original_height, original_width = image.shape
    padded_height = max(original_height, tile_size)
    padded_width = max(original_width, tile_size)
    pad_height = padded_height - original_height
    pad_width = padded_width - original_width
    if pad_height or pad_width:
        image = F.pad(
            image,
            (0, pad_width, 0, pad_height),
            mode="replicate",
        )

    y_positions = sliding_positions(
        padded_height, tile_size, args.val_overlap
    )
    x_positions = sliding_positions(
        padded_width, tile_size, args.val_overlap
    )
    positions = [(y, x) for y in y_positions for x in x_positions]
    blend = hann_blend_weight(tile_size, device)
    logits_sum = torch.zeros(
        (1, 2, padded_height, padded_width),
        dtype=torch.float32,
        device=device,
    )
    weight_sum = torch.zeros(
        (1, 1, padded_height, padded_width),
        dtype=torch.float32,
        device=device,
    )

    for start in range(0, len(positions), args.val_tile_batch_size):
        batch_positions = positions[start : start + args.val_tile_batch_size]
        tiles = torch.cat(
            [
                image[:, :, y : y + tile_size, x : x + tile_size]
                for y, x in batch_positions
            ],
            dim=0,
        )
        if args.channels_last:
            tiles = tiles.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.use_amp,
        ):
            tile_logits = model(tiles)
            if isinstance(tile_logits, (tuple, list)):
                tile_logits = tile_logits[-1]
            if tile_logits.shape[-2:] != (tile_size, tile_size):
                tile_logits = F.interpolate(
                    tile_logits,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )
        tile_logits = tile_logits.float()
        for batch_index, (y, x) in enumerate(batch_positions):
            logits_sum[
                :, :, y : y + tile_size, x : x + tile_size
            ].add_(tile_logits[batch_index : batch_index + 1] * blend)
            weight_sum[
                :, :, y : y + tile_size, x : x + tile_size
            ].add_(blend)

    logits = logits_sum / weight_sum.clamp_min_(1e-6)
    return logits[:, :, :original_height, :original_width]


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    positive_hists: List[np.ndarray] = []
    negative_hists: List[np.ndarray] = []
    relaxed_supported_predictions = 0
    relaxed_predictions = 0
    relaxed_supported_targets = 0
    relaxed_targets = 0

    progress = tqdm(
        loader,
        desc="Validate",
        leave=False,
        disable=not is_main_process(),
    )
    for images, original_masks, names in progress:
        del names
        images = images.to(device, non_blocking=True)
        original_masks = original_masks.to(device, non_blocking=True)
        logits = sliding_window_logits(model, images, device, args)
        probability = torch.softmax(logits, dim=1)[:, 1]

        target_bool = original_masks > 0
        probability_np = probability[0].cpu().numpy()
        target_np = target_bool[0].cpu().numpy().astype(bool)
        positive_hist, negative_hist = histogram_counts(
            probability_np,
            target_np,
            args.threshold_bins,
        )
        positive_hists.append(positive_hist)
        negative_hists.append(negative_hist)

        radius = args.relaxed_buffer_px
        prediction = probability >= 0.5
        if radius > 0:
            kernel = 2 * radius + 1
            dilated_target = F.max_pool2d(
                target_bool[:, None].float(), kernel, stride=1, padding=radius
            )[:, 0] > 0
            dilated_prediction = F.max_pool2d(
                prediction[:, None].float(), kernel, stride=1, padding=radius
            )[:, 0] > 0
        else:
            dilated_target = target_bool
            dilated_prediction = prediction
        relaxed_supported_predictions += int((prediction & dilated_target).sum())
        relaxed_predictions += int(prediction.sum())
        relaxed_supported_targets += int((target_bool & dilated_prediction).sum())
        relaxed_targets += int(target_bool.sum())

    positive_total = (
        np.stack(positive_hists).sum(axis=0)
        if positive_hists
        else np.zeros(args.threshold_bins, dtype=np.int64)
    )
    negative_total = (
        np.stack(negative_hists).sum(axis=0)
        if negative_hists
        else np.zeros(args.threshold_bins, dtype=np.int64)
    )
    if distributed_active():
        global_hists = torch.from_numpy(
            np.stack((positive_total, negative_total))
        ).to(device=device, dtype=torch.int64)
        dist.all_reduce(global_hists, op=dist.ReduceOp.SUM)
        positive_total = global_hists[0].cpu().numpy()
        negative_total = global_hists[1].cpu().numpy()

        relaxed_counts = torch.tensor(
            [
                relaxed_supported_predictions,
                relaxed_predictions,
                relaxed_supported_targets,
                relaxed_targets,
            ],
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(relaxed_counts, op=dist.ReduceOp.SUM)
        (
            relaxed_supported_predictions,
            relaxed_predictions,
            relaxed_supported_targets,
            relaxed_targets,
        ) = [int(value) for value in relaxed_counts.cpu().tolist()]

    fixed_counts = counts_at_threshold(positive_total, negative_total, 0.5)
    fixed = metrics_from_counts(*fixed_counts)

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + 0.5 * args.threshold_step,
        args.threshold_step,
    )
    selected_threshold = 0.5
    selected = fixed
    selected_counts = fixed_counts
    for threshold in thresholds.tolist():
        counts = counts_at_threshold(positive_total, negative_total, threshold)
        metrics = metrics_from_counts(*counts)
        if metrics["road_iou"] > selected["road_iou"]:
            selected_threshold = float(threshold)
            selected = metrics
            selected_counts = counts

    per_image_ious = []
    for positive_hist, negative_hist in zip(positive_hists, negative_hists):
        counts = counts_at_threshold(
            positive_hist, negative_hist, selected_threshold
        )
        per_image_ious.append(metrics_from_counts(*counts)["road_iou"])
    macro_sum = float(sum(per_image_ious))
    macro_count = len(per_image_ious)
    if distributed_active():
        macro_statistics = torch.tensor(
            [macro_sum, float(macro_count)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(macro_statistics, op=dist.ReduceOp.SUM)
        macro_sum = float(macro_statistics[0].item())
        macro_count = int(macro_statistics[1].item())

    relaxed_precision = relaxed_supported_predictions / max(
        relaxed_predictions, 1
    )
    relaxed_recall = relaxed_supported_targets / max(relaxed_targets, 1)
    relaxed_f1 = 2.0 * relaxed_precision * relaxed_recall / max(
        relaxed_precision + relaxed_recall, 1e-9
    )
    del selected_counts
    return {
        "fixed_road_iou": fixed["road_iou"],
        "fixed_miou": fixed["miou"],
        "fixed_f1": fixed["f1"],
        "fixed_precision": fixed["precision"],
        "fixed_recall": fixed["recall"],
        "selected_threshold": selected_threshold,
        "selected_road_iou_micro": selected["road_iou"],
        "selected_road_iou_macro": macro_sum / max(macro_count, 1),
        "selected_miou": selected["miou"],
        "selected_f1": selected["f1"],
        "selected_precision": selected["precision"],
        "selected_recall": selected["recall"],
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": relaxed_f1,
    }


def safe_torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_state(
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    epoch: int,
    best_fixed_iou: float,
    best_selected_iou: float,
    best_threshold: float,
    args: argparse.Namespace,
) -> Dict:
    return {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "ema": ema.module.state_dict(),
        "ema_updates": ema.updates,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_fixed_iou": best_fixed_iou,
        "best_selected_iou": best_selected_iou,
        "best_threshold": best_threshold,
        "args": vars(args),
    }


def atomic_torch_save(state: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def resume_training(
    checkpoint_path: str | Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler,
    device: torch.device,
) -> Tuple[int, float, float, float]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.module.load_state_dict(checkpoint.get("ema", checkpoint["model"]), strict=True)
    ema.updates = int(checkpoint.get("ema_updates", 0))
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_fixed_iou", 0.0)),
        float(checkpoint.get("best_selected_iou", 0.0)),
        float(checkpoint.get("best_threshold", 0.5)),
    )


def append_jsonl(path: str | Path, record: Dict) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RepVGG road segmentation with centerline auxiliary and "
            "optional torchrun DDP"
        )
    )
    parser.add_argument(
        "--dataset", choices=("massachusetts", "deepglobe"), default="massachusetts"
    )
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--train_image_dir", default=None)
    parser.add_argument("--train_mask_dir", default=None)
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--val_mask_dir", default=None)
    parser.add_argument("--test_image_dir", default=None)
    parser.add_argument("--test_mask_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    parser.add_argument(
        "--crop_size",
        "--image_size",
        dest="crop_size",
        type=int,
        default=1024,
        help="Native-resolution training crop size; --image_size is an alias",
    )
    parser.add_argument("--road_crop_probability", type=float, default=0.70)
    parser.add_argument("--road_crop_min_fraction", type=float, default=0.002)
    parser.add_argument("--road_crop_tries", type=int, default=8)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--decoder_channels", type=int, default=128)
    parser.add_argument("--local_blocks", nargs=3, type=int, default=(2, 2, 2))
    parser.add_argument("--global_blocks", nargs=2, type=int, default=(3, 4))
    parser.add_argument("--deep_blocks", type=int, default=2)
    parser.add_argument("--half_refine_channels", type=int, default=64)
    parser.add_argument(
        "--use_fullres_head", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_factor", type=float, default=0.10)
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--road_weight_cap", type=float, default=2.0)
    parser.add_argument("--main_dice_weight", type=float, default=1.0)
    parser.add_argument("--aux_weight", type=float, default=0.20)
    parser.add_argument("--aux_warmup_epochs", type=int, default=5)
    parser.add_argument("--centerline_pos_weight", type=float, default=8.0)
    parser.add_argument("--centerline_dice_weight", type=float, default=1.0)
    parser.add_argument("--skeleton_iterations", type=int, default=6)

    parser.add_argument("--threshold_min", type=float, default=0.20)
    parser.add_argument("--threshold_max", type=float, default=0.80)
    parser.add_argument("--threshold_step", type=float, default=0.02)
    parser.add_argument("--threshold_bins", type=int, default=1001)
    parser.add_argument("--relaxed_buffer_px", type=int, default=3)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--val_tile_size", type=int, default=1024)
    parser.add_argument("--val_overlap", type=int, default=256)
    parser.add_argument("--val_tile_batch_size", type=int, default=2)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Workers per GPU process; T4x2 therefore uses four workers total",
    )
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--channels_last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_dir", default="./checkpoints/repvgg_roadaux_1024")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.crop_size < 32 or args.crop_size % 32 != 0:
        raise ValueError("crop_size must be >= 32 and divisible by 32")
    if args.val_tile_size < 32 or args.val_tile_size % 32 != 0:
        raise ValueError("val_tile_size must be >= 32 and divisible by 32")
    if not 0 <= args.val_overlap < args.val_tile_size:
        raise ValueError("val_overlap must be in [0, val_tile_size)")
    if args.val_tile_batch_size < 1:
        raise ValueError("val_tile_batch_size must be >= 1")
    if not 0.0 <= args.road_crop_probability <= 1.0:
        raise ValueError("road_crop_probability must be in [0, 1]")
    if not 0.0 <= args.road_crop_min_fraction <= 1.0:
        raise ValueError("road_crop_min_fraction must be in [0, 1]")
    if args.road_crop_tries < 1:
        raise ValueError("road_crop_tries must be >= 1")
    if args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("batch_size and accumulation_steps must be >= 1")
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if not 0.0 <= args.aux_weight:
        raise ValueError("aux_weight must be non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    if not 0.0 <= args.threshold_min < args.threshold_max <= 1.0:
        raise ValueError("threshold range must lie inside [0, 1]")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    (
        args.distributed,
        args.rank,
        args.local_rank,
        args.world_size,
        device,
    ) = init_distributed()

    try:
        configure_dataset_paths(args)
        validate_args(args)
        seed_everything(args.seed + args.rank)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if not args.distributed and torch.cuda.device_count() > 1:
                rank_zero_print(
                    "WARNING: multiple GPUs are visible but DDP is inactive. "
                    "Launch with torchrun --standalone --nproc_per_node=2."
                )
        args.use_amp = bool(args.use_amp and device.type == "cuda")

        (
            train_loader,
            val_loader,
            train_pairs,
            train_sampler,
        ) = make_loaders(args)
        raw_imbalance, road_class_weight = distributed_road_weight(
            train_pairs, args.road_weight_cap, device
        )
        rank_zero_print(
            f"Road imbalance={raw_imbalance:.4f}; "
            f"two-class road weight={road_class_weight:.4f}"
        )

        base_model = RoadRepVGGNet(
            channels=args.channels,
            decoder_channels=args.decoder_channels,
            local_blocks=args.local_blocks,
            global_blocks=args.global_blocks,
            deep_blocks=args.deep_blocks,
            half_refine_channels=args.half_refine_channels,
            enable_fullres_head=args.use_fullres_head,
        ).to(device)
        if args.channels_last:
            base_model = base_model.to(memory_format=torch.channels_last)
        if args.distributed:
            model: nn.Module = DDP(
                base_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        else:
            model = base_model

        # DDP construction first broadcasts rank-0 initialization. Copying EMA
        # afterwards guarantees every rank starts from the identical model.
        ema = ModelEMA(base_model, args.ema_decay)
        criterion = RoadSegCenterlineLoss(
            aux_weight=args.aux_weight,
            main_dice_weight=args.main_dice_weight,
            centerline_dice_weight=args.centerline_dice_weight,
            centerline_pos_weight=args.centerline_pos_weight,
            skeleton_iterations=args.skeleton_iterations,
            road_class_weight=road_class_weight,
        ).to(device)
        optimizer = AdamW(
            parameter_groups(model, args.weight_decay),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        updates_per_epoch = math.ceil(
            len(train_loader) / args.accumulation_steps
        )
        scheduler = build_scheduler(optimizer, updates_per_epoch, args)
        scaler = make_grad_scaler(args.use_amp)

        parameter_count = sum(
            parameter.numel() for parameter in base_model.parameters()
        )
        effective_batch = (
            args.batch_size * args.accumulation_steps * args.world_size
        )
        rank_zero_print("=" * 76)
        rank_zero_print("RepVGG two-stream road training")
        rank_zero_print(
            f"DDP={args.distributed} | GPUs={args.world_size} | "
            f"backend={'NCCL' if args.distributed else 'none'} | "
            f"AMP={args.use_amp} | channels_last={args.channels_last}"
        )
        rank_zero_print(
            f"Native crop={args.crop_size}x{args.crop_size} | "
            f"batch/GPU={args.batch_size} | accumulation={args.accumulation_steps} "
            f"| global effective batch={effective_batch}"
        )
        rank_zero_print(
            f"Road-aware crop probability={args.road_crop_probability:.2f} | "
            f"minimum coarse road fraction={args.road_crop_min_fraction:.4f}"
        )
        rank_zero_print(
            f"Parameters={parameter_count:,} | epochs={args.epochs} "
            f"| LR={args.lr:.2e}"
        )
        rank_zero_print(
            "Main loss=weighted CE + Dice | auxiliary=centerline BCE + Dice"
        )
        rank_zero_print(
            f"Distributed native validation: tile={args.val_tile_size}, "
            f"overlap={args.val_overlap}, tile batch/GPU="
            f"{args.val_tile_batch_size}, Hann blending"
        )
        rank_zero_print("=" * 76)

        save_dir = Path(args.save_dir)
        if is_main_process():
            save_dir.mkdir(parents=True, exist_ok=True)
            with (save_dir / "config.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(vars(args), handle, indent=2, ensure_ascii=False)
        if distributed_active():
            dist.barrier()
        log_path = save_dir / "metrics.jsonl"

        start_epoch = 0
        best_fixed_iou = 0.0
        best_selected_iou = 0.0
        best_threshold = 0.5
        if args.resume:
            (
                start_epoch,
                best_fixed_iou,
                best_selected_iou,
                best_threshold,
            ) = resume_training(
                args.resume,
                base_model,
                ema,
                optimizer,
                scheduler,
                scaler,
                device,
            )
            rank_zero_print(
                f"Resumed from epoch {start_epoch + 1}; "
                f"best fixed IoU={best_fixed_iou:.4f}"
            )

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = train_one_epoch(
                model,
                ema,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                args,
            )
            rank_zero_print(
                f"Epoch {epoch + 1:03d}/{args.epochs}: "
                f"loss={train_metrics['total']:.4f}, "
                f"CE={train_metrics['main_ce']:.4f}, "
                f"Dice={train_metrics['main_dice']:.4f}, "
                f"center={train_metrics['aux']:.4f} "
                f"(w={train_metrics['aux_weight']:.3f}), "
                f"road_fraction={train_metrics['road_fraction']:.4f}, "
                f"lr={train_metrics['lr']:.2e}, "
                f"grad={train_metrics['max_gradient']:.2f}, "
                f"skipped={int(train_metrics['skipped_nonfinite'])}"
            )

            validation: Optional[Dict[str, float]] = None
            validate_now = (
                (epoch + 1) % args.val_interval == 0
                or epoch + 1 == args.epochs
            )
            if validate_now:
                broadcast_module(ema.module, source=0)
                validation = validate(ema.module, val_loader, device, args)
                rank_zero_print(
                    f"  Val EMA fixed@0.50: road IoU="
                    f"{validation['fixed_road_iou']:.4f}, "
                    f"mIoU={validation['fixed_miou']:.4f}, "
                    f"F1={validation['fixed_f1']:.4f}"
                )
                rank_zero_print(
                    f"  Calibrated@{validation['selected_threshold']:.2f}: "
                    f"IoU micro={validation['selected_road_iou_micro']:.4f}, "
                    f"macro={validation['selected_road_iou_macro']:.4f}, "
                    f"P={validation['selected_precision']:.4f}, "
                    f"R={validation['selected_recall']:.4f}"
                )
                rank_zero_print(
                    f"  Relaxed ±{args.relaxed_buffer_px}px: "
                    f"P={validation['relaxed_precision']:.4f}, "
                    f"R={validation['relaxed_recall']:.4f}, "
                    f"F1={validation['relaxed_f1']:.4f}"
                )

                fixed_improved = validation["fixed_road_iou"] > best_fixed_iou
                selected_improved = (
                    validation["selected_road_iou_micro"] > best_selected_iou
                )
                if fixed_improved:
                    best_fixed_iou = validation["fixed_road_iou"]
                    best_threshold = 0.5
                if selected_improved:
                    best_selected_iou = validation[
                        "selected_road_iou_micro"
                    ]

                if is_main_process():
                    state = checkpoint_state(
                        model,
                        ema,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_fixed_iou,
                        best_selected_iou,
                        validation["selected_threshold"],
                        args,
                    )
                    if fixed_improved:
                        atomic_torch_save(
                            state, save_dir / "best_fixed_iou.pt"
                        )
                        print(
                            f"  Saved new best fixed IoU="
                            f"{best_fixed_iou:.4f}"
                        )
                    if selected_improved:
                        atomic_torch_save(
                            state, save_dir / "best_selected_iou.pt"
                        )
                        print(
                            "  Saved new best calibrated IoU="
                            f"{best_selected_iou:.4f}"
                        )

            if is_main_process():
                state = checkpoint_state(
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_fixed_iou,
                    best_selected_iou,
                    best_threshold,
                    args,
                )
                atomic_torch_save(state, save_dir / "last.pt")
                record = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "validation": validation,
                    "best_fixed_iou": best_fixed_iou,
                    "best_selected_iou": best_selected_iou,
                }
                append_jsonl(log_path, record)
            if distributed_active():
                dist.barrier()

        rank_zero_print(
            f"Finished. Best fixed@0.50 IoU={best_fixed_iou:.4f}; "
            f"best calibrated IoU={best_selected_iou:.4f}"
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
