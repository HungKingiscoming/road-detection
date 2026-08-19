"""Official Massachusetts test for RoadRepVGGNet, without TTA.

Place this file beside ``train.py``. The script reconstructs the architecture
from the checkpoint, loads EMA weights by default, performs native-resolution
Hann-blended sliding-window inference, and computes exact metrics over the
official test split. The threshold is fixed before testing; it is never tuned
on test labels.

Recommended T4x2 command::

    torchrun --standalone --nproc_per_node=2 test_repvgg_no_tta.py \
        --checkpoint ./checkpoints/repvgg_roadaux_crop1024_ddp2/best_fixed_iou.pt \
        --threshold 0.50 \
        --tile_size 1024 \
        --overlap 256 \
        --tile_batch_size 2 \
        --output_json ./checkpoints/repvgg_roadaux_crop1024_ddp2/test_fixed050.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import train as project_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official no-TTA test for RoadRepVGGNet"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--weights",
        choices=("ema", "model"),
        default="ema",
        help="EMA matches the model used for validation and checkpoint selection",
    )
    parser.add_argument(
        "--test_image_dir",
        default=(
            "/kaggle/input/datasets/balraj98/"
            "massachusetts-roads-dataset/tiff/test"
        ),
    )
    parser.add_argument(
        "--test_mask_dir",
        default=(
            "/kaggle/input/datasets/balraj98/"
            "massachusetts-roads-dataset/tiff/test_labels"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--tile_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--relaxed_buffer_px", type=int, default=3)
    parser.add_argument("--print_per_image", action="store_true")
    parser.add_argument(
        "--use_amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--channels_last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output_json", default="./test_no_tta_metrics.json")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.tile_size < 32 or args.tile_size % 32 != 0:
        raise ValueError("tile_size must be >= 32 and divisible by 32")
    if not 0 <= args.overlap < args.tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if args.tile_batch_size < 1:
        raise ValueError("tile_batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1")
    if args.relaxed_buffer_px < 0:
        raise ValueError("relaxed_buffer_px must be non-negative")


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_file():
        return checkpoint_path.resolve()
    if checkpoint_path.is_dir():
        for name in ("best_fixed_iou.pt", "best_selected_iou.pt", "last.pt"):
            candidate = checkpoint_path / name
            if candidate.is_file():
                if project_train.is_main_process():
                    print(f"Resolved checkpoint directory to: {candidate}")
                return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_checkpoint(path: str | Path) -> Tuple[Path, Dict]:
    checkpoint_path = resolve_checkpoint_path(path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    return checkpoint_path, checkpoint


def remove_ddp_prefix(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return state


def select_state_dict(
    checkpoint: Dict,
    requested_source: str,
) -> Tuple[Dict[str, Tensor], str]:
    source = requested_source
    if source == "ema" and "ema" not in checkpoint:
        source = "model"
        if project_train.is_main_process():
            print("WARNING: checkpoint has no EMA state; falling back to model")

    if source in checkpoint:
        state = checkpoint[source]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        source = "state_dict"
    else:
        state = checkpoint
        source = "checkpoint"
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint entry '{source}' is not a state_dict")
    return remove_ddp_prefix(state), source


def build_model(
    checkpoint: Dict,
    state: Dict[str, Tensor],
    device: torch.device,
    channels_last: bool,
) -> torch.nn.Module:
    saved = dict(checkpoint.get("args", {}))
    has_fullres_weights = any(
        "decode_head.fullres_head." in str(key) for key in state
    )
    model = project_train.RoadRepVGGNet(
        channels=int(saved.get("channels", 40)),
        decoder_channels=int(saved.get("decoder_channels", 128)),
        local_blocks=tuple(saved.get("local_blocks", (2, 2, 2))),
        global_blocks=tuple(saved.get("global_blocks", (3, 4))),
        deep_blocks=int(saved.get("deep_blocks", 2)),
        half_refine_channels=int(saved.get("half_refine_channels", 64)),
        enable_fullres_head=bool(
            saved.get("use_fullres_head", has_fullres_weights)
            or has_fullres_weights
        ),
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint/model mismatch. Ensure modeling/model.py and train.py "
            "are the exact RepVGG versions used during training."
        ) from error
    return model.eval()


def confusion_counts(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction.bool()
    target = target.bool()
    return torch.stack(
        (
            (prediction & target).sum(),
            (prediction & ~target).sum(),
            (~prediction & target).sum(),
            (~prediction & ~target).sum(),
        )
    ).to(dtype=torch.int64)


def relaxed_counts(prediction: Tensor, target: Tensor, radius: int) -> Tensor:
    prediction = prediction.bool()
    target = target.bool()
    if radius > 0:
        kernel_size = 2 * radius + 1
        dilated_target = F.max_pool2d(
            target[:, None].float(),
            kernel_size,
            stride=1,
            padding=radius,
        )[:, 0].bool()
        dilated_prediction = F.max_pool2d(
            prediction[:, None].float(),
            kernel_size,
            stride=1,
            padding=radius,
        )[:, 0].bool()
    else:
        dilated_target = target
        dilated_prediction = prediction
    return torch.stack(
        (
            (prediction & dilated_target).sum(),
            prediction.sum(),
            (target & dilated_prediction).sum(),
            target.sum(),
        )
    ).to(dtype=torch.int64)


def reduce_sum(tensor: Tensor) -> Tensor:
    if project_train.distributed_active():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def gather_per_image(
    local_items: List[Dict[str, float | str]],
    world_size: int,
) -> List[Dict[str, float | str]]:
    if not project_train.distributed_active():
        return local_items
    gathered: List[List[Dict[str, float | str]] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered, local_items)
    return [item for shard in gathered if shard is not None for item in shard]


def metrics_from_totals(
    totals: Tensor,
    relaxed: Tensor,
    per_image: List[Dict[str, float | str]],
) -> Dict[str, float]:
    tp, fp, fn, tn = [int(value) for value in totals.cpu().tolist()]
    prediction_hits, prediction_count, target_hits, target_count = [
        int(value) for value in relaxed.cpu().tolist()
    ]
    road_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    relaxed_precision = prediction_hits / max(prediction_count, 1)
    relaxed_recall = target_hits / max(target_count, 1)
    relaxed_f1 = (
        2.0
        * relaxed_precision
        * relaxed_recall
        / max(relaxed_precision + relaxed_recall, 1e-12)
    )
    macro_iou = float(
        np.mean([float(item["road_iou"]) for item in per_image])
    )
    return {
        "road_iou_micro": road_iou,
        "road_iou_macro": macro_iou,
        "background_iou": background_iou,
        "miou": 0.5 * (road_iou + background_iou),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": relaxed_f1,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    (
        args.distributed,
        args.rank,
        args.local_rank,
        args.world_size,
        device,
    ) = project_train.init_distributed()

    try:
        args.use_amp = bool(args.use_amp and device.type == "cuda")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        checkpoint_path, checkpoint = load_checkpoint(args.checkpoint)
        state, state_source = select_state_dict(checkpoint, args.weights)
        model = build_model(
            checkpoint,
            state,
            device,
            channels_last=args.channels_last,
        )

        pairs = project_train.build_pairs(
            args.test_image_dir,
            args.test_mask_dir,
        )
        dataset = project_train.RoadNativeValidationDataset(pairs)
        sampler = (
            project_train.DistributedEvalSampler(
                dataset,
                rank=args.rank,
                world_size=args.world_size,
            )
            if args.distributed
            else None
        )
        generator = torch.Generator()
        generator.manual_seed(42 + args.rank)
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "sampler": sampler,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.num_workers > 0,
            "worker_init_fn": project_train.seed_worker,
            "generator": generator,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader = DataLoader(dataset, **loader_kwargs)

        # Reuse the exact native sliding-window implementation used by
        # validation in train.py.
        args.val_tile_size = args.tile_size
        args.val_overlap = args.overlap
        args.val_tile_batch_size = args.tile_batch_size

        if project_train.is_main_process():
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            print("=" * 76)
            print("OFFICIAL TEST — NATIVE SLIDING WINDOW, NO TTA")
            print(
                f"Images={len(pairs)} | workers={args.world_size} GPU process(es)"
            )
            print(
                f"Tile={args.tile_size} | overlap={args.overlap} | "
                f"tile batch/GPU={args.tile_batch_size}"
            )
            print(
                f"Threshold locked at {args.threshold:.2f} | "
                f"weights={state_source} | AMP={args.use_amp}"
            )
            print(
                f"Checkpoint epoch={int(checkpoint.get('epoch', -1)) + 1} | "
                f"parameters={parameter_count:,}"
            )
            print("=" * 76)

        totals = torch.zeros(4, dtype=torch.int64, device=device)
        relaxed_totals = torch.zeros(4, dtype=torch.int64, device=device)
        local_per_image: List[Dict[str, float | str]] = []
        progress = tqdm(
            loader,
            desc="Test no-TTA",
            leave=False,
            disable=not project_train.is_main_process(),
        )

        with torch.inference_mode():
            for images, masks, names in progress:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = project_train.sliding_window_logits(
                    model,
                    images,
                    device,
                    args,
                )
                probability = torch.softmax(logits.float(), dim=1)[:, 1]
                prediction = probability >= args.threshold
                target = masks > 0

                counts = confusion_counts(prediction, target)
                totals.add_(counts)
                relaxed_totals.add_(
                    relaxed_counts(
                        prediction,
                        target,
                        args.relaxed_buffer_px,
                    )
                )
                tp, fp, fn = [int(value) for value in counts[:3].tolist()]
                image_iou = tp / max(tp + fp + fn, 1)
                local_per_image.append(
                    {"name": str(names[0]), "road_iou": image_iou}
                )
                progress.set_postfix(iou=f"{image_iou:.4f}")

        reduce_sum(totals)
        reduce_sum(relaxed_totals)
        per_image = gather_per_image(local_per_image, args.world_size)
        per_image.sort(key=lambda item: str(item["name"]))

        if project_train.is_main_process():
            if len(per_image) != len(pairs):
                raise RuntimeError(
                    f"Distributed test coverage mismatch: expected {len(pairs)}, "
                    f"received {len(per_image)}"
                )
            metric_values = metrics_from_totals(
                totals,
                relaxed_totals,
                per_image,
            )
            tp, fp, fn, tn = [int(value) for value in totals.cpu().tolist()]
            metrics = {
                "protocol": (
                    "single-pass native sliding-window; Hann blending; "
                    "no TTA; no test threshold search"
                ),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
                "weights": state_source,
                "images": len(per_image),
                "threshold": args.threshold,
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "relaxed_buffer_px": args.relaxed_buffer_px,
                "confusion_counts": {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                },
                **metric_values,
                "per_image": per_image,
            }

            print("\nTest results (NO TTA)")
            print(
                f"  road IoU micro : {metrics['road_iou_micro']:.6f}"
            )
            print(
                f"  road IoU macro : {metrics['road_iou_macro']:.6f}"
            )
            print(
                f"  background IoU : {metrics['background_iou']:.6f}"
            )
            print(f"  mIoU            : {metrics['miou']:.6f}")
            print(f"  precision       : {metrics['precision']:.6f}")
            print(f"  recall          : {metrics['recall']:.6f}")
            print(f"  F1              : {metrics['f1']:.6f}")
            print(f"  accuracy        : {metrics['accuracy']:.6f}")
            print(
                f"  relaxed F1 ±{args.relaxed_buffer_px}px: "
                f"{metrics['relaxed_f1']:.6f} (diagnostic only)"
            )

            if args.print_per_image:
                print("\nPer-image road IoU (worst to best)")
                for item in sorted(
                    per_image,
                    key=lambda value: float(value["road_iou"]),
                ):
                    print(
                        f"  {str(item['name']):28s} "
                        f"{float(item['road_iou']):.6f}"
                    )

            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2, ensure_ascii=False)
            print(f"Saved metrics to: {output_path.resolve()}")

        if project_train.distributed_active():
            dist.barrier()
    finally:
        project_train.cleanup_distributed()


if __name__ == "__main__":
    main()
