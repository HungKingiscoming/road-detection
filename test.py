"""Evaluate a CoMingNet checkpoint on the official test split without TTA.

The script performs one deterministic sliding-window pass per image.  It does
not search for a threshold on the test set: ``--threshold`` must be fixed in
advance (normally 0.50, or a value selected only on validation data).

Place this file beside ``train.py`` in the road-detection project.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import train as project_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
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
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--tile_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--relaxed_buffer_px", type=int, default=3)
    parser.add_argument("--print_per_image", action="store_true")
    parser.add_argument(
        "--use_amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--output_json", default="./test_no_tta_metrics.json"
    )
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> Dict:
    path = project_train.resolve_checkpoint_path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    return checkpoint


def architecture_args(checkpoint: Dict) -> Namespace:
    saved = dict(checkpoint.get("args", {}))
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Could not find a model state_dict in checkpoint")

    defaults = {
        "model_variant": "coming",
        "channels": 40,
        "decoder_channels": 96,
        "local_blocks": (2, 2, 2),
        "global_blocks": (3, 4),
        "highres_kernel_size": 5,
        "coming_kernel_size": 7,
        "local_expansion": 1.5,
        "global_expansion": 2.0,
        "local_spatial_ratio": 0.50,
        "global_spatial_ratio": 0.50,
        "use_half_refine": True,
        "half_refine_channels": 48,
        "use_fullres_head": False,
        "fullres_channels": 16,
        "dropout": 0.05,
        "aux_weight": 0.0,
        "centerline_weight": 0.0,
    }
    for key, value in defaults.items():
        saved.setdefault(key, value)

    # Older checkpoints predate these config keys. Detect a full-resolution
    # head from its state_dict so such checkpoints still reconstruct exactly.
    has_fullres_weights = any(
        "decode_head.fullres_head." in str(key) for key in state
    )
    if has_fullres_weights:
        saved["use_fullres_head"] = True

    return Namespace(**saved)


def update_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    totals: Dict[str, int],
) -> float:
    prediction = prediction.bool()
    target = target.bool()
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    totals["tp"] += tp
    totals["fp"] += fp
    totals["fn"] += fn
    totals["tn"] += tn
    return tp / max(tp + fp + fn, 1)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.use_amp = bool(args.use_amp and device.type == "cuda")
    checkpoint = load_checkpoint(args.checkpoint)
    model_args = architecture_args(checkpoint)
    model = project_train.build_model(model_args).to(device)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint/model mismatch. Ensure modeling/decoder.py is the "
            "same decoder version used to create this checkpoint."
        ) from error
    model.eval()

    pairs = project_train.build_pairs(
        args.test_image_dir, args.test_mask_dir
    )
    dataset = project_train.RoadDataset(
        pairs,
        crop_size=args.tile_size,
        training=False,
        full_image=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_image: List[Dict[str, float | str]] = []
    relaxed_prediction_hits = 0.0
    relaxed_target_hits = 0.0
    relaxed_prediction_count = 0.0
    relaxed_target_count = 0.0

    print("=" * 72)
    print("OFFICIAL TEST â€” SINGLE-PASS SLIDING WINDOW, NO TTA")
    print(f"Images: {len(pairs)}")
    print(
        f"Tile: {args.tile_size} | overlap: {args.overlap} | "
        f"threshold locked at {args.threshold:.2f}"
    )
    print(f"Checkpoint epoch: {int(checkpoint.get('epoch', -1)) + 1}")
    print("=" * 72)

    progress = tqdm(loader, desc="Test no-TTA")
    for index, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = project_train.sliding_window_inference(
            model=model,
            image=images,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_batch_size=args.tile_batch_size,
            device=device,
            use_amp=args.use_amp,
        )
        probability = torch.sigmoid(project_train.road_logit(logits))[:, 0]
        prediction = probability >= args.threshold
        target = masks > 0
        image_iou = update_counts(prediction, target, totals)
        relaxed = project_train.relaxed_overlap_components(
            prediction, target, args.relaxed_buffer_px
        )
        relaxed_prediction_hits += relaxed[0]
        relaxed_target_hits += relaxed[1]
        relaxed_prediction_count += relaxed[2]
        relaxed_target_count += relaxed[3]
        per_image.append({"name": pairs[index][0].name, "road_iou": image_iou})
        progress.set_postfix(iou=f"{image_iou:.4f}")

    tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
    road_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    relaxed_precision = relaxed_prediction_hits / max(
        relaxed_prediction_count, 1.0
    )
    relaxed_recall = relaxed_target_hits / max(relaxed_target_count, 1.0)
    relaxed_f1 = 2.0 * relaxed_precision * relaxed_recall / max(
        relaxed_precision + relaxed_recall, 1e-12
    )
    macro_iou = float(np.mean([float(item["road_iou"]) for item in per_image]))

    metrics = {
        "protocol": "single-pass sliding-window; no TTA; no test threshold search",
        "images": len(pairs),
        "threshold": args.threshold,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
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
        "per_image": per_image,
    }

    print("\nTest results (NO TTA)")
    print(f"  road IoU micro : {road_iou:.6f}")
    print(f"  road IoU macro : {macro_iou:.6f}")
    print(f"  background IoU : {background_iou:.6f}")
    print(f"  mIoU            : {metrics['miou']:.6f}")
    print(f"  precision       : {precision:.6f}")
    print(f"  recall          : {recall:.6f}")
    print(f"  F1              : {f1:.6f}")
    print(f"  accuracy        : {metrics['accuracy']:.6f}")
    print(
        f"  relaxed F1 Â±{args.relaxed_buffer_px}px: {relaxed_f1:.6f} "
        "(diagnostic only)"
    )

    if args.print_per_image:
        print("\nPer-image road IoU")
        for item in sorted(per_image, key=lambda value: float(value["road_iou"])):
            print(f"  {str(item['name']):28s} {float(item['road_iou']):.6f}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
