"""Distill a trained ResNet road teacher into CoMingNetAccuracy.

This wrapper replaces the original same-architecture deepcopy teacher with a
separately reconstructed ResNet teacher.  It also replaces generic multi-class
KL with road-aware binary dense distillation and a short KD-weight warm-up.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F

import train as project_train

from modeling.accuracy_model import build_accuracy_model
from modeling.resnet_teacher import build_resnet_teacher


_original_parse_args = project_train.parse_args
_original_train_one_epoch = project_train.train_one_epoch
_kd_road_weight = 2.0
_kd_boundary_weight = 1.0
_kd_warmup_epochs = 3


def _parse_args():
    global _kd_road_weight, _kd_boundary_weight, _kd_warmup_epochs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kd_road_weight", type=float, default=2.0)
    parser.add_argument("--kd_boundary_weight", type=float, default=1.0)
    parser.add_argument("--kd_warmup_epochs", type=int, default=3)
    parser.add_argument("--fullres_channels", type=int, default=24)
    custom, remaining = parser.parse_known_args()
    if custom.kd_road_weight < 0 or custom.kd_boundary_weight < 0:
        raise ValueError("KD road/boundary weights must be non-negative")
    if custom.kd_warmup_epochs < 0:
        raise ValueError("kd_warmup_epochs must be non-negative")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = _original_parse_args()
    finally:
        sys.argv = old_argv

    for key, value in vars(custom).items():
        setattr(args, key, value)
    args.model_role = "coming_accuracy_student"
    _kd_road_weight = custom.kd_road_weight
    _kd_boundary_weight = custom.kd_boundary_weight
    _kd_warmup_epochs = custom.kd_warmup_epochs
    if not args.teacher_checkpoint:
        raise ValueError("--teacher_checkpoint is required for distillation")
    return args


def _extract_state(checkpoint):
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Teacher checkpoint does not contain a state_dict")
    return {str(key).removeprefix("module."): value for key, value in state.items()}


def _build_strong_teacher(args, _student, device):
    path = project_train.resolve_checkpoint_path(args.teacher_checkpoint)
    checkpoint = project_train.safe_torch_load(path)
    if not isinstance(checkpoint, dict):
        raise TypeError("Teacher checkpoint must be a dictionary")
    saved_args = dict(checkpoint.get("args", {}))
    role = saved_args.get("model_role")
    if role not in (None, "resnet_teacher"):
        raise ValueError(
            f"Checkpoint model_role={role!r}, expected a ResNet teacher checkpoint"
        )
    saved_args.setdefault("teacher_arch", "resnet50")
    saved_args.setdefault("teacher_decoder_channels", 256)
    saved_args.setdefault("teacher_freeze_backbone_bn", True)
    saved_args.setdefault("aux_weight", 0.0)
    saved_args.setdefault("dropout", 0.10)
    teacher_args = Namespace(**saved_args)
    teacher = build_resnet_teacher(teacher_args, pretrained=False)
    teacher.load_state_dict(_extract_state(checkpoint), strict=True)
    teacher.to(device).eval()
    teacher.requires_grad_(False)

    parameters = sum(parameter.numel() for parameter in teacher.parameters())
    best_iou = checkpoint.get("best_road_iou", "unknown")
    print(
        "Strong teacher loaded: "
        f"arch={teacher_args.teacher_arch}, params={parameters:,}, "
        f"checkpoint_best_iou={best_iou}"
    )
    return teacher


def _soft_boundary(probability: torch.Tensor) -> torch.Tensor:
    maximum = F.max_pool2d(probability, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-probability, 3, stride=1, padding=1)
    return (maximum - minimum).clamp_(0.0, 1.0)


def _road_aware_distillation_loss(student, teacher, temperature):
    """Dense binary KD emphasizing teacher road and boundary pixels.

    The project trains the foreground log-odds (class-1 minus class-0), so the
    same representation is distilled.  This avoids wasting capacity matching
    an arbitrary common offset between two two-channel classifiers.
    """

    if student.shape[1] != 2 or teacher.shape[1] != 2:
        raise ValueError("Road-aware KD expects two-channel logits")
    student_road = student[:, 1:2].float() - student[:, 0:1].float()
    teacher_road = teacher[:, 1:2].float() - teacher[:, 0:1].float()
    teacher_probability = torch.sigmoid(teacher_road / temperature)
    # Bernoulli KL rather than soft-label BCE: both have the same gradient,
    # but KL is zero when student and teacher match and is easier to monitor.
    eps = 1e-6
    teacher_probability = teacher_probability.clamp(eps, 1.0 - eps)
    teacher_background = 1.0 - teacher_probability
    student_log_road = F.logsigmoid(student_road / temperature)
    student_log_background = F.logsigmoid(-student_road / temperature)
    pixel_loss = (
        teacher_probability
        * (teacher_probability.log() - student_log_road)
        + teacher_background
        * (teacher_background.log() - student_log_background)
    )
    with torch.no_grad():
        road_emphasis = teacher_probability
        boundary_emphasis = _soft_boundary(torch.sigmoid(teacher_road))
        weight = (
            1.0
            + _kd_road_weight * road_emphasis
            + _kd_boundary_weight * boundary_emphasis
        )
    return (pixel_loss * weight).sum() / weight.sum().clamp_min(1.0) * temperature**2


def _train_one_epoch_with_kd_warmup(
    model, teacher, loader, criterion, optimizer, scheduler, scaler,
    device, epoch, args,
):
    configured_weight = float(args.kd_weight)
    if _kd_warmup_epochs > 0:
        factor = min(1.0, float(epoch + 1) / float(_kd_warmup_epochs))
    else:
        factor = 1.0
    args.kd_weight = configured_weight * factor
    print(
        f"  KD: weight={args.kd_weight:.4f}/{configured_weight:.4f}, "
        f"T={args.kd_temperature:.2f}, road_w={_kd_road_weight:.2f}, "
        f"boundary_w={_kd_boundary_weight:.2f}"
    )
    try:
        metrics = _original_train_one_epoch(
            model, teacher, loader, criterion, optimizer, scheduler, scaler,
            device, epoch, args,
        )
    finally:
        args.kd_weight = configured_weight
    print(f"  KD raw loss: {metrics['kd']:.6f}")
    return metrics


project_train.parse_args = _parse_args
project_train.build_model = build_accuracy_model
project_train.build_teacher = _build_strong_teacher
project_train.distillation_loss = _road_aware_distillation_loss
project_train.train_one_epoch = _train_one_epoch_with_kd_warmup


if __name__ == "__main__":
    project_train.main()
