"""Distill a validated CRNet checkpoint into CoMingNetAccuracy.

This script reuses the road/boundary-aware KD implementation from
train_resnet_distill.py and replaces only the teacher reconstruction logic.
Do not run it unless CRNet beats the student on the same validation protocol.
"""

from __future__ import annotations

from argparse import Namespace

import train as project_train
import train_resnet_distill as kd_pipeline

from modeling.crnet_teacher import build_crnet_teacher


def _build_crnet_from_checkpoint(args, _student, device):
    path = project_train.resolve_checkpoint_path(args.teacher_checkpoint)
    checkpoint = project_train.safe_torch_load(path)
    if not isinstance(checkpoint, dict):
        raise TypeError("CRNet teacher checkpoint must be a dictionary")
    saved_args = dict(checkpoint.get("args", {}))
    role = saved_args.get("model_role")
    if role != "crnet_teacher":
        raise ValueError(
            f"Checkpoint model_role={role!r}; expected 'crnet_teacher'."
        )
    teacher_crop = int(saved_args.get("crop_size", 512))
    student_crop = int(args.crop_size)
    if teacher_crop != student_crop:
        raise ValueError(
            "Teacher and student tiles must be spatially aligned for dense KD: "
            f"teacher crop={teacher_crop}, student crop={student_crop}."
        )
    saved_args.setdefault("crnet_freeze_backbone_bn", True)
    teacher = build_crnet_teacher(Namespace(**saved_args), pretrained=False)
    teacher.load_state_dict(kd_pipeline._extract_state(checkpoint), strict=True)
    teacher.to(device).eval()
    teacher.requires_grad_(False)
    parameters = sum(parameter.numel() for parameter in teacher.parameters())
    print(
        "CRNet teacher loaded: "
        f"crop={teacher_crop}, params={parameters:,}, "
        f"checkpoint_best_iou={checkpoint.get('best_road_iou', 'unknown')}"
    )
    return teacher


# train_resnet_distill already installed the student factory, custom KD parser,
# road-aware Bernoulli KL and KD warm-up into this same project_train module.
project_train.build_teacher = _build_crnet_from_checkpoint


if __name__ == "__main__":
    project_train.main()
