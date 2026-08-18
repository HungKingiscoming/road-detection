"""Offline shape/contract test for the ResNet road teacher."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from modeling.resnet_teacher import build_resnet_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_arch", choices=("resnet34", "resnet50"), default="resnet50")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    namespace = SimpleNamespace(
        teacher_arch=args.teacher_arch,
        teacher_decoder_channels=256,
        teacher_pretrained=False,
        teacher_freeze_backbone_bn=True,
        aux_weight=0.2,
        dropout=0.1,
    )
    model = build_resnet_teacher(namespace)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    image = torch.randn(1, 3, args.size, args.size)

    model.train()
    aux, centerline, logits = model(image)
    assert centerline is None
    assert aux is not None and aux.shape[:2] == (1, 2)
    assert logits.shape == (1, 2, args.size, args.size)
    backbone_bn_training = [
        module.training
        for module in model.backbone.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]
    assert backbone_bn_training and not any(backbone_bn_training)

    model.eval()
    with torch.no_grad():
        eval_logits = model(image)
    assert eval_logits.shape == logits.shape
    assert torch.isfinite(eval_logits).all()
    print(
        f"PASS: {args.teacher_arch}, parameters={parameters:,}, "
        f"train_logits={tuple(logits.shape)}, aux={tuple(aux.shape)}, "
        f"eval_logits={tuple(eval_logits.shape)}"
    )


if __name__ == "__main__":
    main()
