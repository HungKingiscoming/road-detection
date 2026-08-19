"""Shape, backward and checkpoint-contract test for corrected CRNet."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from modeling.crnet_teacher import build_crnet_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    namespace = SimpleNamespace(
        crop_size=args.crop_size,
        crnet_pretrained=False,
        crnet_freeze_backbone_bn=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_crnet_teacher(namespace).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    image = torch.randn(1, 3, args.crop_size, args.crop_size, device=device)

    model.train()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        aux, centerline, logits = model(image)
    assert aux is None and centerline is None
    assert logits.shape == (1, 2, args.crop_size, args.crop_size)
    road_logit = logits[:, 1:2] - logits[:, 0:1]
    if args.backward:
        target = torch.zeros_like(road_logit)
        loss = F.binary_cross_entropy_with_logits(road_logit, target)
        loss.backward()
        assert model.final[-1].weight.grad is not None

    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        eval_logits = model(image)
    assert eval_logits.shape == logits.shape
    assert torch.isfinite(eval_logits).all()
    print(
        f"PASS: CRNet crop={args.crop_size}, device={device}, params={parameters:,}, "
        f"logits={tuple(eval_logits.shape)}, backward={args.backward}"
    )


if __name__ == "__main__":
    main()
