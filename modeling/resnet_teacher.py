"""Strong, convolution-only ResNet teacher for binary road segmentation.

The teacher is intentionally independent from CoMingNet.  It combines an
ImageNet-pretrained ResNet encoder, multi-dilation context, top-down skip
decoding, and a shallow full-resolution RGB refinement path.  Training follows
the same return contract as the project GCNetHead; evaluation returns logits.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _norm(channels: int) -> nn.GroupNorm:
    """Batch-size-independent normalization for the newly trained decoder."""

    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            _norm(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ResidualRefine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3),
            ConvNormAct(channels, channels, 3, activation=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class ResNetEncoder(nn.Module):
    """Expose a torchvision ResNet as a five-scale segmentation encoder."""

    def __init__(self, architecture: str, pretrained: bool) -> None:
        super().__init__()
        try:
            from torchvision.models import (
                ResNet34_Weights,
                ResNet50_Weights,
                resnet34,
                resnet50,
            )
        except ImportError as error:
            raise ImportError(
                "The ResNet teacher requires torchvision. Install a torchvision "
                "version compatible with the installed PyTorch build."
            ) from error

        architecture = architecture.lower()
        if architecture == "resnet34":
            # ResNet34 currently has a single official ImageNet-1K recipe.
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            network = resnet34(weights=weights)
            self.channels = (64, 64, 128, 256, 512)
        elif architecture == "resnet50":
            # Pin V2 instead of DEFAULT so checkpoint creation is reproducible
            # if TorchVision changes the DEFAULT alias in a future release.
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            network = resnet50(weights=weights)
            self.channels = (64, 256, 512, 1024, 2048)
        else:
            raise ValueError("teacher_arch must be 'resnet34' or 'resnet50'")

        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu)
        self.pool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        s2 = self.stem(x)
        s4 = self.layer1(self.pool(s2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)
        return {"s2": s2, "s4": s4, "s8": s8, "s16": s16, "s32": s32}


class MultiDilationContext(nn.Module):
    """D-Link/ASPP-style context without attention or a Transformer."""

    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, channels, 1)
        self.branches = nn.ModuleList(
            [
                ConvNormAct(channels, channels // 2, 3, dilation=dilation)
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.global_projection = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        merged_channels = channels + 5 * (channels // 2)
        self.fuse = ConvNormAct(merged_channels, channels, 1)
        self.refine = ResidualRefine(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project(x)
        branches = [branch(x) for branch in self.branches]
        global_context = self.global_projection(F.adaptive_avg_pool2d(x, 1))
        global_context = F.interpolate(
            global_context, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(self.fuse(torch.cat([x, *branches, global_context], dim=1)))


class DecoderStage(nn.Module):
    """Concatenative top-down decoder stage; accuracy is prioritized here."""

    def __init__(self, deep_channels: int, skip_channels: int, channels: int) -> None:
        super().__init__()
        self.deep_projection = ConvNormAct(deep_channels, channels, 1)
        self.skip_projection = ConvNormAct(skip_channels, channels, 1)
        self.fuse = ConvNormAct(channels * 2, channels, 3)
        self.refine = ResidualRefine(channels)

    def forward(self, deep: Tensor, skip: Tensor) -> Tensor:
        deep = F.interpolate(
            self.deep_projection(deep),
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        skip = self.skip_projection(skip)
        return self.refine(self.fuse(torch.cat([deep, skip], dim=1)))


class ResNetRoadHead(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int = 256,
        num_classes: int = 2,
        dropout: float = 0.10,
        enable_aux: bool = True,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 5:
            raise ValueError("encoder_channels must describe s2, s4, s8, s16, s32")
        c2, c4, c8, c16, c32 = encoder_channels
        d = decoder_channels
        self.enable_aux = bool(enable_aux)

        self.context = MultiDilationContext(c32, d)
        self.decode16 = DecoderStage(d, c16, d)
        self.decode8 = DecoderStage(d, c8, d // 2)
        self.decode4 = DecoderStage(d // 2, c4, d // 2)
        self.decode2 = DecoderStage(d // 2, c2, d // 4)

        full_channels = max(d // 8, 32)
        self.up_to_full = nn.Sequential(
            nn.ConvTranspose2d(
                d // 4, full_channels, kernel_size=4, stride=2,
                padding=1, bias=False,
            ),
            _norm(full_channels),
            nn.ReLU(inplace=True),
        )
        # Direct RGB detail is useful for thin road boundaries that have been
        # attenuated by the stride-2 encoder stem.
        self.rgb_detail = nn.Sequential(
            ConvNormAct(3, full_channels, 3),
            ConvNormAct(full_channels, full_channels, 3),
        )
        self.full_refine = nn.Sequential(
            ConvNormAct(full_channels * 2, full_channels, 3),
            ResidualRefine(full_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Conv2d(full_channels, num_classes, 1)
        self.aux_classifier = nn.Sequential(
            ConvNormAct(d // 2, d // 4, 3),
            nn.Conv2d(d // 4, num_classes, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for classifier in (self.classifier, self.aux_classifier[-1]):
            nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)

    def forward(
        self, features: Dict[str, Tensor], image: Tensor
    ) -> Union[Tensor, Tuple[Optional[Tensor], Optional[Tensor], Tensor]]:
        d32 = self.context(features["s32"])
        d16 = self.decode16(d32, features["s16"])
        d8 = self.decode8(d16, features["s8"])
        d4 = self.decode4(d8, features["s4"])
        d2 = self.decode2(d4, features["s2"])
        full = self.up_to_full(d2)
        if full.shape[-2:] != image.shape[-2:]:
            full = F.interpolate(
                full, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        detail = self.rgb_detail(image)
        logits = self.classifier(self.full_refine(torch.cat([full, detail], dim=1)))

        if self.training:
            aux = self.aux_classifier(d8) if self.enable_aux else None
            return aux, None, logits
        return logits


class ResNetRoadTeacher(nn.Module):
    def __init__(
        self,
        architecture: str = "resnet50",
        decoder_channels: int = 256,
        pretrained: bool = True,
        freeze_backbone_bn: bool = True,
        enable_aux: bool = True,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = ResNetEncoder(architecture, pretrained=pretrained)
        self.decode_head = ResNetRoadHead(
            self.backbone.channels,
            decoder_channels=decoder_channels,
            enable_aux=enable_aux,
            dropout=dropout,
        )
        self.freeze_backbone_bn = bool(freeze_backbone_bn)

    def _freeze_bn_statistics(self) -> None:
        if not self.freeze_backbone_bn:
            return
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True) -> "ResNetRoadTeacher":
        super().train(mode)
        if mode:
            self._freeze_bn_statistics()
        return self

    def forward(self, image: Tensor):
        return self.decode_head(self.backbone(image), image)

    def switch_to_deploy(self) -> "ResNetRoadTeacher":
        return self


def build_resnet_teacher(args, *, pretrained: Optional[bool] = None) -> ResNetRoadTeacher:
    """Factory compatible with the project's train.py Namespace."""

    if pretrained is None:
        pretrained = bool(getattr(args, "teacher_pretrained", True))
    model = ResNetRoadTeacher(
        architecture=str(getattr(args, "teacher_arch", "resnet50")),
        decoder_channels=int(getattr(args, "teacher_decoder_channels", 256)),
        pretrained=pretrained,
        freeze_backbone_bn=bool(getattr(args, "teacher_freeze_backbone_bn", True)),
        enable_aux=float(getattr(args, "aux_weight", 0.0)) > 0,
        dropout=float(getattr(args, "dropout", 0.10)),
    )
    return model


__all__ = [
    "ResNetEncoder",
    "ResNetRoadHead",
    "ResNetRoadTeacher",
    "build_resnet_teacher",
]
