"""CNN-only multi-scale decoder for CoMingNet road extraction.

The public ``GCNetHead`` name and its training output contract are preserved.
The head consumes s32/s16/s8/s4/s2 features, refines every skip with ordinary
convolutions, and learns the final H/2-to-H reconstruction.  No attention or
Transformer operation is used.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import CoMingBlock, ConvBNAct


class ConvRelationFusion(nn.Module):
    """Filter a shallow skip using the corresponding deeper representation."""

    def __init__(
        self,
        deep_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int,
        expansion: float,
        spatial_ratio: float,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.deep_proj = ConvBNAct(deep_channels, out_channels, 1, padding=0)
        self.skip_proj = ConvBNAct(
            skip_channels, out_channels, 1, padding=0, activation=False
        )
        self.relation = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, 3),
            ConvBNAct(out_channels, out_channels, 3, activation=False),
        )
        self.activation = nn.ReLU(inplace=True)
        self.refine = CoMingBlock(
            out_channels,
            kernel_size=kernel_size,
            expansion=expansion,
            spatial_ratio=spatial_ratio,
            deploy=deploy,
            zero_init_residual=True,
        )

    def forward(self, deep: Tensor, skip: Tensor) -> Tensor:
        deep = F.interpolate(
            self.deep_proj(deep),
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        skip = self.skip_proj(skip)
        relation = self.relation(torch.cat((deep, skip), dim=1))
        return self.refine(self.activation(deep + relation))


class FullResolutionHead(nn.Module):
    """Learned output-stride-2 to full-resolution road reconstruction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if hidden_channels < 8:
            raise ValueError("fullres_channels must be >= 8")
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, 3),
            ConvBNAct(hidden_channels, hidden_channels, 3),
        )
        self.correction = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, feature: Tensor, coarse_logits: Tensor) -> Tensor:
        full_feature = self.refine(self.up(feature))
        coarse_full = F.interpolate(
            coarse_logits,
            size=full_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return coarse_full + self.correction(full_feature)


class GCNetHead(nn.Module):
    """Compatibility name retained for the existing training project.

    During training returns ``(aux_logits_s8, centerline_logits_s4,
    main_logits_fullres)``.  At evaluation it returns full-resolution logits.
    """

    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 128,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (128, 128, 128),
        stem_channels: int = 40,
        dropout_ratio: float = 0.05,
        highres_kernel_size: int = 5,
        context_kernel_size: int = 7,
        local_expansion: float = 1.5,
        global_expansion: float = 2.0,
        local_spatial_ratio: float = 0.5,
        global_spatial_ratio: float = 0.5,
        enable_seg_aux: bool = False,
        enable_centerline_aux: bool = False,
        enable_half_refine: bool = True,
        half_refine_channels: int = 64,
        enable_fullres_head: bool = True,
        fullres_channels: int = 24,
        deploy: bool = False,
        align_corners: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        del in_channels, enable_half_refine, enable_fullres_head, align_corners
        if len(feature_channels) != 3:
            raise ValueError("feature_channels must describe s4, s8 and s16")
        if half_refine_channels < 8:
            raise ValueError("half_refine_channels must be >= 8")

        s4_channels, s8_channels, s16_channels = feature_channels
        # s32 is compressed by the backbone to the same width as s16.
        s32_channels = s16_channels
        self.enable_seg_aux = bool(enable_seg_aux)
        self.enable_centerline_aux = bool(enable_centerline_aux)

        self.fuse16 = ConvRelationFusion(
            s32_channels,
            s16_channels,
            channels,
            context_kernel_size,
            global_expansion,
            global_spatial_ratio,
            deploy,
        )
        self.fuse8 = ConvRelationFusion(
            channels,
            s8_channels,
            channels,
            highres_kernel_size,
            global_expansion,
            global_spatial_ratio,
            deploy,
        )
        self.fuse4 = ConvRelationFusion(
            channels,
            s4_channels,
            channels,
            highres_kernel_size,
            local_expansion,
            local_spatial_ratio,
            deploy,
        )
        self.fuse2 = ConvRelationFusion(
            channels,
            stem_channels,
            half_refine_channels,
            highres_kernel_size,
            local_expansion,
            max(0.5, local_spatial_ratio),
            deploy,
        )

        aux_channels = max(channels // 2, 32)
        self.aux_classifier = nn.Sequential(
            ConvBNAct(channels, aux_channels, 3),
            nn.Conv2d(aux_channels, num_classes, 1),
        )
        self.centerline_classifier = nn.Sequential(
            ConvBNAct(channels, aux_channels, 3),
            nn.Conv2d(aux_channels, 1, 1),
        )
        self.main_classifier = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(half_refine_channels, num_classes, 1),
        )
        self.fullres_head = FullResolutionHead(
            half_refine_channels, fullres_channels, num_classes
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        classifiers = (
            self.aux_classifier[-1],
            self.centerline_classifier[-1],
            self.main_classifier[-1],
            self.fullres_head.correction,
        )
        for classifier in classifiers:
            nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)

    def forward(
        self, features: Dict[str, Tensor]
    ) -> Union[
        Tensor,
        Tuple[Optional[Tensor], Optional[Tensor], Tensor],
    ]:
        required = {"s2", "s4", "s8", "s16", "s32"}
        missing = required.difference(features)
        if missing:
            raise KeyError(f"Missing backbone features: {sorted(missing)}")

        d16 = self.fuse16(features["s32"], features["s16"])
        d8 = self.fuse8(d16, features["s8"])
        d4 = self.fuse4(d8, features["s4"])
        d2 = self.fuse2(d4, features["s2"])
        coarse_logits = self.main_classifier(d2)
        main_logits = self.fullres_head(d2, coarse_logits)

        if self.training:
            return (
                self.aux_classifier(d8) if self.enable_seg_aux else None,
                self.centerline_classifier(d4)
                if self.enable_centerline_aux
                else None,
                main_logits,
            )
        return main_logits

    def switch_to_deploy(self) -> "GCNetHead":
        for module in list(self.modules()):
            if isinstance(module, CoMingBlock):
                module.switch_to_deploy()
        return self


__all__ = ["GCNetHead", "ConvRelationFusion", "FullResolutionHead"]
