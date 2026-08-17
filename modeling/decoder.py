"""Lightweight additive decoder for CoMingNet.

The decoder never concatenates high-resolution features. It projects the
backbone scales to small widths and uses addition. An optional H/2 refinement
path consumes the already-computed stem feature and replaces the old direct
H/4-to-input bilinear upsampling with a learned, low-width boundary refiner.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import CoMingBlock, ConvBNAct


class HalfResolutionRefine(nn.Module):
    """One cheap learned refinement step at output stride 2.

    The residual branch uses one dense 3x3 convolution. Its final BN is
    zero-initialized, therefore the block starts as ReLU(identity) and does
    not inject a large random perturbation at the beginning of training.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(
            self.conv.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.bn(self.conv(x)))


class GCNetHead(nn.Module):
    """Compatibility name retained for the existing training project.

    During training returns ``(aux_logits_s8, centerline_logits_s4,
    main_logits)``; a disabled auxiliary is returned as ``None`` and is not
    executed. With H/2 refinement enabled, ``main_logits`` has output stride 2;
    otherwise the original output-stride-4 path is preserved exactly.
    """

    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 96,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (128, 128, 128),
        stem_channels: int = 32,
        dropout_ratio: float = 0.05,
        highres_kernel_size: int = 5,
        context_kernel_size: int = 7,
        local_expansion: float = 1.5,
        global_expansion: float = 2.0,
        local_spatial_ratio: float = 0.25,
        global_spatial_ratio: float = 0.5,
        enable_seg_aux: bool = False,
        enable_centerline_aux: bool = False,
        enable_half_refine: bool = False,
        half_refine_channels: int = 32,
        deploy: bool = False,
        align_corners: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("feature_channels must describe s4, s8 and s16")
        s4_channels, s8_channels, s16_channels = feature_channels
        self.align_corners = align_corners
        self.enable_seg_aux = bool(enable_seg_aux)
        self.enable_centerline_aux = bool(enable_centerline_aux)
        self.enable_half_refine = bool(enable_half_refine)
        if half_refine_channels < 8:
            raise ValueError("half_refine_channels must be >= 8")

        self.context_proj = ConvBNAct(s16_channels, channels, 1, padding=0)
        self.context_refine = CoMingBlock(
            channels,
            context_kernel_size,
            expansion=global_expansion,
            spatial_ratio=global_spatial_ratio,
            deploy=deploy,
        )

        self.mid_proj = ConvBNAct(s8_channels, channels, 1, padding=0)
        self.mid_refine = CoMingBlock(
            channels,
            highres_kernel_size,
            expansion=global_expansion,
            spatial_ratio=global_spatial_ratio,
            deploy=deploy,
        )

        self.local_proj = ConvBNAct(s4_channels, channels, 1, padding=0)
        self.local_refine = CoMingBlock(
            channels,
            highres_kernel_size,
            expansion=local_expansion,
            spatial_ratio=local_spatial_ratio,
            deploy=deploy,
        )

        aux_channels = max(channels // 2, 32)
        main_channels = max(channels * 2 // 3, 48)
        self.aux_classifier = nn.Sequential(
            ConvBNAct(channels, aux_channels, 3),
            nn.Conv2d(aux_channels, num_classes, 1),
        )
        self.centerline_classifier = nn.Sequential(
            ConvBNAct(channels, aux_channels, 3),
            nn.Conv2d(aux_channels, 1, 1),
        )
        if self.enable_half_refine:
            # Project d4 before upsampling: cheaper than a 1x1 convolution at H/2.
            self.d4_to_half = ConvBNAct(
                channels,
                half_refine_channels,
                1,
                padding=0,
            )
            self.s2_proj = ConvBNAct(
                stem_channels,
                half_refine_channels,
                1,
                padding=0,
            )
            self.half_refine = HalfResolutionRefine(half_refine_channels)
            self.main_classifier = nn.Sequential(
                nn.Dropout2d(dropout_ratio)
                if dropout_ratio > 0
                else nn.Identity(),
                nn.Conv2d(half_refine_channels, num_classes, 1),
            )
        else:
            self.main_classifier = nn.Sequential(
                ConvBNAct(channels, main_channels, 3),
                nn.Dropout2d(dropout_ratio)
                if dropout_ratio > 0
                else nn.Identity(),
                nn.Conv2d(main_channels, num_classes, 1),
            )
        self._initialize_classifiers()

    def _initialize_classifiers(self) -> None:
        for module in (
            self.aux_classifier,
            self.centerline_classifier,
            self.main_classifier,
        ):
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    if child is module[-1]:
                        nn.init.normal_(child.weight, mean=0.0, std=0.01)
                    else:
                        nn.init.kaiming_normal_(
                            child.weight, mode="fan_out", nonlinearity="relu"
                        )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    def _resize(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

    def forward(
        self, features: Dict[str, Tensor]
    ) -> Union[
        Tensor,
        Tuple[Optional[Tensor], Optional[Tensor], Tensor],
    ]:
        if not isinstance(features, dict):
            raise TypeError(
                "Balanced GCNetHead expects the feature dictionary returned by CoMingNet"
            )

        s4, s8, s16 = features["s4"], features["s8"], features["s16"]
        d16 = self.context_refine(self.context_proj(s16))
        d8 = self.mid_refine(self._resize(d16, s8.shape[-2:]) + self.mid_proj(s8))
        d4 = self.local_refine(self._resize(d8, s4.shape[-2:]) + self.local_proj(s4))
        if self.enable_half_refine:
            if "s2" not in features:
                raise KeyError(
                    "H/2 refinement requires backbone feature 's2'. "
                    "Return stem_half as features['s2'] from CoMingNet.forward()."
                )
            s2 = features["s2"]
            d2 = self._resize(self.d4_to_half(d4), s2.shape[-2:])
            d2 = self.half_refine(d2 + self.s2_proj(s2))
            main_logits = self.main_classifier(d2)
        else:
            main_logits = self.main_classifier(d4)

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


__all__ = ["GCNetHead", "HalfResolutionRefine"]
