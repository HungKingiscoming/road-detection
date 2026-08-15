"""Lightweight additive decoder for CoMingNet.

The decoder never concatenates full-resolution features.  It projects all
three backbone scales to one small width and uses addition, so activation
memory stays predictable for batch size 8 on 512x512 crops.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import CoMingBlock, ConvBNAct


class GCNetHead(nn.Module):
    """Compatibility name retained for the existing training project.

    During training returns ``(aux_logits_s8, main_logits_s4)``.  During
    evaluation returns only ``main_logits_s4``.  Deep supervision therefore
    has zero inference cost.
    """

    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 96,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (128, 128, 128),
        dropout_ratio: float = 0.05,
        highres_kernel_size: int = 5,
        context_kernel_size: int = 7,
        deploy: bool = False,
        align_corners: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("feature_channels must describe s4, s8 and s16")
        s4_channels, s8_channels, s16_channels = feature_channels
        self.align_corners = align_corners

        self.context_proj = ConvBNAct(s16_channels, channels, 1, padding=0)
        self.context_refine = CoMingBlock(
            channels, context_kernel_size, deploy=deploy
        )

        self.mid_proj = ConvBNAct(s8_channels, channels, 1, padding=0)
        self.mid_refine = CoMingBlock(
            channels, highres_kernel_size, deploy=deploy
        )

        self.local_proj = ConvBNAct(s4_channels, channels, 1, padding=0)
        self.local_refine = CoMingBlock(
            channels, highres_kernel_size, deploy=deploy
        )

        aux_channels = max(channels // 2, 32)
        main_channels = max(channels * 2 // 3, 48)
        self.aux_classifier = nn.Sequential(
            ConvBNAct(channels, aux_channels, 3),
            nn.Conv2d(aux_channels, num_classes, 1),
        )
        self.main_classifier = nn.Sequential(
            ConvBNAct(channels, main_channels, 3),
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(main_channels, num_classes, 1),
        )
        self._initialize_classifiers()

    def _initialize_classifiers(self) -> None:
        for module in (self.aux_classifier, self.main_classifier):
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    if child.out_channels == self.main_classifier[-1].out_channels:
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
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not isinstance(features, dict):
            raise TypeError(
                "Balanced GCNetHead expects the feature dictionary returned by CoMingNet"
            )

        s4, s8, s16 = features["s4"], features["s8"], features["s16"]
        d16 = self.context_refine(self.context_proj(s16))
        d8 = self.mid_refine(self._resize(d16, s8.shape[-2:]) + self.mid_proj(s8))
        d4 = self.local_refine(self._resize(d8, s4.shape[-2:]) + self.local_proj(s4))
        main_logits = self.main_classifier(d4)

        if self.training:
            return self.aux_classifier(d8), main_logits
        return main_logits

    def switch_to_deploy(self) -> "GCNetHead":
        for module in list(self.modules()):
            if isinstance(module, CoMingBlock):
                module.switch_to_deploy()
        return self


__all__ = ["GCNetHead"]
