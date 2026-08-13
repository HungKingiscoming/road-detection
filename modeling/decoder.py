"""High-resolution decoder for CoMingNet V2.

The class name ``GCNetHead`` is intentionally retained so the existing
``train.py`` and ``Segmentor`` wrapper can import it without structural edits.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    # Kaggle project layout: place both files in modeling/.
    from .backbone import CoMingBlock
except ImportError:
    try:
        from backbone import CoMingBlock
    except ImportError:
        from coming_model import CoMingBlock


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, act=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class GCNetHead(nn.Module):
    """H/4 -> H/2 road decoder with shallow-detail fusion.

    Expected backbone dictionary (base channels C=32):
      fused:    [B, 4C, H/4, W/4]
      stem_half:[B,  C, H/2, W/2]
      aux:      [B, 2C, H/4, W/4]

    Training returns ``(aux_logits, main_logits)`` to remain compatible with
    the current trainer. Inference returns main logits only.
    """

    def __init__(self, in_channels: int = 128, channels: int = 128,
                 num_classes: int = 2, stem_channels: int = 32,
                 aux_in_channels: Optional[int] = None,
                 align_corners: bool = False, dropout_ratio: float = 0.1,
                 ignore_index: int = 255, loss_weight_aux: float = 0.3,
                 norm_cfg=None, act_cfg=None, init_cfg=None, **kwargs):
        super().__init__()
        self.align_corners = align_corners
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.loss_weight_aux = loss_weight_aux
        aux_in_channels = aux_in_channels or in_channels // 2

        # Main fused feature is projected before H/2 upsampling.
        self.main_project = ConvBNAct(in_channels, 96, 1, padding=0)
        self.stem_project = ConvBNAct(stem_channels, 32, 1, padding=0)
        self.detail_merge = ConvBNAct(128, 128, 1, padding=0)
        self.detail_refine = nn.Sequential(
            CoMingBlock(128, kernel_size=7, expansion=2.0),
            CoMingBlock(128, kernel_size=7, expansion=2.0),
        )
        self.dropout = (nn.Dropout2d(dropout_ratio)
                        if dropout_ratio > 0 else nn.Identity())
        self.cls_seg = nn.Conv2d(128, num_classes, 1)

        # Training-only auxiliary road segmentation at H/4.
        self.aux_head = nn.Sequential(
            ConvBNAct(aux_in_channels, 64, 3, padding=1),
            CoMingBlock(64, kernel_size=7, expansion=2.0),
            nn.Conv2d(64, num_classes, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _main_logits(self, features: Dict[str, Tensor]) -> Tensor:
        fused = self.main_project(features["fused"])
        stem = self.stem_project(features["stem_half"])
        fused = F.interpolate(
            fused, size=stem.shape[-2:], mode="bilinear",
            align_corners=self.align_corners,
        )
        x = self.detail_merge(torch.cat([fused, stem], dim=1))
        x = self.detail_refine(x)
        return self.cls_seg(self.dropout(x))

    def forward(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError(
                "CoMingNet V2 decoder expects the feature dictionary returned "
                "by coming_model.CoMingNet"
            )
        main = self._main_logits(inputs)
        if self.training:
            aux = self.aux_head(inputs["aux"])
            return aux, main
        return main

    def predict(self, inputs: Dict[str, Tensor],
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        logits = self.forward(inputs)
        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        if img_size is not None:
            logits = F.interpolate(
                logits, size=img_size, mode="bilinear",
                align_corners=self.align_corners,
            )
        return logits

    def switch_to_deploy(self) -> "GCNetHead":
        for module in list(self.modules()):
            if isinstance(module, CoMingBlock):
                module.switch_to_deploy()
        return self
