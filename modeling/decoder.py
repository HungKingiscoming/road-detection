"""CNN-only detail-preserving decoder for CoMingNet v4.

Drop-in replacement for ``modeling/decoder.py``.  Training returns
``(aux_logits_s8, centerline_logits_s4, main_logits_fullres)`` and evaluation
returns only the full-resolution logits.

Compared with the previous decoder this version:
  * keeps a direct projected skip path at every fusion level;
  * zero-initializes only the learned relation residual;
  * actually respects the half/full-resolution feature flags;
  * uses PixelShuffle instead of transposed convolution;
  * optionally uses a tiny RGB detail branch at full resolution.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import CoMingBlock, ConvBNAct


class ConvRelationFusion(nn.Module):
    """Fuse deep semantics and a shallow skip without discarding the skip."""

    def __init__(
        self,
        deep_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int,
        expansion: float,
        spatial_ratio: float,
        deploy: bool = False,
        refine_blocks: int = 1,
    ) -> None:
        super().__init__()
        if refine_blocks < 1:
            raise ValueError("refine_blocks must be >= 1")

        self.deep_proj = ConvBNAct(deep_channels, out_channels, 1, padding=0)
        self.skip_proj = ConvBNAct(
            skip_channels, out_channels, 1, padding=0, activation=False
        )
        # deep, skip and their signed difference let ordinary convolutions
        # learn agreement/disagreement while retaining a direct skip residual.
        self.relation = nn.Sequential(
            ConvBNAct(out_channels * 3, out_channels, 3),
            ConvBNAct(out_channels, out_channels, 3, activation=False),
        )
        self.activation = nn.ReLU(inplace=True)
        self.refine = nn.Sequential(
            *[
                CoMingBlock(
                    out_channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    spatial_ratio=spatial_ratio,
                    deploy=deploy,
                    zero_init_residual=True,
                )
                for _ in range(refine_blocks)
            ]
        )

    def zero_init_relation(self) -> None:
        final_bn = self.relation[-1][-1]
        if not isinstance(final_bn, nn.BatchNorm2d):
            raise TypeError("Relation residual must end with BatchNorm2d")
        nn.init.zeros_(final_bn.weight)
        nn.init.zeros_(final_bn.bias)

    def forward(self, deep: Tensor, skip: Tensor) -> Tensor:
        deep = F.interpolate(
            self.deep_proj(deep),
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        skip = self.skip_proj(skip)
        base = deep + skip
        relation = self.relation(torch.cat((deep, skip, deep - skip), dim=1))
        return self.refine(self.activation(base + relation))


class FullResolutionHead(nn.Module):
    """Learn H/2-to-H reconstruction with PixelShuffle and RGB detail."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        detail_channels: int = 16,
    ) -> None:
        super().__init__()
        if hidden_channels < 8:
            raise ValueError("fullres_channels must be >= 8")
        if detail_channels < 4:
            raise ValueError("detail_channels must be >= 4")

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.rgb_detail = nn.Sequential(
            ConvBNAct(3, detail_channels, 3),
            ConvBNAct(detail_channels, detail_channels, 3),
        )
        merged_channels = hidden_channels + detail_channels
        self.refine = nn.Sequential(
            ConvBNAct(merged_channels, hidden_channels, 3),
            ConvBNAct(hidden_channels, hidden_channels, 3),
        )
        self.correction = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(
        self,
        feature: Tensor,
        coarse_logits: Tensor,
        image: Tensor,
    ) -> Tensor:
        full_feature = self.up(feature)
        detail = self.rgb_detail(image)
        if detail.shape[-2:] != full_feature.shape[-2:]:
            detail = F.interpolate(
                detail,
                size=full_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        full_feature = self.refine(torch.cat((full_feature, detail), dim=1))
        coarse_full = F.interpolate(
            coarse_logits,
            size=full_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return coarse_full + self.correction(full_feature)


class GCNetHead(nn.Module):
    """Compatibility name retained for the existing CoMingNet project."""

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
        detail_channels: int = 16,
        deploy: bool = False,
        align_corners: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        del in_channels, align_corners
        if len(feature_channels) != 3:
            raise ValueError("feature_channels must describe s4, s8 and s16")
        if enable_fullres_head and not enable_half_refine:
            raise ValueError("The full-resolution head requires half refinement")
        if half_refine_channels < 8:
            raise ValueError("half_refine_channels must be >= 8")

        s4_channels, s8_channels, s16_channels = feature_channels
        s32_channels = s16_channels
        self.enable_seg_aux = bool(enable_seg_aux)
        self.enable_centerline_aux = bool(enable_centerline_aux)
        self.enable_half_refine = bool(enable_half_refine)
        self.enable_fullres_head = bool(enable_fullres_head)

        self.fuse16 = ConvRelationFusion(
            s32_channels, s16_channels, channels, context_kernel_size,
            global_expansion, global_spatial_ratio, deploy,
        )
        self.fuse8 = ConvRelationFusion(
            channels, s8_channels, channels, highres_kernel_size,
            global_expansion, global_spatial_ratio, deploy,
        )
        self.fuse4 = ConvRelationFusion(
            channels, s4_channels, channels, highres_kernel_size,
            local_expansion, local_spatial_ratio, deploy, refine_blocks=2,
        )

        main_in_channels = channels
        if self.enable_half_refine:
            self.fuse2 = ConvRelationFusion(
                channels, stem_channels, half_refine_channels,
                highres_kernel_size, local_expansion,
                max(0.5, local_spatial_ratio), deploy, refine_blocks=2,
            )
            main_in_channels = half_refine_channels
        else:
            self.fuse2 = None

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
            nn.Conv2d(main_in_channels, num_classes, 1),
        )
        self.fullres_head: Optional[FullResolutionHead]
        if self.enable_fullres_head:
            self.fullres_head = FullResolutionHead(
                half_refine_channels,
                fullres_channels,
                num_classes,
                detail_channels=detail_channels,
            )
        else:
            self.fullres_head = None

        self._initialize()
        self._zero_init_residuals()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for classifier in (
            self.aux_classifier[-1],
            self.centerline_classifier[-1],
            self.main_classifier[-1],
        ):
            nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)
        if self.fullres_head is not None:
            # Start as bilinear coarse prediction, then learn only a correction.
            nn.init.zeros_(self.fullres_head.correction.weight)
            nn.init.zeros_(self.fullres_head.correction.bias)

    def _zero_init_residuals(self) -> None:
        for module in self.modules():
            if isinstance(module, ConvRelationFusion):
                module.zero_init_relation()
            elif isinstance(module, CoMingBlock):
                module.zero_init_residual()

    def forward(
        self, features: Dict[str, Tensor]
    ) -> Union[Tensor, Tuple[Optional[Tensor], Optional[Tensor], Tensor]]:
        required = {"input", "s2", "s4", "s8", "s16", "s32"}
        missing = required.difference(features)
        if missing:
            raise KeyError(f"Missing backbone features: {sorted(missing)}")

        d16 = self.fuse16(features["s32"], features["s16"])
        d8 = self.fuse8(d16, features["s8"])
        d4 = self.fuse4(d8, features["s4"])
        decoded = self.fuse2(d4, features["s2"]) if self.fuse2 else d4
        coarse_logits = self.main_classifier(decoded)

        if self.fullres_head is not None:
            main_logits = self.fullres_head(
                decoded, coarse_logits, features["input"]
            )
        else:
            main_logits = F.interpolate(
                coarse_logits,
                size=features["input"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.training:
            return (
                self.aux_classifier(d8) if self.enable_seg_aux else None,
                self.centerline_classifier(d4)
                if self.enable_centerline_aux else None,
                main_logits,
            )
        return main_logits

    def switch_to_deploy(self) -> "GCNetHead":
        for module in list(self.modules()):
            if isinstance(module, CoMingBlock):
                module.switch_to_deploy()
        return self


__all__ = ["GCNetHead", "ConvRelationFusion", "FullResolutionHead"]
