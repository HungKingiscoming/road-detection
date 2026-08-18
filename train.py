from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import CoMingBlock, ConvBNAct


class HalfResolutionRefine(nn.Module):
    """Fuse semantic s4 and shallow-detail s2 features at output stride 2.

    The semantic path is dominant at initialization.  ``detail_scale`` is a
    learnable per-channel parameter, initialized to 0.1, so the network can
    selectively introduce useful edges from s2 without allowing shallow image
    texture to overwrite the road semantics.

    A residual dense 3x3 fusion convolution first aligns the two sources.  A
    partial reparameterizable CoMingBlock then learns road-boundary structure
    at H/2 while keeping deployment conversion available.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        expansion: float = 1.5,
        spatial_ratio: float = 0.25,
        detail_scale_init: float = 0.1,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("channels must be >= 8")
        if not 0.0 <= detail_scale_init <= 1.0:
            raise ValueError("detail_scale_init must be in [0, 1]")

        self.detail_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(detail_scale_init))
        )
        self.pre_activation = nn.ReLU(inplace=False)
        self.fusion_conv = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            activation=False,
        )
        self.fusion_activation = nn.ReLU(inplace=True)
        self.refine = CoMingBlock(
            channels,
            kernel_size=kernel_size,
            expansion=expansion,
            spatial_ratio=spatial_ratio,
            deploy=deploy,
            zero_init_residual=True,
        )

        # Keep this BN normally initialized.  The project's build_model()
        # zero-initializes every CoMingBlock residual, so also zeroing this
        # fusion would make the complete H/2 adaptation path learn too slowly.

    def forward(self, semantic: Tensor, detail: Tensor) -> Tensor:
        if semantic.shape != detail.shape:
            raise ValueError(
                "semantic and detail features must have the same shape, got "
                f"{tuple(semantic.shape)} and {tuple(detail.shape)}"
            )
        fused = self.pre_activation(semantic + self.detail_scale * detail)
        fused = self.fusion_activation(fused + self.fusion_conv(fused))
        return self.refine(fused)


class FullResolutionResidualHead(nn.Module):
    """Learn a cheap full-resolution correction to coarse H/2 logits."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if hidden_channels < 8:
            raise ValueError("fullres_channels must be >= 8")

        # Produce four sub-pixel groups at H/2 and rearrange them to H.
        self.subpixel_projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels * 4,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.correction = nn.Conv2d(hidden_channels, num_classes, 1)

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

        # Exact function-preserving start: initially output only the resized
        # old classifier logits. Gradients first train this final layer, then
        # flow into the preceding full-resolution refinement layers.
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, d2: Tensor, coarse_logits: Tensor) -> Tensor:
        detail = self.refine(self.subpixel_projection(d2))
        coarse_full = F.interpolate(
            coarse_logits,
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return coarse_full + self.correction(detail)


class GCNetHead(nn.Module):
    """Additive CoMingNet decoder with optional semantic-controlled H/2 head."""

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
        half_refine_channels: int = 48,
        enable_fullres_head: bool = False,
        fullres_channels: int = 16,
        half_refine_kernel_size: int = 5,
        half_refine_expansion: float = 1.5,
        half_refine_spatial_ratio: float = 0.25,
        detail_scale_init: float = 0.1,
        deploy: bool = False,
        align_corners: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("feature_channels must describe s4, s8 and s16")
        if half_refine_channels < 8:
            raise ValueError("half_refine_channels must be >= 8")

        del in_channels  # Kept only for compatibility with the existing caller.
        s4_channels, s8_channels, s16_channels = feature_channels
        self.align_corners = align_corners
        self.enable_seg_aux = bool(enable_seg_aux)
        self.enable_centerline_aux = bool(enable_centerline_aux)
        self.enable_half_refine = bool(enable_half_refine)
        self.enable_fullres_head = bool(enable_fullres_head)
        if self.enable_fullres_head and not self.enable_half_refine:
            raise ValueError("Full-resolution head requires H/2 refinement")

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
            # Project at H/4 before resizing to keep the expensive H/2 path narrow.
            self.d4_to_half = ConvBNAct(
                channels,
                half_refine_channels,
                1,
                padding=0,
            )
            # No activation here: preserve signed shallow detail until fusion.
            self.s2_proj = ConvBNAct(
                stem_channels,
                half_refine_channels,
                1,
                padding=0,
                activation=False,
            )
            self.half_refine = HalfResolutionRefine(
                channels=half_refine_channels,
                kernel_size=half_refine_kernel_size,
                expansion=half_refine_expansion,
                spatial_ratio=half_refine_spatial_ratio,
                detail_scale_init=detail_scale_init,
                deploy=deploy,
            )
            self.main_classifier = nn.Sequential(
                nn.Dropout2d(dropout_ratio)
                if dropout_ratio > 0
                else nn.Identity(),
                nn.Conv2d(half_refine_channels, num_classes, 1),
            )
            self.fullres_head = (
                FullResolutionResidualHead(
                    in_channels=half_refine_channels,
                    hidden_channels=fullres_channels,
                    num_classes=num_classes,
                )
                if self.enable_fullres_head
                else None
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
            final_conv = module[-1]
            for child in module.modules():
                if not isinstance(child, nn.Conv2d):
                    continue
                if child is final_conv:
                    nn.init.normal_(child.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_normal_(
                        child.weight,
                        mode="fan_out",
                        nonlinearity="relu",
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
        self,
        features: Dict[str, Tensor],
    ) -> Union[
        Tensor,
        Tuple[Optional[Tensor], Optional[Tensor], Tensor],
    ]:
        if not isinstance(features, dict):
            raise TypeError(
                "GCNetHead expects the feature dictionary returned by CoMingNet"
            )

        missing = {"s4", "s8", "s16"}.difference(features)
        if missing:
            raise KeyError(f"Missing backbone features: {sorted(missing)}")

        s4, s8, s16 = features["s4"], features["s8"], features["s16"]
        d16 = self.context_refine(self.context_proj(s16))
        d8 = self.mid_refine(
            self._resize(d16, s8.shape[-2:]) + self.mid_proj(s8)
        )
        d4 = self.local_refine(
            self._resize(d8, s4.shape[-2:]) + self.local_proj(s4)
        )

        if self.enable_half_refine:
            if "s2" not in features:
                raise KeyError(
                    "H/2 refinement requires backbone feature 's2'. "
                    "Return stem_half as features['s2'] from CoMingNet.forward()."
                )
            s2 = features["s2"]
            semantic = self._resize(
                self.d4_to_half(d4),
                s2.shape[-2:],
            )
            detail = self.s2_proj(s2)
            d2 = self.half_refine(semantic, detail)
            coarse_logits = self.main_classifier(d2)
            main_logits = (
                self.fullres_head(d2, coarse_logits)
                if self.fullres_head is not None
                else coarse_logits
            )
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


__all__ = [
    "GCNetHead",
    "HalfResolutionRefine",
    "FullResolutionResidualHead",
]
