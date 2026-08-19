"""CoMingNet two-stream road model with RepVGG blocks and one road auxiliary.

This module is a standalone replacement for the previous CoMingNet backbone
and decoder.  Every CoMingBlock has been replaced by a standard RepVGG block:

    3x3 Conv-BN + 1x1 Conv-BN + identity-BN -> one 3x3 Conv at deployment.

The only auxiliary task is road-centerline prediction at output stride 4.
During inference the auxiliary head is discarded and only the full-resolution
road logits are returned.

Training output:
    centerline_logits_s4, road_logits_full

Evaluation output:
    road_logits_full

The accompanying RoadSegCenterlineLoss creates a centerline target directly
from a binary road mask and optimizes:

    L = CE_road + Dice_road
        + aux_weight * (weighted_BCE_centerline + Dice_centerline)

There is deliberately no boundary, direction, connectivity, or clDice term in
this control version.  Add those only through later ablation experiments.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNAct(nn.Sequential):
    """Convolution, BatchNorm, and optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        activation: bool = True,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvBN(nn.Sequential):
    """Conv-BN branch used inside RepVGGBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    @property
    def conv(self) -> nn.Conv2d:
        return self[0]

    @property
    def bn(self) -> nn.BatchNorm2d:
        return self[1]


class RepVGGBlock(nn.Module):
    """Standard RepVGG training block with exact 3x3 deployment fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.deploy = bool(deploy)
        self.activation = nn.ReLU(inplace=True)

        if self.deploy:
            self.reparam = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.stride,
                padding=1,
                bias=True,
            )
        else:
            self.branch_3x3 = ConvBN(
                self.in_channels, self.out_channels, 3, self.stride, 1
            )
            self.branch_1x1 = ConvBN(
                self.in_channels, self.out_channels, 1, self.stride, 0
            )
            self.branch_identity: Optional[nn.BatchNorm2d]
            if self.in_channels == self.out_channels and self.stride == 1:
                self.branch_identity = nn.BatchNorm2d(self.in_channels)
            else:
                self.branch_identity = None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.activation(self.reparam(x))
        identity: Union[Tensor, int]
        identity = self.branch_identity(x) if self.branch_identity else 0
        return self.activation(
            self.branch_3x3(x) + self.branch_1x1(x) + identity
        )

    @staticmethod
    def _fuse_conv_bn(branch: ConvBN) -> Tuple[Tensor, Tensor]:
        weight = branch.conv.weight
        bn = branch.bn
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        kernel = weight * scale.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * scale
        return kernel, bias

    def _fuse_identity_bn(self) -> Tuple[Union[Tensor, int], Union[Tensor, int]]:
        if self.branch_identity is None:
            return 0, 0
        bn = self.branch_identity
        kernel = bn.weight.new_zeros(
            self.out_channels, self.in_channels, 3, 3
        )
        indices = torch.arange(self.in_channels, device=kernel.device)
        kernel[indices, indices, 1, 1] = 1.0
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        kernel = kernel * scale.reshape(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * scale
        return kernel, bias

    @staticmethod
    def _pad_1x1_to_3x3(kernel: Tensor) -> Tensor:
        return F.pad(kernel, (1, 1, 1, 1))

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        kernel_3x3, bias_3x3 = self._fuse_conv_bn(self.branch_3x3)
        kernel_1x1, bias_1x1 = self._fuse_conv_bn(self.branch_1x1)
        kernel_identity, bias_identity = self._fuse_identity_bn()
        kernel = (
            kernel_3x3
            + self._pad_1x1_to_3x3(kernel_1x1)
            + kernel_identity
        )
        bias = bias_3x3 + bias_1x1 + bias_identity
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            stride=self.stride,
            padding=1,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.reparam = reparam
        del self.branch_3x3
        del self.branch_1x1
        del self.branch_identity
        self.deploy = True


class DeepPyramidContext(nn.Module):
    """CNN pyramid context operating at output stride 32."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_channels: int,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.pool_sizes = (1, 2, 4, 8)
        self.shortcut = ConvBNAct(
            in_channels, out_channels, 1, padding=0, activation=False
        )
        self.pool_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 1, bias=True),
                    nn.ReLU(inplace=True),
                )
                for _ in self.pool_sizes
            ]
        )
        self.pool_process = nn.ModuleList(
            [
                ConvBNAct(branch_channels, branch_channels, 3)
                for _ in self.pool_sizes
            ]
        )
        merged_channels = out_channels + len(self.pool_sizes) * branch_channels
        self.fuse = ConvBNAct(merged_channels, out_channels, 1, padding=0)
        self.refine = RepVGGBlock(out_channels, deploy=deploy)

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]
        pooled_features = []
        for pool_size, projection, process in zip(
            self.pool_sizes, self.pool_projections, self.pool_process
        ):
            pooled = F.adaptive_avg_pool2d(x, pool_size)
            pooled = projection(pooled)
            pooled = F.interpolate(
                pooled,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            pooled_features.append(process(pooled))
        shortcut = self.shortcut(x)
        fused = self.fuse(torch.cat([shortcut, *pooled_features], dim=1))
        return self.refine(shortcut + fused)


class CoMingNet(nn.Module):
    """Two-stream backbone using only standard RepVGG blocks.

    Old CoMingBlock-specific arguments remain explicit for builder
    compatibility, but no longer affect the model.  This prevents old training
    scripts from crashing while making the architectural change unambiguous.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 40,
        local_blocks: Sequence[int] = (2, 2, 2),
        global_blocks: Sequence[int] = (3, 4),
        deep_blocks: int = 2,
        deploy: bool = False,
        highres_kernel_size: int = 5,
        context_kernel_size: int = 7,
        coming_kernel_size: Optional[int] = None,
        local_expansion: float = 1.5,
        global_expansion: float = 2.0,
        local_spatial_ratio: float = 0.5,
        global_spatial_ratio: float = 0.5,
        deep_spatial_ratio: float = 0.75,
        kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        del (
            highres_kernel_size,
            context_kernel_size,
            coming_kernel_size,
            local_expansion,
            global_expansion,
            local_spatial_ratio,
            global_spatial_ratio,
            deep_spatial_ratio,
            kernel_size,
        )
        if len(local_blocks) != 3:
            raise ValueError("local_blocks must contain three integers")
        if len(global_blocks) not in (2, 3):
            raise ValueError("global_blocks must contain two integers")
        if deep_blocks < 1:
            raise ValueError("deep_blocks must be >= 1")
        global_blocks = tuple(global_blocks[:2])

        c1, c2, c4, c8 = channels, channels * 2, channels * 4, channels * 8

        self.stem_half = nn.Sequential(
            ConvBNAct(in_channels, c1, 3, stride=2),
            ConvBNAct(c1, c1, 3),
        )
        self.stem_quarter = ConvBNAct(c1, c2, 3, stride=2)

        self.local_stage1 = self._stage(c2, local_blocks[0], deploy)
        self.global_stage1 = nn.Sequential(
            ConvBNAct(c2, c4, 3, stride=2),
            self._stage(c4, global_blocks[0], deploy),
        )

        self.g2l1_proj = ConvBNAct(c4, c2, 1, padding=0, activation=False)
        self.l2g1_proj = ConvBNAct(c2, c4, 3, stride=2, activation=False)

        self.local_stage2 = self._stage(c2, local_blocks[1], deploy)
        self.global_stage2 = nn.Sequential(
            ConvBNAct(c4, c8, 3, stride=2),
            self._stage(c8, global_blocks[1], deploy),
        )

        self.g2l2_proj = ConvBNAct(c8, c2, 1, padding=0, activation=False)
        self.l2g2_proj = nn.Sequential(
            ConvBNAct(c2, c4, 3, stride=2),
            ConvBNAct(c4, c8, 3, stride=2, activation=False),
        )

        self.local_transition = ConvBNAct(c2, c4, 3)
        self.local_stage3 = self._stage(c4, local_blocks[2], deploy)

        self.s16_proj = ConvBNAct(c8, c4, 1, padding=0)
        self.deep_downsample = ConvBNAct(c8, c8, 3, stride=2)
        self.deep_stage = self._stage(c8, deep_blocks, deploy)
        self.deep_context = DeepPyramidContext(
            c8, c4, branch_channels=channels, deploy=deploy
        )
        self.activation = nn.ReLU(inplace=True)
        self.deploy = bool(deploy)
        self._initialize()

    @staticmethod
    def _stage(channels: int, blocks: int, deploy: bool) -> nn.Sequential:
        if blocks < 1:
            raise ValueError("Each stage must contain at least one block")
        return nn.Sequential(
            *[RepVGGBlock(channels, deploy=deploy) for _ in range(blocks)]
        )

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

    @staticmethod
    def _resize_like(source: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        s2 = self.stem_half(x)
        shared = self.stem_quarter(s2)

        local1_old = self.local_stage1(shared)
        global1_old = self.global_stage1(shared)
        local1 = self.activation(
            local1_old
            + self._resize_like(self.g2l1_proj(global1_old), local1_old)
        )
        global1 = self.activation(global1_old + self.l2g1_proj(local1_old))

        local2_old = self.local_stage2(local1)
        global2_old = self.global_stage2(global1)
        local2 = self.activation(
            local2_old
            + self._resize_like(self.g2l2_proj(global2_old), local2_old)
        )
        local_to_global = self.l2g2_proj(local2_old)
        if local_to_global.shape[-2:] != global2_old.shape[-2:]:
            local_to_global = self._resize_like(local_to_global, global2_old)
        global2 = self.activation(global2_old + local_to_global)

        s4 = self.local_stage3(self.local_transition(local2))
        s16 = self.s16_proj(global2)
        s32 = self.deep_context(self.deep_stage(self.deep_downsample(global2)))
        return {"s2": s2, "s4": s4, "s8": global1, "s16": s16, "s32": s32}

    def switch_to_deploy(self) -> "CoMingNet":
        for module in list(self.modules()):
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()
        self.deploy = True
        return self


class ConvRelationFusion(nn.Module):
    """Deep-guided skip fusion followed by RepVGG refinement."""

    def __init__(
        self,
        deep_channels: int,
        skip_channels: int,
        out_channels: int,
        deploy: bool = False,
        refine_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.deep_proj = ConvBNAct(deep_channels, out_channels, 1, padding=0)
        self.skip_proj = ConvBNAct(
            skip_channels, out_channels, 1, padding=0, activation=False
        )
        self.relation = nn.Sequential(
            ConvBNAct(out_channels * 3, out_channels, 3),
            ConvBNAct(out_channels, out_channels, 3, activation=False),
        )
        self.activation = nn.ReLU(inplace=True)
        self.refine = nn.Sequential(
            *[
                RepVGGBlock(out_channels, deploy=deploy)
                for _ in range(refine_blocks)
            ]
        )

    def zero_init_relation(self) -> None:
        final_bn = self.relation[-1][-1]
        if not isinstance(final_bn, nn.BatchNorm2d):
            raise TypeError("Relation branch must end with BatchNorm2d")
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
        relation = self.relation(torch.cat((deep, skip, deep - skip), dim=1))
        return self.refine(self.activation(deep + skip + relation))


class FullResolutionHead(nn.Module):
    """Optional H/2-to-H reconstruction with an RGB detail correction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        detail_channels: int = 16,
    ) -> None:
        super().__init__()
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
        self.refine = nn.Sequential(
            ConvBNAct(hidden_channels + detail_channels, hidden_channels, 3),
            ConvBNAct(hidden_channels, hidden_channels, 3),
        )
        self.correction = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(
        self, feature: Tensor, coarse_logits: Tensor, image: Tensor
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
        correction = self.correction(
            self.refine(torch.cat((full_feature, detail), dim=1))
        )
        coarse = F.interpolate(
            coarse_logits,
            size=correction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return coarse + correction


class GCNetHead(nn.Module):
    """Decoder with one road-centerline auxiliary head at output stride 4."""

    def __init__(
        self,
        channels: int = 128,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (160, 160, 160),
        stem_channels: int = 40,
        dropout_ratio: float = 0.05,
        enable_half_refine: bool = True,
        half_refine_channels: int = 64,
        enable_fullres_head: bool = False,
        fullres_channels: int = 24,
        detail_channels: int = 16,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("feature_channels must describe s4, s8, and s16")
        if enable_fullres_head and not enable_half_refine:
            raise ValueError("Full-resolution head requires half refinement")
        s4_channels, s8_channels, s16_channels = feature_channels
        s32_channels = s16_channels
        self.enable_half_refine = bool(enable_half_refine)
        self.enable_fullres_head = bool(enable_fullres_head)

        self.fuse16 = ConvRelationFusion(
            s32_channels, s16_channels, channels, deploy
        )
        self.fuse8 = ConvRelationFusion(channels, s8_channels, channels, deploy)
        self.fuse4 = ConvRelationFusion(
            channels, s4_channels, channels, deploy, refine_blocks=2
        )
        if self.enable_half_refine:
            self.fuse2: Optional[ConvRelationFusion] = ConvRelationFusion(
                channels,
                stem_channels,
                half_refine_channels,
                deploy,
                refine_blocks=2,
            )
            main_channels = half_refine_channels
        else:
            self.fuse2 = None
            main_channels = channels

        aux_channels = max(channels // 2, 32)
        self.centerline_aux = nn.Sequential(
            ConvBNAct(channels, aux_channels, 3),
            nn.Conv2d(aux_channels, 1, 1),
        )
        self.main_classifier = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(main_channels, num_classes, 1),
        )
        if self.enable_fullres_head:
            self.fullres_head: Optional[FullResolutionHead] = FullResolutionHead(
                half_refine_channels,
                fullres_channels,
                num_classes,
                detail_channels,
            )
        else:
            self.fullres_head = None
        self._initialize()

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
        nn.init.normal_(self.centerline_aux[-1].weight, std=0.01)
        nn.init.normal_(self.main_classifier[-1].weight, std=0.01)
        for module in self.modules():
            if isinstance(module, ConvRelationFusion):
                module.zero_init_relation()
        if self.fullres_head is not None:
            nn.init.zeros_(self.fullres_head.correction.weight)
            nn.init.zeros_(self.fullres_head.correction.bias)

    def forward(
        self, features: Dict[str, Tensor]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
            return self.centerline_aux(d4), main_logits
        return main_logits

    def switch_to_deploy(self) -> "GCNetHead":
        for module in list(self.modules()):
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()
        return self


class RoadRepVGGNet(nn.Module):
    """Convenience wrapper joining backbone and road decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 40,
        decoder_channels: int = 128,
        local_blocks: Sequence[int] = (2, 2, 2),
        global_blocks: Sequence[int] = (3, 4),
        deep_blocks: int = 2,
        num_classes: int = 2,
        half_refine_channels: int = 64,
        enable_fullres_head: bool = False,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = CoMingNet(
            in_channels=in_channels,
            channels=channels,
            local_blocks=local_blocks,
            global_blocks=global_blocks,
            deep_blocks=deep_blocks,
            deploy=deploy,
        )
        feature_channels = (channels * 4, channels * 4, channels * 4)
        self.decode_head = GCNetHead(
            channels=decoder_channels,
            num_classes=num_classes,
            feature_channels=feature_channels,
            stem_channels=channels,
            half_refine_channels=half_refine_channels,
            enable_fullres_head=enable_fullres_head,
            deploy=deploy,
        )

    def forward(
        self, image: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        features = self.backbone(image)
        features["input"] = image
        return self.decode_head(features)

    def switch_to_deploy(self) -> "RoadRepVGGNet":
        self.eval()
        self.backbone.switch_to_deploy()
        self.decode_head.switch_to_deploy()
        return self


def _soft_erode(mask: Tensor) -> Tensor:
    vertical = -F.max_pool2d(-mask, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-mask, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(mask: Tensor) -> Tensor:
    return F.max_pool2d(mask, 3, stride=1, padding=1)


def _soft_open(mask: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(mask))


def soft_skeletonize(mask: Tensor, iterations: int = 10) -> Tensor:
    """Differentiable morphology used here without gradients for GT creation."""
    opened = _soft_open(mask)
    skeleton = F.relu(mask - opened)
    for _ in range(iterations):
        mask = _soft_erode(mask)
        opened = _soft_open(mask)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp_(0.0, 1.0)


@torch.no_grad()
def build_centerline_target(
    road_target: Tensor,
    output_size: Tuple[int, int],
    skeleton_iterations: int = 10,
) -> Tensor:
    """Create an OS4 centerline target from BHW or B1HW road masks."""
    if road_target.ndim == 3:
        road_target = road_target.unsqueeze(1)
    if road_target.ndim != 4 or road_target.shape[1] != 1:
        raise ValueError("road_target must have shape BHW or B1HW")
    road_target = (road_target > 0).to(dtype=torch.float32)
    centerline = soft_skeletonize(road_target, skeleton_iterations)
    if centerline.shape[-2:] != output_size:
        centerline = F.adaptive_max_pool2d(centerline, output_size)
    return centerline


def binary_dice_loss(probability: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


class RoadSegCenterlineLoss(nn.Module):
    """Main road segmentation plus exactly one centerline auxiliary loss."""

    def __init__(
        self,
        aux_weight: float = 0.20,
        main_dice_weight: float = 1.0,
        centerline_dice_weight: float = 1.0,
        centerline_pos_weight: float = 8.0,
        skeleton_iterations: int = 10,
        road_class_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if aux_weight < 0:
            raise ValueError("aux_weight must be non-negative")
        self.aux_weight = float(aux_weight)
        self.main_dice_weight = float(main_dice_weight)
        self.centerline_dice_weight = float(centerline_dice_weight)
        self.centerline_pos_weight = float(centerline_pos_weight)
        self.skeleton_iterations = int(skeleton_iterations)
        self.road_class_weight = float(road_class_weight)

    def forward(
        self,
        outputs: Tuple[Tensor, Tensor],
        road_target: Tensor,
    ) -> Dict[str, Tensor]:
        centerline_logits, main_logits = outputs
        if main_logits.shape[1] != 2:
            raise ValueError("RoadSegCenterlineLoss expects two-class main logits")
        if road_target.ndim == 4:
            road_target = road_target.squeeze(1)
        if road_target.ndim != 3:
            raise ValueError("road_target must have shape BHW or B1HW")
        labels = (road_target > 0).long()
        road_mask = labels.unsqueeze(1).to(dtype=main_logits.dtype)

        class_weights = main_logits.new_tensor([1.0, self.road_class_weight])
        loss_main_ce = F.cross_entropy(main_logits, labels, weight=class_weights)
        road_probability = torch.softmax(main_logits, dim=1)[:, 1:2]
        loss_main_dice = binary_dice_loss(road_probability, road_mask)

        centerline_target = build_centerline_target(
            labels,
            output_size=centerline_logits.shape[-2:],
            skeleton_iterations=self.skeleton_iterations,
        ).to(dtype=centerline_logits.dtype)
        positive_weight = centerline_logits.new_tensor(
            self.centerline_pos_weight
        )
        loss_centerline_bce = F.binary_cross_entropy_with_logits(
            centerline_logits,
            centerline_target,
            pos_weight=positive_weight,
        )
        loss_centerline_dice = binary_dice_loss(
            torch.sigmoid(centerline_logits), centerline_target
        )
        loss_aux = (
            loss_centerline_bce
            + self.centerline_dice_weight * loss_centerline_dice
        )
        loss_total = (
            loss_main_ce
            + self.main_dice_weight * loss_main_dice
            + self.aux_weight * loss_aux
        )
        return {
            "loss_total": loss_total,
            "loss_main_ce": loss_main_ce.detach(),
            "loss_main_dice": loss_main_dice.detach(),
            "loss_aux_centerline": loss_aux.detach(),
            "loss_centerline_bce": loss_centerline_bce.detach(),
            "loss_centerline_dice": loss_centerline_dice.detach(),
        }


@torch.no_grad()
def verify_reparameterization(
    model: RoadRepVGGNet,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    atol: float = 2e-4,
) -> Tuple[float, float]:
    """Compare full-resolution logits before and after RepVGG fusion."""
    model.eval()
    parameter = next(model.parameters())
    image = torch.randn(
        input_shape, device=parameter.device, dtype=parameter.dtype
    )
    before = model(image)
    model.switch_to_deploy()
    after = model(image)
    error = (before - after).abs()
    max_error = float(error.max())
    mean_error = float(error.mean())
    if max_error > atol:
        raise AssertionError(
            f"Reparameterization error {max_error:.6g} exceeds atol={atol}"
        )
    return max_error, mean_error
