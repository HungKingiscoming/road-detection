"""Lightweight asymmetric decoder and road-specific auxiliary loss.

The decoder intentionally stops at output stride 2 and uses bilinear resize for
the final logits.  It is therefore much cheaper than a symmetric U-Net decoder.

Training output:
    centerline_logits_s4, road_logits_full

Evaluation output:
    road_logits_full

The complete objective contains exactly four primitive terms::

    L = CE_road + lambda_dice * Dice_road
        + lambda_aux * (BCE_centerline + lambda_center * Dice_centerline)

"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNAct(nn.Sequential):
    """Convolution followed by BatchNorm and an optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
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
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvBN(nn.Sequential):
    """Linear Conv-BN branch used by :class:`RepVGGBlock`."""

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
    """RepVGG block that can be exactly fused to one 3x3 convolution."""

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
            if self.in_channels == self.out_channels and self.stride == 1:
                self.branch_identity: Optional[nn.BatchNorm2d] = nn.BatchNorm2d(
                    self.in_channels
                )
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
        return (
            weight * scale.reshape(-1, 1, 1, 1),
            bn.bias - bn.running_mean * scale,
        )

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
        return (
            kernel * scale.reshape(-1, 1, 1, 1),
            bn.bias - bn.running_mean * scale,
        )

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        kernel_3, bias_3 = self._fuse_conv_bn(self.branch_3x3)
        kernel_1, bias_1 = self._fuse_conv_bn(self.branch_1x1)
        kernel_id, bias_id = self._fuse_identity_bn()
        kernel = kernel_3 + F.pad(kernel_1, (1, 1, 1, 1)) + kernel_id
        return kernel, bias_3 + bias_1 + bias_id

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


class SemanticSpatialRefinement(nn.Module):
    """Use OS16 semantics to suppress or amplify the detailed OS4 feature.

    The zero-initialized gate starts as an identity mapping.  This is safer
    than forcing a random sigmoid gate onto pretrained encoder features.
    """

    def __init__(self, shallow_channels: int, deep_channels: int) -> None:
        super().__init__()
        self.gate = nn.Conv2d(deep_channels, shallow_channels, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, shallow: Tensor, deep: Tensor) -> Tensor:
        guidance = F.interpolate(
            deep,
            size=shallow.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        modulation = torch.tanh(self.gate(guidance))
        return shallow * (1.0 + modulation)


class CompactRoadDecoder(nn.Module):
    """Asymmetric additive decoder with one train-only centerline head."""

    def __init__(
        self,
        stem_channels: int = 64,
        shallow_channels: int = 64,
        middle_channels: int = 128,
        deep_channels: int = 96,
        decoder_channels: int = 96,
        half_channels: int = 48,
        num_classes: int = 2,
        dropout: float = 0.05,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.semantic_refine = SemanticSpatialRefinement(
            shallow_channels, deep_channels
        )

        # Three features meet only at OS4. Additive fusion keeps memory and
        # parameters far below concatenation-heavy U-Net decoders.
        self.shallow_proj = ConvBNAct(
            shallow_channels, decoder_channels, 1, padding=0
        )
        self.middle_proj = ConvBNAct(
            middle_channels, decoder_channels, 1, padding=0
        )
        self.deep_proj = ConvBNAct(
            deep_channels, decoder_channels, 1, padding=0
        )
        self.p4_refine = RepVGGBlock(
            decoder_channels, decoder_channels, deploy=deploy
        )

        self.p4_to_p2 = ConvBNAct(
            decoder_channels, half_channels, 1, padding=0
        )
        self.stem_proj = ConvBNAct(stem_channels, half_channels, 1, padding=0)
        self.p2_refine = RepVGGBlock(
            half_channels, half_channels, deploy=deploy
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Conv2d(half_channels, num_classes, 1)

        # The only auxiliary. It is never evaluated at inference time.
        aux_channels = max(24, decoder_channels // 2)
        self.centerline_head = nn.Sequential(
            ConvBNAct(decoder_channels, aux_channels, 3),
            nn.Conv2d(aux_channels, 1, 1),
        )

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )

    def forward(
        self,
        stem_s2: Tensor,
        shallow_s4: Tensor,
        middle_s8: Tensor,
        deep_s16: Tensor,
        output_size: Tuple[int, int],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        shallow_s4 = self.semantic_refine(shallow_s4, deep_s16)
        p4_size = shallow_s4.shape[-2:]
        p4 = self.shallow_proj(shallow_s4)
        p4 = p4 + self._resize(self.middle_proj(middle_s8), p4_size)
        p4 = p4 + self._resize(self.deep_proj(deep_s16), p4_size)
        p4 = self.p4_refine(p4)

        p2 = self._resize(self.p4_to_p2(p4), stem_s2.shape[-2:])
        p2 = self.p2_refine(p2 + self.stem_proj(stem_s2))
        road_logits = self.classifier(self.dropout(p2))
        road_logits = self._resize(road_logits, output_size)

        if self.training:
            return self.centerline_head(p4), road_logits
        return road_logits

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()


def _soft_erode(mask: Tensor) -> Tensor:
    vertical = -F.max_pool2d(-mask, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-mask, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(mask: Tensor) -> Tensor:
    return F.max_pool2d(mask, 3, stride=1, padding=1)


def _soft_open(mask: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(mask))


def soft_skeletonize(mask: Tensor, iterations: int = 6) -> Tensor:
    """Differentiability is not required; this creates a target from the mask."""
    opened = _soft_open(mask)
    skeleton = F.relu(mask - opened)
    for _ in range(max(0, int(iterations))):
        mask = _soft_erode(mask)
        opened = _soft_open(mask)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp_(0.0, 1.0)


def binary_dice_loss(
    probability: Tensor, target: Tensor, eps: float = 1e-6
) -> Tensor:
    probability = probability.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


class RoadSegCenterlineLoss(nn.Module):
    """Weighted road CE + Dice and one centerline BCE + Dice auxiliary."""

    def __init__(
        self,
        road_class_weight: float = 2.0,
        main_dice_weight: float = 1.0,
        aux_weight: float = 0.20,
        centerline_pos_weight: float = 8.0,
        centerline_dice_weight: float = 1.0,
        skeleton_iterations: int = 6,
    ) -> None:
        super().__init__()
        self.road_class_weight = float(road_class_weight)
        self.main_dice_weight = float(main_dice_weight)
        self.aux_weight = float(aux_weight)
        self.centerline_pos_weight = float(centerline_pos_weight)
        self.centerline_dice_weight = float(centerline_dice_weight)
        self.skeleton_iterations = int(skeleton_iterations)

    def forward(
        self,
        outputs: Tuple[Tensor, Tensor],
        target: Tensor,
    ) -> Dict[str, Tensor]:
        centerline_logits, road_logits = outputs
        labels = (target > 0).long()
        road_mask = labels.unsqueeze(1).float()
        class_weights = road_logits.new_tensor([1.0, self.road_class_weight])
        loss_main_ce = F.cross_entropy(
            road_logits.float(), labels, weight=class_weights
        )
        road_probability = road_logits.float().softmax(dim=1)[:, 1:2]
        loss_main_dice = binary_dice_loss(road_probability, road_mask)

        # Build the topology-oriented target on GPU, then supervise at OS4.
        with torch.no_grad():
            centerline_target = soft_skeletonize(
                road_mask.float(), self.skeleton_iterations
            )
            centerline_target = F.interpolate(
                centerline_target,
                size=centerline_logits.shape[-2:],
                mode="nearest",
            )
        centerline_logits_f = centerline_logits.float()
        pos_weight = centerline_logits_f.new_tensor(
            [self.centerline_pos_weight]
        )
        loss_centerline_bce = F.binary_cross_entropy_with_logits(
            centerline_logits_f,
            centerline_target,
            pos_weight=pos_weight,
        )
        loss_centerline_dice = binary_dice_loss(
            centerline_logits_f.sigmoid(), centerline_target
        )
        loss_aux = (
            loss_centerline_bce
            + self.centerline_dice_weight * loss_centerline_dice
        )
        total = (
            loss_main_ce
            + self.main_dice_weight * loss_main_dice
            + self.aux_weight * loss_aux
        )
        return {
            "loss_total": total,
            "loss_main_ce": loss_main_ce.detach(),
            "loss_main_dice": loss_main_dice.detach(),
            "loss_aux_centerline": loss_aux.detach(),
            "loss_centerline_bce": loss_centerline_bce.detach(),
            "loss_centerline_dice": loss_centerline_dice.detach(),
        }


@torch.no_grad()
def verify_reparameterization(
    block: RepVGGBlock,
    shape: Tuple[int, int, int, int] = (2, 48, 32, 32),
) -> float:
    """Return maximum absolute error before/after RepVGG fusion."""
    if block.in_channels != shape[1]:
        raise ValueError("shape channels must match block.in_channels")
    block.eval()
    x = torch.randn(shape, device=next(block.parameters()).device)
    reference = block(x)
    block.switch_to_deploy()
    return float((reference - block(x)).abs().max())
