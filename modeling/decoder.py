from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBNAct(nn.Sequential):
    """Convolution, BatchNorm, and an optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: bool = True,
        zero_init_bn: bool = False,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        norm = nn.BatchNorm2d(out_channels)
        if zero_init_bn:
            nn.init.zeros_(norm.weight)
        layers: list[nn.Module] = [convolution, norm]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvGNAct(nn.Sequential):
    """Conv-GroupNorm-ReLU used on pooled maps, including 1x1 maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        activation: bool = True,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvBN(nn.Sequential):
    """Linear Conv-BN branch used by re-parameterizable blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int,
        padding: Union[int, Tuple[int, int]],
        groups: int = 1,
    ) -> None:
        super().__init__(
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
        )

    @property
    def conv(self) -> nn.Conv2d:
        return self[0]

    @property
    def bn(self) -> nn.BatchNorm2d:
        return self[1]


def _fuse_conv_bn(branch: ConvBN) -> Tuple[Tensor, Tensor]:
    weight = branch.conv.weight
    norm = branch.bn
    std = torch.sqrt(norm.running_var + norm.eps)
    scale = norm.weight / std
    return (
        weight * scale.reshape(-1, 1, 1, 1),
        norm.bias - norm.running_mean * scale,
    )


class RepVGGBlock(nn.Module):
    """RepVGG block exactly fused to one dense 3x3 convolution at deploy."""

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

    def _fuse_identity_bn(self) -> Tuple[Union[Tensor, int], Union[Tensor, int]]:
        if self.branch_identity is None:
            return 0, 0
        norm = self.branch_identity
        kernel = norm.weight.new_zeros(
            self.out_channels, self.in_channels, 3, 3
        )
        indices = torch.arange(self.in_channels, device=kernel.device)
        kernel[indices, indices, 1, 1] = 1.0
        std = torch.sqrt(norm.running_var + norm.eps)
        scale = norm.weight / std
        return (
            kernel * scale.reshape(-1, 1, 1, 1),
            norm.bias - norm.running_mean * scale,
        )

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        kernel_3, bias_3 = _fuse_conv_bn(self.branch_3x3)
        kernel_1, bias_1 = _fuse_conv_bn(self.branch_1x1)
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


class RepDepthwiseBlock(nn.Module):
    """Road refinement with a deployable depthwise 5x5 spatial kernel.

    During training, 3x3, 1x5, 5x1, and identity paths learn complementary
    road geometry. The four paths are exactly fused into one depthwise 5x5
    convolution for inference. The inexpensive pointwise mixer remains.
    """

    def __init__(self, channels: int, deploy: bool = False) -> None:
        super().__init__()
        self.channels = int(channels)
        self.deploy = bool(deploy)
        self.spatial_activation = nn.ReLU(inplace=True)
        self.output_activation = nn.ReLU(inplace=True)
        if self.deploy:
            self.spatial_reparam = nn.Conv2d(
                channels,
                channels,
                5,
                padding=2,
                groups=channels,
                bias=True,
            )
        else:
            self.branch_3x3 = ConvBN(
                channels, channels, 3, 1, 1, groups=channels
            )
            self.branch_1x5 = ConvBN(
                channels, channels, (1, 5), 1, (0, 2), groups=channels
            )
            self.branch_5x1 = ConvBN(
                channels, channels, (5, 1), 1, (2, 0), groups=channels
            )
            self.branch_identity = nn.BatchNorm2d(channels)
        self.pointwise = ConvBNAct(
            channels, channels, 1, padding=0, activation=False
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            spatial = self.spatial_reparam(x)
        else:
            spatial = (
                self.branch_3x3(x)
                + self.branch_1x5(x)
                + self.branch_5x1(x)
                + self.branch_identity(x)
            )
        spatial = self.spatial_activation(spatial)
        return self.output_activation(x + self.pointwise(spatial))

    @staticmethod
    def _pad_to_5x5(kernel: Tensor) -> Tensor:
        height, width = kernel.shape[-2:]
        pad_h, pad_w = 5 - height, 5 - width
        return F.pad(
            kernel,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

    def _fuse_identity(self) -> Tuple[Tensor, Tensor]:
        norm = self.branch_identity
        kernel = norm.weight.new_zeros(self.channels, 1, 5, 5)
        kernel[:, 0, 2, 2] = 1.0
        std = torch.sqrt(norm.running_var + norm.eps)
        scale = norm.weight / std
        return (
            kernel * scale.reshape(-1, 1, 1, 1),
            norm.bias - norm.running_mean * scale,
        )

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.spatial_reparam.weight, self.spatial_reparam.bias
        kernels, biases = [], []
        for branch in (self.branch_3x3, self.branch_1x5, self.branch_5x1):
            kernel, bias = _fuse_conv_bn(branch)
            kernels.append(self._pad_to_5x5(kernel))
            biases.append(bias)
        kernel_id, bias_id = self._fuse_identity()
        return sum(kernels, kernel_id), sum(biases, bias_id)

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            5,
            padding=2,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.spatial_reparam = reparam
        del self.branch_3x3
        del self.branch_1x5
        del self.branch_5x1
        del self.branch_identity
        self.deploy = True


class SeparableConvBNAct(nn.Sequential):
    """Depthwise-separable full-resolution prediction refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNAct(
                in_channels,
                in_channels,
                3,
                groups=in_channels,
            ),
            ConvBNAct(in_channels, out_channels, 1, padding=0),
        )


class RoadReconstructionDecoder(nn.Module):
    """S8-to-S1 road decoder with S4/S2 skips and one centerline head."""

    def __init__(
        self,
        stem_channels: int = 64,
        shallow_channels: int = 64,
        fused_channels: int = 96,
        s4_channels: int = 64,
        s2_channels: int = 32,
        full_channels: int = 24,
        num_classes: int = 2,
        dropout: float = 0.05,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        shallow_skip_channels = max(24, s4_channels // 2)
        stem_skip_channels = max(16, s2_channels // 2)

        self.fused_proj = ConvBNAct(fused_channels, s4_channels, 1, padding=0)
        self.shallow_proj = ConvBNAct(
            shallow_channels, shallow_skip_channels, 1, padding=0
        )
        self.s4_fuse = ConvBNAct(
            s4_channels + shallow_skip_channels,
            s4_channels,
            1,
            padding=0,
        )
        self.s4_refine = nn.Sequential(
            RepDepthwiseBlock(s4_channels, deploy=deploy),
            RepDepthwiseBlock(s4_channels, deploy=deploy),
        )

        self.stem_proj = ConvBNAct(
            stem_channels, stem_skip_channels, 1, padding=0
        )
        self.s2_fuse = ConvBNAct(
            s4_channels + stem_skip_channels,
            s2_channels,
            1,
            padding=0,
        )
        self.s2_refine = nn.Sequential(
            RepDepthwiseBlock(s2_channels, deploy=deploy),
            RepDepthwiseBlock(s2_channels, deploy=deploy),
        )

        self.full_refine = SeparableConvBNAct(s2_channels, full_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Conv2d(full_channels, num_classes, 1)

        auxiliary_channels = max(24, s4_channels // 2)
        self.centerline_head = nn.Sequential(
            ConvBNAct(s4_channels, auxiliary_channels, 3),
            nn.Conv2d(auxiliary_channels, 1, 1),
        )

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        stem_s2: Tensor,
        shallow_s4: Tensor,
        fused_s8: Tensor,
        output_size: Tuple[int, int],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        p4 = self._resize(self.fused_proj(fused_s8), shallow_s4.shape[-2:])
        p4 = self.s4_fuse(torch.cat((p4, self.shallow_proj(shallow_s4)), dim=1))
        p4 = self.s4_refine(p4)

        p2 = self._resize(p4, stem_s2.shape[-2:])
        p2 = self.s2_fuse(torch.cat((p2, self.stem_proj(stem_s2)), dim=1))
        p2 = self.s2_refine(p2)

        full = self._resize(p2, output_size)
        road_logits = self.classifier(self.dropout(self.full_refine(full)))
        if self.training:
            return self.centerline_head(p4), road_logits
        return road_logits

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(module, (RepVGGBlock, RepDepthwiseBlock)):
                module.switch_to_deploy()


def _soft_erode(mask: Tensor) -> Tensor:
    vertical = -F.max_pool2d(-mask, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-mask, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(mask: Tensor) -> Tensor:
    return F.max_pool2d(mask, 3, stride=1, padding=1)


def _soft_open(mask: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(mask))


def soft_skeletonize(mask: Tensor, iterations: int = 8) -> Tensor:
    """Morphological skeleton target generation using only PyTorch ops."""
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


def binary_tversky_loss(
    probability: Tensor,
    target: Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    eps: float = 1e-6,
) -> Tensor:
    """One centerline loss; beta > alpha penalizes broken roads more."""
    probability = probability.float().flatten(1)
    target = target.float().flatten(1)
    true_positive = (probability * target).sum(dim=1)
    false_positive = (probability * (1.0 - target)).sum(dim=1)
    false_negative = ((1.0 - probability) * target).sum(dim=1)
    score = (true_positive + eps) / (
        true_positive
        + float(alpha) * false_positive
        + float(beta) * false_negative
        + eps
    )
    return (1.0 - score).mean()


class RoadSegCenterlineTverskyLoss(nn.Module):
    """Main road CE+Dice and exactly one auxiliary centerline Tversky."""

    def __init__(
        self,
        road_class_weight: float = 2.0,
        main_dice_weight: float = 1.0,
        aux_weight: float = 0.20,
        centerline_alpha: float = 0.30,
        centerline_beta: float = 0.70,
        skeleton_iterations: int = 8,
        centerline_dilation: int = 1,
        fast_centerline_target: bool = True,
    ) -> None:
        super().__init__()
        self.road_class_weight = float(road_class_weight)
        self.main_dice_weight = float(main_dice_weight)
        self.aux_weight = float(aux_weight)
        self.centerline_alpha = float(centerline_alpha)
        self.centerline_beta = float(centerline_beta)
        self.skeleton_iterations = int(skeleton_iterations)
        self.centerline_dilation = int(centerline_dilation)
        self.fast_centerline_target = bool(fast_centerline_target)

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

        with torch.no_grad():
            target_dilation = self.centerline_dilation
            if self.fast_centerline_target:
                # The auxiliary head lives at S4. Skeletonizing a 512x512 mask
                # performs roughly 40 full-resolution pooling passes for the
                # default settings. Pooling first preserves thin positives and
                # cuts this auxiliary-target cost by well over an order of
                # magnitude without changing the Tversky objective.
                target_height, target_width = centerline_logits.shape[-2:]
                scale = max(
                    road_mask.shape[-2] / max(target_height, 1),
                    road_mask.shape[-1] / max(target_width, 1),
                )
                skeleton_input = F.adaptive_max_pool2d(
                    road_mask.float(), (target_height, target_width)
                )
                target_iterations = (
                    max(1, math.ceil(self.skeleton_iterations / scale))
                    if self.skeleton_iterations > 0
                    else 0
                )
                # The CLI dilation is expressed in input pixels. Convert it
                # to the auxiliary grid so dilation=1 does not accidentally
                # become a +/-4-pixel tolerance at S4.
                target_dilation = max(
                    0, int(math.floor(self.centerline_dilation / scale + 0.5))
                )
                centerline_target = soft_skeletonize(
                    skeleton_input, target_iterations
                )
            else:
                centerline_target = soft_skeletonize(
                    road_mask.float(), self.skeleton_iterations
                )
            if target_dilation > 0:
                kernel = 2 * target_dilation + 1
                centerline_target = F.max_pool2d(
                    centerline_target,
                    kernel,
                    stride=1,
                    padding=target_dilation,
                )
            if centerline_target.shape[-2:] != centerline_logits.shape[-2:]:
                # Full-resolution compatibility/ablation path.
                centerline_target = F.adaptive_max_pool2d(
                    centerline_target, centerline_logits.shape[-2:]
                )

        loss_centerline = binary_tversky_loss(
            centerline_logits.float().sigmoid(),
            centerline_target,
            alpha=self.centerline_alpha,
            beta=self.centerline_beta,
        )
        total = (
            loss_main_ce
            + self.main_dice_weight * loss_main_dice
            + self.aux_weight * loss_centerline
        )
        return {
            "loss_total": total,
            "loss_main_ce": loss_main_ce.detach(),
            "loss_main_dice": loss_main_dice.detach(),
            "loss_aux_centerline": loss_centerline.detach(),
            "loss_centerline_tversky": loss_centerline.detach(),
        }


@torch.no_grad()
def verify_reparameterization(
    block: Union[RepVGGBlock, RepDepthwiseBlock],
    shape: Tuple[int, int, int, int],
) -> float:
    """Return max absolute output error before and after branch fusion."""
    block.eval()
    x = torch.randn(shape, device=next(block.parameters()).device)
    reference = block(x)
    block.switch_to_deploy()
    return float((reference - block(x)).abs().max())
