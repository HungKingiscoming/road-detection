"""CoMingNet balanced backbone for Massachusetts Roads.

Pure CNN, no attention and no transformer.  The network keeps a high-resolution
local stream (output stride 4), a compact context stream (output stride 8/16),
and performs exactly two bilateral fusions.  CoMingBlock is reparameterizable:
four depthwise branches used during training become one depthwise KxK branch
for deployment.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _pair(value: int | Tuple[int, int]) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
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


class DWConvBN(nn.Module):
    """Depthwise convolution followed by BN; used by CoMingBlock branches."""

    def __init__(
        self,
        channels: int,
        kernel_size: int | Tuple[int, int],
        padding: int | Tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))

    def equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        weight = self.conv.weight
        conv_bias = torch.zeros(
            weight.shape[0], device=weight.device, dtype=weight.dtype
        )
        std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale = self.bn.weight / std
        kernel = weight * scale.reshape(-1, 1, 1, 1)
        bias = self.bn.bias + (conv_bias - self.bn.running_mean) * scale
        return kernel, bias


def _center_pad(kernel: Tensor, target_size: int) -> Tensor:
    height, width = kernel.shape[-2:]
    if height > target_size or width > target_size:
        raise ValueError(f"Cannot pad {height}x{width} to {target_size}x{target_size}")
    pad_h = target_size - height
    pad_w = target_size - width
    return F.pad(
        kernel,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
    )


class CoMingBlock(nn.Module):
    """Road-oriented reparameterizable convolution block.

    Training branches: 3x3, 1xK, Kx1 and KxK depthwise convolutions.
    Deployment branch: one KxK depthwise convolution.  An inverted pointwise
    MLP expands channels, applies a non-linearity, then projects back to C.
    The public name is intentionally preserved.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        expansion: float = 1.0,
        deploy: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if expansion < 1.0:
            raise ValueError("expansion must be >= 1.0")

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.expansion = float(expansion)
        self.deploy = deploy
        hidden_channels = max(
            channels,
            int(round(channels * self.expansion / 8.0)) * 8,
        )
        self.hidden_channels = hidden_channels

        if deploy:
            self.reparam_spatial = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=self.padding,
                groups=channels,
                bias=True,
            )
        else:
            self.branch_local = DWConvBN(channels, 3, 1)
            self.branch_horizontal = DWConvBN(
                channels, (1, kernel_size), (0, self.padding)
            )
            self.branch_vertical = DWConvBN(
                channels, (kernel_size, 1), (self.padding, 0)
            )
            self.branch_context = DWConvBN(
                channels, kernel_size, self.padding
            )

        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)
        if zero_init_residual:
            self.zero_init_residual()

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            spatial = self.reparam_spatial(x)
        else:
            spatial = (
                self.branch_local(x)
                + self.branch_horizontal(x)
                + self.branch_vertical(x)
                + self.branch_context(x)
            )
        return self.activation(x + self.channel_mixer(spatial))

    def zero_init_residual(self) -> None:
        """Start the residual path at zero without touching other BN layers."""
        project_bn = self.channel_mixer[-1]
        if not isinstance(project_bn, nn.BatchNorm2d):
            raise TypeError("CoMingBlock channel mixer must end with BatchNorm2d")
        nn.init.zeros_(project_bn.weight)
        nn.init.zeros_(project_bn.bias)

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam_spatial.weight, self.reparam_spatial.bias

        branches = (
            self.branch_local,
            self.branch_horizontal,
            self.branch_vertical,
            self.branch_context,
        )
        kernels, biases = zip(
            *(branch.equivalent_kernel_bias() for branch in branches)
        )
        kernel = sum(_center_pad(item, self.kernel_size) for item in kernels)
        bias = sum(biases)
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=self.padding,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.reparam_spatial = reparam
        del self.branch_local
        del self.branch_horizontal
        del self.branch_vertical
        del self.branch_context
        self.deploy = True


class CompactPyramidContext(nn.Module):
    """Small CNN pyramid context at output stride 16.

    Adaptive pooling gives image-level and regional context without a
    quadratic attention map.  Pooled branches intentionally avoid BN because
    the 1x1 branch is unsafe for small batches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_channels: int = 32,
        kernel_size: int = 7,
        expansion: float = 2.0,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.pool_sizes = (1, 2, 4)
        self.pool_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 1, bias=True),
                    nn.ReLU(inplace=True),
                )
                for _ in self.pool_sizes
            ]
        )
        merged_channels = in_channels + len(self.pool_sizes) * branch_channels
        self.fuse = ConvBNAct(merged_channels, out_channels, 1, padding=0)
        self.shortcut = ConvBNAct(
            in_channels, out_channels, 1, padding=0, activation=False
        )
        self.refine = CoMingBlock(
            out_channels,
            kernel_size=kernel_size,
            expansion=expansion,
            deploy=deploy,
        )

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        features = [x]
        for size, projection in zip(self.pool_sizes, self.pool_projections):
            pooled = F.adaptive_avg_pool2d(x, size)
            pooled = projection(pooled)
            features.append(
                F.interpolate(
                    pooled,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.refine(self.fuse(torch.cat(features, dim=1)) + self.shortcut(x))


class CoMingNet(nn.Module):
    """Balanced two-stream backbone.

    Feature contract returned to the decoder:
        s4:  local geometry, 1/4 resolution, 4C channels
        s8:  mid-level context, 1/8 resolution, 4C channels
        s16: pyramid context, 1/16 resolution, 4C channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 48,
        local_blocks: Sequence[int] = (2, 2, 2),
        global_blocks: Sequence[int] = (2, 3),
        highres_kernel_size: int = 5,
        context_kernel_size: int = 7,
        local_expansion: float = 1.5,
        global_expansion: float = 2.0,
        kernel_size: int | None = None,
        deploy: bool = False,
        zero_init_residual: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if len(local_blocks) != 3:
            raise ValueError("local_blocks must have three integers")
        if len(global_blocks) not in (2, 3):
            raise ValueError("global_blocks must have two integers (three is accepted for compatibility)")
        if kernel_size is not None:
            context_kernel_size = kernel_size
        global_blocks = tuple(global_blocks[:2])

        c1, c2, c4, c8 = channels, channels * 2, channels * 4, channels * 8

        self.stem_half = nn.Sequential(
            ConvBNAct(in_channels, c1, 3, stride=2),
            ConvBNAct(c1, c1, 3),
        )
        self.stem_quarter = ConvBNAct(c1, c2, 3, stride=2)

        self.local_stage1 = self._stage(
            c2, local_blocks[0], highres_kernel_size, local_expansion, deploy
        )
        self.global_stage1 = nn.Sequential(
            ConvBNAct(c2, c4, 3, stride=2),
            self._stage(
                c4, global_blocks[0], highres_kernel_size, global_expansion, deploy
            ),
        )

        # Bilateral fusion 1: s4 <-> s8.
        self.g2l1_proj = ConvBNAct(c4, c2, 1, padding=0, activation=False)
        self.l2g1_proj = ConvBNAct(c2, c4, 3, stride=2, activation=False)
        self.local_fusion1 = nn.ReLU(inplace=True)
        self.global_fusion1 = nn.ReLU(inplace=True)

        self.local_stage2 = self._stage(
            c2, local_blocks[1], highres_kernel_size, local_expansion, deploy
        )
        self.global_stage2 = nn.Sequential(
            ConvBNAct(c4, c8, 3, stride=2),
            self._stage(
                c8, global_blocks[1], context_kernel_size, global_expansion, deploy
            ),
        )

        # Bilateral fusion 2: s4 <-> s16.
        self.g2l2_proj = ConvBNAct(c8, c2, 1, padding=0, activation=False)
        self.l2g2_proj = nn.Sequential(
            ConvBNAct(c2, c4, 3, stride=2),
            ConvBNAct(c4, c8, 3, stride=2, activation=False),
        )
        self.local_fusion2 = nn.ReLU(inplace=True)
        self.global_fusion2 = nn.ReLU(inplace=True)

        self.local_transition = ConvBNAct(c2, c4, 3)
        self.local_stage3 = self._stage(
            c4, local_blocks[2], highres_kernel_size, local_expansion, deploy
        )
        self.context = CompactPyramidContext(
            c8,
            c4,
            branch_channels=channels,
            kernel_size=context_kernel_size,
            expansion=global_expansion,
            deploy=deploy,
        )
        self.deploy = deploy
        self._initialize()
        if zero_init_residual:
            self.zero_init_residuals()

    @staticmethod
    def _stage(
        channels: int,
        blocks: int,
        kernel_size: int,
        expansion: float,
        deploy: bool,
    ) -> nn.Sequential:
        if blocks < 1:
            raise ValueError("Each stage needs at least one CoMingBlock")
        return nn.Sequential(
            *[
                CoMingBlock(
                    channels,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    deploy=deploy,
                    zero_init_residual=False,
                )
                for _ in range(blocks)
            ]
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

    def zero_init_residuals(self) -> int:
        count = 0
        for module in self.modules():
            if isinstance(module, CoMingBlock):
                module.zero_init_residual()
                count += 1
        return count

    @staticmethod
    def _resize_like(source: Tensor, target: Tensor) -> Tensor:
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        shared = self.stem_quarter(self.stem_half(x))

        local1_old = self.local_stage1(shared)
        global1_old = self.global_stage1(shared)
        local1 = self.local_fusion1(
            local1_old + self._resize_like(self.g2l1_proj(global1_old), local1_old)
        )
        global1 = self.global_fusion1(global1_old + self.l2g1_proj(local1_old))

        local2_old = self.local_stage2(local1)
        global2_old = self.global_stage2(global1)
        local2 = self.local_fusion2(
            local2_old + self._resize_like(self.g2l2_proj(global2_old), local2_old)
        )
        local_to_global2 = self.l2g2_proj(local2_old)
        if local_to_global2.shape[-2:] != global2_old.shape[-2:]:
            local_to_global2 = self._resize_like(local_to_global2, global2_old)
        global2 = self.global_fusion2(global2_old + local_to_global2)

        local3 = self.local_stage3(self.local_transition(local2))
        context = self.context(global2)
        return {"s4": local3, "s8": global1, "s16": context}

    def switch_to_deploy(self) -> "CoMingNet":
        for module in list(self.modules()):
            if isinstance(module, CoMingBlock):
                module.switch_to_deploy()
        self.deploy = True
        return self


@torch.no_grad()
def verify_reparameterization(
    model: CoMingNet,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    atol: float = 2e-4,
) -> Tuple[float, float]:
    """Check all backbone feature maps before/after branch fusion."""
    model.eval()
    parameter = next(model.parameters())
    sample = torch.randn(input_shape, device=parameter.device, dtype=parameter.dtype)
    before = model(sample)
    model.switch_to_deploy()
    after = model(sample)
    errors = torch.cat(
        [(before[key] - after[key]).abs().flatten() for key in before]
    )
    max_error = float(errors.max())
    mean_error = float(errors.mean())
    if max_error > atol:
        raise AssertionError(
            f"Reparameterization error {max_error:.6g} exceeds atol={atol}"
        )
    return max_error, mean_error


__all__ = ["CoMingBlock", "CoMingNet", "verify_reparamete
