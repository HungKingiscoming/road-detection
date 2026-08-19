"""Two-stream CoMingNet backbone for road extraction.

The local stream stays at output stride 4 while the global stream reaches
stride 8/16. Exactly two additive bilateral fusions exchange geometry and
context. ``CoMingBlock`` is a partial standard-convolution block: only a
fraction of channels enters its multi-branch spatial operator, and all spatial
branches are fused exactly into one dense KxK convolution for deployment.
There is no attention or transformer operation.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


class RepConvBN(nn.Module):
    """Dense convolution followed by BN for a reparameterizable branch."""

    def __init__(
        self,
        channels: int,
        kernel_size: int | Tuple[int, int],
        padding: int | Tuple[int, int],
        dilation: int | Tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=1,
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


class IdentityBN(nn.Module):
    """Identity branch with BN, convertible to a dense 1x1 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x)

    def equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        dtype = self.bn.weight.dtype
        device = self.bn.weight.device
        kernel = torch.zeros(
            self.channels, self.channels, 1, 1, dtype=dtype, device=device
        )
        index = torch.arange(self.channels, device=device)
        kernel[index, index, 0, 0] = 1.0
        std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale = self.bn.weight / std
        kernel = kernel * scale.reshape(-1, 1, 1, 1)
        bias = self.bn.bias - self.bn.running_mean * scale
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


def _expand_dilated_kernel(
    kernel: Tensor,
    dilation: int,
    target_size: int,
) -> Tensor:
    """Materialize a dilated kernel as a dense target-size kernel."""
    height, width = kernel.shape[-2:]
    effective_h = (height - 1) * dilation + 1
    effective_w = (width - 1) * dilation + 1
    if effective_h > target_size or effective_w > target_size:
        raise ValueError("Dilated branch is larger than the deploy kernel")
    expanded = kernel.new_zeros(
        kernel.shape[0], kernel.shape[1], target_size, target_size
    )
    offset_h = (target_size - effective_h) // 2
    offset_w = (target_size - effective_w) // 2
    expanded[
        :, :, offset_h : offset_h + effective_h : dilation,
        offset_w : offset_w + effective_w : dilation,
    ] = kernel
    return expanded


def _channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch, channels, height, width = x.shape
    if groups <= 1 or channels % groups != 0:
        return x
    x = x.reshape(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().reshape(batch, channels, height, width)


class CoMingBlock(nn.Module):
    """Partial reparameterized spatial block; public name is preserved.

    Only ``spatial_ratio`` of the channels is processed by dense convolutions.
    Training uses local 3x3, dilated 3x3, horizontal 1xK, vertical Kx1 and
    identity-BN branches. At deployment they become one standard KxK conv.
    Bypass channels preserve cheap detail; shuffle and pointwise expansion mix
    processed and bypass channels.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        expansion: float = 1.0,
        spatial_ratio: float = 0.25,
        shuffle_groups: int = 4,
        deploy: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 5 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 5")
        if expansion < 1.0:
            raise ValueError("expansion must be >= 1.0")
        if not 0.0 < spatial_ratio <= 1.0:
            raise ValueError("spatial_ratio must be in (0, 1]")

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.expansion = float(expansion)
        self.spatial_ratio = float(spatial_ratio)
        self.shuffle_groups = int(shuffle_groups)
        self.deploy = deploy
        processed = int(round(channels * self.spatial_ratio))
        self.processed_channels = min(channels, max(1, processed))
        self.context_dilation = (kernel_size - 1) // 2
        hidden_channels = max(
            channels,
            int(round(channels * self.expansion / 8.0)) * 8,
        )
        self.hidden_channels = hidden_channels

        if deploy:
            self.reparam_spatial = nn.Conv2d(
                self.processed_channels,
                self.processed_channels,
                kernel_size,
                padding=self.padding,
                groups=1,
                bias=True,
            )
        else:
            p = self.processed_channels
            self.branch_local = RepConvBN(p, 3, 1)
            self.branch_dilated = RepConvBN(
                p, 3, self.context_dilation, dilation=self.context_dilation
            )
            self.branch_horizontal = RepConvBN(
                p, (1, kernel_size), (0, self.padding)
            )
            self.branch_vertical = RepConvBN(
                p, (kernel_size, 1), (self.padding, 0)
            )
            self.branch_identity = IdentityBN(p)

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
        spatial_input = x[:, : self.processed_channels]
        bypass = x[:, self.processed_channels :]
        if self.deploy:
            spatial = self.reparam_spatial(spatial_input)
        else:
            spatial = (
                self.branch_local(spatial_input)
                + self.branch_dilated(spatial_input)
                + self.branch_horizontal(spatial_input)
                + self.branch_vertical(spatial_input)
                + self.branch_identity(spatial_input)
            )
        spatial = self.activation(spatial)
        mixed_input = torch.cat((spatial, bypass), dim=1) if bypass.numel() else spatial
        mixed_input = _channel_shuffle(mixed_input, self.shuffle_groups)
        return self.activation(x + self.channel_mixer(mixed_input))

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
            self.branch_dilated,
            self.branch_horizontal,
            self.branch_vertical,
            self.branch_identity,
        )
        kernels, biases = zip(
            *(branch.equivalent_kernel_bias() for branch in branches)
        )
        kernel = (
            _center_pad(kernels[0], self.kernel_size)
            + _expand_dilated_kernel(
                kernels[1], self.context_dilation, self.kernel_size
            )
            + _center_pad(kernels[2], self.kernel_size)
            + _center_pad(kernels[3], self.kernel_size)
            + _center_pad(kernels[4], self.kernel_size)
        )
        bias = sum(biases)
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.processed_channels,
            self.processed_channels,
            self.kernel_size,
            padding=self.padding,
            groups=1,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.reparam_spatial = reparam
        del self.branch_local
        del self.branch_dilated
        del self.branch_horizontal
        del self.branch_vertical
        del self.branch_identity
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
        spatial_ratio: float = 0.5,
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
            spatial_ratio=spatial_ratio,
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


class _CoMingNetBase(nn.Module):
    """Balanced two-stream backbone.

    Feature contract returned to the decoder:
        s2:  shallow edge/detail feature, 1/2 resolution, C channels
        s4:  local geometry, 1/4 resolution, 4C channels
        s8:  mid-level context, 1/8 resolution, 4C channels
        s16: pyramid context, 1/16 resolution, 4C channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 40,
        local_blocks: Sequence[int] = (2, 2, 2),
        global_blocks: Sequence[int] = (3, 4),
        highres_kernel_size: int = 5,
        context_kernel_size: int = 7,
        local_expansion: float = 1.5,
        global_expansion: float = 2.0,
        local_spatial_ratio: float = 0.25,
        global_spatial_ratio: float = 0.5,
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
            c2, local_blocks[0], highres_kernel_size, local_expansion,
            local_spatial_ratio, deploy
        )
        self.global_stage1 = nn.Sequential(
            ConvBNAct(c2, c4, 3, stride=2),
            self._stage(
                c4, global_blocks[0], highres_kernel_size, global_expansion,
                global_spatial_ratio, deploy
            ),
        )

        # Bilateral fusion 1: s4 <-> s8.
        self.g2l1_proj = ConvBNAct(c4, c2, 1, padding=0, activation=False)
        self.l2g1_proj = ConvBNAct(c2, c4, 3, stride=2, activation=False)
        self.local_fusion1 = nn.ReLU(inplace=True)
        self.global_fusion1 = nn.ReLU(inplace=True)

        self.local_stage2 = self._stage(
            c2, local_blocks[1], highres_kernel_size, local_expansion,
            local_spatial_ratio, deploy
        )
        self.global_stage2 = nn.Sequential(
            ConvBNAct(c4, c8, 3, stride=2),
            self._stage(
                c8, global_blocks[1], context_kernel_size, global_expansion,
                global_spatial_ratio, deploy
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
            c4, local_blocks[2], highres_kernel_size, local_expansion,
            local_spatial_ratio, deploy
        )
        self.context = CompactPyramidContext(
            c8,
            c4,
            branch_channels=channels,
            kernel_size=context_kernel_size,
            expansion=global_expansion,
            spatial_ratio=global_spatial_ratio,
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
        spatial_ratio: float,
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
                    spatial_ratio=spatial_ratio,
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
        stem_half = self.stem_half(x)
        shared = self.stem_quarter(stem_half)

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
        return {
            "s2": stem_half,
            "s4": local3,
            "s8": global1,
            "s16": context,
        }

    def switch_to_deploy(self) -> "CoMingNet":
        for module in list(self.modules()):
            if isinstance(module, CoMingBlock):
                module.switch_to_deploy()
        self.deploy = True
        return self


class DeepPyramidContext(nn.Module):
    """Cascaded CNN pyramid context used on the new output-stride-32 stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_channels: int,
        kernel_size: int,
        expansion: float,
        spatial_ratio: float,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.pool_sizes = (1, 2, 4, 8)
        self.shortcut = ConvBNAct(
            in_channels, out_channels, 1, padding=0, activation=False
        )
        self.seed = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=True),
            nn.ReLU(inplace=True),
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
        self.process = nn.ModuleList(
            [ConvBNAct(branch_channels, branch_channels, 3) for _ in self.pool_sizes]
        )
        merged_channels = out_channels + len(self.pool_sizes) * branch_channels
        self.fuse = ConvBNAct(merged_channels, out_channels, 1, padding=0)
        self.refine = CoMingBlock(
            out_channels,
            kernel_size=kernel_size,
            expansion=expansion,
            spatial_ratio=spatial_ratio,
            deploy=deploy,
            zero_init_residual=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        previous = self.seed(x)
        pyramid = []
        for pool_size, projection, process in zip(
            self.pool_sizes, self.pool_projections, self.process
        ):
            pooled = projection(F.adaptive_avg_pool2d(x, pool_size))
            pooled = F.interpolate(
                pooled, size=size, mode="bilinear", align_corners=False
            )
            previous = process(pooled + previous)
            pyramid.append(previous)
        shortcut = self.shortcut(x)
        fused = self.fuse(torch.cat([shortcut, *pyramid], dim=1))
        return self.refine(shortcut + fused)


class CoMingNet(_CoMingNetBase):
    """CNN-only CoMingNet with local OS4 and semantic OS8/16/32 streams.

    Existing stage names and public class name are preserved so checkpoints
    from the earlier OS16 model can initialize all compatible backbone layers
    with ``--resume_mode transfer``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 40,
        local_blocks: Sequence[int] = (2, 2, 2),
        global_blocks: Sequence[int] = (3, 4),
        highres_kernel_size: int = 5,
        context_kernel_size: int = 7,
        local_expansion: float = 1.5,
        global_expansion: float = 2.0,
        local_spatial_ratio: float = 0.5,
        global_spatial_ratio: float = 0.5,
        deep_blocks: int = 2,
        deep_spatial_ratio: float = 0.75,
        kernel_size: int | None = None,
        deploy: bool = False,
        zero_init_residual: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            local_blocks=local_blocks,
            global_blocks=global_blocks,
            highres_kernel_size=highres_kernel_size,
            context_kernel_size=context_kernel_size,
            local_expansion=local_expansion,
            global_expansion=global_expansion,
            local_spatial_ratio=local_spatial_ratio,
            global_spatial_ratio=global_spatial_ratio,
            kernel_size=kernel_size,
            deploy=deploy,
            zero_init_residual=False,
            **kwargs,
        )
        if deep_blocks < 1:
            raise ValueError("deep_blocks must be >= 1")
        if kernel_size is not None:
            context_kernel_size = kernel_size

        c4, c8 = channels * 4, channels * 8
        # The old context module is replaced by deeper semantic processing.
        del self.context
        self.s16_proj = ConvBNAct(c8, c4, 1, padding=0)
        self.deep_downsample = ConvBNAct(c8, c8, 3, stride=2)
        self.deep_stage = self._stage(
            c8,
            deep_blocks,
            context_kernel_size,
            global_expansion,
            deep_spatial_ratio,
            deploy,
        )
        self.deep_context = DeepPyramidContext(
            c8,
            c4,
            branch_channels=channels,
            kernel_size=context_kernel_size,
            expansion=global_expansion,
            spatial_ratio=max(0.5, global_spatial_ratio),
            deploy=deploy,
        )
        self._initialize()
        if zero_init_residual:
            self.zero_init_residuals()

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        stem_half = self.stem_half(x)
        shared = self.stem_quarter(stem_half)

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
        semantic16 = self.s16_proj(global2)
        semantic32 = self.deep_context(
            self.deep_stage(self.deep_downsample(global2))
        )
        return {
            "s2": stem_half,
            "s4": local3,
            "s8": global1,
            "s16": semantic16,
            "s32": semantic32,
        }


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


__all__ = ["CoMingBlock", "CoMingNet", "verify_reparameterization"]
