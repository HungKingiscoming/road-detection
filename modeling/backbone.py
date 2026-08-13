"""CoMingNet V2 backbone for binary road extraction.

The public names ``CoMingBlock`` and ``CoMingNet`` are kept compatible with
the original project.  During training each CoMingBlock uses five spatial
branches.  ``switch_to_deploy`` fuses them into one grouped convolution.

Backbone output is a feature dictionary consumed by ``coming_decoder.py``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _valid_groups(channels: int, channels_per_group: int = 32) -> int:
    groups = max(1, channels // channels_per_group)
    while channels % groups:
        groups -= 1
    return groups


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1,
                 padding=1, groups=1, act=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class GroupConvBN(nn.Module):
    def __init__(self, channels: int, kernel_size, padding, groups: int,
                 dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, padding=padding,
            dilation=dilation, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))

    def fused_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        weight = self.conv.weight
        scale = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        kernel = weight * scale[:, None, None, None]
        bias = self.bn.bias - self.bn.running_mean * scale
        return kernel, bias


def _to_dense_kernel(kernel: Tensor, dilation: Union[int, Tuple[int, int]],
                     target_size: int) -> Tensor:
    """Embed a small or dilated grouped-conv kernel into KxK coordinates."""
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    kh, kw = kernel.shape[-2:]
    effective_h = (kh - 1) * dh + 1
    effective_w = (kw - 1) * dw + 1
    if effective_h > target_size or effective_w > target_size:
        raise ValueError("Branch effective kernel exceeds deployment kernel")
    out = kernel.new_zeros(*kernel.shape[:-2], target_size, target_size)
    top = (target_size - effective_h) // 2
    left = (target_size - effective_w) // 2
    out[..., top:top + effective_h:dh, left:left + effective_w:dw] = kernel
    return out


class CoMingBlock(nn.Module):
    """Road-oriented, attention-free, re-parameterizable convolution block.

    Train: KxK + 3x3 + 1xK + Kx1 + dilated-3x3 grouped convolutions.
    Deploy: one KxK grouped convolution.  A full-channel pointwise FFN follows
    the spatial operator and prevents isolated convolution groups.
    """

    def __init__(self, channels: int, kernel_size: int = 7,
                 deploy: bool = False, expansion: float = 2.0,
                 channels_per_group: int = 32,
                 zero_init_residual: bool = False,
                 act_layer=None) -> None:
        super().__init__()
        if kernel_size not in (5, 7, 9, 11) or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be one of 5, 7, 9, 11")
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.dilation = self.padding
        self.groups = _valid_groups(channels, channels_per_group)
        self.deploy = deploy

        if deploy:
            self.reparam_spatial = nn.Conv2d(
                channels, channels, kernel_size, padding=self.padding,
                groups=self.groups, bias=True
            )
        else:
            self.branch_large = GroupConvBN(
                channels, kernel_size, self.padding, self.groups)
            self.branch_local = GroupConvBN(channels, 3, 1, self.groups)
            self.branch_horizontal = GroupConvBN(
                channels, (1, kernel_size), (0, self.padding), self.groups)
            self.branch_vertical = GroupConvBN(
                channels, (kernel_size, 1), (self.padding, 0), self.groups)
            self.branch_dilated = GroupConvBN(
                channels, 3, self.dilation, self.groups,
                dilation=self.dilation)

        hidden = max(channels, int(round(channels * expansion)))
        self.spatial_act = nn.SiLU(inplace=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        if zero_init_residual:
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            spatial = self.reparam_spatial(x)
        else:
            spatial = self.branch_large(x)
            spatial = spatial + self.branch_local(x)
            spatial = spatial + self.branch_horizontal(x)
            spatial = spatial + self.branch_vertical(x)
            spatial = spatial + self.branch_dilated(x)
        spatial = self.spatial_act(spatial)
        return x + self.ffn(spatial)

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam_spatial.weight, self.reparam_spatial.bias
        branches = (
            self.branch_large, self.branch_local, self.branch_horizontal,
            self.branch_vertical, self.branch_dilated,
        )
        kernel_sum = None
        bias_sum = None
        for branch in branches:
            kernel, bias = branch.fused_kernel_bias()
            kernel = _to_dense_kernel(
                kernel, branch.conv.dilation, self.kernel_size)
            kernel_sum = kernel if kernel_sum is None else kernel_sum + kernel
            bias_sum = bias if bias_sum is None else bias_sum + bias
        return kernel_sum, bias_sum

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(
            self.channels, self.channels, self.kernel_size,
            padding=self.padding, groups=self.groups, bias=True
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            conv.weight.copy_(kernel)
            conv.bias.copy_(bias)
        self.reparam_spatial = conv
        for name in (
            "branch_large", "branch_local", "branch_horizontal",
            "branch_vertical", "branch_dilated"
        ):
            delattr(self, name)
        self.deploy = True


class _Fusion(nn.Module):
    """Concat-convolution residual fusion; no attention or gating."""
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.merge = ConvBNAct(channels * 2, channels, 1, padding=0)
        self.refine = CoMingBlock(channels, kernel_size, expansion=2.0)

    def forward(self, base: Tensor, transferred: Tensor) -> Tensor:
        update = self.merge(torch.cat([base, transferred], dim=1))
        return self.refine(base + update)


class CoMingNet(nn.Module):
    """Dual geometry-context CoMingNet backbone.

    Geometry stays at output stride 4. Context progresses to stride 32. Two
    bilateral exchanges use concat-convolution fusion. A stride-2 stem feature
    is retained for the decoder to reconstruct thin roads and boundaries.
    """

    def __init__(self, in_channels: int = 3, channels: int = 32,
                 ppm_channels: int = 128,
                 local_blocks: Sequence[int] = (2, 2, 2),
                 global_blocks: Sequence[int] = (2, 3, 2),
                 kernel_size: int = 7, align_corners: bool = False,
                 deploy: bool = False, zero_init_residual: bool = False,
                 norm_cfg=None, act_cfg=None, init_cfg=None,
                 use_checkpoint: bool = False, **kwargs) -> None:
        super().__init__()
        if len(local_blocks) != 3 or len(global_blocks) != 3:
            raise ValueError("local_blocks/global_blocks require three values")
        c1, c2, c4, c8, c16 = (
            channels, channels * 2, channels * 4,
            channels * 8, channels * 16,
        )
        self.align_corners = align_corners
        self.use_checkpoint = use_checkpoint

        self.stem_half = nn.Sequential(
            ConvBNAct(in_channels, c1, 3, stride=2, padding=1),
            CoMingBlock(c1, 7, deploy, 2.0, 32, zero_init_residual),
        )
        self.stem_quarter = ConvBNAct(c1, c2, 3, stride=2, padding=1)

        self.local_stage1 = self._stage(c2, local_blocks[0], 9, deploy)
        self.local_stage2 = self._stage(c2, local_blocks[1], 9, deploy)
        self.local_transition = ConvBNAct(c2, c4, 3, padding=1)
        self.local_stage3 = self._stage(c4, local_blocks[2], 9, deploy)

        self.global_stage1 = self._down_stage(c2, c4, global_blocks[0], 7, deploy)
        self.global_stage2 = self._down_stage(c4, c8, global_blocks[1], 5, deploy)
        self.global_stage3 = self._down_stage(c8, c16, global_blocks[2], 5, deploy)

        self.g2l1_proj = ConvBNAct(c4, c2, 1, padding=0, act=False)
        self.l2g1_proj = ConvBNAct(c2, c4, 3, stride=2, padding=1, act=False)
        self.local_fusion1 = _Fusion(c2, 9)
        self.global_fusion1 = _Fusion(c4, 7)

        self.g2l2_proj = ConvBNAct(c8, c2, 1, padding=0, act=False)
        self.l2g2_proj = nn.Sequential(
            ConvBNAct(c2, c4, 3, stride=2, padding=1),
            ConvBNAct(c4, c8, 3, stride=2, padding=1, act=False),
        )
        self.local_fusion2 = _Fusion(c2, 9)
        self.global_fusion2 = _Fusion(c8, 5)

        # Re-parameterizable context replaces DAPPM.
        self.context = nn.Sequential(
            CoMingBlock(c16, 11, deploy, 2.0, 32, zero_init_residual),
            ConvBNAct(c16, c4, 1, padding=0),
        )
        self.final_merge = ConvBNAct(c4 * 2, c4, 1, padding=0)
        self.final_refine = nn.Sequential(
            CoMingBlock(c4, 9, deploy, 2.0, 32, zero_init_residual),
            CoMingBlock(c4, 9, deploy, 2.0, 32, zero_init_residual),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _stage(channels: int, count: int, kernel: int, deploy: bool):
        return nn.Sequential(*[
            CoMingBlock(channels, kernel, deploy, expansion=2.0)
            for _ in range(count)
        ])

    @classmethod
    def _down_stage(cls, in_ch: int, out_ch: int, count: int,
                    kernel: int, deploy: bool):
        return nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, stride=2, padding=1),
            cls._stage(out_ch, count, kernel, deploy),
        )

    def _resize(self, x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, ref.shape[-2:], mode="bilinear",
                             align_corners=self.align_corners)

    def forward(self, x: Tensor, return_aux: Optional[bool] = None,
                return_features: bool = False) -> Dict[str, Tensor]:
        stem_half = self.stem_half(x)                  # 1/2, C
        shared = self.stem_quarter(stem_half)          # 1/4, 2C

        local1_old = self.local_stage1(shared)         # 1/4, 2C
        global1_old = self.global_stage1(shared)       # 1/8, 4C
        g2l1 = self._resize(self.g2l1_proj(global1_old), local1_old)
        l2g1 = self.l2g1_proj(local1_old)
        if l2g1.shape[-2:] != global1_old.shape[-2:]:
            l2g1 = self._resize(l2g1, global1_old)
        local1 = self.local_fusion1(local1_old, g2l1)
        global1 = self.global_fusion1(global1_old, l2g1)

        local2_old = self.local_stage2(local1)         # 1/4, 2C
        global2_old = self.global_stage2(global1)      # 1/16, 8C
        g2l2 = self._resize(self.g2l2_proj(global2_old), local2_old)
        l2g2 = self.l2g2_proj(local2_old)
        if l2g2.shape[-2:] != global2_old.shape[-2:]:
            l2g2 = self._resize(l2g2, global2_old)
        local2 = self.local_fusion2(local2_old, g2l2)
        global2 = self.global_fusion2(global2_old, l2g2)

        geometry = self.local_stage3(self.local_transition(local2))  # 1/4, 4C
        global3 = self.global_stage3(global2)                         # 1/32,16C
        context = self.context(global3)                               # 1/32, 4C
        context_up = self._resize(context, geometry)
        fused = self.final_merge(torch.cat([geometry, context_up], dim=1))
        fused = self.final_refine(fused)                              # 1/4, 4C

        return {
            "stem_half": stem_half,
            "aux": local1,
            "geometry": geometry,
            "global8": global1,
            "global16": global2,
            "global32": global3,
            "context": context,
            "fused": fused,
        }

    def switch_to_deploy(self) -> "CoMingNet":
        for module in list(self.modules()):
            if isinstance(module, CoMingBlock):
                module.switch_to_deploy()
        return self


@torch.no_grad()
def verify_reparameterization(model: CoMingNet,
                              input_shape=(1, 3, 256, 256),
                              atol: float = 2e-4):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(input_shape, device=device)
    before = model(x)["fused"]
    model.switch_to_deploy()
    after = model(x)["fused"]
    error = (before - after).abs()
    if error.max().item() > atol:
        raise AssertionError(f"reparameterization max error={error.max().item():.6g}")
    return error.max().item(), error.mean().item()
