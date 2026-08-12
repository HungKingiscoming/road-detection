import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Optional, Sequence, Tuple, Union

from components.components import (
    BaseModule,
    ConvModule,
    DAPPM,
    OptConfigType,
    resize,
)


KernelSize = Union[int, Tuple[int, int]]


class DWConvBN(nn.Module):
    """Depthwise convolution followed by BatchNorm, without activation."""

    def __init__(
        self,
        channels: int,
        kernel_size: KernelSize,
        stride: KernelSize = 1,
        padding: KernelSize = 0,
        dilation: KernelSize = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(self.conv(x))

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        """Fuse the depthwise convolution and its BatchNorm."""
        kernel = self.conv.weight

        if self.conv.bias is None:
            conv_bias = torch.zeros(
                kernel.shape[0], device=kernel.device, dtype=kernel.dtype
            )
        else:
            conv_bias = self.conv.bias

        std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale = self.bn.weight / std

        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = self.bn.bias + (conv_bias - self.bn.running_mean) * scale
        return fused_kernel, fused_bias


def center_pad_kernel(kernel: Tensor, target_size: int) -> Tensor:
    """Center-pad a 3x3, 1xK, or Kx1 kernel to target_size x target_size."""
    current_height, current_width = kernel.shape[-2:]
    pad_height = target_size - current_height
    pad_width = target_size - current_width

    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            f"Cannot pad a {current_height}x{current_width} kernel "
            f"to {target_size}x{target_size}."
        )

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return F.pad(kernel, [pad_left, pad_right, pad_top, pad_bottom])


class CoMingBlock(nn.Module):
    """Road-oriented four-branch reparameterizable CNN block.

    Training spatial branches:
        DWConv 3x3 + BN       (local geometry)
        DWConv 1xK + BN       (horizontal continuity)
        DWConv Kx1 + BN       (vertical continuity)
        DWConv KxK + BN       (broad context)

    Deployment spatial branch:
        one DWConv KxK with bias

    A shared pointwise 1x1 convolution mixes channels after spatial fusion.
    The block preserves both spatial size and channel count.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        deploy: bool = False,
        act_layer: type[nn.Module] = nn.ReLU,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()

        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.deploy = deploy

        if deploy:
            self.reparam_spatial = nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=self.padding,
                groups=channels,
                bias=True,
            )
        else:
            self.branch_local = DWConvBN(channels, 3, padding=1)
            self.branch_horizontal = DWConvBN(
                channels, (1, kernel_size), padding=(0, self.padding)
            )
            self.branch_vertical = DWConvBN(
                channels, (kernel_size, 1), padding=(self.padding, 0)
            )
            self.branch_context = DWConvBN(
                channels, kernel_size, padding=self.padding
            )

        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = act_layer(inplace=True)

        if zero_init_residual:
            nn.init.zeros_(self.channel_mixer[1].weight)
            nn.init.zeros_(self.channel_mixer[1].bias)

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

        return self.act(x + self.channel_mixer(spatial))

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam_spatial.weight, self.reparam_spatial.bias

        kernel_local, bias_local = self.branch_local.get_equivalent_kernel_bias()
        kernel_h, bias_h = self.branch_horizontal.get_equivalent_kernel_bias()
        kernel_v, bias_v = self.branch_vertical.get_equivalent_kernel_bias()
        kernel_context, bias_context = (
            self.branch_context.get_equivalent_kernel_bias()
        )

        kernel_local = center_pad_kernel(kernel_local, self.kernel_size)
        kernel_h = center_pad_kernel(kernel_h, self.kernel_size)
        kernel_v = center_pad_kernel(kernel_v, self.kernel_size)

        equivalent_kernel = kernel_context + kernel_local + kernel_h + kernel_v
        equivalent_bias = bias_context + bias_local + bias_h + bias_v
        return equivalent_kernel, equivalent_bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_spatial = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)

        with torch.no_grad():
            reparam_spatial.weight.copy_(kernel)
            reparam_spatial.bias.copy_(bias)

        self.reparam_spatial = reparam_spatial
        del self.branch_local
        del self.branch_horizontal
        del self.branch_vertical
        del self.branch_context
        self.deploy = True


class CoMingNet(BaseModule):
    """Global-local CNN backbone for road extraction.

    Resolution schedule relative to the input image:
        shared stem:       1/4,  2C
        local branch:      1/4,  2C -> 4C
        global stage 1:    1/8,  4C
        global stage 2:   1/16,  8C
        global stage 3:   1/32, 16C
        DAPPM output:     1/32,  4C
        fused output:      1/4,  4C

    Bilateral fusion is performed at global resolutions 1/8 and 1/16.

    Default output follows the GCNet-style contract:
        training:  (aux_feature, fused_feature)
        inference: fused_feature

    Set ``return_features=True`` to receive a feature dictionary in either mode.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        ppm_channels: int = 128,
        local_blocks: Sequence[int] = (2, 2, 2),
        global_blocks: Sequence[int] = (2, 3, 2),
        kernel_size: int = 7,
        align_corners: bool = False,
        norm_cfg: OptConfigType = dict(type="BN", requires_grad=True),
        act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
        init_cfg: OptConfigType = None,
        deploy: bool = False,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__(init_cfg)

        if len(local_blocks) != 3 or len(global_blocks) != 3:
            raise ValueError("local_blocks and global_blocks must contain 3 values.")
        if any(n < 1 for n in (*local_blocks, *global_blocks)):
            raise ValueError("Every stage must contain at least one block.")

        self.in_channels = in_channels
        self.channels = channels
        self.ppm_channels = ppm_channels
        self.kernel_size = kernel_size
        self.align_corners = align_corners
        self.deploy = deploy

        c1, c2, c4, c8, c16 = (
            channels,
            channels * 2,
            channels * 4,
            channels * 8,
            channels * 16,
        )

        # Shared shallow stem: RGB -> 1/4 feature.
        self.stem = nn.Sequential(
            ConvModule(
                in_channels,
                c1,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                c1,
                c2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        # Local branch stays at output stride 4.
        self.local_stage1 = self._make_coming_stage(
            c2, local_blocks[0], kernel_size, deploy, zero_init_residual
        )
        self.local_stage2 = self._make_coming_stage(
            c2, local_blocks[1], kernel_size, deploy, zero_init_residual
        )
        self.local_transition = ConvModule(
            c2,
            c4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.local_stage3 = self._make_coming_stage(
            c4, local_blocks[2], kernel_size, deploy, zero_init_residual
        )

        # Global branch progressively downsamples to output stride 32.
        self.global_stage1 = self._make_global_stage(
            c2,
            c4,
            global_blocks[0],
            kernel_size,
            norm_cfg,
            act_cfg,
            deploy,
            zero_init_residual,
        )
        self.global_stage2 = self._make_global_stage(
            c4,
            c8,
            global_blocks[1],
            kernel_size,
            norm_cfg,
            act_cfg,
            deploy,
            zero_init_residual,
        )
        self.global_stage3 = self._make_global_stage(
            c8,
            c16,
            global_blocks[2],
            kernel_size,
            norm_cfg,
            act_cfg,
            deploy,
            zero_init_residual,
        )

        # Bilateral fusion 1: local 1/4 <-> global 1/8.
        self.global_to_local1 = ConvModule(
            c4,
            c2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.local_to_global1 = ConvModule(
            c2,
            c4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        # Bilateral fusion 2: local 1/4 <-> global 1/16.
        self.global_to_local2 = ConvModule(
            c8,
            c2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.local_to_global2 = nn.Sequential(
            ConvModule(
                c2,
                c4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                c4,
                c8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
        )

        # Multi-scale global context and final local-global fusion.
        self.context = DAPPM(
            c16,
            ppm_channels,
            c4,
            num_scales=5,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.local_projection = ConvModule(
            c4,
            c4,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.final_refine = CoMingBlock(
            c4,
            kernel_size=kernel_size,
            deploy=deploy,
            zero_init_residual=zero_init_residual,
        )

        if init_cfg is None:
            self.apply(self._init_module_weights)

        # Parent-level initialization above resets all BN scales to one, so
        # apply optional residual zero-initialization afterwards.
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, CoMingBlock):
                    nn.init.zeros_(module.channel_mixer[1].weight)
                    nn.init.zeros_(module.channel_mixer[1].bias)

    @staticmethod
    def _make_coming_stage(
        stage_channels: int,
        num_blocks: int,
        kernel_size: int,
        deploy: bool,
        zero_init_residual: bool,
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                CoMingBlock(
                    stage_channels,
                    kernel_size=kernel_size,
                    deploy=deploy,
                    zero_init_residual=zero_init_residual,
                )
                for _ in range(num_blocks)
            ]
        )

    @classmethod
    def _make_global_stage(
        cls,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int,
        norm_cfg: OptConfigType,
        act_cfg: OptConfigType,
        deploy: bool,
        zero_init_residual: bool,
    ) -> nn.Sequential:
        return nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            cls._make_coming_stage(
                out_channels,
                num_blocks,
                kernel_size,
                deploy,
                zero_init_residual,
            ),
        )

    @staticmethod
    def _init_module_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _resize(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        return resize(
            x,
            size=size,
            mode="bilinear",
            align_corners=self.align_corners,
        )

    def forward(
        self,
        x: Tensor,
        return_aux: Optional[bool] = None,
        return_features: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tensor],
        Dict[str, Optional[Tensor]],
    ]:
        use_aux = self.training if return_aux is None else return_aux

        shared = self.stem(x)  # 1/4, 2C

        # Stage 1 and simultaneous bilateral fusion.
        local1_old = self.local_stage1(shared)  # 1/4, 2C
        global1_old = self.global_stage1(shared)  # 1/8, 4C

        global1_to_local = self._resize(
            self.global_to_local1(global1_old), local1_old.shape[-2:]
        )
        local1_to_global = self.local_to_global1(local1_old)
        if local1_to_global.shape[-2:] != global1_old.shape[-2:]:
            local1_to_global = self._resize(
                local1_to_global, global1_old.shape[-2:]
            )

        local1 = local1_old + global1_to_local
        global1 = global1_old + local1_to_global
        aux_feature = local1 if use_aux else None

        # Stage 2 and simultaneous bilateral fusion.
        local2_old = self.local_stage2(local1)  # 1/4, 2C
        global2_old = self.global_stage2(global1)  # 1/16, 8C

        global2_to_local = self._resize(
            self.global_to_local2(global2_old), local2_old.shape[-2:]
        )
        local2_to_global = self.local_to_global2(local2_old)
        if local2_to_global.shape[-2:] != global2_old.shape[-2:]:
            local2_to_global = self._resize(
                local2_to_global, global2_old.shape[-2:]
            )

        local2 = local2_old + global2_to_local
        global2 = global2_old + local2_to_global

        # Final local and global stages.
        local3 = self.local_stage3(self.local_transition(local2))  # 1/4, 4C
        global3 = self.global_stage3(global2)  # 1/32, 16C
        global_context = self.context(global3)  # 1/32, 4C
        global_up = self._resize(global_context, local3.shape[-2:])

        fused = self.final_refine(self.local_projection(local3) + global_up)

        if return_features:
            return {
                "shared": shared,
                "aux": aux_feature,
                "local": local3,
                "global": global3,
                "context": global_context,
                "fused": fused,
            }

        if use_aux:
            assert aux_feature is not None
            return aux_feature, fused
        return fused

    def switch_to_deploy(self) -> "CoMingNet":
        """Convert every CoMingBlock spatial graph to one DWConv KxK."""
        blocks = [m for m in self.modules() if isinstance(m, CoMingBlock)]
        for block in blocks:
            block.switch_to_deploy()
        self.deploy = True
        return self


def verify_reparameterization(
    model: CoMingNet,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    atol: float = 1e-4,
) -> Tuple[float, float]:
    """Compare backbone output immediately before and after conversion."""
    model.eval()
    parameter = next(model.parameters())
    x = torch.randn(input_shape, device=parameter.device, dtype=parameter.dtype)

    with torch.no_grad():
        before = model(x, return_aux=False)
        model.switch_to_deploy()
        after = model(x, return_aux=False)

    if not isinstance(before, Tensor) or not isinstance(after, Tensor):
        raise TypeError("Expected tensor outputs while verifying deployment.")

    error = (before - after).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()

    if max_error > atol:
        raise AssertionError(
            f"Reparameterization error {max_error:.6g} exceeds atol={atol}."
        )
    return max_error, mean_error
