import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple, Union, List

from components.components import (
    BaseModule,
    ConvModule,
    DAPPM,
    build_norm_layer,
    build_activation_layer,
    resize,
    OptConfigType,
)


class GCNet(BaseModule):
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_blocks_per_stage: List = [4, 4, [5, 4], [5, 4], [2, 2]],
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 deploy: bool = False):
        super().__init__(init_cfg)

        self.in_channels          = in_channels
        self.channels             = channels
        self.ppm_channels         = ppm_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.align_corners        = align_corners
        self.norm_cfg             = norm_cfg
        self.act_cfg              = act_cfg
        self.deploy               = deploy

        # ------------------------------------------------------------------ #
        # Stage 1 — Thay FoggyAwareNorm bằng ConvModule (sử dụng norm_cfg)   #
        # ------------------------------------------------------------------ #
        self.stem_conv1 = ConvModule(
            in_channels, channels, kernel_size=3, stride=2, padding=1,
            bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.stem_conv2 = ConvModule(
            channels, channels, kernel_size=3, stride=2, padding=1,
            bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.stem_stage2 = nn.Sequential(
            *[GCBlock(channels, channels, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[0])]
        )

        self.stem_stage3 = nn.Sequential(
            GCBlock(channels, channels * 2, stride=2,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(channels * 2, channels * 2, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[1] - 1)]
        )

        self.relu = build_activation_layer(act_cfg)

        # ------------------------------------------------------------------ #
        # Semantic branch (stage 4 → 6)                                      #
        # ------------------------------------------------------------------ #
        self.semantic_branch_layers = nn.ModuleList()

        self.semantic_branch_layers.append(nn.Sequential(
            GCBlock(channels * 2, channels * 4, stride=2,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(channels * 4, channels * 4, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[2][0] - 2)],
            GCBlock(channels * 4, channels * 4, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        self.semantic_branch_layers.append(nn.Sequential(
            GCBlock(channels * 4, channels * 8, stride=2,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(channels * 8, channels * 8, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[3][0] - 2)],
            GCBlock(channels * 8, channels * 8, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        self.semantic_branch_layers.append(nn.Sequential(
            GCBlock(channels * 8, channels * 16, stride=2,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(channels * 16, channels * 16, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[4][0] - 2)],
            GCBlock(channels * 16, channels * 16, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        # ------------------------------------------------------------------ #
        # Bilateral fusion                                                   #
        # ------------------------------------------------------------------ #
        self.compression_1 = ConvModule(
            channels * 4, channels * 2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2, channels * 4, kernel_size=3, stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8, channels * 2, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(channels * 2, channels * 4,
                       kernel_size=3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels * 4, channels * 8,
                       kernel_size=3, stride=2, padding=1,
                       norm_cfg=norm_cfg, act_cfg=None),
        )

        # ------------------------------------------------------------------ #
        # Detail branch (stage 4 → 6)                                        #
        # ------------------------------------------------------------------ #
        self.detail_branch_layers = nn.ModuleList()

        self.detail_branch_layers.append(nn.Sequential(
            *[GCBlock(channels * 2, channels * 2, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[2][1] - 1)],
            GCBlock(channels * 2, channels * 2, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        self.detail_branch_layers.append(nn.Sequential(
            *[GCBlock(channels * 2, channels * 2, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[3][1] - 1)],
            GCBlock(channels * 2, channels * 2, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        self.detail_branch_layers.append(nn.Sequential(
            GCBlock(channels * 2, channels * 4, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy),
            *[GCBlock(channels * 4, channels * 4, stride=1,
                      norm_cfg=norm_cfg, act_cfg=act_cfg, deploy=deploy)
              for _ in range(num_blocks_per_stage[4][1] - 2)],
            GCBlock(channels * 4, channels * 4, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, act=False, deploy=deploy),
        ))

        self.spp = DAPPM(
            channels * 16, ppm_channels, channels * 4,
            num_scales=5,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.kaiming_init()

    def forward(self,
                x: Tensor,
                return_aux: Optional[bool] = None,
                return_semantic: bool = False,
                ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:

        use_aux = self.training if return_aux is None else return_aux

        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))

        # ---- Stage 1–3 --------------------------------------------------- #
        x = self.stem_conv1(x)
        x = self.stem_conv2(x)
        x = self.stem_stage2(x)
        x = self.stem_stage3(x)      # 1/8, channels*2

        # ---- Stage 4 ---------------------------------------------------- #
        x_s = self.semantic_branch_layers[0](x)      # 1/16, channels*4
        x_d = self.detail_branch_layers[0](x)        # 1/8,  channels*2

        comp_c = self.compression_1(self.relu(x_s))
        x_s    = x_s + self.down_1(self.relu(x_d))
        x_d    = x_d + resize(comp_c, size=out_size,
                               mode='bilinear',
                               align_corners=self.align_corners)

        c4_feat = x_d.clone() if use_aux else None

        # ---- Stage 5 ---------------------------------------------------- #
        x_s = self.semantic_branch_layers[1](self.relu(x_s))  # 1/32, channels*8
        x_d = self.detail_branch_layers[1](self.relu(x_d))    # 1/8,  channels*2

        comp_c = self.compression_2(self.relu(x_s))
        x_s    = x_s + self.down_2(self.relu(x_d))
        x_d    = x_d + resize(comp_c, size=out_size,
                               mode='bilinear',
                               align_corners=self.align_corners)

        # ---- Stage 6 ---------------------------------------------------- #
        x_d   = self.detail_branch_layers[2](self.relu(x_d))    # 1/8,  channels*4
        x_s   = self.semantic_branch_layers[2](self.relu(x_s))  # 1/64, channels*16
        x_spp = self.spp(x_s)
        x_spp = resize(x_spp, size=out_size,
                       mode='bilinear', align_corners=self.align_corners)

        fused = x_d + x_spp

        if return_semantic:
            return (c4_feat, x_spp, fused) if use_aux else (x_spp, fused)
        return (c4_feat, fused) if use_aux else fused

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, GCBlock):
                m.switch_to_deploy()
        self.deploy = True

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# =============================================================================
# Block1x1
# =============================================================================

class Block1x1(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 deploy: bool = False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.bias         = bias
        self.deploy       = deploy

        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1, stride=stride,
                                  padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(in_channels, out_channels,
                                    kernel_size=1, stride=stride,
                                    padding=padding, bias=bias,
                                    norm_cfg=norm_cfg, act_cfg=None)
            self.conv2 = ConvModule(out_channels, out_channels,
                                    kernel_size=1, stride=1,
                                    padding=padding, bias=bias,
                                    norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def _fuse_bn_tensor(self, conv: ConvModule):
        kernel       = conv.conv.weight
        bias         = (conv.conv.bias if conv.conv.bias is not None
                        else torch.zeros(kernel.shape[0], device=kernel.device))
        running_mean = conv.bn.running_mean
        running_var  = conv.bn.running_var
        gamma        = conv.bn.weight
        beta         = conv.bn.bias
        eps          = conv.bn.eps
        std          = (running_var + eps).sqrt()
        t            = (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias   = (beta + (bias - running_mean) * gamma / std
                        if self.bias else beta - running_mean * gamma / std)
        return kernel * t, fused_bias

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=1, stride=self.stride,
                              padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data   = bias2 + kernel2.squeeze(-1).squeeze(-1) @ bias1
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


# =============================================================================
# Block3x3
# =============================================================================

class Block3x3(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 deploy: bool = False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.padding      = padding
        self.bias         = bias
        self.deploy       = deploy

        if deploy:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3, stride=stride,
                                  padding=padding, bias=True)
        else:
            self.conv1 = ConvModule(in_channels, out_channels,
                                    kernel_size=3, stride=stride,
                                    padding=padding, bias=bias,
                                    norm_cfg=norm_cfg, act_cfg=None)
            self.conv2 = ConvModule(out_channels, out_channels,
                                    kernel_size=1, stride=1, padding=0,
                                    bias=bias, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.conv(x)
        return self.conv2(self.conv1(x))

    def _fuse_bn_tensor(self, conv: ConvModule):
        kernel       = conv.conv.weight
        bias         = (conv.conv.bias if conv.conv.bias is not None
                        else torch.zeros(kernel.shape[0], device=kernel.device))
        running_mean = conv.bn.running_mean
        running_var  = conv.bn.running_var
        gamma        = conv.bn.weight
        beta         = conv.bn.bias
        eps          = conv.bn.eps
        std          = (running_var + eps).sqrt()
        t            = (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias   = (beta + (bias - running_mean) * gamma / std
                        if self.bias else beta - running_mean * gamma / std)
        return kernel * t, fused_bias

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn_tensor(self.conv1)
        kernel2, bias2 = self._fuse_bn_tensor(self.conv2)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              kernel_size=3, stride=self.stride,
                              padding=self.padding, bias=True)
        self.conv.weight.data = torch.einsum(
            'oi,icjk->ocjk', kernel2.squeeze(3).squeeze(2), kernel1)
        self.conv.bias.data   = bias2 + kernel2.squeeze(-1).squeeze(-1) @ bias1
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.deploy = True


# =============================================================================
# GCBlock
# =============================================================================

class GCBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 act: bool = True,
                 deploy: bool = False):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.deploy       = deploy

        assert kernel_size == 3
        assert padding    == 1

        padding_11 = padding - kernel_size // 2

        self.relu = build_activation_layer(act_cfg) if act else nn.Identity()

        if deploy:
            self.reparam_3x3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=True,
                padding_mode=padding_mode)
        else:
            if (out_channels == in_channels) and (stride == 1):
                self.path_residual = build_norm_layer(norm_cfg, in_channels)[1]
            else:
                self.path_residual = None

            self.path_3x3_1 = Block3x3(
                in_channels, out_channels, stride=stride,
                padding=padding, bias=False, norm_cfg=norm_cfg)
            self.path_3x3_2 = Block3x3(
                in_channels, out_channels, stride=stride,
                padding=padding, bias=False, norm_cfg=norm_cfg)
            self.path_1x1 = Block1x1(
                in_channels, out_channels, stride=stride,
                padding=padding_11, bias=False, norm_cfg=norm_cfg)

    def forward(self, inputs: Tensor) -> Tensor:
        if hasattr(self, 'reparam_3x3'):
            return self.relu(self.reparam_3x3(inputs))

        id_out = 0 if self.path_residual is None else self.path_residual(inputs)

        return self.relu(
            self.path_3x3_1(inputs)
            + self.path_3x3_2(inputs)
            + self.path_1x1(inputs)
            + id_out
        )

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        self.path_3x3_1.switch_to_deploy()
        k3_1, b3_1 = self.path_3x3_1.conv.weight.data, self.path_3x3_1.conv.bias.data
        self.path_3x3_2.switch_to_deploy()
        k3_2, b3_2 = self.path_3x3_2.conv.weight.data, self.path_3x3_2.conv.bias.data
        self.path_1x1.switch_to_deploy()
        k1x1, b1x1 = self.path_1x1.conv.weight.data, self.path_1x1.conv.bias.data
        kid, bid   = self._fuse_bn_tensor(self.path_residual)

        return (k3_1 + k3_2 + self._pad_1x1_to_3x3(k1x1) + kid,
                b3_1 + b3_2 + b1x1 + bid)

    def _pad_1x1_to_3x3(self, kernel1x1: Optional[Tensor]) -> Tensor:
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, conv: nn.Module) -> Tuple:
        if conv is None:
            return 0, 0

        if isinstance(conv, ConvModule):
            kernel       = conv.conv.weight
            running_mean = conv.bn.running_mean
            running_var  = conv.bn.running_var
            gamma        = conv.bn.weight
            beta         = conv.bn.bias
            eps          = conv.bn.eps
        else:
            assert isinstance(conv, (nn.BatchNorm2d, nn.SyncBatchNorm))
            if not hasattr(self, 'id_tensor'):
                kv = np.zeros(
                    (self.in_channels, self.in_channels, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kv[i, i, 1, 1] = 1.0
                self.id_tensor = torch.from_numpy(kv).to(conv.weight.device)
            kernel       = self.id_tensor
            running_mean = conv.running_mean
            running_var  = conv.running_var
            gamma        = conv.weight
            beta         = conv.bias
            eps          = conv.eps

        std = (running_var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_3x3'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_3x3 = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, bias=True)
        self.reparam_3x3.weight.data = kernel
        self.reparam_3x3.bias.data   = bias
        for p in self.parameters():
            p.detach_()
        self.__delattr__('path_3x3_1')
        self.__delattr__('path_3x3_2')
        self.__delattr__('path_1x1')
        if hasattr(self, 'path_residual') and self.path_residual is not None:
            self.__delattr__('path_residual')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True