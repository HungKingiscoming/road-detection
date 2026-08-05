
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

OptConfigType = Optional[Dict]
SampleList = List[Dict]

BATCH_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)

NORM_TYPES = BATCH_NORM_TYPES + (
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


# ============================================================================
# BASE MODULE
# ============================================================================

class BaseModule(nn.Module):
    def __init__(self, init_cfg: OptConfigType = None):
        super().__init__()
        self.init_cfg = init_cfg
        self._is_init = False

    def init_weights(self):
        if self._is_init:
            return
        if self.init_cfg is not None:
            self._initialize_weights(self.init_cfg)
        self._is_init = True

    def _initialize_weights(self, init_cfg: Dict):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, NORM_TYPES):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ============================================================================
# NORMALIZATION
# ============================================================================

def build_norm_layer(cfg: Union[Dict, str], num_features: int) -> Tuple[str, nn.Module]:
    if isinstance(cfg, str):
        cfg = {'type': cfg}

    norm_type = cfg.get('type', 'BN')

    if norm_type in ['BN', 'BatchNorm2d']:
        layer = nn.BatchNorm2d(num_features, **cfg.get('kwargs', {}))
        name  = 'bn'
    elif norm_type in ['GN', 'GroupNorm']:
        num_groups = cfg.get('num_groups', 32)
        layer = nn.GroupNorm(num_groups, num_features, **cfg.get('kwargs', {}))
        name  = 'gn'
    elif norm_type in ['LN', 'LayerNorm']:
        layer = nn.LayerNorm(num_features, **cfg.get('kwargs', {}))
        name  = 'ln'
    elif norm_type in ['IN', 'InstanceNorm2d']:
        layer = nn.InstanceNorm2d(num_features, **cfg.get('kwargs', {}))
        name  = 'in'
    elif norm_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_features, **cfg.get('kwargs', {}))
        name  = 'sync_bn'
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')

    return name, layer


# ============================================================================
# ACTIVATION
# ============================================================================

def build_activation_layer(cfg: Union[Dict, str, None]) -> Optional[nn.Module]:
    if cfg is None:
        return None
    if isinstance(cfg, str):
        cfg = {'type': cfg}

    act_type = cfg.get('type', 'ReLU')

    if act_type == 'ReLU':
        return nn.ReLU(inplace=cfg.get('inplace', True))
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=cfg.get('negative_slope', 0.01),
                            inplace=cfg.get('inplace', True))
    elif act_type == 'PReLU':
        return nn.PReLU(num_parameters=cfg.get('num_parameters', 1))
    elif act_type == 'ReLU6':
        return nn.ReLU6(inplace=cfg.get('inplace', True))
    elif act_type == 'ELU':
        return nn.ELU(alpha=cfg.get('alpha', 1.0), inplace=cfg.get('inplace', True))
    elif act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'Tanh':
        return nn.Tanh()
    elif act_type == 'Hardswish':
        return nn.Hardswish(inplace=cfg.get('inplace', True))
    elif act_type in ('SiLU', 'Swish'):
        return nn.SiLU(inplace=cfg.get('inplace', True))
    elif act_type == 'Identity':
        return nn.Identity()
    else:
        raise ValueError(f'Unsupported activation type: {act_type}')


# ============================================================================
# CONVOLUTION MODULE
# ============================================================================

class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = 'auto',
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        order: Tuple[str, ...] = ('conv', 'norm', 'act'),
    ):
        super().__init__()

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg  = act_cfg
        self.order    = order

        self.with_norm       = norm_cfg is not None
        self.with_activation = act_cfg is not None

        # ---- bias ---- #
        if bias == 'auto':
            bias = not (self.with_norm and
                        'norm' in order and 'conv' in order and
                        order.index('norm') > order.index('conv'))

        # ---- Conv ---- #
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )

        # ---- Norm ---- #
        if self.with_norm:
            if 'norm' in order and 'conv' in order:
                norm_before_conv = order.index('norm') < order.index('conv')
            else:
                norm_before_conv = False
            norm_channels = in_channels if norm_before_conv else out_channels

            norm_name, norm_layer = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(norm_name, norm_layer)
            object.__setattr__(self, '_norm_key', norm_name)

        # ---- Activation ---- #
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)

    @property
    def norm(self):
        """Trả norm layer — dùng self._modules trực tiếp để tránh property loop."""
        if self.with_norm:
            return self._modules[self._norm_key]
        return None

    @property
    def bn(self):
        """Alias .bn cho _fuse_bn_tensor — dùng self._modules trực tiếp."""
        if self.with_norm:
            return self._modules[self._norm_key]
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_name in self.order:
            if layer_name == 'conv':
                x = self.conv(x)
            elif layer_name == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer_name == 'act' and self.with_activation:
                x = self.activate(x)
        return x


# ============================================================================
# RESIZE
# ============================================================================

def resize(input, size=None, scale_factor=None, mode='nearest',
           align_corners=None, warning=True):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# ============================================================================
# DAPPM
# ============================================================================

class DAPPM(BaseModule):
    """DAPPM — Deep Aggregation Pyramid Pooling Module.

    Với ConvModule đã fix, order=('norm','act','conv') hoạt động đúng:
    BN được build với in_channels khi đứng trước conv.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear',
                 init_cfg: Optional[Dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.num_scales    = num_scales
        self.upsample_mode = upsample_mode
        self.in_channels   = in_channels
        self.branch_channels = branch_channels
        self.out_channels  = out_channels
        self.norm_cfg      = norm_cfg
        self.act_cfg       = act_cfg
        self.conv_cfg      = conv_cfg

        self.scales = nn.ModuleList([
            ConvModule(in_channels, branch_channels, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg, **conv_cfg)
        ])

        for i in range(1, num_scales - 1):
            self.scales.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=kernel_sizes[i - 1],
                             stride=strides[i - 1],
                             padding=paddings[i - 1]),
                ConvModule(in_channels, branch_channels, kernel_size=1,
                           norm_cfg=norm_cfg, act_cfg=act_cfg, **conv_cfg)
            ))

        self.scales.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(in_channels, branch_channels, kernel_size=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg, **conv_cfg)
        ))

        self.processes = nn.ModuleList([
            ConvModule(branch_channels, branch_channels, kernel_size=3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg, **conv_cfg)
            for _ in range(num_scales - 1)
        ])

        self.compression = ConvModule(
            branch_channels * num_scales, out_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, **conv_cfg)

        self.shortcut = ConvModule(
            in_channels, out_channels, kernel_size=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, **conv_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = [self.scales[0](inputs)]

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.upsample_mode,
                align_corners=False if self.upsample_mode == 'bilinear' else None)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats, dim=1)) + self.shortcut(inputs)


# ============================================================================
# PAPPM
# ============================================================================

class PAPPM(DAPPM):
    def __init__(self, in_channels, branch_channels, out_channels, num_scales,
                 kernel_sizes=[5, 9, 17], strides=[2, 4, 8], paddings=[2, 4, 8],
                 norm_cfg=dict(type='BN', momentum=0.1),
                 act_cfg=dict(type='ReLU', inplace=True),
                 conv_cfg=dict(order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode='bilinear'):
        super().__init__(in_channels, branch_channels, out_channels, num_scales,
                         kernel_sizes, strides, paddings, norm_cfg, act_cfg,
                         conv_cfg, upsample_mode)
        self.processes = ConvModule(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3, padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, **self.conv_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_   = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs), size=inputs.shape[2:],
                mode=self.upsample_mode, align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out], dim=1)) + self.shortcut(inputs)


# ============================================================================
# RPPM + RepParallel
# ============================================================================

class RepParallel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, padding_mode='zeros',
                 norm_cfg=dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 act_cfg=dict(type='ReLU', inplace=True),
                 deploy=False):
        super().__init__()
        self.deploy      = deploy
        self.groups      = groups
        self.in_channels  = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3 and padding == 1

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.relu = build_activation_layer(act_cfg)

        if deploy:
            self.reparam_parallel = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=True,
                padding_mode=padding_mode)
        else:
            self.parallel1 = ConvModule(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False,
                norm_cfg=norm_cfg, act_cfg=None)
            self.parallel2 = ConvModule(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False,
                norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.relu(self.norm(inputs))
        if hasattr(self, 'reparam_parallel'):
            return self.reparam_parallel(inputs)
        return self.parallel1(inputs) + self.parallel2(inputs)

    def get_equivalent_kernel_bias(self):
        k1, b1 = self._fuse_bn_tensor(self.parallel1)
        k2, b2 = self._fuse_bn_tensor(self.parallel2)
        return k1 + k2, b1 + b2

    def _fuse_bn_tensor(self, conv):
        kernel       = conv.conv.weight
        running_mean = conv.bn.running_mean
        running_var  = conv.bn.running_var
        gamma        = conv.bn.weight
        beta         = conv.bn.bias
        eps          = conv.bn.eps
        std          = (running_var + eps).sqrt()
        t            = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'reparam_parallel'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_parallel = nn.Conv2d(
            in_channels=self.parallel1.conv.in_channels,
            out_channels=self.parallel1.conv.out_channels,
            kernel_size=self.parallel1.conv.kernel_size,
            stride=self.parallel1.conv.stride,
            padding=self.parallel1.conv.padding,
            groups=self.parallel1.conv.groups, bias=True)
        self.reparam_parallel.weight.data = kernel
        self.reparam_parallel.bias.data   = bias
        for p in self.parameters():
            p.detach_()
        self.__delattr__('parallel1')
        self.__delattr__('parallel2')
        self.deploy = True


class RPPM(DAPPM):
    def __init__(self, in_channels, branch_channels, out_channels, num_scales,
                 kernel_sizes=[5, 9, 17], strides=[2, 4, 8], paddings=[2, 4, 8],
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 conv_cfg=dict(order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode='bilinear', deploy=False):
        super().__init__(in_channels, branch_channels, out_channels, num_scales,
                         kernel_sizes, strides, paddings, norm_cfg, act_cfg,
                         conv_cfg, upsample_mode)
        self.deploy    = deploy
        self.processes = RepParallel(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3, padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg, deploy=self.deploy)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x_   = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs), size=inputs.shape[2:],
                mode=self.upsample_mode, align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out], dim=1)) + self.shortcut(inputs)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepParallel):
                m.switch_to_deploy()


# ============================================================================
# BASE DECODE HEAD
# ============================================================================

class BaseDecodeHead(BaseModule):
    def __init__(self, in_channels, channels, num_classes,
                 dropout_ratio=0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels   = in_channels
        self.channels      = channels
        self.num_classes   = num_classes
        self.dropout_ratio = dropout_ratio
        self.norm_cfg      = norm_cfg
        self.act_cfg       = act_cfg
        self.align_corners = align_corners
        self.dropout       = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg      = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        raise NotImplementedError

    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)


# ============================================================================
# BasicBlock / Bottleneck (giữ nguyên từ bản gốc)
# ============================================================================

class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU', inplace=True),
                 act_cfg_out=dict(type='ReLU', inplace=True), init_cfg=None):
        super().__init__(init_cfg)
        self.conv1      = ConvModule(in_channels, channels, 3, stride=stride,
                                     padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2      = ConvModule(channels, channels, 3, padding=1,
                                     norm_cfg=norm_cfg, act_cfg=None)
        self.downsample = downsample
        if act_cfg_out:
            self.act = build_activation_layer(act_cfg_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out      = self.conv2(self.conv1(x))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        if hasattr(self, 'act'):
            out = self.act(out)
        return out


class Bottleneck(BaseModule):
    expansion = 2

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU', inplace=True),
                 act_cfg_out=None, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1      = ConvModule(in_channels, channels, 1,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2      = ConvModule(channels, channels, 3, stride=stride,
                                     padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3      = ConvModule(channels, channels * self.expansion, 1,
                                     norm_cfg=norm_cfg, act_cfg=None)
        self.downsample = downsample
        if act_cfg_out:
            self.act = build_activation_layer(act_cfg_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out      = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        if hasattr(self, 'act'):
            out = self.act(out)
        return out
