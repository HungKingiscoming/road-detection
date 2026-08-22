"""Modern compact two-stream network for aerial road extraction.

The detail stream starts at stride 4; the semantic stream uses full pretrained
ResNet-34 features at strides 8/16/32. Bilateral
exchanges happen at S8 and S16. Low-resolution semantic stages use dilated
large-kernel structural re-parameterization while detail stages use efficient
MobileNetV4-style universal inverted bottlenecks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .decoder import (
    ConvBN,
    ConvBNAct,
    ConvGNAct,
    ModernRoadDecoder,
    RepDepthwiseBlock,
    RepVGGBlock,
    _fuse_conv_bn,
)


def _extract_state_dict(checkpoint: object) -> Dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Encoder checkpoint must contain a state dictionary")
    for key in ("state_dict", "model", "ema"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    if not isinstance(checkpoint, dict):
        raise TypeError("Could not find a state dictionary")
    state: Dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, Tensor):
            continue
        clean = str(key)
        for prefix in ("module.", "encoder.backbone.", "backbone."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        state[clean] = value
    return state


def _build_resnet34(
    imagenet_pretrained: bool,
    encoder_weights_path: Optional[str],
) -> nn.Module:
    try:
        from torchvision.models import ResNet34_Weights, resnet34

        weights = (
            ResNet34_Weights.DEFAULT
            if imagenet_pretrained and not encoder_weights_path
            else None
        )
        backbone = resnet34(weights=weights)
    except ImportError as error:
        raise ImportError("torchvision is required for ResNet-34") from error
    except TypeError:
        from torchvision.models import resnet34

        backbone = resnet34(
            pretrained=bool(imagenet_pretrained and not encoder_weights_path)
        )

    if encoder_weights_path:
        path = Path(encoder_weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"Encoder weights not found: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state = _extract_state_dict(checkpoint)
        missing, _ = backbone.load_state_dict(state, strict=False)
        matched = len(backbone.state_dict()) - len(missing)
        if matched < 100:
            raise RuntimeError(
                f"Only {matched} ResNet tensors matched {path}; wrong weights?"
            )
    return backbone


class TruncatedResNet34(nn.Module):
    """ImageNet ResNet-34 through layer4 (S32), without classifier."""

    out_channels = (64, 64, 128, 256, 512)

    def __init__(
        self,
        imagenet_pretrained: bool = True,
        encoder_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        backbone = _build_resnet34(
            imagenet_pretrained=imagenet_pretrained,
            encoder_weights_path=encoder_weights_path,
        )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4


class SqueezeExcite(nn.Module):
    """Small channel gate used only on low-resolution semantic maps."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // max(1, reduction))
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.adaptive_avg_pool2d(x, 1)
        gate = F.relu(self.reduce(gate), inplace=True)
        return x * torch.sigmoid(self.expand(gate))


class UniversalInvertedBottleneck(nn.Module):
    """Efficient Extra-DW UIB instantiation inspired by MobileNetV4."""

    def __init__(
        self,
        channels: int,
        expansion: float = 1.5,
        start_kernel: int = 5,
        middle_kernel: int = 3,
    ) -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * expansion / 8.0)) * 8)
        if start_kernel > 0:
            self.start_dw: nn.Module = ConvBNAct(
                channels, channels, start_kernel, groups=channels
            )
        else:
            self.start_dw = nn.Identity()
        self.expand = ConvBNAct(channels, hidden, 1, padding=0)
        if middle_kernel > 0:
            self.middle_dw: nn.Module = ConvBNAct(
                hidden, hidden, middle_kernel, groups=hidden
            )
        else:
            self.middle_dw = nn.Identity()
        self.project = ConvBN(hidden, channels, 1, 1, 0)
        nn.init.zeros_(self.project.bn.weight)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.project(self.middle_dw(self.expand(self.start_dw(x))))
        return self.activation(x + residual)


class DilatedReparamBlock(nn.Module):
    """Dilated training branches exactly fused to one DW large kernel.

    The design follows the structural idea of UniRepLKNet: a large depthwise
    branch sees wide context directly, while parallel small/dilated branches
    ease optimization. Deployment retains one depthwise convolution.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 13,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size < 7 or kernel_size % 2 == 0:
            raise ValueError("large kernel must be odd and at least 7")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.deploy = bool(deploy)
        self.activation = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(channels)
        # Near-identity initialization protects the ImageNet feature
        # distribution seen by the following pretrained ResNet stage.
        # Keep this learnable scale one-dimensional.  A 4-D parameter with
        # singleton spatial dimensions is re-strided by ``channels_last``;
        # the broadcast gradient then has a different layout from DDP's
        # bucket view and triggers a reducer performance warning.
        self.residual_scale = nn.Parameter(torch.full((channels,), 1e-3))
        self.channel_mixer = UniversalInvertedBottleneck(
            channels, expansion=2.0, start_kernel=0, middle_kernel=0
        )

        if self.deploy:
            self.reparam = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=True,
            )
        else:
            self.large_branch = ConvBN(
                channels,
                channels,
                kernel_size,
                1,
                kernel_size // 2,
                groups=channels,
            )
            configurations = self._valid_branch_configurations(kernel_size)
            self.small_branches = nn.ModuleList(
                ConvBN(
                    channels,
                    channels,
                    small_kernel,
                    1,
                    dilation * (small_kernel // 2),
                    groups=channels,
                    dilation=dilation,
                )
                for small_kernel, dilation in configurations
            )
            self.branch_configurations = configurations
            self.identity = nn.BatchNorm2d(channels)

    @staticmethod
    def _valid_branch_configurations(
        kernel_size: int,
    ) -> Tuple[Tuple[int, int], ...]:
        candidates = ((5, 1), (3, 2), (3, 3), (3, 5))
        return tuple(
            (kernel, dilation)
            for kernel, dilation in candidates
            if 1 + (kernel - 1) * dilation <= kernel_size
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            spatial = self.reparam(x)
        else:
            spatial = self.large_branch(x) + self.identity(x)
            for branch in self.small_branches:
                spatial = spatial + branch(x)
        scale = self.residual_scale.view(1, -1, 1, 1)
        x = x + scale * self.se(self.activation(spatial))
        return self.channel_mixer(x)

    def _embed_dilated_kernel(self, kernel: Tensor, dilation: int) -> Tensor:
        target = kernel.new_zeros(
            kernel.shape[0], kernel.shape[1], self.kernel_size, self.kernel_size
        )
        small = kernel.shape[-1]
        center = self.kernel_size // 2
        start = center - (small // 2) * dilation
        stop = start + small * dilation
        target[:, :, start:stop:dilation, start:stop:dilation] = kernel
        return target

    def _fuse_identity(self) -> Tuple[Tensor, Tensor]:
        norm = self.identity
        kernel = norm.weight.new_zeros(
            self.channels, 1, self.kernel_size, self.kernel_size
        )
        kernel[:, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        std = torch.sqrt(norm.running_var + norm.eps)
        scale = norm.weight / std
        return (
            kernel * scale.reshape(-1, 1, 1, 1),
            norm.bias - norm.running_mean * scale,
        )

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        kernel, bias = _fuse_conv_bn(self.large_branch)
        identity_kernel, identity_bias = self._fuse_identity()
        kernel, bias = kernel + identity_kernel, bias + identity_bias
        for configuration, branch in zip(
            self.branch_configurations, self.small_branches
        ):
            _, dilation = configuration
            branch_kernel, branch_bias = _fuse_conv_bn(branch)
            kernel += self._embed_dilated_kernel(branch_kernel, dilation)
            bias += branch_bias
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.reparam = reparam
        del self.large_branch
        del self.small_branches
        del self.identity
        self.deploy = True


def _uib_stage(channels: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(
        *[
            UniversalInvertedBottleneck(channels)
            for _ in range(max(1, int(blocks)))
        ]
    )


def _large_kernel_stage(
    channels: int,
    blocks: int,
    kernel_size: int,
    deploy: bool,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            DilatedReparamBlock(
                channels, kernel_size=kernel_size, deploy=deploy
            )
            for _ in range(max(1, int(blocks)))
        ]
    )


class CompactPyramidContext(nn.Module):
    """Independent 1/2/4-grid context aggregation on the S32 stream."""

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        pool_sizes: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        sizes = tuple(sorted({int(size) for size in pool_sizes}))
        if not sizes or min(sizes) < 1:
            raise ValueError("context pool sizes must be positive")
        self.pool_sizes = sizes
        self.local = ConvGNAct(in_channels, branch_channels, 1, padding=0)
        self.projections = nn.ModuleList(
            ConvGNAct(in_channels, branch_channels, 1, padding=0)
            for _ in sizes
        )
        self.processes = nn.ModuleList(
            ConvGNAct(branch_channels, branch_channels, 3) for _ in sizes
        )
        self.compression = ConvGNAct(
            branch_channels * (len(sizes) + 1),
            out_channels,
            1,
            padding=0,
            activation=False,
        )
        self.shortcut = ConvGNAct(
            in_channels, out_channels, 1, padding=0, activation=False
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]
        outputs = [self.local(x)]
        for configured_size, projection, process in zip(
            self.pool_sizes, self.projections, self.processes
        ):
            grid = max(1, min(configured_size, *output_size))
            pooled = F.adaptive_avg_pool2d(x, (grid, grid))
            pooled = process(projection(pooled))
            outputs.append(
                F.interpolate(
                    pooled,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.activation(
            self.compression(torch.cat(outputs, dim=1)) + self.shortcut(x)
        )


class ModernDualResolutionContext(nn.Module):
    """Detail S4 plus semantic S8/S16/S32 with bilateral exchange."""

    def __init__(
        self,
        detail_channels: int = 80,
        semantic_channels: int = 256,
        context_channels: int = 48,
        context_pool_sizes: Sequence[int] = (1, 2, 4),
        detail_blocks: Sequence[int] = (2, 1, 1),
        semantic_blocks: int = 2,
        fusion_blocks: int = 2,
        large_kernel_size: int = 13,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if len(detail_blocks) != 3:
            raise ValueError("detail_blocks must contain three stage depths")
        self.detail_projection = ConvBNAct(64, detail_channels, 1, padding=0)
        self.detail_stages = nn.ModuleList(
            _uib_stage(detail_channels, depth) for depth in detail_blocks
        )

        self.semantic_to_detail_1 = ConvBNAct(
            128, detail_channels, 1, padding=0, activation=False
        )
        self.detail_to_semantic_1 = ConvBNAct(
            detail_channels,
            128,
            3,
            stride=2,
            activation=False,
            zero_init_bn=True,
        )

        # Keep S16 at 256 channels so the pretrained ResNet layer4 remains
        # structurally intact after the second bilateral exchange.
        self.semantic_s16_refine = _large_kernel_stage(
            256,
            1,
            large_kernel_size,
            deploy,
        )
        self.semantic_to_detail_2 = ConvBNAct(
            256, detail_channels, 1, padding=0, activation=False
        )
        intermediate = max(96, detail_channels)
        self.detail_to_semantic_2 = nn.Sequential(
            ConvBNAct(detail_channels, intermediate, 3, stride=2),
            ConvBNAct(
                intermediate,
                256,
                3,
                stride=2,
                activation=False,
                zero_init_bn=True,
            ),
        )

        self.semantic_s32_refine = nn.Sequential(
            ConvBNAct(512, semantic_channels, 1, padding=0),
            _large_kernel_stage(
                semantic_channels,
                semantic_blocks,
                large_kernel_size,
                deploy,
            ),
        )
        self.context = CompactPyramidContext(
            semantic_channels,
            context_channels,
            detail_channels,
            pool_sizes=context_pool_sizes,
        )
        self.final_refine = _uib_stage(detail_channels, fusion_blocks)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def fuse_s8(
        self, shallow_s4: Tensor, shared_s8: Tensor
    ) -> Tuple[Tensor, Tensor]:
        detail = self.detail_stages[0](self.detail_projection(shallow_s4))
        detail_before, semantic_before = detail, shared_s8
        detail = self.activation(
            detail_before
            + self._resize(
                self.semantic_to_detail_1(semantic_before),
                detail_before.shape[-2:],
            )
        )
        semantic = self.activation(
            semantic_before + self.detail_to_semantic_1(detail_before)
        )
        return self.detail_stages[1](detail), semantic

    def fuse_s16(
        self, detail_s4: Tensor, resnet_s16: Tensor
    ) -> Tuple[Tensor, Tensor]:
        semantic = self.semantic_s16_refine(resnet_s16)
        detail_before, semantic_before = detail_s4, semantic
        detail = self.activation(
            detail_before
            + self._resize(
                self.semantic_to_detail_2(semantic_before),
                detail_before.shape[-2:],
            )
        )
        semantic = self.activation(
            semantic_before + self.detail_to_semantic_2(detail_before)
        )
        return self.detail_stages[2](detail), semantic

    def finish(self, detail_s4: Tensor, resnet_s32: Tensor) -> Tensor:
        semantic = self.context(self.semantic_s32_refine(resnet_s32))
        semantic = self._resize(semantic, detail_s4.shape[-2:])
        return self.final_refine(detail_s4 + semantic)


class ModernRoadNet(nn.Module):
    """Modern compact road network with optional staged unfreezing support."""

    PHASE_NAMES = {
        0: "head_only",
        1: "head_plus_dual_branch",
        2: "plus_resnet_layer3_layer4",
        3: "plus_resnet_layer2",
        4: "all_trainable",
    }

    def __init__(
        self,
        num_classes: int = 2,
        detail_channels: int = 80,
        semantic_channels: int = 256,
        context_channels: int = 48,
        context_pool_sizes: Sequence[int] = (1, 2, 4),
        detail_blocks: Sequence[int] = (2, 1, 1),
        semantic_blocks: int = 2,
        fusion_blocks: int = 2,
        large_kernel_size: int = 13,
        decoder_s4_channels: int = 80,
        decoder_s2_channels: int = 40,
        full_channels: int = 32,
        dropout: float = 0.05,
        imagenet_pretrained: bool = True,
        encoder_weights_path: Optional[str] = None,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TruncatedResNet34(
            imagenet_pretrained=imagenet_pretrained,
            encoder_weights_path=encoder_weights_path,
        )
        self.dual_branch = ModernDualResolutionContext(
            detail_channels=detail_channels,
            semantic_channels=semantic_channels,
            context_channels=context_channels,
            context_pool_sizes=context_pool_sizes,
            detail_blocks=detail_blocks,
            semantic_blocks=semantic_blocks,
            fusion_blocks=fusion_blocks,
            large_kernel_size=large_kernel_size,
            deploy=deploy,
        )
        self.decode_head = ModernRoadDecoder(
            stem_channels=64,
            shallow_channels=64,
            fused_channels=detail_channels,
            s4_channels=decoder_s4_channels,
            s2_channels=decoder_s2_channels,
            full_channels=full_channels,
            num_classes=num_classes,
            dropout=dropout,
            deploy=deploy,
        )
        self.current_phase = 4

    def forward(self, image: Tensor) -> Tensor:
        output_size = image.shape[-2:]
        stem = self.encoder.stem(image)
        shallow = self.encoder.layer1(self.encoder.maxpool(stem))
        shared = self.encoder.layer2(shallow)
        detail, semantic_s8 = self.dual_branch.fuse_s8(shallow, shared)
        semantic_s16 = self.encoder.layer3(semantic_s8)
        detail, semantic_s16 = self.dual_branch.fuse_s16(
            detail, semantic_s16
        )
        semantic_s32 = self.encoder.layer4(semantic_s16)
        fused_s4 = self.dual_branch.finish(detail, semantic_s32)
        return self.decode_head(stem, shallow, fused_s4, output_size)

    def _phase_modules(self, phase: int) -> Tuple[nn.Module, ...]:
        modules: list[nn.Module] = [self.decode_head]
        if phase >= 1:
            modules.append(self.dual_branch)
        if phase >= 2:
            modules.extend((self.encoder.layer3, self.encoder.layer4))
        if phase >= 3:
            modules.append(self.encoder.layer2)
        if phase >= 4:
            modules.extend((self.encoder.stem, self.encoder.layer1))
        return tuple(modules)

    def set_trainable_phase(self, phase: int) -> str:
        phase = int(phase)
        if phase not in self.PHASE_NAMES:
            raise ValueError(f"Unknown trainable phase: {phase}")
        self.current_phase = phase
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in self._phase_modules(phase):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return self.PHASE_NAMES[phase]

    def enforce_frozen_norm_eval(self, freeze_encoder_bn: bool = False) -> None:
        trainable_modules = set(self._phase_modules(self.current_phase))
        groups = (
            self.dual_branch,
            self.encoder.layer4,
            self.encoder.layer3,
            self.encoder.layer2,
            self.encoder.stem,
            self.encoder.layer1,
        )
        for group in groups:
            if group not in trainable_modules:
                for module in group.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
        if freeze_encoder_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def trainable_parameter_counts(self) -> Tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return trainable, total

    def optimization_modules(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "head": self.decode_head.parameters(),
            "dual_branch": self.dual_branch.parameters(),
            "layer4": self.encoder.layer4.parameters(),
            "layer3": self.encoder.layer3.parameters(),
            "layer2": self.encoder.layer2.parameters(),
            "early_encoder": (
                parameter
                for module in (self.encoder.stem, self.encoder.layer1)
                for parameter in module.parameters()
            ),
        }

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(
                module, (RepVGGBlock, RepDepthwiseBlock, DilatedReparamBlock)
            ):
                module.switch_to_deploy()


DualBranchRoadNet = ModernRoadNet


def build_model(args) -> ModernRoadNet:
    """Build from an argparse Namespace or compatible attribute container."""
    return ModernRoadNet(
        num_classes=2,
        detail_channels=int(args.detail_channels),
        semantic_channels=int(args.semantic_channels),
        context_channels=int(args.dappm_channels),
        context_pool_sizes=tuple(int(value) for value in args.dappm_pool_sizes),
        detail_blocks=tuple(int(value) for value in args.detail_blocks),
        semantic_blocks=int(args.semantic_blocks),
        fusion_blocks=int(args.fusion_blocks),
        large_kernel_size=int(args.large_kernel_size),
        decoder_s4_channels=int(args.decoder_s4_channels),
        decoder_s2_channels=int(args.decoder_s2_channels),
        full_channels=int(args.full_channels),
        dropout=float(args.dropout),
        imagenet_pretrained=bool(args.imagenet_pretrained),
        encoder_weights_path=args.encoder_weights_path,
    )


@torch.no_grad()
def verify_large_kernel_reparameterization(
    block: DilatedReparamBlock,
    shape: Tuple[int, int, int, int],
) -> float:
    block.eval()
    sample = torch.randn(shape, device=next(block.parameters()).device)
    reference = block(sample)
    block.switch_to_deploy()
    return float((reference - block(sample)).abs().max())
