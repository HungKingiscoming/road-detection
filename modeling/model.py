"""Compact local-global road extraction network.

Architecture
------------
* one ImageNet-pretrained ResNet-34 encoder truncated after ``layer3`` (OS16);
* one compact bilateral bottleneck at OS16;
* local branch: a re-parameterizable RepVGG block at 96 channels;
* global branch: 96 -> 32, pooled grids 1/2/4/8, depthwise processing,
  aggregation, then 32 -> 96;
* a light asymmetric additive decoder, not a second U-Net;
* one centerline auxiliary head used during training only.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .decoder import CompactRoadDecoder, ConvBNAct, RepVGGBlock


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
        raise ImportError(
            "torchvision is required for the ResNet-34 encoder"
        ) from error
    except TypeError:
        # torchvision < 0.13 compatibility.
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
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        # layer4/fc may be absent in a deliberately truncated checkpoint.  A
        # completely unrelated file, however, should fail loudly.
        matched = len(backbone.state_dict()) - len(missing)
        if matched < 100:
            raise RuntimeError(
                f"Only {matched} ResNet tensors matched {path}; wrong weights?"
            )
        del unexpected
    return backbone


class TruncatedResNet34(nn.Module):
    """ResNet-34 feature encoder returning strides 2, 4, 8, and 16."""

    out_channels = (64, 64, 128, 256)

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        stem_s2 = self.stem(x)
        shallow_s4 = self.layer1(self.maxpool(stem_s2))
        middle_s8 = self.layer2(shallow_s4)
        deep_s16 = self.layer3(middle_s8)
        return stem_s2, shallow_s4, middle_s8, deep_s16


class GlobalPoolBranch(nn.Module):
    """Pool cheaply, resize, then apply spatial depthwise processing."""

    def __init__(self, channels: int, pool_size: int) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        groups = 8 if channels % 8 == 0 else 1
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
                padding_mode="replicate",
            ),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, output_size: Tuple[int, int]) -> Tensor:
        x = F.adaptive_avg_pool2d(x, self.pool_size)
        x = self.channel_mix(x)
        x = F.interpolate(
            x, size=output_size, mode="bilinear", align_corners=False
        )
        return self.depthwise(x)


class CompactPyramidGlobal(nn.Module):
    """96 -> 32 multi-grid context -> 96 with depthwise spatial mixing."""

    def __init__(
        self,
        channels: int = 96,
        global_channels: int = 32,
        pool_sizes: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        pool_sizes = tuple(sorted({int(size) for size in pool_sizes}))
        if not pool_sizes or pool_sizes[0] < 1:
            raise ValueError("pool_sizes must contain positive integers")
        self.pool_sizes = pool_sizes
        self.reduce = ConvBNAct(channels, global_channels, 1, padding=0)
        self.branches = nn.ModuleList(
            GlobalPoolBranch(global_channels, size) for size in pool_sizes
        )
        self.expand = ConvBNAct(
            global_channels, channels, 1, padding=0, activation=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        output_size = x.shape[-2:]
        context = torch.stack(
            [branch(x, output_size) for branch in self.branches], dim=0
        ).mean(dim=0)
        return self.expand(context)


class CompactLocalGlobalBottleneck(nn.Module):
    """Bilateral local/global processing at OS16 with gated residual fusion."""

    def __init__(
        self,
        in_channels: int = 256,
        channels: int = 96,
        global_channels: int = 32,
        pool_sizes: Sequence[int] = (1, 2, 4, 8),
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.reduce = ConvBNAct(in_channels, channels, 1, padding=0)
        self.local = RepVGGBlock(channels, channels, deploy=deploy)
        self.global_context = CompactPyramidGlobal(
            channels=channels,
            global_channels=global_channels,
            pool_sizes=pool_sizes,
        )
        self.fusion_gate = nn.Conv2d(channels * 2, channels, 1, bias=True)
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)
        self.refine = RepVGGBlock(channels, channels, deploy=deploy)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.reduce(x)
        local = self.local(residual)
        global_context = self.global_context(residual)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((local, global_context), dim=1))
        )
        fused = residual + gate * local + (1.0 - gate) * global_context
        return self.refine(fused)


class CompactRoadNet(nn.Module):
    """Resource-efficient road model with progressive-unfreezing support."""

    PHASE_NAMES = {
        0: "head_only",
        1: "head_plus_bottleneck",
        2: "plus_resnet_layer3",
        3: "plus_resnet_layer2",
        4: "all_trainable",
    }

    def __init__(
        self,
        num_classes: int = 2,
        bottleneck_channels: int = 96,
        global_channels: int = 32,
        decoder_channels: int = 96,
        half_channels: int = 48,
        pool_sizes: Sequence[int] = (1, 2, 4, 8),
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
        self.bottleneck = CompactLocalGlobalBottleneck(
            in_channels=256,
            channels=bottleneck_channels,
            global_channels=global_channels,
            pool_sizes=pool_sizes,
            deploy=deploy,
        )
        self.decode_head = CompactRoadDecoder(
            stem_channels=64,
            shallow_channels=64,
            middle_channels=128,
            deep_channels=bottleneck_channels,
            decoder_channels=decoder_channels,
            half_channels=half_channels,
            num_classes=num_classes,
            dropout=dropout,
            deploy=deploy,
        )
        self.current_phase = 4

    def forward(self, image: Tensor):
        output_size = image.shape[-2:]
        # During progressive fine-tuning, frozen stages run under no_grad.
        # Parameters keep requires_grad=True so DDP hooks remain valid when a
        # later epoch opens the stage; find_unused_parameters handles the
        # temporarily detached stages.
        if not self.training or self.current_phase >= 4:
            stem, shallow, middle, deep = self.encoder(image)
        elif self.current_phase <= 1:
            with torch.no_grad():
                stem, shallow, middle, deep = self.encoder(image)
        elif self.current_phase == 2:
            with torch.no_grad():
                stem = self.encoder.stem(image)
                shallow = self.encoder.layer1(self.encoder.maxpool(stem))
                middle = self.encoder.layer2(shallow)
            deep = self.encoder.layer3(middle)
        else:
            with torch.no_grad():
                stem = self.encoder.stem(image)
                shallow = self.encoder.layer1(self.encoder.maxpool(stem))
            middle = self.encoder.layer2(shallow)
            deep = self.encoder.layer3(middle)

        if self.training and self.current_phase == 0:
            with torch.no_grad():
                deep = self.bottleneck(deep)
        else:
            deep = self.bottleneck(deep)
        return self.decode_head(
            stem, shallow, middle, deep, output_size=output_size
        )

    def set_trainable_phase(self, phase: int) -> str:
        """Configure the requested fine-tuning phase.

        This changes the autograd path rather than mutating ``requires_grad``.
        That distinction makes gradual unfreezing safe after DDP has installed
        its reducer hooks.  Optimizer groups also stay stable across phases.
        """
        phase = int(phase)
        if phase not in self.PHASE_NAMES:
            raise ValueError(f"Unknown trainable phase: {phase}")
        self.current_phase = phase
        return self.PHASE_NAMES[phase]

    def enforce_frozen_norm_eval(self, freeze_encoder_bn: bool = True) -> None:
        """Prevent small per-GPU batches from corrupting pretrained BN stats."""
        frozen_modules: list[nn.Module] = []
        if self.current_phase == 0:
            frozen_modules.append(self.bottleneck)
        if self.current_phase <= 1:
            frozen_modules.append(self.encoder.layer3)
        if self.current_phase <= 2:
            frozen_modules.append(self.encoder.layer2)
        if self.current_phase <= 3:
            frozen_modules.extend((self.encoder.stem, self.encoder.layer1))
        for frozen in frozen_modules:
            for module in frozen.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        if freeze_encoder_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def trainable_parameter_counts(self) -> Tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        modules: list[nn.Module] = [self.decode_head]
        if self.current_phase >= 1:
            modules.append(self.bottleneck)
        if self.current_phase >= 2:
            modules.append(self.encoder.layer3)
        if self.current_phase >= 3:
            modules.append(self.encoder.layer2)
        if self.current_phase >= 4:
            modules.extend((self.encoder.stem, self.encoder.layer1))
        trainable = sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
        )
        return trainable, total

    def optimization_modules(self) -> Dict[str, Iterable[nn.Parameter]]:
        """Stable module groups for head/bottleneck/backbone differential LR."""
        return {
            "head": self.decode_head.parameters(),
            "bottleneck": self.bottleneck.parameters(),
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
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()


def build_model(args) -> CompactRoadNet:
    """Build from an argparse Namespace or another attribute container."""
    return CompactRoadNet(
        num_classes=2,
        bottleneck_channels=int(args.bottleneck_channels),
        global_channels=int(args.global_channels),
        decoder_channels=int(args.decoder_channels),
        half_channels=int(args.half_channels),
        pool_sizes=tuple(int(value) for value in args.global_pool_sizes),
        dropout=float(args.dropout),
        imagenet_pretrained=bool(args.imagenet_pretrained),
        encoder_weights_path=args.encoder_weights_path,
    )
