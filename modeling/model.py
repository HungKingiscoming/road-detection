from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .decoder import (
    ConvBNAct,
    ConvGNAct,
    RepDepthwiseBlock,
    RepVGGBlock,
    RoadReconstructionDecoder,
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
    """Return ResNet features C2/S2, C4/S4, C8/S8, and C16/S16."""

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
        shared_s8 = self.layer2(shallow_s4)
        semantic_s16 = self.layer3(shared_s8)
        return stem_s2, shallow_s4, shared_s8, semantic_s16


def _rep_stage(channels: int, blocks: int, deploy: bool) -> nn.Sequential:
    return nn.Sequential(
        *[
            RepVGGBlock(channels, channels, deploy=deploy)
            for _ in range(max(1, int(blocks)))
        ]
    )


class ProgressiveDAPPM(nn.Module):
    """Progressively aggregate adaptive pooled context at the semantic S32 map.

    GroupNorm keeps the global 1x1 branch valid for small per-GPU batches.
    Pooling proceeds from finer to coarser grids, so each stage adds broader
    context to the previous representation before concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        pool_sizes: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        sizes = tuple(sorted({int(size) for size in pool_sizes}, reverse=True))
        if not sizes or min(sizes) < 1:
            raise ValueError("DAPPM pool sizes must be positive")
        self.pool_sizes = sizes
        self.scale0 = ConvGNAct(in_channels, branch_channels, 1, padding=0)
        self.pool_projections = nn.ModuleList(
            ConvGNAct(in_channels, branch_channels, 1, padding=0)
            for _ in sizes
        )
        self.processes = nn.ModuleList(
            ConvGNAct(branch_channels, branch_channels, 3)
            for _ in sizes
        )
        self.compression = ConvGNAct(
            branch_channels * (len(sizes) + 1),
            out_channels,
            1,
            padding=0,
            activation=False,
        )
        self.shortcut = ConvGNAct(
            in_channels,
            out_channels,
            1,
            padding=0,
            activation=False,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]
        previous = self.scale0(x)
        outputs = [previous]
        for configured_size, projection, process in zip(
            self.pool_sizes, self.pool_projections, self.processes
        ):
            grid = max(1, min(configured_size, *output_size))
            pooled = F.adaptive_avg_pool2d(x, (grid, grid))
            pooled = projection(pooled)
            pooled = F.interpolate(
                pooled,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            previous = process(previous + pooled)
            outputs.append(previous)
        context = self.compression(torch.cat(outputs, dim=1))
        return self.activation(context + self.shortcut(x))


class DualResolutionContext(nn.Module):
    """Detail S8 and semantic S16/S32 branches with two bilateral fusions."""

    def __init__(
        self,
        detail_channels: int = 96,
        semantic_channels: int = 192,
        dappm_channels: int = 32,
        dappm_pool_sizes: Sequence[int] = (1, 2, 4, 8),
        detail_blocks: Sequence[int] = (2, 2, 1),
        semantic_blocks: int = 2,
        fusion_blocks: int = 2,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if len(detail_blocks) != 3:
            raise ValueError("detail_blocks must contain three stage depths")
        self.detail_projection = ConvBNAct(128, detail_channels, 1, padding=0)
        self.detail_stages = nn.ModuleList(
            _rep_stage(detail_channels, depth, deploy)
            for depth in detail_blocks
        )

        self.semantic_downstage = nn.Sequential(
            ConvBNAct(256, semantic_channels, 3, stride=2),
            _rep_stage(semantic_channels, semantic_blocks, deploy),
        )

        # S16 <-> S8. Only detail-to-semantic is zero-initialized, protecting
        # the pretrained ResNet semantic representation at the start.
        self.semantic_to_detail_1 = ConvBNAct(
            256, detail_channels, 1, padding=0, activation=False
        )
        self.detail_to_semantic_1 = ConvBNAct(
            detail_channels,
            256,
            3,
            stride=2,
            activation=False,
            zero_init_bn=True,
        )

        # S32 <-> S8. The detail path is downsampled twice to align exactly.
        self.semantic_to_detail_2 = ConvBNAct(
            semantic_channels,
            detail_channels,
            1,
            padding=0,
            activation=False,
        )
        intermediate_channels = max(128, detail_channels)
        self.detail_to_semantic_2 = nn.Sequential(
            ConvBNAct(
                detail_channels,
                intermediate_channels,
                3,
                stride=2,
            ),
            ConvBNAct(
                intermediate_channels,
                semantic_channels,
                3,
                stride=2,
                activation=False,
                zero_init_bn=True,
            ),
        )

        self.dappm = ProgressiveDAPPM(
            semantic_channels,
            dappm_channels,
            detail_channels,
            pool_sizes=dappm_pool_sizes,
        )
        self.final_refine = _rep_stage(detail_channels, fusion_blocks, deploy)
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, shared_s8: Tensor, semantic_s16: Tensor) -> Tensor:
        detail = self.detail_stages[0](self.detail_projection(shared_s8))
        semantic = semantic_s16

        detail_before, semantic_before = detail, semantic
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

        detail = self.detail_stages[1](detail)
        semantic = self.semantic_downstage(semantic)
        detail_before, semantic_before = detail, semantic
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

        detail = self.detail_stages[2](detail)
        semantic = self.dappm(semantic)
        semantic = self._resize(semantic, detail.shape[-2:])
        return self.final_refine(detail + semantic)


class DualBranchRoadNet(nn.Module):
    """Dual-resolution road model with progressive-unfreezing support."""

    PHASE_NAMES = {
        0: "head_only",
        1: "head_plus_dual_branch",
        2: "plus_resnet_layer3",
        3: "plus_resnet_layer2",
        4: "all_trainable",
    }

    def __init__(
        self,
        num_classes: int = 2,
        detail_channels: int = 96,
        semantic_channels: int = 192,
        dappm_channels: int = 32,
        dappm_pool_sizes: Sequence[int] = (1, 2, 4, 8),
        detail_blocks: Sequence[int] = (2, 2, 1),
        semantic_blocks: int = 2,
        fusion_blocks: int = 2,
        decoder_s4_channels: int = 64,
        decoder_s2_channels: int = 32,
        full_channels: int = 24,
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
        self.dual_branch = DualResolutionContext(
            detail_channels=detail_channels,
            semantic_channels=semantic_channels,
            dappm_channels=dappm_channels,
            dappm_pool_sizes=dappm_pool_sizes,
            detail_blocks=detail_blocks,
            semantic_blocks=semantic_blocks,
            fusion_blocks=fusion_blocks,
            deploy=deploy,
        )
        self.decode_head = RoadReconstructionDecoder(
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

    def forward(self, image: Tensor):
        output_size = image.shape[-2:]
        if not self.training or self.current_phase >= 4:
            stem, shallow, shared, semantic = self.encoder(image)
        elif self.current_phase <= 1:
            with torch.no_grad():
                stem, shallow, shared, semantic = self.encoder(image)
        elif self.current_phase == 2:
            with torch.no_grad():
                stem = self.encoder.stem(image)
                shallow = self.encoder.layer1(self.encoder.maxpool(stem))
                shared = self.encoder.layer2(shallow)
            semantic = self.encoder.layer3(shared)
        else:
            with torch.no_grad():
                stem = self.encoder.stem(image)
                shallow = self.encoder.layer1(self.encoder.maxpool(stem))
            shared = self.encoder.layer2(shallow)
            semantic = self.encoder.layer3(shared)

        if self.training and self.current_phase == 0:
            with torch.no_grad():
                fused = self.dual_branch(shared, semantic)
        else:
            fused = self.dual_branch(shared, semantic)
        return self.decode_head(stem, shallow, fused, output_size)

    def set_trainable_phase(self, phase: int) -> str:
        phase = int(phase)
        if phase not in self.PHASE_NAMES:
            raise ValueError(f"Unknown trainable phase: {phase}")
        self.current_phase = phase
        return self.PHASE_NAMES[phase]

    def enforce_frozen_norm_eval(self, freeze_encoder_bn: bool = True) -> None:
        frozen_modules: list[nn.Module] = []
        if self.current_phase == 0:
            frozen_modules.append(self.dual_branch)
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
            modules.append(self.dual_branch)
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
        return {
            "head": self.decode_head.parameters(),
            "dual_branch": self.dual_branch.parameters(),
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
            if isinstance(module, (RepVGGBlock, RepDepthwiseBlock)):
                module.switch_to_deploy()


def build_model(args) -> DualBranchRoadNet:
    """Build from an argparse Namespace or compatible attribute container."""
    return DualBranchRoadNet(
        num_classes=2,
        detail_channels=int(args.detail_channels),
        semantic_channels=int(args.semantic_channels),
        dappm_channels=int(args.dappm_channels),
        dappm_pool_sizes=tuple(int(value) for value in args.dappm_pool_sizes),
        detail_blocks=tuple(int(value) for value in args.detail_blocks),
        semantic_blocks=int(args.semantic_blocks),
        fusion_blocks=int(args.fusion_blocks),
        decoder_s4_channels=int(args.decoder_s4_channels),
        decoder_s2_channels=int(args.decoder_s2_channels),
        full_channels=int(args.full_channels),
        dropout=float(args.dropout),
        imagenet_pretrained=bool(args.imagenet_pretrained),
        encoder_weights_path=args.encoder_weights_path,
    )
