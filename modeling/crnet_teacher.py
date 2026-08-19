"""Resolution-configurable CRNet teacher for road extraction.

This is a cleaned integration of the supplied CRNet implementation.  It keeps
the ResNet34 encoder, cross-view context, relation-refined skip fusion and
LinkNet decoder, while fixing the original fixed-1024 reshapes and attention
layout.  The relation computation is vectorized over spatial groups to avoid
hundreds of Python-loop iterations per forward pass.

The network produces two raw logits so it is directly compatible with the
existing RoadLoss and road-aware knowledge-distillation code.  No sigmoid is
applied inside the model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet34_Weights, resnet34
        except ImportError as error:
            raise ImportError(
                "CRNet teacher requires torchvision compatible with PyTorch."
            ) from error
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        network = resnet34(weights=weights)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu)
        self.pool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        s2 = self.stem(image)
        s4 = self.layer1(self.pool(s2))
        s8 = self.layer2(s4)
        s16 = self.layer3(s8)
        s32 = self.layer4(s16)
        return {"s2": s2, "s4": s4, "s8": s8, "s16": s16, "s32": s32}


class TransformerBlock(nn.Module):
    """Correct batch-first self-attention used only by the CRNet teacher."""

    def __init__(self, channels: int, heads: int, expansion: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * expansion, channels),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm1(x)
        attention, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + self.dropout(attention)
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class AxialTransformer(nn.Module):
    """Apply self-attention independently along horizontal and vertical axes."""

    def __init__(self, channels: int = 512, heads: int = 8) -> None:
        super().__init__()
        self.horizontal = TransformerBlock(channels, heads)
        self.vertical = TransformerBlock(channels, heads)
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, feature: Tensor) -> Tensor:
        batch, channels, height, width = feature.shape
        horizontal = feature.permute(0, 2, 3, 1).reshape(
            batch * height, width, channels
        )
        horizontal = self.horizontal(horizontal).reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2)

        vertical = feature.permute(0, 3, 2, 1).reshape(
            batch * width, height, channels
        )
        vertical = self.vertical(vertical).reshape(
            batch, width, height, channels
        ).permute(0, 3, 2, 1)
        return self.fuse(torch.cat([horizontal, vertical], dim=1))


class CrossViewContext(nn.Module):
    """CRNet cross-view context with dynamic quadrant reconstruction."""

    def __init__(self, channels: int = 512, heads: int = 8) -> None:
        super().__init__()
        self.global_context = AxialTransformer(channels, heads)
        self.quadrant_refine = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(4)]
        )
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, feature: Tensor) -> Tensor:
        _, _, height, width = feature.shape
        if height % 2 or width % 2:
            raise ValueError(
                "CRNet context requires even stride-32 feature dimensions; "
                f"received {height}x{width}."
            )
        half_h, half_w = height // 2, width // 2
        quadrants = (
            feature[:, :, :half_h, :half_w],
            feature[:, :, :half_h, half_w:],
            feature[:, :, half_h:, :half_w],
            feature[:, :, half_h:, half_w:],
        )
        refined = [layer(value) for layer, value in zip(self.quadrant_refine, quadrants)]
        top = torch.cat((refined[0], refined[1]), dim=3)
        bottom = torch.cat((refined[2], refined[3]), dim=3)
        local_context = torch.cat((top, bottom), dim=2)
        return self.output(self.global_context(feature) + local_context)


class SpatialRelation(nn.Module):
    """Original CRNet patch relation, evaluated for all patches in one batch."""

    def __init__(self, source_elements: int, target_elements: int) -> None:
        super().__init__()
        self.projection = nn.Linear(source_elements, target_elements)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        # source/target: [B*patches, 1, elements]
        source = self.projection(source)
        attention = torch.bmm(target.transpose(1, 2), source)
        attention = F.softmax(attention, dim=-1)
        return torch.bmm(target, attention)


def _grid_to_batch(feature: Tensor, groups: int) -> Tuple[Tensor, Tuple[int, ...]]:
    batch, channels, height, width = feature.shape
    if height % groups or width % groups:
        raise ValueError(
            f"Feature {height}x{width} is not divisible by relation grid {groups}."
        )
    patch_h, patch_w = height // groups, width // groups
    patches = feature.reshape(
        batch, channels, groups, patch_h, groups, patch_w
    ).permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(batch * groups * groups, channels, patch_h, patch_w)
    return patches, (batch, groups, channels, patch_h, patch_w)


def _batch_to_grid(patches: Tensor, metadata: Tuple[int, ...]) -> Tensor:
    batch, groups, channels, patch_h, patch_w = metadata
    return patches.reshape(
        batch, groups, groups, channels, patch_h, patch_w
    ).permute(0, 3, 1, 4, 2, 5).reshape(
        batch, channels, groups * patch_h, groups * patch_w
    )


class RelationRefine(nn.Module):
    """Vectorized semantic and spatial relation refinement."""

    def __init__(
        self,
        deep_channels: int,
        skip_channels: int,
        source_elements: int,
        target_elements: int,
        channel_groups: int = 8,
    ) -> None:
        super().__init__()
        if deep_channels % channel_groups or skip_channels % channel_groups:
            raise ValueError("Relation channels must be divisible by channel_groups")
        deep_group = deep_channels // channel_groups
        skip_group = skip_channels // channel_groups
        self.channel_groups = channel_groups
        self.deep_mlp = nn.Sequential(
            nn.Conv1d(deep_group, skip_group, 1), nn.ReLU(inplace=True)
        )
        self.skip_mlp = nn.Sequential(
            nn.Conv1d(skip_group, skip_group, 1), nn.ReLU(inplace=True)
        )
        self.deep_position = nn.Conv2d(deep_channels, 1, 1)
        self.skip_position = nn.Conv2d(skip_channels, 1, 1)
        self.spatial_relation = SpatialRelation(source_elements, target_elements)
        self.output = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
        )

    def _semantic_refine(self, deep: Tensor, skip: Tensor) -> Tensor:
        batch, deep_channels = deep.shape[:2]
        skip_channels = skip.shape[1]
        groups = self.channel_groups
        deep_group = deep_channels // groups
        skip_group = skip_channels // groups

        deep_descriptor = F.adaptive_avg_pool2d(deep, 1).reshape(
            batch * groups, deep_group, 1
        )
        skip_descriptor = F.adaptive_avg_pool2d(skip, 1).reshape(
            batch * groups, skip_group, 1
        )
        deep_descriptor = self.deep_mlp(deep_descriptor)
        skip_descriptor = self.skip_mlp(skip_descriptor)
        relation = torch.bmm(deep_descriptor, skip_descriptor.transpose(1, 2))
        channel_weight = F.softmax(
            torch.bmm(relation, skip_descriptor), dim=1
        ).reshape(batch, skip_channels, 1, 1)
        return skip * channel_weight

    def _spatial_refine(self, deep: Tensor, skip: Tensor, grid: int) -> Tensor:
        deep_patches, deep_meta = _grid_to_batch(deep, grid)
        skip_patches, skip_meta = _grid_to_batch(skip, grid)
        deep_position = self.deep_position(deep_patches).flatten(2)
        skip_position = self.skip_position(skip_patches).flatten(2)
        spatial = self.spatial_relation(deep_position, skip_position)
        spatial = spatial.reshape(
            skip_patches.shape[0], 1, skip_patches.shape[2], skip_patches.shape[3]
        )
        spatial_meta = (
            skip_meta[0], skip_meta[1], 1, skip_meta[3], skip_meta[4]
        )
        return _batch_to_grid(spatial, spatial_meta)

    def forward(self, deep: Tensor, skip: Tensor, grid: int) -> Tensor:
        semantic = self._semantic_refine(deep, skip)
        spatial = self._spatial_refine(deep, skip, grid)
        return self.output(semantic + spatial)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden, hidden, 3, stride=2, padding=1,
                output_padding=1, bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CRNetTeacher(nn.Module):
    """CRNet adapted to the training/evaluation contract of this project."""

    def __init__(
        self,
        crop_size: int = 512,
        pretrained: bool = True,
        freeze_backbone_bn: bool = True,
    ) -> None:
        super().__init__()
        if crop_size < 256 or crop_size % 64:
            raise ValueError("CRNet crop_size must be >=256 and divisible by 64")
        self.crop_size = int(crop_size)
        # This preserves the original per-patch element counts at both 512 and
        # 1024: e4 patch=2x2, e3 patch=4x4, ..., stem patch=32x32.
        self.relation_grid = self.crop_size // 64
        self.freeze_backbone_bn = bool(freeze_backbone_bn)
        self.backbone = ResNet34Encoder(pretrained=pretrained)
        self.context = CrossViewContext(512, heads=8)

        self.rr4 = RelationRefine(512, 256, 4, 16)
        self.rr3 = RelationRefine(256, 128, 16, 64)
        self.rr2 = RelationRefine(128, 64, 64, 256)
        self.rr1 = RelationRefine(64, 64, 256, 1024)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        self._initialize_new_layers()

    def _initialize_new_layers(self) -> None:
        for name, module in self.named_modules():
            if name.startswith("backbone"):
                continue
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.final[-1].weight, mean=0.0, std=0.01)

    def _freeze_bn_statistics(self) -> None:
        if not self.freeze_backbone_bn:
            return
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True) -> "CRNetTeacher":
        super().train(mode)
        if mode:
            self._freeze_bn_statistics()
        return self

    @staticmethod
    def _two_class_logits(road_logit: Tensor) -> Tensor:
        # class-1 minus class-0 remains exactly road_logit.
        return torch.cat((-0.5 * road_logit, 0.5 * road_logit), dim=1)

    def forward(self, image: Tensor) -> Union[
        Tensor, Tuple[Optional[Tensor], Optional[Tensor], Tensor]
    ]:
        if image.shape[-2:] != (self.crop_size, self.crop_size):
            raise ValueError(
                f"CRNet was built for {self.crop_size}x{self.crop_size} tiles, "
                f"but received {tuple(image.shape[-2:])}."
            )
        features = self.backbone(image)
        x = features["s2"]
        e1, e2, e3 = features["s4"], features["s8"], features["s16"]
        e4 = self.context(features["s32"])
        grid = self.relation_grid

        d4 = self.decoder4(e4) + self.rr4(e4, e3, grid)
        d3 = self.decoder3(d4) + self.rr3(d4, e2, grid)
        d2 = self.decoder2(d3) + self.rr2(d3, e1, grid)
        d1 = self.decoder1(d2) + self.rr1(d2, x, grid)
        logits = self._two_class_logits(self.final(d1))

        if self.training:
            return None, None, logits
        return logits

    def switch_to_deploy(self) -> "CRNetTeacher":
        return self


def build_crnet_teacher(args, *, pretrained: Optional[bool] = None) -> CRNetTeacher:
    if pretrained is None:
        pretrained = bool(getattr(args, "crnet_pretrained", True))
    return CRNetTeacher(
        crop_size=int(getattr(args, "crop_size", 512)),
        pretrained=pretrained,
        freeze_backbone_bn=bool(getattr(args, "crnet_freeze_backbone_bn", True)),
    )


__all__ = ["CRNetTeacher", "build_crnet_teacher"]
