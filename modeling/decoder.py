import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from components.components import (
    BaseModule,
    ConvModule,
    build_norm_layer,
    build_activation_layer,
    resize,
    OptConfigType,
    SampleList,
)


# =============================================================================
# Accuracy helper
# =============================================================================

def accuracy(pred: Tensor,
             target: Tensor,
             ignore_index: int = 255) -> Tensor:
    """Tính pixel accuracy, bỏ qua các pixel có nhãn = ignore_index."""
    pred_label = pred.argmax(dim=1)
    mask       = target != ignore_index
    correct    = (pred_label[mask] == target[mask]).sum().float()
    total      = mask.sum().float().clamp(min=1)
    return correct / total * 100.0


# =============================================================================
# Cross-entropy loss wrapper
# =============================================================================

class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight  = loss_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss_weight * F.cross_entropy(
            pred, target, ignore_index=self.ignore_index)


# =============================================================================
# OHEM Cross-entropy loss
# =============================================================================

class OHEMCrossEntropyLoss(nn.Module):

    def __init__(self,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0,
                 thresh: float = 1.5,
                 min_kept: int = 100_000):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight  = loss_weight
        self.thresh       = thresh
        self.min_kept     = min_kept

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        losses = F.cross_entropy(
            pred, target,
            ignore_index=self.ignore_index,
            reduction='none'
        ).view(-1)

        valid_mask = target.view(-1) != self.ignore_index
        losses     = losses[valid_mask]

        if losses.numel() == 0:
            return pred.sum() * 0.0

        losses_sorted, _ = losses.sort(descending=True)

        n_above_thresh = (losses_sorted > self.thresh).sum().item()
        n_keep         = int(max(n_above_thresh, self.min_kept))
        n_keep         = min(n_keep, losses_sorted.numel())

        return self.loss_weight * losses_sorted[:n_keep].mean()


# =============================================================================
# Fog Consistency Loss
# =============================================================================

class FogConsistencyLoss(nn.Module):
    def __init__(self,
                 temperature: float = 4.0,
                 loss_weight: float = 0.1):
        super().__init__()
        self.T           = temperature
        self.loss_weight = loss_weight

    def forward(self,
                logit_light: Tensor,
                logit_heavy: Tensor) -> Tensor:
        assert logit_light.shape == logit_heavy.shape, (
            f"FogConsistencyLoss: shape mismatch "
            f"{logit_light.shape} vs {logit_heavy.shape}"
        )

        # Q = heavy fog distribution (student, được tối ưu)
        # P = light fog distribution (teacher, target)
        # Minimize KL(P || Q) = sum P * log(P/Q)
        log_q = F.log_softmax(logit_heavy / self.T, dim=1)   # log Q (student)
        p     = F.softmax(logit_light  / self.T, dim=1)       # P     (teacher)

        kl = F.kl_div(log_q, p, reduction='batchmean') * (self.T ** 2)

        return self.loss_weight * kl


# =============================================================================
# GCNetHead
# =============================================================================

class GCNetHead(BaseModule):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 loss_weight_aux: float = 0.4,
                 dropout_ratio: float = 0.1,
                 # FIX: thresh default 1.5 (từ 0.7) — xem OHEMCrossEntropyLoss
                 ohem_thresh: float = 1.5,
                 ohem_min_kept: int = 100_000,
                 fog_consistency_weight: float = 0.1,
                 fog_temperature: float = 4.0,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels         = in_channels
        self.channels            = channels
        self.num_classes         = num_classes
        self.norm_cfg            = norm_cfg
        self.act_cfg             = act_cfg
        self.align_corners       = align_corners
        self.ignore_index        = ignore_index
        self.loss_weight_aux     = loss_weight_aux

        # ---- Main head (c6) ---------------------------------------------- #
        self.head = self._make_base_head(in_channels, channels)

        # ---- Auxiliary head (c4) ----------------------------------------- #
        # in_channels // 2: c4_feat là channels*2 = in_channels // 2
        self.aux_head_c4    = self._make_base_head(in_channels // 2, channels)
        self.aux_cls_seg_c4 = nn.Conv2d(channels, num_classes, kernel_size=1)

        # ---- Final classifiers ------------------------------------------- #
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg  = nn.Conv2d(channels, num_classes, kernel_size=1)

        # ---- Loss functions ---------------------------------------------- #
        self.loss_c4 = CrossEntropyLoss(ignore_index=ignore_index,
                                         loss_weight=loss_weight_aux)

        self.loss_c6 = OHEMCrossEntropyLoss(
            ignore_index=ignore_index,
            loss_weight=1.0,
            thresh=ohem_thresh,
            min_kept=ohem_min_kept,
        )

        self.fog_consistency_weight = fog_consistency_weight
        if fog_consistency_weight > 0.0:
            self.loss_fog = FogConsistencyLoss(
                temperature=fog_temperature,
                loss_weight=fog_consistency_weight,
            )
        else:
            self.loss_fog = None

        self.init_weights()

    # ---------------------------------------------------------------------- #
    # Weight init                                                              #
    # ---------------------------------------------------------------------- #

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self,
                inputs: Union[Tensor, Tuple[Tensor, Tensor]]
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.training:
            assert isinstance(inputs, (tuple, list)) and len(inputs) == 2, (
                f"GCNetHead training mode expects (c4_feat, c6_feat) tuple, "
                f"got {type(inputs)}. "
                f"Kiểm tra backbone.forward() có return_aux=True không."
            )
            c4_feat, c6_feat = inputs

            c4_logit = self.aux_cls_seg_c4(self.aux_head_c4(c4_feat))
            c6_logit = self.cls_seg(self.dropout(self.head(c6_feat)))

            return c4_logit, c6_logit

        else:
            # Inference: inputs có thể là tuple (nếu backbone.return_aux=True)
            # hoặc Tensor đơn — handle cả hai
            if isinstance(inputs, (tuple, list)):
                # Lấy c6_feat (index 1), bỏ c4_feat
                c6_feat = inputs[1]
            else:
                c6_feat = inputs
            return self.cls_seg(self.dropout(self.head(c6_feat)))

    # ---------------------------------------------------------------------- #
    # Loss                                                                     #
    # ---------------------------------------------------------------------- #

    def loss(self,
             seg_logits: Tuple[Tensor, Tensor],
             seg_label: Tensor) -> Dict[str, Tensor]:
        c4_logit, c6_logit = seg_logits

        # FIX: normalize seg_label shape → luôn (B, H, W)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        # FIX: dùng shape[-2:] thay vì shape[1:]
        target_size = seg_label.shape[-2:]

        c4_logit = resize(c4_logit, size=target_size,
                          mode='bilinear', align_corners=self.align_corners)
        c6_logit = resize(c6_logit, size=target_size,
                          mode='bilinear', align_corners=self.align_corners)

        losses = {
            'loss_c4': self.loss_c4(c4_logit, seg_label),
            'loss_c6': self.loss_c6(c6_logit, seg_label),
            'acc_seg': accuracy(c6_logit, seg_label,
                                ignore_index=self.ignore_index),
        }
        return losses

    def compute_fog_consistency(self,
                                 logit_light: Tensor,
                                 logit_heavy: Tensor) -> Optional[Tensor]:
        if self.loss_fog is None:
            return None
        return self.loss_fog(logit_light, logit_heavy)

    # ---------------------------------------------------------------------- #
    # Helper                                                                   #
    # ---------------------------------------------------------------------- #

    def _make_base_head(self, in_channels: int, channels: int) -> nn.Sequential:
        return nn.Sequential(
            build_norm_layer(self.norm_cfg, in_channels)[1],   # [0] BN standalone
            build_activation_layer(self.act_cfg),              # [1] ReLU
            ConvModule(                                        # [2] Conv→BN→ReLU
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('conv', 'norm', 'act'),
            ),
        )

    # ---------------------------------------------------------------------- #
    # Inference helper                                                         #
    # ---------------------------------------------------------------------- #

    def predict(self,
                inputs: Union[Tensor, Tuple[Tensor, Tensor]],
                img_size: Optional[Tuple[int, int]] = None) -> Tensor:
        self.eval()
        with torch.no_grad():
            logit = self.forward(inputs)
            if img_size is not None:
                logit = resize(logit, size=img_size,
                               mode='bilinear', align_corners=self.align_corners)
        return logit