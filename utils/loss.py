from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Dice loss thuần
# ============================================================================

def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0, multiclass: bool = False) -> torch.Tensor:
    # Ep contiguous de tranh loi alignment CUDA trong Multi-GPU
    logits = logits.contiguous()
    target = target.contiguous()

    if multiclass:
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        if target.dim() == 3:
            target_oh = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        else:
            target_oh = target.float()
    else:
        probs = torch.sigmoid(logits)
        target_oh = target.float()

    dims = (0, 2, 3)
    intersection = (probs * target_oh).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_oh.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


# ============================================================================
# 2. Loss cho segmentation nhị phân (seg_out_channels == 1)
# ============================================================================

class BCEDiceLoss(nn.Module):
    """BCE + Dice cho segmentation nhị phân (road vs background, C=1)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: Optional[float] = None, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ép contiguous() và float() để bảo đảm tính tương thích bộ nhớ trên GPU
        logits = logits.contiguous().float()
        target = target.contiguous().float()

        pw = None
        if self.pos_weight is not None:
            pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype).contiguous()

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        dice = dice_loss(logits, target, smooth=self.smooth, multiclass=False)
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================================
# 3. Loss cho segmentation nhiều lớp (seg_out_channels > 1, num_classes > 2)
# ============================================================================

class CEDiceLoss(nn.Module):
    """CrossEntropy + Dice cho segmentation nhiều lớp. target: [B,H,W] nhãn long."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 class_weights: Optional[Union[list, torch.Tensor]] = None,
                 smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer('class_weights', torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous().float()
        target = target.contiguous()

        w = None
        if self.class_weights is not None:
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype).contiguous()

        ce = F.cross_entropy(logits, target.long(), weight=w, ignore_index=self.ignore_index)
        dice = dice_loss(logits, target, smooth=self.smooth, multiclass=True)
        return self.ce_weight * ce + self.dice_weight * dice


# ============================================================================
# 4. Wrapper tự chọn binary/multi-class theo số kênh logits
# ============================================================================

class SegmentationLoss(nn.Module):
    """
    Dùng cho `seg` và `aux_seg` (cùng số kênh = seg_out_channels trong CoANet).
    Tự chọn BCEDiceLoss nếu logits.shape[1] == 1, ngược lại dùng CEDiceLoss.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: Optional[float] = None,
                 class_weights: Optional[Union[list, torch.Tensor]] = None,
                 smooth: float = 1.0):
        super().__init__()
        self.binary_loss = BCEDiceLoss(bce_weight, dice_weight, pos_weight, smooth)
        self.multiclass_loss = CEDiceLoss(bce_weight, dice_weight, class_weights, smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return self.binary_loss(logits, target)
        return self.multiclass_loss(logits, target)


# ============================================================================
# 5. Loss cho connect / connect_d1 (multi-label)
# ============================================================================

class ConnectivityLoss(nn.Module):
    """
    Dùng cho `connect` và `connect_d1`, shape [B, num_neighbor, H, W].
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 pos_weight: Optional[Union[float, list, torch.Tensor]] = None,
                 smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous().float()
        target = target.contiguous().float()

        pw = None
        if self.pos_weight is not None:
            pw = self.pos_weight.to(device=logits.device, dtype=logits.dtype).contiguous()

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        dice = dice_loss(logits, target, smooth=self.smooth, multiclass=False)
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================================
# 6. Loss tổng: khớp với output của model
# ============================================================================

class TopoLoss(nn.Module):
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return logits.sum() * 0.0


class CoANetLoss(nn.Module):
    def __init__(self,
                 seg_bce_weight: float = 1.0,
                 seg_dice_weight: float = 1.0,
                 seg_pos_weight: Optional[float] = None,
                 class_weights: Optional[Union[list, torch.Tensor]] = None,
                 connect_bce_weight: float = 1.0,
                 connect_dice_weight: float = 1.0,
                 connect_pos_weight: Optional[Union[float, list, torch.Tensor]] = None,
                 seg_loss_weight: float = 1.0,
                 connect_loss_weight: float = 1.0,
                 connect_d1_loss_weight: float = 1.0,
                 aux_loss_weight: float = 0.4,
                 smooth: float = 1.0):
        super().__init__()

        self.seg_criterion = SegmentationLoss(
            bce_weight=seg_bce_weight, dice_weight=seg_dice_weight,
            pos_weight=seg_pos_weight, class_weights=class_weights, smooth=smooth)

        self.aux_criterion = SegmentationLoss(
            bce_weight=seg_bce_weight, dice_weight=seg_dice_weight,
            pos_weight=seg_pos_weight, class_weights=class_weights, smooth=smooth)

        self.connect_criterion = ConnectivityLoss(
            bce_weight=connect_bce_weight, dice_weight=connect_dice_weight,
            pos_weight=connect_pos_weight, smooth=smooth)
        self.connect_d1_criterion = ConnectivityLoss(
            bce_weight=connect_bce_weight, dice_weight=connect_dice_weight,
            pos_weight=connect_pos_weight, smooth=smooth)

        self.seg_loss_weight = seg_loss_weight
        self.connect_loss_weight = connect_loss_weight
        self.connect_d1_loss_weight = connect_d1_loss_weight
        self.aux_loss_weight = aux_loss_weight

    def forward(self,
                seg: torch.Tensor,
                connect: torch.Tensor,
                connect_d1: torch.Tensor,
                aux_seg: Optional[torch.Tensor],
                gt_seg: torch.Tensor,
                gt_connect: torch.Tensor,
                gt_connect_d1: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # Căn chỉnh kích thước Ground Truth
        gt_seg_matched = self._match_size(gt_seg, seg)
        gt_connect_matched = self._match_size(gt_connect, connect)
        gt_connect_d1_matched = self._match_size(gt_connect_d1, connect_d1)

        loss_seg = self.seg_criterion(seg, gt_seg_matched)
        loss_connect = self.connect_criterion(connect, gt_connect_matched)
        loss_connect_d1 = self.connect_d1_criterion(connect_d1, gt_connect_d1_matched)

        total = (self.seg_loss_weight * loss_seg +
                 self.connect_loss_weight * loss_connect +
                 self.connect_d1_loss_weight * loss_connect_d1)

        loss_dict = {
            'loss_seg': loss_seg.detach(),
            'loss_connect': loss_connect.detach(),
            'loss_connect_d1': loss_connect_d1.detach(),
        }

        # Xử lý aux_seg nếu có
        if aux_seg is not None:
            gt_aux_matched = self._match_size(gt_seg, aux_seg)
            loss_aux = self.aux_criterion(aux_seg, gt_aux_matched)
            total = total + self.aux_loss_weight * loss_aux
            loss_dict['loss_aux'] = loss_aux.detach()

        loss_dict['loss_total'] = total.detach()
        return total, loss_dict

    @staticmethod
    def _match_size(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Resize GT về đúng kích thước pred nếu bị lệch, bảo đảm tính liên tục bộ nhớ."""
        gt = gt.contiguous()
        pred = pred.contiguous()

        if gt.shape[-2:] == pred.shape[-2:]:
            return gt

        if gt.dim() == 3:  # [B,H,W] nhãn lớp (multi-class index)
            gt = gt.unsqueeze(1).float()
            gt = F.interpolate(gt, size=pred.shape[-2:], mode='nearest')
            return gt.squeeze(1).long().contiguous()

        mode = 'nearest' if gt.dtype in (torch.long, torch.int64, torch.bool) else 'bilinear'
        kwargs = {} if mode == 'nearest' else {'align_corners': True}
        return F.interpolate(gt.float(), size=pred.shape[-2:], mode=mode, **kwargs).contiguous()
