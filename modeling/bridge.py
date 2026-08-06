import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

from modeling.coanet import CoANet
from modeling.toponet import TopoNet, BilinearSampler
from modeling.fusion import rasterize_graph_vectorized, fuse_mask_and_graph


# ============================================================================
# Config
# ============================================================================

@dataclass
class TopoConfig:
    # --- Bắt buộc bởi TopoNet / BilinearSampler ---
    TOPONET_VERSION: str = 'full'       # 'full' | 'no_tgt_features' | 'no_offset' | 'no_transformer'
    coord_format: str = 'pixel'         # 'pixel' | 'norm01' | 'norm11'

    # --- Trích điểm graph từ seg mask ---
    max_points: int = 128               # Số điểm graph tối đa mỗi ảnh
    mask_threshold: float = 0.5         # Ngưỡng nhị phân hoá seg mask trước khi skeletonize
    min_point_spacing: int = 4          # Khoảng cách tối thiểu (pixel) giữa 2 điểm graph (NMS thưa)

    # --- Sinh candidate edges ---
    k_neighbors: int = 5                # Số hàng xóm gần nhất mỗi điểm dùng để tạo candidate edge
    max_pairs: Optional[int] = None     # None -> tự set = max_points * k_neighbors

    # --- Gán nhãn GT cho candidate edges (suy từ GT mask) ---
    edge_gt_num_samples: int = 8        # Số điểm lấy mẫu dọc theo 1 cạnh để kiểm tra
    edge_gt_mask_ratio: float = 0.8     # Tỉ lệ điểm dọc cạnh phải nằm trong GT mask -> coi là cạnh thật

    # --- Rasterize graph -> mask & fusion ---
    graph_mask_sigma: float = 2.0
    graph_mask_score_threshold: float = 0.3
    fusion_mode: str = 'gated_v3'        # 'baseline_v0' | 'union_v1' | 'conditional_v2' | 'gated_v3'
    fusion_threshold: float = 0.5

    def __post_init__(self):
        if self.max_pairs is None:
            self.max_pairs = self.max_points * self.k_neighbors


# ============================================================================
# 1. Trích điểm graph từ seg mask (skeletonize, không differentiable)
# ============================================================================

def _skeleton_points_single(mask_np: np.ndarray, max_points: int, min_spacing: int) -> np.ndarray:
    if skeletonize is None:
        raise ImportError(
            "Cần cài scikit-image để dùng skeleton-based point sampling: "
            "pip install scikit-image"
        )
    if not mask_np.any():
        return np.zeros((0, 2), dtype=np.float32)

    skel = skeletonize(mask_np)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    if min_spacing > 0 and len(pts) > 1:
        rng = np.random.RandomState(0)
        order = rng.permutation(len(pts))
        kept = []
        for i in order:
            p = pts[i]
            if kept:
                occ = np.stack(kept, axis=0)
                d = np.sqrt(((occ - p) ** 2).sum(axis=1))
                if d.min() < min_spacing:
                    continue
            kept.append(p)
            if len(kept) >= max_points:
                break
        pts = np.stack(kept, axis=0) if kept else np.zeros((0, 2), dtype=np.float32)

    if len(pts) > max_points:
        idx = np.random.RandomState(0).choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


@torch.no_grad()
def extract_graph_points(mask: torch.Tensor, cfg: TopoConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    B = mask.shape[0]
    device = mask.device
    mask_np = (mask.detach().squeeze(1) >= cfg.mask_threshold).cpu().numpy()

    all_points = np.zeros((B, cfg.max_points, 2), dtype=np.float32)
    all_valid = np.zeros((B, cfg.max_points), dtype=bool)

    for b in range(B):
        pts = _skeleton_points_single(mask_np[b], cfg.max_points, cfg.min_point_spacing)
        n = pts.shape[0]
        if n > 0:
            all_points[b, :n] = pts
            all_valid[b, :n] = True

    points = torch.from_numpy(all_points).to(device=device, dtype=torch.float32)
    points_valid = torch.from_numpy(all_valid).to(device=device)
    return points, points_valid


# ============================================================================
# 2. Sinh candidate edges (k-NN) giữa các điểm
# ============================================================================

@torch.no_grad()
def build_knn_pairs(points: torch.Tensor, points_valid: torch.Tensor,
                     cfg: TopoConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = points.shape
    device = points.device
    pairs = torch.zeros(B, cfg.max_pairs, 2, dtype=torch.long, device=device)
    valid = torch.zeros(B, cfg.max_pairs, dtype=torch.bool, device=device)

    k = min(cfg.k_neighbors, N - 1) if N > 1 else 0
    if k <= 0:
        return pairs.unsqueeze(1), valid.unsqueeze(1)

    dist = torch.cdist(points, points)
    invalid = ~points_valid
    dist = dist.masked_fill(invalid.unsqueeze(1), float('inf'))
    dist = dist.masked_fill(invalid.unsqueeze(2), float('inf'))
    dist.diagonal(dim1=1, dim2=2).fill_(float('inf'))

    knn_dist, knn_idx = dist.topk(k, dim=-1, largest=False)

    src = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
    cand_pairs = torch.stack([src, knn_idx], dim=-1).reshape(B, N * k, 2)
    cand_valid = knn_dist.reshape(B, N * k) < float('inf')

    i_idx, j_idx = cand_pairs[..., 0], cand_pairs[..., 1]
    cand_valid = cand_valid & (i_idx < j_idx)

    n_take = min(cand_pairs.shape[1], cfg.max_pairs)
    pairs[:, :n_take] = cand_pairs[:, :n_take]
    valid[:, :n_take] = cand_valid[:, :n_take]

    return pairs.unsqueeze(1), valid.unsqueeze(1)


# ============================================================================
# 3. Gán nhãn GT cho candidate edges (suy từ GT mask)
# ============================================================================

@torch.no_grad()
def compute_edge_gt_labels(points: torch.Tensor, pairs: torch.Tensor, pairs_valid: torch.Tensor,
                            gt_mask: torch.Tensor, cfg: TopoConfig) -> torch.Tensor:
    B, S, P, _ = pairs.shape
    device = points.device
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, S, P)
    p1 = points[batch_idx, pairs[..., 0]]
    p2 = points[batch_idx, pairs[..., 1]]

    t = torch.linspace(0, 1, cfg.edge_gt_num_samples, device=device).view(1, 1, 1, -1, 1)
    seg_pts = p1.unsqueeze(3) * (1 - t) + p2.unsqueeze(3) * t

    H, W = gt_mask.shape[-2:]
    gx = (seg_pts[..., 0] / max(W - 1, 1)) * 2 - 1
    gy = (seg_pts[..., 1] / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).reshape(B, S * P * cfg.edge_gt_num_samples, 1, 2)

    sampled = F.grid_sample(gt_mask.float(), grid, mode='bilinear', align_corners=False)
    sampled = sampled.view(B, S, P, cfg.edge_gt_num_samples)

    on_road_ratio = (sampled >= cfg.mask_threshold).float().mean(dim=-1)
    edge_gt = (on_road_ratio >= cfg.edge_gt_mask_ratio).float()
    edge_gt = edge_gt * pairs_valid.float()
    return edge_gt


# ============================================================================
# 4. TopoGraphHead
# ============================================================================

class TopoGraphHead(nn.Module):
    def __init__(self, feature_dim: int, cfg: TopoConfig):
        super().__init__()
        self.cfg = cfg
        self.sampler = BilinearSampler(cfg)
        self.toponet = TopoNet(cfg, feature_dim)

    def forward(self, feature_map: torch.Tensor, seg_prob: torch.Tensor,
                gt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        points, points_valid = extract_graph_points(seg_prob, cfg)
        pairs, pairs_valid = build_knn_pairs(points, points_valid, cfg)
        point_features = self.sampler(feature_map, points, coord_format='pixel')

        logits, scores = self.toponet(points, point_features, pairs, pairs_valid)

        p_out = logits.shape[2]
        out = {
            'points': points,
            'points_valid': points_valid,
            'pairs': pairs[:, :, :p_out],
            'pairs_valid': pairs_valid[:, :, :p_out],
            'logits': logits,
            'scores': scores,
        }

        if gt_mask is not None:
            out['edge_gt'] = compute_edge_gt_labels(
                points, out['pairs'], out['pairs_valid'], gt_mask, cfg)

        return out


# ============================================================================
# 5. Rasterize graph & fusion
# ============================================================================

def build_fused_mask(seg_logits: torch.Tensor, topo_out: Dict[str, torch.Tensor],
                      cfg: TopoConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W = seg_logits.shape[-2:]
    graph_mask = rasterize_graph_vectorized(
        graph_points=topo_out['points'],
        pairs=topo_out['pairs'],
        valid=topo_out['pairs_valid'],
        topo_scores=topo_out['scores'],
        out_size=(H, W),
        sigma=cfg.graph_mask_sigma,
        score_threshold=cfg.graph_mask_score_threshold,
    )
    fused_mask = fuse_mask_and_graph(
        seg_logits, graph_mask,
        threshold=cfg.fusion_threshold,
        mode=cfg.fusion_mode,
    )
    return fused_mask, graph_mask


# ============================================================================
# 6. Loss cho nhánh topology
# ============================================================================

def compute_topo_loss(logits: torch.Tensor, edge_gt: torch.Tensor,
                       pairs_valid: torch.Tensor, pos_weight: float = 3.0) -> torch.Tensor:
    logits = logits.squeeze(1).squeeze(-1)
    edge_gt = edge_gt.squeeze(1)
    pairs_valid = pairs_valid.squeeze(1)

    n_valid = pairs_valid.float().sum()
    if n_valid.item() == 0:
        return logits.sum() * 0.0

    weight_tensor = torch.ones_like(edge_gt)
    weight_tensor[edge_gt == 1.0] = pos_weight

    loss = F.binary_cross_entropy_with_logits(logits, edge_gt, weight=weight_tensor, reduction='none')
    loss = (loss * pairs_valid.float()).sum() / n_valid
    return loss


# ============================================================================
# 7. Model tổng: CoANetWithTopo
# ============================================================================

class CoANetWithTopo(nn.Module):
    """
    Bọc CoANet + TopoGraphHead + Fusion Module.
    Đã điều chỉnh linh hoạt: Có thể truyền vào 1 instance CoANet HOẶC truyền các arguments đơn lẻ.
    """
    def __init__(self, 
                 coanet: Optional[Union[nn.Module, str]] = None, 
                 topo_cfg: Optional[TopoConfig] = None, 
                 decoder_feature_dim: int = 64,
                 backbone: str = 'gcnet',
                 num_classes: int = 1,
                 num_neighbor: int = 9,
                 sync_bn: bool = False,
                 freeze_bn: bool = False,
                 convnext_model_name: str = 'convnextv2_tiny.fcmae_ft_in22k_in1k',
                 pretrained: bool = True,
                 max_points: int = 128,
                 k_neighbors: int = 5):
        super().__init__()

        # 1. Khởi tạo hoặc gán CoANet
        if isinstance(coanet, nn.Module):
            self.coanet = coanet
        else:
            # Nếu coanet truyền vào là string (hoặc None) thì dùng các argument để tự khởi tạo CoANet
            actual_backbone = coanet if isinstance(coanet, str) else backbone
            self.coanet = CoANet(
                backbone=actual_backbone,
                num_classes=num_classes,
                num_neighbor=num_neighbor,
                sync_bn=sync_bn,
                freeze_bn=freeze_bn,
                convnext_model_name=convnext_model_name,
                pretrained=pretrained
            )

        # 2. Cấu hình TopoConfig
        if topo_cfg is None:
            self.topo_cfg = TopoConfig(max_points=max_points, k_neighbors=k_neighbors)
        else:
            self.topo_cfg = topo_cfg

        # 3. Khởi tạo TopoGraphHead
        self.topo_head = TopoGraphHead(decoder_feature_dim, self.topo_cfg)

    def forward(self, input: torch.Tensor, gt_mask: Optional[torch.Tensor] = None,
                return_aux: bool = False) -> Dict[str, torch.Tensor]:
        e1, e2, e3, e4, x_spp, fused_backbone, c4_feat = self.coanet.backbone(
            input, return_aux=(return_aux or self.training))

        feat = self.coanet.decoder(e1, e2, e3, e4)
        seg_logits, con0, con1 = self.coanet.connect(feat)

        seg_prob = torch.sigmoid(seg_logits.detach())
        topo_out = self.topo_head(feat, seg_prob, gt_mask=gt_mask)

        fused_mask, graph_mask = build_fused_mask(seg_logits, topo_out, self.topo_cfg)

        result = {
            'seg_logits': seg_logits,
            'connect': con0,
            'connect_d1': con1,
            'fused_mask': fused_mask,
            'graph_mask': graph_mask,
            'topo': topo_out,
        }

        if (return_aux or self.training) and c4_feat is not None:
            aux_seg = self.coanet.aux_head(c4_feat)
            aux_seg = F.interpolate(aux_seg, size=input.shape[2:], mode='bilinear', align_corners=True)
            result['aux_seg'] = aux_seg

        return result
