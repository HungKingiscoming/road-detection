"""topo_bridge.py
================
Cầu nối giữa CoANet và TopoNet (đã tối ưu hóa tốc độ cực đại).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from skimage.morphology import skeletonize
except ImportError:
  skeletonize = None

from modeling.fusion import fuse_mask_and_graph, rasterize_graph_vectorized
from modeling.toponet import BilinearSampler, TopoNet


@dataclass
class TopoConfig:
  TOPONET_VERSION: str = 'full'
  coord_format: str = 'pixel'
  max_points: int = 128
  mask_threshold: float = 0.5
  min_point_spacing: int = 4
  k_neighbors: int = 5
  max_pairs: Optional[int] = None
  edge_gt_num_samples: int = 8
  edge_gt_mask_ratio: float = 0.8
  graph_mask_sigma: float = 2.0
  graph_mask_score_threshold: float = 0.3
  fusion_mode: str = 'conditional_v2'
  fusion_threshold: float = 0.5

  def __post_init__(self):
    if self.max_pairs is None:
      self.max_pairs = self.max_points * self.k_neighbors


# ============================================================================
# 1. Trích điểm Graph: Dùng Grid Subsampling siêu nhanh
# ============================================================================


def _skeleton_points_fast(
    mask_np: np.ndarray, max_points: int, min_spacing: int
) -> np.ndarray:
  if skeletonize is None or not mask_np.any():
    return np.zeros((0, 2), dtype=np.float32)

  skel = skeletonize(mask_np)
  ys, xs = np.nonzero(skel)
  if len(xs) == 0:
    return np.zeros((0, 2), dtype=np.float32)

  pts = np.stack([xs, ys], axis=1).astype(np.float32)

  if min_spacing > 1 and len(pts) > 1:
    grid_coords = (pts / min_spacing).astype(np.int32)
    _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
    pts = pts[unique_indices]

  if len(pts) > max_points:
    idx = np.random.choice(len(pts), size=max_points, replace=False)
    pts = pts[idx]

  return pts


@torch.no_grad()
def extract_graph_points(
    mask: torch.Tensor, cfg: TopoConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
  B, _, H, W = mask.shape
  device = mask.device

  target_size = (256, 256)
  scale_x = W / target_size[1]
  scale_y = H / target_size[0]

  mask_small = F.interpolate(mask, size=target_size, mode='nearest')
  mask_np = (mask_small.detach().squeeze(1) >= cfg.mask_threshold).cpu().numpy()

  all_points = np.zeros((B, cfg.max_points, 2), dtype=np.float32)
  all_valid = np.zeros((B, cfg.max_points), dtype=bool)

  def process_item(b):
    return b, _skeleton_points_fast(
        mask_np[b], cfg.max_points, cfg.min_point_spacing
    )

  with ThreadPoolExecutor(max_workers=min(B, 4)) as executor:
    results = executor.map(process_item, range(B))

  for b, pts in results:
    n = pts.shape[0]
    if n > 0:
      pts[:, 0] *= scale_x
      pts[:, 1] *= scale_y
      all_points[b, :n] = pts
      all_valid[b, :n] = True

  points = torch.from_numpy(all_points).to(device=device, dtype=torch.float32)
  points_valid = torch.from_numpy(all_valid).to(device=device)
  return points, points_valid


# ============================================================================
# 2. Candidate Edges (GPU Vectorized)
# ============================================================================


@torch.no_grad()
def build_knn_pairs(
    points: torch.Tensor, points_valid: torch.Tensor, cfg: TopoConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
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
# 3. Ground Truth Labels
# ============================================================================


@torch.no_grad()
def compute_edge_gt_labels(
    points: torch.Tensor,
    pairs: torch.Tensor,
    pairs_valid: torch.Tensor,
    gt_mask: torch.Tensor,
    cfg: TopoConfig,
) -> torch.Tensor:
  B, S, P, _ = pairs.shape
  device = points.device
  batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, S, P)

  p1 = points[batch_idx, pairs[..., 0]]
  p2 = points[batch_idx, pairs[..., 1]]

  t = torch.linspace(0, 1, cfg.edge_gt_num_samples, device=device).view(
      1, 1, 1, -1, 1
  )
  seg_pts = p1.unsqueeze(3) * (1 - t) + p2.unsqueeze(3) * t

  H, W = gt_mask.shape[-2:]
  gx = (seg_pts[..., 0] / max(W - 1, 1)) * 2 - 1
  gy = (seg_pts[..., 1] / max(H - 1, 1)) * 2 - 1

  grid = torch.stack([gx, gy], dim=-1).reshape(
      B, S * P * cfg.edge_gt_num_samples, 1, 2
  )
  sampled = F.grid_sample(
      gt_mask.float(), grid, mode='bilinear', align_corners=False
  )
  sampled = sampled.view(B, S, P, cfg.edge_gt_num_samples)

  on_road_ratio = (sampled >= cfg.mask_threshold).float().mean(dim=-1)
  edge_gt = (on_road_ratio >= cfg.edge_gt_mask_ratio).float()
  return edge_gt * pairs_valid.float()


# ============================================================================
# 4. TopoGraphHead Module
# ============================================================================


class TopoGraphHead(nn.Module):

  def __init__(self, feature_dim: int, cfg: TopoConfig):
    super().__init__()
    self.cfg = cfg
    self.sampler = BilinearSampler(cfg)
    self.toponet = TopoNet(cfg, feature_dim)

  def forward(
      self,
      feature_map: torch.Tensor,
      seg_prob: torch.Tensor,
      gt_mask: Optional[torch.Tensor] = None,
  ) -> Dict[str, torch.Tensor]:
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
          points, out['pairs'], out['pairs_valid'], gt_mask, cfg
      )

    return out


# ============================================================================
# 5. Hàm tính Loss cho Topology
# ============================================================================


def compute_topo_loss(
    logits: torch.Tensor, edge_gt: torch.Tensor, pairs_valid: torch.Tensor
) -> torch.Tensor:
  """Tính BCE loss cho nhánh Topo, bỏ qua các padding pairs."""
  logits = logits.squeeze(1).squeeze(-1)  # [B, P]
  edge_gt = edge_gt.squeeze(1)  # [B, P]
  pairs_valid = pairs_valid.squeeze(1)  # [B, P]

  n_valid = pairs_valid.float().sum()
  if n_valid.item() == 0:
    return logits.sum() * 0.0

  loss = F.binary_cross_entropy_with_logits(logits, edge_gt, reduction='none')
  loss = (loss * pairs_valid.float()).sum() / n_valid
  return loss


# ============================================================================
# 6. CoANetWithTopo Module
# ============================================================================


class CoANetWithTopo(nn.Module):

  def __init__(
      self,
      coanet: nn.Module,
      topo_cfg: TopoConfig,
      decoder_feature_dim: int = 64,
  ):
    super().__init__()
    self.coanet = coanet
    self.topo_cfg = topo_cfg
    self.topo_head = TopoGraphHead(decoder_feature_dim, topo_cfg)
    self.freeze_coanet = True

  def set_freeze_coanet(self, freeze: bool = True):
    self.freeze_coanet = freeze
    for p in self.coanet.parameters():
      p.requires_grad = not freeze

  def forward(
      self,
      input: torch.Tensor,
      gt_mask: Optional[torch.Tensor] = None,
      return_aux: bool = False,
  ) -> Dict[str, torch.Tensor]:

    if self.training and self.freeze_coanet:
      with torch.no_grad():
        e1, e2, e3, e4 = self.coanet.backbone(input)
        e4_aspp = self.coanet.aspp(e4)
        feat = self.coanet.decoder(e1, e2, e3, e4_aspp)
        seg_logits, con0, con1 = self.coanet.connect(feat)

      feat = feat.detach()
      seg_logits_detach = seg_logits.detach()

    else:
      e1, e2, e3, e4 = self.coanet.backbone(input)
      e4_aspp = self.coanet.aspp(e4)
      feat = self.coanet.decoder(e1, e2, e3, e4_aspp)
      seg_logits, con0, con1 = self.coanet.connect(feat)
      seg_logits_detach = seg_logits.detach()

    seg_prob = torch.sigmoid(seg_logits_detach)
    topo_out = self.topo_head(feat, seg_prob, gt_mask=gt_mask)

    # --- FIX: fused_mask LUÔN LUÔN là xác suất trong [0, 1], bất kể train/eval ---
    # Trước đây: lúc training fused_mask = seg_logits (logits thô, chưa fuse),
    # lúc eval fused_mask = fuse_mask_and_graph(...) (đã là xác suất, do hàm này
    # tự áp sigmoid bên trong). Hai nhánh trả về 2 loại giá trị khác nhau khiến
    # nơi tiêu thụ (train.py) áp sigmoid thêm 1 lần nữa cho nhánh eval -> lỗi
    # "double sigmoid" (mọi giá trị >= sigmoid(0) = 0.5 -> Recall ~ 1, Precision sập).
    if not self.training:
      with torch.no_grad():
        fused_mask, graph_mask = rasterize_and_fuse(
            seg_logits, topo_out, self.topo_cfg
        )
    else:
      fused_mask = torch.sigmoid(seg_logits)  # xác suất, KHÔNG phải logits thô
      graph_mask = None

    return {
        'seg_logits': seg_logits,
        'connect': con0,
        'connect_d1': con1,
        'fused_mask': fused_mask,
        'graph_mask': graph_mask,
        'topo': topo_out,
    }


def rasterize_and_fuse(
    seg_logits: torch.Tensor, topo_out: Dict, cfg: TopoConfig
):
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
      seg_logits,
      graph_mask,
      threshold=cfg.fusion_threshold,
      mode=cfg.fusion_mode,
  )
  return fused_mask, graph_mask
