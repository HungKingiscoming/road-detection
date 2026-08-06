import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def rasterize_graph_vectorized(
    graph_points: torch.Tensor,  # [B, N_points, 2]
    pairs: torch.Tensor,         # [B, N_pairs, 2] hoặc [B, N_samples, N_pairs, 2]
    valid: torch.Tensor,         # [B, N_pairs] hoặc [B, N_samples, N_pairs]
    topo_scores: torch.Tensor,   # [B, N_pairs] hoặc [B, N_samples, N_pairs, 1]
    out_size: Tuple[int, int],   # (H, W)
    sigma: float = 2.0,
    score_threshold: float = 0.3,
    chunk_size: int = 32         # Giảm từ 256 xuống 32 để chống bùng nổ Peak VRAM
) -> torch.Tensor:
    """
    Chuyển đổi danh sách cạnh đồ thị thưa thành 2D Mask Heatmap [B, 1, H, W].
    Được tối ưu bộ nhớ VRAM triệt để bằng cách tách Batch + Micro-Chunking.
    """
    B = graph_points.shape[0]
    H, W = out_size
    device = graph_points.device

    # Flatten về dạng 2D [B, N_flat_pairs]
    if pairs.dim() == 4:
        pairs = pairs.view(B, -1, 2)
    if valid.dim() == 3:
        valid = valid.view(B, -1)
    if topo_scores.dim() >= 3:
        topo_scores = topo_scores.view(B, -1)

    N_pairs = pairs.shape[1]
    if N_pairs == 0:
        return torch.zeros(B, 1, H, W, device=device)

    # Lọc các cạnh hợp lệ
    active_mask = (valid > 0) & (topo_scores >= score_threshold)  # [B, N_pairs]

    # Khởi tạo Grid không gian [1, 1, H, W, 2]
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]

    graph_mask = torch.zeros(B, 1, H, W, device=device)

    # Đảm bảo không lưu Autograd Graph
    with torch.no_grad():
        # Xử lý từng mẫu trong Batch riêng biệt để tránh nhân kích thước Tensor với B
        for b in range(B):
            b_act = active_mask[b]  # [N_pairs]
            if not b_act.any():
                continue

            b_pairs = pairs[b, b_act]            # [N_valid_pairs, 2]
            b_scores = topo_scores[b, b_act]      # [N_valid_pairs]
            b_points = graph_points[b]           # [N_points, 2]

            N_valid = b_pairs.shape[0]

            # Xử lý theo đợt nhỏ (Micro-chunk)
            for start_idx in range(0, N_valid, chunk_size):
                end_idx = min(start_idx + chunk_size, N_valid)

                p1 = b_points[b_pairs[start_idx:end_idx, 0]]  # [chunk, 2]
                p2 = b_points[b_pairs[start_idx:end_idx, 1]]  # [chunk, 2]
                sc = b_scores[start_idx:end_idx]               # [chunk]

                p1_c = p1.unsqueeze(1).unsqueeze(1)  # [chunk, 1, 1, 2]
                p2_c = p2.unsqueeze(1).unsqueeze(1)  # [chunk, 1, 1, 2]
                sc_c = sc.unsqueeze(-1).unsqueeze(-1) # [chunk, 1, 1]

                v = p2_c - p1_c
                v_len2 = (v ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-6)

                grid_dir = grid - p1_c
                t = ((grid_dir * v).sum(dim=-1, keepdim=True) / v_len2).clamp(0.0, 1.0)

                proj = p1_c + t * v
                dist2 = ((grid - proj) ** 2).sum(dim=-1)  # [chunk, H, W]

                line_resp = torch.exp(-dist2 / (2.0 * (sigma ** 2))) * sc_c  # [chunk, H, W]

                chunk_max, _ = line_resp.max(dim=0)  # Shape: [H, W]
                graph_mask[b, 0] = torch.maximum(graph_mask[b, 0], chunk_max)

                # Thu gom rác bộ nhớ ngay trong loop
                del grid_dir, proj, dist2, line_resp, chunk_max

    return graph_mask


def fuse_mask_and_graph(
    seg_logits: torch.Tensor,       # [B, 1, H, W] từ Connect Head
    graph_mask: torch.Tensor,       # [B, 1, H, W] từ rasterize_graph_vectorized
    threshold: float = 0.5,
    mode: str = 'conditional_v2'
) -> torch.Tensor:
    """
    Hợp nhất Segmentation Mask (Dense) và Graph Mask (Sparse đã rasterize).
    
    Modes:
      - 'baseline_v0':     Chỉ lấy Seg Mask gốc.
      - 'union_v1':        Union thô max(seg, graph). (Rủi ro FP cao nếu graph nhiễu).
      - 'conditional_v2':  Chỉ cho Graph lấp vào vùng Seg nghi ngờ (low-confidence < threshold).
    """
    seg_prob = torch.sigmoid(seg_logits)

    if mode == 'baseline_v0':
        return seg_prob

    elif mode == 'union_v1':
        return torch.maximum(seg_prob, graph_mask)

    elif mode == 'conditional_v2':
        # Chỉ lấp vào vùng nghi ngờ (seg_prob < threshold)
        low_confidence_zone = (seg_prob < threshold).float()
        
        # Mức độ đóng góp của Graph tỉ lệ nghịch với độ tự tin của Seg Mask: (1 - seg_prob)
        final_mask = seg_prob + low_confidence_zone * graph_mask * (1.0 - seg_prob)
        return torch.clamp(final_mask, 0.0, 1.0)

    else:
        raise ValueError(f"Unsupported fusion mode: {mode}")
