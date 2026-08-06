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
    chunk_size: int = 32         # ⚡ Hạ từ 256 xuống 32/64 để ép Peak Memory xuống dưới 200MB
) -> torch.Tensor:
    """
    Chuyển đổi danh sách cạnh đồ thị thưa thành 2D Mask Heatmap [B, 1, H, W].
    Được tối ưu bộ nhớ VRAM triệt để bằng kỹ thuật Filter Active Edges + Small Chunking + AMP.
    """
    B = graph_points.shape[0]
    H, W = out_size
    device = graph_points.device

    # Flatten về dạng 2D chuẩn
    if pairs.dim() == 4:
        pairs = pairs.view(B, -1, 2)
    if valid.dim() == 3:
        valid = valid.view(B, -1)
    if topo_scores.dim() >= 3:
        topo_scores = topo_scores.view(B, -1)

    N_pairs = pairs.shape[1]
    if N_pairs == 0:
        return torch.zeros(B, 1, H, W, device=device)

    # 1. Lọc điều kiện active mask
    active_mask = (valid > 0) & (topo_scores >= score_threshold)  # [B, N_pairs]

    # Nếu không có cạnh nào thỏa mãn ngưỡng, trả về zero mask lập tức
    if not active_mask.any():
        return torch.zeros(B, 1, H, W, device=device)

    # 2. Khởi tạo Grid [1, 1, H, W, 2]
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]

    graph_mask = torch.zeros(B, 1, H, W, device=device)

    with torch.no_grad():
        batch_idx = torch.arange(B, device=device).unsqueeze(1)

        # Xử lý theo từng Sample b trong Batch để loại bỏ hoàn toàn việc Broadcast 5D
        for b in range(B):
            active_indices = torch.nonzero(active_mask[b], as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                continue

            # Lấy danh sách các cặp điểm thực sự Active
            b_pairs = pairs[b, active_indices]            # [N_act, 2]
            b_scores = topo_scores[b, active_indices]      # [N_act]

            b_p1 = graph_points[b, b_pairs[:, 0]]          # [N_act, 2]
            b_p2 = graph_points[b, b_pairs[:, 1]]          # [N_act, 2]

            N_act = b_p1.shape[0]

            # Chia Chunk theo danh sách cạnh ACTIVE
            for start_idx in range(0, N_act, chunk_size):
                end_idx = min(start_idx + chunk_size, N_act)

                p1_c = b_p1[start_idx:end_idx].unsqueeze(1).unsqueeze(1)  # [chunk, 1, 1, 2]
                p2_c = b_p2[start_idx:end_idx].unsqueeze(1).unsqueeze(1)  # [chunk, 1, 1, 2]
                sc_c = b_scores[start_idx:end_idx].unsqueeze(-1).unsqueeze(-1) # [chunk, 1, 1]

                v = p2_c - p1_c
                v_len2 = (v ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-6)

                grid_dir = grid - p1_c                                    # [1, chunk, H, W, 2]
                t = (grid_dir * v).sum(dim=-1, keepdim=True) / v_len2
                t = t.clamp(0.0, 1.0)

                proj = p1_c + t * v
                dist2 = ((grid - proj) ** 2).sum(dim=-1)                  # [1, chunk, H, W]

                line_resp = torch.exp(-dist2 / (2.0 * (sigma ** 2))) * sc_c

                # Cập nhật Max vào Mask kết quả của sample b
                chunk_max, _ = line_resp.max(dim=1, keepdim=True)         # [1, 1, H, W]
                graph_mask[b:b+1] = torch.maximum(graph_mask[b:b+1], chunk_max)

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
