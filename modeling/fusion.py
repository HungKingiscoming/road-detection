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
    chunk_size: int = 256        # Tối ưu: Xử lý theo từng chunk để chống OOM
) -> torch.Tensor:
    """
    Chuyển đổi danh sách cạnh đồ thị thưa thành 2D Mask Heatmap [B, 1, H, W].
    Được tối ưu bộ nhớ VRAM bằng kỹ thuật Chunking + torch.no_grad().
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

    # Khởi tạo Mask kết quả tích lũy
    graph_mask = torch.zeros(B, 1, H, W, device=device)

    # Chạy không cần tính Gradient để tiết kiệm tối đa VRAM
    with torch.no_grad():
        batch_idx = torch.arange(B, device=device).unsqueeze(1)

        # Xử lý cắt lớp theo Chunk để kiểm soát Memory Peak
        for start_idx in range(0, N_pairs, chunk_size):
            end_idx = min(start_idx + chunk_size, N_pairs)
            
            p1_chunk = graph_points[batch_idx, pairs[:, start_idx:end_idx, 0]]  # [B, chunk_size, 2]
            p2_chunk = graph_points[batch_idx, pairs[:, start_idx:end_idx, 1]]  # [B, chunk_size, 2]
            act_chunk = active_mask[:, start_idx:end_idx]                        # [B, chunk_size]
            score_chunk = topo_scores[:, start_idx:end_idx]                     # [B, chunk_size]

            p1_c = p1_chunk.unsqueeze(2).unsqueeze(2)  # [B, chunk_size, 1, 1, 2]
            p2_c = p2_chunk.unsqueeze(2).unsqueeze(2)  # [B, chunk_size, 1, 1, 2]

            v = p2_c - p1_c
            v_len2 = (v ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-6)

            grid_dir = grid - p1_c
            t = (grid_dir * v).sum(dim=-1, keepdim=True) / v_len2
            t = t.clamp(0.0, 1.0)

            proj = p1_c + t * v
            dist2 = ((grid - proj) ** 2).sum(dim=-1)  # [B, chunk_size, H, W]

            scores = (score_chunk * act_chunk.float()).unsqueeze(-1).unsqueeze(-1)  # [B, chunk_size, 1, 1]
            line_resp = torch.exp(-dist2 / (2.0 * (sigma ** 2))) * scores             # [B, chunk_size, H, W]

            # Cập nhật MAX tích lũy qua từng chunk
            if line_resp.shape[1] > 0:
                chunk_max, _ = line_resp.max(dim=1, keepdim=True)  # [B, 1, H, W]
                graph_mask = torch.maximum(graph_mask, chunk_max)

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
