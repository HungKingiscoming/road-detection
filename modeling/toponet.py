import torch
import torch.nn.functional as F
from torch import nn

# Only needed for the ablation experiment of using a ViT-B model without SA-1B pre-training.
# It depends on detectron2 library. Not super important. 
# import vitdet


class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points, coord_format=None):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2].
            coord_format (str): One of {'norm11', 'norm01', 'pixel'}.
        Returns:
            Tensor: Sampled feature vectors of shape [B, N_points, D].
        """
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape

        coord_format = self.config.coord_format if coord_format is None else coord_format
        sample_points = sample_points.clone()
        sample_points_x = sample_points[..., 0]
        sample_points_y = sample_points[..., 1]

        if coord_format == 'norm11':
            # Coordinates already in [-1, 1].
            pass
        elif coord_format == 'norm01':
            sample_points_x = sample_points_x * 2.0 - 1.0
            sample_points_y = sample_points_y * 2.0 - 1.0
        elif coord_format == 'pixel':
            sample_points_x = (sample_points_x / W) * 2.0 - 1.0
            sample_points_y = (sample_points_y / H) * 2.0 - 1.0
        else:
            raise ValueError(f"coord_format='{coord_format}' không được hỗ trợ!")

        sample_points = torch.stack([sample_points_x, sample_points_y], dim=-1)

        # sample_points from [B, N_points, 2] to [B, N_points, 1, 2] for grid_sample
        sample_points = sample_points.unsqueeze(2)

        # Use grid_sample for bilinear sampling. Align_corners set to False to use -1 to 1 grid space.
        # [B, D, N_points, 1]
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)

        # sampled_features is [B, N_points, D]
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features
    

class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim)

        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid):
        point_features = F.relu(self.feature_proj(point_features))
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)
        
        batch_indices = torch.arange(batch_size, device=points.device).view(-1, 1).expand(-1, n_samples * n_pairs)
        
        src_features = point_features[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        src_points = points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        offset = tgt_points - src_points

        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.cat([src_features, torch.zeros_like(tgt_features), offset], dim=2)
        elif self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.cat([src_features, tgt_features, torch.zeros_like(offset)], dim=2)
        else:
            pair_features = torch.cat([src_features, tgt_features, offset], dim=2)
        
        pair_features = F.relu(self.pair_proj(pair_features))
        
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # SỬA: Chống NaN chuẩn xác khi một sample có n_valid = 0
        all_invalid_samples = (pairs_valid.sum(dim=-1) == 0)
        if all_invalid_samples.any():
            pairs_valid = pairs_valid.clone()
            pairs_valid[all_invalid_samples, 0] = True  # Mở giả định 1 điểm để không bị bẫy Mask All-False

        padding_mask = ~pairs_valid
        
        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)
        
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        logits = self.output_proj(pair_features)
        scores = torch.sigmoid(logits)

        return logits, scores
