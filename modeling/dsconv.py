"""Dynamic Snake Convolution adapted for CoANet's strip-convolution module.

The implementation intentionally returns a *linear* branch output.  CoANet
already applies BatchNorm and ReLU after concatenating the four SCM branches;
adding GroupNorm/ReLU inside every DSConv branch would destroy the behaviour
learned by an existing CoANet checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv2d(nn.Module):
    """One orientation of 2-D Dynamic Snake Convolution.

    ``morph=0`` starts from a horizontal K-point kernel and learns vertical
    displacements. ``morph=1`` starts from a vertical kernel and learns
    horizontal displacements. The output has the same HxW shape as the input.
    """

    def __init__(self, in_channels, out_channels, kernel_size=9,
                 extend_scope=1.0, morph=0, if_offset=True):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("DSConv2d requires an odd kernel_size")
        if morph not in (0, 1):
            raise ValueError("morph must be 0 (horizontal) or 1 (vertical)")

        self.kernel_size = kernel_size
        self.extend_scope = float(extend_scope)
        self.morph = morph
        self.if_offset = if_offset

        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size, kernel_size=3, padding=1
        )
        if morph == 0:
            # Samples are packed along H, then collapsed back to H.
            sample_kernel = (kernel_size, 1)
            sample_stride = (kernel_size, 1)
        else:
            # Samples are packed along W, then collapsed back to W.
            sample_kernel = (1, kernel_size)
            sample_stride = (1, kernel_size)
        self.sample_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=sample_kernel,
            stride=sample_stride, padding=0, bias=True
        )

        # Zero offset makes the module start as the corresponding pretrained
        # strip convolution. This is the key to stable transfer learning.
        self.reset_offset_parameters()

    def reset_offset_parameters(self):
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    @staticmethod
    def _accumulate_from_center(offset):
        """Apply DSCNet's topology constraint without in-place autograd ops."""
        kernel_size = offset.shape[1]
        center = kernel_size // 2
        left = torch.flip(
            torch.cumsum(torch.flip(offset[:, :center], dims=(1,)), dim=1),
            dims=(1,),
        )
        middle = torch.zeros_like(offset[:, center:center + 1])
        right = torch.cumsum(offset[:, center + 1:], dim=1)
        return torch.cat((left, middle, right), dim=1)

    @staticmethod
    def _normalize_grid(coordinate, maximum):
        if maximum <= 0:
            return torch.zeros_like(coordinate)
        return coordinate.mul(2.0 / maximum).sub(1.0)

    def _coordinate_grid(self, offset, height, width):
        batch_size = offset.shape[0]
        dtype = offset.dtype
        device = offset.device
        k = self.kernel_size
        center = k // 2

        y_offset, x_offset = torch.split(offset, k, dim=1)
        base_y = torch.arange(height, dtype=dtype, device=device).view(1, 1, height, 1)
        base_x = torch.arange(width, dtype=dtype, device=device).view(1, 1, 1, width)

        if self.morph == 0:
            spread_x = torch.arange(-center, center + 1, dtype=dtype, device=device)
            y = base_y.expand(batch_size, k, height, width)
            x = base_x.add(spread_x.view(1, k, 1, 1)).expand(batch_size, k, height, width)
            if self.if_offset:
                y = y + self._accumulate_from_center(y_offset) * self.extend_scope
            # [B,K,H,W] -> [B,H*K,W]
            y = y.permute(0, 2, 1, 3).reshape(batch_size, height * k, width)
            x = x.permute(0, 2, 1, 3).reshape(batch_size, height * k, width)
        else:
            spread_y = torch.arange(-center, center + 1, dtype=dtype, device=device)
            y = base_y.add(spread_y.view(1, k, 1, 1)).expand(batch_size, k, height, width)
            x = base_x.expand(batch_size, k, height, width)
            if self.if_offset:
                x = x + self._accumulate_from_center(x_offset) * self.extend_scope
            # [B,K,H,W] -> [B,H,W*K]
            y = y.permute(0, 2, 3, 1).reshape(batch_size, height, width * k)
            x = x.permute(0, 2, 3, 1).reshape(batch_size, height, width * k)

        grid_y = self._normalize_grid(y, height - 1)
        grid_x = self._normalize_grid(x, width - 1)
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, x):
        offset = torch.tanh(self.offset_conv(x))
        grid = self._coordinate_grid(offset, x.shape[-2], x.shape[-1])
        sampled = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return self.sample_conv(sampled)
