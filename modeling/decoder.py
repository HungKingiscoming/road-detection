import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, base_channels=32, in_channels_list=None):
        super(Decoder, self).__init__()

        # Xác định số channel đầu vào từ 4 stages của backbone
        if in_channels_list is not None:
            c1, c2, c3, c4 = in_channels_list
        else:
            c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # 1x1 Conv để nén channels của e1, e2, e3 về độ rộng cố định cho Decoder
        self.conv_e3 = nn.Sequential(nn.Conv2d(c3, 128, 1, bias=False), BatchNorm(128), nn.ReLU(inplace=True))
        self.conv_e2 = nn.Sequential(nn.Conv2d(c2, 64, 1, bias=False), BatchNorm(64), nn.ReLU(inplace=True))
        self.conv_e1 = nn.Sequential(nn.Conv2d(c1, 32, 1, bias=False), BatchNorm(32), nn.ReLU(inplace=True))

        # Các khối Decoder khôi phục độ phân giải
        self.decoder4 = DecoderBlock(c4, 256, BatchNorm)
        self.decoder3 = DecoderBlock(256 + 128, 128, BatchNorm)
        self.decoder2 = DecoderBlock(128 + 64, 64, BatchNorm)
        self.decoder1 = DecoderBlock(64 + 32, 64, BatchNorm)

        # Output projection về feature dimension cho Connect Head
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, e1, e2, e3, e4):
        # e4 (1/32) -> Upsample lên (1/16)
        d4 = self.decoder4(e4)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate với e3 (1/16) -> Upsample lên (1/8)
        proj_e3 = self.conv_e3(e3)
        d3 = self.decoder3(torch.cat([d4, proj_e3], dim=1))
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate với e2 (1/8) -> Upsample lên (1/4)
        proj_e2 = self.conv_e2(e2)
        d2 = self.decoder2(torch.cat([d3, proj_e2], dim=1))
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate với e1 (1/4) -> Upsample về độ phân giải gốc (1/1)
        proj_e1 = self.conv_e1(e1)
        d1 = self.decoder1(torch.cat([d2, proj_e1], dim=1))
        out = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=True)

        return self.final_conv(out)


def build_decoder(num_classes, backbone, BatchNorm, base_channels=32, in_channels_list=None):
    return Decoder(num_classes, backbone, BatchNorm, base_channels=base_channels, in_channels_list=in_channels_list)
