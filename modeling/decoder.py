import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
except ImportError:
    SynchronizedBatchNorm2d = nn.BatchNorm2d


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        # 4 nhánh (in_channels // 8) gộp lại = in_channels // 2
        self.bn2 = BatchNorm(in_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels // 2, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), dim=1)

        if self.inp:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        return x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        return x[..., 0: shape[-2]]

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        return x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1).permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        return x[..., 0: shape[-2]].permute(0, 1, 3, 2)


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()

        if backbone == 'resnet':
            in_inplanes = 256
            # Cấu hình chuẩn inp=True cho decoder4, decoder3, decoder2
            self.decoder4 = DecoderBlock(in_inplanes, 256, BatchNorm, inp=True)
            self.decoder3 = DecoderBlock(256 + 256, 128, BatchNorm, inp=True)
            self.decoder2 = DecoderBlock(128 + 128, 64, BatchNorm, inp=True)
            self.decoder1 = DecoderBlock(64 + 64, 64, BatchNorm, inp=True)

            self.conv_e3 = nn.Sequential(
                nn.Conv2d(1024, 256, 1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True)
            )
            self.conv_e2 = nn.Sequential(
                nn.Conv2d(512, 128, 1, bias=False),
                BatchNorm(128),
                nn.ReLU(inplace=True)
            )
            self.conv_e1 = nn.Sequential(
                nn.Conv2d(256, 64, 1, bias=False),
                BatchNorm(64),
                nn.ReLU(inplace=True)
            )
        else:
            raise NotImplementedError(f"Backbone {backbone} chưa được hỗ trợ.")

        self._init_weight()

    def _match_size(self, source, target_tensor):
        """Đảm bảo hai tensor khớp HxW trước khi torch.cat."""
        if source.shape[-2:] != target_tensor.shape[-2:]:
            source = F.interpolate(source, size=target_tensor.shape[-2:], mode='bilinear', align_corners=True)
        return source

    def forward(self, e1, e2, e3, e4):
        out_d4 = self.decoder4(e4)
        feat_e3 = self._match_size(self.conv_e3(e3), out_d4)
        d4 = torch.cat((out_d4, feat_e3), dim=1)

        out_d3 = self.decoder3(d4)
        feat_e2 = self._match_size(self.conv_e2(e2), out_d3)
        d3 = torch.cat((out_d3, feat_e2), dim=1)

        out_d2 = self.decoder2(d3)
        feat_e1 = self._match_size(self.conv_e1(e1), out_d2)
        d2 = torch.cat((out_d2, feat_e1), dim=1)

        d1 = self.decoder1(d2)

        # Upsample về 100% kích thước ảnh đầu vào gốc
        return F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
