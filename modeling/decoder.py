import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.dsconv import DSConv2d

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False,
                 scm_type='strip', dsconv_kernel_size=9,
                 dsconv_extend_scope=1.0):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.scm_type = scm_type
        branch_in = in_channels // 4
        branch_out = in_channels // 8
        if scm_type == 'strip':
            self.deconv1 = nn.Conv2d(branch_in, branch_out, (1, 9), padding=(0, 4))
            self.deconv2 = nn.Conv2d(branch_in, branch_out, (9, 1), padding=(4, 0))
            self.deconv3 = nn.Conv2d(branch_in, branch_out, (9, 1), padding=(4, 0))
            self.deconv4 = nn.Conv2d(branch_in, branch_out, (1, 9), padding=(0, 4))
        elif scm_type == 'dsconv':
            dsconv_args = dict(
                in_channels=branch_in,
                out_channels=branch_out,
                kernel_size=dsconv_kernel_size,
                extend_scope=dsconv_extend_scope,
                if_offset=True,
            )
            self.dsconv1 = DSConv2d(morph=0, **dsconv_args)
            self.dsconv2 = DSConv2d(morph=1, **dsconv_args)
            self.dsconv3 = DSConv2d(morph=1, **dsconv_args)
            self.dsconv4 = DSConv2d(morph=0, **dsconv_args)
        else:
            raise ValueError("scm_type must be 'strip' or 'dsconv'")

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()
        if self.scm_type == 'dsconv':
            for module in self.modules():
                if isinstance(module, DSConv2d):
                    module.reset_offset_parameters()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        if self.scm_type == 'strip':
            x1 = self.deconv1(x)
            x2 = self.deconv2(x)
            x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
            x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        else:
            x1 = self.dsconv1(x)
            x2 = self.dsconv2(x)
            x3 = self.inv_h_transform(self.dsconv3(self.h_transform(x)))
            x4 = self.inv_v_transform(self.dsconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, scm_type='strip',
                 dsconv_kernel_size=9, dsconv_extend_scope=1.0):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            in_inplanes = 256
        else:
            raise NotImplementedError

        block_args = dict(
            BatchNorm=BatchNorm,
            scm_type=scm_type,
            dsconv_kernel_size=dsconv_kernel_size,
            dsconv_extend_scope=dsconv_extend_scope,
        )
        self.decoder4 = DecoderBlock(in_inplanes, 256, **block_args)
        self.decoder3 = DecoderBlock(512, 128, **block_args)
        self.decoder2 = DecoderBlock(256, 64, inp=True, **block_args)
        self.decoder1 = DecoderBlock(128, 64, inp=True, **block_args)

        self.conv_e3 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False),
                                     BatchNorm(128),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                     BatchNorm(64),
                                     nn.ReLU())

        self._init_weight()
        # Decoder._init_weight recursively visits child convolutions, so reset
        # all newly introduced offsets once more after the complete init pass.
        for module in self.modules():
            if isinstance(module, DSConv2d):
                module.reset_offset_parameters()


    def forward(self, e1, e2, e3, e4):
        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.conv_e2(e2)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1)
        d1 = self.decoder1(d2)
        x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, scm_type='strip',
                  dsconv_kernel_size=9, dsconv_extend_scope=1.0):
    return Decoder(num_classes, backbone, BatchNorm, scm_type,
                   dsconv_kernel_size, dsconv_extend_scope)
