import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
except ImportError:
    SynchronizedBatchNorm2d = nn.BatchNorm2d


class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Connect(nn.Module):
    def __init__(self, num_classes, num_neighbor, BatchNorm, in_channels=64, reduction=3):
        super(Connect, self).__init__()
        
        seg_out_channels = 1 if num_classes <= 2 else num_classes
        self.seg_out_channels = seg_out_channels

        # Nhánh Semantic Segmentation
        self.seg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, seg_out_channels, kernel_size=1, stride=1)
        )

        # Nhánh Local Connectivity (dilated = 1)
        self.connect_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_neighbor, 3, padding=1, dilation=1)
        )
        self.se = SELayer(num_neighbor, reduction)

        # Nhánh Non-local / Context Connectivity (dilated = 3)
        self.connect_branch_d1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_neighbor, 3, padding=3, dilation=3)
        )
        self.se_d1 = SELayer(num_neighbor, reduction)

        self._init_weight()

    def forward(self, input):
        seg = self.seg_branch(input)

        con = self.connect_branch(input)
        con0 = self.se(con)

        con_d1 = self.connect_branch_d1(input)
        con1 = self.se_d1(con_d1)

        return seg, con0, con1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()


def build_connect(num_classes, num_neighbor, BatchNorm, in_channels=64):
    return Connect(num_classes, num_neighbor, BatchNorm, in_channels=in_channels)