import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.connect import build_connect

class CoANet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, num_neighbor=9,
                 sync_bn=True, freeze_bn=False, scm_type='strip',
                 dsconv_kernel_size=9, dsconv_extend_scope=1.0,
                 backbone_pretrained=True):
        super(CoANet, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(
            backbone, output_stride, BatchNorm, pretrained=backbone_pretrained
        )
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(
            num_classes, backbone, BatchNorm,
            scm_type=scm_type,
            dsconv_kernel_size=dsconv_kernel_size,
            dsconv_extend_scope=dsconv_extend_scope,
        )

        self.connect = build_connect(num_classes, num_neighbor, BatchNorm)

        self.freeze_bn_flag = freeze_bn
        self.scm_type = scm_type
        self.transfer_stage = 4

    def forward(self, input):
        e1, e2, e3, e4 = self.backbone(input)
        e4 = self.aspp(e4)
        x = self.decoder(e1, e2, e3, e4)
        seg, connect, connect_d1 = self.connect(x)

        return seg, connect, connect_d1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.eval()

    def set_transfer_stage(self, stage):
        """Progressively unfreeze pretrained CoANet around the new offsets.

        0: offset predictors only
        1: decoder + connectivity heads
        2: stage 1 + ASPP + ResNet layer4
        3: stage 2 + ResNet layer3
        4: full network
        """
        if stage not in range(5):
            raise ValueError("transfer stage must be an integer from 0 to 4")

        for parameter in self.parameters():
            parameter.requires_grad = False

        if stage == 0:
            for name, parameter in self.decoder.named_parameters():
                if '.offset_conv.' in name:
                    parameter.requires_grad = True
        else:
            for module in (self.decoder, self.connect):
                for parameter in module.parameters():
                    parameter.requires_grad = True

        if stage >= 2:
            for module in (self.aspp, self.backbone.layer4):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        if stage >= 3:
            for parameter in self.backbone.layer3.parameters():
                parameter.requires_grad = True
        if stage >= 4:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True

        self.transfer_stage = stage
        self.enforce_frozen_batchnorm()

    def enforce_frozen_batchnorm(self):
        """Keep statistics fixed in frozen parts after every ``model.train()``."""
        for module in self.modules():
            if isinstance(module, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                parameters = list(module.parameters())
                if self.freeze_bn_flag or not any(p.requires_grad for p in parameters):
                    module.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn_flag:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_2x_lr_params(self):
        modules = [self.aspp, self.decoder, self.connect]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn_flag:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = CoANet(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
