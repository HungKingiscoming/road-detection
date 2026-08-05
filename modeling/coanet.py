import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
except ImportError:
    SynchronizedBatchNorm2d = nn.BatchNorm2d

from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.connect import build_connect
from modeling.backbone import GCNet


class GCNetBackboneWrapper(GCNet):
    """
    Wrapper kế thừa từ GCNet để trích xuất đúng 4 tầng feature maps (e1, e2, e3, e4)
    đồng thời giữ nguyên bilateral fusion và DAPPM path của backbone gốc.
    """
    def forward(
        self,
        x,
        return_aux: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        use_aux = self.training if return_aux is None else return_aux
        out_size = (math.ceil(x.shape[-2] / 8), math.ceil(x.shape[-1] / 8))

        # ---- Stage 1–3 --------------------------------------------------- #
        x = self.stem_conv1(x)
        x = self.stem_conv2(x)
        e1 = self.stem_stage2(x)
        e2 = self.stem_stage3(e1)

        # ---- Stage 4 ---------------------------------------------------- #
        x_s = self.semantic_branch_layers[0](e2)
        x_d = self.detail_branch_layers[0](e2)

        comp_c = self.compression_1(self.relu(x_s))
        x_s = x_s + self.down_1(self.relu(x_d))
        x_d = x_d + self._resize(comp_c, out_size)
        c4_feat = x_d.clone() if use_aux else None

        # ---- Stage 5 ---------------------------------------------------- #
        e3 = x_s
        x_s = self.semantic_branch_layers[1](self.relu(x_s))
        x_d = self.detail_branch_layers[1](self.relu(x_d))

        comp_c = self.compression_2(self.relu(x_s))
        x_s = x_s + self.down_2(self.relu(x_d))
        x_d = x_d + self._resize(comp_c, out_size)
        e4 = x_s

        # ---- Stage 6 ---------------------------------------------------- #
        x_d = self.detail_branch_layers[2](self.relu(x_d))
        x_s = self.semantic_branch_layers[2](self.relu(x_s))
        x_spp = self.spp(x_s)
        x_spp = self._resize(x_spp, out_size)
        fused = x_d + x_spp

        return e1, e2, e3, e4, x_spp, fused, c4_feat


    def _resize(self, x, size):
        return F.interpolate(
            x,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners,
        )


class CoANet(nn.Module):
    def __init__(self, 
                 backbone='gcnet', 
                 output_stride=16, 
                 num_classes=1, 
                 num_neighbor=9,
                 base_channels=32,
                 sync_bn=False, 
                 freeze_bn=False,
                 deploy=False):
        super(CoANet, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone_type = backbone
        self.freeze_bn_flag = freeze_bn
        self.deploy = deploy

        # 1. Khởi tạo Backbone
        if backbone == 'gcnet':
            self.backbone = GCNetBackboneWrapper(
                in_channels=3,
                channels=base_channels,
                norm_cfg=dict(type='BN', requires_grad=True),
                deploy=deploy
            )
        else:
            raise NotImplementedError(
                f"Backbone '{backbone}' chưa được hỗ trợ trong build model hiện tại."
            )



        # 3. Khởi tạo Decoder
        if backbone == 'gcnet':
            self.decoder = build_decoder(num_classes, backbone, BatchNorm, base_channels=base_channels)
        else:
            self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        # 4. Khởi tạo Connect Head
        # Với road/background, head segmentation nên là binary 1-channel.
        self.connect = build_connect(num_classes, num_neighbor, BatchNorm)

        # 5. Khởi tạo Auxiliary Head cho Stage 4 của backbone
        aux_in_channels = base_channels * 2
        seg_out_channels = 1 if num_classes <= 2 else num_classes
        self.aux_head = nn.Sequential(
            nn.Conv2d(aux_in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(base_channels, seg_out_channels, kernel_size=1),
        )
        self.aux_loss_weight = 0.4
        self._init_weight()

    def forward(self, input, return_aux: bool = False):
        # Trích xuất 4 mức đặc trưng + x_spp, fused từ backbone trong 1 lần forward duy nhất
        e1, e2, e3, e4, x_spp, fused, c4_feat = self.backbone(
            input,
            return_aux=(return_aux or self.training),
        )

        # Đưa qua Decoder
        x = self.decoder(e1, e2, e3, e4)

        # Kết quả dự đoán segmentation và connectivity maps
        seg, connect, connect_d1 = self.connect(x)

        aux_seg = None
        if (return_aux or self.training) and c4_feat is not None:
            aux_seg = self.aux_head(c4_feat)
            aux_seg = F.interpolate(
                aux_seg,
                size=input.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

        if return_aux:
            return seg, connect, connect_d1, aux_seg
        return seg, connect, connect_d1

    def _init_weight(self):
        for m in self.aux_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def switch_to_deploy(self):
        """Chuyển đổi toàn bộ GCBlock trong GCNet backbone sang chế độ Inference tối ưu."""
        if hasattr(self.backbone, 'switch_to_deploy'):
            self.backbone.switch_to_deploy()
        self.deploy = True

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.eval()

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
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_2x_lr_params(self):
        modules = [self.decoder, self.connect]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn_flag:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    # Test thử nghiệm khởi tạo và chạy thử dữ liệu
    model = CoANet(backbone='gcnet', output_stride=16, num_classes=19)
    model.eval()

    input_tensor = torch.randn(1, 3, 512, 512)
    seg, connect, connect_d1 = model(input_tensor)

    print("--- Kết quả Test Shape ---")
    print("Segmentation Output:", seg.shape)        # [1, 19, 512, 512]
    print("Connect Map 0 Shape:", connect.shape)     # [1, 9, 512, 512]
    print("Connect Map 1 Shape:", connect_d1.shape)  # [1, 9, 512, 512]

    model.train()
    seg, connect, connect_d1, aux_seg = model(input_tensor, return_aux=True)
    print("Auxiliary Output:", aux_seg.shape)        # [1, 19, 512, 512]

    # Test switch sang Deploy Mode
    model.switch_to_deploy()
    print("Chuyển đổi switch_to_deploy() thành công!")