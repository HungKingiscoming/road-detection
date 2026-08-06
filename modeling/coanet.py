import math
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
except ImportError:
    SynchronizedBatchNorm2d = nn.BatchNorm2d

import timm
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


class ConvNeXtV2BackboneWrapper(nn.Module):
    """
    Wrapper tích hợp ConvNeXt-V2 thông qua thư viện timm,
    trả về đồng bộ 7 đầu ra tương thích hoàn toàn với kiến trúc CoANet.
    """
    def __init__(self, model_name='convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3) # Stage 1 (1/4), Stage 2 (1/8), Stage 3 (1/16), Stage 4 (1/32)
        )
        # Danh sách số channels của 4 stages (Ví dụ Tiny: [96, 192, 384, 768])
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x, return_aux: Optional[bool] = None):
        use_aux = self.training if return_aux is None else return_aux
        features = self.backbone(x)
        e1, e2, e3, e4 = features[0], features[1], features[2], features[3]

        x_spp = e4
        fused = e2
        c4_feat = e2.clone() if use_aux else None

        return e1, e2, e3, e4, x_spp, fused, c4_feat


class CoANet(nn.Module):
    def __init__(self, 
                 backbone='gcnet', 
                 output_stride=16, 
                 num_classes=1, 
                 num_neighbor=9,
                 base_channels=32,
                 convnext_model_name='convnextv2_tiny.fcmae_ft_in22k_in1k',
                 pretrained=True,
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

        # 1. Khởi tạo Backbone & xác định channels linh hoạt
        if backbone == 'gcnet':
            self.backbone = GCNetBackboneWrapper(
                in_channels=3,
                channels=base_channels,
                norm_cfg=dict(type='BN', requires_grad=True),
                deploy=deploy
            )
            in_channels_list = None
            aux_in_channels = base_channels * 2 # Stage 3 / c4 channels của GCNet
            aux_hidden_channels = base_channels
        elif backbone.startswith('convnext'):
            self.backbone = ConvNeXtV2BackboneWrapper(
                model_name=convnext_model_name,
                pretrained=pretrained
            )
            in_channels_list = self.backbone.out_channels
            aux_in_channels = in_channels_list[1] # Stage 2 channels (e2) của ConvNeXt
            aux_hidden_channels = 64
        else:
            raise NotImplementedError(
                f"Backbone '{backbone}' chưa được hỗ trợ. Hãy chọn 'gcnet' hoặc 'convnext...'"
            )

        # 2. Khởi tạo Decoder
        if backbone == 'gcnet':
            self.decoder = build_decoder(num_classes, backbone, BatchNorm, base_channels=base_channels)
        else:
            # Truyền in_channels_list cho Decoder để tự thích ứng số kênh
            self.decoder = build_decoder(num_classes, backbone, BatchNorm, in_channels_list=in_channels_list)

        # 3. Khởi tạo Connect Head
        self.connect = build_connect(num_classes, num_neighbor, BatchNorm)

        # 4. Khởi tạo Auxiliary Head
        seg_out_channels = 1 if num_classes <= 2 else num_classes
        self.aux_head = nn.Sequential(
            nn.Conv2d(aux_in_channels, aux_hidden_channels, kernel_size=3, padding=1, bias=False),
            BatchNorm(aux_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(aux_hidden_channels, seg_out_channels, kernel_size=1),
        )
        self.aux_loss_weight = 0.4
        self._init_weight()

    def forward(self, input, return_aux: bool = False):
        # Trích xuất 4 mức đặc trưng từ backbone được chọn
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
        """Chuyển đổi sang chế độ Inference tối ưu (chỉ áp dụng cho GCNet)."""
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
                    if isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d)):
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
                    if isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    input_tensor = torch.randn(2, 3, 512, 512)

    print("=== TEST BACKBONE: GCNET ===")
    model_gc = CoANet(backbone='gcnet', num_classes=1)
    seg, connect, connect_d1 = model_gc(input_tensor)
    print("Seg Output shape:", seg.shape)

    print("\n=== TEST BACKBONE: CONVNEXT-V2 ===")
    model_convnext = CoANet(backbone='convnextv2_tiny', num_classes=1)
    seg_c, connect_c, connect_d1_c, aux_c = model_convnext(input_tensor, return_aux=True)
    print("Seg Output shape:", seg_c.shape)
    print("Aux Output shape:", aux_c.shape)
