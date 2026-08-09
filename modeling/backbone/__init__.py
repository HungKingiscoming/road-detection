from modeling.backbone import resnet

def build_backbone(backbone, output_stride, BatchNorm, pretrained=True):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=pretrained)
    else:
        raise NotImplementedError
