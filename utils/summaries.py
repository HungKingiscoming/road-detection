import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def get_spacenet_labels():
    """Tạo bảng màu chuẩn cho SpaceNet (0: Background - Đen, 1: Road - Trắng)."""
    return np.asarray([[0, 0, 0], [255, 255, 255]])


def decode_segmap(label_mask, dataset='spacenet', plot=False):
    """
    Chuyển đổi Mask nhị phân/Index sang ảnh RGB.
    """
    if torch.is_tensor(label_mask):
        label_mask = label_mask.detach().cpu().numpy()

    if label_mask.ndim == 3:
        label_mask = label_mask.squeeze(0)

    n_classes = 2
    label_colours = get_spacenet_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def decode_seg_map_sequence(label_masks, dataset='spacenet'):
    """Chuyển đổi một batch mask sang Tensor dạng (B, C, H, W) phục vụ TensorBoard."""
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


class TensorboardSummary(object):
    """Class quản lý SummaryWriter và trực quan hóa hình ảnh trên TensorBoard."""
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=self.directory)
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step, split='Val'):
        """
        Trực quan hóa Ảnh gốc (Input Image), Ground Truth Mask và Predicted Mask lên TensorBoard.
        """
        # 1. Lấy mẫu đầu tiên trong batch
        grid_image = image[0].detach().cpu().numpy()
        grid_target = target[0].detach().cpu().numpy()
        
        # Nếu output là logits từ fused_mask, đưa qua Sigmoid và Threshold 0.5
        if output.shape[1] == 1:
            grid_output = (torch.sigmoid(output[0]) > 0.5).detach().cpu().numpy().astype(np.uint8)
        else:
            grid_output = torch.argmax(output[0], dim=0).detach().cpu().numpy().astype(np.uint8)

        # 2. Un-normalize ảnh gốc từ ImageNet mean/std về [0, 1]
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        grid_image = grid_image * std + mean
        grid_image = np.clip(grid_image, 0, 1)

        # 3. Decode mask sang RGB
        grid_target_rgb = decode_segmap(grid_target, dataset).transpose([2, 0, 1])
        grid_output_rgb = decode_segmap(grid_output, dataset).transpose([2, 0, 1])

        # 4. Ghi lên TensorBoard
        writer.add_image(f'{split}/1_Image', grid_image, global_step)
        writer.add_image(f'{split}/2_GroundTruth', grid_target_rgb, global_step)
        writer.add_image(f'{split}/3_Prediction', grid_output_rgb, global_step)
