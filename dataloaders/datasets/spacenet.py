import glob
import json
import os
import pickle
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


def generate_connectivity_gt(
    gt_mask_np: np.ndarray, dilation: int = 1
) -> np.ndarray:
    """Tạo GT Connectivity 9 kênh [9, H, W] chuẩn CoANet gốc.

    Thứ tự 9 kênh tương ứng với 3 file _0.png, _1.png, _2.png của CoANet:
    - _0.png (Kênh 0, 1, 2): Hàng trên    [(-d, -d), (-d, 0), (-d, +d)]
    - _1.png (Kênh 3, 4, 5): Hàng giữa   [( 0, -d), ( 0, 0), ( 0, +d)] (Kênh 4 là Center)
    - _2.png (Kênh 6, 7, 8): Hàng dưới   [(+d, -d), (+d, 0), (+d, +d)]
    """
    h, w = gt_mask_np.shape
    pad_size = 2 * dilation
    step = 2 * dilation

    # Nhị phân hóa mask về 0.0 hoặc 1.0
    gt_bin = (gt_mask_np > 0).astype(np.float32)

    # Pad viền 0 xung quanh mask
    mask_pad = np.pad(
        gt_bin, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant'
    )

    # Trích xuất 3 hàng theo đúng cách tính slicing của CoANet gốc
    row0 = [
        mask_pad[0:h, 0:w],
        mask_pad[0:h, step : w + step],
        mask_pad[0:h, 2 * step : w + 2 * step],
    ]

    row1 = [
        mask_pad[step : h + step, 0:w],
        mask_pad[step : h + step, step : w + step],  # Kênh 4 (Center)
        mask_pad[step : h + step, 2 * step : w + 2 * step],
    ]

    row2 = [
        mask_pad[2 * step : h + 2 * step, 0:w],
        mask_pad[2 * step : h + 2 * step, step : w + step],
        mask_pad[2 * step : h + 2 * step, 2 * step : w + 2 * step],
    ]

    # Ghép 9 kênh lại thành tensor [9, H, W]
    conn_channels = row0 + row1 + row2
    conn_map = np.stack(conn_channels, axis=0).astype(np.float32)

    # Đảm bảo chỉ giữ lại kết nối tại những vị trí pixel tâm thuộc đường (Ground Truth > 0)
    center_mask = gt_bin[None, ...]  # [1, H, W]
    conn_map = conn_map * center_mask

    return conn_map  # [9, H, W]


class SpaceNetDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        transform=None,
        is_train: bool = True,
        load_graph_data: bool = True,
    ):
        """Args:
        data_dir: Thư mục chứa các file mẫu (RGB, GT, json)
        transform: Các phép biến đổi Augmentation tùy chỉnh
        is_train: Chế độ Train hay Val
        load_graph_data: Có đọc kèm file JSON Graph hay không
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.load_graph_data = load_graph_data

        # Lấy danh sách các prefix file hợp lệ
        rgb_files = sorted(glob.glob(os.path.join(data_dir, '*__rgb.png')))
        self.prefixes = []

        for f in rgb_files:
            prefix = f.replace('__rgb.png', '')
            gt_path = f'{prefix}__gt.png'
            if os.path.exists(gt_path):
                self.prefixes.append(prefix)

        if len(self.prefixes) == 0:
            raise FileNotFoundError(
                f'Không tìm thấy các cặp file (*__rgb.png, *__gt.png) hợp lệ nào trong {data_dir}'
            )

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]

        # 1. Đường dẫn các file tương ứng
        rgb_path = f'{prefix}__rgb.png'
        gt_path = f'{prefix}__gt.png'
        graph_json_path = f'{prefix}__gt_graph_dense_spacenet.json'

        # 2. Đọc ảnh RGB & GT dưới dạng PIL Image để tương thích hoàn toàn với custom transforms
        rgb_img = Image.open(rgb_path).convert('RGB')
        gt_mask_img = Image.open(gt_path).convert('L')

        # 3. Tính toán trước ma trận Connectivity (9 kênh) dạng numpy array từ mask gốc
        gt_mask_np = (np.array(gt_mask_img) > 0).astype(np.float32)
        gt_connect_d1 = generate_connectivity_gt(gt_mask_np, dilation=1)  # [9, H, W]
        gt_connect_d3 = generate_connectivity_gt(gt_mask_np, dilation=2)  # [9, H, W]

        # Chuyển đổi 6 phân đoạn connectivity thành các PIL Image (mỗi phần 3 kênh HxWx3) để apply đồng bộ transform
        sample = {
            'image': rgb_img,
            'label': gt_mask_img,
            'connect0': Image.fromarray((gt_connect_d3[0:3].transpose(1, 2, 0) * 255).astype(np.uint8)),
            'connect1': Image.fromarray((gt_connect_d3[3:6].transpose(1, 2, 0) * 255).astype(np.uint8)),
            'connect2': Image.fromarray((gt_connect_d3[6:9].transpose(1, 2, 0) * 255).astype(np.uint8)),
            'connect_d1_0': Image.fromarray((gt_connect_d1[0:3].transpose(1, 2, 0) * 255).astype(np.uint8)),
            'connect_d1_1': Image.fromarray((gt_connect_d1[3:6].transpose(1, 2, 0) * 255).astype(np.uint8)),
            'connect_d1_2': Image.fromarray((gt_connect_d1[6:9].transpose(1, 2, 0) * 255).astype(np.uint8)),
        }

        # 4. Thực thi chuỗi Transform (Augmentation + Normalize + ToTensor) nếu có truyền vào
        if self.transform is not None:
            sample = self.transform(sample)
        else:
            # Fallback mặc định nếu không truyền transform ngoài
            img_np = np.array(sample['image']).astype(np.float32) / 255.0
            img_tensor = TF.normalize(
                torch.from_numpy(img_np).permute(2, 0, 1).float(),
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            mask_np = (np.array(sample['label']) > 0).astype(np.float32)
            
            sample = {
                'image': img_tensor,
                'label': torch.from_numpy(mask_np).float(),
                'connect0': torch.from_numpy(gt_connect_d3[0:3]).float(),
                'connect1': torch.from_numpy(gt_connect_d3[3:6]).float(),
                'connect2': torch.from_numpy(gt_connect_d3[6:9]).float(),
                'connect_d1_0': torch.from_numpy(gt_connect_d1[0:3]).float(),
                'connect_d1_1': torch.from_numpy(gt_connect_d1[3:6]).float(),
                'connect_d1_2': torch.from_numpy(gt_connect_d1[6:9]).float(),
            }

        # Gắn thêm tên định danh file mẫu vào dictionary trả về
        sample['prefix'] = os.path.basename(prefix)

        # 5. Đọc thông tin Graph cấu trúc cho TopoNet (nếu có)
        if self.load_graph_data and os.path.exists(graph_json_path):
            try:
                with open(graph_json_path, 'r') as f:
                    graph_json = json.load(f)
                sample['graph_json'] = graph_json
            except Exception:
                pass

        return sample


def build_spacenet_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    transform_train=None,
    transform_val=None,
):
    """Hàm khởi tạo DataLoader cho Train và Validation hỗ trợ custom transform."""
    train_dataset = SpaceNetDataset(
        data_dir=data_dir, transform=transform_train, is_train=True, load_graph_data=False
    )
    val_dataset = SpaceNetDataset(
        data_dir=data_dir, transform=transform_val, is_train=False, load_graph_data=False
    )

    # Chia Dataset theo tỷ lệ train/val cố định seed
    total_size = len(train_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
