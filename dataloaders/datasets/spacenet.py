import os
import glob
import pickle
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def generate_connectivity_gt(gt_mask_np: np.ndarray, dilation: int = 1) -> np.ndarray:
    """
    Tạo GT Connectivity 9 kênh từ Binary Mask (Numpy) [9, H, W].
    dilation = 1 -> connect_d1
    dilation = 3 -> connect_d3
    """
    h, w = gt_mask_np.shape
    pad = 2 * dilation
    stride = 2 * dilation

    mask_pad = np.pad(gt_mask_np, pad_width=pad, mode='constant', constant_values=0)

    conn_channels = []
    for i in range(3):
        for j in range(3):
            r_start = i * stride
            c_start = j * stride
            patch = mask_pad[r_start : r_start + h, c_start : c_start + w]
            conn_channels.append(patch)

    conn_map = np.stack(conn_channels, axis=0).astype(np.float32)
    # Chỉ giữ giá trị kết nối tại các vị trí đường giao thông (mask > 0)
    conn_map *= (gt_mask_np > 0).astype(np.float32)[None, ...]
    return conn_map


class SpaceNetDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        transform=None, 
        is_train: bool = True,
        load_graph_data: bool = True
    ):
        """
        Args:
            data_dir: Thư mục chứa các file mẫu (ví dụ: RGB, GT, pickle, json)
            transform: Các phép biến đổi Augmentation (Albumentations)
            is_train: Chế độ Train hay Val
            load_graph_data: Có đọc kèm file Pickle/JSON Graph hay không
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.load_graph_data = load_graph_data

        # Lấy danh sách các prefix file (chỉ giữ lại những file có đủ cặp __rgb.png và __gt.png)
        rgb_files = sorted(glob.glob(os.path.join(data_dir, "*__rgb.png")))
        self.prefixes = []
        
        for f in rgb_files:
            prefix = f.replace("__rgb.png", "")
            gt_path = f"{prefix}__gt.png"
            if os.path.exists(gt_path):
                self.prefixes.append(prefix)

        if len(self.prefixes) == 0:
            raise FileNotFoundError(f"Không tìm thấy các cặp file (*__rgb.png, *__gt.png) hợp lệ nào trong {data_dir}")

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]

        # 1. Đường dẫn các file tương ứng
        rgb_path = f"{prefix}__rgb.png"
        gt_path = f"{prefix}__gt.png"
        graph_json_path = f"{prefix}__gt_graph_dense_spacenet.json"

        # 2. Đọc ảnh RGB & GT Mask dưới dạng NumPy Array
        rgb_img = np.array(Image.open(rgb_path).convert('RGB'))

        gt_mask_img = Image.open(gt_path).convert('L')
        gt_mask = (np.array(gt_mask_img) > 0).astype(np.float32)

        # 3. Áp dụng Data Augmentation (Albumentations)
        if self.transform is not None:
            augmented = self.transform(
                image=rgb_img, 
                mask=gt_mask
            )
            rgb_img = augmented['image']
            gt_mask = augmented['mask']
            
            # Chuyển đổi về numpy nếu transform trả về Tensor trước khi tính connectivity
            if isinstance(rgb_img, torch.Tensor):
                rgb_img = rgb_img.numpy()
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.numpy()

        # 4. Tính toán Connectivity Maps (d1 & d3) trên NumPy Array đã Augment
        gt_connect_d1 = generate_connectivity_gt(gt_mask, dilation=1)  # [9, H, W]
        gt_connect_d3 = generate_connectivity_gt(gt_mask, dilation=3)  # [9, H, W]

        # 5. Chuyển đổi sang PyTorch Tensor & Normalization
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        rgb_tensor = TF.normalize(
            rgb_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

        gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).float()  # [1, H, W]
        gt_conn_d1_tensor = torch.from_numpy(gt_connect_d1).float()      # [9, H, W]
        gt_conn_d3_tensor = torch.from_numpy(gt_connect_d3).float()      # [9, H, W]

        sample = {
            'image': rgb_tensor,
            'gt_mask': gt_mask_tensor,
            'gt_connect_d1': gt_conn_d1_tensor,
            'gt_connect_d3': gt_conn_d3_tensor,
            'prefix': os.path.basename(prefix)
        }

        # 6. Đọc thông tin Graph cấu trúc cho TopoNet (nếu có)
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
    train_ratio: float = 0.8
):
    """
    Hàm khởi tạo DataLoader cho Train và Validation.
    """
    dataset = SpaceNetDataset(data_dir=data_dir, load_graph_data=False)
    
    # Chia Dataset thành Train / Val
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
