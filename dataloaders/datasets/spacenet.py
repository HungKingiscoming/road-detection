import glob
import json
import os
import pickle
import random
from pathlib import Path
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
        split: str | None = None,
        base_size: int = 512,
        crop_size: int | None = 512,
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
        self.base_size = base_size
        self.crop_size = crop_size
        self.samples = []

        prepared_root = self._find_prepared_root(Path(data_dir))
        if split is not None and prepared_root is not None:
            self._discover_prepared_split(prepared_root, split)
        else:
            self._discover_flat_layout(Path(data_dir))

        if not self.samples:
            raise FileNotFoundError(
                'Không tìm thấy dữ liệu SpaceNet hợp lệ trong '
                f'{data_dir}. Hỗ trợ: (*__rgb.png, *__gt.png) hoặc '
                '(train|test)/(images|gt)/*.tif'
            )

        print(
            f"SpaceNet {split or 'flat'}: {len(self.samples)} cặp ảnh-mask "
            f"từ {prepared_root or data_dir}"
        )

    @staticmethod
    def _find_prepared_root(data_dir: Path):
        candidates = [data_dir]
        if data_dir.is_dir():
            candidates.extend(path for path in data_dir.iterdir() if path.is_dir())
        for candidate in candidates:
            if (candidate / 'train' / 'images').is_dir() and (candidate / 'train' / 'gt').is_dir():
                return candidate
        return None

    def _discover_prepared_split(self, root: Path, split: str):
        split_dir = root / split
        image_dir = split_dir / 'images'
        gt_dir = split_dir / 'gt'
        if not image_dir.is_dir() or not gt_dir.is_dir():
            return

        for image_path in sorted(image_dir.glob('*')):
            if not image_path.is_file():
                continue
            # RGB-PanSharpen_AOI_2_Vegas_img1.tif -> AOI_2_Vegas_img1.tif
            gt_name = image_path.name
            if gt_name.startswith('RGB-PanSharpen_'):
                gt_name = gt_name[len('RGB-PanSharpen_'):]
            gt_path = gt_dir / gt_name
            if gt_path.is_file():
                self.samples.append((image_path, gt_path, None, image_path.stem))

    def _discover_flat_layout(self, data_dir: Path):
        for image_path in sorted(data_dir.glob('*__rgb.png')):
            prefix = str(image_path)[:-len('__rgb.png')]
            gt_path = Path(f'{prefix}__gt.png')
            if gt_path.is_file():
                graph_path = Path(f'{prefix}__gt_graph_dense_spacenet.json')
                self.samples.append((
                    image_path, gt_path,
                    graph_path if graph_path.is_file() else None,
                    image_path.name[:-len('__rgb.png')],
                ))

    def __len__(self):
        return len(self.samples)

    def _resize_or_crop(self, image, mask):
        if self.crop_size is None:
            return image, mask

        crop_size = self.crop_size
        if not self.is_train:
            return (
                image.resize((crop_size, crop_size), Image.BILINEAR),
                mask.resize((crop_size, crop_size), Image.NEAREST),
            )

        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = image.size
        if height > width:
            out_width = short_size
            out_height = int(height * short_size / width)
        else:
            out_height = short_size
            out_width = int(width * short_size / height)
        image = image.resize((out_width, out_height), Image.BILINEAR)
        mask = mask.resize((out_width, out_height), Image.NEAREST)

        pad_width = max(crop_size - out_width, 0)
        pad_height = max(crop_size - out_height, 0)
        if pad_width or pad_height:
            image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=0)

        width, height = image.size
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        box = (left, top, left + crop_size, top + crop_size)
        image = image.crop(box)
        mask = mask.crop(box)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def __getitem__(self, idx):
        rgb_path, gt_path, graph_json_path, sample_id = self.samples[idx]

        # 2. Đọc ảnh RGB & GT dưới dạng PIL Image để tương thích hoàn toàn với custom transforms
        rgb_img = Image.open(rgb_path).convert('RGB')
        gt_mask_img = Image.open(gt_path).convert('L')
        rgb_img, gt_mask_img = self._resize_or_crop(rgb_img, gt_mask_img)

        # 3. Tính toán trước ma trận Connectivity (9 kênh) dạng numpy array từ mask gốc
        # CoANet reference masks are soft uint8 Gaussian masks. The upstream
        # normalization binarizes them at 0.5, i.e. at 128 in uint8 space.
        gt_mask_np = (np.array(gt_mask_img) >= 128).astype(np.float32)
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
            mask_np = (np.array(sample['label']) >= 128).astype(np.float32)
            
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
        sample['prefix'] = sample_id

        # 5. Đọc thông tin Graph cấu trúc cho TopoNet (nếu có)
        if self.load_graph_data and graph_json_path is not None and graph_json_path.exists():
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
    base_size: int = 512,
    crop_size: int = 512,
):
    """Hàm khởi tạo DataLoader cho Train và Validation hỗ trợ custom transform."""
    prepared_root = SpaceNetDataset._find_prepared_root(Path(data_dir))
    if prepared_root is not None:
        # The prepared reference dataset already contains the official split.
        val_split = 'val' if (prepared_root / 'val').is_dir() else 'test'
        train_subset = SpaceNetDataset(
            data_dir=data_dir, transform=transform_train, is_train=True,
            load_graph_data=False, split='train', base_size=base_size,
            crop_size=crop_size,
        )
        val_subset = SpaceNetDataset(
            data_dir=data_dir, transform=transform_val, is_train=False,
            load_graph_data=False, split=val_split, base_size=base_size,
            crop_size=crop_size,
        )
    else:
        train_dataset = SpaceNetDataset(
            data_dir=data_dir, transform=transform_train, is_train=True,
            load_graph_data=False, base_size=base_size, crop_size=crop_size,
        )
        val_dataset = SpaceNetDataset(
            data_dir=data_dir, transform=transform_val, is_train=False,
            load_graph_data=False, base_size=base_size, crop_size=crop_size,
        )
        total_size = len(train_dataset)
        train_size = int(train_ratio * total_size)
        indices = torch.randperm(
            total_size, generator=torch.Generator().manual_seed(42)
        ).tolist()
        train_subset = torch.utils.data.Subset(train_dataset, indices[:train_size])
        val_subset = torch.utils.data.Subset(val_dataset, indices[train_size:])

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
