import glob
import json
import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


def generate_connectivity_gt(
    gt_mask_np: np.ndarray, dilation: int = 1
) -> np.ndarray:
  """Tạo GT Connectivity 9 kênh [9, H, W] chuẩn CoANet gốc.

  Thứ tự 9 kênh tương ứng với 3 file _0.png, _1.png, _2.png của CoANet:
  - _0.png (Kênh 0, 1, 2): Hàng trên    [(-d, -d), (-d, 0), (-d, +d)]
  - _1.png (Kênh 3, 4, 5): Hàng giữa   [( 0, -d), ( 0, 0), ( 0, +d)] (Kênh 4 là
  Center)
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
  # Hàng 0 (Top)
  row0 = [
      mask_pad[0:h, 0:w],
      mask_pad[0:h, step : w + step],
      mask_pad[0:h, 2 * step : w + 2 * step],
  ]

  # Hàng 1 (Center)
  row1 = [
      mask_pad[step : h + step, 0:w],
      mask_pad[step : h + step, step : w + step],  # Kênh 4 (Center)
      mask_pad[step : h + step, 2 * step : w + 2 * step],
  ]

  # Hàng 2 (Bottom)
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

    data_dir: Thư mục chứa các file mẫu (ví dụ: RGB, GT, pickle, json) transform:
    Các phép biến đổi Augmentation (Albumentations) is_train: Chế độ Train hay
    Val load_graph_data: Có đọc kèm file Pickle/JSON Graph hay không
    """
    self.data_dir = data_dir
    self.transform = transform
    self.is_train = is_train
    self.load_graph_data = load_graph_data

    # Lấy danh sách các prefix file (chỉ giữ lại những file có đủ cặp __rgb.png và __gt.png)
    rgb_files = sorted(glob.glob(os.path.join(data_dir, '*__rgb.png')))
    self.prefixes = []

    for f in rgb_files:
      prefix = f.replace('__rgb.png', '')
      gt_path = f'{prefix}__gt.png'
      if os.path.exists(gt_path):
        self.prefixes.append(prefix)

    if len(self.prefixes) == 0:
      raise FileNotFoundError(
          'Không tìm thấy các cặp file (*__rgb.png, *__gt.png) hợp lệ nào trong'
          f' {data_dir}'
      )

  def __len__(self):
    return len(self.prefixes)

  def __getitem__(self, idx):
        prefix = self.prefixes[idx]

        # 1. Đường dẫn các file tương ứng
        rgb_path = f'{prefix}__rgb.png'
        gt_path = f'{prefix}__gt.png'
        graph_json_path = f'{prefix}__gt_graph_dense_spacenet.json'

        # 2. Đọc ảnh RGB & GT Mask dưới dạng NumPy Array
        rgb_img = np.array(Image.open(rgb_path).convert('RGB'))

        gt_mask_img = Image.open(gt_path).convert('L')
        gt_mask = (np.array(gt_mask_img) > 0).astype(np.float32)

        # 3. Áp dụng Data Augmentation (Albumentations)
        if self.transform is not None:
            augmented = self.transform(image=rgb_img, mask=gt_mask)
            rgb_img = augmented['image']
            gt_mask = augmented['mask']

            # Chuyển đổi về numpy nếu transform trả về Tensor trước khi tính connectivity
            if isinstance(rgb_img, torch.Tensor):
                rgb_img = rgb_img.numpy()
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.numpy()

        # 4. Tính toán Connectivity Maps (d1 & d3) chuẩn thuật toán CoANet gốc
        gt_connect_d1 = generate_connectivity_gt(gt_mask, dilation=1)  # [9, H, W]
        gt_connect_d3 = generate_connectivity_gt(gt_mask, dilation=2)  # [9, H, W]

        # 5. Chuyển đổi sang PyTorch Tensor & Normalization
        rgb_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        rgb_tensor = TF.normalize(
            rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # LƯU Ý: Bỏ unsqueeze(0) để mask giữ nguyên shape [H, W] thay vì [1, H, W].
        # Vì trong train.py tác giả đã dùng target = torch.unsqueeze(target, 1) rồi.
        gt_mask_tensor = torch.from_numpy(gt_mask).float() 
        
        gt_conn_d1_tensor = torch.from_numpy(gt_connect_d1).float()  # [9, H, W]
        gt_conn_d3_tensor = torch.from_numpy(gt_connect_d3).float()  # [9, H, W]

        # 🟢 ĐỔI TÊN KEYS VÀ CHIA TENSOR ĐỂ KHỚP 100% VỚI TRAIN.PY GỐC
        sample = {
            'image': rgb_tensor,
            'label': gt_mask_tensor,
            
            # Chia tensor [9, H, W] thành 3 phần [3, H, W] cho connect_d3 (tương ứng connect0, 1, 2)
            'connect0': gt_conn_d3_tensor[0:3, :, :],
            'connect1': gt_conn_d3_tensor[3:6, :, :],
            'connect2': gt_conn_d3_tensor[6:9, :, :],
            
            # Chia tensor [9, H, W] thành 3 phần [3, H, W] cho connect_d1
            'connect_d1_0': gt_conn_d1_tensor[0:3, :, :],
            'connect_d1_1': gt_conn_d1_tensor[3:6, :, :],
            'connect_d1_2': gt_conn_d1_tensor[6:9, :, :],
            
            'prefix': os.path.basename(prefix),
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
    train_ratio: float = 0.8,
):
  """Hàm khởi tạo DataLoader cho Train và Validation."""
  dataset = SpaceNetDataset(data_dir=data_dir, load_graph_data=False)

  # Chia Dataset thành Train / Val
  train_size = int(train_ratio * len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = torch.utils.data.random_split(
      dataset,
      [train_size, val_size],
      generator=torch.Generator().manual_seed(42),
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
      drop_last=True,
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
      drop_last=False,
  )

  return train_loader, val_loader
