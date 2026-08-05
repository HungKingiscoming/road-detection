import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        img = np.array(sample['image'], dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std

        gt_mask = np.array(sample['gt_mask'], dtype=np.float32)
        gt_mask = (gt_mask >= 0.5).astype(np.float32)

        gt_connect_d1 = np.array(sample['gt_connect_d1'], dtype=np.float32) / 255.0
        gt_connect_d3 = np.array(sample['gt_connect_d3'], dtype=np.float32) / 255.0

        sample['image'] = img
        sample['gt_mask'] = gt_mask
        sample['gt_connect_d1'] = gt_connect_d1
        sample['gt_connect_d3'] = gt_connect_d3
        return sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['image']
        gt_mask = sample['gt_mask']
        gt_connect_d1 = sample['gt_connect_d1']
        gt_connect_d3 = sample['gt_connect_d3']

        # Chuyển HWC -> CHW nếu là ảnh NumPy 3 chiều
        if isinstance(img, np.ndarray) and img.ndim == 3:
            img = img.transpose((2, 0, 1))
        if isinstance(gt_connect_d1, np.ndarray) and gt_connect_d1.ndim == 3:
            gt_connect_d1 = gt_connect_d1.transpose((2, 0, 1))
        if isinstance(gt_connect_d3, np.ndarray) and gt_connect_d3.ndim == 3:
            gt_connect_d3 = gt_connect_d3.transpose((2, 0, 1))

        sample['image'] = torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img
        sample['gt_mask'] = torch.from_numpy(gt_mask).float().unsqueeze(0) if isinstance(gt_mask, np.ndarray) and gt_mask.ndim == 2 else torch.from_numpy(gt_mask).float()
        sample['gt_connect_d1'] = torch.from_numpy(gt_connect_d1).float() if isinstance(gt_connect_d1, np.ndarray) else gt_connect_d1
        sample['gt_connect_d3'] = torch.from_numpy(gt_connect_d3).float() if isinstance(gt_connect_d3, np.ndarray) else gt_connect_d3

        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            for key in ['image', 'gt_mask', 'gt_connect_d1', 'gt_connect_d3']:
                if key in sample and isinstance(sample[key], Image.Image):
                    sample[key] = sample[key].transpose(Image.FLIP_LEFT_RIGHT)
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        for key in ['image', 'gt_mask', 'gt_connect_d1', 'gt_connect_d3']:
            if key in sample and isinstance(sample[key], Image.Image):
                resample = Image.BILINEAR if key == 'image' else Image.NEAREST
                sample[key] = sample[key].rotate(rotate_degree, resample)
        return sample