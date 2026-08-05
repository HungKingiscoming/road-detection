import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='spacenet'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset='spacenet', plot=False):
    if torch.is_tensor(label_mask):
        label_mask = label_mask.detach().cpu().numpy()
        
    if label_mask.ndim == 3:
        label_mask = label_mask.squeeze(0)

    if dataset in ['spacenet', 'DeepGlobe']:
        n_classes = 2
        label_colours = get_spacenet_labels()
    else:
        raise NotImplementedError(f"Dataset {dataset} chưa được hỗ trợ.")

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

def get_spacenet_labels():
    return np.asarray([[0, 0, 0], [255, 255, 255]])

def get_deepglobe_labels():
    return np.asarray([[0, 0, 0], [255, 255, 255]])