# CoANet with Dynamic Snake Convolution

This repository adapts CoANet for road extraction by replacing the four fixed
strip-convolution branches in each SCM decoder block with Dynamic Snake
Convolution. The model preserves CoANet's segmentation and connectivity heads
and supports progressive transfer learning from a pretrained CoANet checkpoint.

## Highlights

- Four-direction DSConv SCM, including transformed diagonal branches.
- Automatic migration of all pretrained SCM sampling kernels.
- Zero-initialized offsets, making the transferred model initially equivalent
  to the original CoANet up to floating-point interpolation error.
- Five-stage gradual unfreezing with discriminative learning rates.
- Original segmentation and connectivity supervision retained.

## Installation

Kaggle already includes PyTorch and torchvision:

```bash
python -m pip install -r requirements.txt
```

On a local machine, install a CUDA-compatible PyTorch build first, then run the
same command.

## Training

```bash
python train.py \
  --dataset spacenet \
  --data-dir /path/to/spacenet-prepared \
  --coanet-weights /path/to/CoANet-spacenet.pth.tar \
  --scm-type dsconv \
  --dsconv-kernel-size 9 \
  --unfreeze-epochs 5 15 30 45 \
  --lr 0.001 \
  --batch-size 2 \
  --epochs 150 \
  --checkname CoANet-DSConv
```

See [DSCONV_TRANSFER.md](DSCONV_TRANSFER.md) for architecture and transfer
details, and [KAGGLE.md](KAGGLE.md) for the complete Kaggle workflow.

Datasets, checkpoints and training outputs are deliberately excluded from Git.
Publish the pretrained checkpoint as a Kaggle Dataset or GitHub Release asset.

The `CoANet-main` and `DSCNet-main` directories contain upstream reference code;
refer to their respective README and license files for attribution and terms.
