# CoANet with Dynamic Snake SCM

The rewritten model keeps CoANet's four-direction SCM layout but replaces each
fixed strip convolution with a topology-constrained `DSConv2d`.  A legacy
`CoANet-spacenet.pth.tar` checkpoint is migrated automatically: all horizontal,
vertical and transformed diagonal strip kernels are copied into DSConv sampling
kernels, while the new offset predictors start at zero.

## Recommended transfer run

```bash
python train.py \
  --dataset spacenet \
  --data-dir RGB_1.0_meter \
  --coanet-weights CoANet-spacenet.pth.tar \
  --scm-type dsconv \
  --dsconv-kernel-size 9 \
  --dsconv-extend-scope 1.0 \
  --unfreeze-epochs 5 15 30 45 \
  --lr 0.001 \
  --batch-size 2 \
  --epochs 150 \
  --checkname CoANet-DSConv
```

DSConv expands a feature map along its nine sampling points, so start with
batch size 1 or 2 at 512x512 and increase only after checking GPU memory.

## Unfreezing stages

| Stage | Epoch (default) | Trainable modules |
|---|---:|---|
| 0 | 0 | DSConv offset predictors only |
| 1 | 5 | decoder and connectivity heads |
| 2 | 15 | stage 1, ASPP and ResNet layer4 |
| 3 | 30 | stage 2 and ResNet layer3 |
| 4 | 45 | the complete model |

The optimizer contains every parameter from the beginning. Frozen parameters
therefore become trainable at a later stage without rebuilding the optimizer.
BatchNorm statistics remain frozen wherever their owning layer is frozen.

For an ablation with the original SCM, pass `--scm-type strip`. To disable the
schedule and fine-tune all layers immediately, pass `--no-progressive-transfer`.

The existing segmentation and two connectivity losses are retained. They give
the DSConv offsets road-connectivity supervision without discarding CoANet's
original topological learning objective.
