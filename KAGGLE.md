# Train CoANet-DSConv on Kaggle

## 1. Clone and install

In a Kaggle notebook, enable a GPU and Internet, then run:

```bash
!git clone https://github.com/USER/REPOSITORY.git
%cd REPOSITORY
!python -m pip install -q -r requirements.txt
```

Kaggle supplies `torch` and `torchvision`; `requirements.txt` deliberately does
not reinstall them. Verify the runtime before training:

```python
import torch, torchvision
print(torch.__version__, torchvision.__version__)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Restart the notebook kernel after `pip install` if Kaggle reports that a loaded
package was replaced.

## 2. Attach data and pretrained weights

Do not commit SpaceNet imagery or `CoANet-spacenet.pth.tar` to GitHub. The
checkpoint is larger than GitHub's normal file limit. Upload them as private or
public Kaggle Datasets and attach those datasets to the notebook.

Typical read-only Kaggle paths are:

```text
/kaggle/input/spacenet-prepared/
/kaggle/input/coanet-pretrained/CoANet-spacenet.pth.tar
```

Kaggle inputs are read-only. Training outputs should go under `/kaggle/working`.

## 3. Start progressive transfer learning

```bash
!python train.py \
  --dataset spacenet \
  --data-dir /kaggle/input/spacenet-prepared \
  --coanet-weights /kaggle/input/coanet-pretrained/CoANet-spacenet.pth.tar \
  --scm-type dsconv \
  --dsconv-kernel-size 9 \
  --dsconv-extend-scope 1.0 \
  --unfreeze-epochs 5 15 30 45 \
  --lr 0.001 \
  --batch-size 2 \
  --workers 2 \
  --gpu-ids 0 \
  --epochs 150 \
  --checkname CoANet-DSConv-Kaggle
```

If CUDA runs out of memory, reduce `--batch-size` to 1. Kaggle notebooks usually
work more reliably with `--workers 2` or `--workers 4` than the desktop default.

Use `--scm-type strip` with the same split and seed to obtain the baseline for
an ablation comparison.

## 4. Save results

The repository ignores checkpoints and run outputs. Before the Kaggle session
ends, save the useful checkpoint as notebook output or create a new Kaggle
Dataset version from `/kaggle/working`.
