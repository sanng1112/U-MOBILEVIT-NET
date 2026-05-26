# U-MobileViT-Net

A lightweight hybrid segmentation network combining **U-Net**, **MobileNetV2** inverted-residual blocks, and **MobileViTv2-style linear self-attention**. Designed as a VRAM-friendly alternative to the standard U-Net baseline for binary/foreground segmentation on COCO-style datasets.

## Overview

This repository implements two segmentation models built on a small in-house ML library (`cv_nets`):

- **UNetLite** (`models/basemodels.py`) — a clean U-Net baseline with double-conv blocks, used as the reference model.
- **UNetMobileViT** (`models/u_mobilevit_net_base.py`) — the proposed hybrid model:
  - Encoder/decoder built from `MV2Block` (MobileNetV2 inverted residual, depthwise-separable conv).
  - `LinearSelfAttention` (MobileViTv2 style) applied at the bottleneck and the two deepest decoder stages.
  - Residual addition (instead of channel concat) on the shallower decoder stages to reduce memory.
  - `QuantStub` / `DeQuantStub` / `FloatFunctional` wiring so the model is ready for post-training quantization.

The model targets `320×320` inputs and outputs a single-channel logits map (`num_classes=1` by default).

## Project structure

```
U-MOBILEVIT-NET/
├── cv_nets/                       # in-house ML building blocks
│   ├── blocks/                    # ConvBNAct, ResnetBlock, MV2Block
│   ├── config/demo.yaml           # example config (activation, linear, dropout)
│   ├── layers/                    # Conv2d, ConvTranspose2d, Dropout, Linear,
│   │                              # LinearSelfAttention, activation/, normalization/, pooling/
│   ├── loss_fn/                   # base criterion + classification losses
│   ├── utils/                     # config_helper, registry, logger, import_utils
│   └── main.py                    # config loader demo
├── models/
│   ├── basemodels.py              # UNetLite baseline
│   ├── u_mobilevit_net_base.py    # UNetMobileViT (main model)
│   └── unet_base/                 # baseline checkpoints + training_history.json
│       ├── best_unet_base.pth
│       ├── last_unet_base.pth
│       └── training_history.json
├── u_mobilevit_net_base.ipynb     # training/eval notebook for UNetMobileViT
├── unet_base.ipynb                # training/eval notebook for the U-Net baseline
├── unet_mobilevit_weights.pth     # pretrained UNetMobileViT weights
└── init.py                        # `from cv_nets import *`
```

## Requirements

- Python 3.9+
- PyTorch (with `torch.ao.quantization` and `torch.nn.quantized`)
- torchvision
- numpy, Pillow, matplotlib, tqdm, PyYAML
- pycocotools (for the COCO dataloader used in the notebooks)

Install with:

```bash
pip install torch torchvision numpy pillow matplotlib tqdm pyyaml pycocotools
```

## Quick start

### Build the model

```python
from models.u_mobilevit_net_base import UNetMobileViT

class DummyOpts: pass
opts = DummyOpts()

model = UNetMobileViT(opts=opts, num_classes=1)
```

The `opts` object is passed through to layer builders (activation/normalization/pooling). A YAML-driven config example is available at `cv_nets/config/demo.yaml` and loaded via `cv_nets/main.py`.

### Convergence sanity check

`u_mobilevit_net_base.py` ships with two helpers used during development:

```python
from models.u_mobilevit_net_base import (
    UNetMobileViT, check_model_convergence, save_and_profile_model,
)

model = UNetMobileViT(opts=opts, num_classes=1)

# Force the model to overfit a single synthetic batch for 100 epochs.
check_model_convergence(model, device="cuda")

# Save weights + print parameter count / file size.
save_and_profile_model(model, filepath="unet_mobilevit_weights.pth")
```

Run directly:

```bash
python models/u_mobilevit_net_base.py
```

### Load pretrained weights

```python
import torch
from models.u_mobilevit_net_base import UNetMobileViT

model = UNetMobileViT(opts=opts, num_classes=1)
model.load_state_dict(torch.load("unet_mobilevit_weights.pth", map_location="cpu"))
model.eval()
```

### Training notebooks

End-to-end training and evaluation on a COCO-style dataset (binary foreground segmentation, BCEWithLogitsLoss, Adam) are provided in:

- `unet_base.ipynb` — U-Net baseline.
- `u_mobilevit_net_base.ipynb` — U-MobileViT-Net.

The baseline run's metrics are persisted to `models/unet_base/training_history.json` and the best/last checkpoints to `models/unet_base/`.

## Model details

**Encoder.** Four `MV2Block` stages with channel widths `[16, 32, 64, 128]`, each followed by a strided `2×2` Conv2d for downsampling.

**Bottleneck.** `MV2Block` expands `128 → 256` channels, followed by `Attention2DWrapper(LinearSelfAttention, patch_size=2)`, followed by another `MV2Block`.

**Decoder.** Mirrors the encoder with `ConvTranspose2d` upsamples and `MV2Block` refinements. The two deepest decoder stages (channels `128`, `64`) use `LinearSelfAttention` to fuse skip connections; the two shallower stages use residual add to save memory.

**Head.** A `1×1` Conv2d projects to `num_classes`.

**Quantization-ready.** Inputs/outputs pass through `QuantStub` / `DeQuantStub`, and residual additions use `FloatFunctional`, so the model can be exported through PyTorch's eager-mode static quantization flow.

## Notes

- `init.py` re-exports the `cv_nets` package (`from cv_nets import *`).
- `cv_nets/main.py` is a small demo that loads `config/demo.yaml` into a `SimpleNamespace` — useful as a reference for the `opts` interface consumed by layer builders.
- Code comments and print statements in the model files are partially in Vietnamese.
