# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Conda env**: `vision_env` (Python at `/home/sanng/miniconda3/envs/vision_env/bin/python`, torch 2.4.0 with CUDA)
- **PYTHONPATH**: Always set `PYTHONPATH=$PWD` from project root before running scripts. The codebase uses absolute imports like `from cv_nets...` and `from models...`.
- **CUDA memory**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `train.py` to prevent OOM fragmentation.

## Common Commands

```bash
# Activate environment
conda activate vision_env
export PYTHONPATH=$PWD

# Train a model
python train.py --dataset camvid --variant base
python train.py --dataset voc --epochs 150 --batch-size 16 --class-weights --amp
python train.py --dataset drive --epochs 100 --batch-size 16 --variant nano

# Run full benchmark
python benchmark/run_benchmark.py --all
python benchmark/run_benchmark.py --variant base --dataset camvid

# Download datasets
python tools/download_datasets.py --dataset all
python tools/download_datasets.py --dataset camvid pascal_voc

# Generate paper-quality plots from benchmark results
python tools/paper_plots.py --results-dir benchmark/results --output paper_figures/
```

There is no linting setup, test suite, or package build step in this project. The primary workflow is training via `train.py` and evaluating via benchmark scripts.

## Architecture

U-MobileViT-Net is a lightweight U-Net style semantic segmentation model using MobileViTv2-style separable self-attention transformer blocks. It targets edge/mobile deployment with variants from ~0.3M to ~7M parameters.

### Two major source trees

1. **`cv_nets/`** — A reusable, framework-style computer vision library with custom layers (`Conv2d`, specialized activations, pooling, normalizations), building blocks (MobileNetV2 inverted bottlenecks, MobileViT attention, transformers), and utilities (config helpers, registry, functional ops like `unfold_custom`/`fold_custom`). This is akin to a lightweight `torchvision` replacement and is a **dependency** of the model code — model classes inherit from `cv_nets.layers.base_layer.BaseLayer`.

2. **`models/` + `tools/` + `train.py`** — The U-MobileViT-Net project proper.

### Model architecture (`models/u_mobilevit_net/`)

The model (`UMobileViT` in `u_models.py`) has three stages:

```
Input → Encoder → Decoder → SegHead → Output
```

- **Encoder** (`encoder_block.py`): A 3-stage stem (Conv→ReLU downsample chains producing features at 1/2, 1/4, 1/8 resolution) followed by 3 MobileViT encoder stages with depthwise downsampling. Each stage contains `UMobileViTEncoderLayer` instances (1, 2, 2 copies respectively). Stage 3 uses `patch_size=(1,1)` (no unfolding).

- **Decoder** (`decoder_block.py`): 3-stage upsampling path. Stage 1 (2 layers, x2 upsample), Stage 2 (1 layer, x4 total), and an output block (x8 total with a `UMobileViTDecoderConcatLayer`). Each decoder layer uses cross-attention (`TransformerDecoderLayer`) between the upsampled feature and encoder skip connection.

- **Core building block** — `_UMobileViTLayer` (`module.py`): Every encoder/decoder layer is composed of:
  1. **Local block**: depthwise 3×3 → pointwise 1×1 → GroupNorm (convolutional feature extraction)
  2. **Global block**: N transformer blocks operating on unfolded patches (self-attention in encoder, cross-attention in decoder)
  3. **Expansion block**: MobileNetV2-style inverted bottleneck (expand → depthwise → project) with ReLU6 + GroupNorm
  4. **Output norm**: GroupNorm on the residual connection
  5. Gradient checkpointing is enabled by default on global and expansion blocks to save VRAM (~40%).

- **Segmentation head** (`seg_head.py`):
  - `UpsampleHead`: 3-stage upsampling (x2, x4, x8) with skip connections from encoder stem outputs via `UMobileViTDecoderConcatLayer` (additive memory projection, no transformer).
  - `FeatureRefinementBlock`: Dilated conv + spatial attention for refining boundaries.
  - `ContextAwareSegHead`: Single-task head with upsample → refinement → lightweight classifier (depthwise 3×3 + pointwise 1×1).
  - `MultiTaskDenseHead`: Multi-task variant that shares the expensive upsample + refinement pipeline and only duplicates the final lightweight classifier per task. This is the **default** head.

- **Transformer** (`transfomer.py`): Implements separable self-attention and cross-attention (MobileViTv2-style) with linear attention as the default. The `separable_attention_forward` function in `cv_nets/utils/functional.py` is the core op.

### Variants (`configs.py`)

| Variant | d_model | Transformer Blocks | Expansion | ~Params |
|---------|---------|-------------------|-----------|---------|
| nano    | 48      | 1                 | 2.0       | ~0.3M   |
| base    | 64      | 2                 | 3.0       | ~0.6M   |
| pro     | 128     | 3                 | 3.0       | ~2.0M   |
| promax  | 256     | 3                 | 4.0       | ~7.0M   |

The `d_model` must be divisible by 4, 2, and 8 for stem/seg_head compatibility. `GroupNorm` groups are auto-computed by `get_groupnorm_groups()` to find compatible divisors when channel counts don't divide evenly.

### `train.py` — Unified training script

Uses argparse with dataset-specific defaults in `DATASET_DEFAULTS`. Components:
- **Optimizer**: AdamW with configurable weight decay
- **Scheduler**: Linear warmup (5 epochs by default) + CosineAnnealingLR or PolyLR
- **Loss**: Combined CE/BCE + Dice loss (50/50 split), with optional class weights (inverse frequency), label smoothing, and pos_weight for binary tasks
- **EMA**: Exponential Moving Average of model weights for validation
- **AMP**: Automatic mixed precision via `torch.cuda.amp`
- **Gradient checkpointing**: On by default, auto-disabled on NaN recovery
- **NaN recovery**: If all batches are skipped due to NaN, reloads best checkpoint, reduces LR 10×, and disables checkpointing (up to 3 attempts)

### `tools/training_utils.py` — Training engine

`SegmentationTrainer` is the main training loop class. Key features:
- Gradient accumulation support (`grad_accum_steps`)
- Per-epoch GPU cache cleanup to prevent fragmentation OOM
- Global mIoU accumulation (intersection/union over entire validation set, not per-batch averaging)
- Checkpointing saves `best_model.pth` (full state) and `last_model.pth` (state_dict only for memory)

### `tools/dataset_loader.py` — Data pipeline

Dataset registry in `DATASET_INFO` dict supporting: `voc` (21 classes), `camvid` (11 classes), `coco_leaf` (8 classes), `drive`, `kvasir`, `isic` (binary). `ComposeAugmentation` applies synchronized geometric transforms to image+mask with three intensity levels (`light`/`medium`/`strong`). Validation uses only Resize + CenterCrop (no random augmentation).

### `benchmark/` — Experiment orchestration

`run_benchmark.py` orchestrates multi-dataset, multi-variant training runs by shelling out to `train.py` as a subprocess. Results go to `benchmark/results/<dataset>/<variant>_<timestamp>/`.

## Key conventions

- All model classes inherit from `cv_nets.layers.base_layer.BaseLayer` and use `cv_nets.utils.config_helper.get_param()` for configuration resolution (argparse opts → explicit arg → default).
- Model code is primarily in Vietnamese with English technical terms. Comments and docstrings are mixed Vietnamese/English.
- Convolution uses the custom `Conv2d` from `cv_nets.layers.conv_layer` throughout, not `torch.nn.Conv2d`.
- NaN handling is a first-class concern: input validation, gradient checking, weight initialization checks, and auto-recovery are built into the training loop.
- The model supports fully dynamic input sizes (no fixed spatial dimensions) — tested with rectangular inputs.
