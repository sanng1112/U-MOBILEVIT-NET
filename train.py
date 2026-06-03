#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    U-MobileViT-Net — Unified Training Script                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Script huấn luyện thống nhất cho tất cả các bộ dữ liệu semantic segmentation.

Hỗ trợ:
    - PASCAL VOC 2012  (21 classes, multi-class)
    - CamVid            (11 classes, multi-class)
    - DRIVE             (1 class, binary — retinal vessel)
    - Kvasir-SEG        (1 class, binary — polyp)
    - ISIC 2018         (1 class, binary — skin lesion)

Cải tiến so với notebooks cũ:
    ✦ Optimizer: AdamW (weight_decay=1e-4) thay cho Adam
    ✦ Scheduler: Linear Warmup (5 epochs) + CosineAnnealingLR
    ✦ Validation: Chỉ dùng Resize + CenterCrop (không augmentation ngẫu nhiên)
    ✦ Class Weights: Tự động tính inverse class frequency cho multi-class datasets

Cách dùng:
    # Cơ bản
    python train.py --dataset camvid

    # Đầy đủ tham số
    python train.py --dataset voc --epochs 150 --batch-size 32 --lr 1e-3 \\
                    --image-size 320 320 --class-weights

    # DRIVE (binary)
    python train.py --dataset drive --epochs 100 --batch-size 16

Yêu cầu:
    conda activate vision_env
    export PYTHONPATH=$PWD
"""

import sys
import os

# [OOM Fix] Ngăn CUDA memory fragmentation — PyTorch khuyến nghị trong error message.
# - expandable_segments: cho phép mở rộng segment thay vì fail khi không tìm được block liên tục
# - max_split_size_mb: giới hạn split block lớn → giảm phân mảnh (quan trọng cho promax ~7M params)
# - garbage_collection_threshold: trigger GC sớm hơn khi có nhiều block rác
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6'
)

import argparse
import json
import time
from pathlib import Path

# ── Thiết lập PYTHONPATH ──────────────────────────────────────────
def _setup_path():
    """Tự động thêm project root vào sys.path."""
    current = Path(__file__).resolve().parent
    if str(current) not in sys.path:
        sys.path.insert(0, str(current))
    # Đảm bảo PYTHONPATH chứa project root
    if str(current) not in os.environ.get('PYTHONPATH', ''):
        os.environ['PYTHONPATH'] = str(current)

_setup_path()

import numpy as np
import torch
from torch import nn

# U-MobileViT-Net
from models.u_mobilevit_net.u_models import umobilevit, UMobileViT
from models.u_mobilevit_net.configs import get_variant

# Dataset & Training tools
from tools.data import (
    create_dataloaders, label_to_color, denormalize,
)
from tools.training import (
    SegmentationTrainer, TrainingConfig,
    compute_class_weights, compute_pos_weight,
)
from tools.visualization import (
    plot_training_curves, show_dataset_samples, show_predictions,
)


# ═══════════════════════════════════════════════════════════════════
# Argument Parser
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='U-MobileViT-Net: Unified Training for Semantic Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python train.py --dataset camvid
  python train.py --dataset voc --epochs 150 --class-weights
  python train.py --dataset drive --epochs 100 --batch-size 16
  python train.py --dataset isic --lr 5e-4
        """
    )

    # ── Dataset ──
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['voc', 'camvid', 'drive', 'kvasir', 'isic', 'coco_leaf'],
                        help='Tên bộ dữ liệu (voc/camvid/drive/kvasir/isic/coco_leaf)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Đường dẫn tùy chỉnh đến thư mục dữ liệu')

    # ── Model ──
    parser.add_argument('--variant', type=str, default='base',
                        choices=['nano', 'base', 'pro', 'promax'],
                        help='Model variant: nano|base|pro|promax (mặc định: base)')
    parser.add_argument('--image-size', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help='Kích thước ảnh đầu vào (mặc định: dataset-specific)')
    parser.add_argument('--head', type=str, default='single',
                        choices=['single'],
                        help='Loại segmentation head')

    # ── Training ──
    parser.add_argument('--epochs', type=int, default=300,
                        help='Tổng số epoch huấn luyện (mặc định: 300)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (mặc định: tự động chọn theo dataset)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Số worker cho DataLoader (mặc định: 8)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (mặc định: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay cho AdamW (mặc định: 1e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Số epoch linear warmup (mặc định: 5)')
    parser.add_argument('--min-lr-ratio', type=float, default=0.01,
                        help='Tỷ lệ min_lr / lr cho cosine annealing (mặc định: 0.01)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm (mặc định: 1.0)')

    # ── Augmentation ──
    parser.add_argument('--aug-intensity', type=str, default='medium',
                        choices=['light', 'medium', 'strong'],
                        help='Cường độ data augmentation (mặc định: medium)')

    # ── Regularization ──
    parser.add_argument('--class-weights', action='store_true', default=False,
                        help='Tính class weights dựa trên inverse class frequency '
                             '(khuyến nghị cho multi-class datasets như CamVid, VOC)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing epsilon (0 = tắt, 0.1 khuyến nghị cho multi-class)')
    parser.add_argument('--grad-centralization', action='store_true', default=False,
                        help='Bật Gradient Centralization để cải thiện stability')

    # ── Scheduler ──
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'poly'],
                        help='Loại LR scheduler (mặc định: cosine)')
    parser.add_argument('--poly-power', type=float, default=0.9,
                        help='Poly power cho PolyLR scheduler (mặc định: 0.9)')

    # ── EMA ──
    parser.add_argument('--no-ema', action='store_true', default=False,
                        help='Tắt Exponential Moving Average (EMA)')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay factor (mặc định: 0.999)')

    # ── Mixed Precision ──
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Bật Automatic Mixed Precision (AMP) — TẮT mặc định để tránh NaN float16')
    parser.add_argument('--no-cudnn-benchmark', action='store_true', default=False,
                        help='Tắt cuDNN benchmark')

    # ── Early Stopping ──
    parser.add_argument('--patience', type=int, default=None,
                        help='Patience cho early stopping (mặc định: tự động)')
    parser.add_argument('--min-epochs', type=int, default=30,
                        help='Số epoch tối thiểu trước khi early stopping (mặc định: 30)')

    # ── Output ──
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Thư mục lưu checkpoint (mặc định: ./checkpoints/<dataset>)')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Không hiển thị biểu đồ (non-interactive mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed cho reproducibility (mặc định: 42)')

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Dataset-Specific Defaults
# ═══════════════════════════════════════════════════════════════════

DATASET_DEFAULTS = {
    'camvid': {
        'batch_size': 24,
        'image_size': (320, 320),
        'epochs': 500,
        'patience': 20,
        'use_class_weights': True,
        'aug_intensity': 'strong',
        'scheduler': 'poly',
        'lr': 1e-3,
    },
    'voc': {
        'batch_size': 16,
        'image_size': (320, 320),
        'epochs': 200,
        'patience': 20,
        'use_class_weights': True,
        'aug_intensity': 'strong',
        'scheduler': 'poly',
        'lr': 1e-3,
    },
    'drive': {
        'batch_size': 16,
        'image_size': (320, 320),
        'epochs': 150,
        'patience': 15,
        'use_class_weights': False,
        'aug_intensity': 'medium',
        'scheduler': 'cosine',
        'lr': 5e-4,
    },
    'kvasir': {
        'batch_size': 16,
        'image_size': (320, 320),
        'epochs': 150,
        'patience': 15,
        'use_class_weights': False,
        'aug_intensity': 'medium',
        'scheduler': 'cosine',
        'lr': 5e-4,
    },
    'isic': {
        'batch_size': 16,
        'image_size': (320, 320),
        'epochs': 150,
        'patience': 15,
        'use_class_weights': False,
        'aug_intensity': 'medium',
        'scheduler': 'cosine',
        'lr': 5e-4,
    },
    'coco_leaf': {
        'batch_size': 16,
        'image_size': (320, 320),
        'epochs': 300,
        'patience': 20,
        'use_class_weights': True,
        'aug_intensity': 'medium',
        'scheduler': 'poly',
        'lr': 1e-3,
    },
}


# ═══════════════════════════════════════════════════════════════════
# Main Training Routine
# ═══════════════════════════════════════════════════════════════════

def build_model(num_classes: int, image_size: tuple, variant: str = 'base',
                head_type: str = 'single'):
    """Khởi tạo U-MobileViT-Net với variant được chọn.

    [ĐÃ SỬA] Dùng Namespace rỗng thay vì parse_known_args().
    Trước đây, ArgumentParser với defaults của Encoder/Decoder (d_model=64,
    expansion_factor=3.0, num_transformer_blocks=2 — khớp variant base) đã
    âm thầm ghi đè cấu hình của variant pro/promax qua get_param().

    Với Namespace rỗng, get_param() sẽ fallback về default_val trong variant
    config, đảm bảo mỗi variant nhận đúng tham số của nó.
    """
    from argparse import Namespace
    opts = Namespace()
    return umobilevit(opts=opts, variant=variant, head=head_type,
                     out_channels=num_classes)


def set_seed(seed: int):
    """Cố định random seed cho reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    # ── Thiết lập mặc định theo dataset ──
    defaults = DATASET_DEFAULTS[args.dataset]
    if args.batch_size is None:
        args.batch_size = defaults['batch_size']
    if args.epochs is None:
        args.epochs = defaults['epochs']
    if args.patience is None:
        args.patience = defaults['patience']
    if args.image_size is None:
        args.image_size = defaults['image_size']
    if args.aug_intensity is None:
        args.aug_intensity = defaults.get('aug_intensity', 'medium')
    if args.scheduler is None:
        args.scheduler = defaults.get('scheduler', 'cosine')
    if not args.class_weights:
        args.class_weights = defaults.get('use_class_weights', False)

    image_size = tuple(args.image_size)

    # ── Seed ──
    set_seed(args.seed)

    # ── Device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.no_cudnn_benchmark:
        torch.backends.cudnn.benchmark = (device.type == 'cuda')

    # ── Output Directory ──
    save_dir = args.save_dir or f'./checkpoints/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # Print Configuration
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  U-MobileViT-Net — Unified Training")
    print(f"{'='*70}")
    print(f"  Dataset:      {args.dataset.upper()}")
    print(f"  Variant:      {args.variant.upper()}")
    print(f"  Image size:   {image_size}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Device:       {device}")
    print(f"  LR:           {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup:       {args.warmup_epochs} epochs")
    print(f"  Scheduler:    {args.scheduler.upper()}")
    print(f"  Optimizer:    AdamW")
    print(f"  EMA:          {'No' if args.no_ema else f'Yes (decay={args.ema_decay})'}")
    print(f"  Class weights:{'Yes' if args.class_weights else 'No'}")
    print(f"  Label smooth: {args.label_smoothing if args.label_smoothing > 0 else 'No'}")
    print(f"  Grad Central: {'Yes' if args.grad_centralization else 'No'}")
    print(f"  AMP:          {'Yes' if args.amp else 'No (FP32)'}")
    print(f"  Augmentation: {args.aug_intensity}")
    print(f"  Seed:         {args.seed}")
    print(f"  Save dir:     {os.path.abspath(save_dir)}")
    print(f"{'='*70}\n")

    # ═══════════════════════════════════════════════════════════════
    # 1. Nạp Dữ liệu
    # ═══════════════════════════════════════════════════════════════
    print("[1/5] Đang nạp dữ liệu...")
    train_loader, val_loader, info = create_dataloaders(
        args.dataset,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_intensity=args.aug_intensity,
        data_root=args.data_root,
    )

    print(f"\n  {info['name']}")
    print(f"  Classes:       {info['num_classes']} ({info['type']})")
    print(f"  Train samples: {info['train_size']:,}")
    print(f"  Val samples:   {info['val_size']:,}")
    if info.get('class_names'):
        class_names = info['class_names']
        print(f"  Class names:   {', '.join(class_names[:5])}"
              f"{'...' if len(class_names) > 5 else ''}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 2. Khởi tạo Mô hình
    # ═══════════════════════════════════════════════════════════════
    print("[2/5] Đang khởi tạo mô hình...")
    model = build_model(info['num_classes'], image_size, args.variant, args.head)
    n_params = sum(p.numel() for p in model.parameters())
    variant_cfg = get_variant(args.variant)
    print(f"  U-MobileViT-Net [{args.variant.upper()}] | {n_params:,.0f} params ({n_params/1e6:.2f}M)")
    print(f"    d_model={variant_cfg['d_model']}, "
          f"transformer_blocks={variant_cfg['num_transformer_blocks']}, "
          f"expansion={variant_cfg['expansion_factor']}")

    # Smoke test
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, *image_size)
        out = model(dummy)
        expected_shape = (2, info['num_classes'], *image_size)
        assert out.shape == expected_shape, \
            f"[LỖI] Output shape mismatch: {tuple(out.shape)} vs {expected_shape}"
        print(f"  [Smoke test] ✓ {tuple(dummy.shape)} → {tuple(out.shape)}")

    # NaN check
    model = model.to(device)
    nan_params = [name for name, p in model.named_parameters()
                  if not torch.isfinite(p).all()]
    if nan_params:
        raise RuntimeError(
            f"[LỖI] {len(nan_params)} tham số chứa NaN/Inf: {nan_params[:5]}..."
        )
    print(f"  [NaN check] ✓ All weights are finite")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 3. Class Weights (nếu được yêu cầu)
    # ═══════════════════════════════════════════════════════════════
    class_weights = None
    pos_weight = None
    if args.class_weights and info['type'] == 'multi-class':
        print("[3/5] Đang tính class weights...")
        class_weights = compute_class_weights(
            train_loader, info['num_classes'],
            ignore_index=info.get('ignore_index', 255),
            device=device,
        )
        print()
    elif info['type'] == 'binary':
        print("[3/5] Đang tính pos_weight cho binary segmentation...")
        pos_weight = compute_pos_weight(train_loader, device=device)
        print()
    else:
        print(f"[3/5] Bỏ qua class weights "
              f"({'binary dataset' if info['type'] == 'binary' else '--class-weights not set'})\n")

    # ═══════════════════════════════════════════════════════════════
    # 4. Thiết lập Trainer
    # ═══════════════════════════════════════════════════════════════
    print("[4/5] Đang thiết lập trainer...")
    config = TrainingConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        scheduler_type=args.scheduler,
        poly_power=args.poly_power,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        use_amp=args.amp,  # [FP32] Mặc định False — không dùng AMP
        grad_clip_norm=args.grad_clip,
        label_smoothing=args.label_smoothing,
    )
    trainer = SegmentationTrainer(
        model=model,
        device=device,
        dataset_info=info,
        save_dir=save_dir,
        config=config,
        class_weights=class_weights,
        pos_weight=pos_weight,
    )

    print(f"  Loss:      {trainer.criterion.__class__.__name__}")
    print(f"  Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"  Scheduler: {args.scheduler.upper()} + Warmup ({args.warmup_epochs} epochs)")
    print(f"  Grad Clip: {config.grad_clip_norm}")
    print(f"  EMA:       {'Yes' if not args.no_ema else 'No'}")
    print(f"  AMP:       {'Yes' if args.amp else 'No (FP32)'}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 5. Huấn luyện
    # ═══════════════════════════════════════════════════════════════
    print("[5/5] Bắt đầu huấn luyện...")
    print(f"{'='*70}\n")

    t_start = time.time()

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        save_prefix='best',
        min_lr_ratio=args.min_lr_ratio,
    )

    total_time = time.time() - t_start
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # ═══════════════════════════════════════════════════════════════
    # Tổng kết
    # ═══════════════════════════════════════════════════════════════
    metric_name = 'Dice' if info['type'] == 'binary' else 'mIoU'
    best_epoch = trainer.best_epoch
    best_metric = max(history['val_metric'])
    best_val_loss = min(history['val_loss'])

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — {info['name']}")
    print(f"{'='*70}")
    print(f"  Total time:         {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"  Model params:       {n_params:,.0f} ({n_params/1e6:.2f}M)")
    print(f"  Input size:         {image_size}")
    print(f"  Classes:            {info['num_classes']} ({info['type']})")
    print(f"  Train/Val samples:  {info['train_size']:,} / {info['val_size']:,}")
    print(f"  Augmentation:       {args.aug_intensity}")
    print(f"  Best epoch:         {best_epoch}")
    print(f"  Best {metric_name}:          {best_metric:.4f}")
    print(f"  Best val loss:      {best_val_loss:.4f}")
    print(f"  Checkpoint:         {os.path.abspath(save_dir)}")
    print(f"{'='*70}")

    # ── Biểu đồ ──
    if not args.no_plot:
        try:
            plot_training_curves(history,
                                 save_path=os.path.join(save_dir, 'training_curves.png'))
        except Exception as e:
            print(f"[Cảnh báo] Không thể vẽ biểu đồ: {e}")

    return model, trainer, history


# ═══════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
