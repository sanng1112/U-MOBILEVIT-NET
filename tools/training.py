"""
Training engine for U-MobileViT-Net semantic segmentation experiments.

Provides loss functions, learning-rate schedulers, exponential moving
average (EMA), gradient checkpointing, and the central
:class:`SegmentationTrainer` class that handles the full training loop
with mixed precision, NaN auto-recovery, early stopping, and checkpoint
management.

Usage:
    from tools.data import create_dataloaders
    from tools.training import SegmentationTrainer, TrainingConfig

    train_loader, val_loader, info = create_dataloaders("coco_leaf")
    model = umobilevit_base(out_channels=info.num_classes, head="single")
    config = TrainingConfig(lr=1e-3, scheduler_type="poly")
    trainer = SegmentationTrainer(model, device, info, "checkpoints/", config)
    history = trainer.train(train_loader, val_loader, epochs=300)
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameter configuration for :class:`SegmentationTrainer`.

    Attributes:
        lr: Peak learning rate after warmup.
        weight_decay: AdamW weight decay coefficient.
        warmup_epochs: Number of linear warmup epochs.
        scheduler_type: ``cosine`` or ``poly``.
        poly_power: Exponent for PolyLR schedule.
        min_lr_ratio: Minimum LR as a fraction of *lr*.
        use_ema: Enable exponential moving average of model weights.
        ema_decay: EMA decay factor (closer to 1 = smoother).
        use_amp: Enable automatic mixed precision (CUDA only).
        grad_clip_norm: Maximum gradient norm.
        grad_accum_steps: Number of batches to accumulate before stepping.
        label_smoothing: Label smoothing epsilon (0 = disabled).
        use_checkpointing: Enable gradient checkpointing for VRAM savings.
        nan_recovery: Automatically reload best checkpoint on NaN crisis.
    """
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    scheduler_type: str = "poly"         # "cosine" | "poly"
    poly_power: float = 0.9
    min_lr_ratio: float = 0.01
    use_ema: bool = True
    ema_decay: float = 0.999
    use_amp: bool = False  # [FP32] Tắt AMP mặc định để tránh NaN từ float16 overflow
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    label_smoothing: float = 0.0
    use_checkpointing: bool = True
    nan_recovery: bool = True


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class BinarySegLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation.

    Loss = (1 - λ) * BCE + λ * DiceLoss
    """

    def __init__(
        self, bce_weight: float = 0.5, dice_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: inputs (B,1,H,W) logits, targets (B,H,W) or (B,1,H,W) binary."""
        inputs = inputs.float()
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()

        bce = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = (1.0 - dice).mean()
        combined = self.bce_weight * bce + self.dice_weight * dice_loss
        return combined, bce.detach(), dice_loss.detach()


class MultiClassSegLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss for multi-class segmentation.

    Loss = (1 - λ) * CE + λ * DiceLoss (averaged over present classes).

    Supports class weights and label smoothing.
    """

    def __init__(
        self, ce_weight: float = 0.5, dice_weight: float = 0.5,
        num_classes: int = 21, ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: inputs (B,C,H,W) logits, targets (B,H,W) class indices."""
        inputs = inputs.float()
        targets = targets.long()

        ce = F.cross_entropy(
            inputs, targets, weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        probs = F.softmax(inputs, dim=1)
        valid_mask = (targets != self.ignore_index).float()
        dice_sum = torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)
        present = 0
        for c in range(self.num_classes):
            target_c = (targets == c).float()
            if target_c.sum() == 0:
                continue
            pred_c = probs[:, c, :, :] * valid_mask
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_c = (2.0 * intersection + smooth) / (union + smooth)
            dice_sum += 1.0 - dice_c
            present += 1

        dice = dice_sum / max(present, 1)
        combined = self.ce_weight * ce + self.dice_weight * dice
        return combined, ce.detach(), dice.detach()


class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights well-classified pixels, focusing training on hard examples
    such as boundaries and small objects.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCELoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    Replaces one-hot targets with a soft distribution to reduce overconfidence
    and improve calibration.
    """

    def __init__(
        self, smoothing: float = 0.1, ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.long()
        batch_size, num_classes = inputs.shape[:2]
        mask = (targets != self.ignore_index).float()
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(
                1, targets.unsqueeze(1),
                1.0 - self.smoothing + self.smoothing / (num_classes - 1),
            )
            smooth_targets = smooth_targets * mask.unsqueeze(1)
        if self.class_weights is not None:
            weight = self.class_weights.view(1, -1, 1, 1).expand_as(log_probs)
            loss = -(smooth_targets * log_probs * weight).sum(dim=1) * mask
        else:
            loss = -(smooth_targets * log_probs).sum(dim=1) * mask
        return loss.sum() / (mask.sum() + 1e-6)


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def create_loss_fn(
    dataset_type: str,
    num_classes: int,
    ignore_index: int = 255,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Return the appropriate loss for a given dataset type.

    Args:
        dataset_type: ``binary`` or ``multi-class``.
        num_classes: Number of output classes.
        ignore_index: Pixel value to ignore.
        class_weights: Per-class weights for CE loss.
        label_smoothing: Epsilon for label smoothing.
        pos_weight: Positive-class weight for BCE loss.
    """
    if dataset_type == "binary" or num_classes == 1:
        return BinarySegLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)
    return MultiClassSegLoss(
        ce_weight=0.5, dice_weight=0.5,
        num_classes=num_classes, ignore_index=ignore_index,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(
    dataloader: DataLoader, num_classes: int,
    ignore_index: int = 255,
    device: torch.device = torch.device("cpu"),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training data.

    weight[c] = median_freq / (freq[c] + eps), normalised to mean = 1.
    """
    print("[Class Weights] Scanning training set for class frequencies ...")
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, masks in tqdm(dataloader, desc="[Class Weights]", unit="batch", leave=False):
        masks_flat = masks.flatten()
        valid = (masks_flat != ignore_index)
        pixels = masks_flat[valid].long()
        class_counts.scatter_add_(0, pixels, torch.ones_like(pixels, dtype=torch.float32))

    total_pixels = class_counts.sum()
    if total_pixels == 0:
        print("[Class Weights] No valid pixels found — using uniform weights.")
        return torch.ones(num_classes, device=device)

    class_freqs = class_counts / total_pixels
    median_freq = class_freqs[class_freqs > 0].median()
    weights = median_freq / (class_freqs + eps)
    weights[class_freqs == 0] = 0.0

    active = class_freqs > 0
    if active.any():
        weights[active] = weights[active] / weights[active].mean()
    weights = torch.clamp(weights, min=0.0, max=10.0)

    for c in range(num_classes):
        print(f"  Class {c:2d}: {class_freqs[c].item()*100:5.2f}%  "
              f"weight={weights[c].item():.4f}")
    return weights.to(device)


def compute_pos_weight(
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Compute pos_weight = neg_pixels / pos_pixels for BCEWithLogitsLoss."""
    print("[Pos Weight] Scanning training set ...")
    total_pos, total_neg = 0, 0
    for _, masks in tqdm(dataloader, desc="[Pos Weight]", unit="batch", leave=False):
        masks_flat = masks.flatten().long()
        total_pos += (masks_flat == 1).sum().item()
        total_neg += (masks_flat == 0).sum().item()

    if total_pos == 0:
        print("[Pos Weight] Zero positive pixels — using pos_weight=1.0")
        return torch.tensor([1.0], device=device)

    pw = min(total_neg / total_pos, 100.0)
    print(f"[Pos Weight] pos={total_pos:,}  neg={total_neg:,}  →  pos_weight={pw:.2f}")
    return torch.tensor([pw], device=device)


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of all trainable parameters and updates them
    with decay *after* each optimizer step.  The shadow weights are applied
    during validation to produce a more stable evaluation.

    Args:
        model: The training model.
        decay: EMA decay factor (default 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone().detach()

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self) -> None:
        if not self.backup:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


# ---------------------------------------------------------------------------
# Polynomial LR scheduler
# ---------------------------------------------------------------------------

class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate decay with optional linear warmup.

    lr = initial_lr * (1 - progress)^power
    """

    def __init__(
        self, optimizer: optim.Optimizer, total_epochs: int,
        power: float = 0.9, min_lr: float = 1e-6,
        warmup_epochs: int = 0, initial_lr: Optional[float] = None,
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr or optimizer.param_groups[0]["lr"]
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = (epoch + 1) / max(1, self.warmup_epochs)
            return [self.initial_lr * alpha for _ in self.base_lrs]
        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs,
        )
        factor = (1.0 - min(progress, 1.0)) ** self.power
        return [max(self.min_lr, self.initial_lr * factor) for _ in self.base_lrs]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_miou(
    preds: torch.Tensor, targets: torch.Tensor,
    num_classes: int, ignore_index: int = 255,
) -> float:
    """Compute mean IoU for a single batch."""
    preds = preds.argmax(dim=1)
    valid_mask = (targets != ignore_index)
    ious = []
    for c in range(num_classes):
        target_c = (targets == c)
        if target_c.sum() == 0:
            continue
        pred_c = (preds == c) & valid_mask
        intersection = (pred_c & target_c).float().sum()
        union = (pred_c | target_c).float().sum()
        ious.append(((intersection + 1e-6) / (union + 1e-6)).item())
    return sum(ious) / len(ious) if ious else 0.0


def compute_miou_from_accum(
    intersection: torch.Tensor, union: torch.Tensor,
) -> float:
    """Compute mean IoU from global intersection/union accumulators."""
    ious = []
    for c in range(len(intersection)):
        if union[c] > 0:
            ious.append(((intersection[c] + 1e-6) / (union[c] + 1e-6)).item())
    return sum(ious) / len(ious) if ious else 0.0


@torch.no_grad()
def compute_dice_binary(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
) -> float:
    """Compute Dice coefficient for a binary batch."""
    preds = (torch.sigmoid(preds) > threshold).float()
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return ((2.0 * intersection + 1e-6) / (union + 1e-6)).item()


# ---------------------------------------------------------------------------
# Segmentation Trainer
# ---------------------------------------------------------------------------

class SegmentationTrainer:
    """Training harness for U-MobileViT-Net.

    Features:
        - AdamW optimiser with weight decay
        - Linear warmup + CosineAnnealingLR or PolyLR
        - Automatic mixed precision (AMP)
        - Exponential moving average (EMA) for validation
        - Gradient checkpointing for VRAM efficiency
        - Gradient accumulation for small-batch training
        - NaN auto-recovery (reload best checkpoint, reduce LR)
        - Early stopping on validation loss
        - Full history tracking for post-hoc analysis

    Args:
        model: The segmentation model (moved to *device* automatically).
        device: torch device.
        dataset_info: :class:`DatasetInfo` from :func:`tools.data.create_dataloaders`.
        save_dir: Directory for checkpoints and history.
        config: :class:`TrainingConfig` hyperparameters.
        class_weights: Pre-computed per-class weights (multi-class only).
        pos_weight: Positive-class weight for BCE (binary only).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        dataset_info,            # DatasetInfo
        save_dir: str = "./checkpoints",
        config: Optional[TrainingConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.info = dataset_info
        self.save_dir = save_dir
        self.config = config or TrainingConfig()

        os.makedirs(save_dir, exist_ok=True)

        # Loss
        self.criterion = create_loss_fn(
            self.info.type,
            self.info.num_classes,
            self.info.ignore_index,
            class_weights=class_weights,
            label_smoothing=self.config.label_smoothing,
            pos_weight=pos_weight,
        )

        # Optimiser
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler: Optional[object] = None
        # [FP32] Không dùng GradScaler — toàn bộ training chạy float32

        # EMA
        self.ema = (
            ModelEMA(model, decay=self.config.ema_decay)
            if self.config.use_ema else None
        )

        # State
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_ce": [], "train_dice": [],
            "val_loss": [], "val_ce": [], "val_dice": [], "val_metric": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.best_epoch = 0

        # NaN recovery
        self._nan_recovery_count = 0
        self._nan_recovery_max = 3
        self._set_checkpointing(model, self.config.use_checkpointing)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_nan(*tensors: torch.Tensor) -> bool:
        for t in tensors:
            if t is not None and not torch.isfinite(t).all():
                return True
        return False

    @staticmethod
    def _cleanup_memory(aggressive: bool = False) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if aggressive:
            import gc
            gc.collect()

    @staticmethod
    def _set_checkpointing(model: nn.Module, enabled: bool) -> None:
        from models.u_mobilevit_net.module import _UMobileViTLayer
        count = 0
        for module in model.modules():
            if isinstance(module, _UMobileViTLayer):
                module.use_checkpointing = enabled
                count += 1
        if count > 0:
            action = "enabled" if enabled else "disabled"
            print(f"[Checkpointing] {action} on {count} layers")

    # ------------------------------------------------------------------
    # Training / evaluation loops
    # ------------------------------------------------------------------

    def train_epoch(
        self, train_loader: DataLoader, epoch: int, epochs: int,
    ) -> Tuple[float, float, float]:
        self.model.train()
        running_loss, running_ce, running_dice = 0.0, 0.0, 0.0
        skipped = 0
        batches_processed = 0        # số batch thực sự đã qua vòng lặp
        aborted_early = False        # True nếu epoch bị cắt do NaN liên tiếp
        consecutive_nan = 0          # NaN liên tiếp → early abort epoch
        max_consecutive_nan = 20     # ~5% batches trên dataset nhỏ như camvid (367)
        accum_counter = 0
        self.optimizer.zero_grad()

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                   unit="batch", leave=False)
        for images, masks in bar:
            batches_processed += 1
            # ---------- NaN input ---------------------------------------------
            if self._has_nan(images, masks):
                skipped += 1
                continue
            images, masks = images.to(self.device), masks.to(self.device)

            # ---------- forward + NaN output (FP32) --------------------------
            outputs = self.model(images)

            if self._has_nan(outputs):
                skipped += 1
                consecutive_nan += 1
                # Xóa gradient tích lũy dở dang (nếu có) để tránh state bleed
                if accum_counter > 0:
                    self.optimizer.zero_grad()
                    accum_counter = 0
                # Nếu NaN liên tiếp quá nhiều → model đã hỏng, abort epoch
                if consecutive_nan >= max_consecutive_nan:
                    tqdm.write(
                        f"[Train] {consecutive_nan} consecutive NaN forward "
                        f"passes — model likely corrupted, aborting epoch"
                    )
                    aborted_early = True
                    break
                continue
            consecutive_nan = 0  # reset khi gặp batch tốt

            # ---------- loss NaN ----------------------------------------------
            loss, ce, dice = self.criterion(outputs, masks)
            if self._has_nan(loss):
                skipped += 1
                consecutive_nan += 1
                if accum_counter > 0:
                    self.optimizer.zero_grad()
                    accum_counter = 0
                if consecutive_nan >= max_consecutive_nan:
                    tqdm.write(
                        f"[Train] {consecutive_nan} consecutive NaN loss "
                        f"batches — aborting epoch"
                    )
                    aborted_early = True
                    break
                continue
            consecutive_nan = 0

            # ---------- valid batch: backward (FP32) -------------------------
            loss = loss / self.config.grad_accum_steps
            loss.backward()
            accum_counter += 1
            running_loss += loss.item() * self.config.grad_accum_steps
            running_ce += ce.item()
            running_dice += dice.item()
            bar.set_postfix(
                loss=f"{loss.item() * self.config.grad_accum_steps:.4f}",
                dice=f"{dice.item():.4f}",
            )

            # ---------- optimizer step ---------------------------------------
            if accum_counter >= self.config.grad_accum_steps:
                if self._grads_have_nan():
                    skipped += accum_counter
                    self.optimizer.zero_grad()
                    accum_counter = 0
                    continue

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                accum_counter = 0
                if self.ema is not None:
                    self.ema.update()

        # ---------- partial accumulation at end of epoch (FP32) --------------
        if accum_counter > 0:
            if not self._grads_have_nan():
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.update()
            else:
                tqdm.write("[Train] NaN in partial gradients — skipped")
                skipped += accum_counter
                self.optimizer.zero_grad()

        # [NaN Fix] Nếu epoch bị cắt sớm do NaN liên tiếp → báo 0.0 để
        # train() trigger NaN recovery ngay, thay vì chờ hết epoch.
        if aborted_early:
            tqdm.write(
                f"[Train] Epoch aborted early after {batches_processed}/"
                f"{len(train_loader)} batches — model corrupted, "
                f"triggering NaN recovery"
            )
            return 0.0, 0.0, 0.0

        valid_batches = batches_processed - skipped
        if skipped:
            tqdm.write(f"[Train] Skipped {skipped}/{batches_processed} NaN batches")
        if valid_batches == 0:
            tqdm.write(f"[Train] ALL {batches_processed} batches skipped (NaN)")
            return 0.0, 0.0, 0.0
        return running_loss / valid_batches, running_ce / valid_batches, running_dice / valid_batches

    def _grads_have_nan(self) -> bool:
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    @torch.no_grad()
    def eval_epoch(
        self, val_loader: DataLoader, epoch: int, epochs: int,
    ) -> Tuple[float, float, float, float]:
        if len(val_loader) == 0:
            tqdm.write("[Val] Empty DataLoader")
            return 0.0, 0.0, 0.0, 0.0

        # Apply EMA shadow for validation
        ema_applied = False
        if self.ema is not None:
            if not self._has_nan(*self.ema.shadow.values()):
                self.ema.apply_shadow()
                ema_applied = True
            else:
                tqdm.write("[Val] EMA shadow contains NaN — using raw model")

        try:
            self.model.eval()
            loss_sum = ce_sum = dice_sum = metric_sum = 0.0
            skipped = 0
            num_classes = self.info.num_classes
            ignore_index = self.info.ignore_index

            if self.info.type != "binary":
                global_intersection = torch.zeros(num_classes, device=self.device)
                global_union = torch.zeros(num_classes, device=self.device)

            bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ",
                       unit="batch", leave=False)
            for images, masks in bar:
                if self._has_nan(images, masks):
                    skipped += 1
                    continue
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)  # [FP32] Không dùng autocast

                if self._has_nan(outputs):
                    skipped += 1
                    continue

                loss, ce, dice = self.criterion(outputs, masks)
                if self._has_nan(loss):
                    skipped += 1
                    continue

                loss_sum += loss.item()
                ce_sum += ce.item()
                dice_sum += dice.item()

                if self.info.type == "binary":
                    metric_sum += compute_dice_binary(outputs, masks)
                else:
                    preds = outputs.argmax(dim=1)
                    valid_mask = (masks != ignore_index)
                    for c in range(num_classes):
                        target_c = (masks == c)
                        pred_c = (preds == c) & valid_mask
                        global_intersection[c] += (pred_c & target_c).float().sum()
                        global_union[c] += (pred_c | target_c).float().sum()

                del outputs, loss, ce, dice, images, masks

            n = len(val_loader) - skipped
            if skipped:
                tqdm.write(f"[Val] Skipped {skipped}/{len(val_loader)} NaN batches")
            if n == 0:
                tqdm.write(f"[Val] ALL {len(val_loader)} batches skipped (NaN)")
                return 0.0, 0.0, 0.0, 0.0

            if self.info.type != "binary":
                metric = compute_miou_from_accum(global_intersection, global_union)
            else:
                metric = metric_sum / n

            return loss_sum / n, ce_sum / n, dice_sum / n, metric
        finally:
            if self.ema is not None and ema_applied:
                self.ema.restore()
                # [OOM Fix] Giải phóng backup + clear cache ngay sau validation
                # để tránh phân mảnh tích lũy qua nhiều epoch (đặc biệt promax).
                self._cleanup_memory(aggressive=True)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 300,
        patience: int = 20,
        min_epochs: int = 30,
        save_prefix: str = "best",
    ) -> Dict[str, List[float]]:
        """Run the full training loop with early stopping.

        Returns:
            History dict with keys: train_loss, train_ce, train_dice,
            val_loss, val_ce, val_dice, val_metric, lr.
        """
        metric_name = "Dice" if self.info.type == "binary" else "mIoU"
        best_metric = 0.0

        # Setup scheduler
        if self.config.scheduler_type == "poly":
            self.scheduler = PolyLR(
                self.optimizer,
                total_epochs=epochs,
                power=self.config.poly_power,
                min_lr=self.config.lr * self.config.min_lr_ratio,
                warmup_epochs=self.config.warmup_epochs,
                initial_lr=self.config.lr,
            )
        else:
            cosine_epochs = epochs - self.config.warmup_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cosine_epochs,
                eta_min=self.config.lr * self.config.min_lr_ratio,
            )

        for epoch in range(epochs):
            t0 = time.time()

            # -- LR schedule -----------------------------------------------
            if self.config.scheduler_type == "poly":
                self.scheduler.step()
            elif epoch < self.config.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.config.lr * warmup_factor
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            # [OOM Fix] Cleanup đầu epoch — giải phóng fragment từ epoch trước
            self._cleanup_memory(aggressive=True)

            # -- Train -----------------------------------------------------
            train_loss, train_ce, train_dice = self.train_epoch(
                train_loader, epoch, epochs,
            )

            # -- NaN recovery ----------------------------------------------
            if self.config.nan_recovery and train_loss == 0.0:
                if self._nan_recovery_count < self._nan_recovery_max:
                    self._nan_recovery_count += 1
                    tqdm.write(
                        f"[NaN Recovery #{self._nan_recovery_count}] "
                        f"All batches NaN — reloading best checkpoint, "
                        f"reducing LR, disabling checkpointing ..."
                    )
                    best_path = os.path.join(self.save_dir, f"{save_prefix}_model.pth")
                    if os.path.exists(best_path):
                        ckpt = torch.load(best_path, map_location=self.device,
                                         weights_only=True)
                        self.model.load_state_dict(ckpt["model_state_dict"])
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        new_lr = self.optimizer.param_groups[0]["lr"] / 10.0
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = new_lr
                        self._set_checkpointing(self.model, False)
                        self.config.use_checkpointing = False
                        tqdm.write(
                            f"[NaN Recovery] Restored epoch {ckpt['epoch']}, "
                            f"new LR={new_lr:.2e}, checkpointing=OFF"
                        )
                        self._cleanup_memory()
                        continue
                    else:
                        tqdm.write("[NaN Recovery] No best checkpoint found!")
                else:
                    tqdm.write(
                        f"[NaN Recovery] Max attempts ({self._nan_recovery_max}) "
                        f"exceeded — stopping."
                    )
                    break

            # [OOM Fix] Cleanup sau train, trước validation — giải phóng grad + activation
            self._cleanup_memory(aggressive=True)

            # -- Validate --------------------------------------------------
            val_loss, val_ce, val_dice, val_metric = self.eval_epoch(
                val_loader, epoch, epochs,
            )

            # -- Log -------------------------------------------------------
            elapsed = time.time() - t0
            if self.config.scheduler_type == "poly":
                phase = "[PolyLR]"
            elif epoch < self.config.warmup_epochs:
                phase = "[Warmup]"
            else:
                phase = "[Cosine]"
            print(
                f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} {phase} | "
                f"Time: {elapsed:.0f}s | "
                f"Train: loss={train_loss:.4f} ce={train_ce:.4f} "
                f"dice={train_dice:.4f} | "
                f"Val: loss={val_loss:.4f} ce={val_ce:.4f} "
                f"dice={val_dice:.4f} {metric_name}={val_metric:.4f}"
            )

            # -- Record history --------------------------------------------
            self.history["train_loss"].append(train_loss)
            self.history["train_ce"].append(train_ce)
            self.history["train_dice"].append(train_dice)
            self.history["val_loss"].append(val_loss)
            self.history["val_ce"].append(val_ce)
            self.history["val_dice"].append(val_dice)
            self.history["val_metric"].append(val_metric)
            self.history["lr"].append(current_lr)

            # -- Checkpoint ------------------------------------------------
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss, "val_metric": val_metric,
                "history": self.history,
            }

            if val_metric > best_metric:
                best_metric = val_metric
                torch.save(ckpt, os.path.join(self.save_dir, f"{save_prefix}_model.pth"))
                print(f"  Best model ({metric_name}={best_metric:.4f})")
                self.best_epoch = epoch + 1

            last_ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "val_loss": val_loss, "val_metric": val_metric,
                "history": self.history,
            }
            torch.save(last_ckpt, os.path.join(self.save_dir, "last_model.pth"))
            del ckpt, last_ckpt

            # [OOM Fix] Cleanup sau checkpoint save MỖI epoch — state_dict clone
            # khi torch.save có thể để lại memory fragment lớn.
            self._cleanup_memory(aggressive=True)

            # -- Early stopping --------------------------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            elif (epoch + 1) > min_epochs:
                self.early_stop_counter += 1
                print(f"[Early Stop] {self.early_stop_counter}/{patience}")
                if self.early_stop_counter >= patience:
                    print(f"==== Early stopping at epoch {epoch+1} ====")
                    break

            print()

        # Save history
        with open(os.path.join(self.save_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best {metric_name}: {best_metric:.4f} "
              f"(epoch {self.best_epoch})")
        return self.history
