"""
Mean Teacher semi-supervised learning cho U-MobileViT-Net segmentation.

Triển khai framework Mean Teacher (Tarvainen & Valpola, 2017):
- Student model: huấn luyện với gradient trên cả labeled (supervised) và
  unlabeled (consistency) data
- Teacher model: EMA của student weights, cung cấp pseudo-targets cho
  consistency loss trên unlabeled data
- Supervised loss: Combined CE + Dice trên labeled samples
- Consistency loss: MSE giữa student và teacher softmax outputs trên
  unlabeled samples (với các augmentation khác nhau)
- Sigmoid ramp-up: consistency weight tăng dần trong những epoch đầu

Usage:
    from tools.training_semi import MeanTeacherConfig, MeanTeacherTrainer

    config = MeanTeacherConfig(lr=1e-3, labeled_ratio=0.1, consistency_weight=10.0)
    trainer = MeanTeacherTrainer(model, device, info, "checkpoints/", config)
    history = trainer.train(labeled_loader, unlabeled_loader, val_loader, epochs=300)
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

from tools.training import (
    TrainingConfig, PolyLR,
    create_loss_fn, compute_miou_from_accum, compute_dice_binary,
)


# ---------------------------------------------------------------------------
# Mean Teacher Configuration
# ---------------------------------------------------------------------------

@dataclass
class MeanTeacherConfig(TrainingConfig):
    """Cấu hình mở rộng cho Mean Teacher semi-supervised training.

    Kế thừa toàn bộ thuộc tính từ :class:`TrainingConfig` và bổ sung:

    Attributes:
        consistency_weight: Trọng số cho consistency loss (MSE giữa
            student và teacher predictions trên unlabeled data).
        teacher_ema_decay: Hệ số EMA khi cập nhật teacher từ student.
        labeled_ratio: Tỉ lệ dữ liệu huấn luyện được gán nhãn (0.0-1.0).
        consistency_rampup: Số epoch để ramp-up consistency weight (sigmoid).
    """
    consistency_weight: float = 10.0
    teacher_ema_decay: float = 0.999
    labeled_ratio: float = 0.1
    consistency_rampup: int = 50
    # Tắt EMA của student — teacher thay thế vai trò của standard EMA
    use_ema: bool = False


# ---------------------------------------------------------------------------
# Sigmoid ramp-up
# ---------------------------------------------------------------------------

def sigmoid_rampup(current: int, rampup_length: int) -> float:
    """Sigmoid ramp-up function cho consistency weight.

    Tăng dần từ 0 lên 1 theo hàm sigmoid, giúp tránh teacher chưa ổn định
    tạo ra noisy consistency targets trong giai đoạn đầu.

    Reference:
        Laine & Aila (2017) "Temporal Ensembling for Semi-Supervised Learning"

    Args:
        current: Epoch hiện tại (0-indexed).
        rampup_length: Tổng số epoch ramp-up.

    Returns:
        Giá trị ramp-up trong khoảng [0, 1].
    """
    if rampup_length == 0:
        return 1.0
    t = max(0.0, min(1.0, current / max(1, rampup_length)))
    return float(np.exp(-5.0 * (1.0 - t) ** 2))


# ---------------------------------------------------------------------------
# Mean Teacher Trainer
# ---------------------------------------------------------------------------

class MeanTeacherTrainer:
    """Training harness cho Mean Teacher semi-supervised segmentation.

    Kiến trúc:
        - **Student model**: huấn luyện với gradient trên cả supervised và
          consistency loss.
        - **Teacher model**: EMA của student, chỉ dùng để inference (không
          gradient), cung cấp pseudo-targets ổn định cho consistency loss.
        - **Supervised loss**: CE + Dice (50/50) trên labeled data.
        - **Consistency loss**: MSE giữa student và teacher softmax outputs
          trên unlabeled data với augmentation khác nhau.

    Reference:
        Tarvainen & Valpola (2017) "Mean teachers are better role models"
        https://arxiv.org/abs/1703.01780

    Args:
        student_model: Model sẽ được huấn luyện.
        device: torch device.
        dataset_info: :class:`DatasetInfo` từ data pipeline.
        save_dir: Thư mục lưu checkpoints và history.
        config: :class:`MeanTeacherConfig` hyperparameters.
        class_weights: Pre-computed per-class weights (multi-class only).
        pos_weight: Positive-class weight cho BCE (binary only).
    """

    def __init__(
        self,
        student_model: nn.Module,
        device: torch.device,
        dataset_info,            # DatasetInfo
        save_dir: str = "./checkpoints",
        config: Optional[MeanTeacherConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        self.student = student_model.to(device)
        self.device = device
        self.info = dataset_info
        self.save_dir = save_dir
        self.config = config or MeanTeacherConfig()

        os.makedirs(save_dir, exist_ok=True)

        # ── Teacher model: deep copy student, freeze toàn bộ tham số ──
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # ── Loss functions ──
        self.criterion = create_loss_fn(
            self.info.type,
            self.info.num_classes,
            self.info.ignore_index,
            class_weights=class_weights,
            label_smoothing=self.config.label_smoothing,
            pos_weight=pos_weight,
        )
        self.consistency_criterion = nn.MSELoss()

        # ── Optimiser ──
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler: Optional[object] = None

        # ── State ──
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_sup_loss": [], "train_cons_loss": [],
            "train_ce": [], "train_dice": [],
            "val_loss": [], "val_ce": [], "val_dice": [], "val_metric": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.best_epoch = 0

        # ── NaN recovery ──
        self._nan_recovery_count = 0
        self._nan_recovery_max = 3
        self._set_checkpointing(self.student, self.config.use_checkpointing)

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

    def _grads_have_nan(self) -> bool:
        for p in self.student.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return True
        return False

    def _update_teacher(self) -> None:
        """Cập nhật teacher weights bằng EMA của student weights.

        θ_teacher = α * θ_teacher + (1 - α) * θ_student
        """
        alpha = self.config.teacher_ema_decay
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher.parameters(), self.student.parameters(),
            ):
                t_param.data.mul_(alpha).add_(s_param.data, alpha=1.0 - alpha)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_student_loader: DataLoader,
        unlabeled_teacher_loader: DataLoader,
        epoch: int,
        epochs: int,
    ) -> Tuple[float, float, float, float, float]:
        """Chạy một epoch Mean Teacher training.

        Sử dụng 2 DataLoader unlabeled riêng biệt cho student và teacher
        để mỗi nhánh thấy augmentation khác nhau (đây là cơ chế cốt lõi
        của Mean Teacher — consistency under different perturbations).

        labeled_loader được cycle để đảm bảo mỗi batch unlabeled đều có
        một batch labeled tương ứng.

        Returns:
            (avg_loss, avg_sup_loss, avg_cons_loss, avg_ce, avg_dice)
        """
        self.student.train()
        self.teacher.train()  # teacher vẫn cần train mode cho BN stats (nếu có)

        running_loss = 0.0
        running_sup = 0.0
        running_cons = 0.0
        running_ce = 0.0
        running_dice = 0.0
        skipped = 0
        batches_processed = 0
        aborted_early = False
        consecutive_nan = 0
        max_consecutive_nan = 20
        accum_counter = 0
        self.optimizer.zero_grad()

        # ── Ramp-up weight ──
        rampup_weight = sigmoid_rampup(epoch, self.config.consistency_rampup)
        cons_weight = self.config.consistency_weight * rampup_weight

        # ── Zip 3 loaders: labeled (cycled) + 2 unlabeled ──
        from itertools import cycle
        labeled_iter = cycle(labeled_loader)

        n_steps = len(unlabeled_student_loader)
        bar = tqdm(
            zip(unlabeled_student_loader, unlabeled_teacher_loader, labeled_iter),
            total=n_steps,
            desc=f"Epoch {epoch+1}/{epochs} [MTrain]",
            unit="batch", leave=False,
        )
        for (uimgs_s, _), (uimgs_t, _), (limgs, lmasks) in bar:
            batches_processed += 1

            # ── NaN input check ──
            if self._has_nan(limgs, lmasks, uimgs_s, uimgs_t):
                skipped += 1
                continue

            limgs = limgs.to(self.device)
            lmasks = lmasks.to(self.device)
            uimgs_s = uimgs_s.to(self.device)
            uimgs_t = uimgs_t.to(self.device)

            # ── Student forward: labeled ──
            l_outputs = self.student(limgs)
            if self._has_nan(l_outputs):
                skipped += 1
                consecutive_nan += 1
                if accum_counter > 0:
                    self.optimizer.zero_grad()
                    accum_counter = 0
                if consecutive_nan >= max_consecutive_nan:
                    tqdm.write(
                        f"[MTrain] {consecutive_nan} consecutive NaN "
                        f"— aborting epoch"
                    )
                    aborted_early = True
                    break
                continue
            consecutive_nan = 0

            # ── Supervised loss ──
            sup_loss, ce, dice = self.criterion(l_outputs, lmasks)
            if self._has_nan(sup_loss):
                skipped += 1
                consecutive_nan += 1
                if accum_counter > 0:
                    self.optimizer.zero_grad()
                    accum_counter = 0
                if consecutive_nan >= max_consecutive_nan:
                    aborted_early = True
                    break
                continue
            consecutive_nan = 0

            # ── Student forward: unlabeled (augmentation A) ──
            u_outputs_student = self.student(uimgs_s)
            if self._has_nan(u_outputs_student):
                skipped += 1
                consecutive_nan += 1
                if accum_counter > 0:
                    self.optimizer.zero_grad()
                    accum_counter = 0
                if consecutive_nan >= max_consecutive_nan:
                    aborted_early = True
                    break
                continue

            # ── Teacher forward: unlabeled (augmentation B, no grad) ──
            with torch.no_grad():
                u_outputs_teacher = self.teacher(uimgs_t)

            if self._has_nan(u_outputs_teacher):
                skipped += 1
                consecutive_nan += 1
                if accum_counter > 0:
                    self.optimizer.zero_grad()
                    accum_counter = 0
                if consecutive_nan >= max_consecutive_nan:
                    aborted_early = True
                    break
                continue
            consecutive_nan = 0

            # ── Consistency loss: MSE(student_softmax, teacher_softmax) ──
            student_softmax = F.softmax(u_outputs_student, dim=1)
            teacher_softmax = F.softmax(u_outputs_teacher, dim=1)
            cons_loss = self.consistency_criterion(
                student_softmax, teacher_softmax,
            )

            # ── Total loss ──
            total_loss = sup_loss + cons_weight * cons_loss

            # ── Backward ──
            total_loss = total_loss / self.config.grad_accum_steps
            total_loss.backward()
            accum_counter += 1

            running_loss += total_loss.item() * self.config.grad_accum_steps
            running_sup += sup_loss.item()
            running_cons += cons_loss.item()
            running_ce += ce.item()
            running_dice += dice.item()

            bar.set_postfix(
                loss=f"{total_loss.item() * self.config.grad_accum_steps:.4f}",
                sup=f"{sup_loss.item():.4f}",
                cons=f"{cons_loss.item():.4f}",
                ramp=f"{rampup_weight:.2f}",
            )

            # ── Optimizer step ──
            if accum_counter >= self.config.grad_accum_steps:
                if self._grads_have_nan():
                    skipped += accum_counter
                    self.optimizer.zero_grad()
                    accum_counter = 0
                    continue

                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.grad_clip_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                accum_counter = 0

                # ── Cập nhật teacher bằng EMA sau mỗi optimizer step ──
                self._update_teacher()

        # ── Partial accumulation cuối epoch ──
        if accum_counter > 0:
            if not self._grads_have_nan():
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.grad_clip_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_teacher()
            else:
                tqdm.write("[MTrain] NaN in partial gradients — skipped")
                skipped += accum_counter
                self.optimizer.zero_grad()

        if aborted_early:
            tqdm.write(
                f"[MTrain] Epoch aborted early after {batches_processed}/"
                f"{n_steps} batches — triggering NaN recovery"
            )
            return 0.0, 0.0, 0.0, 0.0, 0.0

        valid_batches = batches_processed - skipped
        if skipped:
            tqdm.write(
                f"[MTrain] Skipped {skipped}/{batches_processed} NaN batches"
            )
        if valid_batches == 0:
            tqdm.write(f"[MTrain] ALL {batches_processed} batches skipped (NaN)")
            return 0.0, 0.0, 0.0, 0.0, 0.0

        return (
            running_loss / valid_batches,
            running_sup / valid_batches,
            running_cons / valid_batches,
            running_ce / valid_batches,
            running_dice / valid_batches,
        )

    # ------------------------------------------------------------------
    # Validation epoch — dùng teacher model để đánh giá
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_epoch(
        self, val_loader: DataLoader, epoch: int, epochs: int,
    ) -> Tuple[float, float, float, float]:
        """Đánh giá teacher model trên tập validation."""
        if len(val_loader) == 0:
            tqdm.write("[Val] Empty DataLoader")
            return 0.0, 0.0, 0.0, 0.0

        self.teacher.eval()
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

            outputs = self.teacher(images)  # [FP32] Dùng teacher để validate

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

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        labeled_loader: DataLoader,
        unlabeled_student_loader: DataLoader,
        unlabeled_teacher_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 300,
        patience: int = 20,
        min_epochs: int = 30,
        save_prefix: str = "best",
    ) -> Dict[str, List[float]]:
        """Chạy toàn bộ training loop với early stopping.

        Args:
            labeled_loader: DataLoader cho tập labeled (10% dữ liệu).
            unlabeled_student_loader: DataLoader unlabeled cho student
                (augmentation riêng, khác với teacher).
            unlabeled_teacher_loader: DataLoader unlabeled cho teacher
                (augmentation riêng, khác với student).
            val_loader: DataLoader cho tập validation.
            epochs: Tổng số epoch.
            patience: Số epoch chờ trước khi early stop.
            min_epochs: Số epoch tối thiểu trước khi early stop có hiệu lực.
            save_prefix: Tiền tố cho tên file checkpoint.

        Returns:
            History dict với các key: train_loss, train_sup_loss,
            train_cons_loss, train_ce, train_dice, val_loss, val_ce,
            val_dice, val_metric, lr.
        """
        metric_name = "Dice" if self.info.type == "binary" else "mIoU"
        best_metric = 0.0

        # ── Setup scheduler ──
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

        n_labeled = len(labeled_loader.dataset)
        n_unlabeled = len(unlabeled_student_loader.dataset)
        print(f"\n{'='*60}")
        print(f"  Mean Teacher Semi-Supervised Training")
        print(f"{'='*60}")
        print(f"  Labeled:           {n_labeled} samples "
              f"({n_labeled / max(1, n_labeled + n_unlabeled) * 100:.1f}%)")
        print(f"  Unlabeled:         {n_unlabeled} samples "
              f"({n_unlabeled / max(1, n_labeled + n_unlabeled) * 100:.1f}%)")
        print(f"  Consistency weight:{self.config.consistency_weight}")
        print(f"  Consistency rampup:{self.config.consistency_rampup} epochs")
        print(f"  Teacher EMA decay: {self.config.teacher_ema_decay}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            t0 = time.time()

            # ── LR schedule ──
            if self.config.scheduler_type == "poly":
                self.scheduler.step()
            elif epoch < self.config.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.config.lr * warmup_factor
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self._cleanup_memory(aggressive=True)

            # ── Train ──
            train_loss, sup_loss, cons_loss, train_ce, train_dice = self.train_epoch(
                labeled_loader, unlabeled_student_loader,
                unlabeled_teacher_loader, epoch, epochs,
            )

            # ── NaN recovery ──
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
                        self.student.load_state_dict(ckpt["model_state_dict"])
                        # Cũng restore teacher từ checkpoint
                        self.teacher = copy.deepcopy(self.student)
                        for p in self.teacher.parameters():
                            p.requires_grad = False
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        new_lr = self.optimizer.param_groups[0]["lr"] / 10.0
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = new_lr
                        self._set_checkpointing(self.student, False)
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

            self._cleanup_memory(aggressive=True)

            # ── Validate ──
            val_loss, val_ce, val_dice, val_metric = self.eval_epoch(
                val_loader, epoch, epochs,
            )

            # ── Log ──
            elapsed = time.time() - t0
            rampup = sigmoid_rampup(epoch, self.config.consistency_rampup)
            if self.config.scheduler_type == "poly":
                phase = "[PolyLR]"
            elif epoch < self.config.warmup_epochs:
                phase = "[Warmup]"
            else:
                phase = "[Cosine]"
            print(
                f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} {phase} | "
                f"Time: {elapsed:.0f}s | "
                f"Train: loss={train_loss:.4f} sup={sup_loss:.4f} "
                f"cons={cons_loss:.4f} (×{rampup:.2f}) | "
                f"Val: loss={val_loss:.4f} ce={val_ce:.4f} "
                f"dice={val_dice:.4f} {metric_name}={val_metric:.4f}"
            )

            # ── Record history ──
            self.history["train_loss"].append(train_loss)
            self.history["train_sup_loss"].append(sup_loss)
            self.history["train_cons_loss"].append(cons_loss)
            self.history["train_ce"].append(train_ce)
            self.history["train_dice"].append(train_dice)
            self.history["val_loss"].append(val_loss)
            self.history["val_ce"].append(val_ce)
            self.history["val_dice"].append(val_dice)
            self.history["val_metric"].append(val_metric)
            self.history["lr"].append(current_lr)

            # ── Checkpoint (lưu student + teacher) ──
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": self.student.state_dict(),
                "teacher_state_dict": self.teacher.state_dict(),
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
                "model_state_dict": self.student.state_dict(),
                "teacher_state_dict": self.teacher.state_dict(),
                "val_loss": val_loss, "val_metric": val_metric,
                "history": self.history,
            }
            torch.save(last_ckpt, os.path.join(self.save_dir, "last_model.pth"))
            del ckpt, last_ckpt

            self._cleanup_memory(aggressive=True)

            # ── Early stopping ──
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

        # ── Save history ──
        with open(os.path.join(self.save_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best {metric_name}: {best_metric:.4f} "
              f"(epoch {self.best_epoch})")
        return self.history
