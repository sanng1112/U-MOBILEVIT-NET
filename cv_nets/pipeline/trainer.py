from __future__ import annotations
import json, os
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from cv_nets.pipeline.config import TrainingConfig, DatasetConfig


class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.data.clone().detach() for k, v in model.named_parameters() if v.requires_grad}
        self.backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply(self, model: nn.Module):
        self.backup = {n: v.data.clone() for n, v in model.named_parameters() if v.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data = self.shadow[n].clone()

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n].clone()
        self.backup = {}



class UnifiedTrainer:
    def __init__(self, model: nn.Module, train_cfg: TrainingConfig,
                 dataset_cfg: DatasetConfig, device: Union[str, torch.device] = "auto"):
        self.model = model
        self.train_cfg = train_cfg
        self.dataset_cfg = dataset_cfg
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

        opt = train_cfg.optimizer
        if opt == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        elif opt == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        elif opt == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=train_cfg.lr, momentum=0.9, weight_decay=train_cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.ema = _EMA(model, train_cfg.ema_decay) if train_cfg.use_ema else None
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

    def _train_epoch(self, loader: DataLoader, epoch: int, total: int):
        self.model.train()
        tl = tc = ts = 0.0
        bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total} [Train]", unit="batch", leave=False)
        for inputs, targets in bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            if self.train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.optimizer.step()
            if self.ema:
                self.ema.update(self.model)
            _, preds = torch.max(outputs, 1)
            tl += loss.item() * inputs.size(0)
            tc += (preds == targets).sum().item()
            ts += inputs.size(0)
            bar.set_postfix(loss=loss.item(), acc=tc/ts)
        return tl/ts, tc/ts

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, epoch: int, total: int):
        if self.ema:
            self.ema.apply(self.model)
        self.model.eval()
        tl = tc = ts = 0.0
        bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total} [Val]", unit="batch", leave=False)
        for inputs, targets in bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
            tl += loss.item() * inputs.size(0)
            tc += (preds == targets).sum().item()
            ts += inputs.size(0)
        if self.ema:
            self.ema.restore(self.model)
        return tl/ts, tc/ts

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        total = self.train_cfg.epochs
        os.makedirs(self.train_cfg.save_dir, exist_ok=True)
        if self.train_cfg.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total - self.train_cfg.warmup_epochs)

        for epoch in range(total):
            if epoch < self.train_cfg.warmup_epochs:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.train_cfg.lr * (epoch+1) / max(1, self.train_cfg.warmup_epochs)
            train_loss, train_acc = self._train_epoch(train_loader, epoch, total)
            if self.scheduler and epoch >= self.train_cfg.warmup_epochs:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            val_loss, val_acc = self._eval_epoch(val_loader or train_loader, epoch, total)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)
            print(f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({"epoch": epoch+1, "model_state_dict": self.model.state_dict(),
                            "val_acc": val_acc, "history": self.history},
                           os.path.join(self.train_cfg.save_dir, "best_model.pth"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            elif epoch+1 > self.train_cfg.min_epochs:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.train_cfg.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            print()
        return self.history
