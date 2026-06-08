import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, num_classes: int) -> dict:
        self.model.eval()
        total_correct = total_samples = total_loss = 0.0
        all_preds, all_targets = [], []
        criterion = nn.CrossEntropyLoss()
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "correct": total_correct, "total": total_samples,
            "predictions": np.array(all_preds), "targets": np.array(all_targets),
        }

    def confusion_matrix(self, preds, targets, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(targets, preds):
            cm[t, p] += 1
        return cm

    def report(self, metrics: dict, num_classes: int) -> str:
        lines = ["# Evaluation Report", "",
                 "| Metric | Value |", "|--------|-------|",
                 f"| Loss | {metrics['loss']:.4f} |",
                 f"| Accuracy | {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']}) |", ""]
        cm = self.confusion_matrix(metrics["predictions"], metrics["targets"], num_classes)
        lines.append("## Confusion Matrix\n```")
        for i in range(num_classes):
            lines.append(f"Class {i}: " + " ".join(f"{cm[i,j]:5d}" for j in range(num_classes)))
        lines.append("```")
        return "\n".join(lines)
