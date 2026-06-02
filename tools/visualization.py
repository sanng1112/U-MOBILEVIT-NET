"""
Publication-quality visualisation for U-MobileViT-Net experiments.

All plotting functions use a consistent matplotlib style (serif fonts,
Paul Tol colourblind-safe palette, 300 DPI saved figures) and accept an
optional ``save_path`` argument for writing publication-ready figures to
disk.

Usage:
    from tools.visualization import configure_paper_style, plot_training_curves

    configure_paper_style()
    plot_training_curves(history, metric_name="mIoU", save_path="figures/loss.pdf")
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tools.data import denormalize, label_to_color


# ---------------------------------------------------------------------------
# Paul Tol colourblind-safe palette (8 colours)
# ---------------------------------------------------------------------------

TOL_PALETTE = [
    "#4477AA",  # blue
    "#228833",  # green
    "#CCBB44",  # yellow
    "#EE6677",  # red
    "#AA3377",  # purple
    "#66CCEE",  # cyan
    "#BBBBBB",  # grey
    "#332288",  # indigo
]

VARIANT_COLORS = {
    "nano": TOL_PALETTE[0],
    "base": TOL_PALETTE[1],
    "pro": TOL_PALETTE[2],
    "promax": TOL_PALETTE[3],
}

# Estimated parameter counts (for scatter / table annotations)
MODEL_PARAMS_ESTIMATE = {
    "nano": 0.3,
    "base": 0.6,
    "pro": 2.0,
    "promax": 7.0,
}


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def configure_paper_style() -> None:
    """Apply consistent publication-ready matplotlib rcParams."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.prop_cycle": plt.cycler(color=TOL_PALETTE),
    })


# ---------------------------------------------------------------------------
# Data exploration plots
# ---------------------------------------------------------------------------

def show_dataset_samples(
    dataloader: DataLoader,
    info,                          # DatasetInfo
    num_samples: int = 8,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    save_path: Optional[str] = None,
) -> None:
    """Display a grid of training samples: image, ground-truth mask, colour overlay."""
    palette = info.palette
    ignore_index = info.ignore_index
    n_rows = min(4, max(1, num_samples // 2))
    n_cols = 4 if palette is not None else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    plt.subplots_adjust(wspace=0.05, hspace=0.3)

    shown = 0
    for images, masks in dataloader:
        for i in range(images.size(0)):
            if shown >= num_samples:
                break
            row = shown // 2
            img = denormalize(images[i], mean, std)
            mask_np = masks[i].numpy()

            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Sample {shown + 1}")
            axes[row, 0].axis("off")

            if palette is not None:
                axes[row, 1].imshow(label_to_color(mask_np, palette, ignore_index))
                axes[row, 1].set_title("Ground Truth")
                axes[row, 1].axis("off")
                # overlay
                overlay = img.copy()
                alpha = 0.5
                for c in range(palette.shape[0]):
                    overlay[mask_np == c] = (
                        (1 - alpha) * overlay[mask_np == c]
                        + alpha * palette[c] / 255.0
                    )
                axes[row, 2].imshow(overlay)
                axes[row, 2].set_title("Overlay")
                axes[row, 2].axis("off")
                # blank remaining columns
                for c in range(3, n_cols):
                    axes[row, c].axis("off")
            else:
                axes[row, 1].imshow(mask_np, cmap="gray")
                axes[row, 1].set_title("Ground Truth")
                axes[row, 1].axis("off")
                for c in range(2, n_cols):
                    axes[row, c].axis("off")

            shown += 1
        if shown >= num_samples:
            break

    title = f"Dataset Samples — {info.name} ({info.num_classes} class{'es' if info.num_classes > 1 else ''})"
    fig.suptitle(title, fontsize=14, y=1.01)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def show_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    info,                          # DatasetInfo
    num_samples: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """Display model predictions alongside ground truth for qualitative evaluation."""
    model.eval()
    palette = info.palette
    ignore_index = info.ignore_index
    is_binary = info.type == "binary"
    n_cols = 4 if palette is not None else 3
    n_rows = min(4, max(1, num_samples // 2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    plt.subplots_adjust(wspace=0.05, hspace=0.3)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    shown = 0

    for images, masks in dataloader:
        images_gpu = images.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(images_gpu)
        images, masks = images.cpu(), masks.cpu()

        for i in range(images.size(0)):
            if shown >= num_samples:
                break
            row = shown // 2
            img = denormalize(images[i], mean, std)
            mask_np = masks[i].numpy()

            if is_binary:
                preds = (torch.sigmoid(outputs[i]) > 0.5).float().squeeze().cpu().numpy()
            else:
                preds = outputs[i].argmax(dim=0).cpu().numpy()

            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Sample {shown + 1}")
            axes[row, 0].axis("off")

            if palette is not None:
                axes[row, 1].imshow(label_to_color(mask_np, palette, ignore_index))
            else:
                axes[row, 1].imshow(mask_np, cmap="gray")
            axes[row, 1].set_title("Ground Truth")
            axes[row, 1].axis("off")

            if palette is not None:
                axes[row, 2].imshow(label_to_color(preds, palette, ignore_index))
            else:
                axes[row, 2].imshow(preds, cmap="gray")
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis("off")

            if palette is not None and n_cols >= 4:
                overlay = img.copy()
                alpha = 0.5
                for c in range(palette.shape[0]):
                    overlay[preds == c] = (
                        (1 - alpha) * overlay[preds == c]
                        + alpha * palette[c] / 255.0
                    )
                axes[row, 3].imshow(overlay)
                axes[row, 3].set_title("Prediction Overlay")
                axes[row, 3].axis("off")

            shown += 1
        if shown >= num_samples:
            break

    fig.suptitle(f"Predictions — {info.name}", fontsize=14, y=1.01)
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    metric_name: str = "mIoU",
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and metric curves from a training history dict.

    Produces a 2 × 2 grid: combined loss, CE/BCE loss, Dice loss, and
    validation metric with overlaid learning rate.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Combined loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", lw=2)
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation", lw=2)
    axes[0, 0].set_title("Combined Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, ls="--", alpha=0.7)

    # CE / BCE
    axes[0, 1].plot(epochs, history["train_ce"], label="Train", lw=2)
    axes[0, 1].plot(epochs, history["val_ce"], label="Validation", lw=2)
    axes[0, 1].set_title("Cross-Entropy / BCE Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, ls="--", alpha=0.7)

    # Dice
    axes[1, 0].plot(epochs, history["train_dice"], label="Train", lw=2)
    axes[1, 0].plot(epochs, history["val_dice"], label="Validation", lw=2)
    axes[1, 0].set_title("Dice Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, ls="--", alpha=0.7)

    # Validation metric
    axes[1, 1].plot(epochs, history["val_metric"], color=TOL_PALETTE[1], label=metric_name, lw=2)
    axes[1, 1].set_title(f"Validation {metric_name}")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel(metric_name)
    axes[1, 1].legend(loc="upper left")
    axes[1, 1].grid(True, ls="--", alpha=0.7)

    if history.get("lr"):
        ax_lr = axes[1, 1].twinx()
        ax_lr.plot(epochs, history["lr"], color=TOL_PALETTE[5], ls="--", lw=1, alpha=0.5)
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.legend(["LR"], loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Convergence comparison (multi-variant overlay)
# ---------------------------------------------------------------------------

def plot_convergence_comparison(
    results: Dict[str, Dict[str, List[float]]],
    dataset: str,
    metric_name: str = "mIoU",
    save_path: Optional[str] = None,
) -> None:
    """Overlay validation curves for multiple variants on one dataset.

    Args:
        results: ``{variant_name: history_dict}`` mapping.
        dataset: Dataset name for the plot title.
        metric_name: Metric label.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, history in results.items():
        epochs = range(1, len(history["val_metric"]) + 1)
        color = VARIANT_COLORS.get(variant, TOL_PALETTE[0])
        ax.plot(epochs, history["val_metric"], color=color, lw=2, label=variant.upper())

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Validation {metric_name} — {dataset}")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Per-class analysis
# ---------------------------------------------------------------------------

def plot_per_class_iou(
    class_names: List[str],
    per_class_iou: List[float],
    title: str = "Per-Class IoU",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of per-class IoU with a mean reference line.

    Args:
        class_names: List of class name strings.
        per_class_iou: IoU value (0--1) for each class.
    """
    n = len(class_names)
    mean_iou = np.mean(per_class_iou)
    colors = [
        TOL_PALETTE[1] if v >= mean_iou else TOL_PALETTE[3]
        for v in per_class_iou
    ]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5))
    bars = ax.bar(range(n), per_class_iou, color=colors, edgecolor="white", lw=0.5)
    ax.axhline(mean_iou, color=TOL_PALETTE[0], ls="--", lw=1.5,
               label=f"Mean = {mean_iou:.4f}")

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45 if n > 8 else 0, ha="right" if n > 8 else "center")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.5)

    # Annotate values above bars
    for bar, val in zip(bars, per_class_iou):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    cm_norm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot raw and row-normalised confusion matrices side by side.

    Args:
        cm: Raw count confusion matrix (C × C).
        cm_norm: Row-normalised matrix (each row sums to 1).
        class_names: Class labels.
    """
    n = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(max(24, n * 1.5), max(10, n * 0.6)))

    im0 = axes[0].imshow(cm, cmap="Blues", aspect="auto")
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(class_names, rotation=90, fontsize=7)
    axes[0].set_yticklabels(class_names, fontsize=7)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(cm_norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Normalised Confusion Matrix (Recall)")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(class_names, rotation=90, fontsize=7)
    axes[1].set_yticklabels(class_names, fontsize=7)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_error_distribution(
    per_image_error: List[float],
    per_image_iou: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Histograms of per-image pixel error rate and per-image mIoU.

    Args:
        per_image_error: List of per-image error rates (0--1).
        per_image_iou: List of per-image mIoU values (0--1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(per_image_error, bins=40, color=TOL_PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(per_image_error), color=TOL_PALETTE[3], ls="--", lw=2,
                    label=f"Mean = {np.mean(per_image_error):.4f}")
    axes[0].axvline(np.median(per_image_error), color=TOL_PALETTE[2], ls="-.", lw=2,
                    label=f"Median = {np.median(per_image_error):.4f}")
    axes[0].set_xlabel("Pixel Error Rate")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-Image Pixel Error Rate Distribution")
    axes[0].legend()

    axes[1].hist(per_image_iou, bins=40, color=TOL_PALETTE[1], edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(per_image_iou), color=TOL_PALETTE[3], ls="--", lw=2,
                    label=f"Mean = {np.mean(per_image_iou):.4f}")
    axes[1].axvline(np.median(per_image_iou), color=TOL_PALETTE[2], ls="-.", lw=2,
                    label=f"Median = {np.median(per_image_iou):.4f}")
    axes[1].set_xlabel("Per-Image mIoU")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-Image mIoU Distribution")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Parameter-accuracy trade-off
# ---------------------------------------------------------------------------

def plot_params_vs_accuracy(
    results: Dict[str, Dict],            # {variant: {dataset: metric}}
    model_params: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Scatter plot of model parameters versus segmentation accuracy.

    Args:
        results: Nested dict ``{variant: {dataset: best_metric}}``.
        model_params: ``{variant: M_params}`` override.  Defaults to
            :data:`MODEL_PARAMS_ESTIMATE`.
    """
    if model_params is None:
        model_params = MODEL_PARAMS_ESTIMATE

    fig, ax = plt.subplots(figsize=(10, 7))
    datasets = sorted({ds for v in results for ds in results[v]})
    markers = ["o", "s", "D", "^", "v", "<"]

    for di, ds in enumerate(datasets):
        xs, ys, labels = [], [], []
        for variant in results:
            if ds in results[variant]:
                xs.append(model_params.get(variant, 0))
                ys.append(results[variant][ds])
                labels.append(variant.upper())
        color = TOL_PALETTE[di % len(TOL_PALETTE)]
        ax.scatter(xs, ys, c=color, marker=markers[di % len(markers)],
                   s=100, edgecolors="white", linewidth=0.5, label=ds, zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.8)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Best Validation Metric")
    ax.set_title("Parameter–Accuracy Trade-off")
    ax.set_xscale("log")
    ax.legend(loc="lower right")
    ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# LaTeX results table
# ---------------------------------------------------------------------------

def generate_results_table(
    results: Dict[str, Dict[str, float]],
    model_params: Optional[Dict[str, float]] = None,
    metric_map: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> str:
    """Generate a LaTeX-formatted results table.

    Args:
        results: ``{variant: {dataset: best_metric}}``.
        model_params: ``{variant: M_params}`` for the first column.
        metric_map: ``{dataset: metric_name}`` for column headers.
        save_path: If given, write LaTeX source to this file.

    Returns:
        LaTeX tabular source as a string.
    """
    if model_params is None:
        model_params = MODEL_PARAMS_ESTIMATE
    if metric_map is None:
        metric_map = {}

    variants = sorted(results.keys(), key=lambda v: model_params.get(v, 0))
    datasets = sorted({ds for v in results for ds in results[v]})

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    cols = "l" + "c" * (len(datasets) + 1)
    lines.append(r"  \begin{tabular}{" + cols + r"}")
    lines.append(r"    \toprule")

    # Header
    header = "    Model & Params (M) & " + " & ".join(datasets) + r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    for variant in variants:
        params = model_params.get(variant, 0)
        row = f"    {variant.upper()} & {params:.1f}"
        for ds in datasets:
            val = results[variant].get(ds, None)
            if val is not None:
                # Bold the best in each column
                col_vals = [results[v].get(ds, 0) for v in variants]
                best = max(col_vals) if col_vals else 0
                if val == best:
                    row += f" & \\textbf{{{val:.4f}}}"
                else:
                    row += f" & {val:.4f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    # Caption with metric info
    metric_notes = []
    for ds in datasets:
        mn = metric_map.get(ds, "mIoU/Dice")
        metric_notes.append(f"{ds}: {mn}")
    lines.append(
        r"  \caption{U-MobileViT-Net benchmark results across "
        + ", ".join(metric_notes) + r".}"
    )
    lines.append(r"  \label{tab:benchmark}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex)
    return latex


# ---------------------------------------------------------------------------
# Result loading helpers (shared with notebooks)
# ---------------------------------------------------------------------------

def load_training_history(path: str) -> Dict[str, List[float]]:
    """Load a ``training_history.json`` file."""
    with open(path) as f:
        return json.load(f)


def load_all_results(
    results_dir: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict]]]:
    """Scan a benchmark results directory and aggregate best metrics.

    Expects::

        results_dir/<dataset>/<variant>_<timestamp>/training_history.json

    Returns:
        ``(best_metrics, all_histories)`` where *best_metrics* is
        ``{variant: {dataset: best_metric}}`` and *all_histories* is
        ``{variant: {dataset: history_dict}}``.
    """
    root = Path(results_dir)
    best_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
    all_histories: Dict[str, Dict[str, Dict]] = defaultdict(dict)

    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for exp_dir in sorted(ds_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            parts = exp_dir.name.split("_")
            variant = parts[0] if parts else "unknown"
            history_path = exp_dir / "training_history.json"
            if not history_path.exists():
                continue
            history = load_training_history(str(history_path))
            best_idx = int(np.argmax(history["val_metric"]))
            best_metrics[variant][dataset] = float(history["val_metric"][best_idx])
            all_histories[variant][dataset] = history

    return dict(best_metrics), dict(all_histories)
