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
        outputs = model(images_gpu)  # [FP32] Không dùng autocast
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

    Produces **4 separate figures**, each displayed individually:
    1. Combined Loss (train + val) with best-val annotation
    2. Cross-Entropy / BCE Loss (train + val)
    3. Dice Loss (train + val)
    4. Validation metric with overlaid learning rate and best-metric annotation

    If *save_path* is provided, it is used as a stem and ``_1`` … ``_4``
    suffixes are appended (e.g. ``curves_1.pdf``).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    train_color = TOL_PALETTE[0]      # blue
    val_color = TOL_PALETTE[3]        # red
    lr_color = TOL_PALETTE[5]         # cyan

    def _maybe_save(fig, suffix: str) -> None:
        if save_path:
            stem = Path(save_path)
            out = stem.parent / f"{stem.stem}_{suffix}{stem.suffix}"
            fig.savefig(str(out))
        fig.show()

    # ---- Figure 1: Combined Loss -----------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(epochs, history["train_loss"], color=train_color, label="Train", lw=2.5)
    ax1.plot(epochs, history["val_loss"], color=val_color, label="Validation", lw=2.5)
    # Annotate best validation loss
    best_val_idx = int(np.argmin(history["val_loss"]))
    best_val_loss = history["val_loss"][best_val_idx]
    ax1.annotate(
        f"Best: {best_val_loss:.4f}",
        xy=(best_val_idx + 1, best_val_loss),
        xytext=(best_val_idx + 1 + len(epochs) * 0.08, best_val_loss * 1.15),
        arrowprops=dict(arrowstyle="->", color=val_color, lw=1.5),
        fontsize=10, color=val_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=val_color, alpha=0.8),
    )
    ax1.set_title("Combined Loss (CE + Dice)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, ls="--", alpha=0.6)
    fig1.tight_layout()
    _maybe_save(fig1, "1")

    # ---- Figure 2: Cross-Entropy / BCE Loss ------------------------------
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    ax2.plot(epochs, history["train_ce"], color=train_color, label="Train", lw=2.5)
    ax2.plot(epochs, history["val_ce"], color=val_color, label="Validation", lw=2.5)
    best_val_ce = min(history["val_ce"])
    best_val_ce_idx = int(np.argmin(history["val_ce"]))
    ax2.annotate(
        f"Best: {best_val_ce:.4f}",
        xy=(best_val_ce_idx + 1, best_val_ce),
        xytext=(best_val_ce_idx + 1 + len(epochs) * 0.08, best_val_ce * 1.2),
        arrowprops=dict(arrowstyle="->", color=val_color, lw=1.5),
        fontsize=10, color=val_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=val_color, alpha=0.8),
    )
    ax2.set_title("Cross-Entropy / BCE Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, ls="--", alpha=0.6)
    fig2.tight_layout()
    _maybe_save(fig2, "2")

    # ---- Figure 3: Dice Loss ---------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    ax3.plot(epochs, history["train_dice"], color=train_color, label="Train", lw=2.5)
    ax3.plot(epochs, history["val_dice"], color=val_color, label="Validation", lw=2.5)
    best_val_dice = min(history["val_dice"])
    best_val_dice_idx = int(np.argmin(history["val_dice"]))
    ax3.annotate(
        f"Best: {best_val_dice:.4f}",
        xy=(best_val_dice_idx + 1, best_val_dice),
        xytext=(best_val_dice_idx + 1 + len(epochs) * 0.08, best_val_dice * 1.2),
        arrowprops=dict(arrowstyle="->", color=val_color, lw=1.5),
        fontsize=10, color=val_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=val_color, alpha=0.8),
    )
    ax3.set_title("Dice Loss", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Loss", fontsize=12)
    ax3.legend(fontsize=11, loc="upper right")
    ax3.grid(True, ls="--", alpha=0.6)
    fig3.tight_layout()
    _maybe_save(fig3, "3")

    # ---- Figure 4: Validation Metric + LR --------------------------------
    fig4, ax4 = plt.subplots(figsize=(11, 6))
    ax4.plot(epochs, history["val_metric"], color=TOL_PALETTE[1], label=metric_name, lw=2.5)
    best_metric_idx = int(np.argmax(history["val_metric"]))
    best_metric = history["val_metric"][best_metric_idx]
    ax4.annotate(
        f"Best {metric_name}: {best_metric:.4f}\n(Epoch {best_metric_idx + 1})",
        xy=(best_metric_idx + 1, best_metric),
        xytext=(best_metric_idx + 1 - len(epochs) * 0.12, best_metric * 0.92),
        arrowprops=dict(arrowstyle="->", color=TOL_PALETTE[1], lw=1.5),
        fontsize=10, color=TOL_PALETTE[1], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=TOL_PALETTE[1], alpha=0.9),
    )
    ax4.set_title(f"Validation {metric_name}", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel(metric_name, fontsize=12, color=TOL_PALETTE[1])
    ax4.tick_params(axis="y", labelcolor=TOL_PALETTE[1])
    ax4.legend([metric_name], loc="upper left", fontsize=11)
    ax4.grid(True, ls="--", alpha=0.6)

    if history.get("lr"):
        ax_lr = ax4.twinx()
        ax_lr.plot(epochs, history["lr"], color=lr_color, ls="--", lw=1.5, alpha=0.6)
        ax_lr.set_ylabel("Learning Rate", fontsize=12, color=lr_color)
        ax_lr.tick_params(axis="y", labelcolor=lr_color)
        ax_lr.set_yscale("log")
        ax_lr.legend(["LR"], loc="lower right", fontsize=10)

    fig4.tight_layout()
    _maybe_save(fig4, "4")


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
    """Plot a single, publication-quality confusion matrix with bold numbers.

    Produces one large matrix showing **both** raw counts and normalised
    recall-percentage values in every cell.  Diagonal cells (correct
    predictions) are clearly separated from off-diagonal errors.

    Args:
        cm: Raw count confusion matrix (C × C).
        cm_norm: Row-normalised matrix (each row sums to 1).
        class_names: Class labels.
    """
    n = len(class_names)
    cell_size = max(1.4, min(2.2, 18.0 / max(n, 1)))
    figsize = (n * cell_size + 2.5, n * cell_size + 1.8)
    fig, ax = plt.subplots(figsize=figsize)

    ann_font_main = max(9, min(14, 110 / max(n, 1)))
    ann_font_sub = max(6, min(10, 90 / max(n, 1)))

    # Clip to [0,1] for display
    cm_norm_clipped = np.clip(cm_norm, 0, 1)
    im = ax.imshow(cm_norm_clipped, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Draw grid lines separating every cell
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", lw=2.5)
        ax.axvline(i - 0.5, color="white", lw=2.5)

    # Diagonal emphasis — draw a subtle border around diagonal cells
    for i in range(n):
        rect = plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor=TOL_PALETTE[7], lw=3, zorder=10,
        )
        ax.add_patch(rect)

    # Annotate every cell with BOTH count and percentage
    for i in range(n):
        for j in range(n):
            count_val = int(cm[i, j])
            pct_val = cm_norm[i, j]

            if i == j:
                # Diagonal: correct predictions in bold
                main_color = "white" if pct_val > 0.5 else "black"
                sub_color = "white" if pct_val > 0.5 else "#444444"
                weight = "bold"
            else:
                main_color = "white" if pct_val > 0.65 else "black"
                sub_color = "white" if pct_val > 0.65 else "#555555"
                weight = "normal"

            # Main text: count (large, bold)
            count_str = f"{count_val:,}" if count_val < 10000 else f"{count_val/1000:.0f}k"
            ax.text(j, i - 0.15, count_str,
                    ha="center", va="center",
                    fontsize=ann_font_main, color=main_color,
                    fontweight=weight)

            # Sub text: percentage (smaller, below count)
            ax.text(j, i + 0.2, f"{pct_val * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=ann_font_sub, color=sub_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45 if n > 8 else 30,
                       ha="right", fontsize=max(8, min(11, 60 / max(n, 1))))
    ax.set_yticklabels(class_names, fontsize=max(8, min(11, 60 / max(n, 1))))

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Compute overall accuracy
    accuracy = np.trace(cm) / max(cm.sum(), 1) * 100
    title = (
        f"Confusion Matrix — "
        f"Accuracy: {accuracy:.1f}%  |  "
        f"mIoU: {np.mean([cm_norm[i,i] / (2 - cm_norm[i,i] + 1e-8) for i in range(n)]):.3f}"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

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
