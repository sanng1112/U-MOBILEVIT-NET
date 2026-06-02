#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         U-MobileViT-Net — Benchmark Summarizer & Comparison Plots          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Tổng hợp kết quả từ benchmark/results/ và tạo biểu đồ so sánh.

COLOR MAP nhất quán: mỗi model variant có MỘT màu cố định, áp dụng
xuyên suốt mọi biểu đồ (mọi dataset).

Cách dùng:
    python benchmark/summarize.py
    python benchmark/summarize.py --results-dir benchmark/results --output benchmark/results/summary
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ═══════════════════════════════════════════════════════════════
# CONSISTENT COLOR MAP — Áp dụng cho MỌI biểu đồ
# ═══════════════════════════════════════════════════════════════

VARIANT_COLORS = {
    'nano':   '#5DADE2',  # Xanh dương nhạt
    'base':   '#27AE60',  # Xanh lá
    'pro':    '#E67E22',  # Cam
    'promax': '#E74C3C',  # Đỏ
}

VARIANT_MARKERS = {
    'nano': 's', 'base': 'o', 'pro': '^', 'promax': 'D',
}

VARIANT_ORDER = ['nano', 'base', 'pro', 'promax']

DATASET_NAMES = {
    'camvid': 'CamVid', 'voc': 'PASCAL VOC', 'drive': 'DRIVE',
    'kvasir': 'Kvasir-SEG', 'isic': 'ISIC 2018', 'coco_leaf': 'COCO Tea Leaf',
}

# Model params estimate
VARIANT_PARAMS = {'nano': 0.3, 'base': 0.6, 'pro': 2.0, 'promax': 7.0}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_all_results(results_dir: Path) -> Dict:
    """Load tất cả kết quả benchmark.

    Returns:
        Dict[dataset][variant] = {
            'best_metric': float,  # mIoU hoặc Dice
            'best_loss': float,
            'best_epoch': int,
            'history': dict,       # training_history.json
            'exp_dir': str,
        }
    """
    all_results = defaultdict(dict)

    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith('summary'):
            continue
        dataset = ds_dir.name
        for exp_dir in sorted(ds_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            # Extract variant name (format: <variant>_<timestamp>)
            parts = exp_dir.name.split('_')
            variant = parts[0]

            history_path = exp_dir / "training_history.json"
            if not history_path.exists():
                continue

            with open(history_path) as f:
                history = json.load(f)

            if not history.get('val_metric'):
                continue

            best_idx = int(np.argmax(history['val_metric']))
            best_metric = float(history['val_metric'][best_idx])
            best_loss = float(min(history['val_loss']))
            best_epoch = best_idx + 1

            # Chỉ giữ kết quả tốt nhất nếu có nhiều lần chạy
            if variant in all_results[dataset]:
                if best_metric > all_results[dataset][variant]['best_metric']:
                    all_results[dataset][variant] = {
                        'best_metric': best_metric,
                        'best_loss': best_loss,
                        'best_epoch': best_epoch,
                        'history': history,
                        'exp_dir': str(exp_dir),
                    }
            else:
                all_results[dataset][variant] = {
                    'best_metric': best_metric,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'history': history,
                    'exp_dir': str(exp_dir),
                }

    return dict(all_results)


def compute_summary(all_results: Dict) -> Dict:
    """Tính toán summary metrics."""
    summary = {}
    for dataset, variants in all_results.items():
        summary[dataset] = {}
        for variant, data in variants.items():
            summary[dataset][variant] = {
                'best_metric': data['best_metric'],
                'best_loss': data['best_loss'],
                'best_epoch': data['best_epoch'],
                'params_M': VARIANT_PARAMS.get(variant, 0),
                'exp_dir': data['exp_dir'],
            }
    return summary


# ═══════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════

def plot_convergence_curves(all_results: Dict, output_dir: Path):
    """Vẽ đường cong hội tụ (val_metric theo epoch) cho từng dataset.

    Mỗi dataset = 1 figure, mỗi model variant = 1 đường với màu cố định.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, variants in all_results.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ds_name = DATASET_NAMES.get(dataset, dataset)
        metric_name = 'Dice Score' if dataset in ('drive', 'kvasir', 'isic') else 'mIoU'

        for variant in VARIANT_ORDER:
            if variant not in variants:
                continue
            history = variants[variant].get('history', {})
            val_metrics = history.get('val_metric', [])
            if not val_metrics:
                continue

            color = VARIANT_COLORS.get(variant, '#999999')
            marker = VARIANT_MARKERS.get(variant, 'o')

            epochs = list(range(1, len(val_metrics) + 1))
            # Plot every Nth point to avoid clutter
            step = max(1, len(epochs) // 50)
            ax.plot(epochs[::step], val_metrics[::step],
                    color=color, marker=marker, markevery=max(1, len(epochs) // 5),
                    markersize=4, linewidth=1.5, label=f'{variant} ({VARIANT_PARAMS.get(variant, 0):.1f}M)',
                    alpha=0.85)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{ds_name} — Validation {metric_name} Convergence')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

        fig.tight_layout()
        fig.savefig(output_dir / f'{dataset}_convergence.png')
        plt.close(fig)

    print(f"  → Convergence curves saved to {output_dir}")


def plot_per_dataset_comparison(all_results: Dict, output_dir: Path):
    """Vẽ bar chart so sánh best metric giữa các variants cho từng dataset.

    Mỗi dataset = 1 figure. Thanh bar dùng màu cố định của variant.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_datasets = len(all_results)
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (dataset, variants) in zip(axes, all_results.items()):
        ds_name = DATASET_NAMES.get(dataset, dataset)
        metric_name = 'Dice Score' if dataset in ('drive', 'kvasir', 'isic') else 'mIoU'

        present_variants = [v for v in VARIANT_ORDER if v in variants]
        if not present_variants:
            ax.set_title(f'{ds_name} — No data')
            continue

        values = [variants[v]['best_metric'] for v in present_variants]
        colors = [VARIANT_COLORS.get(v, '#999999') for v in present_variants]
        labels = [f'{v}\n({VARIANT_PARAMS.get(v, 0):.1f}M)' for v in present_variants]

        bars = ax.bar(range(len(present_variants)), values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(present_variants)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(f'{ds_name}')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    # Hide unused subplots
    for i in range(len(all_results), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('U-MobileViT-Net: Best Validation Metric by Dataset & Variant',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / 'per_dataset_comparison.png', bbox_inches='tight')
    plt.close(fig)

    print(f"  → Per-dataset comparison saved to {output_dir}")


def plot_cross_dataset_summary(all_results: Dict, output_dir: Path):
    """Vẽ grouped bar chart: mỗi dataset là 1 group, mỗi variant là 1 bar cùng màu."""
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(all_results.keys())
    if not datasets:
        return

    # Xác định variants có mặt trong ít nhất 1 dataset
    all_variants = set()
    for variants in all_results.values():
        all_variants.update(variants.keys())
    present_variants = [v for v in VARIANT_ORDER if v in all_variants]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(datasets))
    width = 0.8 / len(present_variants)

    for i, variant in enumerate(present_variants):
        values = []
        ds_labels = []
        for dataset in datasets:
            if variant in all_results.get(dataset, {}):
                values.append(all_results[dataset][variant]['best_metric'])
                ds_labels.append(DATASET_NAMES.get(dataset, dataset))
            else:
                values.append(0)

        color = VARIANT_COLORS.get(variant, '#999999')
        offset = (i - len(present_variants) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=f'{variant} ({VARIANT_PARAMS.get(variant, 0):.1f}M)',
                      color=color, edgecolor='white', linewidth=0.5)

        # Giá trị trên bar
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Best Metric (mIoU / Dice)')
    ax.set_title('U-MobileViT-Net: Cross-Dataset Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets], fontsize=9)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(output_dir / 'cross_dataset_summary.png')
    plt.close(fig)

    print(f"  → Cross-dataset summary saved to {output_dir}")


def plot_params_vs_accuracy(all_results: Dict, output_dir: Path):
    """Vẽ scatter plot: Params vs Best Metric cho mỗi dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset, variants in all_results.items():
        ds_name = DATASET_NAMES.get(dataset, dataset)
        xs, ys, cs = [], [], []
        for variant in VARIANT_ORDER:
            if variant not in variants:
                continue
            xs.append(VARIANT_PARAMS.get(variant, 0))
            ys.append(variants[variant]['best_metric'])
            cs.append(VARIANT_COLORS.get(variant, '#999999'))

        if xs:
            ax.scatter(xs, ys, c=cs, s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
            # Connect points with dashed line for same dataset
            sorted_pairs = sorted(zip(xs, ys))
            ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
                    '--', alpha=0.3, linewidth=1)

            # Label last point
            ax.annotate(ds_name, (xs[-1], ys[-1]), fontsize=8, alpha=0.7,
                        textcoords="offset points", xytext=(5, 0))

    # Legend for variants (color only)
    legend_elements = []
    for variant in VARIANT_ORDER:
        if any(variant in variants for variants in all_results.values()):
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=VARIANT_COLORS[variant],
                           markersize=8, label=f'{variant} ({VARIANT_PARAMS[variant]:.1f}M)')
            )
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    ax.set_xlabel('Model Parameters (Millions)')
    ax.set_ylabel('Best Metric (mIoU / Dice)')
    ax.set_title('U-MobileViT-Net: Accuracy vs Model Size')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(output_dir / 'params_vs_accuracy.png')
    plt.close(fig)

    print(f"  → Params vs Accuracy plot saved to {output_dir}")


def generate_latex_table(all_results: Dict, output_dir: Path):
    """Tạo bảng LaTeX cho paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(all_results.keys())

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{U-MobileViT-Net variants across datasets. Best mIoU/Dice reported.}")
    lines.append(r"  \label{tab:benchmark}")

    # Columns: Dataset + 4 variants
    cols = "l" + "c" * len(VARIANT_ORDER)
    lines.append(r"  \begin{tabular}{" + cols + r"}")
    lines.append(r"    \toprule")
    header = "Dataset & " + " & ".join(f"\\texttt{{{v}}} ({VARIANT_PARAMS[v]:.1f}M)" for v in VARIANT_ORDER) + r" \\"
    lines.append("    " + header)
    lines.append(r"    \midrule")

    for dataset in datasets:
        ds_name = DATASET_NAMES.get(dataset, dataset)
        row = f"    {ds_name}"
        best_val = -1
        best_var = None
        for variant in VARIANT_ORDER:
            if variant in all_results.get(dataset, {}):
                metric = all_results[dataset][variant]['best_metric']
                row += f" & {metric:.4f}"
                if metric > best_val:
                    best_val = metric
                    best_var = variant
            else:
                row += " & ---"
        # Bold best
        if best_var:
            for variant in VARIANT_ORDER:
                if variant == best_var:
                    row = row.replace(f" {best_val:.4f}", f" \\textbf{{{best_val:.4f}}}", 1)
        row += r" \\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex_path = output_dir / "results_table.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  → LaTeX table saved to {latex_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="U-MobileViT-Net Benchmark Summarizer")
    parser.add_argument('--results-dir', type=str, default='benchmark/results',
                       help='Thư mục chứa kết quả benchmark')
    parser.add_argument('--output', type=str, default=None,
                       help='Thư mục output cho biểu đồ (mặc định: <results-dir>/summary)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Không vẽ biểu đồ, chỉ tạo JSON + LaTeX')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 60)
    print("  U-MobileViT-Net Benchmark Summarizer")
    print("═" * 60)
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir:  {output_dir}")

    # Load
    print("\n[1/4] Loading results...")
    all_results = load_all_results(results_dir)

    if not all_results:
        print("  ✗ No results found!")
        return

    n_experiments = sum(len(variants) for variants in all_results.values())
    print(f"  ✓ Loaded {n_experiments} experiments across {len(all_results)} datasets")

    # Summary JSON
    print("\n[2/4] Computing summary...")
    summary = compute_summary(all_results)
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary saved to {summary_path}")

    # Print summary table
    print("\n  Best metrics:")
    for dataset, variants in all_results.items():
        ds_name = DATASET_NAMES.get(dataset, dataset)
        metric_name = 'Dice' if dataset in ('drive', 'kvasir', 'isic') else 'mIoU'
        print(f"    {ds_name}:")
        for variant in VARIANT_ORDER:
            if variant in variants:
                print(f"      {variant:8s} → {metric_name}={variants[variant]['best_metric']:.4f} "
                      f"(epoch {variants[variant]['best_epoch']})")

    if args.no_plots:
        print("\n  [Skipping plots per --no-plots]")
        return

    # Plots
    print("\n[3/4] Generating plots with consistent color map...")
    print(f"  Color map: nano=#5DADE2, base=#27AE60, pro=#E67E22, promax=#E74C3C")

    plot_convergence_curves(all_results, output_dir / 'convergence')
    plot_per_dataset_comparison(all_results, output_dir)
    plot_cross_dataset_summary(all_results, output_dir)
    plot_params_vs_accuracy(all_results, output_dir)

    # LaTeX
    print("\n[4/4] Generating LaTeX table...")
    generate_latex_table(all_results, output_dir)

    print(f"\n{'═' * 60}")
    print(f"  Done! All outputs in: {output_dir}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
