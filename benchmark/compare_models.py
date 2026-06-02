#!/usr/bin/env python3
"""
So sánh kết quả giữa các models trên tất cả datasets.

Cách dùng:
    python benchmark/compare_models.py --results-dir benchmark/results
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_all_results(results_dir: Path) -> Dict:
    """Load và aggregate tất cả results."""
    all_results = defaultdict(lambda: defaultdict(list))

    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for exp_dir in sorted(ds_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            parts = exp_dir.name.split('_')
            variant = parts[0] if parts else "unknown"

            history_path = exp_dir / "training_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
                best_epoch = int(np.argmax(history["val_metric"]) + 1)
                best_metric = float(max(history["val_metric"]))
                best_loss = float(min(history["val_loss"]))
                all_results[dataset][variant].append({
                    "best_metric": best_metric,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "exp_dir": str(exp_dir),
                })

    return dict(all_results)


def compare_models(results_dir: str = "benchmark/results"):
    """In bảng so sánh đầy đủ."""
    results = load_all_results(Path(results_dir))

    if not results:
        print("No results found.")
        return

    # Header
    datasets = sorted(results.keys())
    variants = sorted(set(v for ds in results for v in results[ds]))

    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*80}")

    # Print per dataset
    for dataset in datasets:
        print(f"\n{'─'*80}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"  {'Model':<18} {'Best Metric':>12} {'Best Loss':>12} {'Best Epoch':>10}")
        print(f"  {'─'*50}")

        ds_results = results[dataset]
        for variant in sorted(ds_results.keys()):
            runs = ds_results[variant]
            if runs:
                best = max(runs, key=lambda r: r["best_metric"])
                print(f"  {variant:<18} {best['best_metric']:>12.4f} "
                      f"{best['best_loss']:>12.4f} {best['best_epoch']:>10}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*80}")
    header = f"  {'Model':<18}"
    for ds in datasets:
        header += f" {ds[:8]:>9}"
    print(header)
    print(f"  {'─'*(18 + 10*len(datasets))}")

    for variant in variants:
        row = f"  {variant:<18}"
        for dataset in datasets:
            runs = results.get(dataset, {}).get(variant, [])
            if runs:
                best = max(r["best_metric"] for r in runs)
                row += f" {best:>9.4f}"
            else:
                row += f" {'--':>9}"
        print(row)

    print(f"\n{'='*80}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results-dir", type=str, default="benchmark/results")
    args = parser.parse_args()
    compare_models(args.results_dir)
