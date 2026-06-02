#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              U-MobileViT-Net — Benchmark Orchestrator                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Chạy benchmark toàn diện: train + eval cho mọi model variant trên mọi dataset.

Cách dùng:
    # Benchmark một model trên một dataset
    python benchmark/run_benchmark.py --variant base --dataset camvid

    # Benchmark tất cả variants trên tất cả datasets
    python benchmark/run_benchmark.py --all

    # Chỉ benchmark comparison models
    python benchmark/run_benchmark.py --comparison --dataset camvid

    # Dry run (kiểm tra config)
    python benchmark/run_benchmark.py --all --dry-run

Output:
    benchmark/results/<dataset>/<variant>_<timestamp>/
    ├── best_model.pth
    ├── training_history.json
    ├── metrics.json
    └── training_curves.png
"""

import sys
import os
import argparse
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Experiment Configs
# ═══════════════════════════════════════════════════════════════

UMOBILEVIT_VARIANTS = ["nano", "base", "pro", "promax"]

DATASETS = ["camvid", "voc", "drive", "kvasir", "isic", "coco_leaf"]

COMPARISON_MODELS = {
    "unet": {"type": "comparison", "model": "unet", "backbone": "resnet34"},
    "deeplabv3p": {"type": "comparison", "model": "deeplabv3p", "backbone": "mobilenet_v2"},
    "segformer": {"type": "comparison", "model": "segformer", "variant": "b0"},
}

# Dataset-specific configs
DATASET_CONFIGS = {
    "camvid":    {"epochs": 500, "patience": 20, "image_size": "360 480",
                  "scheduler": "poly", "class_weights": True, "label_smoothing": 0.1},
    "voc":       {"epochs": 200, "patience": 20, "image_size": "384 384",
                  "scheduler": "poly", "class_weights": True, "label_smoothing": 0.1},
    "drive":     {"epochs": 150, "patience": 15, "image_size": "256 256",
                  "scheduler": "cosine", "class_weights": False, "label_smoothing": 0.0},
    "kvasir":    {"epochs": 150, "patience": 15, "image_size": "256 256",
                  "scheduler": "cosine", "class_weights": False, "label_smoothing": 0.0},
    "isic":      {"epochs": 150, "patience": 15, "image_size": "256 256",
                  "scheduler": "cosine", "class_weights": False, "label_smoothing": 0.0},
    "coco_leaf": {"epochs": 300, "patience": 20, "image_size": "320 320",
                  "scheduler": "poly", "class_weights": True, "label_smoothing": 0.1},
}


def get_experiment_dir(dataset: str, variant: str) -> Path:
    """Tạo thư mục experiment với timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RESULTS_DIR / dataset / f"{variant}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def build_train_command(dataset: str, variant: str, exp_dir: Path,
                        extra_args: List[str] = None) -> List[str]:
    """Xây dựng lệnh train.py với đầy đủ tham số."""
    ds_cfg = DATASET_CONFIGS.get(dataset, {})

    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--variant", variant,
        "--epochs", str(ds_cfg.get("epochs", 300)),
        "--patience", str(ds_cfg.get("patience", 15)),
        "--image-size"] + ds_cfg.get("image_size", "320 320").split()
    cmd += ["--save-dir", str(exp_dir)]
    cmd += ["--no-plot"]  # Non-interactive mode

    # Dataset-specific defaults
    if ds_cfg.get("class_weights", False):
        cmd.append("--class-weights")
    if ds_cfg.get("scheduler"):
        cmd += ["--scheduler", ds_cfg["scheduler"]]
    if ds_cfg.get("label_smoothing", 0.0) > 0:
        cmd += ["--label-smoothing", str(ds_cfg["label_smoothing"])]

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_single_experiment(dataset: str, variant: str,
                          extra_args: List[str] = None,
                          dry_run: bool = False) -> Tuple[bool, Path]:
    """Chạy một experiment đơn lẻ."""
    exp_dir = get_experiment_dir(dataset, variant)
    cmd = build_train_command(dataset, variant, exp_dir, extra_args)

    print(f"\n{'='*60}")
    print(f"  [{variant.upper()}] {dataset}")
    print(f"  Dir: {exp_dir}")
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True, exp_dir
    print(f"{'='*60}")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

        result = subprocess.run(
            cmd, cwd=str(Path(__file__).resolve().parent.parent),
            env=env, capture_output=False,
        )

        if result.returncode == 0:
            # Save experiment metadata
            meta = {
                "dataset": dataset,
                "variant": variant,
                "timestamp": datetime.now().isoformat(),
                "command": " ".join(cmd),
                "status": "completed",
            }
            with open(exp_dir / "experiment.json", "w") as f:
                json.dump(meta, f, indent=2)

            return True, exp_dir
        else:
            print(f"[FAILED] Return code: {result.returncode}")
            return False, exp_dir

    except Exception as e:
        print(f"[ERROR] {e}")
        return False, exp_dir


def run_benchmark(
    variants: List[str] = None,
    datasets: List[str] = None,
    comparison: bool = False,
    extra_args: List[str] = None,
    dry_run: bool = False,
) -> Dict[str, Dict[str, bool]]:
    """Chạy benchmark toàn diện.

    Args:
        variants: List model variants (None = all U-MobileViT variants)
        datasets: List datasets (None = all)
        comparison: Nếu True, thêm comparison models
        extra_args: Tham số bổ sung cho train.py
        dry_run: Chỉ in lệnh, không chạy

    Returns:
        Dict[dataset][variant] = success
    """
    variants = variants or UMOBILEVIT_VARIANTS
    datasets = datasets or DATASETS
    results = {}

    # U-MobileViT variants
    for dataset in datasets:
        results[dataset] = {}
        for variant in variants:
            success, exp_dir = run_single_experiment(dataset, variant, extra_args, dry_run)
            results[dataset][f"umobilevit_{variant}"] = success

    # Comparison models
    if comparison:
        for dataset in datasets:
            for model_name, model_cfg in COMPARISON_MODELS.items():
                comp_args = (extra_args or []) + [
                    "--comparison-model", model_cfg["model"],
                    "--comparison-backbone", model_cfg.get("backbone", ""),
                ]
                success, exp_dir = run_single_experiment(
                    dataset, model_name, comp_args, dry_run
                )
                results[dataset][model_name] = success

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    n_total = 0
    n_success = 0
    for dataset, variants_map in results.items():
        for variant, success in variants_map.items():
            n_total += 1
            if success:
                n_success += 1
            status = "✓" if success else "✗"
            print(f"  [{status}] {dataset:12s} {variant}")

    print(f"\n  Completed: {n_success}/{n_total}")
    print(f"  Results:   {RESULTS_DIR}")
    print(f"{'='*60}")

    return results


def list_experiments():
    """Liệt kê tất cả experiments đã chạy."""
    experiments = []
    for ds_dir in sorted(RESULTS_DIR.iterdir()):
        if ds_dir.is_dir():
            for exp_dir in sorted(ds_dir.iterdir()):
                if exp_dir.is_dir():
                    meta_path = exp_dir / "experiment.json"
                    if meta_path.exists():
                        with open(meta_path) as f:
                            meta = json.load(f)
                        experiments.append(meta)

    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'Dataset':<15} {'Variant':<20} {'Status':<12} {'Date'}")
    print("-" * 65)
    for exp in experiments:
        print(f"{exp.get('dataset', '?'):<15} {exp.get('variant', '?'):<20} "
              f"{exp.get('status', '?'):<12} {exp.get('timestamp', '?')[:19]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-MobileViT-Net Benchmark Runner")
    parser.add_argument("--variant", type=str, default=None,
                       help="Model variant (single run)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset (single run)")
    parser.add_argument("--all", action="store_true",
                       help="Run ALL variants on ALL datasets")
    parser.add_argument("--comparison", action="store_true",
                       help="Include comparison models")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--list", action="store_true",
                       help="List all past experiments")
    parser.add_argument("--extra-args", type=str, default="",
                       help="Extra arguments for train.py")

    args = parser.parse_args()

    if args.list:
        list_experiments()
    elif args.variant and args.dataset:
        run_single_experiment(
            args.dataset, args.variant,
            extra_args=args.extra_args.split() if args.extra_args else None,
            dry_run=args.dry_run,
        )
    elif args.all or (not args.variant and not args.dataset):
        run_benchmark(
            variants=[args.variant] if args.variant else None,
            datasets=[args.dataset] if args.dataset else None,
            comparison=args.comparison,
            extra_args=args.extra_args.split() if args.extra_args else None,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
