#!/usr/bin/env python3
"""
============================================================================
Mobile-U-ViT: Architecture Visualization & Summary Generator
============================================================================
Sinh biểu đồ kiến trúc và bảng tổng kết thông số mô hình.
Có thể chạy độc lập hoặc import như module.

Usage:
    python report/main.py                    # In thông tin kiến trúc ra console
    python report/main.py --plot             # Vẽ biểu đồ kiến trúc (matplotlib)
    python report/main.py --plot --save DIR  # Lưu biểu đồ vào thư mục
    python report/main.py --compare          # So sánh các biến thể
============================================================================
"""

import argparse
import textwrap
from typing import Dict, List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# CẤU HÌNH CÁC BIẾN THỂ
# ═══════════════════════════════════════════════════════════════

VARIANTS: Dict[str, Dict] = {
    "nano": {
        "d_model": 48,
        "num_transformer_blocks": 1,
        "expansion_factor": 2.0,
        "stem_channels": [12, 24, 48],
        "params": "~0.3M",
        "encoder_layers": [1, 2, 2],
        "decoder_layers": [2, 1, 1],  # last is ConcatLayer
        "target": "Raspberry Pi 5 / Jetson Nano (edge)",
    },
    "base": {
        "d_model": 64,
        "num_transformer_blocks": 2,
        "expansion_factor": 3.0,
        "stem_channels": [16, 32, 64],
        "params": "~0.6M",
        "encoder_layers": [1, 2, 2],
        "decoder_layers": [2, 1, 1],
        "target": "Jetson Xavier NX / mobile GPU",
    },
    "pro": {
        "d_model": 128,
        "num_transformer_blocks": 3,
        "expansion_factor": 3.0,
        "stem_channels": [32, 64, 128],
        "params": "~2.0M",
        "encoder_layers": [1, 2, 2],
        "decoder_layers": [2, 1, 1],
        "target": "Desktop GPU (RTX 3060+)",
    },
    "promax": {
        "d_model": 256,
        "num_transformer_blocks": 3,
        "expansion_factor": 4.0,
        "stem_channels": [64, 128, 256],
        "params": "~7.0M",
        "encoder_layers": [1, 2, 2],
        "decoder_layers": [2, 1, 1],
        "target": "High-end GPU Server",
    },
}


# ═══════════════════════════════════════════════════════════════
# MÔ TẢ KIẾN TRÚC
# ═══════════════════════════════════════════════════════════════

ARCHITECTURE_DESCRIPTION = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    MOBILE-U-VIT  —  KIẾN TRÚC TỔNG QUÁT                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Input (3 × H × W)                                                       ║
║       │                                                                  ║
║       ▼                                                                  ║
║  ┌─────────────────────────────────────────┐                             ║
║  │  STEM (3 tầng tích chập, stride=2)       │                             ║
║  │  Stem₀: 3 → d/4   (H×W → H/2×W/2)      │──► S₀ (skip đến SegHead)   ║
║  │  Stem₁: d/4 → d/2 (→ H/4×W/4) + GN     │──► S₁ (skip đến SegHead)   ║
║  │  Stem₂: d/2 → d   (→ H/8×W/8) + GN     │──► E₀                       ║
║  └─────────────────────────────────────────┘                             ║
║       │                                                                  ║
║       ▼                                                                  ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │  ENCODER (3 stage, mỗi stage: Downsample + N×UMobileViTLayer)    │    ║
║  │                                                                   │    ║
║  │  Stage 1: DW Down 2× → EncLayer ×1  → E₁ (H/16, d_model)        │    ║
║  │  Stage 2: DW Down 2× → EncLayer ×2  → E₂ (H/32, d_model)        │    ║
║  │  Stage 3: DW Down 2× → EncLayer ×2  → E₃ (H/64, d_model)        │    ║
║  │              (patch_size=(1,1): self-attention toàn cục)          │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║       │                                                                  ║
║       │  E₃ (latent)                                                      ║
║       ▼                                                                  ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │  DECODER (3 stage, mỗi stage: Upsample + N×UMobileViTLayer)      │    ║
║  │                                                                   │    ║
║  │  Stage 1: Up 2× → DecLayer ×2  ←── cross-attn từ E₂              │    ║
║  │  Stage 2: Up 2× → DecLayer ×1  ←── cross-attn từ E₁              │    ║
║  │  Out Blk:  Up 2× → ConcatLayer ←── additive proj từ E₀           │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║       │                                                                  ║
║       ▼                                                                  ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │  SEGMENTATION HEAD                                                │    ║
║  │                                                                   │    ║
║  │  UpsampleHead: ×2 → ×4 → ×8  (skip từ S₁, S₀ qua ConcatLayer)   │    ║
║  │  FeatureRefinementBlock: Dilated Conv + Spatial Attention         │    ║
║  │  Task Classifiers (×K): DW 3×3 + PW 1×1 → K output masks        │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║       │                                                                  ║
║       ▼                                                                  ║
║  Output (Cₖ × H × W) per task                                            ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  _UMobileViTLayer (khối xây dựng cốt lõi):                               ║
║                                                                          ║
║  X ──► [Local Block] ──► [Global Block] ──► [Expansion Block] ──► + ──► GN ──► Out
║         DW 3×3+PW 1×1    Unfold→N×Trans     PW expand→DW→PW        │
║         +ReLU+GN          former→Fold        +ReLU6+GN+Drop        │
║                                                                    │
║         ◄────────────────── Residual (X) ──────────────────────────┘
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════
# COMPONENT DETAILS
# ═══════════════════════════════════════════════════════════════

COMPONENTS = {
    "Stem": {
        "class": "ModuleList[Sequential]",
        "layers": 3,
        "description": (
            "3 tầng Conv2d (kernel=3, stride=2, padding=1) + ReLU(inplace=True). "
            "Tầng 1-2 thêm GroupNorm. Giảm độ phân giải 8× (H×W → H/8×W/8), "
            "mở rộng kênh 3 → d_model."
        ),
    },
    "UMobileViTEncoder": {
        "class": "BaseLayer",
        "stages": 3,
        "description": (
            "3 stage encoder. Mỗi stage: depthwise downsample (DW Conv 3×3, stride=2, "
            "groups=d_model) + 1-2 UMobileViTEncoderLayer (self-attention transformer). "
            "Stage 3 dùng patch_size=(1,1) → self-attention toàn cục."
        ),
    },
    "UMobileViTDecoder": {
        "class": "BaseLayer",
        "stages": 3,
        "description": (
            "3 stage decoder. Mỗi stage: upsample (nearest ×2 + DW Conv 3×3) + "
            "1-2 UMobileViTDecoderLayer (cross-attention) hoặc ConcatLayer (additive proj). "
            "Skip connections từ encoder được dùng làm memory cho cross-attention."
        ),
    },
    "_UMobileViTLayer": {
        "class": "BaseLayer (abstract)",
        "blocks": ["Local", "Global", "Expansion"],
        "description": (
            "Khối xây dựng cốt lõi. Pipeline: Local → Global → Expansion → Residual + GN."
        ),
    },
    "Local Block": {
        "ops": ["DW Conv 3×3 (groups=C)", "PW Conv 1×1", "ReLU", "GroupNorm"],
        "complexity": "O(C·H·W)",
        "description": "Trích xuất đặc trưng không gian tinh (cạnh, kết cấu, màu sắc).",
    },
    "Global Block": {
        "ops": ["unfold_custom (P patches)", "N × Transformer Block", "fold_custom"],
        "complexity": "O(C²·P·S) — tuyến tính theo số patch",
        "description": (
            "Separable self/cross-attention MobileViTv2. Encoder: self-attention. "
            "Decoder: self-attn + cross-attn với encoder skip features."
        ),
    },
    "Expansion Block": {
        "ops": ["PW Expand 1×1 (C→E·C)", "DW Conv 3×3", "PW Project 1×1 (E·C→C)"],
        "complexity": "O(E·C·H·W)",
        "description": (
            "Inverted bottleneck MobileNetV2. ReLU6 + GroupNorm sau mỗi conv. "
            "Hệ số E ∈ {2.0, 3.0, 4.0}. Dropout ở đầu ra."
        ),
    },
    "UpsampleHead": {
        "class": "BaseLayer",
        "stages": 3,
        "description": (
            "Phục hồi độ phân giải: ×2 → ×4 → ×8. Dùng nearest-neighbor upsample + "
            "DW Conv + skip connections từ stem qua UMobileViTDecoderConcatLayer."
        ),
    },
    "FeatureRefinementBlock": {
        "ops": ["Dilated Conv 3×3 (dilation=2)", "Spatial Attention (avg+max pool → Conv 7×7 → Sigmoid)", "PW Mix 1×1"],
        "description": (
            "Tinh chỉnh đặc trưng: dilated conv mở rộng receptive field, "
            "spatial attention lọc nhiễu không gian, residual connection."
        ),
    },
    "Classifier": {
        "ops": ["DW Conv 3×3 (groups=C)", "ReLU", "PW Conv 1×1 (C→num_classes)"],
        "description": (
            "Bộ phân loại pixel nhẹ. MultiTask: mỗi task có classifier riêng, "
            "phần upsample + refinement được chia sẻ."
        ),
    },
}


def print_architecture():
    """In mô tả kiến trúc tổng quát ra console."""
    print(ARCHITECTURE_DESCRIPTION)


def print_components():
    """In chi tiết từng thành phần."""
    print("\n" + "=" * 78)
    print("  CHI TIẾT TỪNG THÀNH PHẦN KIẾN TRÚC")
    print("=" * 78)
    for name, info in COMPONENTS.items():
        print(f"\n{'─' * 60}")
        print(f"  📦 {name}")
        print(f"{'─' * 60}")
        for k, v in info.items():
            if k == "description":
                print(f"  {v}")
            elif k == "ops":
                print(f"  Operations: {' → '.join(v)}")
            elif k == "complexity":
                print(f"  Complexity: {v}")
            else:
                print(f"  {k}: {v}")


def print_variant_table():
    """In bảng so sánh các biến thể."""
    print("\n" + "=" * 100)
    print("  SO SÁNH CÁC BIẾN THỂ MOBILE-U-VIT")
    print("=" * 100)

    header = f"{'Variant':<10} {'d_model':<10} {'Trans Blocks':<14} {'Exp Factor':<12} {'Params':<12} {'Target'}"
    print(header)
    print("-" * 100)

    for name, cfg in VARIANTS.items():
        row = (
            f"{name:<10} "
            f"{cfg['d_model']:<10} "
            f"{cfg['num_transformer_blocks']:<14} "
            f"{cfg['expansion_factor']:<12} "
            f"{cfg['params']:<12} "
            f"{cfg['target']}"
        )
        print(row)

    print("-" * 100)
    print("\n  Encoder: Stem (3 conv) + Stage 1 (1 layer) + Stage 2 (2 layers) + Stage 3 (2 layers)")
    print("  Decoder: Stage 1 (2 layers) + Stage 2 (1 layer) + Out Block (1 ConcatLayer)")
    print("  SegHead: UpsampleHead (×8) + FeatureRefinement + K Task Classifiers")
    print("  Ghi chú: d_model phải chia hết cho 4, 2, và 8 (ràng buộc stem + seg_head)")


def plot_architecture(save_dir: Optional[str] = None):
    """Vẽ biểu đồ kiến trúc bằng matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        import numpy as np
    except ImportError:
        print("[!] matplotlib chưa được cài đặt. Cài đặt: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))
    fig.suptitle("Mobile-U-ViT Architecture Overview", fontsize=16, fontweight="bold", y=0.98)

    # ── LEFT: Overall Architecture ──
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 14)
    ax1.axis("off")
    ax1.set_title("Overall Architecture (Encoder–Decoder U-Net style)", fontsize=12, fontweight="bold", pad=10)

    # Colors
    C_STEM = "#a8d8ea"
    C_ENC = "#ffd3b6"
    C_DEC = "#d5ecc2"
    C_HEAD = "#d5c6e0"
    C_SKIP = "#aaaaaa"
    C_ATTN = "#ffaaa5"

    def draw_box(ax, x, y, w, h, text, color, fontsize=7, edgecolor="black", linewidth=0.8):
        """Vẽ hộp với text."""
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor=edgecolor, linewidth=linewidth,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontfamily="monospace")

    def draw_arrow(ax, x1, y1, x2, y2, color="black", style="->", lw=0.8):
        """Vẽ mũi tên."""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        )

    def draw_skip(ax, x1, y1, x2, y2):
        """Vẽ skip connection (nét đứt)."""
        ax.plot([x1, x2], [y1, y2], color=C_SKIP, linestyle="--", linewidth=0.6)
        # arrowhead
        ax.annotate(
            "", xy=(x2, y2), xytext=(x2 - 0.15, y2),
            arrowprops=dict(arrowstyle="->", color=C_SKIP, lw=0.6),
        )

    # ── Encoder (left side) ──
    ex, ey_start = 2.5, 13.0
    dy = 0.85

    # Input
    draw_box(ax1, ex, ey_start, 2.2, 0.55, "Input 3×H×W", C_STEM, fontsize=8)
    ey = ey_start - dy

    # Stem
    draw_box(ax1, ex, ey, 2.2, 0.55, "Stem₀: Conv→ReLU\n3→d/4, H/2", C_STEM)
    ey_s0 = ey
    ey -= dy
    draw_box(ax1, ex, ey, 2.2, 0.55, "Stem₁: Conv→ReLU→GN\nd/4→d/2, H/4", C_STEM)
    ey_s1 = ey
    ey -= dy
    draw_box(ax1, ex, ey, 2.2, 0.55, "Stem₂: Conv→ReLU→GN\nd/2→d, H/8", C_STEM)
    ey_e0 = ey

    # Encoder stages
    for i, (label, n_layers, patch, y_pos) in enumerate([
        ("Enc Stage 1\nDown+EncLayer×1", 1, "(2,2)", ey - 1.1),
        ("Enc Stage 2\nDown+EncLayer×2", 2, "(2,2)", ey - 2.1),
        ("Enc Stage 3\nDown+EncLayer×2\npatch=(1,1)", 2, "(1,1)", ey - 3.1),
    ]):
        draw_box(ax1, ex, y_pos, 2.4, 0.75, label, C_ENC, fontsize=7)
        if i == 0:
            ey_e1 = y_pos
        elif i == 1:
            ey_e2 = y_pos
        else:
            ey_e3 = y_pos

    # ── Decoder (right side) ──
    dx = 7.5

    # Decoder out block + stages
    dec_stages = [
        ("Dec Out Block\nUp×2+ConcatLayer", C_ATTN, ey_e0 + 0.2),
        ("Dec Stage 2\nUp×2+DecLayer×1\nCross-Attn", C_DEC, ey_e1 + 0.2),
        ("Dec Stage 1\nUp×2+DecLayer×2\nCross-Attn", C_DEC, ey_e2 + 0.2),
    ]

    for label, color, y_pos in dec_stages:
        draw_box(ax1, dx, y_pos, 2.4, 0.75, label, color, fontsize=7)

    # SegHead
    seg_y = ey_s1 + 0.5
    draw_box(ax1, dx, seg_y, 2.5, 0.7, "UpsampleHead\n×2→×4→×8 + Stem Skips", C_HEAD, fontsize=7)
    seg_y2 = seg_y + 1.0
    draw_box(ax1, dx, seg_y2, 2.5, 0.55, "FeatureRefinement\nDilated Conv+Spatial Attn", C_HEAD, fontsize=7)
    seg_y3 = seg_y2 + 0.8
    draw_box(ax1, dx, seg_y3, 2.5, 0.55, "Classifier\nDW 3×3+PW 1×1", C_HEAD, fontsize=7)
    seg_y4 = seg_y3 + 0.85
    draw_box(ax1, dx, seg_y4, 2.2, 0.55, "Output C×H×W", C_STEM, fontsize=8)

    # ── Forward arrows (encoder) ──
    for y1, y2 in [
        (ey_start - 0.28, ey_s0 + 0.28),
        (ey_s0 - 0.28, ey_s1 + 0.28),
        (ey_s1 - 0.28, ey_e0 + 0.28),
        (ey_e0 - 0.28, ey_e1 + 0.38),
        (ey_e1 - 0.38, ey_e2 + 0.38),
        (ey_e2 - 0.38, ey_e3 + 0.38),
    ]:
        draw_arrow(ax1, ex, y1, ex, y2)

    # ── Forward arrows (decoder, going up) ──
    for y1, y2 in [
        (ey_e3 + 0.38, ey_e2 + 0.05),
        (ey_e2 - 0.05, ey_e1 + 0.05),
        (ey_e1 - 0.05, ey_e0 + 0.05),
        (ey_e0 - 0.05, seg_y + 0.35),
        (seg_y - 0.35, seg_y2 + 0.28),
        (seg_y2 - 0.28, seg_y3 + 0.28),
        (seg_y3 - 0.28, seg_y4 + 0.28),
    ]:
        draw_arrow(ax1, dx, y1, dx, y2)

    # ── Latent connection (encoder → decoder) ──
    ax1.annotate(
        "", xy=(dx - 1.2, ey_e3), xytext=(ex + 1.2, ey_e3),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0,
                       connectionstyle="arc3,rad=0"),
    )

    # ── Skip connections ──
    for enc_y, dec_y in [
        (ey_e2, ey_e2 + 0.2),
        (ey_e1, ey_e1 + 0.2),
        (ey_e0, ey_e0 + 0.2),
    ]:
        ax1.plot([ex + 1.2, dx - 1.2], [enc_y, dec_y],
                color=C_SKIP, linestyle="--", linewidth=0.6)
        ax1.annotate("", xy=(dx - 1.2, dec_y), xytext=(dx - 1.0, dec_y),
                    arrowprops=dict(arrowstyle="->", color=C_SKIP, lw=0.6))

    # Stem skips to SegHead
    ax1.plot([ex + 1.2, dx - 1.2], [ey_s1, seg_y],
            color=C_SKIP, linestyle="--", linewidth=0.5)
    ax1.plot([ex + 1.2, dx - 1.2], [ey_s0, seg_y + 0.1],
            color=C_SKIP, linestyle="--", linewidth=0.5)

    # ── Legend ──
    legend_y = 1.5
    patches = [
        mpatches.Patch(color=C_STEM, label="Stem / I/O"),
        mpatches.Patch(color=C_ENC, label="Encoder"),
        mpatches.Patch(color=C_DEC, label="Decoder"),
        mpatches.Patch(color=C_ATTN, label="ConcatLayer"),
        mpatches.Patch(color=C_HEAD, label="SegHead"),
        mpatches.Patch(color="white", ec=C_SKIP, linestyle="--", label="Skip Connection"),
    ]
    ax1.legend(handles=patches, loc="lower center", ncol=3, fontsize=7, framealpha=0.8)

    # ── RIGHT: _UMobileViTLayer Detail ──
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 14)
    ax2.axis("off")
    ax2.set_title("Core Building Block: _UMobileViTLayer", fontsize=12, fontweight="bold", pad=10)

    lx = 4.0
    ly = 13.0
    ldy = 0.9

    # Input
    draw_box(ax2, lx, ly, 3.5, 0.55, "Input X ∈ R^{B×C×H×W}", "#e8e8e8", fontsize=8)
    ly_in = ly

    # Local block
    ly -= ldy
    draw_box(ax2, lx, ly, 3.5, 0.7, "Local Block\nDW 3×3 → PW 1×1 → ReLU → GN", "#a8d8ea", fontsize=7)
    ly_loc_out = ly - 0.4

    # Global block box
    ly -= ldy + 0.1
    rect = FancyBboxPatch((lx - 1.9, ly - 1.2), 3.8, 2.4, boxstyle="round,pad=0.1",
                          facecolor="none", edgecolor="orange", linewidth=0.8, linestyle="--")
    ax2.add_patch(rect)
    ax2.text(lx + 2.1, ly, "Global Block", fontsize=7, color="orange", fontweight="bold", rotation=90, va="center")

    draw_box(ax2, lx, ly + 0.8, 3.2, 0.5, "unfold_custom → P patches", "#ffd3b6", fontsize=7)
    draw_box(ax2, lx, ly, 3.2, 0.7, "N × Transformer Block\nSelf-Attn (Enc) / Cross-Attn (Dec)", "#ffd3b6", fontsize=7)
    draw_box(ax2, lx, ly - 0.8, 3.2, 0.5, "fold_custom → spatial", "#ffd3b6", fontsize=7)
    ly_glob_out = ly - 1.45

    # Expansion block box
    ly -= 1.8
    rect = FancyBboxPatch((lx - 2.0, ly - 1.3), 4.0, 2.6, boxstyle="round,pad=0.1",
                          facecolor="none", edgecolor="green", linewidth=0.8, linestyle="--")
    ax2.add_patch(rect)
    ax2.text(lx + 2.2, ly, "Expansion Block", fontsize=7, color="green", fontweight="bold", rotation=90, va="center")

    draw_box(ax2, lx, ly + 0.9, 3.5, 0.55, "PW Expand 1×1: C → ⌊E·C⌋ + GN + ReLU6", "#d5ecc2", fontsize=7)
    draw_box(ax2, lx, ly + 0.15, 3.5, 0.55, "DW Conv 3×3: groups=⌊E·C⌋ + GN + ReLU6", "#d5ecc2", fontsize=7)
    draw_box(ax2, lx, ly - 0.6, 3.5, 0.55, "PW Project 1×1: ⌊E·C⌋ → C + GN + Dropout", "#d5ecc2", fontsize=7)
    ly_exp_out = ly - 1.6

    # Residual connection
    ly_res = ly_exp_out - 0.3
    draw_box(ax2, lx, ly_res, 0.3, 0.3, "+", "white", fontsize=10)
    ly_res2 = ly_res - 0.5
    draw_box(ax2, lx, ly_res2, 2.0, 0.45, "GroupNorm", "#e0e0e0", fontsize=7)
    ly_out = ly_res2 - 0.6
    draw_box(ax2, lx, ly_out, 3.5, 0.55, "Output ∈ R^{B×C×H×W}", "#e8e8e8", fontsize=8)

    # Forward arrows
    for y1, y2 in [
        (ly_in - 0.28, ly_in - ldy + 0.35),
        (ly_loc_out + 0.05, ly_in - ldy - 0.35),
        (ly_in - ldy - 0.45, ly_in - 2 * ldy - 0.35),
        (ly_glob_out + 0.05, ly_in - 2 * ldy - 1.1),
        (ly_in - 2 * ldy - 1.82, ly - 0.9 + 0.28),
        (ly - 0.9 - 0.28, ly + 0.15 + 0.28),
        (ly + 0.15 - 0.28, ly - 0.6 + 0.28),
        (ly_exp_out + 0.05, ly_res + 0.15),
        (ly_res - 0.15, ly_res2 + 0.22),
        (ly_res2 - 0.22, ly_out + 0.28),
    ]:
        draw_arrow(ax2, lx, y1, lx, y2)

    # Residual line
    ax2.plot([lx - 2.0, lx - 2.0, lx - 0.15], [ly_in, ly_res, ly_res],
             color="gray", linestyle="--", linewidth=0.8)
    ax2.annotate("", xy=(lx - 0.15, ly_res), xytext=(lx - 0.3, ly_res),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Formula
    ax2.text(lx, ly_out - 0.6, "Output = GN(X + Expand(Global(Local(X))))",
             ha="center", fontsize=8, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    # Legend
    patches2 = [
        mpatches.Patch(color="#a8d8ea", label="Local Block"),
        mpatches.Patch(color="#ffd3b6", label="Global Block (Transformer)"),
        mpatches.Patch(color="#d5ecc2", label="Expansion Block (Inverted Bottleneck)"),
        mpatches.Patch(color="white", ec="gray", linestyle="--", label="Residual Connection"),
    ]
    ax2.legend(handles=patches2, loc="lower center", ncol=2, fontsize=7, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "mobile_u_vit_architecture.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[+] Đã lưu biểu đồ kiến trúc: {path}")
    else:
        # Try to show interactively
        try:
            plt.show()
        except Exception:
            print("[!] Không thể hiển thị biểu đồ tương tác. Dùng --save DIR để lưu file.")


def main():
    parser = argparse.ArgumentParser(
        description="Mobile-U-ViT Architecture Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python report/main.py                     # In kiến trúc ra console
              python report/main.py --plot              # Vẽ biểu đồ (hiển thị)
              python report/main.py --plot --save figs  # Lưu biểu đồ vào figs/
              python report/main.py --compare           # So sánh các biến thể
              python report/main.py --all               # Tất cả
        """),
    )
    parser.add_argument("--plot", action="store_true", help="Vẽ biểu đồ kiến trúc bằng matplotlib")
    parser.add_argument("--save", type=str, default=None, help="Thư mục lưu biểu đồ")
    parser.add_argument("--compare", action="store_true", help="In bảng so sánh các biến thể")
    parser.add_argument("--components", action="store_true", help="In chi tiết từng thành phần")
    parser.add_argument("--all", action="store_true", help="Chạy tất cả các chế độ")

    args = parser.parse_args()

    do_all = args.all

    if do_all or (not args.plot and not args.compare and not args.components):
        print_architecture()

    if do_all or args.components:
        print_components()

    if do_all or args.compare:
        print_variant_table()

    if do_all or args.plot:
        plot_architecture(save_dir=args.save)


if __name__ == "__main__":
    main()
