"""
Cấu hình các biến thể U-MobileViT-Net.

Hỗ trợ 4 biến thể:
  - nano   (0.75x width, ~0.3M params) — siêu nhẹ cho edge devices
  - base   (1.0x  width, ~0.6M params) — cân bằng accuracy/ speed
  - pro    (2.0x  width, ~2.0M params) — accuracy cao
  - promax (4.0x  width, ~7.0M params) — maximum accuracy

Mỗi variant điều chỉnh:
  - d_model: số channels chính (phải chia hết cho 4, 2, 8 để tương thích stem + seg_head)
  - num_transformer_blocks: số transformer block mỗi layer (độ sâu)
  - expansion_factor: hệ số mở rộng cho inverted bottleneck
  - target_groups: số GroupNorm groups mong muốn

Tham khảo:
  MobileViTv2: Separable Self-Attention for Mobile Vision Transformers
"""

from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════
# Variant Definitions
# ═══════════════════════════════════════════════════════════════

UMOBILEVIT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "nano": {
        "d_model": 48,
        "num_transformer_blocks": 1,
        "expansion_factor": 2.0,
        "target_groups": 4,
        "description": "Nano (0.75x) — ultra-lightweight for edge/mobile devices",
        # stem:        [12, 24, 48]  — all divisible by 2,3,4,6,12
        # seg_head:    48→24→12→6    — 6 needs groups=3 or 2 or 1
        # head_out:    6 channels at classifier
        "params_estimate": "~0.3M",
    },
    "base": {
        "d_model": 64,
        "num_transformer_blocks": 2,
        "expansion_factor": 3.0,
        "target_groups": 4,
        "description": "Base (1.0x) — balanced accuracy/speed (reference model)",
        # stem:        [16, 32, 64]  — all divisible by 2,4,8,16
        # seg_head:    64→32→16→8    — 8 divisible by 2,4,8
        # head_out:    8 channels at classifier
        "params_estimate": "~0.6M",
    },
    "pro": {
        "d_model": 128,
        "num_transformer_blocks": 3,
        "expansion_factor": 3.0,
        "target_groups": 4,
        "description": "Pro (2.0x) — high accuracy for workstation inference",
        # stem:        [32, 64, 128] — all divisible by 2,4,8,16,32
        # seg_head:    128→64→32→16  — 16 divisible by 2,4,8,16
        # head_out:    16 channels at classifier
        "params_estimate": "~2.0M",
    },
    "promax": {
        "d_model": 256,
        "num_transformer_blocks": 3,
        "expansion_factor": 4.0,
        "target_groups": 8,
        "description": "ProMax (4.0x) — maximum accuracy, server-grade",
        # stem:        [64, 128, 256] — all divisible by 2,4,8,16,32,64
        # seg_head:    256→128→64→32   — 32 divisible by 2,4,8,16,32
        # head_out:    32 channels at classifier
        "params_estimate": "~7.0M",
    },
}


def get_variant(variant: str = "base") -> Dict[str, Any]:
    """Lấy config cho một variant.

    Args:
        variant: "nano", "base", "pro", hoặc "promax"

    Returns:
        Dict chứa d_model, num_transformer_blocks, expansion_factor, target_groups

    Raises:
        ValueError nếu variant không hợp lệ
    """
    variant = variant.lower().strip()
    if variant not in UMOBILEVIT_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Valid options: {list(UMOBILEVIT_VARIANTS.keys())}"
        )
    return UMOBILEVIT_VARIANTS[variant].copy()


def list_variants() -> Dict[str, str]:
    """Liệt kê tất cả các variant và mô tả."""
    return {
        name: cfg["description"]
        for name, cfg in UMOBILEVIT_VARIANTS.items()
    }


# ═══════════════════════════════════════════════════════════════
# Stem Output Channels (computed from d_model)
# ═══════════════════════════════════════════════════════════════

def get_stem_channels(d_model: int) -> list:
    """Trả về [c1, c2, c3] cho stem block outputs."""
    return [d_model // 4, d_model // 2, d_model]


def get_seghead_channels(d_model: int) -> list:
    """Trả về [c1, c2, c3] cho UpsampleHead (x2, x4, x8)."""
    return [d_model // 2, d_model // 4, d_model // 8]
