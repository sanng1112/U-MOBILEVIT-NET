"""
Thư viện khối (building blocks) cho thị giác máy tính & transformer.

Hai nhóm:
  * Khối tích hợp framework (dùng hệ thống ``opts``/builder của dự án):
      - ConvBNAct, MV2Block, ResNetBasicBlock
  * Khối "thuần" ``torch.nn`` (dùng trực tiếp, dễ đọc, dễ test, không cần ``opts``):
      - ConvNormAct, DropPath
      - SqueezeExcite, ECA, CBAM (ChannelAttention, SpatialAttention)
      - DepthwiseSeparableConv, InvertedResidual (MBConv)
      - PatchEmbed, MultiHeadSelfAttention, Mlp, TransformerEncoderBlock
      - MobileViTBlock

Ví dụ:
    from cv_nets.blocks import InvertedResidual, MobileViTBlock, CBAM
"""

# --- Khối tích hợp framework ---
from cv_nets.blocks.ConvBNAct import ConvBNAct
from cv_nets.blocks.mv2block import MV2Block
from cv_nets.blocks.ResnetBlock import ResNetBasicBlock

# --- Helper dùng chung ---
from cv_nets.blocks._common import (
    ConvNormAct,
    DropPath,
    drop_path,
    get_activation,
    make_divisible,
    autopad,
)

# --- Attention ---
from cv_nets.blocks.attention import (
    SqueezeExcite,
    ECA,
    ChannelAttention,
    SpatialAttention,
    CBAM,
)

# --- Khối convolution nhẹ ---
from cv_nets.blocks.mobile import (
    DepthwiseSeparableConv,
    InvertedResidual,
)

# --- Khối transformer ---
from cv_nets.blocks.transformer import (
    PatchEmbed,
    MultiHeadSelfAttention,
    Mlp,
    TransformerEncoderBlock,
)

# --- Khối lai conv + transformer ---
from cv_nets.blocks.mobilevit import MobileViTBlock

# --- INLA: Inverted Nonlinear Low-rank Attention (tính mới) ---
from cv_nets.blocks.inla import INLAAttention, INLATransformerBlock


__all__ = [
    # framework
    "ConvBNAct", "MV2Block", "ResNetBasicBlock",
    # common
    "ConvNormAct", "DropPath", "drop_path", "get_activation", "make_divisible", "autopad",
    # attention
    "SqueezeExcite", "ECA", "ChannelAttention", "SpatialAttention", "CBAM",
    # mobile conv
    "DepthwiseSeparableConv", "InvertedResidual",
    # transformer
    "PatchEmbed", "MultiHeadSelfAttention", "Mlp", "TransformerEncoderBlock",
    # hybrid
    "MobileViTBlock",
    # INLA (tính mới)
    "INLAAttention", "INLATransformerBlock",
]
