"""
Các khối Transformer cho thị giác (Vision Transformer style).

Gồm:
    - PatchEmbed: chia ảnh thành các patch và chiếu thành token (Conv2d stride=patch).
    - MultiHeadSelfAttention: self-attention đa đầu chuẩn (scaled dot-product).
    - Mlp: feed-forward 2 lớp của transformer.
    - TransformerEncoderBlock: khối encoder pre-norm (LN -> MHSA -> LN -> MLP) kèm
      residual và stochastic depth.

Quy ước token: tensor dạng (B, N, C) với N = số token, C = embed_dim.
"""
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from cv_nets.blocks._common import DropPath, get_activation


__all__ = ["PatchEmbed", "MultiHeadSelfAttention", "Mlp", "TransformerEncoderBlock"]


class PatchEmbed(nn.Module):
    """
    Chia ảnh (B, C, H, W) thành lưới patch và chiếu mỗi patch thành một token.

    Trả về (tokens, (H', W')) với tokens dạng (B, H'*W', embed_dim) để các lớp
    transformer phía sau xử lý; (H', W') dùng để khôi phục về dạng ảnh nếu cần.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.proj(x)                       # (B, E, H', W')
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N, E)
        return x, (h, w)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention đa đầu (scaled dot-product). Đầu vào/ra: (B, N, C)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) phải chia hết cho num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]        # mỗi cái (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    """Feed-forward 2 lớp của transformer (Linear -> Act -> Drop -> Linear -> Drop)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act: str = "gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = get_activation(act)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerEncoderBlock(nn.Module):
    """
    Khối Transformer encoder pre-norm:
        x = x + DropPath(MHSA(LN(x)))
        x = x + DropPath(MLP(LN(x)))
    Đầu vào/ra giữ nguyên dạng (B, N, C).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act: str = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), act=act, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
