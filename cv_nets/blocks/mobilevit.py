"""
MobileViT block (Mehta & Rastegari, 2021) — kết hợp tính cục bộ của convolution và
ngữ cảnh toàn cục của transformer trong một khối nhẹ, phù hợp thiết bị biên.

Luồng xử lý:
    1. Local representation: conv nxn (cục bộ) + conv 1x1 chiếu lên `transformer_dim`.
    2. Unfold: tách feature map thành các patch không chồng lấp -> chuỗi token.
    3. Global representation: vài lớp Transformer encoder học quan hệ giữa các patch.
    4. Fold: ghép token trở lại feature map.
    5. Fusion: chiếu về số kênh gốc, nối (concat) với đầu vào, rồi conv nxn hợp nhất.
"""
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from cv_nets.blocks._common import ConvNormAct, get_activation
from cv_nets.blocks.transformer import TransformerEncoderBlock


__all__ = ["MobileViTBlock"]


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        out_channels: Optional[int] = None,
        n_transformer_blocks: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        patch_h: int = 2,
        patch_w: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: Union[str, nn.Module] = "silu",
        norm_layer: Optional[type] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.patch_h = patch_h
        self.patch_w = patch_w

        # 1. Local representation
        self.local_rep = nn.Sequential(
            ConvNormAct(in_channels, in_channels, kernel_size, groups=1, act=act, norm_layer=norm_layer),
            ConvNormAct(in_channels, transformer_dim, kernel_size=1, act=None, norm_layer=None),
        )

        # 3. Global representation (transformer)
        self.global_rep = nn.ModuleList([
            TransformerEncoderBlock(
                dim=transformer_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=dropout, attn_drop=attn_dropout, act="gelu",
            )
            for _ in range(n_transformer_blocks)
        ])
        self.global_norm = nn.LayerNorm(transformer_dim)

        # 5. Fusion
        self.proj = ConvNormAct(transformer_dim, in_channels, kernel_size=1, act=act, norm_layer=norm_layer)
        self.fusion = ConvNormAct(2 * in_channels, out_channels, kernel_size, act=act, norm_layer=norm_layer)

    def _unfold(self, x: Tensor):
        """(B, C, H, W) -> tokens (B*patch_area, num_patches, C) + metadata để fold."""
        B, C, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w
        # pad cho chia hết patch nếu cần (giữ tính linh hoạt kích thước đầu vào)
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        nph, npw = Hp // ph, Wp // pw

        x = x.reshape(B, C, nph, ph, npw, pw)
        x = x.permute(0, 3, 5, 2, 4, 1)              # (B, ph, pw, nph, npw, C)
        x = x.reshape(B * ph * pw, nph * npw, C)     # (B*P, N, C)
        info = (B, C, Hp, Wp, nph, npw, pad_h, pad_w)
        return x, info

    def _fold(self, x: Tensor, info) -> Tensor:
        """tokens (B*patch_area, num_patches, C) -> (B, C, H, W)."""
        B, C, Hp, Wp, nph, npw, pad_h, pad_w = info
        ph, pw = self.patch_h, self.patch_w
        x = x.reshape(B, ph, pw, nph, npw, C)
        x = x.permute(0, 5, 3, 1, 4, 2)              # (B, C, nph, ph, npw, pw)
        x = x.reshape(B, C, Hp, Wp)
        if pad_h or pad_w:
            x = x[:, :, : Hp - pad_h, : Wp - pad_w]  # bỏ phần đã pad
        return x

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        y = self.local_rep(x)                        # (B, transformer_dim, H, W)

        tokens, info = self._unfold(y)
        for blk in self.global_rep:
            tokens = blk(tokens)
        tokens = self.global_norm(tokens)
        y = self._fold(tokens, info)                 # (B, transformer_dim, H, W)

        y = self.proj(y)                             # về in_channels
        y = self.fusion(torch.cat([residual, y], dim=1))
        return y
