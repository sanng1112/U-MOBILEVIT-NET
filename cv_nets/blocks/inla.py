"""
INLA — Inverted Nonlinear Low-rank Attention.

Cơ chế attention nhẹ nhằm kiểm soát hiện tượng *rank collapse* / *spectral
degeneration* của linear attention trong các backbone gọn nhẹ. Ý tưởng cốt lõi:
trước khi thực hiện tổng hợp linear attention (độ phức tạp O(N) theo số token),
ta "nâng" (lift) đặc trưng Query/Key qua một ánh xạ phi tuyến dạng inverted
bottleneck:

    Φ(X) = σ(X W_low + b_low) W_exp + b_exp,   với  d_k < d < r

tức nén (d -> d_k) -> phi tuyến (σ) -> mở rộng (d_k -> r). Việc query/key tương
tác trong một cơ sở (basis) r-chiều giàu hơn giúp làm chậm tốc độ suy giảm hạng
hiệu dụng khi mạng được xếp chồng sâu, trong khi vẫn giữ độ phức tạp tuyến tính.

Công thức attention (giữ tính kết hợp của nhân ma trận để có O(N)):

    S = K̂ᵀ V                      (r × d_v)   — nén ngữ cảnh toàn cục theo basis
    O = D⁻¹ (Q̂ S),  d = Q̂ (K̂ᵀ 1) — truy xuất theo query + chuẩn hóa

Quy ước token: tensor (B, N, C). Đặt ``use_lifting=False`` để thu được baseline
linear attention (cùng pipeline, không có lifting) phục vụ ablation công bằng.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from cv_nets.blocks._common import DropPath, get_activation


__all__ = ["INLAAttention", "INLATransformerBlock"]


def _nonneg_feature_map(x: Tensor) -> Tensor:
    """Kernel feature map không âm (elu+1) để mẫu số chuẩn hóa luôn dương -> ổn định."""
    return F.elu(x) + 1.0


class INLAAttention(nn.Module):
    """
    Inverted Nonlinear Low-rank Attention (single global context, O(N)).

    Args:
        dim: số kênh đầu vào/đầu ra (d).
        dim_compress: chiều nén d_k (mặc định d // 2, thỏa d_k < d).
        dim_expand: chiều mở rộng r (mặc định 2*d, thỏa r > d).
        act: hàm phi tuyến trong lifting (gelu/silu/...).
        use_lifting: True dùng INLA; False -> baseline linear attention (Q̂=Q, K̂=K).
        proj_drop: dropout ở đầu ra.
        eps: hằng số ổn định mẫu số.
    """

    def __init__(
        self,
        dim: int,
        dim_compress: Optional[int] = None,
        dim_expand: Optional[int] = None,
        act: str = "gelu",
        use_lifting: bool = True,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.use_lifting = use_lifting
        self.eps = eps

        dim_compress = dim_compress or max(1, dim // 2)
        dim_expand = dim_expand or (2 * dim)
        self.dim_compress = dim_compress
        self.dim_expand = dim_expand

        # Phép chiếu Q, K, V tiêu chuẩn
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        if use_lifting:
            # Lifting inverted bottleneck: d -> d_k -> (σ) -> r. Dùng chung cho Q và K.
            self.lift = nn.Sequential(
                nn.Linear(dim, dim_compress),
                get_activation(act),
                nn.Linear(dim_compress, dim_expand),
            )
            self._feat_dim = dim_expand
        else:
            self.lift = nn.Identity()
            self._feat_dim = dim

        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _phi(self, x: Tensor) -> Tensor:
        """Lifting Φ rồi áp kernel feature map không âm."""
        return _nonneg_feature_map(self.lift(x))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, d) hoặc (N, d) — thêm batch dim nếu thiếu
        if x.dim() == 2:
            x = x.unsqueeze(0)             # (N, d) → (1, N, d)
            squeeze_out = True
        else:
            squeeze_out = False

        q = self._phi(self.q_proj(x))      # (B, N, F)  F = r nếu lifting, ngược lại d
        k = self._phi(self.k_proj(x))      # (B, N, F)
        v = self.v_proj(x)                 # (B, N, d)

        # S = Kᵀ V : (B, F, d) — nén ngữ cảnh toàn cục theo basis F-chiều
        context = torch.einsum("bnf,bnd->bfd", k, v)

        # mẫu số chuẩn hóa: z = Kᵀ 1 (B, F) ; denom = Q z (B, N)
        k_sum = k.sum(dim=1)                          # (B, F)
        denom = torch.einsum("bnf,bf->bn", q, k_sum)  # (B, N)
        denom = denom.clamp_min(self.eps).unsqueeze(-1)

        # O = D⁻¹ (Q S) : (B, N, d)
        out = torch.einsum("bnf,bfd->bnd", q, context) / denom
        out = self.proj_drop(self.out_proj(out))

        if squeeze_out:
            out = out.squeeze(0)           # (1, N, d) → (N, d)
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, dim_compress={self.dim_compress}, "
                f"dim_expand={self.dim_expand}, use_lifting={self.use_lifting}")


class INLATransformerBlock(nn.Module):
    """
    Khối transformer pre-norm dùng INLAAttention:
        x = x + DropPath(INLA(LN(x)))
        x = x + DropPath(MLP(LN(x)))
    Đầu vào/ra dạng (B, N, C).
    """

    def __init__(
        self,
        dim: int,
        dim_compress: Optional[int] = None,
        dim_expand: Optional[int] = None,
        mlp_ratio: float = 2.0,
        act: str = "gelu",
        use_lifting: bool = True,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = INLAAttention(
            dim, dim_compress=dim_compress, dim_expand=dim_expand,
            act=act, use_lifting=use_lifting, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), get_activation(act), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
