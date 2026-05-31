"""
Công cụ đo cấu trúc phổ của biểu diễn — phục vụ chẩn đoán *rank collapse* và
*spectral degeneration* trong attention.

Các hàm nhận một ma trận biểu diễn ``M`` (token × feature), dạng (..., N, d) hoặc
(N, d), và trả về các đại lượng phổ tính từ các giá trị kỳ dị (singular values).
"""
from typing import Tuple

import torch
from torch import Tensor


__all__ = ["singular_values", "effective_rank", "spectral_entropy", "singular_value_decay"]


def singular_values(M: Tensor, center: bool = True) -> Tensor:
    """Trả về vector singular values (giảm dần) của M (N×d). Có thể trừ trung bình token."""
    if M.dim() > 2:
        M = M.reshape(-1, M.shape[-1])
    if center:
        M = M - M.mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(M.float())


def effective_rank(M: Tensor, center: bool = True, eps: float = 1e-12) -> float:
    """
    Hạng hiệu dụng (Roy & Vetterli 2007): erank = exp(H(p)), với p là phân bố
    chuẩn hóa của singular values và H là entropy Shannon. Đo "số chiều thực sự
    đóng góp" cho biểu diễn — càng cao càng ít suy giảm hạng.
    """
    s = singular_values(M, center=center)
    p = s / (s.sum() + eps)
    p = p[p > 0]
    entropy = -(p * (p + eps).log()).sum()
    return float(torch.exp(entropy))


def spectral_entropy(M: Tensor, center: bool = True, eps: float = 1e-12, normalize: bool = True) -> float:
    """Entropy của phân bố năng lượng phổ (chuẩn hóa theo log(#sv) nếu normalize=True)."""
    s = singular_values(M, center=center)
    p = s / (s.sum() + eps)
    p = p[p > 0]
    entropy = -(p * (p + eps).log()).sum()
    if normalize and p.numel() > 1:
        entropy = entropy / torch.log(torch.tensor(float(p.numel())))
    return float(entropy)


def singular_value_decay(M: Tensor, center: bool = True, eps: float = 1e-12) -> Tuple[Tensor, float]:
    """
    Trả về (phổ chuẩn hóa s/s_max, tỷ lệ năng lượng top-1 = σ₁²/Σσ²).
    Tỷ lệ top-1 càng lớn => phổ càng dốc => suy giảm phổ càng mạnh.
    """
    s = singular_values(M, center=center)
    energy = (s ** 2)
    top1_ratio = float(energy[0] / (energy.sum() + eps))
    return s / (s[0] + eps), top1_ratio
