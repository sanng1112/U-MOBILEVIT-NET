"""
Các khối attention nhẹ cho thị giác máy tính.

Gồm:
    - SqueezeExcite (SE): tái hiệu chỉnh kênh bằng global context.
    - ECA: attention kênh hiệu quả, gần như không thêm tham số.
    - ChannelAttention / SpatialAttention / CBAM: attention kênh + không gian.

Tất cả đều giữ nguyên kích thước (in_channels == out_channels) và rất rẻ về FLOPs,
phù hợp gắn vào mạng nhẹ chạy thời gian thực trên thiết bị biên.
"""
import math
from typing import Union

import torch
from torch import nn, Tensor

from cv_nets.blocks._common import make_divisible, get_activation


__all__ = ["SqueezeExcite", "ECA", "ChannelAttention", "SpatialAttention", "CBAM"]


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation (Hu et al., 2018).

    Nén thông tin không gian bằng global average pooling, học trọng số quan trọng
    cho từng kênh rồi nhân lại (gating). `rd_ratio` điều khiển độ thắt bottleneck.
    """

    def __init__(
        self,
        channels: int,
        rd_ratio: float = 0.25,
        act: str = "relu",
        gate: str = "sigmoid",
    ) -> None:
        super().__init__()
        rd_channels = make_divisible(channels * rd_ratio, divisor=8)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.act = get_activation(act)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = get_activation(gate)

    def forward(self, x: Tensor) -> Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)        # squeeze: global avg pool
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))                   # excitation: trọng số kênh [0,1]
        return x * s


class ECA(nn.Module):
    """
    Efficient Channel Attention (Wang et al., 2020).

    Thay 2 lớp FC của SE bằng một conv 1D trên trục kênh -> gần như không thêm tham số.
    Kích thước kernel được chọn thích ứng theo số kênh.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1            # đảm bảo lẻ
        k_size = max(3, k_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        s = x.mean(dim=(2, 3))                     # (N, C)
        s = self.conv(s.unsqueeze(1)).squeeze(1)  # tương tác kênh lân cận
        s = self.gate(s).unsqueeze(-1).unsqueeze(-1)
        return x * s


class ChannelAttention(nn.Module):
    """Nhánh attention kênh của CBAM: gộp avg-pool và max-pool qua MLP chia sẻ."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return x * self.gate(avg + mx)


class SpatialAttention(nn.Module):
    """Nhánh attention không gian của CBAM: từ thống kê kênh -> bản đồ trọng số HxW."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        att = self.gate(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018):
    áp dụng tuần tự attention kênh rồi attention không gian.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_att(self.channel_att(x))
