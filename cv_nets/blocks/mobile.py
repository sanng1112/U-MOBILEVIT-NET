"""
Các khối convolution nhẹ cho mạng di động / thiết bị biên.

Gồm:
    - DepthwiseSeparableConv: depthwise + pointwise (MobileNetV1).
    - InvertedResidual (MBConv): khối nghịch đảo của MobileNetV2/V3, tùy chọn SE và
      stochastic depth — đơn vị nền tảng của hầu hết backbone real-time hiện đại.
"""
from typing import Optional, Union

from torch import nn, Tensor

from cv_nets.blocks._common import ConvNormAct, DropPath, autopad
from cv_nets.blocks.attention import SqueezeExcite


__all__ = ["DepthwiseSeparableConv", "InvertedResidual"]


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution (MobileNetV1).

    Tách conv chuẩn thành: depthwise (mỗi kênh một nhân) + pointwise (1x1 trộn kênh),
    giảm mạnh FLOPs/tham số so với conv thường.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        act: Optional[Union[str, nn.Module]] = "relu",
        norm_layer: Optional[type] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.dw = ConvNormAct(
            in_channels, in_channels, kernel_size, stride=stride,
            groups=in_channels, dilation=dilation, act=act, norm_layer=norm_layer,
        )
        self.pw = ConvNormAct(
            in_channels, out_channels, kernel_size=1, act=act, norm_layer=norm_layer,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """
    Inverted Residual / MBConv (MobileNetV2, V3, EfficientNet).

    Luồng: PW mở rộng (expand_ratio) -> DW (kxk, stride) -> [SE] -> PW chiếu (tuyến tính).
    Có residual khi `stride == 1` và `in_channels == out_channels`.

    Args:
        expand_ratio: hệ số mở rộng kênh ở nhánh giữa (1 = bỏ lớp mở rộng).
        use_se: gắn Squeeze-Excite sau depthwise.
        act: hàm kích hoạt cho 2 lớp đầu (lớp chiếu là tuyến tính).
        drop_path: xác suất stochastic depth cho nhánh residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: float = 6.0,
        use_se: bool = False,
        act: Union[str, nn.Module] = "relu6",
        norm_layer: Optional[type] = nn.BatchNorm2d,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        assert stride in (1, 2), f"stride phải là 1 hoặc 2, nhận {stride}"
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden_dim, kernel_size=1, act=act, norm_layer=norm_layer))
        # depthwise
        layers.append(ConvNormAct(
            hidden_dim, hidden_dim, kernel_size, stride=stride,
            groups=hidden_dim, act=act, norm_layer=norm_layer,
        ))
        if use_se:
            layers.append(SqueezeExcite(hidden_dim))
        # pointwise chiếu (tuyến tính, không activation)
        layers.append(ConvNormAct(hidden_dim, out_channels, kernel_size=1, act=None, norm_layer=norm_layer))

        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.drop_path(self.block(x))
        return self.block(x)
