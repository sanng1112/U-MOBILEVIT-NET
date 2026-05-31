"""
Tiện ích dùng chung cho thư viện block (`cv_nets.blocks`).

Các block trong thư viện này được viết bằng `torch.nn` thuần để dễ đọc, dễ test và
không phụ thuộc vào hệ thống `opts`/registry của framework — có thể dùng trực tiếp
với tham số số nguyên thông thường. Tên hàm kích hoạt được phân giải qua
`get_activation` nên vẫn linh hoạt cấu hình bằng chuỗi.
"""
from typing import Optional, Union, Tuple

import torch
from torch import nn, Tensor


__all__ = [
    "make_divisible",
    "autopad",
    "get_activation",
    "ConvNormAct",
    "DropPath",
    "drop_path",
]


def make_divisible(value: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Làm tròn số kênh về bội số của `divisor` (chuẩn MobileNet/EfficientNet) để
    thân thiện phần cứng. Đảm bảo không giảm quá 10% so với giá trị gốc.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def autopad(kernel_size: int, dilation: int = 1) -> int:
    """Padding 'same' cho conv stride=1: giữ nguyên kích thước không gian."""
    return (kernel_size - 1) // 2 * dilation


# Bảng ánh xạ tên -> lớp kích hoạt (mở rộng dễ dàng).
_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
    "swish": lambda: nn.SiLU(inplace=True),
    "gelu": lambda: nn.GELU(),
    "hardswish": lambda: nn.Hardswish(inplace=True),
    "hardsigmoid": lambda: nn.Hardsigmoid(inplace=True),
    "mish": lambda: nn.Mish(inplace=True),
    "sigmoid": lambda: nn.Sigmoid(),
    "tanh": lambda: nn.Tanh(),
    "identity": lambda: nn.Identity(),
    "none": lambda: nn.Identity(),
}


def get_activation(name: Optional[Union[str, nn.Module]]) -> nn.Module:
    """
    Trả về module kích hoạt từ tên chuỗi. Nếu truyền sẵn `nn.Module` thì giữ nguyên;
    `None` -> `nn.Identity`.
    """
    if name is None:
        return nn.Identity()
    if isinstance(name, nn.Module):
        return name
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Activation '{name}' chưa hỗ trợ. Các lựa chọn: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


class ConvNormAct(nn.Module):
    """
    Khối Conv -> Norm -> Activation tiêu chuẩn (tự động padding 'same').

    Args:
        in_channels, out_channels: số kênh vào/ra.
        kernel_size: kích thước nhân (mặc định 3).
        stride: bước trượt.
        groups: số nhóm conv (đặt = in_channels để có depthwise).
        dilation: độ giãn nở.
        norm_layer: lớp chuẩn hóa (mặc định BatchNorm2d); `None` để bỏ.
        act: tên hàm kích hoạt hoặc module; `None`/"identity" để bỏ.
        bias: dùng bias cho conv (mặc định tắt khi có norm).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[type] = nn.BatchNorm2d,
        act: Optional[Union[str, nn.Module]] = "relu",
        bias: Optional[bool] = None,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if bias is None:
            bias = norm_layer is None
        if padding is None:
            padding = autopad(kernel_size, dilation)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self.act = get_activation(act)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Stochastic Depth: ngẫu nhiên bỏ (zero-out) toàn bộ nhánh residual theo từng mẫu
    trong batch. Giúp regularize mạng sâu. Khi không train hoặc `drop_prob == 0` thì
    trả về nguyên vẹn.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast theo batch
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Bọc `drop_path` thành module (Stochastic Depth)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
