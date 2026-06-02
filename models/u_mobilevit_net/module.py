import argparse
import math
from typing import (
    Tuple,
    Callable,
    Any,
    List,
    Union,
    Optional,
    Type
)

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import (
    ModuleList,
    Sequential,
    ReLU,
    ReLU6,
    GroupNorm,
    Identity
)


# ═══════════════════════════════════════════════════════════════
# GroupNorm Helper — tự động chọn num_groups tương thích
# ═══════════════════════════════════════════════════════════════

def get_groupnorm_groups(num_channels: int, target_groups: int = 4) -> int:
    """Tìm num_groups tương thích nhất với GroupNorm.

    GroupNorm yêu cầu num_channels % num_groups == 0.
    Khi scale model (nano/pro/promax), channel count có thể không
    chia hết cho target_groups — hàm này tự động chọn ước số
    phù hợp nhất.

    Args:
        num_channels: Số channels hiện tại
        target_groups: Số groups mong muốn (mặc định: 4)

    Returns:
        Số groups hợp lệ, được chọn theo thứ tự ưu tiên:
        1. target_groups nếu num_channels % target_groups == 0
        2. Ước số lớn nhất của num_channels mà ≤ target_groups
        3. Ước số nhỏ nhất của num_channels mà ≥ target_groups
        4. 1 (fallback)
    """
    if num_channels % target_groups == 0:
        return target_groups

    # Tìm tất cả ước số của num_channels
    divisors = set()
    for i in range(1, int(math.sqrt(num_channels)) + 1):
        if num_channels % i == 0:
            divisors.add(i)
            divisors.add(num_channels // i)

    divisors = sorted(divisors)

    # Ưu tiên ước số lớn nhất ≤ target_groups
    smaller = [d for d in divisors if d <= target_groups]
    if smaller:
        return max(smaller)

    # Fallback: ước số nhỏ nhất ≥ target_groups
    larger = [d for d in divisors if d >= target_groups]
    if larger:
        return min(larger)

    return 1
from torch.nn.modules.utils import _pair
from torch.nn.init import zeros_

# Hệ sinh thái Custom Layers của bạn
from cv_nets.layers.base_layer import BaseLayer
from cv_nets.layers.conv_layer import Conv2d
from cv_nets.layers.dropout import Dropout
from cv_nets.utils.config_helper import get_param

# Modules nội bộ dự án
from models.u_mobilevit_net.transfomer import (
    TransformerEncoderLayer, 
    TransformerDecoderLayer,
    _get_initializer
)

def _get_clones(
    module_class: Type[Any], 
    N: int,
    **kwargs
) -> List[Any]:
    """Helper method để tạo list N lớp giống hệt nhau."""
    return [module_class(**kwargs) for _ in range(N)]


def _get_local_block(
    opts: Optional[Any],
    in_channels: int,
    norm_num_groups: int = 1,
    bias: bool = True,
    **factory_kwargs
) -> Sequential:
    """Tạo Local Block với Convolution."""
    effective_groups = get_groupnorm_groups(in_channels, norm_num_groups)
    block = Sequential(
        Conv2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=bias,
        ),
        ReLU(inplace=True),
        Conv2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            bias=bias,
        ),
        ReLU(inplace=True),
        GroupNorm(
            num_groups=effective_groups,
            num_channels=in_channels,
            **factory_kwargs
        )
    )
    return block


def _get_expansion_block(
    opts: Optional[Any],
    in_channels: int,
    expansion_factor: float,
    dropout_p: float,
    bias: bool = True,
    norm_num_groups: int = 4,
    **factory_kwargs
) -> Sequential:
    """Tạo khối mở rộng Inverted Bottleneck (MobileNetV2 style).

    [FIX NaN] Không có normalization + ReLU không giới hạn = activation
    có thể tăng không kiểm soát → Inf trong attention → NaN gradient.

    Fix: thêm GroupNorm sau mỗi conv (giống MobileNetV2 gốc dùng BN)
    + dùng ReLU6 giới hạn activation trong [0, 6].
    """
    expanded_channels = int(expansion_factor * in_channels)
    gn1 = get_groupnorm_groups(expanded_channels, norm_num_groups)
    gn2 = get_groupnorm_groups(expanded_channels, norm_num_groups)
    gn3 = get_groupnorm_groups(in_channels, norm_num_groups)

    block = Sequential(
        # Pointwise expand
        Conv2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        ),
        GroupNorm(num_groups=gn1, num_channels=expanded_channels, **factory_kwargs),
        ReLU6(inplace=True),

        # Depthwise
        Conv2d(
            opts=opts,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=expanded_channels,
            bias=bias,
        ),
        GroupNorm(num_groups=gn2, num_channels=expanded_channels, **factory_kwargs),
        ReLU6(inplace=True),

        # Pointwise Projection (linear — không activation, giống MobileNetV2)
        Conv2d(
            opts=opts,
            in_channels=expanded_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        ),
        GroupNorm(num_groups=gn3, num_channels=in_channels, **factory_kwargs),
        Dropout(opts=opts, p=dropout_p)
    )
    return block


class _UMobileViTLayer(BaseLayer):
    def __init__(
        self, 
        transformer_block: Union[Type[TransformerEncoderLayer], Type[TransformerDecoderLayer], None],
        in_channels: Optional[int] = None, 
        expansion_factor: Optional[float] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        dropout_p: Optional[float] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None,
        num_transformer_block: Optional[int] = None,
        initializer: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        """
        A building layer of UMobileViT is made up of Transformer encoder/decoder blocks,
        local block made of convolutional layers, and out normalization layer.
        """
        super().__init__(*args, **kwargs)
        
        opts = opts or kwargs.get("opts", None)
        
        self.in_channels = get_param(opts, in_channels, "in_channels", None)
        self.expansion_factor = get_param(opts, expansion_factor, "expansion_factor", 3.0)
        self.patch_size = get_param(opts, patch_size, "patch_size", 2)
        self.dropout_p = get_param(opts, dropout_p, "dropout_p", 0.1)
        self.norm_num_groups = get_param(opts, norm_num_groups, "norm_num_groups", 4)
        self.bias = get_param(opts, bias, "bias", True)
        self.num_transformer_block = get_param(opts, num_transformer_block, "num_transformer_blocks", 1)
        self.init_str = get_param(opts, initializer, "initializer", "he_uniform")

        assert self.num_transformer_block > 0, f"num_transformer_block must be > 0, got {self.num_transformer_block}"
        if self.in_channels is None or self.in_channels <= 0: 
            raise ValueError(f"in_channels must be > 0, got {self.in_channels}")
            
        patch_size_tuple = _pair(self.patch_size)
        assert all(size > 0 for size in patch_size_tuple), "patch size elements must be greater than zero"
        assert self.expansion_factor >= 1, f"expansion_factor must be >= 1, got {self.expansion_factor}"
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fold_params = {"kernel_size": patch_size_tuple, "stride": patch_size_tuple}
        self.initializer = _get_initializer(self.init_str)
        
        # global block made of transformer blocks
        if transformer_block is not None:
            # Auto-compute compatible norm groups for transformer
            effective_gn = get_groupnorm_groups(
                self.in_channels, self.norm_num_groups
            )
            transformer_block_kwargs = {
                "in_channels": self.in_channels,
                "dropout_p": self.dropout_p,
                "norm_num_groups": effective_gn,
                "bias": self.bias,
                "initializer": self.init_str,
                "opts": opts
            }
            
            global_block = _get_clones(
                transformer_block, 
                N=self.num_transformer_block,
                **transformer_block_kwargs, 
                **factory_kwargs
            )
            self.global_block = ModuleList(global_block)
            
            # local block is depthwise separable convolution, followed by a group norm layer
            self.local_block = _get_local_block(
                opts=opts,
                in_channels=self.in_channels,
                norm_num_groups=self.norm_num_groups,
                bias=self.bias,
                **factory_kwargs
            )
            
            # expansion block implementation, inspired by MobileNetV2 block
            self.expansion_block = _get_expansion_block(
                opts=opts,
                in_channels=self.in_channels,
                expansion_factor=self.expansion_factor,
                dropout_p=self.dropout_p,
                bias=self.bias,
                norm_num_groups=self.norm_num_groups,
                **factory_kwargs
            )
        else:
            self.local_block = Identity()
            self.global_block = ModuleList([])
            self.expansion_block = Dropout(opts=opts, p=self.dropout_p)
                
        # out normalization — auto-select compatible num_groups
        effective_out_groups = get_groupnorm_groups(
            self.in_channels, self.norm_num_groups
        )
        self.out_norm = GroupNorm(
            num_groups=effective_out_groups,
            num_channels=self.in_channels,
            **factory_kwargs
        )
        
        self._reset_parameters()

        # ── Gradient checkpointing flag ──
        # Có thể bị ghi đè bởi SegmentationTrainer để tắt checkpointing
        # khi model bất ổn định (NaN). Mặc định BẬT cho tiết kiệm VRAM.
        self.use_checkpointing = True

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(f"Arguments for {cls.__name__}")
        # Hầu hết các arguments đã được kế thừa từ Encoder/Decoder
        return parser

    def _reset_parameters(self) -> None:
        if not isinstance(self.local_block, Identity):
            for layer in self.local_block:
                if isinstance(layer, Conv2d):
                    self.initializer(layer.weight)
                    if getattr(layer, "bias", None) is not None:
                        zeros_(layer.bias)

        if not isinstance(self.expansion_block, Dropout):
            for layer in self.expansion_block:
                if isinstance(layer, Conv2d):
                    self.initializer(layer.weight)
                    if getattr(layer, "bias", None) is not None:
                        zeros_(layer.bias)

    # ── Gradient Checkpointing helpers (chống OOM trên GPU nhỏ) ──

    def _global_forward(self, Z: Tensor, *extra_args: Tensor) -> Tensor:
        """Forward global block — tách riêng để `torch.utils.checkpoint` có thể recompute.

        Khi wrapped với checkpoint(): intermediate activations của transformer blocks
        KHÔNG được lưu → backward sẽ tính lại từ đầu → tiết kiệm 30-40% VRAM.
        """
        for block in self.global_block:
            if extra_args and extra_args[0] is not None:
                Z = block(Z, *extra_args)
            else:
                Z = block(Z)
        return Z

    def _expansion_forward(self, Z: Tensor) -> Tensor:
        """Forward expansion block — tách riêng để checkpoint có thể recompute.

        Đây là phần nặng nhất: expansion_factor=4.0 tạo tensor (B, C*4, H, W)
        → ~1.6 GB cho promax ở độ phân giải 160×160. Checkpointing giúp
        không phải lưu activations khổng lồ này.
        """
        return self.expansion_block(Z)
