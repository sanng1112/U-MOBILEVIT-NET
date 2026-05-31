import argparse
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
from torch.nn import (
    ModuleList,
    Sequential,
    ReLU,
    GroupNorm,
    Identity
)
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
            num_groups=norm_num_groups,
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
    **factory_kwargs
) -> Sequential:
    """Tạo khối mở rộng Inverted Bottleneck (MobileNetV2 style)."""
    expanded_channels = int(expansion_factor * in_channels)
    block = Sequential(
        # Pointwise
        Conv2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        ),
        ReLU(inplace=True),
        
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
        ReLU(inplace=True),
        
        # Pointwise Projection
        Conv2d(
            opts=opts,
            in_channels=expanded_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        ),
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
        self.norm_num_groups = get_param(opts, norm_num_groups, "norm_num_groups", 1)
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
            transformer_block_kwargs = {
                "in_channels": self.in_channels, 
                "dropout_p": self.dropout_p,
                "norm_num_groups": self.norm_num_groups,
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
                **factory_kwargs
            )
        else:
            self.local_block = Identity()
            self.global_block = ModuleList([])
            self.expansion_block = Dropout(opts=opts, p=self.dropout_p)
                
        # out normalization
        self.out_norm = GroupNorm(
            num_groups=self.norm_num_groups,
            num_channels=self.in_channels,
            **factory_kwargs
        )
        
        self._reset_parameters()

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
