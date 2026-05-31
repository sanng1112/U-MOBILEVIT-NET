import argparse
from typing import Optional, Tuple, Callable, Any

import torch
from torch import Tensor
from torch.nn import GroupNorm
from torch.nn.parameter import Parameter
from torch.nn.init import (
    xavier_normal_,
    xavier_uniform_,
    kaiming_normal_,
    kaiming_uniform_,
    zeros_
)

from cv_nets.layers.base_layer import BaseLayer
from cv_nets.layers.dropout import Dropout
from cv_nets.utils.config_helper import get_param

from cv_nets.utils.functional import separable_attention_forward

_INITIALIZERS = {
    "glorot_normal": xavier_normal_,
    "glorot_uniform": xavier_uniform_,
    "he_normal": kaiming_normal_,
    "he_uniform": kaiming_uniform_,
    "zeros": zeros_
}

def _get_initializer(name: str | Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    if isinstance(name, str):
        try:
            return _INITIALIZERS[name]
        except KeyError:
            raise ValueError(
                f"Valid initializers are {list(_INITIALIZERS.keys())}, "
                f"Got {name} instead."
            )
    return name


class SeparableAttention(BaseLayer):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        dropout_p: Optional[float] = None,
        bias: Optional[bool] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        initializer: Optional[str | Callable[[Tensor], Tensor]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        opts = opts or kwargs.get("opts", None)
        
        self.in_channels = get_param(opts, in_channels, "in_channels", None)
        self.dropout_p = get_param(opts, dropout_p, "attn_dropout", 0.1)
        use_bias = get_param(opts, bias, "attn_bias", True)
        self.k_channels = get_param(opts, k_channels, "k_channels", self.in_channels)
        self.v_channels = get_param(opts, v_channels, "v_channels", self.in_channels)
        init_val = get_param(opts, initializer, "attn_init", "he_uniform")

        if self.in_channels is None or self.in_channels <= 0: 
            raise ValueError(
                f"in_channels must be greater than 0, "
                f"Got in_channels={self.in_channels} instead."
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        
        self._qkv_same_channels = (self.k_channels == self.v_channels) and (self.in_channels == self.v_channels)
        self.initializer = _get_initializer(init_val)
        
        if not self._qkv_same_channels:
            self.q_proj_weight = Parameter(torch.empty((1, self.in_channels), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((self.in_channels, self.k_channels), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((self.in_channels, self.k_channels), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
            self.in_proj_weight = Parameter(torch.empty((1 + 2 * self.in_channels, self.in_channels), **factory_kwargs))
            
        self.out_proj_weight = Parameter(torch.empty((self.in_channels, self.in_channels), **factory_kwargs))
        
        if use_bias:
            self.in_proj_bias = Parameter(torch.empty(1 + 2 * self.in_channels, **factory_kwargs))
            self.out_proj_bias = Parameter(torch.empty(self.in_channels, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
            self.register_parameter('out_proj_bias', None)
        
        self._reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(f"Arguments for {cls.__name__}")
        group.add_argument("--attn-dropout", type=float, default=0.1, help="Xác suất Dropout trong Attention")
        group.add_argument("--attn-bias", action="store_true", help="Sử dụng bias cho In/Out Projection")
        group.add_argument("--k-channels", type=int, default=None, help="Số kênh cho Key (mặc định: in_channels)")
        group.add_argument("--v-channels", type=int, default=None, help="Số kênh cho Value (mặc định: in_channels)")
        group.add_argument(
            "--attn-init", 
            type=str, 
            default="he_uniform", 
            choices=["glorot_normal", "glorot_uniform", "he_normal", "he_uniform", "zeros"],
            help="Phương pháp khởi tạo trọng số cho Attention"
        )
        return parser
    
    def _reset_parameters(self) -> None:
        if self._qkv_same_channels:
            self.initializer(self.in_proj_weight)
        else:
            self.initializer(self.q_proj_weight)
            self.initializer(self.k_proj_weight)
            self.initializer(self.v_proj_weight)
        
        if self.in_proj_bias is not None:
            zeros_(self.in_proj_bias)
            zeros_(self.out_proj_bias)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_context_scores: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        
        assert query.dim() == 4, f"Expected query have 4 dimensions, got {query.shape}."
        assert key.dim() == 4, f"Expected key have 4 dimensions, got {key.shape}."
        assert value.dim() == 4, f"Expected value have 4 dimensions, got {value.shape}."
        
        return separable_attention_forward(
            query=query,
            key=key,
            value=value,
            channels_to_check=self.in_channels,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            dropout_p=self.dropout_p,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            use_separate_proj_weight=not self._qkv_same_channels,
            training=self.training,
            need_context_scores=need_context_scores
        )


class TransformerEncoderLayer(BaseLayer):
    def __init__(
        self, 
        in_channels: Optional[int] = None,
        dropout_p: Optional[float] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None,
        initializer: Optional[str | Callable[[Tensor], Tensor]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        opts = opts or kwargs.get("opts", None)
        
        self.in_channels = get_param(opts, in_channels, "in_channels", None)
        self.dropout_p = get_param(opts, dropout_p, "attn_dropout", 0.1)
        self.norm_num_groups = get_param(opts, norm_num_groups, "norm_num_groups", 1)
        self.bias = get_param(opts, bias, "attn_bias", True)
        self.init_str = get_param(opts, initializer, "attn_init", "he_uniform")

        if self.in_channels is None or self.in_channels <= 0: 
            raise ValueError(
                f"in_channels must be greater than 0, "
                f"Got in_channels={self.in_channels} instead."
            )
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.self_attn = SeparableAttention(
            opts=opts,
            in_channels=self.in_channels,
            dropout_p=self.dropout_p,
            bias=self.bias,
            initializer=self.init_str,
            **factory_kwargs
        )
        
        self.dropout_self_attn = Dropout(opts=opts, p=self.dropout_p)
        
        self.norm_self_attn = GroupNorm(
            num_groups=self.norm_num_groups, 
            num_channels=self.in_channels,
            **factory_kwargs
        )
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(f"Arguments for {cls.__name__}")
        group.add_argument("--norm-num-groups", type=int, default=1, help="Số nhóm cho GroupNorm trong Transformer Block")
        return parser

    def forward(self, input: Tensor) -> Tensor | Tuple[Tensor, Tensor]:
        assert input.dim() == 4, f"Transformer encoder block expected input have 4 dimensions, got {input.dim()}." 
        Z = self._sa_block(input)
        return Z 
        
    def _sa_block(self, input: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.self_attn(input, input, input)
        Z = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        Z = self.dropout_self_attn(Z)
        Z = self.norm_self_attn(Z + input)
        return Z


class TransformerDecoderLayer(BaseLayer):
    def __init__(
        self, 
        in_channels: Optional[int] = None, 
        dropout_p: Optional[float] = None, 
        norm_num_groups: Optional[int] = None, 
        bias: Optional[bool] = None, 
        initializer: Optional[str | Callable[[Tensor], Tensor]] = None, 
        opts: Optional[Any] = None,
        device=None, 
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        opts = opts or kwargs.get("opts", None)
        
        self.in_channels = get_param(opts, in_channels, "in_channels", None)
        self.dropout_p = get_param(opts, dropout_p, "attn_dropout", 0.1)
        self.norm_num_groups = get_param(opts, norm_num_groups, "norm_num_groups", 1)
        self.bias = get_param(opts, bias, "attn_bias", True)
        self.init_str = get_param(opts, initializer, "attn_init", "he_uniform")

        if self.in_channels is None or self.in_channels <= 0: 
            raise ValueError(f"in_channels must be greater than 0, Got in_channels={self.in_channels} instead.")
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.self_attn = SeparableAttention(
            opts=opts,
            in_channels=self.in_channels,
            dropout_p=self.dropout_p,
            bias=self.bias,
            initializer=self.init_str,
            **factory_kwargs
        )
        self.dropout_self_attn = Dropout(opts=opts, p=self.dropout_p)
        self.norm_self_attn = GroupNorm(num_groups=self.norm_num_groups, num_channels=self.in_channels, **factory_kwargs)
                
        self.cross_attn = SeparableAttention(
            opts=opts,
            in_channels=self.in_channels,
            dropout_p=self.dropout_p,
            bias=self.bias,
            initializer=self.init_str,
            **factory_kwargs
        )
        self.dropout_cross_attn = Dropout(opts=opts, p=self.dropout_p)
        self.norm_cross_attn = GroupNorm(num_groups=self.norm_num_groups, num_channels=self.in_channels, **factory_kwargs)
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def forward(
        self,
        input: Tensor,
        memory: Tensor,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        assert input.dim() == 4, f"Transformer decoder block expected input have 4 dimensions, got {input.dim()}."
        assert memory.dim() == 4, f"Transformer decoder block expected memory have 4 dimensions, got {memory.dim()}." 
        
        Z = self._sa_block(input) 
        Z = self._ca_block(Z, memory) 
        return Z 
   
    def _sa_block(self, input: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.self_attn(input, input, input)
        Z = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        Z = self.dropout_self_attn(Z)
        Z = self.norm_self_attn(Z + input)
        return Z  
        
    def _ca_block(self, input: Tensor, memory: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output = self.cross_attn(input, memory, memory)
        Z = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        Z = self.dropout_cross_attn(Z)
        Z = self.norm_cross_attn(Z + input)
        return Z