import argparse
from typing import Optional, Tuple, Callable, Any, List

import torch
from torch import Tensor
from torch.nn import (
    ModuleList,
    Sequential,
    ReLU,
    GroupNorm,
    Upsample,
)
from torch.nn.init import zeros_

# Hệ sinh thái Custom Layers của bạn
from cv_nets.layers.base_layer import BaseLayer
from cv_nets.layers.conv_layer import Conv2d
from cv_nets.utils.config_helper import get_param

# Các module mô hình gốc
from models.u_mobilevit_net.transfomer import (
    TransformerDecoderLayer,
    _get_initializer
)
from models.u_mobilevit_net.module import (
    _UMobileViTLayer,
    _get_clones
)
from cv_nets.utils.functional import (
    unfold_custom,
    fold_custom
)


class UMobileViTDecoderLayer(_UMobileViTLayer):
    def __init__(self, opts: Optional[Any] = None, **kwargs) -> None:
        """
        Decoder layer of UMobileViT is made up of Transformer decoder blocks,
        local block made of convolutional layers, and out normalization layer.
        """
        self.opts = opts or kwargs.get("opts", None)
        kwargs["opts"] = self.opts
        transformer_block = TransformerDecoderLayer
        super().__init__(transformer_block, **kwargs)
        
    def forward(self, input: Tensor, memory: Tensor) -> Tensor:
        # check integrity: features sizes must be divisible by patch sizes respectively
        assert input.dim() == 4, f"Decoder block expected input have 4 dimensions, got {input.dim()}." 
        assert memory.dim() == 4, f"Decoder block expected memory have 4 dimensions, got {memory.dim()}." 
        
        N, C, *input_size = input.shape
        patch_h, patch_w = self.fold_params["kernel_size"]
        
        assert (
            input_size[0] % patch_h == 0 and input_size[1] % patch_w == 0
        ), f"Height and width of feature map must be divisible by patch sizes." 
        
        # local block forward
        Z = self.local_block(input) # (N, C, H, W)
        
        # global block forward
        # unfolding        
        Z = unfold_custom(Z, kernel_size=self.fold_params["kernel_size"]) # (N, C, P, S)
        memory_unfolded = unfold_custom(memory, kernel_size=self.fold_params["kernel_size"]) # (N, C, P, S)
        
        for block in self.global_block: 
            Z = block(Z, memory_unfolded)
            
        # folding
        Z = fold_custom(Z, output_size=input_size, kernel_size=self.fold_params["kernel_size"]) # (N, C, H, W)
        
        # expansion block forward
        Z = self.expansion_block(Z)
        
        # return normalized residual connection
        return self.out_norm(Z + input)


class UMobileViTDecoderConcatLayer(_UMobileViTLayer):
    def __init__(self, opts: Optional[Any] = None, **kwargs) -> None:
        """
        Implement the custom global block. Instead of performing self-attention and 
        cross-attention, this global block will project the encoder feature-added feature map
        to the model space.
        """
        self.opts = opts or kwargs.get("opts", None)
        kwargs["opts"] = self.opts
        super().__init__(transformer_block=None, **kwargs)
        
        in_channels = kwargs.get("in_channels")
        bias = kwargs.get("bias", True)
        
        self.memory_proj = Sequential(
            Conv2d(
                opts=self.opts,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            ReLU(inplace=True),
        )
        
        self.global_block = Sequential(
            Conv2d(
                opts=self.opts,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=bias,
            ),
            ReLU(inplace=True),
            Conv2d(
                opts=self.opts,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            ReLU(inplace=True),
        )

    def _reset_parameters(self) -> None:
        super()._reset_parameters()
        for layer in self.global_block:
            if isinstance(layer, Conv2d):
                self.initializer(layer.weight)
                if getattr(layer, "bias", None) is not None:
                    zeros_(layer.bias)
    
    def forward(self, input: Tensor, memory: Tensor) -> Tensor:
        assert input.dim() == 4, f"Decoder block expected input have 4 dimensions, got {input.dim()}." 
        assert memory.dim() == 4, f"Decoder block expected memory have 4 dimensions, got {memory.dim()}." 
        assert (
            input.size(1) == memory.size(1)
        ), f"Decoder concatenative block expected input and memory have the same channels, got {input.size(1)} and {memory.size(1)}"
        
        # local block forward
        Z = self.local_block(input) # (N, C, H, W)
        
        # global block forward (Additive memory projection)
        Z = Z + self.memory_proj(memory)
        Z = self.global_block(Z)    
        
        # expansion block forward
        Z = self.expansion_block(Z)
        
        # return normalized residual connection
        return self.out_norm(Z + input)    


def _get_upsample_block(initializer, opts: Optional[Any] = None, **kwargs) -> Sequential:
    block = Sequential(
        Upsample(
            scale_factor=kwargs.get("scale_factor", 2),
            mode=kwargs.get("mode", "nearest")
        ),
        Conv2d(
            opts=opts,
            in_channels=kwargs.get("in_channels"),
            out_channels=kwargs.get("out_channels"),
            kernel_size=kwargs.get("kernel_size", 3),
            stride=kwargs.get("stride", 1),
            padding=kwargs.get("padding", 1),
            groups=kwargs.get("groups", 1),
            bias=kwargs.get("bias", False),
        ),
        ReLU(inplace=True),
    )
    
    initializer(block[1].weight)
    if getattr(block[1], "bias", None) is not None:
        zeros_(block[1].bias)
        
    return block


class UMobileViTDecoder(BaseLayer):
    def __init__(
        self,  
        d_model: Optional[int] = None,
        expansion_factor: Optional[float] = None,
        patch_size: Optional[int | Tuple[int, int]] = None,
        dropout_p: Optional[float] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None, 
        num_transformer_block: Optional[int] = None,
        initializer: Optional[str | Callable[[Tensor], Tensor]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        opts = opts or kwargs.get("opts", None)
        
        self.d_model = get_param(opts, d_model, "d_model", 64)
        self.expansion_factor = get_param(opts, expansion_factor, "expansion_factor", 3.0)
        self.patch_size = get_param(opts, patch_size, "patch_size", 2)
        self.dropout_p = get_param(opts, dropout_p, "dropout_p", 0.1)
        self.norm_num_groups = get_param(opts, norm_num_groups, "norm_num_groups", 4)
        self.bias = get_param(opts, bias, "bias", True)
        self.init_str = get_param(opts, initializer, "initializer", "he_uniform")
        self.num_transformer_block = get_param(opts, num_transformer_block, "num_transformer_blocks", 2)

        assert self.d_model > 0, f"d_model must be greater than zero, got d_model={self.d_model}"
        assert self.num_transformer_block > 0, f"num_transformer_block must be greater than zero, got num_transformer_block={self.num_transformer_block}"
        
        self.initializer = _get_initializer(self.init_str)
        factory_kwargs = {"device": device, "dtype": dtype}
        
        decoder_layer_kwargs = {
            "in_channels": self.d_model,
            "expansion_factor": self.expansion_factor,
            "patch_size": self.patch_size,
            "dropout_p": self.dropout_p,
            "norm_num_groups": self.norm_num_groups,
            "bias": self.bias,
            "num_transformer_block": self.num_transformer_block,
            "initializer": self.init_str,
            "opts": opts 
        }
        upsampling_kwargs = {
            "scale_factor": 2,
            "mode": "nearest"
        }
        upsampling_conv_kwargs = {
            "in_channels": self.d_model,
            "out_channels": self.d_model,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "groups": self.d_model,
            "bias": self.bias,
        }
        decoder_layer = UMobileViTDecoderLayer
        
        # first stage block - upsample factor = 2
        stage_1 = ModuleList([
            _get_upsample_block(
                self.initializer,
                opts=opts,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
            ),
            *_get_clones(
                decoder_layer, 
                N=2,
                **decoder_layer_kwargs, 
                **factory_kwargs
            ),
        ])
        
        # second stage block - upsample factor = 4
        stage_2 = ModuleList([
            _get_upsample_block(
                self.initializer,
                opts=opts,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
            ),
            *_get_clones(
                decoder_layer, 
                N=1,
                **decoder_layer_kwargs, 
                **factory_kwargs
            ),
        ])
        
        # out block - upsample factor = 8
        out_block = ModuleList([
            _get_upsample_block(
                self.initializer,
                opts=opts,
                **upsampling_kwargs, 
                **upsampling_conv_kwargs,
            ),
            UMobileViTDecoderConcatLayer(**decoder_layer_kwargs, **factory_kwargs)
        ])
        
        self.layers = ModuleList([
            stage_1,
            stage_2,
            out_block,
        ])
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(f"Arguments for {cls.__name__}")
        # Hầu hết các arguments đã được xử lý chung bởi Encoder
        return parser

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        """
        Args:
            inputs (Tuple[Tensor, ...]): features of each stage from encoder. Must be arranged
            backward, i.e feature of the first stage must be put in the last, and the second 
            stage is put in the second last, and so forth.
        """
        assert (
            len(inputs) - 1 == len(self.layers)
        ), f"number of inputs is expected to be equal to {len(self.layers) + 1}, got {len(inputs)}"
        
        Z = inputs[0] # get the latent tensor
        for i, memory in enumerate(inputs[1:]): 
            Z = self.layers[i][0](Z) # upsample
            for layer in self.layers[i][1:]: # cross attention / additive layer
                Z = layer(Z, memory)
        
        return Z