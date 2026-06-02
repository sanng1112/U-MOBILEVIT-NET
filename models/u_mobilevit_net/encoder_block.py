import argparse
from typing import Optional, Tuple, Callable, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import (
    ModuleList,
    Sequential,
    ReLU,
    GroupNorm,
)
from torch.nn.init import zeros_

from cv_nets.layers.base_layer import BaseLayer
from cv_nets.layers.conv_layer import Conv2d
from cv_nets.utils.config_helper import get_param

from models.u_mobilevit_net.transfomer import (
    TransformerEncoderLayer, 
    _get_initializer
)
from models.u_mobilevit_net.module import (
    _UMobileViTLayer,
    _get_clones,
    get_groupnorm_groups,
)
from cv_nets.utils.functional import (
    unfold_custom,
    fold_custom
)


class UMobileViTEncoderLayer(_UMobileViTLayer):
    def __init__(self, opts: Optional[Any] = None, **kwargs) -> None:
        """
        Encoder layer of UMobileViT is made up of Transformer encoder blocks,
        local block made of convolutional layers, and out normalization layer.
        """
        self.opts = opts or kwargs.get("opts", None)
        transformer_block = TransformerEncoderLayer
        
        kwargs["opts"] = self.opts
        super().__init__(transformer_block, **kwargs)
    
    def forward(self, input: Tensor) -> Tensor:
        assert input.dim() == 4, f"Encoder block expected input have 4 dimensions, got {input.dim()}."
        N, C, *input_size = input.shape

        Z = self._local_forward(input)

        # Pad spatial dims to be divisible by patch_size (fixes odd-size inputs like 23)
        patch_h, patch_w = self.fold_params["kernel_size"]
        H, W = Z.shape[2], Z.shape[3]
        pad_h = (patch_h - H % patch_h) % patch_h
        pad_w = (patch_w - W % patch_w) % patch_w

        if pad_h > 0 or pad_w > 0:
            Z = F.pad(Z, (0, pad_w, 0, pad_h))

        padded_size = (H + pad_h, W + pad_w)

        Z = unfold_custom(Z, kernel_size=self.fold_params["kernel_size"])

        # ── Global block (transformer) — checkpoint khi training ──
        if self.training and self.use_checkpointing:
            Z = torch_checkpoint(self._global_forward, Z, None, use_reentrant=False)
        else:
            for block in self.global_block:
                Z = block(Z)

        Z = fold_custom(Z, output_size=padded_size, kernel_size=self.fold_params["kernel_size"])

        # Crop back to original spatial size
        if pad_h > 0 or pad_w > 0:
            Z = Z[:, :, :H, :W]

        # ── Expansion block — checkpoint khi training (nặng nhất) ──
        if self.training and self.use_checkpointing:
            Z = torch_checkpoint(self._expansion_forward, Z, use_reentrant=False)
        else:
            Z = self.expansion_block(Z)

        return self.out_norm(Z + input)


def _get_downsample_block(initializer, opts: Optional[Any] = None, **kwargs) -> Sequential:
    block = Sequential(
        Conv2d(
            opts=opts,
            in_channels=kwargs.get("in_channels"),
            out_channels=kwargs.get("out_channels"),
            kernel_size=kwargs.get("kernel_size", 3),
            stride=kwargs.get("stride", 2),
            padding=kwargs.get("padding", 1),
            groups=kwargs.get("groups", 1),
            bias=kwargs.get("bias", False),
        ),
        ReLU(inplace=True),
    )
    
    initializer(block[0].weight)
    if getattr(block[0], "bias", None) is not None:
        zeros_(block[0].bias)
        
    return block


class UMobileViTEncoder(BaseLayer):
    def __init__(
        self, 
        in_channels: Optional[int] = None,
        d_model: Optional[int] = None,
        expansion_factor: Optional[float] = None,
        patch_size: Optional[int | Tuple[int, int]] = None,
        dropout_p: Optional[float] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None, 
        initializer: Optional[str | Callable[[Tensor], Tensor]] = None,
        num_transformer_block: Optional[int] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        opts = opts or kwargs.get("opts", None)
        
        self.in_channels = get_param(opts, in_channels, "in_channels", 3)
        self.d_model = get_param(opts, d_model, "d_model", 64)
        self.expansion_factor = get_param(opts, expansion_factor, "expansion_factor", 3.0)
        self.patch_size = get_param(opts, patch_size, "patch_size", 2)
        self.dropout_p = get_param(opts, dropout_p, "dropout_p", 0.1)
        self.norm_num_groups = get_param(opts, norm_num_groups, "norm_num_groups", 4)
        self.bias = get_param(opts, bias, "bias", True)
        self.init_str = get_param(opts, initializer, "initializer", "he_uniform")
        self.num_transformer_block = get_param(opts, num_transformer_block, "num_transformer_blocks", 2)

        assert self.in_channels > 0, f"in_channels must be > 0, got {self.in_channels}"
        assert self.d_model > 0, f"d_model must be > 0, got {self.d_model}"
        assert self.num_transformer_block > 0, f"num_transformer_block must be > 0, got {self.num_transformer_block}"
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(self.init_str)
        
        stem_conv_kwargs = {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "bias": self.bias,
        }
        
        # Auto-compute compatible GroupNorm groups for each stem stage
        gn1 = get_groupnorm_groups(self.d_model // 2, self.norm_num_groups)
        gn2 = get_groupnorm_groups(self.d_model, self.norm_num_groups)

        self.stem_block = ModuleList([
            Sequential(
                Conv2d(opts=opts, in_channels=self.in_channels, out_channels=self.d_model//4, **stem_conv_kwargs),
                ReLU(inplace=True),
            ),
            Sequential(
                Conv2d(opts=opts, in_channels=self.d_model//4, out_channels=self.d_model//2, **stem_conv_kwargs),
                ReLU(inplace=True),
                GroupNorm(num_groups=gn1, num_channels=self.d_model//2, **factory_kwargs)
            ),
            Sequential(
                Conv2d(opts=opts, in_channels=self.d_model//2, out_channels=self.d_model, **stem_conv_kwargs),
                ReLU(inplace=True),
                GroupNorm(num_groups=gn2, num_channels=self.d_model, **factory_kwargs)
            )
        ])
        
        encoder_layer_kwargs = {
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
        downsampling_block_kwargs = {
            "in_channels": self.d_model,
            "out_channels": self.d_model,
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "groups": self.d_model, 
            "bias": self.bias,
        }
        stage_1 = Sequential(
            _get_downsample_block(self.initializer, opts=opts, **downsampling_block_kwargs),
            *_get_clones(UMobileViTEncoderLayer, N=1, **encoder_layer_kwargs, **factory_kwargs),
        )
        
        stage_2 = Sequential(
            _get_downsample_block(self.initializer, opts=opts, **downsampling_block_kwargs),
            *_get_clones(UMobileViTEncoderLayer, N=2, **encoder_layer_kwargs, **factory_kwargs),
        )
        
        encoder_layer_kwargs_s3 = encoder_layer_kwargs.copy()
        encoder_layer_kwargs_s3["patch_size"] = (1, 1)
        
        stage_3 = Sequential(
            _get_downsample_block(self.initializer, opts=opts, **downsampling_block_kwargs),
            *_get_clones(UMobileViTEncoderLayer, N=2, **encoder_layer_kwargs_s3, **factory_kwargs),
        )
        
        self.layers = ModuleList([stage_1, stage_2, stage_3])
        self._reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(f"Arguments for {cls.__name__}")
        group.add_argument("--d-model", type=int, default=64, help="Kích thước kênh đặc trưng (d_model) của Encoder")
        group.add_argument("--expansion-factor", type=float, default=3.0, help="Hệ số mở rộng cho block Feed-Forward")
        group.add_argument("--patch-size", type=int, default=2, help="Kích thước ô patch dùng để unfold")
        group.add_argument("--num-transformer-blocks", type=int, default=2, help="Số lượng Transformer blocks trong mỗi Stage")
        return parser
        
    def _reset_parameters(self) -> None:
        for layer in self.stem_block:
            for component in layer:
                if isinstance(component, Conv2d):
                    self.initializer(component.weight)
                    if getattr(component, "bias", None) is not None:
                        zeros_(component.bias)
        
        for layer in self.layers:
            for component in layer:
                if isinstance(component, Conv2d):
                    self.initializer(component.weight)
                    if getattr(component, "bias", None) is not None:
                        zeros_(component.bias)
            
    def forward(self, input: Tensor) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        """
        Return features at each stage, including stem stage.
        """
        stem_outputs = []
        stage_outputs = []
        
        Z = input
        for layer in self.stem_block[:-1]:
            Z = layer(Z)
            stem_outputs.append(Z)
            
        # Cuối stem_block
        Z = self.stem_block[-1](Z)
        stage_outputs.append(Z)
        
        for layer in self.layers:
            Z = layer(Z)
            stage_outputs.append(Z)
            
        return tuple(stem_outputs), tuple(stage_outputs)