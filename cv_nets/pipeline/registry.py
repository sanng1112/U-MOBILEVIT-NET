from typing import Any, Callable, Dict
import inspect
from torch import nn

from cv_nets.blocks import (
    ConvBNAct, MV2Block, ResNetBasicBlock, ConvNormAct,
    DepthwiseSeparableConv, InvertedResidual,
    PatchEmbed, MultiHeadSelfAttention, Mlp, TransformerEncoderBlock,
    MobileViTBlock, INLATransformerBlock,
    SqueezeExcite, ECA, CBAM,
)
from cv_nets.layers import Conv2d, LinearLayer, Dropout, Flatten

BLOCK_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "ConvBNAct": ConvBNAct,
    "MV2Block": MV2Block,
    "ResNetBasicBlock": ResNetBasicBlock,
    "ConvNormAct": ConvNormAct,
    "DepthwiseSeparableConv": DepthwiseSeparableConv,
    "InvertedResidual": InvertedResidual,
    "PatchEmbed": PatchEmbed,
    "TransformerEncoderBlock": TransformerEncoderBlock,
    "MobileViTBlock": MobileViTBlock,
    "INLATransformerBlock": INLATransformerBlock,
    "SqueezeExcite": SqueezeExcite,
    "ECA": ECA,
    "CBAM": CBAM,
    "Conv2d": Conv2d,
    "LinearLayer": LinearLayer,
    "FC": LinearLayer,
    "Dropout": Dropout,
    "Flatten": Flatten,
}


def filter_params(cls: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return params
    valid = set(sig.parameters.keys())
    valid -= {"self", "args", "kwargs"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return params
    return {k: v for k, v in params.items() if k in valid}
