"""
U-MobileViT-Net: Lightweight U-Net style semantic segmentation with
MobileViTv2 separable self-attention transformer blocks.

Submodules:
    - configs:      Variant definitions (nano/base/pro/promax) and helpers
    - u_models:     UMobileViT model wrapper and factory functions
    - encoder_block: Encoder with 3-stage stem + 3 MobileViT stages
    - decoder_block: Decoder with 3-stage upsample + cross-attention
    - module:       Core _UMobileViTLayer building block + utility helpers
    - transfomer:   SeparableAttention, TransformerEncoderLayer, TransformerDecoderLayer
    - seg_head:     UpsampleHead, FeatureRefinementBlock, ContextAwareSegHead, MultiTaskDenseHead
"""

from models.u_mobilevit_net.configs import (
    UMOBILEVIT_VARIANTS,
    get_variant,
    list_variants,
    get_stem_channels,
    get_seghead_channels,
)

from models.u_mobilevit_net.u_models import (
    UMobileViT,
    umobilevit,
    umobilevit_nano,
    umobilevit_base,
    umobilevit_pro,
    umobilevit_promax,
)

from models.u_mobilevit_net.encoder_block import (
    UMobileViTEncoder,
    UMobileViTEncoderLayer,
)

from models.u_mobilevit_net.decoder_block import (
    UMobileViTDecoder,
    UMobileViTDecoderLayer,
    UMobileViTDecoderConcatLayer,
)

from models.u_mobilevit_net.seg_head import (
    UpsampleHead,
    FeatureRefinementBlock,
    ContextAwareSegHead,
    MultiTaskDenseHead,
)

from models.u_mobilevit_net.module import (
    _UMobileViTLayer,
    get_groupnorm_groups,
    _get_clones,
    _get_local_block,
    _get_expansion_block,
)

from models.u_mobilevit_net.transfomer import (
    SeparableAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    _get_initializer,
)

__all__ = [
    # Configs
    "UMOBILEVIT_VARIANTS",
    "get_variant",
    "list_variants",
    "get_stem_channels",
    "get_seghead_channels",
    # Model
    "UMobileViT",
    "umobilevit",
    "umobilevit_nano",
    "umobilevit_base",
    "umobilevit_pro",
    "umobilevit_promax",
    # Encoder
    "UMobileViTEncoder",
    "UMobileViTEncoderLayer",
    # Decoder
    "UMobileViTDecoder",
    "UMobileViTDecoderLayer",
    "UMobileViTDecoderConcatLayer",
    # Seg Head
    "UpsampleHead",
    "FeatureRefinementBlock",
    "ContextAwareSegHead",
    "MultiTaskDenseHead",
    # Module
    "_UMobileViTLayer",
    "get_groupnorm_groups",
    "_get_clones",
    "_get_local_block",
    "_get_expansion_block",
    # Transformer
    "SeparableAttention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "_get_initializer",
]
