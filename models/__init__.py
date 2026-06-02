"""
U-MobileViT-Net Models Package.

Usage:
    from models import umobilevit_nano, umobilevit_base, umobilevit_pro, umobilevit_promax
    model = umobilevit_base(out_channels=11)  # 11-class segmentation
"""

from models.u_mobilevit_net.u_models import (
    UMobileViT,
    umobilevit,
    umobilevit_nano,
    umobilevit_base,
    umobilevit_pro,
    umobilevit_promax,
)

from models.u_mobilevit_net.configs import (
    UMOBILEVIT_VARIANTS,
    get_variant,
    list_variants,
)

__all__ = [
    "UMobileViT",
    "umobilevit",
    "umobilevit_nano",
    "umobilevit_base",
    "umobilevit_pro",
    "umobilevit_promax",
    "UMOBILEVIT_VARIANTS",
    "get_variant",
    "list_variants",
]
