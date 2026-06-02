"""
Comparison model wrappers cho U-MobileViT-Net benchmarking.

Hỗ trợ:
  - UNet (ResNet backbone, segmentation-models-pytorch)
  - DeepLabV3+ (MobileNetV2 backbone, torchvision)
  - SegFormer-B0 (transformers library)

Tất cả wrapper có unified interface:
  model = create_benchmark_model("unet", num_classes=11)
  out = model(torch.randn(2, 3, 320, 320))
"""

import torch
import torch.nn as nn
from typing import Optional


class UNetWrapper(nn.Module):
    """UNet với ResNet backbone từ segmentation-models-pytorch."""

    def __init__(self, num_classes: int = 1, backbone: str = "resnet34",
                 in_channels: int = 3):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if in_channels == 3 else None,
                in_channels=in_channels,
                classes=num_classes,
            )
            self._available = True
        except ImportError:
            print("[WARN] segmentation-models-pytorch not installed. "
                  "UNet wrapper will raise error on forward.")
            self.model = None
            self._available = False

    def forward(self, x):
        if not self._available:
            raise RuntimeError("segmentation-models-pytorch not installed")
        return self.model(x)


class DeepLabV3PlusWrapper(nn.Module):
    """DeepLabV3+ với MobileNetV2 backbone từ torchvision."""

    def __init__(self, num_classes: int = 1, backbone: str = "mobilenet_v2",
                 in_channels: int = 3):
        super().__init__()
        try:
            from torchvision.models.segmentation import deeplabv3plus_mobilenet_v3_large
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            if "mobilenet" in backbone:
                self.model = deeplabv3plus_mobilenet_v3_large(
                    num_classes=num_classes,
                )
            else:
                from torchvision.models.segmentation import deeplabv3_resnet50
                self.model = deeplabv3_resnet50(
                    weights=DeepLabV3_ResNet50_Weights.DEFAULT if num_classes == 21 else None,
                    num_classes=num_classes,
                )
            self._available = True
        except ImportError:
            print("[WARN] torchvision segmentation models not available.")
            self.model = None
            self._available = False

    def forward(self, x):
        if not self._available:
            raise RuntimeError("torchvision segmentation models not available")
        result = self.model(x)
        # torchvision models return OrderedDict with 'out' key
        if isinstance(result, dict):
            return result["out"]
        return result


class SegFormerWrapper(nn.Module):
    """SegFormer từ HuggingFace transformers."""

    def __init__(self, num_classes: int = 1, backbone: str = "b0",
                 in_channels: int = 3):
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation

            model_name = f"nvidia/mit-{backbone}"
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            self._available = True
        except ImportError:
            print("[WARN] transformers not installed. "
                  "SegFormer wrapper will raise error on forward.")
            self.model = None
            self._available = False

    def forward(self, x):
        if not self._available:
            raise RuntimeError("transformers not installed")
        # SegFormer expects specific input size (divisible by 32)
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        # Upsample to input size
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits


# ═══════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════

BENCHMARK_MODELS = {
    "unet": UNetWrapper,
    "deeplabv3p": DeepLabV3PlusWrapper,
    "segformer": SegFormerWrapper,
}


def create_benchmark_model(name: str, num_classes: int = 1,
                          backbone: str = None, **kwargs) -> nn.Module:
    """Factory function cho comparison models.

    Args:
        name: "unet", "deeplabv3p", hoặc "segformer"
        num_classes: Số classes đầu ra
        backbone: Backbone cụ thể (optional)
        **kwargs: Tham số bổ sung cho model

    Returns:
        nn.Module wrapped model

    Raises:
        ValueError nếu name không hợp lệ
    """
    if name not in BENCHMARK_MODELS:
        raise ValueError(
            f"Unknown benchmark model '{name}'. "
            f"Available: {list(BENCHMARK_MODELS.keys())}"
        )

    model_cls = BENCHMARK_MODELS[name]
    init_kwargs = {"num_classes": num_classes}

    if backbone:
        init_kwargs["backbone"] = backbone

    init_kwargs.update(kwargs)
    return model_cls(**init_kwargs)


def list_benchmark_models():
    """Liệt kê tất cả comparison models."""
    return list(BENCHMARK_MODELS.keys())


def get_model_params(model: nn.Module) -> int:
    """Đếm số tham số của model."""
    return sum(p.numel() for p in model.parameters())


def get_model_info(name: str, num_classes: int = 1,
                   backbone: str = None) -> dict:
    """Lấy thông tin model (params, description)."""
    try:
        model = create_benchmark_model(name, num_classes, backbone)
        params = get_model_params(model)
        return {
            "name": name,
            "backbone": backbone,
            "num_classes": num_classes,
            "params": params,
            "params_M": f"{params/1e6:.2f}M",
        }
    except Exception as e:
        return {
            "name": name,
            "backbone": backbone,
            "num_classes": num_classes,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Benchmark Models Registry")
    print("=" * 40)
    for name in BENCHMARK_MODELS:
        print(f"\n{name}:")
        try:
            info = get_model_info(name)
            if "error" in info:
                print(f"  [UNAVAILABLE] {info['error']}")
            else:
                print(f"  Params: {info['params_M']}")
        except Exception as e:
            print(f"  [ERROR] {e}")
