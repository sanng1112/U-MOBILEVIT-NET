from typing import Any, Dict, List
import torch
import torch.nn as nn
from cv_nets.pipeline.config import ModelConfig, LayerConfig
from cv_nets.pipeline.registry import BLOCK_REGISTRY, filter_params


class ModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config

    def build(self) -> nn.Module:
        layers = self._build_layers(self.config.layers)
        model = nn.Sequential(*layers)
        return DynamicModel(model, self.config)

    # Các layer cần in_channels
    _CHANNEL_LAYERS = {"Conv2d", "ConvBNAct", "ResNetBasicBlock", "MV2Block",
                       "ConvNormAct", "DepthwiseSeparableConv", "InvertedResidual",
                       "SqueezeExcite", "ECA", "CBAM", "LinearLayer", "FC"}

    def _build_layers(self, layer_configs: List[LayerConfig]) -> List[nn.Module]:
        built = []
        ch = self.config.input_size[0] if len(self.config.input_size) == 3 else 3
        flat_size = None
        state = {"in_channels": ch, "height": self.config.input_size[-2] if len(self.config.input_size) >= 2 else None,
                 "width": self.config.input_size[-1] if len(self.config.input_size) >= 2 else None}

        for lc in layer_configs:
            layer = self._build_single(lc, state)
            built.append(layer)
            if lc.type == "Flatten":
                if state.get("height") and state.get("width"):
                    flat_size = state["in_channels"] * state["height"] * state["width"]
                state.pop("height", None)
                state.pop("width", None)
            elif "out_channels" in lc.params:
                state["in_channels"] = lc.params["out_channels"]
            if "in_features" in lc.params:
                state.setdefault("in_features", lc.params["in_features"])
        # Gán flat_size cho layer FC đầu tiên sau Flatten
        if flat_size:
            for lc in layer_configs:
                if lc.type in ("FC", "LinearLayer"):
                    if "in_features" not in lc.params:
                        lc.params["in_features"] = flat_size
                    break
        return built

    def _build_single(self, lc: LayerConfig, state: Dict[str, Any]) -> nn.Module:
        if lc.type not in BLOCK_REGISTRY:
            raise ValueError(f"Block '{lc.type}' khong ton tai. Co: {list(BLOCK_REGISTRY.keys())}")
        cls = BLOCK_REGISTRY[lc.type]
        params = dict(lc.params)
        # Auto-fill in_channels cho các layer cần
        if lc.type in self._CHANNEL_LAYERS and "in_channels" not in params and "in_features" not in params:
            params["in_channels"] = state.get("in_channels")
        # Auto-fill in_features cho FC nếu có state
        if lc.type in ("FC", "LinearLayer") and "in_features" not in params:
            if "in_features" in state:
                params["in_features"] = state["in_features"]
        valid_params = filter_params(cls, params)
        try:
            return cls(**valid_params)
        except Exception as e:
            raise RuntimeError(f"Loi khoi tao {lc.type} voi {valid_params}: {e}")


class DynamicModel(nn.Module):
    def __init__(self, backbone: nn.Module, config: ModelConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def __repr__(self) -> str:
        return f"DynamicModel({self.config.name})\n{self.backbone.__repr__()}"
