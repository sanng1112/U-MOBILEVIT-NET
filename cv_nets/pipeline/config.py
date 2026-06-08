from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class LayerConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerConfig":
        d = dict(d)
        layer_type = d.pop("type", None)
        if layer_type is None:
            raise ValueError("Moi layer config phai co truong 'type'")
        return cls(type=layer_type, params=d)


@dataclass
class ModelConfig:
    name: str = "model"
    input_size: List[int] = field(default_factory=lambda: [3, 224, 224])
    output_dim: int = 10
    layers: List[LayerConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        layers_raw = d.pop("layers", [])
        layers = [LayerConfig.from_dict(l) if isinstance(l, dict) else l for l in layers_raw]
        return cls(layers=layers, **d)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.0
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.999
    use_amp: bool = False
    patience: int = 10
    min_epochs: int = 30
    save_dir: str = "./checkpoints"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetConfig:
    name: str = "mnist"
    root: str = "./data"
    image_size: List[int] = field(default_factory=lambda: [28, 28])
    num_classes: int = 10
    task: str = "classification"
    aug_intensity: str = "medium"
    num_workers: int = 4
    val_split: float = 0.1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ResearchConfig:
    enabled: bool = False
    track_effective_rank: bool = False
    track_spectral_entropy: bool = False
    track_attention_maps: bool = False
    visualize_every: int = 10
    ablation_modes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            dataset=DatasetConfig.from_dict(d.get("dataset", {})),
            research=ResearchConfig.from_dict(d.get("research", {})),
            seed=d.get("seed", 42),
            device=d.get("device", "auto"),
        )


def load_config(path: Union[str, Path]) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.from_dict(raw)


def validate_config(cfg: PipelineConfig) -> List[str]:
    errors = []
    if not cfg.model.layers:
        errors.append("Model phai co it nhat 1 layer")
    for i, layer in enumerate(cfg.model.layers):
        if not layer.type:
            errors.append(f"Layer {i} thieu truong 'type'")
    if cfg.dataset.image_size and len(cfg.dataset.image_size) != 2:
        errors.append(f"image_size phai la [H, W], nhan: {cfg.dataset.image_size}")
    if cfg.training.epochs <= 0:
        errors.append("epochs phai > 0")
    if cfg.training.batch_size <= 0:
        errors.append("batch_size phai > 0")
    return errors
