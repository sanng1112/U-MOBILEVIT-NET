from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from cv_nets.blocks import INLATransformerBlock, TransformerEncoderBlock
from cv_nets.utils.spectral import effective_rank, spectral_entropy, singular_value_decay


class SpectralAnalyzer:
    def __init__(self, every_n_epochs: int = 1):
        self.every_n_epochs = every_n_epochs
        self.results: Dict[str, List] = {"epoch": [], "effective_rank": [], "spectral_entropy": [], "top1_ratio": []}

    def analyze(self, model: nn.Module, x: torch.Tensor, epoch: int) -> Dict[str, float]:
        if epoch % self.every_n_epochs != 0:
            return {}
        outputs, hooks = {}, []

        def _hook(name):
            def fn(module, inp, out):
                outputs[name] = out.detach()
            return fn

        for name, module in model.named_modules():
            if isinstance(module, (INLATransformerBlock, TransformerEncoderBlock)):
                hooks.append(module.register_forward_hook(_hook(name)))
        with torch.no_grad():
            model(x)
        results = {}
        for name, out in outputs.items():
            results[f"{name}.erank"] = effective_rank(out)
            results[f"{name}.sent"] = spectral_entropy(out)
            sv, top1 = singular_value_decay(out)
            results[f"{name}.top1"] = top1
        if results:
            keys = [k for k in results if k.endswith(".erank")]
            if keys:
                self.results["epoch"].append(epoch)
                self.results["effective_rank"].append(sum(results[k] for k in keys) / len(keys))
                self.results["spectral_entropy"].append(
                    sum(results[k] for k in results if k.endswith(".sent")) / len(keys))
                self.results["top1_ratio"].append(
                    sum(results[k] for k in results if k.endswith(".top1")) / len(keys))
        for h in hooks:
            h.remove()
        return results

    def summary(self) -> str:
        lines = ["# Spectral Analysis Summary", "",
                 "| Epoch | Effective Rank | Spectral Entropy | Top-1 Ratio |",
                 "|-------|---------------|------------------|-------------|"]
        for i in range(len(self.results["epoch"])):
            lines.append(f"| {self.results['epoch'][i]:4d} | {self.results['effective_rank'][i]:.4f} | "
                         f"{self.results['spectral_entropy'][i]:.4f} | {self.results['top1_ratio'][i]:.4f} |")
        return "\n".join(lines)


class AblationController:
    def __init__(self, model: nn.Module):
        self.model = model
        self._registered: Dict[str, List[Tuple[nn.Module, str]]] = {}
        for name, module in model.named_modules():
            if hasattr(module, "use_lifting"):
                self._register(f"{name}.use_lifting", module, "use_lifting")

    def _register(self, key, module, attr):
        self._registered.setdefault(key, []).append((module, attr))

    def set(self, key: str, value: Any):
        if key not in self._registered:
            raise KeyError(f"Key '{key}' khong ton tai. Co: {list(self._registered.keys())}")
        for module, attr in self._registered[key]:
            setattr(module, attr, value)

    def list_keys(self) -> List[str]:
        return list(self._registered.keys())


class AttentionVisualizer:
    def __init__(self):
        self.scores: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, (INLATransformerBlock, TransformerEncoderBlock)):
                self._hooks.append(module.register_forward_hook(
                    lambda mod, inp, out, n=name: self.scores.update({n: out.detach().cpu()})))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
