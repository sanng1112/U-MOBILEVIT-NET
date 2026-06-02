"""
Controlled experiments for theoretical analysis of U-MobileViT-Net.

Includes:
    - Rank collapse experiment: measures effective rank of token
      representations across attention depth, comparing baseline
      separable attention against INLA (rank-preserving variant).
    - Inference speed benchmarking on GPU / CPU.
    - Width-multiplier (alpha) sensitivity sweep.

Usage:
    from tools.experiments import run_rank_collapse_experiment

    base_erank, inla_erank, base_sent, inla_sent, base_top1, inla_top1 = \
        run_rank_collapse_experiment(dim=64, tokens=196, depth=12)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from cv_nets.blocks.inla import INLAAttention, INLATransformerBlock
from cv_nets.utils.spectral import effective_rank, spectral_entropy, singular_value_decay


# ---------------------------------------------------------------------------
# Rank collapse experiment
# ---------------------------------------------------------------------------

def _correlated_input(
    n_tokens: int, dim: int, rank: int, seed: int,
) -> torch.Tensor:
    """Generate tokens with controlled intrinsic rank.

    Simulates feature maps that have passed through several convolutional
    layers, where representations tend to be low-rank.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    U = torch.randn(n_tokens, rank, generator=g)
    V = torch.randn(rank, dim, generator=g)
    X = U @ V
    X = X / X.std()
    return X


def run_rank_collapse_experiment(
    dim: int = 64,
    n_tokens: int = 196,
    depth: int = 12,
    dim_expand: int = 128,
    seeds: int = 8,
    mode: str = "block",
) -> Tuple[
    List[float], List[float],   # effective rank (baseline, inla)
    List[float], List[float],   # spectral entropy
    List[float], List[float],   # top-1 singular value ratio
]:
    """Measure how token rank evolves across stacked attention layers.

    Compares a baseline separable attention stack against INLA, which
    uses a lifting mechanism designed to preserve token rank.  The
    experiment is repeated across multiple random seeds and averaged.

    Args:
        dim: Token embedding dimension.
        n_tokens: Number of tokens (e.g., 14×14 = 196).
        depth: Number of stacked layers.
        dim_expand: Hidden dimension for the INLA transformer block.
        seeds: Number of random seeds to average over.
        mode: ``"block"`` uses ``INLATransformerBlock`` (attention + MLP);
              ``"attention"`` uses raw ``INLAAttention``.

    Returns:
        Six lists of length *depth*, averaged across seeds:
        (base_erank, inla_erank, base_sent, inla_sent, base_top1, inla_top1).
    """
    base_erank_sum = [0.0] * depth
    inla_erank_sum = [0.0] * depth
    base_sent_sum = [0.0] * depth
    inla_sent_sum = [0.0] * depth
    base_top1_sum = [0.0] * depth
    inla_top1_sum = [0.0] * depth

    for seed in range(seeds):
        x = _correlated_input(n_tokens, dim, rank=dim // 8, seed=seed)

        # Baseline: INLA with lifting disabled
        base = x.clone()
        for d in range(depth):
            if mode == "block":
                layer = INLATransformerBlock(
                    dim=dim, dim_expand=dim_expand, use_lifting=False,
                )
            else:
                layer = INLAAttention(dim=dim, use_lifting=False)
            base = layer(base)
            er = effective_rank(base)
            se = spectral_entropy(base)
            _, sv = singular_value_decay(base)
            top1 = sv[0] / sv.sum() if sv.sum() > 0 else 1.0
            base_erank_sum[d] += er
            base_sent_sum[d] += se
            base_top1_sum[d] += top1

        # INLA: lifting enabled
        inla = x.clone()
        for d in range(depth):
            if mode == "block":
                layer = INLATransformerBlock(
                    dim=dim, dim_expand=dim_expand, use_lifting=True,
                )
            else:
                layer = INLAAttention(dim=dim, use_lifting=True)
            inla = layer(inla)
            er = effective_rank(inla)
            se = spectral_entropy(inla)
            _, sv = singular_value_decay(inla)
            top1 = sv[0] / sv.sum() if sv.sum() > 0 else 1.0
            inla_erank_sum[d] += er
            inla_sent_sum[d] += se
            inla_top1_sum[d] += top1

    avg = lambda s: [v / seeds for v in s]
    return (
        avg(base_erank_sum), avg(inla_erank_sum),
        avg(base_sent_sum), avg(inla_sent_sum),
        avg(base_top1_sum), avg(inla_top1_sum),
    )


# ---------------------------------------------------------------------------
# Inference speed benchmark
# ---------------------------------------------------------------------------

def benchmark_inference_speed(
    model: nn.Module,
    input_size: Tuple[int, int] = (320, 320),
    device: str = "cuda",
    precision: str = "fp16",
    batch_size: int = 1,
    warmup: int = 50,
    repeats: int = 200,
) -> Tuple[float, float]:
    """Measure single-image inference throughput.

    Args:
        model: The segmentation model.
        input_size: (height, width).
        device: ``cuda`` or ``cpu``.
        precision: ``fp32`` or ``fp16``.
        batch_size: Number of images per forward pass.
        warmup: Warmup iterations (not timed).
        repeats: Number of timed iterations.

    Returns:
        ``(fps, ms_per_image)``.
    """
    model = model.to(device)
    model.eval()

    dummy = torch.randn(batch_size, 3, *input_size, device=device)

    use_amp = (precision == "fp16" and device == "cuda")

    # Warmup
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    for _ in range(repeats):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_image = (elapsed / repeats) * 1000 / batch_size
    fps = 1000.0 / max(ms_per_image, 1e-9)
    return fps, ms_per_image


# ---------------------------------------------------------------------------
# Alpha (width multiplier) sweep
# ---------------------------------------------------------------------------

def sweep_alpha(
    alpha_values: List[float],
    input_size: Tuple[int, int] = (320, 320),
    device: str = "cpu",
) -> List[Dict]:
    """Sweep the width multiplier and collect FLOPs / parameter counts.

    Args:
        alpha_values: List of width multipliers to evaluate.
        input_size: (height, width).
        device: torch device string.

    Returns:
        List of dicts with keys: alpha, d_model, params, flops_total.
    """
    from models.u_mobilevit_net.u_models import umobilevit
    from tools.evaluation import compute_flops, compute_parameters

    results = []
    for alpha in alpha_values:
        model = umobilevit(
            alpha=alpha, d_model=64, out_channels=8, head="single",
            expansion_factor=3.0, patch_size=(2, 2),
            dropout_p=0.1, norm_num_groups=4,
            bias=True, num_transformer_block=2,
        )
        model = model.to(device)
        model.eval()

        flops, _ = compute_flops(model, input_size)
        params = compute_parameters(model)
        results.append({
            "alpha": alpha,
            "d_model": int(64 * alpha),
            "params": params,
            "flops_total": flops,
        })
    return results
