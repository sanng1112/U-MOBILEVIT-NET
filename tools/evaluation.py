"""
Model evaluation utilities for U-MobileViT-Net.

Provides accurate (non-thop) FLOPs calculation, parameter counting,
and edge-device latency estimation across a catalogue of representative
hardware platforms (Jetson family, Raspberry Pi, smartphones).

The FLOPs calculator walks the model architecture directly rather than
using a profiler, which ensures correctness with custom ``Conv2d``
subclasses that third-party libraries cannot trace.

Usage:
    from tools.evaluation import compute_flops, compute_parameters, estimate_edge_latency

    model = umobilevit_base(out_channels=8, head="single")
    flops, breakdown = compute_flops(model, (320, 320))
    params = compute_parameters(model)
    latency = estimate_edge_latency(flops, params, "jetson_orin_nano")
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# FLOPs primitives
# ---------------------------------------------------------------------------

def _conv2d_flops(
    in_ch: int, out_ch: int, kernel: int,
    h_in: int, w_in: int, stride: int = 1,
    groups: int = 1, bias: bool = True,
) -> int:
    """FLOPs for one Conv2d layer (MAC × 2 + bias adds)."""
    h_out = h_in // stride
    w_out = w_in // stride
    mac = out_ch * (in_ch // groups) * kernel * kernel * h_out * w_out
    flops = 2 * mac
    if bias:
        flops += out_ch * h_out * w_out
    return flops


def _separable_attn_flops(
    in_ch: int, patch_h: int, patch_w: int, num_patches: int,
) -> int:
    """FLOPs for one separable self-attention layer."""
    P = patch_h * patch_w
    S = num_patches
    # QKV projection
    qkv = 2 * (1 + 2 * in_ch) * in_ch * P * S
    # softmax (~5 ops / token)
    softmax = 5 * P * S
    # context aggregation
    ctx = 2 * in_ch * P * S
    # activation + multiply
    act = 2 * in_ch * P * S
    # output projection
    out = 2 * in_ch * in_ch * P * S
    return qkv + softmax + ctx + act + out


def _local_block_flops(in_ch: int, h: int, w: int) -> int:
    """FLOPs for a local block (DW 3×3 + PW 1×1 + GroupNorm)."""
    flops = _conv2d_flops(in_ch, in_ch, 3, h, w, groups=in_ch)
    flops += _conv2d_flops(in_ch, in_ch, 1, h, w)
    flops += 4 * in_ch * h * w  # GroupNorm
    return flops


def _expansion_block_flops(
    in_ch: int, exp_factor: float, h: int, w: int,
) -> int:
    """FLOPs for an inverted bottleneck expansion block."""
    exp_ch = int(in_ch * exp_factor)
    flops = _conv2d_flops(in_ch, exp_ch, 1, h, w)            # PW expand
    flops += _conv2d_flops(exp_ch, exp_ch, 3, h, w, groups=exp_ch)  # DW
    flops += _conv2d_flops(exp_ch, in_ch, 1, h, w)            # PW project
    return flops


def _downsample_flops(in_ch: int, h: int, w: int) -> int:
    return _conv2d_flops(in_ch, in_ch, 3, h, w, stride=2, groups=in_ch)


def _upsample_flops(in_ch: int, h: int, w: int) -> int:
    return _conv2d_flops(in_ch, in_ch, 3, 2 * h, 2 * w, groups=in_ch)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_flops(
    model: nn.Module, input_size: Tuple[int, int] = (320, 320),
) -> Tuple[int, Dict[str, int]]:
    """Compute total FLOPs and a per-component breakdown.

    Args:
        model: A ``UMobileViT`` instance.
        input_size: (height, width) of the input image.

    Returns:
        ``(total_flops, breakdown)`` where *breakdown* is a dict mapping
        component names to integer FLOP counts.
    """
    H, W = input_size
    d = model.encoder.d_model
    exp_factor = model.expansion_factor
    patch_h, patch_w = model.patch_size
    n_trans = model.num_transformer_block
    breakdown: Dict[str, int] = {}

    # -- Stem ---------------------------------------------------------------
    breakdown["stem_conv1"] = _conv2d_flops(3, d // 4, 3, H, W, stride=2)

    flops = _conv2d_flops(d // 4, d // 2, 3, H // 2, W // 2, stride=2)
    flops += 4 * (d // 2) * (H // 4) * (W // 4)
    breakdown["stem_conv2"] = flops

    flops = _conv2d_flops(d // 2, d, 3, H // 4, W // 4, stride=2)
    flops += 4 * d * (H // 8) * (W // 8)
    breakdown["stem_conv3"] = flops

    # -- Encoder stages -----------------------------------------------------
    encoder_cfg = [
        (H // 16, W // 16, 1, (patch_h, patch_w)),
        (H // 32, W // 32, 2, (patch_h, patch_w)),
        (H // 64, W // 64, 2, (1, 1)),
    ]
    prev_h, prev_w = H // 8, W // 8

    for si, (sh, sw, n_layers, (ph, pw)) in enumerate(encoder_cfg):
        breakdown[f"enc_s{si+1}_downsample"] = _downsample_flops(d, prev_h, prev_w)
        for li in range(n_layers):
            prefix = f"enc_s{si+1}_l{li}"
            breakdown[f"{prefix}_local"] = _local_block_flops(d, sh, sw)
            P = ph * pw
            N = (sh // ph) * (sw // pw)
            for ti in range(n_trans):
                breakdown[f"{prefix}_attn{ti}"] = _separable_attn_flops(d, ph, pw, N)
            breakdown[f"{prefix}_expansion"] = _expansion_block_flops(d, exp_factor, sh, sw)
        prev_h, prev_w = sh, sw

    # -- Decoder ------------------------------------------------------------
    decoder_cfg = [
        (H // 64, W // 64, H // 32, W // 32, 2),
        (H // 32, W // 32, H // 16, W // 16, 1),
        (H // 16, W // 16, H // 8, W // 8, 0),
    ]
    for si, (in_h, in_w, out_h, out_w, n_layers) in enumerate(decoder_cfg):
        breakdown[f"dec_s{si+1}_upsample"] = _upsample_flops(d, in_h, in_w)
        for li in range(n_layers):
            prefix = f"dec_s{si+1}_l{li}"
            breakdown[f"{prefix}_local"] = _local_block_flops(d, out_h, out_w)
            P = 4
            N = (out_h // 2) * (out_w // 2)
            for ti in range(n_trans):
                breakdown[f"{prefix}_self_attn{ti}"] = _separable_attn_flops(d, 2, 2, N)
                breakdown[f"{prefix}_cross_attn{ti}"] = _separable_attn_flops(d, 2, 2, N)
            breakdown[f"{prefix}_expansion"] = _expansion_block_flops(d, exp_factor, out_h, out_w)

    # Decoder concat layer (no transformer)
    ch, oh, ow = d, H // 8, W // 8
    breakdown["dec_concat_local"] = _local_block_flops(ch, oh, ow)
    breakdown["dec_concat_mem_proj"] = _conv2d_flops(ch, ch, 1, oh, ow)
    breakdown["dec_concat_global_dw"] = _conv2d_flops(ch, ch, 3, oh, ow, groups=ch)
    breakdown["dec_concat_global_pw"] = _conv2d_flops(ch, ch, 1, oh, ow)
    breakdown["dec_concat_expansion"] = _expansion_block_flops(ch, exp_factor, oh, ow)

    # -- Segmentation head --------------------------------------------------
    h_step, w_step = H // 8, W // 8
    for step, (ch_in, ch_out) in enumerate([(d, d // 2), (d // 2, d // 4), (d // 4, d // 8)]):
        prefix = f"head_upsample_x{2**(step+1)}"
        breakdown[f"{prefix}_proj"] = _conv2d_flops(ch_in, ch_out, 1, h_step, w_step)
        breakdown[f"{prefix}_dwconv"] = _conv2d_flops(
            ch_out, ch_out, 3, 2 * h_step, 2 * w_step, groups=ch_out,
        )
        if step < 2:
            nch, nh, nw = ch_out, 2 * h_step, 2 * w_step
            breakdown[f"{prefix}_concat_local"] = _local_block_flops(nch, nh, nw)
            breakdown[f"{prefix}_concat_mem"] = _conv2d_flops(nch, nch, 1, nh, nw)
            breakdown[f"{prefix}_concat_global_dw"] = _conv2d_flops(nch, nch, 3, nh, nw, groups=nch)
            breakdown[f"{prefix}_concat_global_pw"] = _conv2d_flops(nch, nch, 1, nh, nw)
            breakdown[f"{prefix}_concat_exp"] = _expansion_block_flops(nch, exp_factor, nh, nw)
        h_step, w_step = 2 * h_step, 2 * w_step

    # Feature refinement
    rch, rh, rw = d // 8, H, W
    breakdown["refine_dilated"] = _conv2d_flops(rch, rch, 3, rh, rw, groups=rch)
    breakdown["refine_mix"] = _conv2d_flops(rch, rch, 1, rh, rw)
    breakdown["refine_spatial_attn"] = _conv2d_flops(2, 1, 7, rh, rw, bias=False)
    breakdown["refine_elemwise"] = 10 * rh * rw

    # Task classifiers — handle both single-task and multi-task heads
    if hasattr(model.seg_head, "out_channels_list"):
        out_ch_list = model.seg_head.out_channels_list
    else:
        out_ch_list = [model.seg_head.out_channels]
    for ti, out_ch in enumerate(out_ch_list):
        prefix = f"classifier_t{ti}"
        breakdown[f"{prefix}_dw"] = _conv2d_flops(rch, rch, 3, rh, rw, groups=rch)
        breakdown[f"{prefix}_pw"] = _conv2d_flops(rch, out_ch, 1, rh, rw)

    total = sum(breakdown.values())
    return total, dict(breakdown)


def compute_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops_by_component(
    model: nn.Module, input_size: Tuple[int, int] = (320, 320),
) -> Dict[str, int]:
    """Aggregate FLOPs into high-level architectural groups.

    Returns a dict with keys: stem, encoder, decoder, head, refinement, classifier.
    """
    _, breakdown = compute_flops(model, input_size)
    groups: Dict[str, int] = defaultdict(int)
    for k, v in breakdown.items():
        if k.startswith("stem_"):
            groups["stem"] += v
        elif k.startswith("enc_"):
            groups["encoder"] += v
        elif k.startswith("dec_"):
            groups["decoder"] += v
        elif k.startswith("head_"):
            groups["head"] += v
        elif k.startswith("refine_"):
            groups["refinement"] += v
        elif k.startswith("classifier_"):
            groups["classifier"] += v
    return dict(groups)


def format_flops(flops: int) -> str:
    """Format an integer FLOP count as a human-readable string."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f}K"
    return str(flops)


# ---------------------------------------------------------------------------
# Edge device catalogue
# ---------------------------------------------------------------------------

DEVICES: Dict[str, dict] = {
    "jetson_nano": {
        "name": "NVIDIA Jetson Nano (4 GB)",
        "ram_mb": 4096,
        "gpu_gflops_fp32": 235,
        "gpu_eff_fp32": 0.35,
        "gpu_gflops_fp16": 472,
        "gpu_eff_fp16": 0.30,
        "gpu_gops_int8": 944,
        "gpu_eff_int8": 0.50,
        "mem_bw_gbs": 25.6,
        "power_w": 10,
        "backend": "TensorRT",
        "kernel_overhead_us": 15,
    },
    "jetson_xavier_nx": {
        "name": "NVIDIA Jetson Xavier NX",
        "ram_mb": 8192,
        "gpu_gflops_fp32": 845,
        "gpu_eff_fp32": 0.40,
        "gpu_gflops_fp16": 1690,
        "gpu_eff_fp16": 0.35,
        "gpu_gops_int8": 3380,
        "gpu_eff_int8": 0.55,
        "mem_bw_gbs": 51.2,
        "power_w": 15,
        "backend": "TensorRT",
        "kernel_overhead_us": 8,
    },
    "jetson_orin_nano": {
        "name": "NVIDIA Jetson Orin Nano (8 GB)",
        "ram_mb": 8192,
        "gpu_gflops_fp32": 1024,
        "gpu_eff_fp32": 0.45,
        "gpu_gflops_fp16": 2048,
        "gpu_eff_fp16": 0.40,
        "gpu_gops_int8": 4096,
        "gpu_eff_int8": 0.55,
        "mem_bw_gbs": 68,
        "power_w": 7,
        "backend": "TensorRT",
        "kernel_overhead_us": 5,
    },
    "raspberry_pi_5": {
        "name": "Raspberry Pi 5 (8 GB, ARM CPU)",
        "ram_mb": 8192,
        "cpu_gflops_fp32": 20,
        "cpu_eff_fp32": 0.50,
        "mem_bw_gbs": 17,
        "power_w": 8,
        "backend": "ONNX Runtime / ncnn",
        "kernel_overhead_us": 50,
    },
    "smartphone_mid": {
        "name": "Mid-range Smartphone (SD 7 Gen1)",
        "ram_mb": 4096,
        "npu_gops_int8": 1000,
        "npu_eff_int8": 0.60,
        "mem_bw_gbs": 22,
        "power_w": 3,
        "backend": "TFLite / QNN",
        "kernel_overhead_us": 20,
    },
    "smartphone_high": {
        "name": "Flagship Smartphone (SD 8 Gen3)",
        "ram_mb": 8192,
        "npu_gops_int8": 4000,
        "npu_eff_int8": 0.65,
        "mem_bw_gbs": 64,
        "power_w": 4,
        "backend": "TFLite / QNN",
        "kernel_overhead_us": 10,
    },
}


# ---------------------------------------------------------------------------
# Latency estimation
# ---------------------------------------------------------------------------

def estimate_edge_latency(
    flops: int,
    params: int,
    input_size: Tuple[int, int] = (320, 320),
    device_key: str = "jetson_orin_nano",
    precision: str = "fp32",
) -> Dict:
    """Estimate inference latency on a representative edge device.

    Uses a heuristic model combining compute-bound and memory-bound regimes:
    latency = max(flops / effective_throughput, params / mem_bw) + kernel_overhead.

    Args:
        flops: Total model FLOPs (from :func:`compute_flops`).
        params: Number of trainable parameters.
        input_size: (height, width) of the input.
        device_key: Key in :data:`DEVICES` catalogue.
        precision: ``fp32``, ``fp16``, or ``int8``.

    Returns:
        Dict with keys: device_name, latency_ms, fps, fits_ram, bottleneck,
        precision, input_size.
    """
    spec = DEVICES[device_key]
    H, W = input_size

    if precision == "fp32":
        if "gpu_gflops_fp32" in spec:
            eff_flops = spec["gpu_gflops_fp32"] * spec["gpu_eff_fp32"] * 1e9
        elif "cpu_gflops_fp32" in spec:
            eff_flops = spec["cpu_gflops_fp32"] * spec["cpu_eff_fp32"] * 1e9
        else:
            eff_flops = 1e9
    elif precision == "fp16":
        eff_flops = spec.get("gpu_gflops_fp16", spec.get("gpu_gflops_fp32", 100)) \
                    * spec.get("gpu_eff_fp16", 0.3) * 1e9
    else:  # int8
        eff_flops = spec.get("gpu_gops_int8", spec.get("npu_gops_int8", 500)) \
                    * spec.get("gpu_eff_int8", spec.get("npu_eff_int8", 0.5)) * 1e9

    compute_time_s = flops / max(eff_flops, 1)
    mem_time_s = (params * 4) / (spec["mem_bw_gbs"] * 1e9)  # 4 bytes per fp32 param
    kernel_overhead_s = spec.get("kernel_overhead_us", 10) * 1e-6
    latency_s = max(compute_time_s, mem_time_s) + kernel_overhead_s

    memory_mb = params * 4 / 1e6  # rough: fp32
    fits_ram = memory_mb < spec["ram_mb"] * 0.8  # 80% threshold

    if compute_time_s > mem_time_s:
        bottleneck = "compute"
    else:
        bottleneck = "memory"

    return {
        "device_name": spec["name"],
        "latency_ms": round(latency_s * 1000, 2),
        "fps": round(1.0 / max(latency_s, 1e-9), 1),
        "fits_ram": fits_ram,
        "bottleneck": bottleneck,
        "precision": precision,
        "input_size": f"{H}×{W}",
        "flops_formatted": format_flops(flops),
    }


def estimate_all_devices(
    flops: int, params: int, input_size: Tuple[int, int] = (320, 320),
) -> List[Dict]:
    """Estimate latency across all devices in the catalogue at fp32."""
    return [
        estimate_edge_latency(flops, params, input_size, key)
        for key in DEVICES
    ]
