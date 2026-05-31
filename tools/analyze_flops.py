"""
Phân tích FLOPs, tham số, memory footprint và đánh giá thiết bị biên
cho U-MobileViT-Net.

Chạy: PYTHONPATH=. python tools/analyze_flops.py [--alpha 1.0] [--sweep]
"""
import argparse
import copy
import sys
import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict

from thop import profile, clever_format

from models.u_mobilevit_net.u_models import umobilevit
from models.u_mobilevit_net.transfomer import SeparableAttention


# ═══════════════════════════════════════════════════════════════
# PARAMETER COUNTING
# ═══════════════════════════════════════════════════════════════
def count_params_by_type(model: nn.Module) -> dict:
    """Đếm params theo loại module."""
    counts = OrderedDict()
    for name, module in model.named_modules():
        t = type(module).__name__
        p = sum(p.numel() for p in module.parameters())
        if p > 0:
            counts[t] = counts.get(t, 0) + p
    return counts


def count_params_by_component(model) -> dict:
    """Đếm params theo thành phần kiến trúc."""
    comps = OrderedDict()
    enc = model.encoder
    comps["Stem"] = sum(p.numel() for p in enc.stem_block.parameters())
    comps["Encoder Stages (1-3)"] = sum(p.numel() for p in enc.layers.parameters())

    dec = model.decoder
    comps["Decoder Stages (1-3)"] = sum(p.numel() for p in dec.layers.parameters())

    head = model.seg_head
    comps["UpsampleHead"] = sum(p.numel() for p in head.upsample_module.parameters())
    comps["FeatureRefinement"] = sum(p.numel() for p in head.refinement_module.parameters())
    comps["TaskClassifiers"] = sum(p.numel() for p in head.task_classifiers.parameters())

    # Attention vs Conv breakdown
    attn_params = 0
    for m in model.modules():
        if isinstance(m, SeparableAttention):
            attn_params += sum(p.numel() for p in m.parameters())
    comps["  └─ Attention (SeparableAttn)"] = attn_params
    comps["  └─ Convolution (all Conv2d)"] = comps.get("Conv2d", 0)

    return comps


# ═══════════════════════════════════════════════════════════════
# FLOPs ESTIMATION (thop-based, full-model only)
# ═══════════════════════════════════════════════════════════════
def profile_model(alpha: float, input_hw: Tuple[int, int], out_channels=(2, 4)):
    """Tạo model + profile với thop."""
    h, w = input_hw
    model = umobilevit(
        alpha=alpha, d_model=64, out_channels=out_channels,
        expansion_factor=3.0, patch_size=(2, 2),
        dropout_p=0.1, norm_num_groups=4,
        bias=True, num_transformer_block=2,
    )
    model.eval()
    dummy = torch.randn(1, 3, h, w)
    flops, _ = profile(copy.deepcopy(model), inputs=(dummy,), verbose=False)
    params = sum(p.numel() for p in model.parameters())

    # Đo riêng encoder bằng cách forward thủ công
    with torch.no_grad():
        stem, stage = model.encoder(dummy)
    enc_flops, _ = profile(copy.deepcopy(model.encoder), inputs=(dummy,), verbose=False)
    # Decoder
    dec_in = tuple(reversed(stage))
    dec_flops, _ = profile(copy.deepcopy(model.decoder), inputs=(dec_in,), verbose=False)
    # Head
    with torch.no_grad():
        dec_out = model.decoder(dec_in)
    head_flops, _ = profile(copy.deepcopy(model.seg_head),
                            inputs=(dec_out, tuple(reversed(stem))), verbose=False)

    return {
        "model": model,
        "params_total": params,
        "flops_total": flops,
        "flops_encoder": enc_flops,
        "flops_decoder": dec_flops,
        "flops_head": head_flops,
        "alpha": alpha,
        "input_hw": input_hw,
    }


# ═══════════════════════════════════════════════════════════════
# MEMORY & LATENCY ESTIMATION
# ═══════════════════════════════════════════════════════════════
def estimate_memory(params: int, input_hw: Tuple[int, int],
                    batch=1, num_tasks=2, dtype_bytes=4) -> dict:
    """Ước tính memory footprint cho inference."""
    B, C, H, W = batch, 3, input_hw[0], input_hw[1]

    model_mb = params * dtype_bytes / (1024 ** 2)
    input_mb = B * C * H * W * dtype_bytes / (1024 ** 2)

    # Activation memory estimate:
    # U-Net lưu ≈ 4 scale × C feature maps cho skip connections
    # + intermediate activations ≈ 2-3× input
    d_model_ratio = params / (3 * H * W)  # estimate channel density
    activations_mb = input_mb * 5.0  # heuristic cho U-Net depth=4

    output_mb = B * H * W * dtype_bytes * num_tasks * 2 / (1024 ** 2)

    peak_mb = model_mb + input_mb + activations_mb + output_mb

    return {
        "model_weights_mb": round(model_mb, 2),
        "input_mb": round(input_mb, 2),
        "activations_est_mb": round(activations_mb, 2),
        "output_mb": round(output_mb, 2),
        "peak_inference_mb": round(peak_mb, 2),
    }


# Edge device specifications (tham khảo datasheet thực tế)
EDGE_DEVICES = {
    "jetson_nano_4gb": {
        "name": "Jetson Nano (4GB, MAX-N)",
        "ram_mb": 4096,
        "compute_fp32_gflops": 118,      # ~118 GFLOPS FP32 thực tế
        "compute_fp16_gflops": 236,
        "compute_int8_gops": 472,
        "mem_bw_gbs": 25.6,
        "power_w": 10,
        "backend": "TensorRT",
    },
    "jetson_xavier_nx": {
        "name": "Jetson Xavier NX (8GB)",
        "ram_mb": 8192,
        "compute_fp32_gflops": 500,
        "compute_fp16_gflops": 1000,
        "compute_int8_gops": 2000,
        "mem_bw_gbs": 51.2,
        "power_w": 15,
        "backend": "TensorRT",
    },
    "jetson_orin_nano": {
        "name": "Jetson Orin Nano (8GB)",
        "ram_mb": 8192,
        "compute_fp32_gflops": 500,
        "compute_fp16_gflops": 1000,
        "compute_int8_gops": 2000,
        "mem_bw_gbs": 68,
        "power_w": 7,
        "backend": "TensorRT",
    },
    "raspberry_pi_5": {
        "name": "Raspberry Pi 5 (CPU-only)",
        "ram_mb": 8192,
        "compute_fp32_gflops": 15,        # ARM Cortex-A76, 4 cores ~15 GFLOPS
        "compute_fp16_gflops": 30,
        "compute_int8_gops": 60,
        "mem_bw_gbs": 17,
        "power_w": 8,
        "backend": "ONNX Runtime / ncnn",
    },
    "smartphone_mid": {
        "name": "Smartphone tầm trung (SNapdragon 7 Gen1)",
        "ram_mb": 4096,
        "compute_fp32_gflops": 200,
        "compute_fp16_gflops": 800,
        "compute_int8_gops": 1600,
        "mem_bw_gbs": 22,
        "power_w": 3,
        "backend": "TFLite / QNN / MNN",
    },
    "smartphone_high": {
        "name": "Smartphone cao cấp (A17 Pro / SD 8 Gen3)",
        "ram_mb": 6144,
        "compute_fp32_gflops": 500,
        "compute_fp16_gflops": 2000,
        "compute_int8_gops": 4000,
        "mem_bw_gbs": 51,
        "power_w": 4,
        "backend": "CoreML / SNPE / MNN",
    },
}


def estimate_latency(flops: float, params: int, input_hw: Tuple[int, int],
                     precision: str = "fp32") -> dict:
    """
    Ước tính latency/FPS trên các thiết bị biên.

    Công thức: latency = max(compute_time, memory_time) + kernel_overhead
    - compute_time: FLOPs / (effective_peak * efficiency)
    - memory_time: bytes_moved / memory_bandwidth
    - kernel_overhead: fixed cost per layer (~0.01-0.05ms/layer trên GPU)

    Real-world efficiency cho edge inference:
    - GPU (TensorRT): ~40-60% peak FLOPS cho model nhỏ
    - CPU: ~30-50% peak FLOPS
    - Model nhẹ như U-MobileViT (~50 layers) bị kernel-launch bound
    """
    # Đếm số lượng Conv2d + Attention layers (ước lượng kernel launches)
    # U-MobileViT-Net với d_model=64 có khoảng 80-100 conv layers
    estimated_kernel_launches = 85  # xấp xỉ số Conv2d + MatMul operations
    kernel_overhead_per_launch_ms = 0.008  # ~8μs mỗi kernel launch trên GPU edge

    results = {}
    h, w = input_hw

    for key, dev in EDGE_DEVICES.items():
        if precision == "fp32":
            peak = dev["compute_fp32_gflops"] * 1e9
            efficiency = 0.45  # 45% peak cho model nhỏ
        elif precision == "fp16":
            peak = dev["compute_fp16_gflops"] * 1e9
            efficiency = 0.40  # 40% — FP16 thường ít hiệu quả hơn trên model conv-heavy
        else:  # int8
            peak = dev["compute_int8_gops"] * 1e9
            efficiency = 0.50  # 50% — INT8 tốt hơn nhờ tensor cores

        effective = peak * efficiency

        # Compute-bound latency
        compute_time_ms = (flops / effective) * 1000

        # Memory-bound latency
        # U-Net: weights đọc 1 lần, activations ≈ 5× input (skip connections)
        total_bytes = params * (4 if precision == "fp32" else 2 if precision == "fp16" else 1)
        total_bytes += h * w * 3 * 4 * 5  # activation traffic
        mem_time_ms = (total_bytes / (dev["mem_bw_gbs"] * 1e9)) * 1000

        # Kernel launch overhead
        kernel_overhead_ms = estimated_kernel_launches * kernel_overhead_per_launch_ms

        # Framework overhead (PyTorch/TensorRT/ONNX runtime)
        framework_overhead_ms = 0.5  # ~0.5ms baseline

        # Total latency
        latency_ms = max(compute_time_ms, mem_time_ms) + kernel_overhead_ms + framework_overhead_ms
        latency_ms = round(latency_ms, 2)

        fps = round(1000.0 / latency_ms, 1) if latency_ms > 0 else float("inf")

        if compute_time_ms > mem_time_ms:
            bottleneck = "Compute"
        else:
            bottleneck = "Memory BW"

        # Memory feasibility
        mem_est = estimate_memory(params, input_hw, num_tasks=2)
        fits_ram = "✓" if mem_est["peak_inference_mb"] < dev["ram_mb"] * 0.7 else "✗"

        results[key] = {
            "device": dev["name"],
            "precision": precision,
            "backend": dev["backend"],
            "latency_ms": latency_ms,
            "fps": fps,
            "bottleneck": bottleneck,
            "fits_ram": fits_ram,
            "peak_mem_mb": mem_est["peak_inference_mb"],
            "ram_mb": dev["ram_mb"],
            "power_w": dev["power_w"],
            "compute_bound_ms": round(compute_time_ms, 2),
            "memory_bound_ms": round(mem_time_ms, 2),
            "kernel_overhead_ms": round(kernel_overhead_ms, 2),
        }

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN REPORT
# ═══════════════════════════════════════════════════════════════
def print_separator(title: str, char: str = "═", width: int = 80):
    print(f"\n{' ' + title + ' ':{char}^{width}}")


def main():
    parser = argparse.ArgumentParser(description="U-MobileViT-Net FLOPs & Edge Analysis")
    parser.add_argument("--alpha", type=float, default=1.0, help="Width multiplier")
    parser.add_argument("--input-height", type=int, default=320)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--sweep", action="store_true", help="Sweep alpha & input sizes")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "int8"])
    args = parser.parse_args()

    H, W = args.input_height, args.input_width
    alpha = args.alpha

    print("╔" + "═" * 78 + "╗")
    print("║" + f"  U-MobileViT-Net: PHÂN TÍCH FLOPs & ĐÁNH GIÁ THIẾT BỊ BIÊN".center(78) + "║")
    print("║" + f"  alpha={alpha}, input=({3}×{H}×{W}), precision={args.precision}".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # ── 1. PROFILE ──────────────────────────────────────────
    print_separator("1. THỐNG KÊ TỔNG QUAN")
    result = profile_model(alpha, (H, W))
    model = result["model"]
    pt = result["params_total"]
    ft = result["flops_total"]

    print(f"""
    Tham số tổng:     {pt:>12,}  ({clever_format([pt], '%.2f')})
    Trainable:        {sum(p.numel() for p in model.parameters() if p.requires_grad):>12,}
    FLOPs tổng:       {ft:>12,.0f}  ({clever_format([ft], '%.2f')})
    FLOPs/Param:      {ft/pt:>12.1f}

    Kích thước file (FP32 weights):  {pt * 4 / (1024**2):.2f} MB
    Kích thước file (FP16 weights):  {pt * 2 / (1024**2):.2f} MB
    Kích thước file (INT8 weights):  {pt * 1 / (1024**2):.2f} MB
    """)

    # ── 2. BREAKDOWN PARAMS ─────────────────────────────────
    print_separator("2. PHÂN BỔ THAM SỐ THEO THÀNH PHẦN")
    comps = count_params_by_component(model)
    for name, p in comps.items():
        if not name.startswith("  "):
            bar = "█" * int(p / pt * 50)
            pct = p / pt * 100
            print(f"  {name:<35} {p:>10,} ({pct:>5.1f}%) {bar}")

    print(f"\n  {'TOTAL':<35} {pt:>10,} (100.0%)")

    # ── 3. BREAKDOWN BY LAYER TYPE ──────────────────────────
    print_separator("3. PHÂN BỔ THAM SỐ THEO LOẠI LỚP")
    type_counts = count_params_by_type(model)
    for t, p in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(p / pt * 50)
        print(f"  {t:<30} {p:>10,} params ({p/pt*100:>5.1f}%) {bar}")

    # ── 4. FLOPs BREAKDOWN ──────────────────────────────────
    print_separator("4. PHÂN BỔ FLOPs (Encoder / Decoder / Head)")
    fe = result["flops_encoder"]
    fd = result["flops_decoder"]
    fh = result["flops_head"]
    print(f"""
    Encoder:        {fe:>12,.0f} FLOPs  ({fe/ft*100:>5.1f}%)  [{clever_format([fe], '%.2f')}]
    Decoder:        {fd:>12,.0f} FLOPs  ({fd/ft*100:>5.1f}%)  [{clever_format([fd], '%.2f')}]
    Segmentation Head: {fh:>12,.0f} FLOPs  ({fh/ft*100:>5.1f}%)  [{clever_format([fh], '%.2f')}]
    """)

    # ── 5. ATTENTION ANALYSIS ───────────────────────────────
    print_separator("5. PHÂN TÍCH ATTENTION")
    total_attn = 0
    for name, mod in model.named_modules():
        if isinstance(mod, SeparableAttention):
            p = sum(p.numel() for p in mod.parameters())
            total_attn += p
            in_ch = mod.in_channels
            print(f"  {name:<55} {p:>8,} params  (in={in_ch})")
    print(f"\n  Tổng Attention params: {total_attn:>10,} ({total_attn/pt*100:.1f}% của model)")

    # ── 6. MEMORY ESTIMATE ──────────────────────────────────
    print_separator("6. ƯỚC LƯỢNG BỘ NHỚ (FP32, batch=1)")
    mem = estimate_memory(pt, (H, W))
    for k, v in mem.items():
        bar = "█" * int(min(v / mem["peak_inference_mb"] * 50, 50))
        print(f"  {k:<30} {v:>8.2f} MB  {bar}")

    # ── 7. EDGE DEVICE EVALUATION ───────────────────────────
    print_separator("7. ĐÁNH GIÁ TRÊN THIẾT BỊ BIÊN")
    latency = estimate_latency(ft, pt, (H, W), precision=args.precision)

    header = f"  {'Thiết bị':<28} {'Latency':>8} {'FPS':>8} {'Mem':>10} {'Fit':>4} {'Power':>6} {'Bottleneck':<12} {'Backend':<18}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for key, dev in latency.items():
        mem_str = f"{dev['peak_mem_mb']:.0f}/{dev['ram_mb']}MB"
        print(f"  {dev['device']:<28} {dev['latency_ms']:>6.1f}ms {dev['fps']:>7.1f} "
              f"{mem_str:>10} {dev['fits_ram']:>4} {dev['power_w']:>4.1f}W "
              f"{dev['bottleneck']:<12} {dev['backend']:<18}")

    # ── 8. QUANTIZATION POTENTIAL ───────────────────────────
    print_separator("8. TIỀM NĂNG QUANTIZATION")
    fp32_lat = latency.get("jetson_nano_4gb", {}).get("latency_ms", 0)
    fp16_lat = estimate_latency(ft, pt, (H, W), precision="fp16")
    int8_lat = estimate_latency(ft, pt, (H, W), precision="int8")

    header = f"{'Precision':<12} {'Model Size':<15} {'Jetson Nano':<15} {'Xavier NX':<15} {'Orin Nano':<15}"
    sep = "─" * 72
    print(f"\n    {header}\n    {sep}")
    for prec, lat_map, size_factor in [
        ("FP32", latency, 4),
        ("FP16", fp16_lat, 2),
        ("INT8", int8_lat, 1),
    ]:
        nano_fps = lat_map.get("jetson_nano_4gb", {}).get("fps", 0)
        xavier_fps = lat_map.get("jetson_xavier_nx", {}).get("fps", 0)
        orin_fps = lat_map.get("jetson_orin_nano", {}).get("fps", 0)
        size_mb = pt * size_factor / (1024 ** 2)
        print(f"    {prec:<12} {size_mb:>8.2f} MB     {nano_fps:>7.1f} fps     "
              f"{xavier_fps:>7.1f} fps     {orin_fps:>7.1f} fps")

    # ── 9. SWEEP (nếu yêu cầu) ──────────────────────────────
    if args.sweep:
        print_separator("9. SWEEP: ALPHA (WIDTH MULTIPLIER)")
        alphas = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        print(f"  {'Alpha':<8} {'d_model':<10} {'Params':<15} {'FLOPs':<15} {'Size(MB)':<10} "
              f"{'vs α=1.0':<10} {'JetsonNano':<12}")
        print("  " + "─" * 85)
        base_result = None
        for a in alphas:
            try:
                r = profile_model(a, (H, W))
                if a == 1.0:
                    base_result = r
                ratio = r["flops_total"] / base_result["flops_total"] if base_result else 1.0
                p_str = clever_format([r["params_total"]], "%.2f")
                f_str = clever_format([r["flops_total"]], "%.2f")
                size_mb = r["params_total"] * 4 / (1024 ** 2)
                l = estimate_latency(r["flops_total"], r["params_total"], (H, W))
                nano_fps = l.get("jetson_nano_4gb", {}).get("fps", 0)
                print(f"  {a:<8} {int(a*64):<10} {p_str:<15} {f_str:<15} "
                      f"{size_mb:<10.2f} {ratio:<10.2f}x {nano_fps:>7.1f} fps")
            except Exception as e:
                # Với alpha rất nhỏ, GroupNorm có thể lỗi (num_channels < num_groups)
                print(f"  {a:<8} {'ERROR':<10} {'(num_groups > channels)':<40}")

        print_separator("10. SWEEP: KÍCH THƯỚC ĐẦU VÀO")
        sizes = [(128, 128), (160, 160), (224, 224), (256, 256),
                 (320, 320), (384, 384), (512, 512), (640, 320), (640, 640)]
        print(f"  {'Size':<15} {'FLOPs':<18} {'Params':<15} {'JetsonNano':<12} {'RPi5':<12}")
        print("  " + "─" * 75)
        for sz_h, sz_w in sizes:
            try:
                r = profile_model(alpha, (sz_h, sz_w))
                f_str = clever_format([r["flops_total"]], "%.2f")
                p_str = clever_format([r["params_total"]], "%.2f")
                l = estimate_latency(r["flops_total"], r["params_total"], (sz_h, sz_w))
                nano_fps = l.get("jetson_nano_4gb", {}).get("fps", 0)
                pi_fps = l.get("raspberry_pi_5", {}).get("fps", 0)
                print(f"  {sz_h:>4}×{sz_w:<10} {f_str:<18} {p_str:<15} "
                      f"{nano_fps:>7.1f} fps  {pi_fps:>7.1f} fps")
            except Exception as e:
                print(f"  {sz_h:>4}×{sz_w:<10} ERROR: {str(e)[:40]}")

    # ── SUMMARY ─────────────────────────────────────────────
    print_separator("TÓM TẮT & KHUYẾN NGHỊ")
    l = estimate_latency(ft, pt, (H, W))
    print(f"""
  Mô hình:       U-MobileViT-Net (alpha={alpha})
  Tham số:       {pt:,} ({pt*4/1024**2:.1f} MB FP32)
  FLOPs:         {ft:,.0f} @ {H}×{W}

  PHÂN LOẠI THIẾT BỊ PHÙ HỢP:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Realtime (≥30fps):  Jetson Xavier NX, Orin Nano, Smartphone cao cấp │
  │ Khả dụng (≥10fps):  Jetson Nano, Smartphone tầm trung               │
  │ Offline (<10fps):    Raspberry Pi 5 (cần tối ưu thêm)               │
  └─────────────────────────────────────────────────────────────────────┘

  KHUYẾN NGHỊ TỐI ƯU:
  1. Dùng FP16/INT8 quantization → tăng 2-4× FPS trên Jetson
  2. Giảm alpha xuống 0.5 → FLOPs giảm ~4×, vẫn giữ chất lượng khá
  3. Giảm input size xuống 256×256 → FLOPs giảm ~1.6×
  4. TensorRT optimization → cải thiện 20-40% latency
  5. Pruning attention heads không cần thiết ở shallow layers
  6. Fuse Conv-BN-ReLU thành 1 kernel với TensorRT
    """)

    print("═" * 80)
    print("  Phân tích hoàn tất!")
    print("═" * 80)


if __name__ == "__main__":
    main()
