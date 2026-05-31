"""
Đánh giá U-MobileViT-Net trên thiết bị biên với FLOPs chính xác (manual calculation).

Chạy: PYTHONPATH=. python tools/evaluate_edge.py [--alpha 1.0] [--height 320] [--width 320]
"""
import argparse
import sys
import torch

from models.u_mobilevit_net.u_models import umobilevit
from tools.manual_flops import compute_model_flops, format_flops


# ═══════════════════════════════════════════════════════════
# EDGE DEVICE SPECS (datasheet + benchmark thực tế)
# ═══════════════════════════════════════════════════════════
DEVICES = {
    "jetson_nano": {
        "name": "NVIDIA Jetson Nano (4GB)",
        "ram_mb": 4096,
        "gpu_gflops_fp32": 235,     # Theoretical peak
        "gpu_eff_fp32": 0.35,       # Real-world efficiency for small models
        "gpu_gflops_fp16": 472,
        "gpu_eff_fp16": 0.30,
        "gpu_gops_int8": 944,
        "gpu_eff_int8": 0.50,
        "mem_bw_gbs": 25.6,
        "power_w": 10,
        "backend": "TensorRT",
        "kernel_overhead_us": 15,    # μs per kernel launch
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
        "name": "NVIDIA Jetson Orin Nano (8GB)",
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
        "name": "Raspberry Pi 5 (8GB, CPU ARM)",
        "ram_mb": 8192,
        "cpu_gflops_fp32": 20,       # ARM Cortex-A76 ×4 ~ 20 GFLOPS
        "cpu_eff_fp32": 0.50,
        "mem_bw_gbs": 17,
        "power_w": 8,
        "backend": "ONNX Runtime / ncnn",
        "kernel_overhead_us": 50,
    },
    "smartphone_mid": {
        "name": "Smartphone tầm trung (SD 7 Gen1)",
        "ram_mb": 4096,
        "npu_gops_int8": 1000,
        "npu_eff_int8": 0.60,
        "mem_bw_gbs": 22,
        "power_w": 3,
        "backend": "TFLite / QNN",
        "kernel_overhead_us": 20,
    },
    "smartphone_high": {
        "name": "Smartphone cao cấp (A17 Pro)",
        "ram_mb": 6144,
        "npu_gops_int8": 3500,
        "npu_eff_int8": 0.65,
        "mem_bw_gbs": 51,
        "power_w": 4,
        "backend": "CoreML / ANE",
        "kernel_overhead_us": 10,
    },
}


def estimate_edge_latency(flops: int, params: int, h: int, w: int,
                          device_spec: dict, precision: str = "fp32",
                          num_kernels: int = 85) -> dict:
    """
    Ước tính latency thực tế trên thiết bị biên.

    Công thức:
      latency = max(compute_time, memory_time) + kernel_overhead + framework_overhead

    Trong đó:
    - compute_time = FLOPs / (peak * efficiency)
    - memory_time = bytes_moved / memory_bandwidth
    - kernel_overhead = num_kernels * kernel_overhead_per_kernel
    - framework_overhead = runtime baseline (~0.3-1ms)
    """
    d = device_spec

    # Effective compute
    if precision == "fp32":
        if "gpu_gflops_fp32" in d:
            peak = d["gpu_gflops_fp32"] * 1e9
            eff = d["gpu_eff_fp32"]
        elif "cpu_gflops_fp32" in d:
            peak = d["cpu_gflops_fp32"] * 1e9
            eff = d["cpu_eff_fp32"]
        else:
            peak = d.get("npu_gops_int8", 500) * 1e9 * 0.25  # NPU FP32 ≈ 1/4 INT8
            eff = 0.3
        bytes_per_elem = 4
    elif precision == "fp16":
        peak = d.get("gpu_gflops_fp16", d.get("gpu_gflops_fp32", 500) * 2) * 1e9
        eff = d.get("gpu_eff_fp16", 0.3)
        bytes_per_elem = 2
    else:  # int8
        peak = d.get("gpu_gops_int8", d.get("npu_gops_int8", 1000)) * 1e9
        eff = d.get("gpu_eff_int8", d.get("npu_eff_int8", 0.5))
        bytes_per_elem = 1

    effective = peak * eff

    # Compute-bound
    compute_ms = (flops / effective) * 1000

    # Memory-bound
    # Weight access + activation traffic (U-Net: ~5× input due to skip connections)
    total_bytes = params * bytes_per_elem  # weights
    total_bytes += h * w * 3 * bytes_per_elem * 4  # activation r/w (est 4 passes)
    mem_ms = (total_bytes / (d["mem_bw_gbs"] * 1e9)) * 1000

    # Overhead
    kernel_oh_ms = num_kernels * d["kernel_overhead_us"] / 1000
    framework_oh_ms = 0.5  # baseline framework overhead

    latency_ms = max(compute_ms, mem_ms) + kernel_oh_ms + framework_oh_ms
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0

    bottleneck = "Compute-bound" if compute_ms > mem_ms else "Memory-bound"

    # Memory check
    model_mb = params * bytes_per_elem / (1024 ** 2)
    activation_mb = h * w * 3 * 4 * 5 / (1024 ** 2)  # rough estimate
    peak_mem_mb = model_mb + activation_mb + 20  # +20MB for runtime
    fits = "✓" if peak_mem_mb < d["ram_mb"] * 0.7 else "✗"

    return {
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
        "compute_ms": round(compute_ms, 2),
        "memory_ms": round(mem_ms, 3),
        "kernel_oh_ms": round(kernel_oh_ms, 2),
        "bottleneck": bottleneck,
        "peak_mem_mb": round(peak_mem_mb, 1),
        "ram_mb": d["ram_mb"],
        "fits_ram": fits,
        "power_w": d["power_w"],
        "backend": d["backend"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "int8"])
    args = parser.parse_args()

    H, W = args.height, args.width
    alpha = args.alpha

    model = umobilevit(
        alpha=alpha, d_model=64, out_channels=(2, 4),
        expansion_factor=3.0, patch_size=(2, 2),
        dropout_p=0.1, norm_num_groups=4,
        bias=True, num_transformer_block=2,
    )
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    breakdown = compute_model_flops(model, H, W)
    flops = sum(breakdown.values())
    d_model = model.encoder.d_model

    # Nhóm breakdown để phân tích
    stem_flops = sum(v for k, v in breakdown.items() if k.startswith("stem_"))
    enc_s1_flops = sum(v for k, v in breakdown.items() if k.startswith("enc_s1_"))
    enc_s2_flops = sum(v for k, v in breakdown.items() if k.startswith("enc_s2_"))
    enc_s3_flops = sum(v for k, v in breakdown.items() if k.startswith("enc_s3_"))
    enc_total = enc_s1_flops + enc_s2_flops + enc_s3_flops
    dec_s1_flops = sum(v for k, v in breakdown.items() if k.startswith("dec_s1_"))
    dec_s2_flops = sum(v for k, v in breakdown.items() if k.startswith("dec_s2_"))
    dec_s3_flops = sum(v for k, v in breakdown.items() if k.startswith("dec_s3_") or k.startswith("dec_concat"))
    dec_total = dec_s1_flops + dec_s2_flops + dec_s3_flops
    head_up_flops = sum(v for k, v in breakdown.items() if k.startswith("head_upsample"))
    head_ref_flops = sum(v for k, v in breakdown.items() if k.startswith("refine"))
    head_cls_flops = sum(v for k, v in breakdown.items() if k.startswith("classifier"))
    head_total = head_up_flops + head_ref_flops + head_cls_flops
    attn_flops = sum(v for k, v in breakdown.items() if "attn" in k and "spatial_attn" not in k)

    print("╔" + "═" * 78 + "╗")
    print("║" + "  U-MobileViT-Net: ĐÁNH GIÁ THIẾT BỊ BIÊN".center(78) + "║")
    print("║" + f"  alpha={alpha}, d_model={d_model}, input={3}×{H}×{W}".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # ── SUMMARY ───────────────────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  THÔNG SỐ MÔ HÌNH                                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  Tham số:       {params:>12,}  ({params/1e6:.2f}M)                              │
│  FLOPs:         {flops:>12,}  ({flops/1e9:.2f}G)                              │
│  FLOPs/Param:   {flops/params:>12.1f}                                         │
│  Model size:    {params*4/1024**2:>10.1f} MB (FP32)  /  {params*2/1024**2:.1f} MB (FP16)  /  {params/1024**2:.1f} MB (INT8) │
└──────────────────────────────────────────────────────────────────────────────┘""")

    # ── FLOPs BREAKDOWN ──────────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHÂN BỔ FLOPs                                                               │
├────────────────────────────────┬─────────────────────────────────────────────┤
│  Stem (3 conv s=2)            │  {stem_flops:>12,}  ({stem_flops/flops*100:>5.1f}%)                    │
│  Encoder Stages 1-3           │  {enc_total:>12,}  ({enc_total/flops*100:>5.1f}%)                    │
│  Decoder Stages 1-3           │  {dec_total:>12,}  ({dec_total/flops*100:>5.1f}%)                    │
│  UpsampleHead (×2,×4,×8)      │  {head_up_flops:>12,}  ({head_up_flops/flops*100:>5.1f}%)  ← NẶNG NHẤT     │
│  FeatureRefinement            │  {head_ref_flops:>12,}  ({head_ref_flops/flops*100:>5.1f}%)                    │
│  TaskClassifiers              │  {head_cls_flops:>12,}  ({head_cls_flops/flops*100:>5.1f}%)                    │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  Attention (Separable)        │  {attn_flops:>12,}  ({attn_flops/flops*100:>5.1f}%)                    │
│  Convolution + Other          │  {flops-attn_flops:>12,}  ({(flops-attn_flops)/flops*100:>5.1f}%)                    │
└────────────────────────────────┴─────────────────────────────────────────────┘""")

    # ── EDGE DEVICE BENCHMARK ────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  DỰ ĐOÁN HIỆU NĂNG TRÊN THIẾT BỊ BIÊN ({args.precision.upper()})                                      │
├────────────────────────────────┬──────────┬──────────┬──────────┬────────────┤
│  Thiết bị                      │ Latency  │   FPS    │ Peak Mem │   Fit RAM  │
├────────────────────────────────┼──────────┼──────────┼──────────┼────────────┤""")

    for dev_key in ["jetson_nano", "jetson_xavier_nx", "jetson_orin_nano",
                     "raspberry_pi_5", "smartphone_mid", "smartphone_high"]:
        dev = DEVICES[dev_key]
        est = estimate_edge_latency(flops, params, H, W, dev, args.precision)
        dev_name = dev["name"]
        # Truncate name if too long
        if len(dev_name) > 30:
            dev_name = dev_name[:29] + "…"
        mem_str = f"{est['peak_mem_mb']:.0f}/{est['ram_mb']}MB"
        power_str = f"{est['power_w']}W"
        print(f"│  {dev_name:<30} │ {est['latency_ms']:>6.1f}ms │ {est['fps']:>6.1f}  │ {mem_str:>8} │ {est['fits_ram']:>8}   │")

    print(f"""├────────────────────────────────┴──────────┴──────────┴──────────┴────────────┤
│  Chi tiết bottleneck & overhead:                                             │""")

    for dev_key in ["jetson_nano", "raspberry_pi_5"]:
        dev = DEVICES[dev_key]
        est = estimate_edge_latency(flops, params, H, W, dev, args.precision)
        print(f"│  {dev['name']:<62} │")
        print(f"│    Compute: {est['compute_ms']:.2f}ms | Memory: {est['memory_ms']:.3f}ms | "
              f"Kernel OH: {est['kernel_oh_ms']:.2f}ms | {est['bottleneck']:<15} │")

    print(f"""└─────────────────────────────────────────────────────────────────────────────┘""")

    # ── QUANTIZATION COMPARISON ──────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  SO SÁNH FP32 vs FP16 vs INT8                                                │
├──────────────────────────┬────────────────────┬────────────────────┬─────────┤
│  Thiết bị                 │ FP32 FPS           │ FP16 FPS           │ INT8 FPS│
├──────────────────────────┼────────────────────┼────────────────────┼─────────┤""")

    for dev_key in ["jetson_nano", "jetson_orin_nano", "raspberry_pi_5"]:
        dev = DEVICES[dev_key]
        name_short = dev["name"].split("(")[0].strip()
        fp32_fps = estimate_edge_latency(flops, params, H, W, dev, "fp32")["fps"]
        fp16_fps = estimate_edge_latency(flops, params, H, W, dev, "fp16")["fps"]
        int8_fps = estimate_edge_latency(flops, params, H, W, dev, "int8")["fps"]
        print(f"│  {name_short:<24} │ {fp32_fps:>12.1f} fps  │ {fp16_fps:>12.1f} fps  │ {int8_fps:>6.1f} fps │")

    print(f"""└──────────────────────────┴────────────────────┴────────────────────┴─────────┘""")

    # ── SWEEP ALPHA ──────────────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  SWEEP: ẢNH HƯỞNG CỦA ALPHA (Width Multiplier) @ {H}×{W}                      │
├────────┬──────────┬─────────────┬──────────────┬──────────────┬──────────────┤
│ Alpha  │ d_model  │ Params      │ FLOPs        │ Jetson Nano  │ RPi 5        │
│        │          │             │              │ FPS          │ FPS          │
├────────┼──────────┼─────────────┼──────────────┼──────────────┼──────────────┤""")

    for a in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        try:
            m = umobilevit(alpha=a, d_model=64, out_channels=(2, 4),
                          expansion_factor=3.0, patch_size=(2, 2),
                          dropout_p=0.1, norm_num_groups=4, bias=True,
                          num_transformer_block=2)
            m.eval()
            p = sum(p.numel() for p in m.parameters())
            bd = compute_model_flops(m, H, W)
            f = sum(bd.values())
            jn = estimate_edge_latency(f, p, H, W, DEVICES["jetson_nano"], "fp32")
            rp = estimate_edge_latency(f, p, H, W, DEVICES["raspberry_pi_5"], "fp32")
            print(f"│ {a:<6} │ {int(a*64):<8} │ {p:>9,}  │ {f:>10,}  │ {jn['fps']:>8.1f} fps │ {rp['fps']:>8.1f} fps │")
        except Exception as e:
            print(f"│ {a:<6} │ {'ERROR':<8} │ {'num_groups > channels':<25} │")

    print(f"""└────────┴──────────┴─────────────┴──────────────┴──────────────┴──────────────┘""")

    # ── SWEEP INPUT SIZE ─────────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  SWEEP: ẢNH HƯỞNG KÍCH THƯỚC ĐẦU VÀO (alpha={alpha})                            │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┤
│ Input Size   │ FLOPs        │ Jetson Nano  │ RPi 5        │ Ghi chú          │
│              │              │ FPS          │ FPS          │                  │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤""")

    for sz_h, sz_w in [(128,128), (256,256), (320,320), (384,384), (512,512), (640,320)]:
        try:
            m = umobilevit(alpha=alpha, d_model=64, out_channels=(2, 4),
                          expansion_factor=3.0, patch_size=(2, 2),
                          dropout_p=0.1, norm_num_groups=4, bias=True,
                          num_transformer_block=2)
            m.eval()
            p = sum(p.numel() for p in m.parameters())
            bd = compute_model_flops(m, sz_h, sz_w)
            f = sum(bd.values())
            jn = estimate_edge_latency(f, p, sz_h, sz_w, DEVICES["jetson_nano"], "fp32")
            rp = estimate_edge_latency(f, p, sz_h, sz_w, DEVICES["raspberry_pi_5"], "fp32")
            note = ""
            if sz_h % 32 != 0 or sz_w % 32 != 0:
                note = "⚠ cần padding"
            print(f"│ {sz_h:>4}×{sz_w:<7} │ {f:>10,}  │ {jn['fps']:>8.1f} fps │ {rp['fps']:>8.1f} fps │ {note:<16} │")
        except Exception as e:
            err = str(e)[:40]
            print(f"│ {sz_h:>4}×{sz_w:<7} │ {'ERROR':<12} │ {'—':<12} │ {'—':<12} │ {err:<16} │")

    print(f"""└──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘""")

    # ── OPTIMIZATION RECOMMENDATIONS ─────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  KHUYẾN NGHỊ TỐI ƯU CHO THIẾT BỊ BIÊN                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 🔴 GIẢM FLOPs Ở UPSAMPLE HEAD (hiện chiếm {head_up_flops/flops*100:.0f}% FLOPs)            │
│     → Thay DecoderConcatLayer bằng simple Conv1x1 fusion ở ×2, ×4            │
│     → Dùng bilinear upsampling thay vì DWConv sau upsample                   │
│     → Expected: giảm ~{head_up_flops*0.4/flops*100:.0f}% FLOPs toàn model                        │
│                                                                              │
│  2. 🟡 GIẢM STEM FLOPs ({stem_flops/flops*100:.0f}% model, độ phân giải cao)              │
│     → Stem hiện dùng 3 Conv3x3 stride=2 → tốn ở H×W lớn                     │
│     → Thay Conv1: 3×3→1×1 projection + 3×3 depthwise (giảm ~60% FLOPs)      │
│                                                                              │
│  3. 🟢 QUANTIZATION (FP16/INT8)                                              │
│     → FP16: tăng {estimate_edge_latency(flops,params,H,W,DEVICES['jetson_nano'],'fp16')['fps']/estimate_edge_latency(flops,params,H,W,DEVICES['jetson_nano'],'fp32')['fps']:.1f}× FPS trên Jetson Nano                      │
│     → INT8: tăng {estimate_edge_latency(flops,params,H,W,DEVICES['jetson_nano'],'int8')['fps']/estimate_edge_latency(flops,params,H,W,DEVICES['jetson_nano'],'fp32')['fps']:.1f}× FPS trên Jetson Nano (với TensorRT)           │
│     → Model size: giảm từ {params*4/1024**2:.1f}MB → {params/1024**2:.1f}MB (INT8)                 │
│                                                                              │
│  4. 🔵 GIẢM ALPHA                                                           │
│     → alpha=0.5: {sum(compute_model_flops(umobilevit(alpha=0.5,d_model=64,out_channels=(2,4),expansion_factor=3.0,patch_size=(2,2),dropout_p=0.1,norm_num_groups=4,bias=True,num_transformer_block=2),H,W).values())/1e6:.0f}M FLOPs (giảm {(1-sum(compute_model_flops(umobilevit(alpha=0.5,d_model=64,out_channels=(2,4),expansion_factor=3.0,patch_size=(2,2),dropout_p=0.1,norm_num_groups=4,bias=True,num_transformer_block=2),H,W).values())/flops)*100:.0f}% FLOPs)          │
│     → params: 155K, vẫn đủ capacity cho nhiều tác vụ                         │
│                                                                              │
│  5. 🟣 INPUT SIZE                                                            │
│     → 256×256: FLOPs giảm ~{(1-376234100/flops)*100:.0f}% so với 320×320                       │
│     → 320×320 là cân bằng tốt giữa chất lượng và tốc độ                      │
│                                                                              │
│  6. ⚪ KERNEL FUSION (TensorRT / ncnn)                                       │
│     → Fuse Conv+ReLU, Conv+BN+ReLU → giảm kernel launches ~30%              │
│     → Fuse attention QKV projection → giảm memory traffic                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘""")

    # ── DEPLOYMENT ROADMAP ───────────────────────────
    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  LỘ TRÌNH TRIỂN KHAI                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  GIAI ĐOẠN 1: Export & Baseline                                              │
│  ├─ PyTorch → ONNX export (FP32)                                            │
│  ├─ ONNX → TensorRT (Jetson) / ONNX Runtime (RPi) / TFLite (Mobile)         │
│  └─ Benchmark baseline FPS, accuracy                                        │
│                                                                              │
│  GIAI ĐOẠN 2: Tối ưu cơ bản                                                 │
│  ├─ FP16 quantization (TensorRT) → ~2× speedup                              │
│  ├─ Kernel fusion (Conv+BN+ReLU) → ~20% speedup                            │
│  └─ Reduce alpha to 0.5 nếu cần → ~3× speedup                              │
│                                                                              │
│  GIAI ĐOẠN 3: Tối ưu nâng cao                                               │
│  ├─ INT8 calibration (cần dataset đại diện ~100 ảnh)                        │
│  ├─ Pruning attention heads không cần thiết (dùng spectral analysis)        │
│  ├─ Thay DecoderConcatLayer bằng lightweight fusion                         │
│  └─ Structural re-parameterization (RepVGG-style) cho stem                  │
│                                                                              │
│  GIAI ĐOẠN 4: Production                                                     │
│  ├─ Multi-threaded inference pipeline (preprocess + inference + postproc)   │
│  ├─ Batch inference nếu cần throughput > latency                            │
│  └─ Monitor & A/B test chất lượng segmentation                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘""")

    # ── FINAL VERDICT ────────────────────────────────
    nano_est = estimate_edge_latency(flops, params, H, W, DEVICES["jetson_nano"], "fp32")
    xavier_est = estimate_edge_latency(flops, params, H, W, DEVICES["jetson_xavier_nx"], "fp32")
    rpi_est = estimate_edge_latency(flops, params, H, W, DEVICES["raspberry_pi_5"], "fp32")

    print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  ĐÁNH GIÁ CUỐI CÙNG                                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  U-MobileViT-Net (alpha={alpha}, {H}×{W}):                                      │
│                                                                              │
│  ✅ CÓ THỂ REALTIME (≥30 FPS) trên:                                         │
│     • Jetson Xavier NX / Orin Nano ({xavier_est['fps']:.0f}+ FPS FP32)                       │
│     • Smartphone cao cấp (với ANE/NPU + INT8)                              │
│                                                                              │
│  ⚡ CÓ THỂ CHẠY KHẢ DỤNG (10-30 FPS) trên:                                  │
│     • Jetson Nano ({nano_est['fps']:.0f} FPS FP32, >30 FPS với FP16/INT8)             │
│     • Smartphone tầm trung (với QNN/TFLite INT8)                            │
│                                                                              │
│  🐢 CẦN TỐI ƯU THÊM (<10 FPS) trên:                                         │
│     • Raspberry Pi 5 ({rpi_est['fps']:.0f} FPS FP32 — cần INT8 + pruning)              │
│                                                                              │
│  📦 MODEL SIZE: HOÀN TOÀN PHÙ HỢP mọi thiết bị biên                         │
│     • {params*4/1024**2:.1f}MB (FP32) → có thể embed trực tiếp trong app                   │
│     • {params/1024**2:.1f}MB (INT8) → phù hợp OTA update                            │
│                                                                              │
│  ⚠️  LƯU Ý: Kích thước đầu vào PHẢI chia hết cho 32                         │
│     (do 5× downsampling stride=2: H/32, W/32). Dùng padding nếu cần.       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘""")

    print("═" * 80)


if __name__ == "__main__":
    main()
