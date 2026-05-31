"""
Công cụ export & benchmark UMobileViT cho thiết bị biên (Jetson Nano/Orin, ...).

Chức năng:
    1. Dựng model UMobileViT (tùy chọn nạp checkpoint).
    2. Fuse Conv+BatchNorm (nếu có) để giảm số op lúc suy luận.
    3. Export TorchScript (.pt) và ONNX (.onnx, dùng cho TensorRT).
    4. Benchmark latency/FPS: PyTorch FP32 & FP16 trên CUDA, INT8 dynamic trên CPU.
    5. Kiểm tra tính đúng đắn của file ONNX bằng onnxruntime.

Ví dụ:
    # Export + benchmark đầy đủ cho ảnh 320x320, model 2 tác vụ (2 và 4 lớp)
    PYTHONPATH=. python tools/export_edge.py --img-size 320 --out-channels 2 4 \
        --onnx --torchscript --benchmark --fp16

    # Nạp trọng số đã train rồi export
    PYTHONPATH=. python tools/export_edge.py --weights runs/best.pt --onnx

Gợi ý dùng trên Jetson:
    trtexec --onnx=umobilevit.onnx --saveEngine=umobilevit_fp16.engine --fp16
"""
import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn

from models.u_mobilevit_net.u_models import umobilevit, UMobileViT


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
class _TupleToList(nn.Module):
    """Bọc model trả về tuple thành list để TorchScript/ONNX export ổn định."""
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return list(out)
        return [out]


def build_model(args: argparse.Namespace) -> nn.Module:
    out_channels = tuple(args.out_channels)
    if len(out_channels) == 1:
        out_channels = out_channels[0]  # int -> single task khi head=single

    model = umobilevit(
        opts=args,
        head=args.head_type,
        out_channels=out_channels,
        alpha=args.alpha,
    )

    if args.weights:
        ckpt = torch.load(args.weights, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[weights] loaded '{args.weights}' | missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    return model


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """
    Fuse Conv2d + BatchNorm2d liền kề trong các Sequential (giảm op khi suy luận,
    rất hợp TensorRT). Model mặc định dùng GroupNorm nên đây thường là no-op,
    nhưng vẫn an toàn nếu sau này đổi sang BatchNorm.
    """
    try:
        from torch.ao.quantization import fuse_modules
    except Exception:
        return model

    fused = 0
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        names = list(dict(module.named_children()).keys())
        for i in range(len(names) - 1):
            m1 = getattr(module, names[i])
            m2 = getattr(module, names[i + 1])
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                try:
                    fuse_modules(module, [names[i], names[i + 1]], inplace=True)
                    fused += 1
                except Exception:
                    pass
    print(f"[fuse] đã fuse {fused} cặp Conv+BN")
    return model


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@torch.no_grad()
def benchmark(model: nn.Module, x: torch.Tensor, n_warmup: int = 10, n_iter: int = 50, device="cpu", tag="") -> float:
    model = model.to(device)
    x = x.to(device)
    cuda = device != "cpu" and torch.cuda.is_available()

    for _ in range(n_warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter

    print(f"[bench] {tag:18s} {dt * 1000:7.2f} ms/img  ({1.0 / dt:6.1f} FPS)  @ {tuple(x.shape)} on {device}")
    return dt


def run_benchmarks(model: nn.Module, args: argparse.Namespace) -> None:
    x = torch.randn(args.batch, 3, args.img_size, args.img_size)

    # CPU FP32
    torch.set_num_threads(args.cpu_threads)
    benchmark(model, x, device="cpu", tag=f"CPU FP32 ({args.cpu_threads}thr)")

    # CPU INT8 dynamic (lượng tử hóa nhanh, không cần calib) — tham khảo cho biên CPU
    try:
        qmodel = torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        benchmark(qmodel, x, device="cpu", tag="CPU INT8 dynamic")
    except Exception as e:
        print(f"[bench] INT8 dynamic bỏ qua: {e}")

    # CUDA FP32 (FP16 xử lý riêng ở benchmark_fp16_cuda để tránh đổi dtype model)
    if torch.cuda.is_available():
        benchmark(model, x, device="cuda", tag="CUDA FP32")
        model.float().to("cpu")  # khôi phục trạng thái trước khi export
    else:
        print("[bench] Không có CUDA — bỏ benchmark GPU (Jetson sẽ chạy được).")


@torch.no_grad()
def benchmark_fp16_cuda(model: nn.Module, args: argparse.Namespace) -> None:
    if not (args.fp16 and torch.cuda.is_available()):
        return
    m = model.to("cuda").half()
    x = torch.randn(args.batch, 3, args.img_size, args.img_size, device="cuda", dtype=torch.float16)
    benchmark(m, x, device="cuda", tag="CUDA FP16")
    model.float()  # khôi phục để export sau đó không bị half


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_torchscript(model: nn.Module, x: torch.Tensor, path: str) -> None:
    wrapped = _TupleToList(model).eval()
    ts = torch.jit.trace(wrapped, x, strict=False)
    ts = torch.jit.optimize_for_inference(ts)
    ts.save(path)
    print(f"[torchscript] đã lưu {path}")


def export_onnx(model: nn.Module, x: torch.Tensor, path: str, dynamic_batch: bool, opset: int, n_outputs: int) -> None:
    wrapped = _TupleToList(model).eval()
    output_names = [f"task_{i}" for i in range(n_outputs)]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})

    torch.onnx.export(
        wrapped, x, path,
        input_names=["input"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"[onnx] đã lưu {path} (opset={opset}, dynamic_batch={dynamic_batch})")

    # Kiểm tra cấu trúc + đối chiếu output với PyTorch qua onnxruntime
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        onnx.checker.check_model(onnx.load(path))
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": x.cpu().numpy()})
        with torch.no_grad():
            ref = wrapped(x)
        max_diff = max(float(np.abs(o - r.cpu().numpy()).max()) for o, r in zip(ort_out, ref))
        print(f"[onnx] check OK | sai lệch lớn nhất so với PyTorch: {max_diff:.2e}")
    except Exception as e:
        print(f"[onnx] bỏ qua bước verify: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export & benchmark UMobileViT cho thiết bị biên")
    UMobileViT.add_arguments(p)
    g = p.add_argument_group("Export/Benchmark")
    g.add_argument("--weights", type=str, default=None, help="Đường dẫn checkpoint (.pt)")
    g.add_argument("--out-channels", type=int, nargs="+", default=[2, 4], help="Số lớp mỗi tác vụ, vd: 2 4")
    g.add_argument("--img-size", type=int, default=320, help="Kích thước ảnh vuông để export/bench")
    g.add_argument("--batch", type=int, default=1)
    g.add_argument("--cpu-threads", type=int, default=4)
    g.add_argument("--out-dir", type=str, default="exports")
    g.add_argument("--fuse", action="store_true", help="Fuse Conv+BN trước khi export")
    g.add_argument("--torchscript", action="store_true")
    g.add_argument("--onnx", action="store_true")
    g.add_argument("--opset", type=int, default=17)
    g.add_argument("--dynamic-batch", action="store_true", help="ONNX cho phép batch động")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--fp16", action="store_true", help="Thêm benchmark CUDA FP16")
    return p


def main() -> None:
    args, _ = build_parser().parse_known_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] UMobileViT | params={n_params / 1e6:.3f} M | head={args.head_type} | out_channels={args.out_channels}")

    if args.fuse:
        model = fuse_conv_bn(model)

    x = torch.randn(args.batch, 3, args.img_size, args.img_size)
    with torch.no_grad():
        sample = model(x)
    n_outputs = len(sample) if isinstance(sample, (tuple, list)) else 1
    print(f"[model] số nhánh đầu ra = {n_outputs}")

    if args.benchmark:
        run_benchmarks(model, args)
        benchmark_fp16_cuda(model, args)
        model.float().to("cpu")  # đảm bảo model về CPU/FP32 trước khi export

    if args.torchscript:
        export_torchscript(model, x, os.path.join(args.out_dir, "umobilevit.pt"))

    if args.onnx:
        export_onnx(
            model, x, os.path.join(args.out_dir, "umobilevit.onnx"),
            dynamic_batch=args.dynamic_batch, opset=args.opset, n_outputs=n_outputs,
        )

    if not (args.benchmark or args.torchscript or args.onnx):
        print("\nKhông có hành động nào được chọn. Thêm --benchmark / --onnx / --torchscript.")


if __name__ == "__main__":
    main()
