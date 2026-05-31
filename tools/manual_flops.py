"""
Manual FLOPs calculator cho U-MobileViT-Net.
Không phụ thuộc vào thop (thop không nhận diện được custom Conv2d subclass).

Chạy: PYTHONPATH=. python tools/manual_flops.py [--alpha 1.0] [--height 320] [--width 320]
"""
import argparse
import torch
from torch import nn
from typing import Tuple, Dict
from collections import defaultdict

from models.u_mobilevit_net.u_models import umobilevit


def conv2d_flops(in_ch: int, out_ch: int, kernel: int,
                 h_in: int, w_in: int, stride: int = 1,
                 groups: int = 1, bias: bool = True) -> int:
    """Tính FLOPs cho 1 Conv2d layer."""
    h_out = h_in // stride
    w_out = w_in // stride
    # MAC = out * (in/groups) * k * k * h_out * w_out
    mac = out_ch * (in_ch // groups) * kernel * kernel * h_out * w_out
    # 1 MAC ≈ 2 FLOPs (multiply + add)
    flops = 2 * mac
    if bias:
        flops += out_ch * h_out * w_out  # bias add
    return flops


def separable_attn_flops(in_ch: int, patch_h: int, patch_w: int,
                         num_patches: int) -> int:
    """Tính FLOPs cho 1 SeparableAttention layer.

    Shape input: (N, C, P, S) với P = patch_h * patch_w, S = num_patches.

    Operations:
    1. QKV projection: Linear(in_ch → 1 + 2*in_ch) trên [N, P, S, C]
    2. softmax trên Q: [N, 1, P, S]
    3. K * context_scores + sum
    4. ReLU(V) * context_vector
    5. out projection: Linear(in_ch → in_ch)
    """
    P = patch_h * patch_w
    S = num_patches

    # 1. QKV projection (dùng F.linear)
    #    in_proj_weight: (1 + 2*in_ch, in_ch)
    #    F.linear: x @ W^T + b, mỗi vị trí = 1 MAC
    qkv_flops = 2 * (1 + 2 * in_ch) * in_ch * P * S

    # 2. Softmax: khoảng 5 ops mỗi token (exp, sum, div)
    softmax_flops = 5 * P * S

    # 3. Context: K * scores + sum → in_ch * P * S multiply + in_ch * P * S add
    context_flops = 2 * in_ch * P * S

    # 4. ReLU + multiply → 1 * in_ch * P * S for relu + in_ch * P * S for multiply
    activation_flops = 2 * in_ch * P * S

    # 5. Out projection: Linear(in_ch → in_ch)
    out_flops = 2 * in_ch * in_ch * P * S

    return qkv_flops + softmax_flops + context_flops + activation_flops + out_flops


def cross_attn_flops(in_ch: int, patch_h: int, patch_w: int,
                     num_patches: int) -> int:
    """FLOPs cho Cross-Attention (giống Self-Attn nhưng Q,K,V tách biệt)."""
    # Tương tự self-attn nhưng có thêm memory projection
    # Self-attn: Q=X, K=X, V=X
    # Cross-attn: Q=X, K=memory, V=memory
    # FLOPs = 2 * self_attn_flops (vì có cả self + cross)
    # Nhưng thực tế cross-attn dùng chung V projection với self-attn?
    # Trong TransformerDecoderLayer: self_attn + cross_attn riêng biệt
    # Mỗi cái đều có QKV projection đầy đủ
    return separable_attn_flops(in_ch, patch_h, patch_w, num_patches)


def local_block_flops(in_ch: int, h: int, w: int) -> int:
    """FLOPs cho LocalBlock (DWConv3x3 + PWConv1x1 + GroupNorm)."""
    flops = conv2d_flops(in_ch, in_ch, 3, h, w, groups=in_ch, bias=True)  # DW
    flops += conv2d_flops(in_ch, in_ch, 1, h, w, bias=True)  # PW
    # GroupNorm: 2 * C (mean, var) + C * 2 (scale, shift)
    flops += 4 * in_ch * h * w
    return flops


def expansion_block_flops(in_ch: int, exp_factor: float, h: int, w: int,
                          dropout_p: float = 0.1) -> int:
    """FLOPs cho ExpansionBlock (Inverted Bottleneck)."""
    exp_ch = int(in_ch * exp_factor)
    flops = conv2d_flops(in_ch, exp_ch, 1, h, w, bias=True)  # PW expand
    flops += conv2d_flops(exp_ch, exp_ch, 3, h, w, groups=exp_ch, bias=True)  # DW
    flops += conv2d_flops(exp_ch, in_ch, 1, h, w, bias=True)  # PW project
    return flops


def downsample_flops(in_ch: int, h: int, w: int) -> int:
    """Downsample: DWConv3x3 stride=2."""
    return conv2d_flops(in_ch, in_ch, 3, h, w, stride=2, groups=in_ch, bias=True)


def upsample_flops(in_ch: int, h: int, w: int) -> int:
    """Upsample: Nearest + DWConv3x3 (h,w là kích thước input, output gấp đôi)."""
    # Nearest upsampling ≈ 0 FLOPs
    # DWConv sau upsample: input đã được resize lên 2h × 2w
    return conv2d_flops(in_ch, in_ch, 3, 2 * h, 2 * w, groups=in_ch, bias=True)


def compute_model_flops(model, H: int, W: int) -> Dict:
    """Tính FLOPs chi tiết cho U-MobileViT-Net."""

    d = model.encoder.d_model  # 64 * alpha
    alpha = model.alpha
    exp_factor = model.expansion_factor
    patch_h, patch_w = model.patch_size  # (2, 2)
    n_trans = model.num_transformer_block  # 2
    n_tasks = len(model.seg_head.out_channels_list)

    breakdown = defaultdict(int)

    # ── STEM ──────────────────────────────────────────
    # Tầng 1: Conv2d(3 → d/4, k=3, s=2) + ReLU
    flops = conv2d_flops(3, d // 4, 3, H, W, stride=2, bias=True)
    breakdown["stem_conv1"] = flops

    # Tầng 2: Conv2d(d/4 → d/2, k=3, s=2) + ReLU + GN
    flops = conv2d_flops(d // 4, d // 2, 3, H // 2, W // 2, stride=2, bias=True)
    flops += 4 * (d // 2) * (H // 4) * (W // 4)  # GN
    breakdown["stem_conv2"] = flops

    # Tầng 3: Conv2d(d/2 → d, k=3, s=2) + ReLU + GN
    flops = conv2d_flops(d // 2, d, 3, H // 4, W // 4, stride=2, bias=True)
    flops += 4 * d * (H // 8) * (W // 8)  # GN
    breakdown["stem_conv3"] = flops

    # ── ENCODER STAGES ────────────────────────────────
    encoder_resolutions = [
        (H // 16, W // 16, 1, (patch_h, patch_w)),   # Stage 1
        (H // 32, W // 32, 2, (patch_h, patch_w)),   # Stage 2
        (H // 64, W // 64, 2, (1, 1)),                # Stage 3
    ]
    prev_h, prev_w = H // 8, W // 8

    for stage_idx, (sh, sw, n_layers, (ph, pw)) in enumerate(encoder_resolutions):
        # Downsample
        ds_flops = downsample_flops(d, prev_h, prev_w)
        breakdown[f"enc_s{stage_idx+1}_downsample"] = ds_flops

        for layer_idx in range(n_layers):
            prefix = f"enc_s{stage_idx+1}_l{layer_idx}"
            # Local block
            breakdown[f"{prefix}_local"] = local_block_flops(d, sh, sw)
            # Global block (transformer)
            P = ph * pw
            N = (sh // ph) * (sw // pw)
            for t_idx in range(n_trans):
                breakdown[f"{prefix}_attn{t_idx}"] = separable_attn_flops(d, ph, pw, N)
            # Expansion block
            breakdown[f"{prefix}_expansion"] = expansion_block_flops(d, exp_factor, sh, sw)

        prev_h, prev_w = sh, sw

    # ── DECODER ───────────────────────────────────────
    decoder_configs = [
        (H // 64, W // 64, H // 32, W // 32, 2),   # Stage 1: 2 decoder layers
        (H // 32, W // 32, H // 16, W // 16, 1),   # Stage 2: 1 decoder layer
        (H // 16, W // 16, H // 8, W // 8, 0),      # Stage 3: ConcatLayer (no transformer)
    ]

    for stage_idx, (in_h, in_w, out_h, out_w, n_layers) in enumerate(decoder_configs):
        # Upsample
        breakdown[f"dec_s{stage_idx+1}_upsample"] = upsample_flops(d, in_h, in_w)

        for layer_idx in range(n_layers):
            prefix = f"dec_s{stage_idx+1}_l{layer_idx}"
            # Local block
            breakdown[f"{prefix}_local"] = local_block_flops(d, out_h, out_w)
            # Global block: Self-Attn + Cross-Attn
            P = 4  # patch = 2×2
            N = (out_h // 2) * (out_w // 2)
            for t_idx in range(n_trans):
                breakdown[f"{prefix}_self_attn{t_idx}"] = separable_attn_flops(d, 2, 2, N)
                breakdown[f"{prefix}_cross_attn{t_idx}"] = cross_attn_flops(d, 2, 2, N)
            # Expansion block
            breakdown[f"{prefix}_expansion"] = expansion_block_flops(d, exp_factor, out_h, out_w)

    # Decoder ConcatLayer (Stage 3, no transformer)
    ch, oh, ow = d, H // 8, W // 8
    breakdown["dec_concat_local"] = local_block_flops(ch, oh, ow)
    breakdown["dec_concat_mem_proj"] = conv2d_flops(ch, ch, 1, oh, ow, bias=True)  # memory proj
    # global block thay thế attention
    breakdown["dec_concat_global_dw"] = conv2d_flops(ch, ch, 3, oh, ow, groups=ch, bias=True)
    breakdown["dec_concat_global_pw"] = conv2d_flops(ch, ch, 1, oh, ow, bias=True)
    breakdown["dec_concat_expansion"] = expansion_block_flops(ch, exp_factor, oh, ow)

    # ── SEGMENTATION HEAD ─────────────────────────────
    # UpsampleHead: ×2, ×4, ×8
    # ×2: d → d/2
    h_step, w_step = H // 8, W // 8
    for step, (ch_in, ch_out) in enumerate([(d, d//2), (d//2, d//4), (d//4, d//8)]):
        prefix = f"head_upsample_x{2**(step+1)}"
        breakdown[f"{prefix}_proj"] = conv2d_flops(ch_in, ch_out, 1, h_step, w_step, bias=True)
        breakdown[f"{prefix}_dwconv"] = conv2d_flops(ch_out, ch_out, 3, 2*h_step, 2*w_step,
                                                      groups=ch_out, bias=True)
        # DecoderConcatLayer ở 2 bước đầu (×2, ×4)
        if step < 2:
            nch, nh, nw = ch_out, 2 * h_step, 2 * w_step
            breakdown[f"{prefix}_concat_local"] = local_block_flops(nch, nh, nw)
            breakdown[f"{prefix}_concat_mem"] = conv2d_flops(nch, nch, 1, nh, nw, bias=True)
            breakdown[f"{prefix}_concat_global_dw"] = conv2d_flops(nch, nch, 3, nh, nw,
                                                                   groups=nch, bias=True)
            breakdown[f"{prefix}_concat_global_pw"] = conv2d_flops(nch, nch, 1, nh, nw, bias=True)
            breakdown[f"{prefix}_concat_exp"] = expansion_block_flops(nch, exp_factor, nh, nw)
        h_step, w_step = 2 * h_step, 2 * w_step

    # FeatureRefinement: DilatedConv + SpatialAttention + MixConv
    rch, rh, rw = d // 8, H, W
    breakdown["refine_dilated"] = conv2d_flops(rch, rch, 3, rh, rw, groups=rch, bias=True)
    breakdown["refine_mix"] = conv2d_flops(rch, rch, 1, rh, rw, bias=True)
    # Spatial attention: avg+max pool + Conv7x7 + Sigmoid
    breakdown["refine_spatial_attn"] = conv2d_flops(2, 1, 7, rh, rw, bias=False)
    # Element-wise ops: avgpool, maxpool, sigmoid, multiply, add ≈ rh*rw * 10
    breakdown["refine_elemwise"] = 10 * rh * rw

    # TaskClassifiers (×N)
    for t_idx, out_ch in enumerate(model.seg_head.out_channels_list):
        prefix = f"classifier_t{t_idx}"
        breakdown[f"{prefix}_dw"] = conv2d_flops(rch, rch, 3, rh, rw, groups=rch, bias=True)
        breakdown[f"{prefix}_pw"] = conv2d_flops(rch, out_ch, 1, rh, rw, bias=True)

    return dict(breakdown)


def format_flops(flops: int) -> str:
    """Format FLOPs thành string đẹp."""
    if flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f}K"
    return str(flops)


def main():
    parser = argparse.ArgumentParser(description="Manual FLOPs Calculator")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
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

    params_total = sum(p.numel() for p in model.parameters())
    d = model.encoder.d_model

    print("╔" + "═" * 78 + "╗")
    print("║" + "  U-MobileViT-Net: PHÂN TÍCH FLOPs CHI TIẾT (Manual)".center(78) + "║")
    print("║" + f"  alpha={alpha}, d_model={d}, input={3}×{H}×{W}".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    breakdown = compute_model_flops(model, H, W)
    total_flops = sum(breakdown.values())

    # Nhóm theo thành phần chính
    groups = {
        "Stem": [k for k in breakdown if k.startswith("stem_")],
        "Encoder Stage 1": [k for k in breakdown if k.startswith("enc_s1_")],
        "Encoder Stage 2": [k for k in breakdown if k.startswith("enc_s2_")],
        "Encoder Stage 3": [k for k in breakdown if k.startswith("enc_s3_")],
        "Decoder Stage 1": [k for k in breakdown if k.startswith("dec_s1_")],
        "Decoder Stage 2": [k for k in breakdown if k.startswith("dec_s2_")],
        "Decoder Stage 3": [k for k in breakdown if k.startswith("dec_s3_") or k.startswith("dec_concat")],
        "UpsampleHead": [k for k in breakdown if k.startswith("head_upsample")],
        "FeatureRefine": [k for k in breakdown if k.startswith("refine")],
        "Classifiers": [k for k in breakdown if k.startswith("classifier")],
    }

    # Sub-group: Attention vs Convolution
    attn_keys = [k for k in breakdown if "attn" in k and not "spatial_attn" in k]
    conv_keys = [k for k in breakdown if k not in attn_keys]
    attn_flops = sum(breakdown[k] for k in attn_keys)
    conv_flops = sum(breakdown[k] for k in conv_keys)

    print(f"\n{'═' * 80}")
    print(f"  TỔNG FLOPs: {total_flops:,.0f} ({format_flops(total_flops)})")
    print(f"  TỔNG PARAMS: {params_total:,} ({params_total/1e6:.2f}M)")
    print(f"  FLOPs/Param: {total_flops/params_total:.1f}")
    print(f"{'═' * 80}")

    print(f"\n  {'Thành phần':<25} {'FLOPs':>15} {'%':>8} {'Phân bổ':>30}")
    print(f"  {'─' * 78}")

    for group_name, keys in groups.items():
        g_flops = sum(breakdown[k] for k in keys)
        pct = g_flops / total_flops * 100
        bar = "█" * int(pct / 2)
        print(f"  {group_name:<25} {g_flops:>13,}  ({pct:>5.1f}%) {bar}")

    print(f"  {'─' * 78}")
    print(f"  {'TOTAL':<25} {total_flops:>13,}  (100.0%)")

    # Attention vs Conv
    print(f"\n  {'Phân loại theo operation':<30} {'FLOPs':>15} {'%':>8}")
    print(f"  {'─' * 55}")
    print(f"  {'Attention (Separable)':<30} {attn_flops:>13,}  ({attn_flops/total_flops*100:>5.1f}%)")
    print(f"  {'Convolution + Other':<30} {conv_flops:>13,}  ({conv_flops/total_flops*100:>5.1f}%)")

    # Chi tiết từng layer trong 1 encoder stage (đại diện)
    print(f"\n  {'─' * 80}")
    print(f"  CHI TIẾT ENCODER STAGE 1 (đại diện):")
    print(f"  {'─' * 80}")
    for k in groups["Encoder Stage 1"]:
        f = breakdown[k]
        print(f"    {k:<45} {f:>12,}  ({f/total_flops*100:>4.1f}%)")
    print(f"    {'TOTAL Stage 1':<45} {sum(breakdown[k] for k in groups['Encoder Stage 1']):>12,}")

    # Chi tiết Decoder Stage 1 (đại diện)
    print(f"\n  {'─' * 80}")
    print(f"  CHI TIẾT DECODER STAGE 1 (đại diện):")
    print(f"  {'─' * 80}")
    for k in groups["Decoder Stage 1"]:
        f = breakdown[k]
        print(f"    {k:<45} {f:>12,}  ({f/total_flops*100:>4.1f}%)")
    print(f"    {'TOTAL Stage 1':<45} {sum(breakdown[k] for k in groups['Decoder Stage 1']):>12,}")

    # Phân tích độ phân giải
    print(f"\n  {'─' * 80}")
    print(f"  FLOPs THEO ĐỘ PHÂN GIẢI:")
    print(f"  {'─' * 80}")
    res_groups = {
        f"H×W ({H}×{W})": [k for k in breakdown if "refine" in k or "classifier" in k or (k.startswith("head_upsample_x8"))],
        f"H/8 ({H//8}×{W//8})": [k for k in breakdown if (f"{H//8}" in k or f"{W//8}" in k) and "stem_conv3" not in k and "dec_concat" not in k],
        f"H/16 ({H//16}×{W//16})": [k for k in breakdown if f"{H//16}" in k or f"{W//16}" in k],
        f"H/32 ({H//32}×{W//32})": [k for k in breakdown if f"{H//32}" in k or f"{W//32}" in k],
        f"H/64 ({H//64}×{W//64})": [k for k in breakdown if f"{H//64}" in k or f"{W//64}" in k],
    }
    # Fix matching
    res_groups_clean = defaultdict(int)
    for k, v in breakdown.items():
        if any(x in k for x in ["refine", "classifier", "head_upsample_x8"]):
            res_groups_clean[f"Full ({H}×{W})"] += v
        elif "stem_conv3" in k or "dec_concat" in k or "head_upsample_x2" in k or "dec_s3" in k:
            res_groups_clean[f"H/8 → H/16"] += v
        elif "enc_s1" in k or "dec_s2" in k or "stem_conv2" in k or "head_upsample_x4" in k:
            res_groups_clean[f"H/16 → H/32"] += v
        elif "enc_s2" in k or "dec_s1" in k or "stem_conv1" in k:
            res_groups_clean[f"H/32 → H/64"] += v
        elif "enc_s3" in k:
            res_groups_clean[f"H/64 → H/128"] += v
        else:
            res_groups_clean["Other"] += v

    for res, f in sorted(res_groups_clean.items()):
        bar = "█" * int(f / total_flops * 50)
        print(f"  {res:<25} {f:>13,}  ({f/total_flops*100:>5.1f}%) {bar}")

    # Tổng kết
    print(f"\n{'═' * 80}")
    print(f"  KẾT LUẬN:")
    print(f"  ─────────")
    print(f"  • Mô hình light-weight: {params_total:,} params, {format_flops(total_flops)} FLOPs")
    print(f"  • Attention chiếm {attn_flops/total_flops*100:.1f}% FLOPs toàn model")
    print(f"  • UpsampleHead + Refinement chiếm {sum(breakdown[k] for k in groups['UpsampleHead'] + groups['FeatureRefine'])/total_flops*100:.1f}% FLOPs")
    print(f"  • Phần lớn FLOPs tập trung ở độ phân giải cao (H/8 → H)")
    print(f"{'═' * 80}")


if __name__ == "__main__":
    main()
