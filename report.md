# U‑MobileViT‑Net: Kiến trúc Chi tiết & Báo cáo Học thuật

> **Mổ xẻ kiến trúc 4 biến thể: Nano · Base · Pro · ProMax**
>
> Pipeline nghiên cứu: Kiến trúc → FLOPs/Params → Huấn luyện → Đánh giá đa miền

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Kiến trúc Tổng thể](#2-kiến-trúc-tổng-thể)
3. [Encoder — Trái tim Mã hóa](#3-encoder--trái-tim-mã-hóa)
4. [Decoder — Cầu nối Giải mã](#4-decoder--cầu-nối-giải-mã)
5. [Segmentation Head — Đầu ra Phân vùng](#5-segmentation-head--đầu-ra-phân-vùng)
6. [4 Biến thể — So sánh Chi tiết](#6-4-biến-thể--so-sánh-chi-tiết)
7. [Phân tích FLOPs & Tham số](#7-phân-tích-flops--tham-số)
8. [Những Cải tiến Kỹ thuật Quan trọng](#8-những-cải-tiến-kỹ-thuật-quan-trọng)
9. [Pipeline Huấn luyện](#9-pipeline-huấn-luyện)
10. [Kết quả Benchmark Đa miền](#10-kết-quả-benchmark-đa-miền)
11. [Kết luận & Hướng Phát triển](#11-kết-luận--hướng-phát-triển)

---

## 1. Tổng quan

### 1.1 Định danh

**U‑MobileViT‑Net** là kiến trúc segmentation lai (CNN + Vision Transformer) được thiết kế cho **thiết bị biên** (edge devices). Nó kết hợp:

- **U‑Net style encoder‑decoder** với skip connections → bảo toàn chi tiết không gian
- **MobileViTv2 separable self‑attention** → mô hình hóa ngữ cảnh toàn cục với chi phí tuyến tính O(N)
- **Inverted Bottleneck (MobileNetV2)** → mở rộng biểu diễn hiệu quả

### 1.2 Triết lý Thiết kế

| Nguyên tắc | Hiện thực |
|------------|-----------|
| **Nhẹ** | < 0.3M–7M tham số, phù hợp Jetson/RPi/Smartphone |
| **Nhanh** | Attention tuyến tính O(N), không O(N²) như ViT gốc |
| **Chính xác** | Cross‑attention decoder + multi‑scale skip connections |
| **Ổn định** | Float32‑protected attention, ReLU6, GroupNorm thay BatchNorm |
| **Linh hoạt** | 4 variants (nano→promax), single/dual‑task head |

### 1.3 Nguồn cảm hứng

| Công trình | Đóng góp vào U‑MobileViT‑Net |
|------------|------------------------------|
| **U‑Net** (Ronneberger, 2015) | Encoder‑decoder + skip connections |
| **MobileNetV2** (Sandler, 2018) | Inverted bottleneck với expansion factor |
| **MobileViT** (Mehta, 2022) | CNN + Transformer fusion cho mobile |
| **MobileViTv2** (Mehta, 2023) | Separable self‑attention O(N) thay O(N²) |
| **YOLOP** (Wu, 2022) | Multi‑task head với shared decoder |

---

## 2. Kiến trúc Tổng thể

### 2.1 Sơ đồ Khối



### 2.2 Luồng Dữ liệu (Forward Pass)



### 2.3 Kích thước Feature Map qua từng Tầng (Base, input 320×320)

| Tầng | Output Size | Channels |
|------|-------------|----------|
| Input | 320×320 | 3 |
| Stem 0 | 160×160 | 16 |
| Stem 1 | 80×80 | 32 |
| Stem 2 | 40×40 | 64 |
| Enc Stage 1 | 20×20 | 64 |
| Enc Stage 2 | 10×10 | 64 |
| Enc Stage 3 | 5×5 | 64 |
| Dec Stage 1 | 10×10 | 64 |
| Dec Stage 2 | 20×20 | 64 |
| Dec Out | 40×40 | 64 |
| SegHead ×2 | 80×80 | 32 |
| SegHead ×4 | 160×160 | 16 |
| SegHead ×8 | 320×320 | 8 |
| Output | 320×320 | num_classes |

---

## 3. Encoder — Trái tim Mã hóa

### 3.1 Stem Block

Stem gồm **3 tầng Conv2d + ReLU** liên tiếp, mỗi tầng giảm spatial resolution đi 2×:

| Tầng | Input Ch | Output Ch | Kernel | Stride | Output Size |
|------|----------|-----------|--------|--------|-------------|
| Stem 0 | 3 (RGB) | `d_model // 4` | 3×3 | 2 | H/2 × W/2 |
| Stem 1 | `d_model // 4` | `d_model // 2` | 3×3 | 2 | H/4 × W/4 |
| Stem 2 | `d_model // 2` | `d_model` | 3×3 | 2 | H/8 × W/8 |

**Skip connections**: Stem 0 và Stem 1 outputs được lưu để đưa vào Segmentation Head.

**Ví dụ Base (d=64)**: `[3→16→32→64]` — chia hết cho target_groups=4.

### 3.2 UMobileViTEncoderLayer

Mỗi Encoder Layer gồm 3 thành phần xử lý tuần tự:

```
Input (B, C, H, W)
    │
    ▼
┌──────────────────┐
│ ① LOCAL BLOCK     │  Depthwise 3×3 → PW 1×1 → ReLU → GroupNorm
│  (spatial detail) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ② GLOBAL BLOCK    │  Unfold → Transformer Encoder (self-attn) → Fold
│  (context)        │  Separable self-attention O(N) — từ MobileViTv2
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ③ EXPANSION BLOCK │  Inverted Bottleneck (MobileNetV2 style)
│  (representation) │  PW expand → DW 3×3 → PW project → GN → Dropout
└────────┬─────────┘
         │
         ▼
    + Input (residual) → GroupNorm → Output
```

#### 3.2.1 Local Block

```
Depthwise Conv2d (3×3, groups=C) → ReLU
  → Pointwise Conv2d (1×1) → ReLU → GroupNorm
```
- **Mục đích**: Trích xuất đặc trưng cục bộ (edges, textures)
- **Chi phí**: O(C × H × W) — rất nhẹ
- **Depthwise**: mỗi channel xử lý riêng biệt

#### 3.2.2 Global Block — Separable Self-Attention

Đây là **trái tim** của kiến trúc — nơi CNN gặp Transformer:

```
Input (B, C, H, W)
    │ Unfold → (B, C, P, S)  [P=patch pixels, S=num patches]
    ▼
Transformer Encoder Layer (×N blocks):
  ┌─────────────────────────────────┐
  │  SeparableAttention (Q=K=V)     │
  │    ├─ QKV projection (linear)   │
  │    ├─ Softmax attention         │
  │    └─ Output projection         │
  │         ↓                       │
  │  Dropout → GroupNorm (+res)     │
  └─────────────────────────────────┘
    │ Fold → (B, C, H, W)
    ▼
```

**Separable Self-Attention** (từ MobileViTv2):
- **Chi phí tuyến tính O(N)**, không O(N²) như ViT gốc
- Q, K, V tính qua linear projection trên channels
- Attention trong từng patch riêng biệt

**Unfold/Fold**:
- `unfold_custom`: (B,C,H,W) → (B,C,P,S) với P = patch_h×patch_w
- `fold_custom`: (B,C,P,S) → (B,C,H,W)
- Patch size: (2,2) cho stage 1,2; (1,1) cho stage 3

#### 3.2.3 Expansion Block — Inverted Bottleneck

```
Pointwise Conv2d (1×1, C → C×E) → GroupNorm → ReLU6
  → Depthwise Conv2d (3×3, groups=C×E) → GroupNorm → ReLU6
  → Pointwise Conv2d (1×1, C×E → C) → GroupNorm → Dropout
```
- **E = expansion_factor**: 2.0 (nano) → 4.0 (promax)
- **ReLU6**: giới hạn activation trong [0, 6] → chống NaN
- **GroupNorm thay BatchNorm**: ổn định với batch nhỏ
- **Linear bottleneck**: không activation ở cuối (giống MobileNetV2)

### 3.3 Encoder Stages

| Stage | Downsample | EncoderLayers | Patch Size | Blocks/Layer |
|-------|-----------|---------------|------------|-------------|
| 1 | H/8→H/16 | 1 | (2,2) | N (variant) |
| 2 | H/16→H/32 | 2 | (2,2) | N |
| 3 | H/32→H/64 | 2 | **(1,1)** | N |

**Tại sao Stage 3 dùng patch_size=(1,1)?** Ở độ phân giải thấp nhất, spatial context đủ nhỏ — attention hoạt động như **global self-attention** trên toàn bộ feature map.

### 3.4 Downsample Block

```
Depthwise Conv2d (3×3, stride=2, groups=C) → ReLU
```
- Giảm spatial 2×, giữ channels
- Depthwise → tiết kiệm tham số hơn Conv2d thường

---

## 4. Decoder — Cầu nối Giải mã

### 4.1 UMobileViTDecoderLayer

Decoder Layer có cấu trúc tương tự Encoder Layer nhưng **thay self-attention bằng cross-attention**:

```
Input (decoder features) + Memory (encoder skip)
    │
    ▼
┌──────────────────┐
│ ① LOCAL BLOCK     │  (giống Encoder)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ② GLOBAL BLOCK    │  Unfold → Transformer DECODER → Fold
│  (cross-attn)     │  Self-attn (input) + Cross-attn (input+memory)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ③ EXPANSION BLOCK │  (giống Encoder)
└────────┬─────────┘
         │
         ▼
    + Input (residual) → GroupNorm → Output
```

#### Transformer Decoder Layer:

```
Input (decoder features) + Memory (encoder features)
    │
    ├──→ Self-Attention (Q=K=V=input) → Dropout → GroupNorm (+res)
    │
    └──→ Cross-Attention (Q=input, K=V=memory) → Dropout → GroupNorm (+res)
```
- **Self‑attention**: mô hình quan hệ nội tại trong decoder
- **Cross‑attention**: query từ decoder, key/value từ encoder → khôi phục chi tiết không gian

### 4.2 UMobileViTDecoderConcatLayer

Layer đặc biệt ở tầng cuối — **không dùng transformer** (nhẹ hơn):

```
Input + Memory (skip từ stem)
    │
    ▼ Local Block
    │
    ▼ Memory Projection (1×1 Conv → ReLU)
    │
    ▼ Z = Z + mem_proj  (additive fusion — KHÔNG cross-attention)
    │
    ▼ Global Block nhẹ: DW 3×3 → ReLU → PW 1×1 → ReLU
    │
    ▼ Expansion Block → + Input (residual) → GroupNorm
```
- **Additive fusion** thay cross‑attention → giảm FLOPs ở tầng cuối
- Dùng khi feature map ở độ phân giải cao (H/8 × W/8)

### 4.3 Decoder Stages

| Stage | Upsample | DecoderLayers | Skip Connection từ |
|-------|----------|---------------|-------------------|
| 1 | ×2 (H/64→H/32) | 2 (cross-attn) | Encoder Stage 2 |
| 2 | ×2 (H/32→H/16) | 1 (cross-attn) | Encoder Stage 1 |
| Out | ×2 (H/16→H/8) | 1 (concat, additive) | Encoder Stem cuối |

**Upsample block**: `Nearest Upsample(×2) → Depthwise Conv2d 3×3 → ReLU`

---

## 5. Segmentation Head — Đầu ra Phân vùng

### 5.1 Single‑Task Head (`ContextAwareSegHead`)

```
Decoder Output (B, d, H/8, W/8)
    │
    ▼
┌──────────────────────────────────────┐
│  UpsampleHead (3 tầng upsample)      │
│  ×2: d→d/2  + skip từ stem[1]        │
│  ×4: d/2→d/4 + skip từ stem[0]       │
│  ×8: d/4→d/8                         │
│  → Output: (B, d/8, H, W)            │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  FeatureRefinementBlock               │
│  ├─ Dilated Conv 3×3 (dilation=2)    │
│  ├─ Spatial Attention (avg+max pool) │
│  └─ Residual connection              │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Classifier                           │
│  Depthwise 3×3 → ReLU → Pointwise 1×1│
│  → Output: (B, num_classes, H, W)    │
└──────────────────────────────────────┘
```

### 5.2 Multi‑Task Head (`MultiTaskDenseHead`)

```
Decoder Output
    │
    ▼
┌────────────────────────────┐
│  UpsampleHead (CHIA SẺ)    │  ← Chạy 1 lần duy nhất
├────────────────────────────┤
│  FeatureRefinement (CHIA SẺ)│
└────────────┬───────────────┘
             │
      ┌──────┴──────┐
      ▼              ▼
┌──────────┐  ┌──────────┐
│Classifier│  │Classifier│  ← Mỗi task 1 classifier riêng
│ Task 1   │  │ Task 2   │     (DW 3×3 + PW 1×1)
└──────────┘  └──────────┘
```

**Tối ưu Edge**: Phần đắt nhất (upsample + refinement) được **chia sẻ** cho mọi tác vụ → chi phí gần như không đổi khi tăng số task. Chuẩn thiết kế của YOLOP/HybridNets.

### 5.3 FeatureRefinementBlock

```
Input (B, C, H, W)
    │
    ├──→ Dilated Conv2d (3×3, dilation=2, groups=C)
    │         ↓
    │    Pointwise Conv2d (1×1)  ← mix channels
    │         ↓
    │    ┌─────────────────┐
    │    │ Spatial Attention │
    │    │ avg_pool + max    │
    │    │ → Conv 7×7 → Sigm│
    │    └────────┬────────┘
    │             ↓
    │    feat * attention_map
    │
    └──→ + Input (residual) → Output
```
- **Dilated Conv (dilation=2)**: mở rộng receptive field gấp đôi
- **Spatial Attention**: lọc nhiễu không gian, tập trung vùng quan trọng
- **Residual connection**: giữ thông tin gốc, chống vanishing gradient

---

## 6. 4 Biến thể — So sánh Chi tiết

### 6.1 Bảng Tham số Cấu hình

| Tham số | Nano | Base | Pro | ProMax |
|---------|------|------|-----|--------|
| **d_model** | 48 | 64 | 128 | 256 |
| **Width multiplier** | 0.75x | 1.0x | 2.0x | 4.0x |
| **Transformer blocks/layer** | 1 | 2 | 3 | 3 |
| **Expansion factor** | 2.0 | 3.0 | 3.0 | 4.0 |
| **GroupNorm target groups** | 4 | 4 | 4 | 8 |
| **Params (ước tính)** | ~0.3M | ~0.6M | ~2.0M | ~7.0M |
| **Mục tiêu** | Edge/Mobile | Cân bằng | Accuracy cao | Server-grade |

### 6.2 Stem Channels theo Biến thể

| Variant | d_model | Stem 0 | Stem 1 | Stem 2 |
|---------|---------|--------|--------|--------|
| Nano | 48 | **12** (3→12) | **24** (12→24) | **48** (24→48) |
| Base | 64 | **16** (3→16) | **32** (16→32) | **64** (32→64) |
| Pro | 128 | **32** (3→32) | **64** (32→64) | **128** (64→128) |
| ProMax | 256 | **64** (3→64) | **128** (64→128) | **256** (128→256) |

### 6.3 SegHead Channels theo Biến thể

| Variant | Upsample x2 | Upsample x4 | Upsample x8 (classifier input) |
|---------|-------------|-------------|-------------------------------|
| Nano | 48→24 | 24→12 | **12→6** |
| Base | 64→32 | 32→16 | **16→8** |
| Pro | 128→64 | 64→32 | **32→16** |
| ProMax | 256→128 | 128→64 | **64→32** |

### 6.4 Số Transformer Blocks

| Variant | Blocks/Layer | Enc S1 | Enc S2 | Enc S3 | Dec S1 (2 layers) | Dec S2 (1 layer) |
|---------|-------------|--------|--------|--------|-------------------|-------------------|
| Nano | **1** | 1 | 2 | 2 | 2/2/0 | 1/0/0 |
| Base | **2** | 2 | 4 | 4 | 4/4/0 | 2/0/0 |
| Pro | **3** | 3 | 6 | 6 | 6/6/0 | 3/0/0 |
| ProMax | **3** | 3 | 6 | 6 | 6/6/0 | 3/0/0 |

> Decoder S1 có 2 decoder layers, mỗi layer có N transformer decoder blocks. Decoder Out không dùng transformer (ConcatLayer additive).

### 6.5 Chiến lược Chọn Biến thể

```
         Độ chính xác →
         Nano     Base      Pro       ProMax
Edge  ───●─────────●─────────●──────────●─── Server
      0.3M      0.6M      2.0M       7.0M

Nano:   Raspberry Pi 5, Smartphone tầm trung
Base:   Jetson Nano, Smartphone cao cấp
Pro:    Jetson Xavier NX, Laptop GPU
ProMax: Jetson Orin, Workstation, Server
```

---

## 7. Phân tích FLOPs & Tham số

### 7.1 Phương pháp Tính FLOPs

FLOPs được tính **trực tiếp từ kiến trúc** (không dùng thop/profiler) để chính xác với custom `Conv2d`:

```python
# Conv2d FLOPs
mac = out_ch * (in_ch // groups) * kernel**2 * H_out * W_out
flops = 2 * mac + (out_ch * H_out * W_out if bias else 0)

# Separable Attention FLOPs (tuyến tính O(N)!)
qkv_proj = 2 * (1 + 2*C) * C * P * S    # QKV projection
softmax  = 5 * P * S                      # softmax
ctx      = 2 * C * P * S                  # context aggregation
out_proj = 2 * C * C * P * S              # output projection
# Tổng = O(C**2 * P * S), không O(P**2 * S**2) như ViT gốc
```

### 7.2 Phân bố FLOPs (Base, 320x320)

| Thành phần | FLOPs (ước tính) | Tỉ lệ |
|------------|-----------------|-------|
| **Stem** | ~15M | ~3% |
| **Encoder** | ~240M | ~45% |
| **Decoder** | ~160M | ~30% |
| **SegHead** | ~95M | ~18% |
| **Refinement + Classifier** | ~20M | ~4% |
| **Tổng** | **~530M** | **100%** |

### 7.3 FLOPs theo Biến thể (ước tính @ 320x320)

| Variant | d_model | FLOPs (ước tính) | vs Base |
|---------|---------|-----------------|---------|
| Nano | 48 | ~250M | 0.47x |
| Base | 64 | ~530M | 1.00x |
| Pro | 128 | ~2.1G | 4.00x |
| ProMax | 256 | ~8.5G | 16.0x |

> FLOPs tỉ lệ với d_model**2 (attention projection) + expansion_factor (inverted bottleneck).

### 7.4 Phân bố Tham số

| Thành phần | Tỉ lệ |
|------------|-------|
| Stem | ~5% |
| Encoder (local + attention + expansion) | ~35% |
| Decoder (local + attention + expansion) | ~30% |
| SegHead (upsample + refinement) | ~25% |
| Classifier(s) | ~5% |

### 7.5 Edge Device Latency Estimation

`estimate_edge_latency()` ước tính latency = max(compute_time, memory_time) + overhead.

| Thiết bị | RAM | Throughput (FP32) | Power |
|----------|-----|-------------------|-------|
| Jetson Nano (4GB) | 4 GB | 82 GFLOPs | 10W |
| Jetson Xavier NX | 8 GB | 338 GFLOPs | 15W |
| Jetson Orin Nano (8GB) | 8 GB | 461 GFLOPs | 7W |
| Raspberry Pi 5 (8GB) | 8 GB | 10 GFLOPs (CPU) | 8W |
| Smartphone Mid-range | 4 GB | 600 GOPS (INT8 NPU) | 3W |
| Flagship Smartphone | 8 GB | 2600 GOPS (INT8 NPU) | 4W |

---

## 8. Những Cải tiến Kỹ thuật Quan trọng

### 8.1 Float32‑Protected Attention (Chống NaN)

**Vấn đề**: AMP (Automatic Mixed Precision) + Gradient Checkpointing gây NaN trong backward.

**Giải pháp**: Luôn ép attention chạy float32 trên CUDA:

```python
# Trong SeparableAttention.forward():
if query.device.type == 'cuda':
    with torch.cuda.amp.autocast(enabled=False):
        result = separable_attention_forward(
            query=query.float(),   # EP VE FLOAT32
            key=key.float(), value=value.float(), ...
        )
        return result.to(query.dtype)  # Cast ve float16
```
- Dam bao ca forward (AMP) va backward recompute (checkpoint) duoc bao ve
- QKV projection de overflow > 65504 trong float16 → NaN gradient

### 8.2 ReLU6 trong Expansion Block

- **ReLU6** gioi han activation trong [0, 6] → ngan chan Inf → NaN
- **GroupNorm** sau moi conv (khong chi truoc/sau block) → on dinh hon

### 8.3 Gradient Checkpointing

```python
if self.training and self.use_checkpointing:
    Z = torch_checkpoint(self._global_forward, Z, None, ...)
    Z = torch_checkpoint(self._expansion_forward, Z, ...)
```
- **Global block** (transformer) va **expansion block** duoc checkpoint
- Giam **30–40% VRAM** (khong luu intermediate activations)
- Expansion block voi exp=4.0 tao tensor ~1.6GB cho ProMax

### 8.4 GroupNorm voi Auto‑Selection

```python
def get_groupnorm_groups(num_channels, target_groups=4):
    if num_channels % target_groups == 0:
        return target_groups
    divisors = sorted(divisors_of(num_channels))
    smaller = [d for d in divisors if d <= target_groups]
    return max(smaller) if smaller else min(divisors)
```
- **Van de**: GroupNorm yeu cau `num_channels % num_groups == 0`
- Khi scale model (nano: d=48), channel count khong chia het cho target_groups
- Ham tu dong chon uoc so phu hop → khong crash

### 8.5 Adaptive Padding cho Odd‑Size Inputs

```python
pad_h = (patch_h - H % patch_h) % patch_h
if pad_h > 0 or pad_w > 0:
    Z = F.pad(Z, (0, pad_w, 0, pad_h))
# ... unfold/fold ...
if pad_h > 0 or pad_w > 0:
    Z = Z[:, :, :H, :W]  # Crop back
```
- Input le (360x480) → tu dong pad → xu ly → crop
- **Khong can resize input** → giu ti le anh goc

### 8.6 Multi‑Task voi Shared Decoder

```
Truoc (ton O(N_task)):
  moi task: upsample + refinement + classifier

Sau (ton O(1)):
  upsample + refinement (chung) → N classifiers nhe
```
- **Tiet kiem**: phan decode dat nhat chay 1 lan
- Moi task chi them 1 classifier nhe (DW 3x3 + PW 1x1)

---

## 9. Pipeline Huấn luyện

### 9.1 Cấu hình Chung

| Hyperparameter | Giá trị |
|----------------|---------|
| Optimizer | **AdamW** |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Warmup Epochs | 5 |
| LR Scheduler | **Poly** (multi-class) / **Cosine** (binary) |
| Label Smoothing | 0.0–0.1 |
| Batch Size | 16 |
| Loss Function | **CE + Dice** (multi-class) / **BCE + Dice** (binary) |
| Initializer | **He Uniform** (kaiming_uniform_) |

### 9.2 Data Augmentation

| Intensity | Áp dụng cho | Augmentations |
|-----------|-------------|---------------|
| **Medium** | COCO Leaf, Cityscapes, Kvasir | Flip, Rotate(+-10), ColorJitter, Crop |
| **Strong** | CamVid, VOC, ISIC | + ElasticTransform, GridDistortion, CoarseDropout |

### 9.3 Class Imbalance Handling

```python
# Multi-class: class_weights tỉ lệ nghịch với tần suất pixel
class_weights = compute_class_weights(train_loader, num_classes, ignore_index)

# Binary: pos_weight cho BCE loss
pos_weight = compute_pos_weight(train_loader)
# = (#negative_pixels) / (#positive_pixels)
```

### 9.4 Early Stopping

- **Patience**: 15–20 epochs (tùy dataset)
- **Min epochs**: 30 (đảm bảo warmup + hội tụ ban đầu)
- **Monitor**: `val_metric` (mIoU hoặc Dice)

---

## 10. Kết quả Benchmark Đa miền

### 10.1 Datasets

| Dataset | Miền | Loại | Số lớp | Kích thước |
|---------|------|------|--------|------------|
| COCO Tea Leaf | Nông nghiệp | Multi-class | 8 | 320x320 |
| CamVid | Đường phố | Multi-class | 32 | 360x480 |
| Cityscapes | Đường phố | Multi-class | 19 | 512x1024 |
| PASCAL VOC | Đa dụng | Multi-class | 21 | 384x384 |
| Kvasir-SEG | Y tế (nội soi) | Binary | 1 | 256x256 |
| ISIC 2018 | Y tế (da liễu) | Binary | 1 | 256x256 |

### 10.2 Metrics

- **Multi-class**: mIoU (Mean Intersection over Union)
- **Binary**: Dice Coefficient (F1-score)

### 10.3 Cấu hình Huấn luyện theo Dataset

| Dataset | Epochs | Patience | Scheduler | Aug Intensity |
|---------|--------|----------|-----------|---------------|
| COCO Tea Leaf | 300 | 20 | Poly | Medium |
| CamVid | 500 | 20 | Poly | Strong |
| Cityscapes | 200 | 20 | Poly | Medium |
| PASCAL VOC | 200 | 20 | Poly | Strong |
| Kvasir-SEG | 150 | 15 | Cosine | Medium |
| ISIC 2018 | 150 | 15 | Cosine | Strong |

---

## 11. Kết luận & Hướng Phát triển

### 11.1 Tóm tắt Đóng góp

1. **Kiến trúc lai CNN-Transformer hiệu quả**: U-Net + MobileViTv2 separable attention → vừa nhẹ vừa chính xác
2. **4 biến thể linh hoạt**: Nano (0.3M) → ProMax (7M), phủ từ Raspberry Pi đến Server
3. **Ổn định huấn luyện**: Float32-protected attention, ReLU6, GroupNorm, gradient checkpointing → không NaN
4. **Thiết kế multi-task**: Shared decoder + lightweight task-specific classifiers
5. **Adaptive architecture**: Tự động pad odd-size inputs, auto-select GroupNorm groups

### 11.2 Hạn chế Hiện tại

- **Patch size cố định (2,2)** cho hầu hết các stage — có thể thử multi-scale patches
- **Chưa tích hợp Knowledge Distillation** từ ProMax → Nano
- **Chưa có quantization-aware training** (QAT) cho INT8 deployment

### 11.3 Hướng Phát triển

| Hướng | Mô tả | Ưu tiên |
|-------|------|---------|
| **QAT + INT8 Deployment** | Quantize model về INT8 cho NPU (Jetson/Smartphone) | Cao |
| **Neural Architecture Search** | Tự động tìm optimal d_model, expansion_factor, num_transformer_blocks | Trung bình |
| **Multi-scale Patches** | Patch size thích ứng theo spatial resolution | Trung bình |
| **Knowledge Distillation** | ProMax (teacher) → Nano (student) | Thấp |
| **Real-time Video Segmentation** | Temporal consistency + optical flow guidance | Thấp |
| **3D Extension** | U-MobileViT-Net 3D cho Medical Volume Segmentation | Thấp |

---

## Phụ lục A: Cấu trúc Thư mục

```
models/u_mobilevit_net/
├── __init__.py          # Public API exports
├── configs.py           # 4 variant definitions (nano/base/pro/promax)
├── u_models.py          # UMobileViT wrapper + factory functions
├── encoder_block.py     # UMobileViTEncoder + EncoderLayer
├── decoder_block.py     # UMobileViTDecoder + DecoderLayer + ConcatLayer
├── module.py            # _UMobileViTLayer base + local/expansion blocks
├── transfomer.py        # SeparableAttention + Transformer Enc/Dec layers
└── seg_head.py          # UpsampleHead + FeatureRefinement + SegHeads

tools/
├── evaluation.py        # FLOPs calculator, params counter, edge latency estimator
├── training.py          # SegmentationTrainer, TrainingConfig, class weight helpers
├── data.py              # create_dataloaders, DatasetInfo, augmentation pipeline
└── visualization.py     # Publication-quality plotting utilities
```

## Phụ lục B: Cách Sử dụng Nhanh

```python
from models.u_mobilevit_net import umobilevit_base, umobilevit_promax
from tools.evaluation import compute_flops, compute_parameters, format_flops

# Khởi tạo
model = umobilevit_base(out_channels=8, head="single")

# Phân tích
params = compute_parameters(model)
flops, breakdown = compute_flops(model, input_size=(320, 320))
print(f"Params: {params/1e6:.2f}M | FLOPs: {format_flops(flops)}")

# Forward
import torch
x = torch.randn(1, 3, 320, 320)
output = model(x)  # (1, 8, 320, 320)
```

---

> **Báo cáo được sinh từ codebase ngày 03/06/2026.**
> Mọi thông tin kiến trúc được trích xuất trực tiếp từ source code tại `models/u_mobilevit_net/`.
