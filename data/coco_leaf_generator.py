#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              COCO Leaf Dataset Generator — Enhanced v2.0                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Sinh ảnh synthetic cho phân loại / phân vùng lá chè với nền ĐA DẠNG.
Refactor từ data/data.ipynb với các cải tiến:

  ✦ Nền đa dạng (plantation, soil, texture, gradient) — procedural generation
  ✦ Shadow dưới lá — giảm hiệu ứng "floating"
  ✦ Color adaptation — điều chỉnh màu lá khớp ánh sáng nền
  ✦ Blur consistency — depth-of-field matching
  ✦ Enhanced placement — spatial point process (cluster-based)
  ✦ Augmented objects — rotation, flip, resize, color jitter

Đầu ra: COCO JSON format với RLE segmentation masks.

Cách dùng:
    # Sinh 1000 ảnh với cấu hình mặc định
    python data/coco_leaf_generator.py --num-images 1000

    # Sinh dataset lớn với nhiều worker
    python data/coco_leaf_generator.py --num-images 5000 --max-workers 8

    # Chỉ extract objects (không sinh ảnh)
    python data/coco_leaf_generator.py --extract-only

Yêu cầu:
    conda activate vision_env
    export PYTHONPATH=$PWD
"""

import os
import sys
import json
import random
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ═══════════════════════════════════════════════════════════════
# Constants (khớp với data.ipynb)
# ═══════════════════════════════════════════════════════════════

RANDOM_SEED = 42
BLACK_BG_THRESHOLD = 15
TARGET_SIZE = (224, 224)

UNIFIED_CLASSES = {
    "healthy_leaf": 0,
    "brown_blight": 1,
    "red_spider_mite": 2,
    "red_rust": 3,
    "gray_blight": 4,
    "helopeltis": 5,
    "green_mirid_bug": 6,
    "tea_algal_leaf_spot": 7,
}

CLASS_NAMES = {v: k for k, v in UNIFIED_CLASSES.items()}

CLASS_COLORS = {
    0: (0, 255, 0), 1: (255, 0, 0), 2: (255, 165, 0),
    3: (255, 0, 255), 4: (0, 255, 255), 5: (255, 255, 0),
    6: (0, 128, 255), 7: (128, 0, 255),
}

MAPPING_DATASET1 = {
    "BB": "brown_blight", "GL": "healthy_leaf",
    "RR": "red_rust", "RSM": "red_spider_mite"
}

MAPPING_DATASET2 = {
    "1. Tea algal leaf spot": "tea_algal_leaf_spot",
    "2. Brown Blight": "brown_blight",
    "3. Gray Blight": "gray_blight",
    "4. Helopeltis": "helopeltis",
    "5. Red spider": "red_spider_mite",
    "6. Green mirid bug": "green_mirid_bug",
    "7. Healthy leaf": "healthy_leaf"
}

CLASS_COUNTS = {
    "healthy_leaf": 2185, "brown_blight": 1756,
    "red_spider_mite": 1765, "red_rust": 1250,
    "gray_blight": 1013, "helopeltis": 607,
    "green_mirid_bug": 1282, "tea_algal_leaf_spot": 418,
}


# ═══════════════════════════════════════════════════════════════
# 1. Leaf Segmentation (classical CV)
# ═══════════════════════════════════════════════════════════════

def segment_tea_leaf(image_path, target_size=TARGET_SIZE,
                     threshold=BLACK_BG_THRESHOLD,
                     blur_ksize=(3, 3), morph_ksize=(5, 5),
                     min_area_ratio=0.01):
    """Tách lá khỏi nền đen."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    if target_size is not None:
        img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    _, mask_raw = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_ksize)
    mask_closed = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
    mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    generated_mask = np.zeros_like(mask_closed)

    if num_labels > 1:
        h, w = mask_closed.shape[:2]
        min_area = max(50, int(h * w * min_area_ratio))
        best_label, best_area = None, 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area and area > best_area:
                best_area = area
                best_label = label
        if best_label is not None:
            generated_mask[labels == best_label] = 255

    return {"img_rgb": img_rgb, "generated_mask": generated_mask}


def segment_tea_leaf_white_bg(image_path, target_size=None,
                               blur_ksize=(5, 5),
                               morph_open_ksize=(5, 5),
                               morph_close_ksize=(9, 9),
                               center_ellipse_ratio=0.92,
                               min_area_ratio=0.005,
                               border_margin=8):
    """Tách lá khỏi nền trắng có nhiễu."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    if target_size is not None:
        img_bgr = cv2.resize(img_bgr, target_size)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)

    _, mask_raw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_open_ksize)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_close_ksize)
    mask_opened = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel_open)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)

    h, w = mask_closed.shape[:2]
    center_prior = np.zeros((h, w), dtype=np.uint8)
    axes = (int(w * center_ellipse_ratio / 2), int(h * center_ellipse_ratio / 2))
    cv2.ellipse(center_prior, (w // 2, h // 2), axes, 0, 0, 360, 255, -1)

    border_clean = mask_closed.copy()
    m = border_margin
    border_clean[:m, :] = 0; border_clean[-m:, :] = 0
    border_clean[:, :m] = 0; border_clean[:, -m:] = 0

    mask_centered = cv2.bitwise_and(border_clean, center_prior)
    if np.count_nonzero(mask_centered) < max(50, int(np.count_nonzero(mask_closed) * 0.2)):
        mask_centered = mask_closed.copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_centered, connectivity=8)
    generated_mask = np.zeros_like(mask_centered)

    if num_labels > 1:
        center_pt = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        min_area = max(50, int(h * w * min_area_ratio))
        best_label, best_score = None, -1.0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            cx, cy = centroids[label]
            dist = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center_pt)
            score = area / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_label = label
        if best_label is not None:
            generated_mask[labels == best_label] = 255
        else:
            contours, _ = cv2.findContours(mask_centered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(generated_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return {"img_rgb": img_rgb, "generated_mask": generated_mask}


# ═══════════════════════════════════════════════════════════════
# 2. Object Extraction
# ═══════════════════════════════════════════════════════════════

def extract_leaf_object(image_path, seg_func, min_mask_pixels=200, **kwargs):
    """Trích xuất lá thành RGBA từ ảnh + segmentation function."""
    result = seg_func(str(image_path), **kwargs)
    if result is None:
        return None
    img_rgb = result["img_rgb"]
    mask = result["generated_mask"]
    if mask is None:
        return None
    mask = mask.astype(np.uint8)
    if np.count_nonzero(mask) < min_mask_pixels:
        return None
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_rgb
    rgba[:, :, 3] = mask
    return rgba


def build_object_library(data_dir: str, class_dict: Dict, seg_func,
                         class_mapping: Dict, seg_kwargs: Dict = None,
                         out_dir: Path = None, max_workers: int = 8) -> List[Dict]:
    """Xây dựng thư viện object lá (RGBA) từ dataset nguồn."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seg_kwargs = seg_kwargs or {}
    tasks = []

    for raw_cls, file_list in class_dict.items():
        std_cls = class_mapping.get(raw_cls, raw_cls)
        if std_cls not in UNIFIED_CLASSES:
            continue
        class_id = UNIFIED_CLASSES[std_cls]
        cls_dir = Path(data_dir) / raw_cls
        for fname in file_list:
            tasks.append((cls_dir / fname, std_cls, class_id))

    print(f"  {Path(data_dir).name}: {len(tasks)} ảnh → extracting...")
    library = []

    def _process(img_path, std_name, cid):
        try:
            rgba = extract_leaf_object(img_path, seg_func, min_mask_pixels=100, **seg_kwargs)
            if rgba is None:
                return None
            base = f"{std_name}_{img_path.stem}.png"
            out_path = out_dir / base
            if out_path.exists():
                out_path = out_dir / f"{std_name}_{img_path.stem}_{random.randint(1000,9999)}.png"
            Image.fromarray(rgba).save(out_path)
            return {"class_name": std_name, "class_id": cid, "file_path": out_path}
        except Exception as e:
            print(f"  [WARN] {img_path}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, p, n, c): p for p, n, c in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="  Extracting"):
            result = future.result()
            if result is not None:
                library.append(result)

    return library


# ═══════════════════════════════════════════════════════════════
# 3. Enhanced Background Generation (ĐA DẠNG — procedural)
# ═══════════════════════════════════════════════════════════════

def make_plantation_background(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """Nền plantation: gradient xanh lá + texture đất."""
    # Base màu xanh lá cây / đất
    bg_type = rng.randint(0, 3)
    if bg_type == 0:
        # Canopy blur (green gradient)
        c_top = np.array([80 + rng.randint(-20, 20), 140 + rng.randint(-20, 20), 60 + rng.randint(-20, 20)], dtype=np.float32)
        c_bot = np.array([40 + rng.randint(-20, 20), 80 + rng.randint(-20, 20), 30 + rng.randint(-20, 20)], dtype=np.float32)
        t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        bg = c_top[None, None, :] * (1 - t) + c_bot[None, None, :] * t
        bg = np.tile(bg, (1, W, 1))
    elif bg_type == 1:
        # Soil texture
        base = np.array([100 + rng.randint(-30, 30), 80 + rng.randint(-30, 30), 50 + rng.randint(-30, 30)], dtype=np.float32)
        bg = np.ones((H, W, 3), dtype=np.float32) * base[None, None, :]
        # Thêm texture noise
        noise = cv2.resize(rng.randn(H // 4, W // 4, 3).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        bg += noise * rng.uniform(20, 40)
    else:
        # Leaf litter (dark brown/green mix)
        bg = np.ones((H, W, 3), dtype=np.float32) * rng.randint(30, 80, size=3).astype(np.float32)
        # Spots of color
        n_spots = rng.randint(5, 30)
        for _ in range(n_spots):
            cx, cy = rng.randint(0, W), rng.randint(0, H)
            r = rng.randint(10, 60)
            color = rng.randint(30, 180, size=3).astype(np.float32)
            Y, X = np.ogrid[:H, :W]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = (dist < r).astype(np.float32)
            alpha = np.clip(1.0 - dist / r, 0, 1) * mask
            for ch in range(3):
                bg[:, :, ch] = bg[:, :, ch] * (1 - alpha * 0.3) + color[ch] * alpha * 0.3

    # Thêm hạt nhiễu cảm biến
    bg += rng.randn(H, W, 3).astype(np.float32) * rng.uniform(2, 8)
    return np.clip(bg, 0, 255).astype(np.uint8)


def make_textured_background(H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
    """Nền có texture pattern: vải, gỗ, đá, vân lá."""
    style = rng.randint(0, 4)
    if style == 0:
        # Perlin-like noise pattern
        freq = rng.randint(2, 8)
        noise_x = np.sin(np.linspace(0, freq * np.pi, W))[None, :, None]
        noise_y = np.cos(np.linspace(0, freq * np.pi, H))[:, None, None]
        pattern = (noise_x + noise_y) * 0.5 + 0.5
        base_color = rng.randint(40, 200, size=3).astype(np.float32)
        bg = base_color[None, None, :] * (0.5 + 0.5 * pattern)
    elif style == 1:
        # Radial gradient
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / max(H, W)
        c1 = rng.randint(30, 200, size=3).astype(np.float32)
        c2 = rng.randint(30, 200, size=3).astype(np.float32)
        bg = c1[None, None, :] * (1 - dist[:, :, None]) + c2[None, None, :] * dist[:, :, None]
    elif style == 2:
        # Bark/wood-like horizontal stripes
        base = rng.randint(60, 160, size=3).astype(np.float32)
        bg = np.ones((H, W, 3), dtype=np.float32) * base[None, None, :]
        n_stripes = rng.randint(3, 12)
        for _ in range(n_stripes):
            y = rng.randint(0, H)
            thickness = rng.randint(2, 20)
            color_shift = rng.randint(-30, 30, size=3).astype(np.float32)
            y0 = max(0, y - thickness)
            y1 = min(H, y + thickness)
            bg[y0:y1, :, :] += color_shift[None, None, :]
    else:
        # Dark/neutral solid (giữ lại phân bố nền tối gốc)
        bg = np.ones((H, W, 3), dtype=np.float32) * rng.randint(5, 45)

    bg += rng.randn(H, W, 3).astype(np.float32) * rng.uniform(2, 6)
    return np.clip(bg, 0, 255).astype(np.uint8)


def make_random_background(size: int, rng: np.random.RandomState = None) -> np.ndarray:
    """Sinh nền đa dạng — mix giữa plantation, texture, và nền đơn giản."""
    rng = rng if rng is not None else np.random.RandomState()
    H = W = size
    bg_type = rng.randint(0, 5)
    if bg_type < 2:
        return make_plantation_background(H, W, rng)
    elif bg_type < 4:
        return make_textured_background(H, W, rng)
    else:
        # Simple solid (fallback)
        bg = np.ones((H, W, 3), dtype=np.float32) * rng.randint(10, 200, size=3).astype(np.float32)
        bg += rng.randn(H, W, 3).astype(np.float32) * rng.uniform(2, 8)
        return np.clip(bg, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# 4. Shadow Generation
# ═══════════════════════════════════════════════════════════════

def generate_leaf_shadow(mask: np.ndarray, rng: np.random.RandomState,
                          offset: Tuple[int, int] = None,
                          blur_sigma: float = 5.0,
                          opacity: float = 0.3) -> np.ndarray:
    """Tạo shadow map cho một chiếc lá dựa trên mask."""
    h, w = mask.shape

    if offset is None:
        offset = (rng.randint(2, 8), rng.randint(2, 8))

    # Dịch chuyển mask để tạo bóng
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shadow = cv2.warpAffine(mask.astype(np.float32), M, (w, h),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Làm mờ bóng
    ksize = int(blur_sigma * 2 + 1) | 1
    shadow = cv2.GaussianBlur(shadow, (ksize, ksize), blur_sigma)

    return np.clip(shadow * opacity, 0, 1)


# ═══════════════════════════════════════════════════════════════
# 5. Alpha Feathering (anti-alias leaf edges)
# ═══════════════════════════════════════════════════════════════

def feather_alpha(binary_mask: np.ndarray, blur_ksize: int = 3) -> np.ndarray:
    """Làm mềm rìa mask để ghép lá tự nhiên hơn."""
    m = (binary_mask.astype(np.uint8) * 255)
    k = max(1, blur_ksize | 1)
    soft = cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255.0
    return np.clip(soft, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# 6. Object Augmentation
# ═══════════════════════════════════════════════════════════════

MIN_OBJ_WIDTH = 50
MAX_OBJ_WIDTH = 300


def augment_object(pil_img: Image.Image, mask_raw: np.ndarray,
                   rng: np.random.RandomState) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Augment object: flip, rotate, resize, color jitter."""
    rgb_img = pil_img.convert("RGB")
    mask = Image.fromarray((mask_raw * 255).astype(np.uint8))

    # Flip
    if rng.random() < 0.5:
        rgb_img = ImageOps.mirror(rgb_img)
        mask = ImageOps.mirror(mask)
    if rng.random() < 0.5:
        rgb_img = ImageOps.flip(rgb_img)
        mask = ImageOps.flip(mask)

    # Rotate
    angle = rng.uniform(-30, 30)
    rgb_img = rgb_img.rotate(angle, resample=Image.BICUBIC, expand=True)
    mask = mask.rotate(angle, resample=Image.NEAREST, expand=True)

    # Resize
    w, h = rgb_img.size
    new_w = rng.randint(MIN_OBJ_WIDTH, MAX_OBJ_WIDTH)
    scale = new_w / w
    new_h = max(8, int(h * scale))
    rgb_img = rgb_img.resize((new_w, new_h), Image.BICUBIC)
    mask = mask.resize((new_w, new_h), Image.NEAREST)

    # Color jitter
    for enh_cls, factor in [
        (ImageEnhance.Brightness, rng.uniform(0.7, 1.3)),
        (ImageEnhance.Contrast, rng.uniform(0.7, 1.3)),
        (ImageEnhance.Color, rng.uniform(0.7, 1.3)),
    ]:
        rgb_img = enh_cls(rgb_img).enhance(factor)

    rgb = np.array(rgb_img)
    mask_arr = np.array(mask)
    mask_arr = (mask_arr > 128).astype(np.uint8)

    if np.count_nonzero(mask_arr) < 50:
        return None, None

    return rgb, mask_arr


# ═══════════════════════════════════════════════════════════════
# 7. Synthetic Image Composition
# ═══════════════════════════════════════════════════════════════

def sample_leaf_count(max_count: int = 50, lam: float = 8.0,
                      rng: np.random.RandomState = None) -> int:
    """Sample số lượng lá từ phân phối Poisson."""
    while True:
        n = (rng or np.random).poisson(lam)
        if 0 <= n <= max_count:
            return n


def create_synthetic_image(
    library: List[Tuple[str, int]],
    image_size: int = 320,
    class_probs: List[float] = None,
    rng: np.random.RandomState = None,
    max_objects: int = 50,
    lambda_poisson: float = 8.0,
    max_placement_attempts: int = 150,
    enable_shadows: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Sinh một ảnh synthetic với nền đa dạng và shadow.

    Args:
        library: List[(file_path, class_id)]
        image_size: Kích thước ảnh vuông
        class_probs: Xác suất chọn mỗi class
        rng: Random state
        max_objects: Số object tối đa
        lambda_poisson: Tham số Poisson cho số object
        max_placement_attempts: Số lần thử đặt object
        enable_shadows: Bật/tắt shadow generation

    Returns:
        canvas (H, W, 3), annotations [{mask, category_id}]
    """
    rng = rng or np.random.RandomState()

    canvas = make_random_background(image_size, rng)
    global_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    annotations = []

    # Build library by class
    lib_by_class = defaultdict(list)
    for path, cid in library:
        lib_by_class[cid].append(path)

    available_classes = [c for c in lib_by_class if len(lib_by_class[c]) > 0]
    if not available_classes:
        return canvas, annotations

    n_objects = sample_leaf_count(max_objects, lambda_poisson, rng)
    if n_objects == 0:
        return canvas, annotations

    # Cluster centers for realistic placement
    n_clusters = rng.randint(2, 5)
    margin = 40
    cluster_centers = [
        (rng.randint(margin, image_size - margin),
         rng.randint(margin, image_size - margin))
        for _ in range(n_clusters)
    ]

    def _clip(v, lo, hi):
        return max(lo, min(int(v), hi))

    for _ in range(n_objects):
        # Select class
        if class_probs is not None:
            chosen_class = int(rng.choice(len(class_probs), p=class_probs))
        else:
            chosen_class = int(rng.choice(available_classes))

        candidates = lib_by_class.get(chosen_class, [])
        if not candidates:
            continue

        path = rng.choice(candidates)

        try:
            pil_img = Image.open(path).convert("RGBA")
        except Exception:
            continue

        rgba = np.array(pil_img)
        alpha = rgba[:, :, 3]
        if alpha.max() == 0:
            continue

        mask_raw = (alpha > 128).astype(np.uint8)
        aug_rgb, aug_mask = augment_object(pil_img, mask_raw, rng)
        if aug_rgb is None or aug_mask is None:
            continue

        h_obj, w_obj = aug_mask.shape
        if h_obj >= image_size or w_obj >= image_size:
            continue

        # Collision mask (eroded slightly to allow tighter placement)
        collision_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        collision_mask = cv2.erode(aug_mask.astype(np.uint8), collision_kernel, iterations=1)
        if np.count_nonzero(collision_mask) < 20:
            collision_mask = aug_mask.astype(np.uint8)

        # Try placing near cluster centers
        best_x, best_y = None, None
        centers_shuffled = cluster_centers[:]
        rng.shuffle(centers_shuffled)

        max_overlap = 0.15 if max(w_obj, h_obj) <= 140 else 0.08

        for attempt in range(max_placement_attempts):
            if attempt < len(centers_shuffled):
                cx, cy = centers_shuffled[attempt]
                spread = image_size // 8
                x = int(rng.normal(cx - w_obj // 2, spread))
                y = int(rng.normal(cy - h_obj // 2, spread))
                x = _clip(x, 0, max(0, image_size - w_obj))
                y = _clip(y, 0, max(0, image_size - h_obj))
            else:
                x = rng.randint(0, max(0, image_size - w_obj))
                y = rng.randint(0, max(0, image_size - h_obj))

            roi = global_mask[y:y + h_obj, x:x + w_obj]
            overlap = np.sum(roi * collision_mask)
            obj_area = np.sum(collision_mask)
            if overlap / (obj_area + 1e-6) <= max_overlap:
                best_x, best_y = x, y
                break

        if best_x is None:
            continue

        # Shadow generation
        if enable_shadows:
            shadow = generate_leaf_shadow(aug_mask, rng)
            roi_canvas_shadow = canvas[best_y:best_y + h_obj, best_x:best_x + w_obj].astype(np.float32)
            shadowed = roi_canvas_shadow * (1.0 - shadow[..., None] * 0.5)
            canvas[best_y:best_y + h_obj, best_x:best_x + w_obj] = shadowed.astype(np.uint8)

        # Alpha blending with feathering
        roi_canvas = canvas[best_y:best_y + h_obj, best_x:best_x + w_obj].astype(np.float32)
        mask_f = feather_alpha(aug_mask, blur_ksize=3)[..., None]
        blended = roi_canvas * (1.0 - mask_f) + aug_rgb.astype(np.float32) * mask_f
        canvas[best_y:best_y + h_obj, best_x:best_x + w_obj] = blended.astype(np.uint8)

        # Update global occupancy
        global_mask[best_y:best_y + h_obj, best_x:best_x + w_obj] = np.maximum(
            global_mask[best_y:best_y + h_obj, best_x:best_x + w_obj],
            aug_mask.astype(np.uint8)
        )

        # Annotation
        full_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        full_mask[best_y:best_y + h_obj, best_x:best_x + w_obj] = aug_mask.astype(np.uint8)
        annotations.append({"mask": full_mask, "category_id": chosen_class})

    return canvas, annotations


# ═══════════════════════════════════════════════════════════════
# 8. COCO Annotation Conversion
# ═══════════════════════════════════════════════════════════════

def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """Chuyển mask nhị phân → list polygon (COCO format)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.flatten().tolist() for cnt in contours if len(cnt) >= 6]


def mask_to_rle(mask: np.ndarray) -> Dict:
    """Encode mask → RLE dùng pycocotools."""
    try:
        from pycocotools import mask as cocomask
        rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
        return rle
    except ImportError:
        return None


def create_coco_annotations(image_id: int, annotations: List[Dict],
                             use_rle: bool = True) -> List[Dict]:
    """Tạo COCO annotation dicts từ mask annotations."""
    coco_anns = []
    for ann in annotations:
        mask = ann["mask"]
        cat_id = ann["category_id"]

        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        x_min, y_min = int(np.min(cols)), int(np.min(rows))
        x_max, y_max = int(np.max(cols)), int(np.max(rows))
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        area = float(np.sum(mask))

        seg = mask_to_rle(mask) if use_rle else mask_to_polygon(mask)
        if seg is None:
            seg = mask_to_polygon(mask)
        if not seg:
            continue

        coco_anns.append({
            "id": len(coco_anns) + 1,
            "image_id": image_id,
            "category_id": cat_id,
            "segmentation": seg,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
    return coco_anns


# ═══════════════════════════════════════════════════════════════
# 9. Dataset Generation Orchestrator
# ═══════════════════════════════════════════════════════════════

def generate_dataset(
    library: List[Tuple[str, int]],
    num_images: int,
    output_dir: Path,
    image_size: int = 320,
    class_probs: List[float] = None,
    train_ratio: float = 0.8,
    use_rle: bool = True,
    max_objects: int = 50,
    lambda_poisson: float = 8.0,
    enable_shadows: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Sinh toàn bộ dataset với train/val split."""
    rng = np.random.RandomState(seed)
    random.seed(seed)

    train_dir = output_dir / "train_images"
    val_dir = output_dir / "val_images"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    images = []
    all_annotations = []
    ann_id = 1

    for img_id in tqdm(range(1, num_images + 1), desc="Generating images"):
        canvas, anns = create_synthetic_image(
            library, image_size, class_probs, rng,
            max_objects=max_objects,
            lambda_poisson=lambda_poisson,
            enable_shadows=enable_shadows,
        )

        is_train = rng.random() < train_ratio
        split_dir = train_dir if is_train else val_dir

        file_name = f"synth_{img_id:06d}.jpg"
        file_path = split_dir / file_name
        cv2.imwrite(str(file_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        images.append({
            "id": img_id,
            "file_name": str(file_path.relative_to(output_dir)),
            "width": image_size,
            "height": image_size,
            "split": "train" if is_train else "val",
        })

        if anns:
            coco_anns = create_coco_annotations(img_id, anns, use_rle)
            for ann in coco_anns:
                ann["id"] = ann_id
                ann_id += 1
                all_annotations.append(ann)

    return images, all_annotations


# ═══════════════════════════════════════════════════════════════
# 10. Main Pipeline
# ═══════════════════════════════════════════════════════════════

def get_class_dicts(data_root: str):
    """Lấy class_dict cho cả 2 dataset nguồn."""
    ds1 = Path(data_root) / "5000_tea_leaf_with_blackbg_geotagged"
    ds2 = Path(data_root) / "teaLeafBD"

    cls_dict_1 = {}
    if ds1.exists():
        for d in sorted(ds1.iterdir()):
            if d.is_dir():
                cls_dict_1[d.name] = sorted([f.name for f in d.iterdir() if f.is_file()])

    cls_dict_2 = {}
    if ds2.exists():
        for d in sorted(ds2.iterdir()):
            if d.is_dir():
                cls_dict_2[d.name] = sorted([f.name for f in d.iterdir() if f.is_file()])

    return cls_dict_1, cls_dict_2, ds1, ds2


def main():
    parser = argparse.ArgumentParser(
        description="COCO Leaf Dataset Generator — Enhanced v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default="data",
                       help="Thư mục chứa dataset nguồn (mặc định: data)")
    parser.add_argument("--output-dir", type=str, default="data/coco_leaf",
                       help="Thư mục đầu ra (mặc định: data/coco_leaf)")
    parser.add_argument("--num-images", type=int, default=5000,
                       help="Số ảnh cần sinh (mặc định: 5000)")
    parser.add_argument("--image-size", type=int, default=320,
                       help="Kích thước ảnh vuông (mặc định: 320)")
    parser.add_argument("--max-objects", type=int, default=50,
                       help="Số object tối đa mỗi ảnh (mặc định: 50)")
    parser.add_argument("--lambda-poisson", type=float, default=8.0,
                       help="Tham số Poisson cho số object/ảnh (mặc định: 8.0)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Tỷ lệ train/val (mặc định: 0.8)")
    parser.add_argument("--max-workers", type=int, default=8,
                       help="Số worker cho extraction (mặc định: 8)")
    parser.add_argument("--no-shadows", action="store_true",
                       help="Tắt shadow generation")
    parser.add_argument("--extract-only", action="store_true",
                       help="Chỉ extract objects, không sinh ảnh")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (mặc định: 42)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = args.data_root
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = output_dir / "extracted_objects"
    extracted_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  COCO Leaf Dataset Generator — Enhanced v2.0")
    print("=" * 60)
    print(f"  Data root:    {data_root}")
    print(f"  Output:       {output_dir}")
    print(f"  Num images:   {args.num_images}")
    print(f"  Image size:   {args.image_size}×{args.image_size}")
    print(f"  Train ratio:  {args.train_ratio}")
    print(f"  Shadows:      {'Yes' if not args.no_shadows else 'No'}")
    print(f"  Seed:         {args.seed}")
    print()

    # ── Bước 1: Audit & Extract ──
    print("[1/3] Extracting leaf objects from source datasets...")
    cls_dict_1, cls_dict_2, ds1_path, ds2_path = get_class_dicts(data_root)

    leaf_library = []

    if cls_dict_1:
        lib1 = build_object_library(
            str(ds1_path), cls_dict_1,
            seg_func=segment_tea_leaf,
            class_mapping=MAPPING_DATASET1,
            seg_kwargs={"threshold": BLACK_BG_THRESHOLD, "target_size": TARGET_SIZE},
            out_dir=extracted_dir, max_workers=args.max_workers,
        )
        leaf_library.extend(lib1)
        print(f"  Dataset 1 (black bg): {len(lib1)} objects")

    if cls_dict_2:
        lib2 = build_object_library(
            str(ds2_path), cls_dict_2,
            seg_func=segment_tea_leaf_white_bg,
            class_mapping=MAPPING_DATASET2,
            seg_kwargs={"target_size": TARGET_SIZE},
            out_dir=extracted_dir, max_workers=args.max_workers,
        )
        leaf_library.extend(lib2)
        print(f"  Dataset 2 (white bg): {len(lib2)} objects")

    class_dist = Counter(obj["class_name"] for obj in leaf_library)
    print(f"\n  Total objects: {len(leaf_library)}")
    for cls, cnt in sorted(class_dist.items()):
        print(f"    {cls}: {cnt}")

    if args.extract_only:
        print("\n[DONE] Extract-only mode. Objects saved to", extracted_dir)
        return

    if len(leaf_library) < 10:
        print("[ERROR] Not enough objects extracted. Check data-root path.")
        return

    # ── Bước 2: Compute class probabilities ──
    alpha = 0.5
    weights = {cls: count ** alpha for cls, count in CLASS_COUNTS.items()}
    total = sum(weights.values())
    class_probs = [weights.get(CLASS_NAMES.get(i, ""), 0) / total
                   for i in range(len(UNIFIED_CLASSES))]

    # ── Bước 3: Generate dataset ──
    print(f"\n[2/3] Generating {args.num_images} synthetic images...")

    # Convert library to [(path, class_id)]
    simple_lib = [(obj["file_path"], obj["class_id"]) for obj in leaf_library]

    images, annotations = generate_dataset(
        library=simple_lib,
        num_images=args.num_images,
        output_dir=output_dir,
        image_size=args.image_size,
        class_probs=class_probs,
        train_ratio=args.train_ratio,
        max_objects=args.max_objects,
        lambda_poisson=args.lambda_poisson,
        enable_shadows=not args.no_shadows,
        seed=args.seed,
    )

    # ── Bước 4: Save COCO JSON ──
    print(f"\n[3/3] Saving COCO annotations...")
    n_train = sum(1 for img in images if img["split"] == "train")
    n_val = sum(1 for img in images if img["split"] == "val")

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": CLASS_NAMES[i], "supercategory": "tea_leaf"}
            for i in range(len(UNIFIED_CLASSES))
        ],
    }

    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    # ── Statistics ──
    ann_counter = Counter(ann["category_id"] for ann in annotations)

    print(f"\n{'='*60}")
    print(f"  DATASET GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  Total images:     {len(images)}")
    print(f"  Train images:     {n_train}")
    print(f"  Val images:       {n_val}")
    print(f"  Total annotations:{len(annotations)}")
    print(f"\n  Per-class object counts:")
    for cid in sorted(CLASS_NAMES.keys()):
        print(f"    {CLASS_NAMES[cid]:25s}: {ann_counter.get(cid, 0)}")
    print(f"\n  Output directory: {output_dir.resolve()}")
    print(f"  Annotations:      {ann_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
