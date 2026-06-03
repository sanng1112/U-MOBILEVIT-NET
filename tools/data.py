"""
Unified dataset loading pipeline for U-MobileViT-Net benchmark experiments.

Provides synchronized image-mask augmentation, per-dataset classes for six
standard semantic segmentation benchmarks, and a single factory function
that returns train/validation DataLoaders along with structured metadata.

Supported datasets:
    - coco_leaf   (8 classes, agricultural — multi-class, mIoU)
    - camvid      (11 classes, urban driving — multi-class, mIoU)
    - cityscapes  (19 classes, urban driving — multi-class, mIoU)
    - pascal_voc  (21 classes, general objects — multi-class, mIoU)
    - kvasir_seg  (1 class,  medical polyp — binary, Dice)
    - isic2018    (1 class,  medical skin lesion — binary, Dice)

Usage:
    from tools.data import create_dataloaders, DatasetInfo

    train_loader, val_loader, info = create_dataloaders("camvid")
    print(f"{info.name}: {info.num_classes} classes, {info.train_size} samples")
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """Structured metadata for a segmentation dataset."""
    name: str
    num_classes: int
    type: str                         # "binary" | "multi-class"
    ignore_index: int = 255
    palette: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    root: str = ""
    train_size: int = 0
    val_size: int = 0
    image_size: Tuple[int, int] = (320, 320)


# ---------------------------------------------------------------------------
# Synchronized image-mask augmentation pipeline
# ---------------------------------------------------------------------------

class ComposeAugmentation:
    """Synchronized image-mask augmentation.

    Geometric transforms are applied identically to both image and mask.
    Mask interpolation always uses NEAREST to preserve class indices.
    Color transforms are applied to the image only.

    Args:
        image_size: Target (height, width) for center crop.
        is_train: If True, apply random augmentations; otherwise resize + center crop only.
        ignore_index: Pixel value to use as fill for mask padding / rotation borders.
        aug_intensity: One of 'very_light', 'minimal', 'light', 'medium', 'strong'.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (320, 320),
        is_train: bool = True,
        ignore_index: int = 255,
        aug_intensity: str = "medium",
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.ignore_index = ignore_index
        self.aug_intensity = aug_intensity

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # -- resize so the short edge is at least crop_min -----------------
        target_min = min(self.image_size)
        if self.is_train:
            scale_min = {"very_light": 1.0, "minimal": 1.0, "light": 1.0, "medium": 1.0, "strong": 0.9}[self.aug_intensity]
            scale_max = {"very_light": 1.05, "minimal": 1.1, "light": 1.3, "medium": 1.6, "strong": 2.0}[self.aug_intensity]
            short_edge = random.randint(
                int(target_min * scale_min), int(target_min * scale_max)
            )
            short_edge = max(target_min, short_edge)
        else:
            short_edge = target_min

        image = TF.resize(image, short_edge, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, short_edge, interpolation=InterpolationMode.NEAREST)

        if not self.is_train:
            # Validation: deterministic resize + center crop only
            image = TF.center_crop(image, self.image_size)
            mask = TF.center_crop(mask, self.image_size)
        else:
            # -- random rotation -------------------------------------------
            if random.random() > 0.5:
                angle_lim = {"very_light": 3, "minimal": 5, "light": 10, "medium": 20, "strong": 45}[self.aug_intensity]
                angle = random.uniform(-angle_lim, angle_lim)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST,
                                 fill=self.ignore_index)

            # -- random crop -----------------------------------------------
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.image_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # -- horizontal flip -------------------------------------------
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # -- vertical flip (strong only) -------------------------------
            if self.aug_intensity == "strong" and random.random() > 0.7:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # -- colour jitter (image only) --------------------------------
            #   very_light: completely disabled — colour is critical for
            #   agricultural disease identification
            if self.aug_intensity != "very_light" and random.random() > 0.3:
                brightness = {"very_light": 0.0, "minimal": 0.1, "light": 0.2, "medium": 0.3, "strong": 0.5}[self.aug_intensity]
                contrast = {"very_light": 0.0, "minimal": 0.1, "light": 0.2, "medium": 0.3, "strong": 0.5}[self.aug_intensity]
                saturation = {"very_light": 0.0, "minimal": 0.1, "light": 0.2, "medium": 0.3, "strong": 0.5}[self.aug_intensity]
                hue = {"very_light": 0.0, "minimal": 0.02, "light": 0.05, "medium": 0.1, "strong": 0.15}[self.aug_intensity]
                image = transforms.ColorJitter(
                    brightness=brightness, contrast=contrast,
                    hue = hue, saturation = saturation
                )(image)

            # -- Gaussian blur (image only) --------------------------------
            #   very_light: completely disabled — blur destroys fine lesion
            #   texture needed for disease classification
            if self.aug_intensity != "very_light" and random.random() > 0.7:
                sigma = {"very_light": (0.1, 0.1), "minimal": (0.1, 0.5), "light": (0.1, 1.0), "medium": (0.1, 2.0), "strong": (0.1, 3.0)}[
                    self.aug_intensity
                ]
                image = transforms.GaussianBlur(
                    kernel_size=int(random.choice([3, 5])), sigma=sigma,
                )(image)

            # -- random gamma (strong only) --------------------------------
            if self.aug_intensity == "strong" and random.random() > 0.7:
                gamma = random.uniform(0.7, 1.3)
                image = TF.adjust_gamma(image, gamma)

        # -- to tensor ----------------------------------------------------
        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class VOCSegDataset(Dataset):
    """PASCAL VOC 2012 semantic segmentation dataset.

    21 classes (20 foreground + background).  Mask border pixels (value 255)
    are treated as ignore index.

    Expected layout::

        root/VOCdevkit/VOC2012/JPEGImages/*.jpg
        root/VOCdevkit/VOC2012/SegmentationClass/*.png
        root/VOCdevkit/VOC2012/ImageSets/Segmentation/{train,val}.txt
    """

    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor",
    ]
    NUM_CLASSES = 21
    IGNORE_INDEX = 255

    PALETTE = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128],
    ], dtype=np.uint8)

    def __init__(
        self, root: str, split: str = "train",
        image_size: Tuple[int, int] = (320, 320),
        aug_intensity: str = "medium",
    ):
        self.root = Path(root)
        self.split = split
        self.is_train = (split == "train")
        self.aug = ComposeAugmentation(
            image_size=image_size, is_train=self.is_train,
            ignore_index=self.IGNORE_INDEX, aug_intensity=aug_intensity,
        )
        split_file = (
            self.root / "VOCdevkit" / "VOC2012" / "ImageSets"
            / "Segmentation" / f"{split}.txt"
        )
        if split_file.exists():
            self.ids = [line.strip() for line in open(split_file) if line.strip()]
        else:
            mask_dir = self.root / "VOCdevkit" / "VOC2012" / "SegmentationClass"
            all_ids = sorted([p.stem for p in mask_dir.glob("*.png")])
            random.seed(42)
            random.shuffle(all_ids)
            n_train = int(len(all_ids) * 0.8)
            self.ids = all_ids[:n_train] if split == "train" else all_ids[n_train:]

        self.img_dir = self.root / "VOCdevkit" / "VOC2012" / "JPEGImages"
        self.mask_dir = self.root / "VOCdevkit" / "VOC2012" / "SegmentationClass"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]
        image = Image.open(self.img_dir / f"{img_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{img_id}.png")
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[mask_np == 255] = self.IGNORE_INDEX
        mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")
        image, mask = self.aug(image, mask)
        return image, mask


class BinarySegDataset(Dataset):
    """Generic binary segmentation dataset.

    Expected layout::

        root/images/{split}/*.{img_suffix}
        root/masks/{split}/*.{mask_suffix}

    Split names are auto-detected (train/training, val/validation/valid).
    If no split directory exists, a random 80/20 split is created.

    Args:
        val_split_ratio: Fraction of data to reserve for validation when
            no pre-existing split is found.
    """

    def __init__(
        self, root: str, split: str = "train",
        image_size: Tuple[int, int] = (320, 320),
        aug_intensity: str = "medium",
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
        val_split_ratio: float = 0.2,
    ):
        self.root = Path(root)
        self.split = split
        self.is_train = (split == "train")
        self.ignore_index = 255
        self.aug = ComposeAugmentation(
            image_size=image_size, is_train=self.is_train,
            ignore_index=255, aug_intensity=aug_intensity,
        )
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        # discover images
        img_dirs = list(self.root.glob("images/*"))
        all_files: List[Tuple[str, str]] = []
        for d in img_dirs:
            if d.is_dir():
                all_files.extend([(d.name, f.stem) for f in d.glob(f"*{img_suffix}")])

        if not all_files:
            raise FileNotFoundError(f"No images found under {self.root}/images/")

        by_split = defaultdict(list)
        for split_name, fname in all_files:
            by_split[split_name].append(fname)

        train_names = [n for n in by_split if n in ("train", "training")]
        val_names = [n for n in by_split if n in ("val", "validation", "valid")]

        if split == "train":
            if train_names:
                self.filenames = [f for sn in train_names for f in by_split[sn]]
                self.split_dir = train_names[0]
            else:
                all_fnames = list({f for fl in by_split.values() for f in fl})
                random.seed(42)
                random.shuffle(all_fnames)
                n_train = int(len(all_fnames) * (1 - val_split_ratio))
                self.filenames = all_fnames[:n_train]
                self.split_dir = list(by_split.keys())[0]
        else:
            if val_names:
                self.filenames = [f for sn in val_names for f in by_split[sn]]
                self.split_dir = val_names[0]
            else:
                all_fnames = list({f for fl in by_split.values() for f in fl})
                random.seed(42)
                random.shuffle(all_fnames)
                n_train = int(len(all_fnames) * (1 - val_split_ratio))
                self.filenames = all_fnames[n_train:]
                self.split_dir = list(by_split.keys())[0]

        self.img_dir = self.root / "images" / self.split_dir
        self.mask_dir = self.root / "masks" / self.split_dir

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        image = Image.open(self.img_dir / f"{fname}{self.img_suffix}").convert("RGB")
        mask = Image.open(self.mask_dir / f"{fname}{self.mask_suffix}").convert("L")
        image, mask = self.aug(image, mask)
        mask = (mask > 127).long()
        return image, mask


class CamVidDataset(Dataset):
    """CamVid urban driving semantic segmentation dataset.

    11 classes.  Masks are colour-coded RGB images that are converted to
    class-index format at load time.

    Expected layout::

        root/images/*.png
        root/labels/*_L.png
    """

    CLASSES = [
        "sky", "building", "pole", "road", "pavement",
        "tree", "signsymbol", "fence", "car", "pedestrian", "bicyclist",
    ]
    NUM_CLASSES = 11
    IGNORE_INDEX = 255

    COLOR_MAP = {
        (128, 128, 128): 0,   # sky
        (128, 0, 0): 1,       # building
        (192, 192, 128): 2,   # pole
        (128, 64, 128): 3,    # road
        (0, 0, 192): 4,       # pavement
        (128, 128, 0): 5,     # tree
        (192, 128, 128): 6,   # signsymbol
        (64, 64, 128): 7,     # fence
        (64, 0, 128): 8,      # car
        (64, 64, 0): 9,       # pedestrian
        (0, 128, 192): 10,    # bicyclist
    }

    PALETTE = np.array([
        [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
        [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
        [64, 0, 128], [64, 64, 0], [0, 128, 192],
    ], dtype=np.uint8)

    def __init__(
        self, root: str, split: str = "train",
        image_size: Tuple[int, int] = (320, 320),
        aug_intensity: str = "medium",
    ):
        self.root = Path(root)
        self.split = split
        self.is_train = (split == "train")
        self.aug = ComposeAugmentation(
            image_size=image_size, is_train=self.is_train,
            ignore_index=self.IGNORE_INDEX, aug_intensity=aug_intensity,
        )
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "labels"

        all_files = sorted([
            f.stem for f in self.img_dir.glob("*.png")
            if not f.stem.endswith("_L")
        ])
        random.seed(42)
        random.shuffle(all_files)
        n = len(all_files)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        if split == "train":
            self.filenames = all_files[:n_train]
        elif split == "val":
            self.filenames = all_files[n_train:n_train + n_val]
        else:
            self.filenames = all_files[n_train + n_val:]

    def __len__(self) -> int:
        return len(self.filenames)

    @staticmethod
    def _rgb_to_class(color_mask: np.ndarray) -> np.ndarray:
        h, w, _ = color_mask.shape
        class_mask = np.full((h, w), 255, dtype=np.uint8)
        for rgb, cls_idx in CamVidDataset.COLOR_MAP.items():
            match = np.all(color_mask == rgb, axis=-1)
            class_mask[match] = cls_idx
        return class_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]
        image = Image.open(self.img_dir / f"{fname}.png").convert("RGB")
        mask_rgb = Image.open(self.mask_dir / f"{fname}_L.png").convert("RGB")
        mask_np = self._rgb_to_class(np.array(mask_rgb))
        mask = Image.fromarray(mask_np, mode="L")
        image, mask = self.aug(image, mask)
        return image, mask


class CityscapesDataset(Dataset):
    """Cityscapes urban scene semantic segmentation dataset.

    19 evaluation classes (plus void/ignore).  Raw label IDs (0--33) are
    mapped to train IDs (0--18, 255) using the standard Cityscapes mapping.

    Expected layout::

        root/leftImg8bit/{train,val,test}/{city}/*_leftImg8bit.png
        root/gtFine/{train,val,test}/{city}/*_gtFine_labelIds.png
    """

    CLASSES = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train",
        "motorcycle", "bicycle",
    ]
    NUM_CLASSES = 19
    IGNORE_INDEX = 255

    # Cityscapes label ID → train ID mapping
    # Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    _LABEL_TO_TRAINID = np.array([
        255, 255, 255, 255, 255, 255, 255,    # 0-6
        0, 1, 255, 255, 2, 3, 4, 255,         # 7-14
        255, 255, 5, 255, 6, 7, 8, 9,         # 15-22
        10, 11, 12, 13, 14, 15, 255, 255,     # 23-30
        16, 17, 18,                            # 31-33
    ], dtype=np.uint8)

    PALETTE = np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32],
    ], dtype=np.uint8)

    def __init__(
        self, root: str, split: str = "train",
        image_size: Tuple[int, int] = (512, 1024),
        aug_intensity: str = "medium",
    ):
        self.root = Path(root)
        self.split = split
        self.is_train = (split == "train")
        self.aug = ComposeAugmentation(
            image_size=image_size, is_train=self.is_train,
            ignore_index=self.IGNORE_INDEX, aug_intensity=aug_intensity,
        )

        img_dir = self.root / "leftImg8bit" / split
        self.samples: List[Tuple[Path, Path]] = []
        for city in sorted(img_dir.iterdir()):
            if not city.is_dir():
                continue
            for img_path in sorted(city.glob("*_leftImg8bit.png")):
                base = img_path.name.replace("_leftImg8bit.png", "")
                label_path = (
                    self.root / "gtFine" / split / city.name
                    / f"{base}_gtFine_labelIds.png"
                )
                if label_path.exists():
                    self.samples.append((img_path, label_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        # Map label IDs → train IDs
        label_np = np.array(label, dtype=np.uint8)
        label_np = self._LABEL_TO_TRAINID[label_np]
        mask = Image.fromarray(label_np, mode="L")

        image, mask = self.aug(image, mask)
        return image, mask


class COCOLeafDataset(Dataset):
    """COCO Tea Leaf disease segmentation dataset (8 classes).

    Custom synthetic dataset with COCO-format JSON annotations.

    Expected layout::

        root/annotations.json
        root/train_images/
        root/val_images/
    """

    CLASSES = [
        "healthy_leaf", "brown_blight", "red_spider_mite", "red_rust",
        "gray_blight", "helopeltis", "green_mirid_bug", "tea_algal_leaf_spot",
    ]
    NUM_CLASSES = 8
    IGNORE_INDEX = 255

    PALETTE = np.array([
        [0, 255, 0],       # 0: healthy_leaf
        [255, 0, 0],       # 1: brown_blight
        [255, 165, 0],     # 2: red_spider_mite
        [255, 0, 255],     # 3: red_rust
        [0, 255, 255],     # 4: gray_blight
        [255, 255, 0],     # 5: helopeltis
        [0, 128, 255],     # 6: green_mirid_bug
        [128, 0, 255],     # 7: tea_algal_leaf_spot
    ], dtype=np.uint8)

    def __init__(
        self, root: str, split: str = "train",
        image_size: Tuple[int, int] = (320, 320),
        aug_intensity: str = "medium",
    ):
        self.root = Path(root)
        self.split = split
        self.is_train = (split == "train")
        self.aug = ComposeAugmentation(
            image_size=image_size, is_train=self.is_train,
            ignore_index=self.IGNORE_INDEX, aug_intensity=aug_intensity,
        )

        ann_path = self.root / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}. "
                f"Generate it with: python data/coco_leaf_generator.py"
            )

        with open(ann_path) as f:
            self.coco_data = json.load(f)

        self.images = [
            img for img in self.coco_data["images"]
            if img.get("split", "train") == split
        ]
        self.ann_by_image = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.ann_by_image[ann["image_id"]].append(ann)
        self.img_by_id = {img["id"]: img for img in self.images}
        self.img_dir = self.root / ("train_images" if split == "train" else "val_images")

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _decode_segmentation(seg, height: int, width: int) -> np.ndarray:
        """Decode COCO segmentation (RLE or polygon) to binary mask."""
        if isinstance(seg, dict):
            try:
                from pycocotools import mask as cocomask
                m = cocomask.decode(seg)
                if m.ndim == 3:
                    m = m[:, :, 0]
                return m.astype(np.uint8)
            except ImportError:
                pass
        mask = np.zeros((height, width), dtype=np.uint8)
        if isinstance(seg, list):
            for poly in seg:
                if isinstance(poly, list):
                    import cv2
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 1)
        return mask

    def _build_class_mask(self, image_id: int, height: int, width: int) -> np.ndarray:
        class_mask = np.zeros((height, width), dtype=np.int64)
        for ann in self.ann_by_image.get(image_id, []):
            cat_id = ann["category_id"]
            seg = ann["segmentation"]
            binary = self._decode_segmentation(seg, height, width)
            class_mask[binary > 0] = cat_id
        return class_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = self.root / file_name
        image = Image.open(img_path).convert("RGB")
        mask_np = self._build_class_mask(img_id, img_info["height"], img_info["width"])
        mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")
        image, mask = self.aug(image, mask)
        return image, mask


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_INFO: Dict[str, dict] = {
    "coco_leaf": {
        "name": "COCO Tea Leaf",
        "num_classes": 8,
        "type": "multi-class",
        "ignore_index": 255,
        "train_class": COCOLeafDataset,
        "val_class": COCOLeafDataset,
        "palette": COCOLeafDataset.PALETTE,
        "class_names": COCOLeafDataset.CLASSES,
    },
    "camvid": {
        "name": "CamVid",
        "num_classes": 11,
        "type": "multi-class",
        "ignore_index": 255,
        "train_class": CamVidDataset,
        "val_class": CamVidDataset,
        "palette": CamVidDataset.PALETTE,
        "class_names": CamVidDataset.CLASSES,
    },
    "cityscapes": {
        "name": "Cityscapes",
        "num_classes": 19,
        "type": "multi-class",
        "ignore_index": 255,
        "train_class": CityscapesDataset,
        "val_class": CityscapesDataset,
        "palette": CityscapesDataset.PALETTE,
        "class_names": CityscapesDataset.CLASSES,
    },
    "voc": {
        "name": "PASCAL VOC 2012",
        "num_classes": 21,
        "type": "multi-class",
        "ignore_index": 255,
        "train_class": VOCSegDataset,
        "val_class": VOCSegDataset,
        "palette": VOCSegDataset.PALETTE,
        "class_names": VOCSegDataset.CLASSES,
    },
    "kvasir": {
        "name": "Kvasir-SEG",
        "num_classes": 1,
        "type": "binary",
        "ignore_index": 255,
        "train_class": BinarySegDataset,
        "val_class": BinarySegDataset,
        "palette": np.array([[0, 0, 0], [0, 255, 0]], dtype=np.uint8),
        "class_names": ["background", "polyp"],
    },
    "isic": {
        "name": "ISIC 2018",
        "num_classes": 1,
        "type": "binary",
        "ignore_index": 255,
        "train_class": BinarySegDataset,
        "val_class": BinarySegDataset,
        "palette": np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8),
        "class_names": ["background", "lesion"],
    },
}

DEFAULT_ROOTS: Dict[str, str] = {
    "coco_leaf": "data/coco_leaf",
    "camvid": "data/camvid",
    "cityscapes": "data/cityscapes",
    "voc": "data/pascal_voc",
    "kvasir": "data/kvasir_seg",
    "isic": "data/isic2018",
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    dataset_name: str,
    image_size: Tuple[int, int] = (320, 320),
    batch_size: int = 16,
    num_workers: int = 4,
    aug_intensity: str = "strong",
    data_root: Optional[str] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DatasetInfo]:
    """Create training and validation DataLoaders for a named dataset.

    Args:
        dataset_name: One of ``coco_leaf``, ``camvid``, ``cityscapes``,
                      ``voc``, ``kvasir``, ``isic``.
        image_size: (height, width) input resolution.
        batch_size: Batch size for training (validation uses same).
        num_workers: DataLoader worker processes.
        aug_intensity: ``light``, ``medium``, or ``strong``.
        data_root: Override default data directory.
        pin_memory: Enable pinned memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader, info) where *info* is a
        :class:`DatasetInfo` dataclass.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_INFO.keys())}"
        )

    entry = DATASET_INFO[dataset_name]
    root = data_root or DEFAULT_ROOTS[dataset_name]
    root = str(Path(root).resolve())

    if not os.path.exists(root):
        print(f"[Warning] Data directory not found: {root}")
        print(f"  Download with: python tools/download_datasets.py "
              f"--dataset {dataset_name}")

    train_kwargs = dict(
        root=root, split="train", image_size=image_size,
        aug_intensity=aug_intensity,
    )
    val_kwargs = dict(
        root=root, split="val", image_size=image_size,
        aug_intensity="medium",
    )

    # Handle datasets with non-standard split names
    if dataset_name in ("camvid", "voc"):
        pass  # already uses split="train"/"val"
    elif dataset_name == "cityscapes":
        pass  # uses split="train"/"val"

    train_ds = entry["train_class"](**train_kwargs)
    val_ds = entry["val_class"](**val_kwargs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    info = DatasetInfo(
        name=entry["name"],
        num_classes=entry["num_classes"],
        type=entry["type"],
        ignore_index=entry.get("ignore_index", 255),
        palette=entry.get("palette"),
        class_names=list(entry.get("class_names", [])),
        root=root,
        train_size=len(train_ds),
        val_size=len(val_ds),
        image_size=image_size,
    )
    return train_loader, val_loader, info


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def label_to_color(
    mask: np.ndarray, palette: np.ndarray, ignore_index: int = 255,
) -> np.ndarray:
    """Convert a class-index mask (H, W) to an RGB colour image (H, W, 3)."""
    h, w = mask.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    colour[mask == ignore_index] = [0, 0, 0]
    for cls_idx in range(palette.shape[0]):
        colour[mask == cls_idx] = palette[cls_idx]
    return colour


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a tensor to a numpy image for display.

    If the tensor has already been ImageNet-normalised (values outside
    [0, 1]), the normalisation is reversed.  Otherwise the tensor is
    assumed to be in [0, 1] range and returned as-is.
    """
    t_min, t_max = tensor.min().item(), tensor.max().item()
    if t_min >= -0.05 and t_max <= 1.05:
        # Already in displayable range — no normalisation to reverse
        return tensor.permute(1, 2, 0).numpy()

    # Reverse ImageNet normalisation
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    tensor = tensor.clone()
    for t, m, s in zip(tensor, inv_mean, inv_std):
        t.mul_(s).add_(m)
    return np.clip(tensor.permute(1, 2, 0).numpy(), 0, 1)
