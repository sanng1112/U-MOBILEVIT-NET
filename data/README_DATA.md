# U-MobileViT-Net — Datasets Directory

Thư mục chứa tất cả bộ dữ liệu benchmark cho tác vụ Image Segmentation.

## Cấu trúc

```
data/
├── README_DATA.md           ← File này
├── camvid/                  🟢 Urban Driving (11 classes, ~570MB)
├── pascal_voc/              🟢 General Objects (21 classes, ~2GB)
├── cityscapes/              🟢 Urban Scene (19 classes, ~10GB) — cần tải thủ công
├── kvasir_seg/              🔵 Medical Polyp (1 class, ~50MB)
├── isic2018/                🔵 Skin Lesion (1 class, ~3GB)
├── drive/                   🔵 Retinal Vessel (1 class, ~33MB)
├── COCO/                    🟡 COCO 2017 (cũ — không khuyến nghị cho segmentation)
├── coco_leaf/               🟡 Custom tea leaf dataset
├── teaLeafBD/               🟡 Raw tea leaf data
└── extracted_leaf_objects/  🟡 Extracted leaf objects
```

## Tổng quan Benchmark

| Dataset | Lớp | Dung lượng | Loại | Phù hợp |
|---------|-----|-----------|------|---------|
| **CamVid** | 11 | ~570MB | Urban driving | ⭐ Prototype nhanh |
| **PASCAL VOC** | 21 | ~2GB | General objects | ⭐ Benchmark chính |
| **Cityscapes** | 19 | ~10GB | Urban scene | ⭐ Edge deployment demo |
| **Kvasir-SEG** | 1 | ~50MB | Medical (polyp) | Medical baseline |
| **ISIC 2018** | 1 | ~3GB | Medical (skin) | Medical baseline |
| **DRIVE** | 1 | ~33MB | Medical (retina) | UNet classic benchmark |
| COCO | 80 | ~25GB | Instance seg. | ❌ Không phù hợp |

## Công cụ quản lý

```bash
# Xem trạng thái datasets
python tools/check_data.py

# Tải dataset
python tools/download_datasets.py --list               # Liệt kê
python tools/download_datasets.py --dataset camvid     # Tải 1 dataset
python tools/download_datasets.py --dataset all        # Tải tất cả
```

## Dataset đặc thù (cũ)

Các thư mục `5000_tea_leaf_with_blackbg_geotagged/`, `teaLeafBD/`, `coco_leaf/`,
và `extracted_leaf_objects/` chứa dữ liệu về bệnh lá trà (8 classes).
Chi tiết xem notebook `data/data.ipynb`.
