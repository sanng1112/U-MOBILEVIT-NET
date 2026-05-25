# 🍃 Synthetic Tea Leaf Disease Dataset Generator (COCO Format)

Project này cung cấp một pipeline hoàn chỉnh để **tự động phân đoạn (segmentation)**, **trích xuất** các chiếc lá trà từ dữ liệu thô (có nền đen hoặc nền trắng/nhiễu), và **tạo ra một tập dữ liệu tổng hợp (synthetic dataset)** mới dành cho bài toán Object Detection & Instance Segmentation.

Tập dữ liệu đầu ra được format theo chuẩn **COCO JSON** (hỗ trợ RLE và Polygon), sẵn sàng để train các mô hình như Mask R-CNN, YOLOv8-Seg, v.v.

---

## ✨ Tính năng chính

1. **Phân đoạn nền thông minh (Smart Background Removal):**
   - **Với nền đen:** Sử dụng Global Thresholding + Morphology + Connected Components.
   - **Với nền trắng/nhiễu:** Sử dụng Otsu's Inverse + Morphology + **Center Prior** (ưu tiên giữ lại vật thể nằm giữa khung hình, tự động xóa rác ở viền).
2. **Xử lý đa luồng (Multithreading):** Tăng tốc độ trích xuất hàng nghìn ảnh từ dataset gốc bằng `ThreadPoolExecutor`.
3. **Data Augmentation phong phú:** Lật (Flip), Xoay ngẫu nhiên (-30° đến 30°), Thu phóng (Resize), và Thay đổi màu sắc (Color Jitter: Brightness, Contrast, Color).
4. **Cân bằng Class (Class Balancing):** Sử dụng trọng số lũy thừa (`alpha = 0.5`) để tăng tỷ lệ xuất hiện của các loại bệnh hiếm (như *tea_algal_leaf_spot*) trong ảnh tổng hợp.
5. **Thuật toán sinh ảnh ngẫu nhiên có kiểm soát:**
   - Số lượng lá phân bổ theo **Poisson Distribution**.
   - Phân cụm (Clustering) lá ở các khu vực tự nhiên.
   - Kiểm soát chồng chéo (Overlap Control) bằng Erosion mask, cho phép lá đè lên nhau một cách hợp lý mà không bị che khuất hoàn toàn.
6. **Chuẩn đầu ra COCO Format:** Hỗ trợ lưu segmentation dưới dạng RLE (nhỏ gọn) hoặc Polygon.

---

## 🛠 Yêu cầu hệ thống (Prerequisites)

Hãy đảm bảo bạn đã cài đặt Python 3.8+ và các thư viện sau:

```bash
pip install numpy pandas matplotlib opencv-python pillow torch tqdm pycocotools
```
*Lưu ý: Thư viện `pycocotools` rất được khuyến khích cài đặt để tối ưu dung lượng file JSON bằng định dạng RLE. Nếu không cài đặt, script sẽ tự động fallback về định dạng Polygon.*

---

## 📁 Cấu trúc thư mục

Trước khi chạy, hãy đảm bảo thư mục gốc của bạn chứa 2 thư mục dataset thô như sau:

```text
.
├── 5000_tea_leaf_with_blackbg_geotagged/  # Dataset 1: Nền đen
│   ├── BB/
│   ├── GL/
│   ├── RR/
│   └── RSM/
├── teaLeafBD/                             # Dataset 2: Nền trắng/nhiễu
│   ├── 1. Tea algal leaf spot/
│   ├── 2. Brown Blight/
│   └── ...
├── data.ipynb                                # File script của bạn
└── README.md
```

### Cấu trúc sau khi chạy script:
```text
.
├── extracted_leaf_objects/                # Chứa các lá đã được tách nền (file .png RGBA)
├── coco_leaf/                             # THƯ MỤC DATASET ĐẦU RA (TỔNG HỢP)
│   ├── train_images/                      # Ảnh tổng hợp cho tập Train
│   ├── val_images/                        # Ảnh tổng hợp cho tập Val
│   └── annotations.json                   # File COCO Annotations
...
```

---

## 🚀 Hướng dẫn luồng hoạt động (Pipeline Workflow)

Script chạy tuần tự qua các bước sau:

1. **Khảo sát (Audit) & Hiển thị dữ liệu gốc:** Quét các class, đếm số lượng ảnh và map (ánh xạ) chúng về 8 class chuẩn (`UNIFIED_CLASSES`).
2. **Tách nền & Lưu thư viện lá (Object Extraction):** 
   - Chạy song song để tách các lá ra khỏi nền.
   - Các lá hợp lệ (diện tích đủ lớn) sẽ được lưu vào thư mục `extracted_leaf_objects` với định dạng RGBA (nền trong suốt).
3. **Sinh dữ liệu tổng hợp (Synthetic Generation):**
   - Lấy ngẫu nhiên các lá từ thư viện (dựa theo xác suất đã được cân bằng).
   - Augment (xoay, đổi màu, kích thước) chiếc lá.
   - Tạo canvas `320x320` pixel (mặc định).
   - "Dán" các chiếc lá lên canvas theo các cụm ngẫu nhiên.
   - Ghi nhận lại tọa độ Bounding Box, Diện tích và Mask cho từng chiếc lá.
4. **Xuất file COCO:** Chia tập dữ liệu thành `train` và `val` theo tỷ lệ (mặc định 80/20) và xuất file `annotations.json`.
5. **Kiểm tra trực quan (Visualization):** Đọc lại file `annotations.json` và in đè mask có màu lên ảnh tổng hợp để người dùng kiểm chứng độ chính xác.

---

## ⚙️ Các tham số có thể cấu hình (Configurations)

Bạn có thể thay đổi các hằng số ở đầu / giữa script để tùy biến bộ dữ liệu đầu ra:

| Biến | Giá trị mặc định | Ý nghĩa |
|------|------------------|---------|
| `TARGET_SIZE` | `(224, 224)` | Kích thước resize ảnh gốc ban đầu khi trích xuất nền. |
| `MAX_WORKERS` | `8` | Số lượng luồng (threads) chạy song song khi tách nền. Chỉnh nhỏ lại nếu máy yếu. |
| `IMAGE_SIZE` | `320` | Kích thước (Width x Height) của ảnh tổng hợp đầu ra (Synthetic image). |
| `MAX_OBJECTS` | `50` | Số lượng lá tối đa được dán vào 1 ảnh tổng hợp. |
| `LAMBDA_POISSON` | `80.0` | Tham số lambda của phân phối Poisson quyết định mật độ lá trung bình. |
| `TRAIN_RATIO` | `0.8` | Tỷ lệ chia tập Train / Val (80% Train, 20% Val). |
| `MIN_OBJ_WIDTH` / `MAX` | `50` / `300` | Phạm vi kích thước (Width) ngẫu nhiên của mỗi chiếc lá khi dán vào ảnh. |

---

## 🏷 Danh sách các nhãn (Unified Classes)

Tất cả dữ liệu từ 2 dataset khác nhau được gom chung về 8 nhãn ID (từ 0 đến 7):

0. `healthy_leaf`
1. `brown_blight`
2. `red_spider_mite`
3. `red_rust`
4. `gray_blight`
5. `helopeltis`
6. `green_mirid_bug`
7. `tea_algal_leaf_spot`

