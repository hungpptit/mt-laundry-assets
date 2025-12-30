# 📘 README - Training AutoEncoder cho Iris Liveness Detection

## 🎯 Mục đích
Train model AutoEncoder để phát hiện iris liveness (REAL vs FAKE) sử dụng **Anomaly Detection approach** - chỉ train trên ảnh REAL.

## 📊 Dataset: UBIPR2
- **Tổng số ảnh gốc**: ~5000 images
- **Sau preprocessing**: 1517 images (128×128 RGB)
- **Phân chia**: Train (70%) + Validation (30%)
- **Loại**: Near-infrared iris images đã loại bỏ lông mày

---

## 🔥 CELLS QUAN TRỌNG (Theo thứ tự ưu tiên)

### ⭐ Cell 4: TIỀN XỬ LÝ ẢNH (Quan trọng nhất)
**Mục đích**: Tạo thư mục `processed_clean` chứa ảnh đã xử lý

**Quy trình xử lý**:
```python
1. Đọc ảnh gốc từ images/
2. Đọc mask từ masks/
3. CẮT 1/3 PHẦN TRÊN của mask (bỏ lông mày)
4. Áp mask lên ảnh gốc (chỉ giữ vùng iris)
5. Resize về 128×128
6. Lưu vào processed_clean/
```

**Input**: 
- `images/*.jpg` - Ảnh iris gốc
- `masks/*.png` - Mask vùng iris
- `split/train.txt` - Danh sách file train

**Output**: `processed_clean/*.jpg` (1517 ảnh 128×128)

**Lý do quan trọng**: 
- ✅ Loại bỏ nhiễu (lông mày, mí mắt)
- ✅ Chuẩn hóa kích thước
- ✅ Tăng độ chính xác model

---

### ⭐ Cell 7: TRAINING LOOP (CHÍNH)
**Mục đích**: Train model AutoEncoder

**Hyperparameters**:
- **Epochs**: 100
- **Batch size**: 32
- **Learning rate**: 1e-3
- **Optimizer**: AdamW (weight_decay=1e-5)
- **Loss**: MSELoss
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

**Early Stopping**:
- Patience: 10 epochs
- Save best model dựa trên validation loss

**Quy trình**:
```
For each epoch:
  1. Train trên training set
  2. Validate trên validation set
  3. Tính loss
  4. Scheduler giảm learning rate nếu loss không giảm
  5. Save model nếu validation loss thấp nhất
  6. Early stopping nếu không cải thiện sau 10 epochs
```

**Output**: 
- Model weights: `models/autoencoder_processed_clean.pt`
- Training history: lists của train_loss, val_loss

---

### ⭐ Cell 6: MODEL DEFINITION
**Mục đích**: Định nghĩa Enhanced AutoEncoder architecture

**Architecture**:
```
INPUT: 128×128×3 (RGB)

ENCODER (Downsampling):
  Conv2d(3→32) + BatchNorm + ReLU → 64×64×32
  Conv2d(32→64) + BatchNorm + ReLU → 32×32×64
  Conv2d(64→128) + BatchNorm + ReLU → 16×16×128
  Conv2d(128→256) + BatchNorm + ReLU + Dropout(0.2) → 8×8×256

DECODER (Upsampling):
  ConvTranspose2d(256→128) + BatchNorm + ReLU → 16×16×128
  ConvTranspose2d(128→64) + BatchNorm + ReLU → 32×32×64
  ConvTranspose2d(64→32) + BatchNorm + ReLU → 64×64×32
  ConvTranspose2d(32→3) + Sigmoid → 128×128×3

OUTPUT: 128×128×3 (Reconstructed RGB)
```

**Parameters**: ~2.5M
**Key features**:
- BatchNorm: Chuẩn hóa, tăng tốc training
- Dropout: Tránh overfitting
- Sigmoid: Output trong [0, 1]

---

### ⭐ Cell 5: DATASET CLASS
**Mục đích**: Load và augment data

**Augmentation (training)**:
```python
- RandomHorizontalFlip(p=0.5)
- RandomRotation(±10 degrees)
- ColorJitter(brightness, contrast, saturation, hue)
- GaussianBlur(kernel=3)
```

**Augmentation (validation)**: None

**Normalization**: ToTensor() (scale 0-1)

---

### ⭐ Cell 10: THRESHOLD CALCULATION
**Mục đích**: Tính ngưỡng phát hiện FAKE

**Phương pháp**:
```python
threshold = mean_MSE + 2 * std_MSE
```

**Ý nghĩa**:
- MSE < threshold → REAL (giống ảnh train)
- MSE ≥ threshold → FAKE (khác ảnh train)

**Output**: Threshold value dùng cho inference

---

## 📦 CELLS THIẾT LẬP (Chạy 1 lần đầu)

### Cell 1: Install Packages
```python
!pip install opencv-python-headless numpy matplotlib torch torchvision tqdm -q
```

### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 3: Imports & Config
- Import libraries
- Định nghĩa paths
- Check CUDA

---

## 📈 CELLS VISUALIZATION (Không quan trọng cho training)

### Cell 8: Training Loss Visualization
- Vẽ đồ thị train loss vs validation loss
- Không ảnh hưởng đến model

### Cell 9: Load Best Model & Reconstruction Check
- Load model tốt nhất
- Test reconstruction trên vài ảnh
- Visualization only

### Cell 11: Report Figures (4 ảnh)
- `report_training_curves.png`
- `report_best_worst_cases.png`
- `report_mse_distribution.png`
- `report_summary.png`

### Cell 12: Optional Model Download
- Download model về local
- Không cần thiết nếu chạy trên Colab

### Cell 13: Performance Evaluation (Upload Images)
- Upload ảnh test để tính metrics
- Accuracy, Precision, Recall, F1, AUC
- Chỉ để đánh giá, không train

### Cell 14: Experimental Setup Tables
- Tạo bảng thông số kỹ thuật
- 6 bảng + 2 PNG + 6 CSV
- Dùng cho báo cáo luận văn

### Cell 15: Chapter 2 Architectural Diagrams
- 5 sơ đồ kiến trúc
- Dùng cho luận văn
- Không liên quan training

---

## 🚀 QUY TRÌNH CHẠY ĐÚNG (Theo thứ tự)

### 1️⃣ GIAI ĐOẠN SETUP (Chỉ 1 lần):
```
Cell 1 → Install packages
Cell 2 → Mount Drive
Cell 3 → Import & Config
```

### 2️⃣ GIAI ĐOẠN XỬ LÝ DỮ LIỆU:
```
Cell 4 → Preprocessing ảnh (QUAN TRỌNG)
Cell 5 → Dataset class
```

### 3️⃣ GIAI ĐOẠN TRAINING (CHÍNH):
```
Cell 6 → Define Model
Cell 7 → Training Loop ⭐⭐⭐
Cell 10 → Calculate Threshold
```

### 4️⃣ GIAI ĐOẠN VISUALIZATION (Tùy chọn):
```
Cell 8 → Loss curves
Cell 9 → Reconstruction check
Cell 11-15 → Report generation
```

---

## ⚙️ THÔNG SỐ QUAN TRỌNG CẦN NHỚ

| Thông số | Giá trị | Ghi chú |
|----------|---------|---------|
| Input size | 128×128×3 | RGB image |
| Batch size | 32 | Có thể giảm nếu out of memory |
| Learning rate | 1e-3 | AdamW optimizer |
| Epochs | 100 | Early stopping patience=10 |
| Model params | 2.5M | Enhanced architecture |
| Loss function | MSELoss | Reconstruction error |
| Augmentation | 4 types | Training only |
| Threshold | Mean + 2×Std | From validation MSE |

---

## 📂 CẤU TRÚC FOLDER

```
/content/drive/MyDrive/dataset/ubipr2/
│
├── images/                    # Ảnh gốc (~5000 ảnh)
│   ├── F001_1.jpg
│   └── ...
│
├── masks/                     # Mask vùng iris
│   ├── F001_1.png
│   └── ...
│
├── split/                     # File phân chia train/test
│   ├── train.txt             # Danh sách file train
│   └── test.txt              # Danh sách file test
│
├── processed_clean/          # ⭐ Ảnh đã xử lý (Cell 4 tạo)
│   ├── F001_1.jpg           # 128×128, đã crop eyebrow
│   └── ... (1517 files)
│
└── models/                    # ⭐ Model sau khi train (Cell 7 tạo)
    └── autoencoder_processed_clean.pt
```

---

## 🔍 KIỂM TRA SAU KHI TRAIN

### 1. Model file tồn tại:
```python
import os
os.path.exists(f"{base_dir}/models/autoencoder_processed_clean.pt")
```

### 2. Validation loss giảm:
- Xem đồ thị Cell 8
- Val loss phải < Train loss cuối cùng

### 3. Reconstruction tốt:
- Chạy Cell 9
- Ảnh reconstruct phải giống ảnh input

### 4. Threshold hợp lý:
- Chạy Cell 10
- Threshold thường trong khoảng 0.01-0.03

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. **Cell 4 PHẢI CHẠY TRƯỚC Cell 7**
- Cell 7 load data từ `processed_clean/`
- Nếu chưa có folder này → Lỗi!

### 2. **Đường dẫn Google Drive**
```python
base_dir = "/content/drive/MyDrive/dataset/ubipr2"
```
- ⚠️ SỬA ĐƯỜNG DẪN NÀY CHO ĐÚNG!
- Phải có: images/, masks/, split/

### 3. **GPU vs CPU**
- GPU: ~10 phút/epoch
- CPU: ~60 phút/epoch
- Khuyến nghị: Dùng GPU (Colab Pro)

### 4. **Early Stopping**
- Nếu val loss không giảm sau 10 epochs → Dừng tự động
- Không cần chờ hết 100 epochs

### 5. **Augmentation chỉ cho Train**
- Validation set KHÔNG augment
- Đảm bảo đánh giá đúng

---

## 🎓 TÓM TẮT CHO NGƯỜI MỚI

**Muốn train model từ đầu**:
1. Chạy Cell 1-3 (setup)
2. Chạy Cell 4 (preprocessing) ⭐
3. Chạy Cell 5-6 (dataset + model)
4. Chạy Cell 7 (TRAINING - đợi ~30-60 phút) ⭐⭐⭐
5. Chạy Cell 10 (threshold)
6. Xong! Model lưu ở `models/autoencoder_processed_clean.pt`

**Các cell còn lại (8, 9, 11-15)**: Chỉ để visualization và report, không bắt buộc.

---

## 📞 TROUBLESHOOTING

### Lỗi: "FileNotFoundError: processed_clean/"
→ **Giải pháp**: Chạy lại Cell 4 (preprocessing)

### Lỗi: "CUDA out of memory"
→ **Giải pháp**: Giảm batch_size từ 32 xuống 16 hoặc 8

### Lỗi: "No such file: images/"
→ **Giải pháp**: Sửa đường dẫn `base_dir` trong Cell 3

### Val loss không giảm:
→ **Nguyên nhân**: Data không đủ đa dạng hoặc augmentation quá mạnh
→ **Giải pháp**: Điều chỉnh augmentation trong Cell 5

---

**Tác giả**: Iris Liveness Detection Project  
**Dataset**: UBIPR2 (Near-infrared iris images)  
**Model**: Enhanced AutoEncoder (2.5M params)  
**Approach**: Anomaly Detection (train on REAL only)
