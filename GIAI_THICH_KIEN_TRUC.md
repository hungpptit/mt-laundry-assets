# GIẢI THÍCH CHI TIẾT KIẾN TRÚC HỆ THỐNG

> **Tài liệu này giải thích chi tiết các hình minh hoạ về kiến trúc, thuật toán, triển khai và đánh giá hệ thống phát hiện liveness mống mắt (Hình 2.1–2.5, Hình 3.1, Hình 3.3).**

---

## 📐 HÌNH 2.1: KIẾN TRÚC TỔNG THỂ HỆ THỐNG

### Tổng quan
Hệ thống được chia thành **2 giai đoạn chính**: Training (huấn luyện) và Inference (suy diễn thời gian thực).

---

### 🔵 PHASE 1: TRAINING (Giai đoạn Huấn luyện)

#### 📦 Input: Dataset UBIPR2
```
Dataset UBIPR2
├─ Chỉ chứa ảnh mống mắt THẬT (REAL iris only)
├─ ~5000 ảnh gốc
└─ 3855 ảnh sau preprocessing
```

**Đặc điểm:**
- ✅ **Chỉ dùng ảnh REAL**: Không cần ảnh giả (FAKE) trong training
- ✅ **One-class learning**: Học đặc trưng của ảnh thật
- 📊 **Data quality**: Ảnh cận hồng ngoại (NIR) chất lượng cao

#### 🔄 Preprocessing (Tiền xử lý)
```
Raw Image
    ↓
[1] Crop eyebrows (1/3 top)
    ↓ Loại bỏ phần lông mày, giữ lại vùng mống mắt
[2] Apply mask
    ↓ Chỉ giữ vùng iris, loại bỏ background
[3] Resize to 128×128
    ↓ Chuẩn hóa kích thước
Clean Image (128×128×3)
```

**Tại sao cần preprocessing?**
- 🎯 **Focus on iris**: Loại bỏ nhiễu từ lông mày, mi mắt
- 📏 **Standardization**: Đồng nhất kích thước đầu vào
- 🧹 **Noise reduction**: Giảm thiểu ảnh hưởng của background

#### 🧠 AutoEncoder Model
```
Input: 128×128×3
    ↓
[Encoder] Compress → Latent Space (8×8×256)
    ↓
[Decoder] Reconstruct
    ↓
Output: 128×128×3
```

**Thông số mô hình:**
- 🔢 **Parameters**: ~2.5M (~0.78M trainable)
- 🏗️ **Architecture**: Convolutional AutoEncoder
- 📊 **Latent dimension**: 8×8×256 = 16,384 features
- ⚡ **Compression ratio**: 49,152 → 16,384 (~48% compression)

#### 💾 Output: Trained Model
```
Trained Model (.pt file)
├─ Encoder weights
├─ Decoder weights
├─ Statistics (Mean, Std)
└─ Threshold = Mean(REAL MSE) + 2×Std(REAL MSE)
```

**Kết quả sau training:**
- 📉 **Loss giảm**: 0.135653 → 0.000215 (99.84%)
- 🎯 **Threshold**: 0.000312
- ✅ **No overfitting**: Validation loss < Training loss

---

### 🔴 PHASE 2: INFERENCE (Giai đoạn Suy diễn Real-time)

#### 📹 Input: Webcam
```
Webcam Input
├─ Live capture
├─ Variable resolution (720p, 1080p)
├─ RGB color
└─ Real-time stream
```

**Thách thức:**
- ⚠️ **Lighting variations**: Ánh sáng thay đổi liên tục
- ⚠️ **Head movements**: Đầu người dùng di chuyển
- ⚠️ **Distance changes**: Khoảng cách camera thay đổi

#### 👁️ Eye Detection (MediaPipe)
```
Webcam Frame
    ↓
[MediaPipe Face Mesh]
    ↓
Face Landmarks (468 points)
    ↓
Extract Eye Region
    ↓
Iris Image
```

**MediaPipe làm gì?**
- 🎯 **Face detection**: Phát hiện khuôn mặt
- 📍 **Landmark detection**: Xác định 468 điểm trên mặt
- 👁️ **Eye extraction**: Cắt vùng mắt từ frame
- ⚡ **Real-time**: Xử lý 30-60 FPS

#### 🔄 Preprocessing
```
Eye Image (variable size)
    ↓
[Same as training]
├─ Crop eyebrows
├─ Apply mask
└─ Resize to 128×128
    ↓
Standardized Image (128×128×3)
```

**Quan trọng:**
- ⚠️ **Must match training pipeline**: Phải giống y hệt với training
- 📏 **Same normalization**: Normalize pixel values [0, 1]
- 🎨 **Same color space**: RGB (nếu training dùng RGB)

#### 🤖 AutoEncoder Inference
```
Input: 128×128×3
    ↓
[Load trained model]
    ↓
[Encoder] Extract features
    ↓
Latent representation (8×8×256)
    ↓
[Decoder] Reconstruct
    ↓
Reconstructed Image (128×128×3)
```

**Model execution:**
- ⚡ **Latency**: 2.84ms (GPU) / 50ms (CPU)
- 🔢 **Batch size**: 1 (single image)
- 💾 **Memory**: ~100MB GPU memory

#### 📊 Calculate MSE & Compare Threshold
```
Original Image (X)
Reconstructed Image (X_recon)
    ↓
MSE = mean((X - X_recon)²)
    ↓
Compare with Threshold (0.000312)
```

**MSE Calculation:**
```python
MSE = (1 / (128×128×3)) × Σ(pixel_original - pixel_recon)²
    = (1 / 49,152) × Σ(differences²)
```

#### 🎯 Decision & Output

**Decision Logic:**
```
IF MSE < Threshold (0.000312):
    ✅ Classification: REAL (Valid Iris)
    📊 MSE: Low reconstruction error
    ✔️ Action: Grant access / Continue
    
ELSE (MSE ≥ Threshold):
    ❌ Classification: FAKE (Spoofed Iris)
    📊 MSE: High reconstruction error
    ⛔ Action: Deny access / Alert
```

**Output Format:**
```json
{
  "classification": "REAL" | "FAKE",
  "mse": 0.000154,
  "threshold": 0.000312,
  "confidence": 0.95,
  "latency_ms": 2.84
}
```

---

### 🔗 Mối quan hệ giữa 2 Phase

```
PHASE 1 (Training)         PHASE 2 (Inference)
      ↓                            ↑
[Learn from REAL] ───────> [Apply learned knowledge]
      ↓                            ↑
[Calculate Threshold] ────> [Use threshold to decide]
      ↓                            ↑
[Save model.pt] ──────────> [Load model.pt]
```

**Key Connection:**
- 📦 **Model transfer**: Model được train ở Phase 1 được dùng ở Phase 2
- 🎯 **Threshold transfer**: Ngưỡng tính từ validation set
- 🔄 **Preprocessing consistency**: Phải giống nhau 100%

---

### 🎨 Color Coding trong Diagram

| Màu | Ý nghĩa |
|-----|---------|
| **Trắng (boxes)** | Data/Process steps |
| **Nét đứt** | Data flow (training → inference) |
| **Nét liền** | Sequential process flow |
| **Xanh dương** | Training phase components |
| **Đỏ/Cam** | Inference phase components |

---

## 🔄 HÌNH 2.2: BIỂU ĐỒ LUỒNG DỮ LIỆU

### Tổng quan
Biểu đồ này mô tả chi tiết **quy trình xử lý từng bước** từ ảnh đầu vào đến quyết định cuối cùng.

---

### 📥 STEP 0: INPUT

```
⚫ Raw Iris Image
   ├─ Variable size (e.g., 640×480, 1920×1080)
   ├─ With eyebrows
   ├─ RGB format
   └─ May contain noise
```

**Đặc điểm ảnh đầu vào:**
- 📸 **Source**: Camera, webcam, hoặc file upload
- 🖼️ **Format**: JPEG, PNG
- 📏 **Size**: Không cố định, phụ thuộc thiết bị
- 🎨 **Quality**: Có thể có nhiễu, mờ, thiếu sáng

---

### 🔧 STEP 1: PREPROCESSING

```
Input: Raw Image (variable size)
    ↓
[1.1] Load and mask image
      - Create circular mask around iris
      - Set background pixels to 0
    ↓
[1.2] Crop eyebrows (top 1/3 = 0)
      - Remove eyebrow region
      - Zero out top third of image
    ↓
[1.3] Apply bitwise_and(image, mask)
      - Keep only iris region
      - Remove eyelids, sclera
    ↓
[1.4] Resize to 128×128
      - Interpolation: bilinear/bicubic
      - Maintain aspect ratio
    ↓
Output: Clean Image (128×128×3)
```

**Visualization of Preprocessing:**
```
Original (480×640)          After Mask            After Crop         Final (128×128)
┌─────────────┐            ┌─────────────┐      ┌─────────────┐    ┌────────┐
│  ┌───────┐  │            │  ┌───────┐  │      │     ███     │    │  ████  │
│ /│ ̄ ̄ ̄ ̄ ̄│\ │            │ /│ ̄ ̄ ̄ ̄ ̄│\ │      │    █████    │    │ ██████ │
│( │ ● ● │ )│    ───>      │( │ ● ● │ )│  ───> │   ███████   │───>│████████│
│ \│_____│/ │            │ \│_____│/ │      │   ███████   │    │████████│
│  └───────┘  │            │  └───────┘  │      │    █████    │    │ ██████ │
└─────────────┘            └─────────────┘      └─────────────┘    └────────┘
  Eyebrows                  Masked              Cropped            Resized
  included                  background          eyebrows           128×128
```

**Chi tiết từng bước:**

**1.1. Create Mask:**
```python
mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(mask, (center_x, center_y), radius, 255, -1)
# Result: Binary mask (1 = iris, 0 = background)
```

**1.2. Crop Eyebrows:**
```python
crop_height = height // 3
image[:crop_height, :] = 0  # Zero out top 1/3
# Removes eyebrow interference
```

**1.3. Apply Mask:**
```python
masked_image = cv2.bitwise_and(image, image, mask=mask)
# Keeps only iris pixels
```

**1.4. Resize:**
```python
resized = cv2.resize(masked_image, (128, 128), 
                     interpolation=cv2.INTER_LINEAR)
# Standardize to model input size
```

---

### 📊 STEP 2: NORMALIZE

```
Input: Clean Image (128×128×3)
      - Pixel values: 0-255 (uint8)
      - RGB channels
    ↓
[Normalization]
X = pixel_values / 255.0
    ↓
Output: Normalized Image (128×128×3)
       - Pixel values: 0.0-1.0 (float32)
       - Shape: (128, 128, 3)
```

**Tại sao cần normalize?**
- 🎯 **Scale consistency**: Neural networks hoạt động tốt với input [0, 1]
- ⚡ **Faster convergence**: Training nhanh hơn
- 📊 **Numerical stability**: Tránh overflow/underflow
- 🔄 **Match training**: Phải giống với training phase

**Before vs After:**
```
Before Normalization:
Pixel = [255, 127, 64]  (uint8)
       = [Red=max, Green=mid, Blue=low]

After Normalization:
Pixel = [1.0, 0.498, 0.251]  (float32)
       = [Red=max, Green=mid, Blue=low]
       ↑ Same relative values, different scale
```

---

### 🧠 STEP 3: AUTOENCODER FORWARD PASS

#### 🔽 ENCODER (Compression)

```
Input: 128×128×3 (49,152 values)
    ↓
Conv2d(32) + BN + ReLU
    ↓ Dimension: 64×64×32 (131,072 features)
    ↓ Compression: 2× spatial, 10.67× increase features
Conv2d(64) + BN + ReLU
    ↓ Dimension: 32×32×64 (65,536 features)
    ↓ Compression: 4× spatial total
Conv2d(128) + BN + ReLU
    ↓ Dimension: 16×16×128 (32,768 features)
    ↓ Compression: 8× spatial total
Conv2d(256) + BN + ReLU + Dropout(0.2)
    ↓ Dimension: 8×8×256 (16,384 features)
    ↓ Compression: 16× spatial total, ~48% compression
LATENT SPACE: 8×8×256
```

**Phân tích Encoder:**

**Layer 1: Conv2d(3→32)**
```
Input:  128×128×3  = 49,152 pixels
         ↓ [Conv 3×3, stride=2, padding=1]
Output: 64×64×32   = 131,072 features

Purpose: 
- Extract low-level features (edges, corners)
- Reduce spatial dimension by 2×
- Increase feature channels 3→32
```

**Layer 2: Conv2d(32→64)**
```
Input:  64×64×32   = 131,072 features
         ↓ [Conv 3×3, stride=2, padding=1]
Output: 32×32×64   = 65,536 features

Purpose:
- Extract mid-level features (textures, patterns)
- Further reduce spatial dimension
- Increase feature richness 32→64
```

**Layer 3: Conv2d(64→128)**
```
Input:  32×32×64   = 65,536 features
         ↓ [Conv 3×3, stride=2, padding=1]
Output: 16×16×128  = 32,768 features

Purpose:
- Extract high-level features (iris structures)
- Continue spatial compression
- Rich feature representation 64→128
```

**Layer 4: Conv2d(128→256)**
```
Input:  16×16×128  = 32,768 features
         ↓ [Conv 3×3, stride=2, padding=1]
         ↓ + Dropout(0.2) for regularization
Output: 8×8×256    = 16,384 features

Purpose:
- Extract highest-level features (iris identity)
- Maximum compression
- Most abstract representation
- Dropout prevents overfitting
```

**Latent Space Properties:**
```
Dimension: 8×8×256 = 16,384 features
Dropout: 0.2 (20% neurons dropped during training)
Compression Ratio: 
  Input:  49,152 values
  Latent: 16,384 values
  Ratio:  49,152 / 16,384 ≈ 3:1 (33% of original)
  Reduction: ~67% compression

Information Content:
- Contains ONLY essential iris features
- Removes redundant information
- Compact representation for comparison
```

#### 🔼 DECODER (Reconstruction)

```
LATENT SPACE: 8×8×256 (16,384 features)
    ↓
ConvTranspose2d(128) + BN + ReLU
    ↓ Dimension: 16×16×128 (32,768 features)
    ↓ Expansion: 2× spatial
ConvTranspose2d(64) + BN + ReLU
    ↓ Dimension: 32×32×64 (65,536 features)
    ↓ Expansion: 4× spatial total
ConvTranspose2d(32) + BN + ReLU
    ↓ Dimension: 64×64×32 (131,072 features)
    ↓ Expansion: 8× spatial total
ConvTranspose2d(3) + Sigmoid
    ↓ Dimension: 128×128×3 (49,152 pixels)
    ↓ Expansion: 16× spatial total
OUTPUT: Reconstructed Image (128×128×3)
```

**Phân tích Decoder:**

**Layer 1: ConvTranspose2d(256→128)**
```
Input:  8×8×256    = 16,384 features
         ↓ [ConvT 3×3, stride=2, padding=1, output_padding=1]
Output: 16×16×128  = 32,768 features

Purpose:
- Begin reconstruction
- Upsample spatial dimension 2×
- Reduce feature channels 256→128
```

**Layer 2: ConvTranspose2d(128→64)**
```
Input:  16×16×128  = 32,768 features
         ↓ [ConvT 3×3, stride=2, padding=1, output_padding=1]
Output: 32×32×64   = 65,536 features

Purpose:
- Continue upsampling
- Reconstruct mid-level features
- Reduce channels 128→64
```

**Layer 3: ConvTranspose2d(64→32)**
```
Input:  32×32×64   = 65,536 features
         ↓ [ConvT 3×3, stride=2, padding=1, output_padding=1]
Output: 64×64×32   = 131,072 features

Purpose:
- Further upsampling
- Reconstruct low-level features (textures)
- Reduce channels 64→32
```

**Layer 4: ConvTranspose2d(32→3) + Sigmoid**
```
Input:  64×64×32   = 131,072 features
         ↓ [ConvT 3×3, stride=2, padding=1, output_padding=1]
         ↓ + Sigmoid activation
Output: 128×128×3  = 49,152 pixels (RGB)

Purpose:
- Final reconstruction to original size
- Reduce to 3 RGB channels
- Sigmoid ensures output in [0, 1]
- Match input image format
```

**Sigmoid Activation:**
```python
output = 1 / (1 + exp(-x))
# Ensures all pixel values are in range [0, 1]
# Critical for image reconstruction
```

---

### 📐 STEP 4: RECONSTRUCTION ERROR

```
Input:
├─ X_original (128×128×3)     - Original normalized image
└─ X_recon (128×128×3)        - Reconstructed image
    ↓
[Compute MSE]
MSE = mean((X_original - X_recon)²)
    ↓
For each pixel:
  difference = original_pixel - reconstructed_pixel
  squared_diff = difference²
    ↓
MSE = sum(all squared_diff) / total_pixels
    = sum(all squared_diff) / (128 × 128 × 3)
    = sum(all squared_diff) / 49,152
    ↓
Output: Single MSE value (e.g., 0.000154)
```

**Chi tiết tính toán MSE:**

**Step-by-step:**
```python
# 1. Calculate pixel-wise difference
diff = X_original - X_recon
# Shape: (128, 128, 3)
# Values: can be positive or negative

# 2. Square each difference
squared_diff = diff ** 2
# Shape: (128, 128, 3)
# Values: all positive (removes sign)

# 3. Sum all squared differences
total_squared_error = np.sum(squared_diff)
# Single value: sum of 49,152 squared differences

# 4. Compute mean
MSE = total_squared_error / (128 * 128 * 3)
    = total_squared_error / 49,152
```

**Ví dụ cụ thể:**
```
Original pixel:      [0.8, 0.6, 0.4]  (RGB)
Reconstructed pixel: [0.7, 0.5, 0.3]  (RGB)

Differences:  [0.1, 0.1, 0.1]
Squared:      [0.01, 0.01, 0.01]
Sum:          0.03

Do this for all 49,152 pixels:
Total squared error = Σ(all squared differences)
MSE = Total / 49,152
```

**Ý nghĩa của MSE:**
- 📉 **MSE thấp** (< 0.000312): Tái tạo tốt → Có thể là REAL
  ```
  Original:      ████████
  Reconstructed: ████████  ← Very similar
  MSE = 0.000154
  ```

- 📈 **MSE cao** (≥ 0.000312): Tái tạo kém → Có thể là FAKE
  ```
  Original:      ████████
  Reconstructed: ██░░░░██  ← Different
  MSE = 0.000450
  ```

---

### ❓ DECISION: MSE < Threshold?

```
MSE value (computed)
    ↓
Compare with Threshold = 0.000312
    ↓
┌─────────────────────────────────┐
│  IF MSE < 0.000312:  → [YES]   │
│  ELSE:               → [NO]     │
└─────────────────────────────────┘
         ↓                    ↓
    [YES Branch]         [NO Branch]
```

**Decision Tree:**
```
                MSE < Threshold?
                      │
         ┌────────────┴────────────┐
         │                         │
       [YES]                     [NO]
    MSE < 0.000312          MSE ≥ 0.000312
         │                         │
    Reconstruction               Reconstruction
    is GOOD                      is POOR
         │                         │
      ✅ REAL                    ❌ FAKE
   (Valid Iris)               (Spoofed Iris)
```

**Ví dụ thực tế:**

**Case 1: REAL iris**
```
Input: Ảnh mống mắt thật từ webcam
  ↓ Preprocessing
  ↓ AutoEncoder Forward Pass
  ↓ MSE = 0.000154
  ↓ Compare: 0.000154 < 0.000312? → YES
  ↓
✅ Result: REAL (Valid Iris)
   Confidence: High (MSE chỉ bằng 49% threshold)
```

**Case 2: FAKE iris (printed photo)**
```
Input: Ảnh mống mắt in trên giấy
  ↓ Preprocessing
  ↓ AutoEncoder Forward Pass
  ↓ MSE = 0.000567
  ↓ Compare: 0.000567 < 0.000312? → NO
  ↓
❌ Result: FAKE (Spoofed Iris)
   Confidence: High (MSE gấp 1.8× threshold)
```

---

### 📤 FINAL OUTPUT

#### Output 1: REAL (Valid Iris) ✅
```json
{
  "status": "REAL",
  "description": "Valid Iris - Reconstruction successful",
  "mse": 0.000154,
  "threshold": 0.000312,
  "confidence": 0.95,
  "reconstruction_quality": "excellent",
  "action": "grant_access"
}
```

**Characteristics:**
- 📉 **Low MSE**: Significantly below threshold
- ✅ **Good reconstruction**: Original ≈ Reconstructed
- 🔓 **Action**: Allow authentication

#### Output 2: FAKE (Spoofed Iris) ❌
```json
{
  "status": "FAKE",
  "description": "Spoofed Iris - High reconstruction error",
  "mse": 0.000567,
  "threshold": 0.000312,
  "confidence": 0.88,
  "reconstruction_quality": "poor",
  "action": "deny_access",
  "alert": "Possible presentation attack detected"
}
```

**Characteristics:**
- 📈 **High MSE**: Above threshold
- ❌ **Poor reconstruction**: Original ≠ Reconstructed
- 🔒 **Action**: Deny authentication, trigger alert

---

### 📊 Performance Metrics

**Latency Breakdown:**
```
Total Latency: 2.84ms (GPU) / 50ms (CPU)

├─ Step 1 (Preprocessing):     0.5ms  (18%)
├─ Step 2 (Normalization):     0.1ms  (4%)
├─ Step 3 (AutoEncoder):       2.0ms  (70%)  ← Bottleneck
└─ Step 4 (MSE + Decision):    0.24ms (8%)
```

**Throughput:**
```
GPU: 352 FPS (1000ms / 2.84ms)
CPU: ~25 FPS (1000ms / 50ms)
```

---

## 🏗️ HÌNH 2.3: KIẾN TRÚC AUTOENCODER CHI TIẾT

### Tổng quan
Hình này mô tả chi tiết **cấu trúc bên trong** của mô hình AutoEncoder, bao gồm từng layer và tham số cụ thể.

---

### 📥 INPUT LAYER

```
Input Shape: (Batch, 3, 128, 128)
             ↑     ↑   ↑    ↑
             │     │   │    └─ Width (pixels)
             │     │   └────── Height (pixels)
             │     └────────── Channels (RGB)
             └──────────────── Batch size

Example: (32, 3, 128, 128)
         = 32 images, each 3 channels, 128×128 pixels
         = 32 × 49,152 = 1,572,864 values total
```

**PyTorch Format:**
- 🔢 **Channel-first**: (N, C, H, W)
- 📦 **Batch processing**: Multiple images at once
- 💾 **Memory**: ~6 MB per batch (32 images, float32)

---

### 🔽 ENCODER (Compression) - Detailed

#### Layer 1: Conv2d(3 → 32)

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 3, 128, 128)             │
│                                          │
│ Conv2d Configuration:                    │
│ ├─ in_channels: 3                       │
│ ├─ out_channels: 32                     │
│ ├─ kernel_size: 3×3                     │
│ ├─ stride: 2                            │
│ ├─ padding: 1                           │
│                                          │
│ BatchNorm2d(32)                         │
│ ReLU Activation                         │
│                                          │
│ OUTPUT: (Batch, 32, 64, 64)            │
└─────────────────────────────────────────┘
```

**Parameters Calculation:**
```
Conv2d weights: (out_ch × in_ch × kernel_h × kernel_w)
              = 32 × 3 × 3 × 3
              = 864 weights

Conv2d bias:    32 (one per output channel)

BatchNorm:      32 × 2 = 64 (gamma + beta)

Total: 864 + 32 + 64 = 960 parameters
```

**Spatial Dimension Calculation:**
```
Output_size = (Input_size + 2×padding - kernel_size) / stride + 1

Height: (128 + 2×1 - 3) / 2 + 1 = 127 / 2 + 1 = 64
Width:  (128 + 2×1 - 3) / 2 + 1 = 127 / 2 + 1 = 64

Result: 64×64 feature maps
```

**What does this layer learn?**
- 🎨 **Low-level features**: Edges, corners, gradients
- 📊 **Color transitions**: RGB channel interactions
- 🔍 **Local patterns**: Small textures (3×3 receptive field)

#### Layer 2: Conv2d(32 → 64)

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 32, 64, 64)              │
│                                          │
│ Conv2d(32→64, kernel=3×3, stride=2)     │
│ BatchNorm2d(64)                         │
│ ReLU                                     │
│                                          │
│ OUTPUT: (Batch, 64, 32, 32)            │
└─────────────────────────────────────────┘
```

**Parameters:**
```
Conv2d: 64 × 32 × 3 × 3 = 18,432
Bias:   64
BN:     64 × 2 = 128
Total:  18,624 parameters
```

**Receptive Field:**
```
Layer 1: 3×3 pixels
Layer 2: 3×3 on 64×64 = 7×7 on original 128×128
```

**What does this layer learn?**
- 🖼️ **Mid-level features**: Textures, small patterns
- 🔄 **Feature combinations**: Combining edge features
- 📐 **Iris structures**: Radial patterns, furrows

#### Layer 3: Conv2d(64 → 128)

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 64, 32, 32)              │
│                                          │
│ Conv2d(64→128, kernel=3×3, stride=2)    │
│ BatchNorm2d(128)                        │
│ ReLU                                     │
│                                          │
│ OUTPUT: (Batch, 128, 16, 16)           │
└─────────────────────────────────────────┘
```

**Parameters:**
```
Conv2d: 128 × 64 × 3 × 3 = 73,728
Bias:   128
BN:     128 × 2 = 256
Total:  74,112 parameters
```

**Receptive Field:**
```
Layer 3: 15×15 on original 128×128
```

**What does this layer learn?**
- 🎯 **High-level features**: Iris collarette, crypts
- 🔍 **Complex patterns**: Multiple texture combinations
- 📊 **Iris-specific structures**: Unique identification features

#### Layer 4: Conv2d(128 → 256) + Dropout

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 128, 16, 16)             │
│                                          │
│ Conv2d(128→256, kernel=3×3, stride=2)   │
│ BatchNorm2d(256)                        │
│ ReLU                                     │
│ Dropout2d(p=0.2)  ← Regularization     │
│                                          │
│ OUTPUT: (Batch, 256, 8, 8)             │
└─────────────────────────────────────────┘
```

**Parameters:**
```
Conv2d: 256 × 128 × 3 × 3 = 294,912
Bias:   256
BN:     256 × 2 = 512
Dropout: 0 (no parameters, just masking)
Total:  295,680 parameters
```

**Dropout Effect:**
```
Training:
  20% of feature maps randomly dropped
  Remaining 80% scaled by 1.25× (to maintain expected output)

Inference:
  No dropout (all features active)
  Prevents overfitting during training
```

**Receptive Field:**
```
Layer 4: 31×31 on original 128×128
         Covers ~24% of input image
```

---

### 🎯 LATENT SPACE (Bottleneck)

```
┌─────────────────────────────────────────┐
│           LATENT SPACE                   │
│                                          │
│ Dimension: 8×8×256 = 16,384 features   │
│                                          │
│ Dropout: 0.2 (training only)            │
│                                          │
│ Compression Ratio: 49,152 → 16,384     │
│                    (~67% reduction)     │
│                                          │
│ Information Content:                     │
│ - Essential iris features only          │
│ - Removes redundancy                    │
│ - Compact representation                │
│ - Enables anomaly detection             │
│                                          │
│ Visualization:                          │
│    8×8 spatial × 256 channels           │
│    = 64 spatial positions               │
│    = Each position has 256 features     │
└─────────────────────────────────────────┘
```

**Why 8×8×256 is important:**

**1. Compression:**
```
Original: 128×128×3 = 49,152 values
Latent:   8×8×256   = 16,384 values
Ratio:    49,152 / 16,384 = 3:1
```

**2. Information Bottleneck:**
```
Forces model to learn ONLY essential features
├─ REAL iris: Can be compressed and reconstructed well
└─ FAKE iris: Cannot be compressed effectively (loses info)
```

**3. Feature Distribution:**
```
8×8 grid = 64 spatial locations
Each location: 256-dimensional feature vector

Example feature map:
┌─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │ │ │ │ │  Each cell: 256 features
├─┼─┼─┼─┼─┼─┼─┼─┤  Represents 16×16 region
│ │ │●│●│●│●│ │ │  of original image
├─┼─┼─┼─┼─┼─┼─┼─┤  ● = High activation
│ │●│●│●│●│●│●│ │    (important features)
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │●│●│●│●│●│●│ │  Captures:
├─┼─┼─┼─┼─┼─┼─┼─┤  - Iris patterns
│ │●│●│●│●│●│●│ │  - Texture density
├─┼─┼─┼─┼─┼─┼─┼─┤  - Color distribution
│ │●│●│●│●│●│●│ │  - Structural features
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │●│●│●│●│ │ │
├─┼─┼─┼─┼─┼─┼─┼─┤
│ │ │ │ │ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┘
```

---

### 🔼 DECODER (Reconstruction) - Detailed

#### Layer 1: ConvTranspose2d(256 → 128)

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 256, 8, 8)               │
│                                          │
│ ConvTranspose2d Configuration:          │
│ ├─ in_channels: 256                     │
│ ├─ out_channels: 128                    │
│ ├─ kernel_size: 3×3                     │
│ ├─ stride: 2                            │
│ ├─ padding: 1                           │
│ └─ output_padding: 1                    │
│                                          │
│ BatchNorm2d(128)                        │
│ ReLU                                     │
│                                          │
│ OUTPUT: (Batch, 128, 16, 16)           │
└─────────────────────────────────────────┘
```

**ConvTranspose (Deconvolution) Explained:**
```
Regular Conv:     Downsample (e.g., 8×8 → 4×4)
ConvTranspose:    Upsample (e.g., 8×8 → 16×16)

Process:
1. Insert zeros between input pixels (stride=2)
2. Apply convolution
3. Remove padding
4. Result: 2× spatial increase
```

**Parameters:**
```
ConvT: 128 × 256 × 3 × 3 = 294,912
Bias:  128
BN:    128 × 2 = 256
Total: 295,296 parameters
```

**Output size calculation:**
```
Output_size = (Input_size - 1) × stride - 2×padding + kernel + output_padding

Height: (8 - 1) × 2 - 2×1 + 3 + 1 = 16
Width:  (8 - 1) × 2 - 2×1 + 3 + 1 = 16
```

#### Layer 2: ConvTranspose2d(128 → 64)

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 128, 16, 16)             │
│                                          │
│ ConvTranspose2d(128→64, 3×3, stride=2)  │
│ BatchNorm2d(64)                         │
│ ReLU                                     │
│                                          │
│ OUTPUT: (Batch, 64, 32, 32)            │
└─────────────────────────────────────────┘
```

**Parameters:**
```
ConvT: 64 × 128 × 3 × 3 = 73,728
Bias:  64
BN:    64 × 2 = 128
Total: 73,920 parameters
```

#### Layer 3: ConvTranspose2d(64 → 32)

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 64, 32, 32)              │
│                                          │
│ ConvTranspose2d(64→32, 3×3, stride=2)   │
│ BatchNorm2d(32)                         │
│ ReLU                                     │
│                                          │
│ OUTPUT: (Batch, 32, 64, 64)            │
└─────────────────────────────────────────┘
```

**Parameters:**
```
ConvT: 32 × 64 × 3 × 3 = 18,432
Bias:  32
BN:    32 × 2 = 64
Total: 18,528 parameters
```

#### Layer 4: ConvTranspose2d(32 → 3) + Sigmoid

```
┌─────────────────────────────────────────┐
│ INPUT: (Batch, 32, 64, 64)              │
│                                          │
│ ConvTranspose2d(32→3, 3×3, stride=2)    │
│ Sigmoid Activation  ← IMPORTANT         │
│                                          │
│ OUTPUT: (Batch, 3, 128, 128)           │
│         Values in range [0, 1]          │
└─────────────────────────────────────────┘
```

**Parameters:**
```
ConvT: 3 × 32 × 3 × 3 = 864
Bias:  3
Total: 867 parameters
```

**Sigmoid Activation:**
```python
sigmoid(x) = 1 / (1 + exp(-x))

Properties:
- Input: any real number (-∞ to +∞)
- Output: [0, 1]
- Smooth, differentiable
- Perfect for image pixels (normalized)

Example:
  x = 2.0  → sigmoid(2.0)  = 0.88
  x = 0.0  → sigmoid(0.0)  = 0.50
  x = -2.0 → sigmoid(-2.0) = 0.12
```

---

### 📊 MODEL SUMMARY

```
┌─────────────────────────────────────────┐
│         MODEL SUMMARY                    │
├─────────────────────────────────────────┤
│ Total Parameters: ~2.5M                  │
│ ├─ Encoder: ~463K                       │
│ ├─ Decoder: ~388K                       │
│ └─ Total Trainable: ~777K (~0.78M)     │
│                                          │
│ Input Shape: (Batch, 3, 128, 128)       │
│ Output Shape: (Batch, 3, 128, 128)      │
│                                          │
│ Output Range: [0, 1] via Sigmoid        │
│                                          │
│ Memory Footprint:                        │
│ ├─ Model weights: ~10 MB (float32)     │
│ ├─ Activation maps: ~50 MB (batch=32)  │
│ └─ Total GPU memory: ~100 MB           │
└─────────────────────────────────────────┘
```

**Detailed Parameter Count:**
```
ENCODER:
├─ Conv2d(3→32):     960
├─ Conv2d(32→64):    18,624
├─ Conv2d(64→128):   74,112
└─ Conv2d(128→256):  295,680
   Subtotal:         389,376

DECODER:
├─ ConvT2d(256→128): 295,296
├─ ConvT2d(128→64):  73,920
├─ ConvT2d(64→32):   18,528
└─ ConvT2d(32→3):    867
   Subtotal:         388,611

TOTAL: 389,376 + 388,611 = 777,987 ≈ 0.78M
```

---

### 🎨 ConvTranspose2d Visualization

**How upsampling works:**

```
Input: 2×2        Stride=2        Output: 4×4
┌─┬─┐            Insert zeros      ┌─┬─┬─┬─┐
│1│2│    ───>    between pixels    │a│b│c│d│
├─┼─┤                          ───>│e│f│g│h│
│3│4│            Apply 3×3 kernel  │i│j│k│l│
└─┴─┘                               │m│n│o│p│
                                    └─┴─┴─┴─┘

Process:
[1, 2]      [1, 0, 2, 0]       [a, b, c, d]
[3, 4]  →   [0, 0, 0, 0]   →   [e, f, g, h]
            [3, 0, 4, 0]       [i, j, k, l]
            [0, 0, 0, 0]       [m, n, o, p]
```

---

### ⚖️ BatchNorm2d Explained

**What is Batch Normalization?**

```
For each channel (feature map):
1. Calculate mean (μ) and std (σ) across batch
2. Normalize: x_norm = (x - μ) / (σ + ε)
3. Scale and shift: y = γ × x_norm + β
   (γ and β are learnable parameters)
```

**Benefits:**
- ⚡ **Faster training**: Normalizes activations
- 📊 **Stable gradients**: Prevents vanishing/exploding
- 🎯 **Higher learning rates**: Can train faster
- 🔄 **Regularization**: Slight regularization effect

**Example:**
```
Input feature map (8×8):
[[0.1, 0.9, 0.3, ...],   μ = 0.5, σ = 0.2
 [0.7, 0.2, 0.8, ...],   
 ...]

After BN:
[[−2.0, 2.0, −1.0, ...],  ← Normalized
 [1.0, −1.5, 1.5, ...],   ← Mean=0, Std=1
 ...]

After scale/shift (γ=0.5, β=0.1):
[[−0.9, 1.1, −0.4, ...],  ← γ×norm + β
 [0.6, −0.65, 0.85, ...],
 ...]
```

---

### 🎯 Dropout2d Explained

**What is Dropout?**

```
Training:
  Randomly drop 20% of feature maps
  Remaining 80% scaled by 1.25×

Input: (Batch, 256, 8, 8)
       256 feature maps
  ↓
Dropout(p=0.2):
  Keep:  80% × 256 = ~205 feature maps
  Drop:  20% × 256 = ~51 feature maps
  Scale: × 1.25 (to maintain expected output)
  ↓
Output: (Batch, 256, 8, 8)
        Same shape, but 20% channels zeroed

Inference (testing):
  No dropout (all features active)
```

**Why Dropout?**
- 🛡️ **Prevents overfitting**: Model can't rely on specific features
- 🔄 **Ensemble effect**: Like training multiple models
- 💪 **Robust features**: Forces learning of diverse features

**Visualization:**
```
Without Dropout:
Feature Maps: [✓][✓][✓][✓][✓][✓][✓][✓]
              All features always active
              → May overfit

With Dropout (training):
Feature Maps: [✓][✗][✓][✓][✗][✓][✗][✓]
              Random 20% dropped
              → Forces redundancy
              → Better generalization
```

---

### 🔑 KEY INSIGHTS

#### 1. Symmetry between Encoder and Decoder
```
Encoder:     3 → 32 → 64 → 128 → 256
Decoder:   256 → 128 → 64 → 32 → 3
             ↑                    ↑
          Mirror structure
```

#### 2. Progressive Compression/Expansion
```
Spatial Dimensions:
128×128 → 64×64 → 32×32 → 16×16 → 8×8  (Encoder)
  8×8 → 16×16 → 32×32 → 64×64 → 128×128 (Decoder)

Feature Channels:
3 → 32 → 64 → 128 → 256  (Encoder: Increase features)
256 → 128 → 64 → 32 → 3  (Decoder: Decrease features)
```

#### 3. Information Flow
```
Input Image (49,152 pixels)
      ↓ Encoder compresses
Latent Space (16,384 features)  ← Bottleneck
      ↓ Decoder expands
Output Image (49,152 pixels)

Information loss happens at bottleneck:
- REAL iris: Minimal loss (essential features retained)
- FAKE iris: Significant loss (can't compress unfamiliar patterns)
```

#### 4. Why This Works for Anomaly Detection
```
REAL Iris:
  Input → [Compress well] → Latent → [Reconstruct well] → Output
  MSE between Input and Output: LOW

FAKE Iris:
  Input → [Compress poorly] → Latent → [Reconstruct poorly] → Output
  MSE between Input and Output: HIGH

Threshold separates these two cases!
```

---

## 📊 TỔNG KẾT 3 HÌNH

### So sánh 3 Perspectives

| Khía cạnh | Hình 2.1 | Hình 2.2 | Hình 2.3 |
|-----------|----------|----------|----------|
| **Góc nhìn** | System-level (toàn hệ thống) | Process-level (quy trình) | Architecture-level (kiến trúc) |
| **Chi tiết** | High-level overview | Step-by-step flow | Layer-by-layer structure |
| **Mục đích** | Hiểu tổng quan workflow | Hiểu quy trình xử lý | Hiểu cấu trúc mô hình |
| **Độc giả** | Project managers, stakeholders | Developers, researchers | ML engineers, researchers |

### Information Flow across 3 Diagrams

```
HÌNH 2.1 (System View):
Training Phase → Trained Model → Inference Phase
                      ↓
HÌNH 2.2 (Process View):
Input → Preprocess → Normalize → AutoEncoder → MSE → Decision
                                       ↓
HÌNH 2.3 (Architecture View):
Encoder (4 layers) → Latent Space → Decoder (4 layers)
```

### Key Concepts Unified

1. **Preprocessing Consistency**
   - Hình 2.1: Mentioned in both phases
   - Hình 2.2: Detailed in Step 1
   - Hình 2.3: Defines input requirements

2. **AutoEncoder Core**
   - Hình 2.1: Black box "AutoEncoder Model"
   - Hình 2.2: Shows forward pass
   - Hình 2.3: Reveals internal structure

3. **Decision Mechanism**
   - Hình 2.1: "Calculate MSE & Compare Threshold"
   - Hình 2.2: "MSE < Threshold?" decision point
   - Hình 2.3: Outputs reconstruction for comparison

---

## 🧭 HÌNH 2.4: FLOWCHART THUẬT TOÁN PHÁT HIỆN LIVENESS

### Ý nghĩa tổng quan
Hình 2.4 là bản “tóm tắt thuật toán” của hệ thống ở chế độ chạy thật: từ lúc **nạp mô hình**, **lấy ảnh**, **tách vùng mắt**, **tiền xử lý**, **tái tạo bằng AutoEncoder**, tính **lỗi tái tạo (MSE)**, sau đó so sánh với **ngưỡng** để kết luận **REAL/FAKE**.

### Diễn giải từng khối trong flowchart

1) **Load Trained Model (`autoencoder_processed_clean.pt`)**
- Nạp trọng số đã huấn luyện (encoder + decoder) và các tham số cần thiết.
- Đây là bước “khởi tạo hệ thống”; sau khi load xong, mỗi frame chỉ cần inference.

2) **Capture Iris Image (Webcam or Upload)**
- Nguồn ảnh có thể là luồng webcam (real-time) hoặc ảnh tải lên (demo).
- Chất lượng và “domain” của ảnh nguồn ảnh hưởng mạnh đến phân bố MSE.

3) **Detect Eye Region (MediaPipe FaceMesh)**
- MediaPipe phát hiện landmark khuôn mặt và suy ra vùng mắt.
- **Nút rẽ nhánh “Eye detected?”** thể hiện tính thực tế của hệ thống: có thể có frame không bắt được mắt do quay mặt, nháy mắt, thiếu sáng.

4) **Nhánh NO: Error → Retry**
- Nếu không phát hiện mắt: hệ thống báo lỗi “No eye detected”.
- Sau đó **Retry**: lấy lại ảnh/frame và chạy lại pipeline.

5) **Nhánh YES: Preprocessing (giống training)**
- Bao gồm: crop eyebrows, apply mask, resize 128×128, normalize về [0, 1].
- Điểm quan trọng nhất của AutoEncoder-based PAD là: **pipeline tiền xử lý khi chạy thật phải giống pipeline lúc huấn luyện**. Nếu khác (ví dụ normalize khác, mask khác), MSE sẽ lệch và ngưỡng mất hiệu lực.

6) **AutoEncoder Forward Pass**
- Đưa ảnh đã chuẩn hoá vào AutoEncoder để tái tạo: $X_{recon} = AE(X)$.

7) **Calculate Reconstruction Error (MSE)**
- Tính: $\text{MSE} = \text{mean}((X - X_{recon})^2)$.
- Đây là “điểm bất thường” (anomaly score):
  - REAL thường có MSE thấp (tái tạo tốt)
  - FAKE thường có MSE cao (tái tạo kém)

8) **Decision: MSE < Threshold?**
- Nếu **YES** → **Result = REAL (Valid Iris)**
- Nếu **NO** → **Result = FAKE (Spoofed Iris)**

### Mapping flowchart ↔ triển khai thực tế
- Trong dự án, luồng này tương ứng với real-time implementation (ví dụ: `main_realtime_new.py`), nơi mỗi frame chạy: detect → preprocess → inference → compute MSE → compare threshold → hiển thị kết quả.

---

## 📉 HÌNH 3.1: BIỂU ĐỒ LOSS CURVE CỦA MÔ HÌNH AUTOENCODER THEO SỐ EPOCH

### Biểu đồ thể hiện gì?
- Trục $x$: số epoch (số lần mô hình “đi qua” toàn bộ tập train).
- Trục $y$: loss (ở đây là **MSE**) — càng nhỏ càng tốt.
- Thường có 2 đường:
  - **Training loss**: lỗi tái tạo trên tập train
  - **Validation loss**: lỗi tái tạo trên tập validation (REAL)

### Cách đọc loss curve đúng trong AutoEncoder

1) **Giảm nhanh giai đoạn đầu**
- Thể hiện mô hình học được các cấu trúc cơ bản của ảnh mống mắt (biên, pattern thô).

2) **Giảm chậm và dần ổn định về sau**
- Thể hiện quá trình “tinh chỉnh”: mô hình cải thiện chi tiết nhỏ, giảm sai số dần.

3) **Training loss và Validation loss bám sát nhau**
- Là dấu hiệu tốt: mô hình không học vẹt (overfitting) một cách rõ rệt.
- Nếu validation loss tăng trong khi training loss vẫn giảm → thường là overfitting.

### Liên hệ với các số liệu trong báo cáo
- Training loss (initial): **0.135653**
- Training loss (final): **0.000215**
- Validation loss (best): **0.000158**
- Early stopping: **Not triggered** (tức validation loss vẫn cải thiện hoặc không “xấu đi” đủ lâu để dừng sớm).

### Ý nghĩa thực tế cho PAD
Loss curve “đẹp” (giảm đều, không vọt tăng) giúp đảm bảo rằng **MSE trên ảnh REAL** ổn định và có thể dùng để:
- ước lượng phân bố MSE (mean/std/percentile)
- tính ngưỡng (threshold) cho anomaly detection

---

## 🧩 HÌNH 2.5: SƠ ĐỒ TRIỂN KHAI (DEPLOYMENT DIAGRAM)

### Mục tiêu của sơ đồ
Hình 2.5 mô tả hệ thống ở góc nhìn “triển khai”: các khối chạy ở đâu, tương tác dữ liệu như thế nào, và artifact nào được dùng lại giữa training và inference.

### Các khối chính trong sơ đồ

1) **Training Pipeline**
- **Data Preprocessing**: chuẩn hoá dữ liệu (crop/mask/resize/normalize).
- **AutoEncoder Training**: huấn luyện mô hình với optimizer AdamW, loss MSE.
- **Model Evaluation**: đánh giá loss/độ ổn định, tính toán thống kê MSE cho threshold.

2) **Development Environment (Google Colab + Google Drive)**
- Dataset UBIPR2, model đã train và báo cáo (reports) được lưu trên **Google Drive**.
- Google Colab sử dụng GPU (Tesla T4) để train/infer nhanh.
- “Drive” đóng vai trò kho lưu trữ trung tâm:
  - **Dataset UBIPR2** (nguồn dữ liệu)
  - **Trained Models** (artifact phục vụ inference)
  - **Reports** (loss curve, thống kê MSE, confusion matrix/ROC ở demo)

3) **Inference System (Real-time)**
- Input từ **Webcam**.
- Khối **Real-time Detector** gồm 3 thành phần chính:
  - **MediaPipe**: detect mặt/mắt
  - **OpenCV**: xử lý ảnh (crop/resize/mask)
  - **PyTorch Model**: inference AutoEncoder

4) **User Interface**
- Chỉ làm nhiệm vụ hiển thị: “REAL/FAKE”, MSE, FPS/latency (tuỳ cách implement).

### Luồng artifact quan trọng
1) Train xong → lưu `*.pt` model lên Drive.
2) Inference system (real-time) → nạp `*.pt` model → chạy trên webcam.
3) Threshold thường được “auto-computed” từ thống kê MSE của ảnh REAL; nếu thay đổi môi trường (webcam khác, ánh sáng khác) có thể cần hiệu chỉnh (calibration).

---

## 🧪 HÌNH 3.3: ĐÁNH GIÁ PHÂN LOẠI (CONFUSION MATRIX, ROC, HISTOGRAM MSE, METRICS)

### 4 thành phần trong hình và cách đọc

#### (1) Confusion Matrix
- Ma trận cho thấy hệ thống **dự đoán toàn bộ là FAKE**.
- Cụ thể với demo n=10 (REAL=5, FAKE=5):
  - FAKE dự đoán đúng: 5
  - REAL bị đoán nhầm thành FAKE: 5
  - Không có mẫu nào được đoán REAL.

=> Hệ quả:
- **Accuracy = 50%** (đúng hết 5 FAKE, sai hết 5 REAL)
- **Precision/Recall/F1 cho lớp REAL** (hoặc theo định nghĩa positive bạn chọn) bị sụp về **0** vì không có dự đoán REAL.

#### (2) ROC Curve (AUC)
- Đường ROC cho thấy **AUC = 1.0**.
- Ý nghĩa của AUC=1.0: nếu bạn thay đổi ngưỡng một cách phù hợp, **điểm số (MSE)** có thể tách 2 nhóm rất tốt trong tập demo.

#### (3) Histogram MSE (REAL vs FAKE) + Threshold
- Biểu đồ histogram thể hiện phân bố MSE của 2 nhóm (REAL và FAKE).
- Đường threshold (0.000312) nằm rất lệch về bên trái (rất nhỏ) so với các cột histogram trong demo.

=> Điều này giải thích vì sao “dự đoán tất cả là FAKE”:
- Nếu toàn bộ MSE của cả REAL và FAKE trong ảnh upload đều **lớn hơn** 0.000312, thì quy tắc $\text{MSE} < \text{threshold}$ không bao giờ đúng → không thể ra REAL.

#### (4) Metrics Summary
- **Accuracy ~ 0.5** vì dataset cân bằng (5/5) và mô hình đoán hết về một phía.
- **AUC = 1.0** vẫn có thể xảy ra đồng thời vì AUC đo “khả năng xếp hạng” (ranking) khi quét ngưỡng, không phụ thuộc vào một ngưỡng cố định.

### Vì sao có nghịch lý “AUC rất cao nhưng metrics rất tệ”?
1) **Threshold mismatch (lệch ngưỡng)**
- Ngưỡng 0.000312 được tính từ validation REAL (UBIPR2 NIR, điều kiện chuẩn).
- Ảnh upload thường là webcam RGB, điều kiện đa dạng → MSE bị “dịch” lên cao.

2) **Domain gap (khác miền dữ liệu)**
- Training: NIR/controlled.
- Demo: RGB/uncontrolled.
- AutoEncoder rất nhạy với pipeline/thiết bị, nên distribution MSE thay đổi mạnh.

3) **Dataset quá nhỏ (n=10)**
- Không có ý nghĩa thống kê; AUC=1.0 ở tập nhỏ có thể “đẹp” nhưng không bền.

### Gợi ý cách diễn giải trong báo cáo
- Nhấn mạnh: Hình 3.3 là **minh hoạ** cơ chế quyết định và “xu hướng tách lớp” của MSE.
- Không dùng trực tiếp các metric ở demo để kết luận hiệu năng chung.
- Nếu muốn đánh giá đúng, cần:
  - calibration threshold theo môi trường triển khai
  - tập test lớn hơn và FAKE đa dạng (print/screen/contact lens…)

---

## 💡 PRACTICAL INSIGHTS

### Cho Developers:
- 📝 **Implementation order**: Follow Hình 2.2 (top to bottom)
- 🔧 **Debugging**: Use Hình 2.3 to inspect layer outputs
- 🎯 **Optimization**: Focus on Step 3 in Hình 2.2 (70% latency)

### Cho Researchers:
- 📊 **Experimental design**: Modify architecture in Hình 2.3
- 🔬 **Ablation studies**: Remove components and measure impact
- 📈 **Improvements**: Consider VAE, attention mechanisms

### Cho Stakeholders:
- 💰 **Cost**: GPU vs CPU trade-offs (Hình 2.1)
- ⏱️ **Performance**: Real-time capability (Hình 2.2)
- 🔒 **Security**: Anomaly detection approach (all 3 diagrams)

---

**📅 Tài liệu được tạo bởi GitHub Copilot**  
**🔗 Nguồn: Kiến trúc hệ thống phát hiện liveness mống mắt**
