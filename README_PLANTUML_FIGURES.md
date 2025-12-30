# README - PlantUML Figures for Chapter 2

## 📚 Hướng Dẫn Sử Dụng Các Hình Minh Họa

Tài liệu này giải thích chi tiết 6 hình minh họa cho **Chương 2: Mô hình đề xuất** trong luận văn về **Phát hiện Liveness mống mắt bằng AutoEncoder**.

---

## 🎨 Danh Sách Các Hình

| STT | Tên File | Mô Tả | Loại |
|-----|----------|-------|------|
| 2.1 | `fig2_1_system_architecture.puml` | Kiến trúc tổng thể hệ thống | PlantUML |
| 2.2 | `fig2_2_data_flow_diagram.puml` | Biểu đồ luồng dữ liệu | PlantUML |
| 2.3 | `fig2_3_autoencoder_architecture.puml` | Kiến trúc AutoEncoder chi tiết | PlantUML |
| 2.4 | `fig2_4_flowchart_algorithm.puml` | Flowchart thuật toán | PlantUML |
| 2.5 | `fig2_5_deployment_diagram.puml` | Sơ đồ triển khai | PlantUML |
| 2.6 | `fig2_6_mathematical_formulas.tex` | Công thức toán học | LaTeX |

---

## 📊 Chi Tiết Từng Hình

### **Hình 2.1: Kiến Trúc Tổng Thể Hệ Thống**
**File:** `fig2_1_system_architecture.puml`

**Mục đích:** Minh họa kiến trúc tổng thể gồm 2 giai đoạn (PHASE 1: Training, PHASE 2: Inference)

**Nội dung chính:**
- **PHASE 1 - Training:**
  - Dataset UBIPR2 (chỉ REAL iris)
  - Preprocessing: Crop eyebrows, apply mask, resize 128×128
  - AutoEncoder Model (2.5M parameters)
  - Trained Model (.pt file)

- **PHASE 2 - Inference (Real-time):**
  - Webcam Input (live capture)
  - Eye Detection (MediaPipe FaceMesh)
  - Preprocessing (giống training)
  - AutoEncoder Inference
  - Calculate MSE & Compare Threshold
  - Result: REAL (low MSE) hoặc FAKE (high MSE)

- **Threshold Formula:**
  ```
  τ = μ_real + 2 × σ_real
  ```

**Kỹ thuật:**
- Package diagram với stereotype (<<training>>, <<inference>>, <<data>>, <<model>>, <<decision>>)
- Monochrome color scheme (in trắng đen)
- Dashed line cho model transfer

---

### **Hình 2.2: Biểu Đồ Luồng Dữ Liệu**
**File:** `fig2_2_data_flow_diagram.puml`

**Mục đích:** Minh họa luồng xử lý từ input đến output (activity diagram)

**Nội dung chính:**
1. **INPUT:** Raw Iris Image (variable size, with eyebrows)
2. **STEP 1 - Preprocessing:**
   - Load mask image
   - `mask[0:h/3] = 0` (crop eyebrows)
   - `bitwise_and(image, mask)`
   - Resize to 128×128
   - Output shape: (128, 128, 3)

3. **STEP 2 - Normalize:**
   - `X = X / 255.0` (normalize to [0, 1])

4. **STEP 3 - AutoEncoder Forward Pass:**
   - **Encoder:** 128×128×3 → 64×64×32 → 32×32×64 → 16×16×128 → 8×8×256 (Latent)
   - **Decoder:** 8×8×256 → 16×16×128 → 32×32×64 → 64×64×32 → 128×128×3
   - Latent Space: 8×8×256 with Dropout 0.2

5. **OUTPUT:** Reconstructed Image X_recon (128×128×3)

6. **STEP 4 - Calculate MSE:**
   - `MSE = mean((X_original - X_recon)²)`

7. **Decision:**
   - If MSE < Threshold → **REAL** (Valid)
   - Else → **FAKE** (Spoofed)

**Kỹ thuật:**
- Activity diagram với if-then-else
- Note boxes cho chi tiết technical
- Grayscale colors

---

### **Hình 2.3: Kiến Trúc AutoEncoder Chi Tiết**
**File:** `fig2_3_autoencoder_architecture.puml`

**Mục đích:** Minh họa chi tiết từng layer của AutoEncoder

**Nội dung chính:**

**ENCODER (Compression):**
- Input: 128×128×3
- Conv2d(32): 64×64×32 + BatchNorm + ReLU
- Conv2d(64): 32×32×64 + BatchNorm + ReLU
- Conv2d(128): 16×16×128 + BatchNorm + ReLU
- Conv2d(256): 8×8×256 + BatchNorm + ReLU + Dropout(0.2)

**LATENT SPACE (Bottleneck):**
- Dimension: 8×8×256 = 16,384
- Dropout: 0.2
- Compression ratio: ~48× (49,152 → 16,384 → 49,152)

**DECODER (Reconstruction):**
- ConvTranspose2d(128): 16×16×128 + BatchNorm + ReLU
- ConvTranspose2d(64): 32×32×64 + BatchNorm + ReLU
- ConvTranspose2d(32): 64×64×32 + BatchNorm + ReLU
- ConvTranspose2d(3): 128×128×3 + Sigmoid
- Output: 128×128×3

**Model Summary:**
- Total Parameters: ~2.5M
- Input shape: (batch, 3, 128, 128)
- Output shape: (batch, 3, 128, 128)
- Output range: [0, 1] via Sigmoid

**Kỹ thuật:**
- Component diagram với nested rectangles
- Notes cho technical details (kernel size, stride, padding)

---

### **Hình 2.4: Flowchart Thuật Toán**
**File:** `fig2_4_flowchart_algorithm.puml`

**Mục đích:** Minh họa luồng hoạt động của hệ thống real-time

**Nội dung chính:**
1. Load Trained Model (`autoencoder_processed_clean.pt`)
2. Capture Iris Image (Webcam or Upload)
3. Detect Eye Region (MediaPipe FaceMesh)
4. **Decision 1:** Eye detected?
   - NO → Error: No eye detected → Try again
   - YES → Continue
5. Preprocessing:
   - Crop eyebrows
   - Apply mask
   - Resize to 128×128
   - Normalize [0, 1]
6. AutoEncoder Forward Pass: `X_recon = model(X_input)`
7. Calculate MSE: `mse = mean((X - X_recon)²)`
8. **Decision 2:** MSE < Threshold?
   - YES → Result: **REAL** (Valid)
   - NO → Result: **FAKE** (Spoofed)
9. Stop

**Kỹ thuật:**
- Activity diagram với multiple if-then-else
- Color coding cho các kết quả khác nhau

---

### **Hình 2.5: Sơ Đồ Triển Khai**
**File:** `fig2_5_deployment_diagram.puml`

**Mục đích:** Minh họa kiến trúc triển khai hệ thống

**Nội dung chính:**

**Development Environment:**
- Google Colab (training platform)
- Google Drive:
  - Dataset UBIPR2
  - Trained Models
  - Reports

**Training Pipeline:**
- Data Preprocessing component
- AutoEncoder Training component
- Model Evaluation component

**Inference System:**
- Real-time Detector component:
  - MediaPipe (eye detection)
  - OpenCV (image processing)
  - PyTorch Model (inference)
- Webcam (input device)

**User Interface:**
- Display Results component

**Connections:**
- Colab → Preprocessing
- Dataset → Preprocessing
- Preprocessing → Training
- Training → Models & Reports
- Models → PyTorch Model
- Webcam → Detector → Display

**Notes:**
- **Training Configuration:**
  - Epochs: 100
  - Batch size: 64
  - Optimizer: AdamW
  - Learning rate: 1e-3
  - Loss: MSE

- **Real-time Performance:**
  - Latency: ~10-50ms
  - FPS: 20-100
  - Device: CPU/GPU
  - Threshold: Auto-computed

**Kỹ thuật:**
- Deployment diagram với nodes và components
- Database stereotype cho storage

---

### **Hình 2.6: Công Thức Toán Học**
**File:** `fig2_6_mathematical_formulas.tex`

**Mục đích:** Trình bày các công thức toán học của AutoEncoder và MSE

**Nội dung chính:**

**Section 1: AutoEncoder Model**
- Encoder: `z = f_enc(x; θ_enc)`
- Decoder: `x̂ = f_dec(z; θ_dec)`
- Complete AutoEncoder: `x̂ = f_AE(x; θ) = f_dec(f_enc(x; θ_enc); θ_dec)`

**Section 2: Loss Function (Training)**
- Mean Squared Error (MSE):
  ```
  L(x, x̂) = (1/N) Σ(xi - x̂i)²
  ```
  where N = 128 × 128 × 3 = 49,152 (total pixels)

- Optimization Objective:
  ```
  θ* = argmin E[L(x, f_AE(x; θ))]
  ```

**Section 3: Anomaly Detection (Inference)**
- Reconstruction Error: `e(x) = L(x, f_AE(x; θ*))`
- Threshold Computation:
  ```
  τ = μ_real + k · σ_real
  ```
  where:
  - μ_real: Mean MSE on REAL validation set
  - σ_real: Std MSE on REAL validation set
  - k = 2: Confidence level (95% of REAL iris)

- Classification Rule:
  ```
  predict(x) = REAL if e(x) < τ
               FAKE if e(x) ≥ τ
  ```

**Section 4: Giả Thuyết (Hypothesis)**
- Model train chỉ với REAL iris → reconstruct REAL tốt (low MSE)
- FAKE iris (printed, displayed, contact lens) → reconstruct kém (high MSE)

**Kỹ thuật:**
- LaTeX document với amsmath, amssymb
- Mathematical notation chuẩn
- Compile với pdflatex hoặc Overleaf

---
