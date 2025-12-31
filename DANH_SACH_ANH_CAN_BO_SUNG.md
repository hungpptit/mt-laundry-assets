# 📊 DANH SÁCH ẢNH CẦN BỔ SUNG CHO CHƯƠNG 3

## ✅ ẢNH ĐÃ CÓ (3 ảnh)

1. ✅ **Hình 3.1:** Loss curve - `Aspose.Words.096562af-d0a4-4330-89bf-2428db5bf9e1.001.png`
2. ✅ **Hình 3.2:** Best/Worst reconstruction - `Aspose.Words.096562af-d0a4-4330-89bf-2428db5bf9e1.002.png`
3. ✅ **Hình 3.3:** Confusion Matrix + ROC + Histogram - `Aspose.Words.096562af-d0a4-4330-89bf-2428db5bf9e1.003.png`

---

## 🔴 ẢNH CẦN BỔ SUNG (7 ảnh được đề xuất trong IOT.md)

### **NHÓM 1: MSE DISTRIBUTION (1 ảnh)**

#### **Hình 3.4: MSE Distribution với Threshold Lines** ⭐ QUAN TRỌNG
**Nội dung:**
- Histogram MSE của validation set (579 REAL images)
- Các đường threshold:
  - Mean (đỏ đứt nét)
  - Mean + 1×Std (cam đứt nét)
  - Mean + 2×Std (xanh lá - recommended)
  - Mean + 3×Std (xanh dương đứt nét)
  - 95th percentile (tím đứt nét)
- Annotation: "95% REAL below this line"

**Lý do cần:**
- Minh họa cho **Bảng 3.X: Phân tích các mức ngưỡng** (Phần 3.5.2.1 Sensitivity Analysis)
- Chứng minh why Mean+2×Std là optimal choice
- Giải thích distribution của reconstruction error

**Code tạo ảnh từ notebook:**
```python
# Từ validation set MSE
plt.figure(figsize=(12, 6))
plt.hist(all_mses, bins=50, alpha=0.7, edgecolor='black', label='MSE Distribution')

# Vẽ các threshold lines
mean_mse = np.mean(all_mses)
std_mse = np.std(all_mses)
percentile_95 = np.percentile(all_mses, 95)

plt.axvline(mean_mse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_mse:.6f}')
plt.axvline(mean_mse + 1*std_mse, color='orange', linestyle='--', linewidth=2, label=f'Mean+1×Std: {mean_mse + 1*std_mse:.6f}')
plt.axvline(mean_mse + 2*std_mse, color='green', linestyle='-', linewidth=3, label=f'Mean+2×Std (Recommended): {mean_mse + 2*std_mse:.6f}')
plt.axvline(mean_mse + 3*std_mse, color='blue', linestyle='--', linewidth=2, label=f'Mean+3×Std: {mean_mse + 3*std_mse:.6f}')
plt.axvline(percentile_95, color='purple', linestyle='--', linewidth=2, label=f'95th Percentile: {percentile_95:.6f}')

plt.xlabel('MSE (Reconstruction Error)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('MSE Distribution với Các Mức Ngưỡng Đề Xuất', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('C3/fig3_4_mse_distribution_thresholds.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

### **NHÓM 2: FAILURE CASES (1-2 ảnh)** ⭐ KHUYẾN NGHỊ CAO

#### **Hình 3.5: Failure Cases Examples** ⭐⭐ RẤT QUAN TRỌNG
**Nội dung:**
- Grid 2 rows × 3 columns (6 ví dụ)
- **Row 1: False Positives (REAL bị classify là FAKE)**
  1. Low light condition → High MSE
  2. Partial occlusion (phản quang) → High MSE
  3. Blurry image → High MSE
- **Row 2: False Negatives (FAKE bị classify là REAL)**
  1. High-quality OLED display → Low MSE
  2. High-resolution printed photo → Low MSE
  3. Clear contact lens → Low MSE

**Mỗi ảnh nhỏ có:**
- Input image
- Reconstructed image
- MSE value
- Ground truth label (REAL/FAKE)
- Predicted label (sai)
- Lý do fail

**Lý do cần:**
- Minh họa cho **Phần 3.5.3: Failure Cases Analysis**
- Giúp giám khảo hiểu limitations
- Chứng minh phân tích failure cases là có thực

**Có thể tạo từ:**
- Validation set: chọn REAL images có MSE cao nhất
- Nếu có test FAKE samples: chọn FAKE có MSE thấp nhất

---

### **NHÓM 3: REAL-TIME PERFORMANCE (1 ảnh)** ⚡ NẾU CÓ DATA

#### **Hình 3.6: Real-time System Performance**
**Nội dung:**
- Screenshot từ `main_realtime_new.py` running
- Hoặc bar chart comparing:
  - FPS (CPU vs GPU)
  - Latency (mean, max)
  - Detection rate

**Lý do cần:**
- Minh họa cho **Phần 3.3.7-3.3.8: Real-time Evaluation**
- Chứng minh system chạy thực tế
- Đáp ứng yêu cầu "hình minh họa thực tế"

**Nếu chưa có data:**
- Có thể skip hoặc chỉ vẽ bar chart từ số liệu (2.84ms, 352 FPS)

---

### **NHÓM 4: COMPARISON CHART (1 ảnh)** 📊 TÙY CHỌN

#### **Hình 3.7: Comparison với State-of-the-art Methods**
**Nội dung:**
- Bar chart hoặc radar chart so sánh:
  - Phương pháp handcrafted
  - CNN supervised
  - AutoEncoder (đề xuất)
- Theo 5 tiêu chí trong Bảng 3.10:
  1. Need FAKE data (Yes/No)
  2. Detect novel attacks (Low/Medium/High)
  3. Model complexity (Low/Medium/High)
  4. Generalization (Low/Medium/High)
  5. Real-time deployment (Low/Medium/High)

**Lý do cần:**
- Visualization cho **Bảng 3.10**
- Dễ hiểu hơn bảng text
- Professional thesis standard

---

### **NHÓM 5: CROSS-DATASET CONCEPT (1 ảnh)** 🌐 TÙY CHỌN

#### **Hình 3.8: Cross-dataset Evaluation Protocol**
**Nội dung:**
- Flowchart hoặc diagram showing:
  - Phase 1: UBIPR2 → UBIPR2 (Intra-dataset)
  - Phase 2: UBIPR2 → LivDet-Iris (Cross-dataset)
  - Phase 3: UBIPR2 → Notre Dame (Contact lens)
  - Phase 4: Sensor A → Sensor B (Cross-sensor)
- Với expected performance degradation

**Lý do cần:**
- Minh họa cho **Phần 4.5: Cross-dataset Evaluation**
- Làm rõ future work
- Thể hiện hiểu biết về research methodology

---

## 🎯 KHUYẾN NGHỊ ƯU TIÊN

### **BẮT BUỘC (Cần tạo ngay):**

1. ⭐⭐⭐ **Hình 3.4: MSE Distribution với Thresholds** 
   - CỰC KỲ QUAN TRỌNG cho Sensitivity Analysis
   - Dễ tạo từ data validation set có sẵn
   - ~10 phút code

2. ⭐⭐⭐ **Hình 3.5: Failure Cases Examples**
   - QUAN TRỌNG cho Failure Analysis
   - Cần chọn examples từ validation/test set
   - ~30 phút (chọn ảnh + tạo grid)

### **NÊN CÓ (Tùy thời gian):**

3. ⭐⭐ **Hình 3.6: Real-time Performance**
   - Nếu có data từ `main_realtime_new.py`
   - Hoặc chỉ vẽ bar chart đơn giản từ số liệu có
   - ~15 phút

4. ⭐ **Hình 3.7: Comparison Chart**
   - Visualization cho Bảng 3.10
   - Không bắt buộc nhưng làm tăng chất lượng
   - ~20 phút

### **TÙY CHỌN (Nếu muốn perfect):**

5. ⚡ **Hình 3.8: Cross-dataset Protocol**
   - Cho phần Future Work
   - Có thể dùng PlantUML như Chương 2
   - ~30 phút

---

## 📝 CODE MẪU TẠO ẢNH

### **1. MSE Distribution với Thresholds (Hình 3.4)**

```python
import matplotlib.pyplot as plt
import numpy as np

# Giả sử đã có all_mses từ validation
mean_mse = np.mean(all_mses)
std_mse = np.std(all_mses)
percentile_95 = np.percentile(all_mses, 95)

plt.figure(figsize=(14, 6))

# Histogram
plt.hist(all_mses, bins=50, alpha=0.7, edgecolor='black', color='skyblue', label='MSE Distribution')

# Threshold lines
plt.axvline(mean_mse, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {mean_mse:.6f}')
plt.axvline(mean_mse + 1*std_mse, color='orange', linestyle='--', linewidth=2, 
            label=f'Mean+1×Std: {mean_mse + 1*std_mse:.6f}')
plt.axvline(mean_mse + 2*std_mse, color='green', linestyle='-', linewidth=3, 
            label=f'Mean+2×Std (Recommended): {mean_mse + 2*std_mse:.6f}')
plt.axvline(mean_mse + 3*std_mse, color='blue', linestyle='--', linewidth=2, 
            label=f'Mean+3×Std: {mean_mse + 3*std_mse:.6f}')
plt.axvline(percentile_95, color='purple', linestyle='--', linewidth=2, 
            label=f'95th Percentile: {percentile_95:.6f}')

# Annotations
plt.text(mean_mse + 2*std_mse, plt.ylim()[1]*0.9, 
         '← 95% REAL below this line\n(Recommended threshold)', 
         ha='left', va='top', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.xlabel('MSE (Reconstruction Error)', fontsize=13, fontweight='bold')
plt.ylabel('Frequency', fontsize=13, fontweight='bold')
plt.title('Distribution of MSE với Các Mức Ngưỡng Đề Xuất\n(Validation Set: 579 REAL Images)', 
          fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('D:/autoencoder_processed_clean/C3/fig3_4_mse_distribution_thresholds.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Đã tạo: fig3_4_mse_distribution_thresholds.png")
```

### **2. Failure Cases Grid (Hình 3.5)**

```python
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Giả sử có lists:
# false_positive_images = [(input, recon, mse, reason), ...]  # REAL nhưng MSE cao
# false_negative_images = [(input, recon, mse, reason), ...]  # FAKE nhưng MSE thấp

fig, axes = plt.subplots(2, 6, figsize=(20, 7))

# Row 1: False Positives (REAL → FAKE)
for i, (input_img, recon_img, mse, reason) in enumerate(false_positive_images[:3]):
    # Original
    axes[0, i*2].imshow(input_img)
    axes[0, i*2].set_title(f'REAL Input\nMSE={mse:.4f}', fontsize=10, color='red')
    axes[0, i*2].axis('off')
    
    # Reconstructed
    axes[0, i*2+1].imshow(recon_img)
    axes[0, i*2+1].set_title(f'Reconstruction\n{reason}', fontsize=9)
    axes[0, i*2+1].axis('off')

# Row 2: False Negatives (FAKE → REAL)
for i, (input_img, recon_img, mse, reason) in enumerate(false_negative_images[:3]):
    # Original
    axes[1, i*2].imshow(input_img)
    axes[1, i*2].set_title(f'FAKE Input\nMSE={mse:.4f}', fontsize=10, color='blue')
    axes[1, i*2].axis('off')
    
    # Reconstructed
    axes[1, i*2+1].imshow(recon_img)
    axes[1, i*2+1].set_title(f'Reconstruction\n{reason}', fontsize=9)
    axes[1, i*2+1].axis('off')

# Labels
fig.text(0.02, 0.75, 'FALSE POSITIVES\n(REAL → FAKE)\nModel fails to\nreconstruct well', 
         ha='left', va='center', fontsize=12, fontweight='bold', color='red')
fig.text(0.02, 0.25, 'FALSE NEGATIVES\n(FAKE → REAL)\nModel reconstructs\ntoo well', 
         ha='left', va='center', fontsize=12, fontweight='bold', color='blue')

plt.suptitle('Failure Cases Analysis: False Positives và False Negatives', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0.05, 0, 1, 0.96])
plt.savefig('D:/autoencoder_processed_clean/C3/fig3_5_failure_cases.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Đã tạo: fig3_5_failure_cases.png")
```

### **3. Real-time Performance Bar Chart (Hình 3.6)**

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['Latency\n(ms)', 'Throughput\n(FPS)', 'Detection Rate\n(%)']
cpu_values = [50, 25, 92]  # Ví dụ
gpu_values = [2.84, 352, 95]  # Từ IOT.md

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, cpu_values, width, label='CPU', color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, gpu_values, width, label='GPU (Tesla T4)', color='lightcoral', edgecolor='black')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Value', fontsize=13, fontweight='bold')
ax.set_title('Real-time System Performance: CPU vs GPU', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('D:/autoencoder_processed_clean/C3/fig3_6_realtime_performance.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("✅ Đã tạo: fig3_6_realtime_performance.png")
```

---

## 🚀 HÀNH ĐỘNG ĐỀ XUẤT

### **Ngay bây giờ (15-30 phút):**
1. Chạy code tạo **Hình 3.4** (MSE Distribution) - QUAN TRỌNG NHẤT
2. Đổi tên 3 ảnh hiện có cho dễ nhận diện:
   - `Aspose...001.png` → `fig3_1_loss_curve.png`
   - `Aspose...002.png` → `fig3_2_best_worst_reconstruction.png`
   - `Aspose...003.png` → `fig3_3_confusion_matrix_roc.png`

### **Trong vài giờ tới (nếu có thời gian):**
3. Tạo **Hình 3.5** (Failure Cases) - chọn examples từ validation set
4. Tạo **Hình 3.6** (Real-time Performance) - simple bar chart

### **Tùy chọn (nếu muốn perfect):**
5. Tạo **Hình 3.7** (Comparison chart)
6. Tạo **Hình 3.8** (Cross-dataset protocol flowchart)

---

## 📊 TÓM TẮT

**Hiện có:** 3 ảnh (đủ minimum)
**Đề xuất thêm:** 2-5 ảnh (để hoàn thiện)
**Ưu tiên cao nhất:** Hình 3.4 (MSE Distribution) và Hình 3.5 (Failure Cases)

**Kết luận:** Với 3 ảnh hiện tại, bạn đã đủ để defend được luận văn. Nhưng nếu thêm được 2 ảnh nữa (3.4 và 3.5), chương 3 sẽ HOÀN HẢO và thể hiện được phân tích sâu sắc! 🎓
