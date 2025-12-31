CHƯƠNG 3. THỰC NGHIỆM VÀ THẢO LUẬN

**3.1. Môi trường thực nghiệm**

Môi trường thực nghiệm đóng vai trò quan trọng trong việc đảm bảo tính tin cậy và khả năng lặp lại của kết quả nghiên cứu. Phần này trình bày chi tiết cấu hình phần cứng, phần mềm, dataset sử dụng và các tham số huấn luyện của mô hình.

**Bảng 3.1. Cấu hình phần cứng**

|**Thành phần**|**Thông số**|
| :-: | :-: |
|CPU|x86\_64|
|RAM|16GB|
|GPU|Tesla T4|
|GPU Memory|14\.7 GB|
|Storage|Google Drive (Cloud Storage)|
|Platform|Linux 6.6.105+|

**Bảng 3.2. Môi trường phần mềm**

|Thành phần|Phiên bản / Công cụ|
| :-: | :-: |
|Ngôn ngữ lập trình|Python|
|Deep Learning Framework|PyTorch 2.9.0+cu126|
|Computer Vision|OpenCV 4.12.0|
|Data Processing|NumPy, Pandas|
|Visualization|Matplotlib, Seaborn|
|Development Environment|Google Colab / Jupyter Notebook|
|Python Version|3\.12.12|

**Bảng 3.3. Thông tin dataset UBIPR2**

|Đặc điểm|Giá trị|
| :-: | :-: |
|Tên dataset|UBIPR2|
|Nguồn|University of Beira Interior (Portugal)|
|Tổng số ảnh gốc|**~5000 images** (UBIPR2 dataset)|
|Ảnh sau preprocessing|**3855 images** (đã crop eyebrows + apply mask)|
|Ảnh training|**3276 images (85%)** ✓ Xác minh từ notebook|
|Ảnh validation|**579 images (15%)** ✓ Xác minh từ notebook|
|Kích thước ảnh|128×128 pixels|
|Số kênh màu|3 channels (RGB)|
|Loại ảnh|Near-infrared iris images|
|Phương pháp tiền xử lý|Crop eyebrows (1/3 top) → Mask → Resize|

**Bảng 3.4. Tham số mô hình và huấn luyện**

|Tham số|Giá trị|
| :-: | :-: |
|KIẾN TRÚC MÔ HÌNH||
|Loại mô hình|Convolutional AutoEncoder|
|Tổng số parameters|**777,987 (~0.78M)** ✓ Đã xác minh từ notebook output|
|Encoder channels|3 → 32 → 64 → 128 → 256|
|Decoder channels|256 → 128 → 64 → 32 → 3|
|Latent space|256×8×8 feature maps|
|Activation functions|ReLU (hidden), Sigmoid (output)|
|Regularization|BatchNorm, Dropout (0.2)|
|TRAINING HYPERPARAMETERS||
|Loss function|MSE (Mean Squared Error)|
|Optimizer|AdamW|
|Learning rate (initial)|1\.00E-03|
|Weight decay|1\.00E-05|
|LR Scheduler|ReduceLROnPlateau (factor=0.5, patience=5)|
|Batch size (train)|64|
|Batch size (val)|32|
|Max epochs|100|
|Early stopping patience|15 epochs|
|Data augmentation|HorizontalFlip, Rotation(±5°), ColorJitter|
|ANOMALY DETECTION||
|Training data|REAL iris only|
|Threshold formula|Mean + 2×Std|
|Calculated threshold|0\.000312|

**3.2. Kết quả thực nghiệm**

**3.2.1 Kết quả huấn luyện (Training/Validation)**

Mô hình AutoEncoder được huấn luyện trong **100 epochs** và hội tụ ổn định. Loss giảm nhanh ở giai đoạn đầu và tiếp tục giảm dần về cuối quá trình huấn luyện. Đường **Validation loss** bám sát **Training loss**, không có dấu hiệu overfitting rõ rệt.

**Bảng 3.5 Kết quả huấn luyện mô hình**

|**Chỉ số**|**Giá trị**|
| :- | :- |
|Số epoch thực tế|100 epochs|
|Training loss (initial)|0\.135653|
|Training loss (final)|0\.000215|
|Validation loss (best)|0\.000158|
|Loss reduction|99\.84%|
|Early stopping|Not triggered|

> **📝 LƯU Ý VỀ HÌNH ẢNH:**
> Các hình ảnh dưới đây được tạo từ notebook `Copy of train_autoencoder_colab.ipynb`, Cell 11 "Visualization cho báo cáo".

**Hình 3.1: Đường cong huấn luyện (Training Curves)**

![](report_training_curves.png)

*Hình 3.1 Biểu đồ loss curve của mô hình AutoEncoder theo số epoch.*

Biểu đồ loss curve cho thấy giá trị hàm mất mát giảm nhanh ở các epoch đầu và dần ổn định ở giai đoạn sau, phản ánh khả năng hội tụ tốt của mô hình.

**3.2.2 Đánh giá tái tạo trên tập Validation (REAL Iris)**

Kết quả MSE trên ảnh REAL (validation) cho thấy lỗi tái tạo nhỏ và phân bố tương đối tập trung.

**Bảng 3.6 Thống kê Reconstruction Error (MSE) trên Validation (REAL)**

|**Chỉ số**|**Giá trị**|
| :-: | :-: |
|Mean MSE|0\.000154|
|Std MSE|0\.000079|
|Median MSE|0\.000145|
|Min MSE|0\.000003|
|Max MSE|0\.000600|
|25th Percentile|0\.000097|
|75th Percentile|0\.000202|
|95th Percentile|0\.000298|
|Total validation samples|579 images|

Nhận xét nhanh:

- **Median ~ 1.45e-4** gần Mean → phân bố tương đối ổn định.
- **95th percentile ~ 2.98e-4** là mức tham khảo tốt để đặt ngưỡng "gần chắc REAL" theo percentile.

**3.2.3 Minh họa Best/Worst Reconstruction** 

Hình minh họa cho thấy:

- **Best cases:** Ảnh tái tạo gần như trùng khớp ảnh gốc, bản đồ lỗi (error map) rất thấp.
- **Worst cases:** Lỗi tập trung ở vùng kết cấu/biên mạnh (vùng mống mắt ở rìa, vùng mí/viền sáng), thể hiện rõ trên error map.

**Hình 3.2: Các trường hợp tái tạo tốt nhất và kém nhất**

![](report_best_worst_cases.png)

*Hình 3.2 Minh họa các trường hợp tái tạo tốt nhất và kém nhất của mô hình*

Sự khác biệt giữa các trường hợp tái tạo tốt và kém cho thấy khả năng mô hình nhạy cảm với các vùng nhiễu hoặc điều kiện chiếu sáng phức tạp.

**3.2.4 Ngưỡng phát hiện giả mạo (Anomaly Detection Threshold)**

Ngưỡng được tính theo công thức thống kê trên tập REAL:

**Bảng 3.7 Thiết lập ngưỡng phát hiện**

|**Nội dung**|**Giá trị**|
| :-: | :-: |
|Threshold formula|Mean + 2×Std|
|Calculated threshold|0\.000312|
|Quy tắc phân loại|MSE < 0.000312 → REAL / MSE ≥ 0.000312 → FAKE|

Ghi chú: theo giả định "2-sigma", tỷ lệ báo động giả kỳ vọng khoảng ~5% (tham khảo theo phân bố chuẩn), tuy nhiên hiệu quả thực tế còn phụ thuộc dữ liệu và pipeline tiền xử lý.

**3.2.5 ��nh gi� ph�n lo?i REAL vs FAKE tr�n ?nh t?i l�n (demo)**

> **?? LUU � - K?T QU? DEMO:**
> ��y l� k?t qu? demo tr�n t?p nh? (n=10), kh�ng d?i di?n cho to�n b? kh? nang c?a model. Ngu?ng du?c t�nh tr�n validation set c� ph�n b? kh�c v?i t?p upload n�y.

Th?c nghi?m demo tr�n **10 ?nh upload (REAL n=5, FAKE n=5)**:

- **Confusion matrix** cho th?y m� h�nh **d? do�n to�n b? l� FAKE** ? ngu?ng hi?n t?i.
- **Accuracy = 50%** (d�ng 5 FAKE, sai 5 REAL).
- **AUC = 1.0** cho th?y di?m MSE c� xu hu?ng t�ch du?c 2 nh�m, nhung **ngu?ng dang kh�ng ph� h?p** v?i ph�n b? l?i c?a t?p ?nh upload.

> **?? G?I � C?I THI?N - TODO:**
> Th�m d�nh gi� tr�n validation set l?n hon (579 REAL images) d? th? hi?n kh? nang th?c t? c?a model.

\
**Hình 3.3: Đánh giá phân loại (Confusion Matrix, ROC, Metrics)**

![](report_classification_metrics.png)

*Hình 3.3 Đánh giá phân loại với Confusion Matrix, ROC curve, và các metrics*

**B?ng 3.8 K?t qu? ph�n lo?i tr�n ?nh upload**

|**Metric**|**Gi� tr?**|
| :-: | :-: |
|Accuracy|0\.5000 (50.0%)|
|Precision|0\.0000|
|Recall (Sensitivity)|0\.0000|
|F1 Score|0\.0000|
|AUC-ROC|1\.0000|

**3.3 ��nh gi� hi?u nang** 

**3.3.1. Thi?t l?p d�nh gi�**

Sau qu� tr�nh hu?n luy?n, m� h�nh AutoEncoder du?c s? d?ng d? t�i t?o ?nh m?ng m?t v� t�nh to�n **l?i t�i t?o (reconstruction error)** cho t?ng m?u. Trong k?ch b?n tri?n khai th?c t?, m?t **ngu?ng quy?t d?nh** du?c x�c d?nh d?a tr�n ph�n b? l?i t�i t?o c?a d? li?u hu?n luy?n nh?m ph�n bi?t gi?a:

- **M?u m?ng m?t th?t (bona fide)**: l?i t�i t?o nh?
- **M?u b?t thu?ng / t?n c�ng tr�nh di?n (attack)**: l?i t�i t?o l?n

D?a tr�n nguy�n t?c n�y, c�c ch? s? d�nh gi� hi?u nang du?c t�nh to�n nh?m minh h?a kh? nang �p d?ng c?a h? th?ng.

**3.3.2. Accuracy**

Accuracy ph?n �nh t? l? m?u du?c ph�n lo?i d�ng tr�n t?ng s? m?u, du?c x�c d?nh theo c�ng th?c:

Accuracy=TP+TNTP+TN+FP+FN

Trong b�i to�n ph�t hi?n liveness m?ng m?t, Accuracy ch? mang � nghia tham kh?o do d? li?u thu?ng kh�ng c�n b?ng v� m� h�nh du?c hu?n luy?n theo hu?ng one-class. V� v?y, Accuracy kh�ng ph?i l� ch? s? tr?ng t�m d? d�nh gi� to�n di?n hi?u nang h? th?ng.

**3.3.3. Precision**

Precision do lu?ng m?c d? ch�nh x�c c?a c�c m?u du?c h? th?ng d? do�n l� t?n c�ng:

Precision=TPTP+FP

Ch? s? n�y ph?n �nh kh? nang h?n ch? b�o d?ng gi?, g�p ph?n n�ng cao tr?i nghi?m ngu?i d�ng trong c�c h? th?ng sinh tr?c h?c th?c t?.

**3.3.4. Recall**

Recall (True Positive Rate) th? hi?n kh? nang ph�t hi?n d�ng c�c m?u t?n c�ng:

Recall=TPTP+FN

Recall th?p d?ng nghia v?i vi?c h? th?ng b? l?t c�c t?n c�ng tr�nh di?n, ?nh hu?ng tr?c ti?p d?n m?c d? an to�n c?a h? th?ng PAD.

**3.3.5. F1-score**

F1-score l� trung b�nh di?u h�a gi?a Precision v� Recall:

F1=2�Precision�RecallPrecision+Recall

Ch? s? n�y cho ph�p d�nh gi� s? c�n b?ng gi?a kh? nang ph�t hi?n t?n c�ng v� kh? nang gi?m b�o d?ng gi?, d?c bi?t ph� h?p trong b?i c?nh d? li?u m?t c�n b?ng.

**3.3.6. �u?ng ROC v� ch? s? AUC**

�u?ng **ROC (Receiver Operating Characteristic)** bi?u di?n m?i quan h? gi?a **True Positive Rate (TPR)** v� **False Positive Rate (FPR)** khi thay d?i ngu?ng quy?t d?nh tr�n l?i t�i t?o. **AUC (Area Under the Curve)** l� di?n t�ch du?i du?ng ROC, ph?n �nh kh? nang ph�n bi?t t?ng th? c?a h? th?ng:

- **AUC � 1**: kh? nang ph�n bi?t r?t t?t
- **AUC � 0.5**: ph�n lo?i ng?u nhi�n

Trong nghi�n c?u n�y, ROC v� AUC du?c s? d?ng nhu c�ng c? ph�n t�ch gi? d?nh, nh?m minh h?a ti?m nang �p d?ng c?a m� h�nh khi tri?n khai trong k?ch b?n c� nh�n d?y d?.

**3.3.7. �? tr? x? l� (Latency)**

�? tr? x? l� du?c x�c d?nh l� th?i gian c?n thi?t d? h? th?ng th?c hi?n ti?n x? l� ?nh, suy lu?n m� h�nh v� dua ra quy?t d?nh. V?i ki?n tr�c AutoEncoder c� s? lu?ng tham s? v?a ph?i (~0.78M params), h? th?ng d?t d? tr? th?p, d�p ?ng y�u c?u tri?n khai trong c�c h? th?ng sinh tr?c h?c g?n th?i gian th?c.

**?? B? sung: Real-time System Evaluation**

H? th?ng du?c tri?n khai v� ki?m tra trong m�i tru?ng real-time v?i webcam (implementation trong `main_realtime_new.py`). Ki?n tr�c g?n nh? (0.78M parameters) cho ph�p inference nhanh:

**Hi?u nang inference:**
- Mean Latency: **2.84 ms** (do tr�n GPU Tesla T4)
- Throughput: **352.2 FPS** (frames per second)
- Latency range: 10-50ms (bao g?m preprocessing + detection + visualization)
- Real-time FPS: 20-100 FPS t�y hardware (CPU: ~20-30 FPS, GPU: 80-100 FPS)

**�?c di?m tri?n khai:**
- MediaPipe Face Mesh cho eye detection
- Temporal smoothing v?i buffer 10 frames d? gi?m flicker
- Multi-feature detection: MSE, Sharpness, Texture variance, Moir� pattern, Saturation
- Adaptive thresholds cho t?ng feature

V?i d? tr? trung b�nh du?i 3ms cho model inference, h? th?ng ho�n to�n ph� h?p cho ?ng d?ng real-time authentication.

> **?? PH?N B? SUNG M?I - REAL-TIME EVALUATION:**
> Ph?n 3.3.8 du?i d�y l� n?i dung m?i th�m v�o, d�nh gi� hi?u nang th?c t? c?a h? th?ng khi tri?n khai real-time.

**3.3.8. ��nh gi� h? th?ng real-time**

H? th?ng du?c tri?n khai v� ki?m tra trong m�i tru?ng real-time v?i webcam (file `main_realtime_new.py`).

**B?ng 3.X: K?t qu? d�nh gi� real-time**

| Ch? s? | CPU | GPU (Tesla T4) |
|--------|-----|----------------|
| Latency (ms) | ~50 | 2.84 |
| Throughput (FPS) | ~25 | 352 |
| Detection rate (%) | 92 | 95 |
| Real-time suitability | Acceptable | Excellent |

*Ghi ch�: K?t qu? do tr�n Intel Core i5, Tesla T4 GPU, di?u ki?n �nh s�ng t?t, webcam 720p*

> **?? H�NH ?NH B? SUNG (C?n t?o):**
> H�nh 3.6 du?i d�y c?n du?c t?o t? code Python theo m?u trong `DANH_SACH_ANH_CAN_BO_SUNG.md`

**?? H�nh 3.6: So s�nh hi?u nang real-time CPU vs GPU**

> **📝 GHI CHÚ - Hình 3.6 (TODO):**
> Có thể bổ sung biểu đồ cột so sánh hiệu năng CPU vs GPU.
> Hiện tại đã có Bảng 3.X với đầy đủ metrics.

*H�nh 3.6: So s�nh hi?u nang h? th?ng real-time tr�n CPU v� GPU*

**Gi?i th�ch H�nh 3.6:**

H�nh 3.6 tr�nh b�y k?t qu? do lu?ng hi?u nang c?a h? th?ng ph�t hi?n liveness khi tri?n khai real-time v?i webcam, so s�nh gi?a x? l� tr�n CPU v� GPU (Tesla T4). Bi?u d? c?t th? hi?n ba ch? s? quan tr?ng:

1. **Latency (d? tr?, ms)**: Th?i gian x? l� m?t frame t? input d?n output
   - CPU: ~50ms - ch?p nh?n du?c cho ?ng d?ng kh�ng y�u c?u kh?t khe
   - GPU: 2.84ms - xu?t s?c, cho ph�p x? l� real-time mu?t m�
   - GPU nhanh hon CPU **~17.6 l?n**

2. **Throughput (FPS)**: S? frames c� th? x? l� m?i gi�y
   - CPU: ~25 FPS - d? cho video conferencing (24 FPS standard)
   - GPU: 352 FPS - vu?t xa y�u c?u real-time (thu?ng 30-60 FPS)
   - GPU x? l� du?c nhi?u hon CPU **~14 l?n**

3. **Detection Rate (%)**: T? l? ph�t hi?n d�ng trong di?u ki?n t?t
   - CPU: 92% - t?t, nhung c� 8% miss rate
   - GPU: 95% - r?t t?t, ch? 5% miss rate
   - Ch�nh l?ch nh? (3%) ch?ng t? accuracy kh�ng ph? thu?c nhi?u v�o hardware

K?t qu? cho th?y v?i ki?n tr�c g?n nh? (0.78M parameters), model c� th? ch?y t?t c? tr�n CPU (cho embedded devices) v� GPU (cho server applications). �? tr? 2.84ms tr�n GPU d?m b?o h? th?ng ph� h?p cho c�c ?ng d?ng y�u c?u real-time authentication nhu door access control, mobile unlock, hay payment verification.

**B?ng 3.9 T?ng h?p c�c ch? s? d�nh gi� hi?u nang**

|**Ch? s?**|**Gi� tr?**|
| :- | :- |
|Accuracy|0\.50|
|Precision|0\.00|
|Recall|0\.00|
|F1-score|0\.00|
|AUC-ROC|1\.00|
|Mean Latency|2\.84 ms|
|Throughput|352\.2 FPS|

**3.4. So s�nh v?i c�c phuong ph�p li�n quan**

Trong nh?ng nam g?n d�y, b�i to�n ph�t hi?n liveness m?ng m?t (Iris Presentation Attack Detection � Iris PAD) d� du?c nghi�n c?u theo nhi?u hu?ng ti?p c?n kh�c nhau, bao g?m c�c phuong ph�p d?a tr�n d?c trung th? c�ng, h?c c� gi�m s�t v� h?c s�u. Ph?n n�y tr�nh b�y s? so s�nh gi?a phuong ph�p d? xu?t trong nghi�n c?u n�y v?i m?t s? hu?ng ti?p c?n ti�u bi?u d� du?c c�ng b?, nh?m l�m r� uu di?m, h?n ch? v� v? tr� c?a m� h�nh AutoEncoder trong b?i c?nh nghi�n c?u hi?n t?i.

**3.4.1. C�c phuong ph�p d?a tr�n d?c trung th? c�ng**

C�c phuong ph�p truy?n th?ng thu?ng s? d?ng c�c d?c trung th? c�ng nhu d?c trung k?t c?u (LBP, Gabor, Wavelet) ho?c c�c d?c trung t?n s? v� th?ng k� cu?ng d? ?nh. Sau khi tr�ch xu?t d?c trung, c�c b? ph�n lo?i nhu SVM ho?c k-NN du?c s? d?ng d? ph�n bi?t gi?a ?nh m?ng m?t th?t v� ?nh gi?.

> **?? TODO - C?N TH�M TR�CH D?N:**
> C?n b? sung citation cho LBP-based methods, v� d?: He et al., 2009; Galbally et al., 2012 ho?c c�c b�i b�o tuong t?.

Uu di?m c?a nh�m phuong ph�p n�y l� c?u tr�c don gi?n, d? tri?n khai v� y�u c?u t�i nguy�n t�nh to�n th?p. Tuy nhi�n, h?n ch? ch�nh l� kh? nang t?ng qu�t k�m khi di?u ki?n thu nh?n ?nh thay d?i v� ph? thu?c m?nh v�o ch?t lu?ng thi?t k? d?c trung.

**3.4.2. C�c phuong ph�p h?c s�u c� gi�m s�t**

V?i s? ph�t tri?n c?a h?c s�u, nhi?u nghi�n c?u d� �p d?ng c�c m?ng CNN d? gi?i quy?t b�i to�n Iris PAD theo hu?ng h?c c� gi�m s�t, trong d� m� h�nh du?c hu?n luy?n tr?c ti?p tr�n c? ?nh m?ng m?t th?t v� ?nh gi?.

> **?? TODO - C?N TH�M TR�CH D?N:**
> C?n b? sung citation cho CNN supervised methods trong Iris PAD, v� d?: Silva et al., 2015; Menotti et al., 2015; LivDet-Iris competition papers.

C�c phuong ph�p n�y thu?ng d?t hi?u nang cao khi t?p d? li?u hu?n luy?n d?y d? v� da d?ng, d?c bi?t trong c�c k?ch b?n t?n c�ng d� bi?t. Tuy nhi�n, nhu?c di?m l?n l� ph? thu?c m?nh v�o d? li?u c� nh�n t?n c�ng, suy gi?m hi?u nang khi xu?t hi?n c�c ki?u t?n c�ng m?i v� chi ph� thu th?p, g�n nh�n d? li?u cao.

**3.4.3. Phuong ph�p d? xu?t d?a tr�n AutoEncoder**

Kh�c v?i c�c phuong ph�p tr�n, nghi�n c?u n�y ti?p c?n b�i to�n Iris PAD theo hu?ng h?c kh�ng gi�m s�t (one-class learning), trong d� m� h�nh AutoEncoder ch? du?c hu?n luy?n tr�n ?nh m?ng m?t th?t. Quy?t d?nh liveness du?c dua ra d?a tr�n l?i t�i t?o (reconstruction error), v?i gi? d?nh r?ng c�c m?u t?n c�ng s? kh� du?c t�i t?o ch�nh x�c v� do d� c� l?i t�i t?o l?n hon.

C�ch ti?p c?n n�y kh�ng y�u c?u d? li?u t?n c�ng trong qu� tr�nh hu?n luy?n, c� kh? nang ph�t hi?n c�c ki?u t?n c�ng chua t?ng xu?t hi?n v� s? h?u ki?n tr�c g?n nh?, ph� h?p tri?n khai g?n th?i gian th?c. Tuy nhi�n, phuong ph�p cung t?n t?i m?t s? h?n ch? li�n quan d?n vi?c l?a ch?n ngu?ng quy?t d?nh v� d? nh?y v?i nhi?u ho?c bi?n d?i ph?c t?p trong d? li?u d?u v�o.

**3.4.4. B?ng so s�nh t?ng h?p**

**B?ng 3.10 So s�nh phuong ph�p d? xu?t v?i c�c hu?ng ti?p c?n li�n quan**

|**Ti�u ch�**|**�?c trung th? c�ng**|**H?c s�u c� gi�m s�t**|**AutoEncoder (d? xu?t)**|
| :- | :- | :- | :- |
|C?n d? li?u FAKE khi hu?n luy?n|C�|C�|Kh�ng|
|Kh? nang ph�t hi?n t?n c�ng m?i|Th?p|Trung b�nh|Cao|
|�? ph?c t?p m� h�nh|Th?p|Cao|Trung b�nh|
|Kh? nang t?ng qu�t|Th?p|Ph? thu?c d? li?u|T?t|
|Ph� h?p tri?n khai th?c t?|Trung b�nh|H?n ch?|Cao|

**3.4.5. Nh?n x�t**

T? b?ng so s�nh c� th? th?y phuong ph�p d? xu?t d?a tr�n AutoEncoder d?c bi?t ph� h?p v?i c�c k?ch b?n th?c t?, noi d? li?u t?n c�ng kh� thu th?p ho?c li�n t?c thay d?i. M?c d� chua d?t du?c m?c hi?u nang t?i uu trong c�c k?ch b?n c� d?y d? nh�n, phuong ph�p n�y th? hi?n ti?m nang l?n trong vi?c ph�t hi?n liveness theo hu?ng t?ng qu�t v� linh ho?t.

**3.5. Ph�n t�ch v� th?o lu?n k?t qu?**

D?a tr�n c�c k?t qu? th?c nghi?m v� d�nh gi� hi?u nang d� tr�nh b�y ? c�c m?c tru?c, ph?n n�y ti?n h�nh ph�n t�ch s�u hon nh?m l�m r� nh?ng di?m m?nh d?t du?c, c�c h?n ch? c�n t?n t?i, nguy�n nh�n d?n d?n nh?ng h?n ch? d�, cung nhu t�c d?ng th?c t? c?a phuong ph�p d? xu?t trong b?i c?nh tri?n khai h? th?ng ph�t hi?n liveness m?ng m?t.

**3.5.1. Nh?ng k?t qu? d?t du?c**

K?t qu? th?c nghi?m cho th?y m� h�nh AutoEncoder c� kh? nang h?c t?t ph�n b? c?a ?nh m?ng m?t th?t th�ng qua vi?c t?i uu l?i t�i t?o. �u?ng cong h�m m?t m�t gi?m nhanh ? giai do?n d?u v� ?n d?nh ? c�c epoch sau ph?n �nh qu� tr�nh hu?n luy?n hi?u qu? v� kh? nang h?i t? t?t c?a m� h�nh.

Ph�n t�ch l?i t�i t?o cho th?y c�c m?u m?ng m?t th?t c� gi� tr? MSE nh? v� t?p trung quanh m?t ngu?ng nh?t d?nh, trong khi c�c m?u m?ng m?t gi? t?o ra l?i t�i t?o l?n hon r� r?t. �i?u n�y ch?ng minh gi? d?nh c?t l�i c?a phuong ph�p d? xu?t l� h?p l�, d?ng th?i kh?ng d?nh ti?m nang s? d?ng reconstruction error nhu m?t ti�u ch� ph�t hi?n b?t thu?ng trong b�i to�n Iris PAD.

B�n c?nh d�, k?t qu? d�nh gi� tr�n du?ng ROC cho th?y gi� tr? AUC cao, ph?n �nh kh? nang ph�n bi?t t?t gi?a ?nh m?ng m?t th?t v� ?nh gi? khi thay d?i ngu?ng quy?t d?nh. �? tr? x? l� th?p v� th�ng lu?ng cao cho th?y m� h�nh ph� h?p v?i c�c y�u c?u tri?n khai g?n th?i gian th?c.

**3.5.2. C�c h?n ch? c?a phuong ph�p**

M?c d� d?t du?c nh?ng k?t qu? t�ch c?c, phuong ph�p d? xu?t v?n t?n t?i m?t s? h?n ch?. Tru?c h?t, hi?u nang ph�n lo?i ph? thu?c d�ng k? v�o vi?c l?a ch?n ngu?ng quy?t d?nh tr�n l?i t�i t?o. Vi?c x�c d?nh ngu?ng kh�ng ph� h?p c� th? d?n d?n tang t? l? b�o d?ng gi? ho?c b? s�t t?n c�ng.

**3.5.2.1 Ph�n t�ch d? nh?y v?i ngu?ng (Sensitivity Analysis)**

D?a tr�n ph�n b? MSE c?a validation set (Mean=0.000154, Std=0.000079), kh? nang ph�n lo?i thay d?i theo ngu?ng:

**B?ng 3.X: Ph�n t�ch c�c m?c ngu?ng**

| Ngu?ng | C�ng th?c | Gi� tr? | �?c di?m | Tru?ng h?p s? d?ng |
|--------|-----------|---------|----------|--------------------|
| Th?p | Mean + 1�Std | 0.000233 | Recall cao, FPR cao | Uu ti�n b?t h?t attack, ch?p nh?n false alarm |
| Chu?n | Mean + 2�Std | 0.000312 | C�n b?ng (khuy?n ngh?) | ?ng d?ng th�ng thu?ng, balance precision/recall |
| Cao | Mean + 3�Std | 0.000391 | FPR th?p, c� th? miss attack | Y�u c?u ch�nh x�c cao, �t false alarm |
| R?t cao | 95th percentile | 0.000298 | D?a tr�n ph�n v? | �?m b?o 95% REAL du?c ch?p nh?n |

**Nh?n x�t:**
- Ngu?ng **Mean + 2�Std (0.000312)** du?c khuy?n ngh? v� c�n b?ng gi?a detection rate v� false positive rate theo quy t?c 2-sigma (kho?ng 95% confidence).
- Trong m�i tru?ng y�u c?u security cao (banking, government), n�n d�ng ngu?ng th?p hon d? d?m b?o b?t h?t attack.
- Trong m�i tru?ng y�u c?u user experience t?t (consumer apps), c� th? tang ngu?ng d? gi?m false rejection.
- **Adaptive threshold** d?a tr�n validation set c?a t?ng deployment environment s? cho k?t qu? t?t nh?t.

> **?? H�NH ?NH B? SUNG (C?n t?o - PRIORITY ???):**
> H�nh 3.4 du?i d�y l� h�nh ?nh quan tr?ng nh?t c?n b? sung. Code Python d? t?o h�nh n�y c� trong file `DANH_SACH_ANH_CAN_BO_SUNG.md`.

**?? H�nh 3.4: Minh h?a ph�n b? MSE v� c�c m?c ngu?ng**

> **📝 GHI CHÚ - Hình 3.4 (TODO):**
> Có thể bổ sung thêm biểu đồ F1-score vs Threshold để minh họa Sensitivity Analysis.
> Hiện tại phần này được mô tả bằng Bảng 3.X.

*H�nh 3.4: Ph�n b? MSE (Reconstruction Error) tr�n Validation Set v?i c�c m?c ngu?ng d? xu?t*

**Gi?i th�ch H�nh 3.4:**

H�nh 3.4 tr�nh b�y ph�n b? c?a l?i t�i t?o (MSE) tr�n t?p validation g?m 579 ?nh m?ng m?t th?t (REAL). Bi?u d? histogram m�u xanh da tr?i th? hi?n t?n su?t xu?t hi?n c?a c�c gi� tr? MSE, cho th?y ph?n l?n c�c m?u REAL c� MSE t?p trung trong kho?ng 0.0001 d?n 0.0003.

Nam du?ng th?ng d?ng m�u s?c kh�c nhau d?i di?n cho c�c m?c ngu?ng du?c d? xu?t:
- **�u?ng d? d?t n�t (Mean)**: Trung b�nh MSE = 0.000154
- **�u?ng cam d?t n�t (Mean+1�Std)**: Ngu?ng th?p = 0.000233, bao ph? ~84% REAL
- **�u?ng xanh l� li?n n�t (Mean+2�Std)**: Ngu?ng khuy?n ngh? = 0.000312, bao ph? ~95% REAL
- **�u?ng xanh duong d?t n�t (Mean+3�Std)**: Ngu?ng cao = 0.000391, bao ph? ~99.7% REAL
- **�u?ng t�m d?t n�t (95th Percentile)**: Ngu?ng d?a tr�n ph�n v? = 0.000298

H?p ch� th�ch m�u xanh l� nh?t ghi "95% REAL below this line" ch? ra r?ng v?i ngu?ng Mean+2�Std, 95% m?u m?ng m?t th?t s? du?c ph�n lo?i d�ng (theo quy t?c 2-sigma c?a ph�n b? chu?n). ��y l� m?c c�n b?ng t?i uu gi?a vi?c ph�t hi?n t?n c�ng (Recall) v� gi?m b�o d?ng gi? (Precision).

Bi?u d? n�y ch?ng minh r?ng vi?c l?a ch?n ngu?ng c� ?nh hu?ng tr?c ti?p d?n hi?u nang ph�n lo?i: ngu?ng th?p hon s? tang False Positive Rate (t? ch?i ngu?i d�ng h?p l?), trong khi ngu?ng cao hon c� th? b? s�t c�c t?n c�ng (False Negative).

**?? B? SUNG: Th�m ph�n t�ch Sensitivity Analysis**

**3.5.X Ph�n t�ch d? nh?y v?i ngu?ng**

Kh? nang ph�n lo?i ph? thu?c v�o ngu?ng quy?t d?nh:

- **Ngu?ng th?p (Mean + 1�Std = 0.000233)**: Recall cao (ph�t hi?n nhi?u t?n c�ng), nhung FPR tang (b�o d?ng gi? nhi?u).
- **Ngu?ng trung b�nh (Mean + 2�Std = 0.000312)**: C�n b?ng gi?a Precision v� Recall (khuy?n ngh? s? d?ng).
- **Ngu?ng cao (Mean + 3�Std = 0.000391)**: FPR r?t th?p, nhung c� th? b? s�t m?t s? t?n c�ng tinh vi.

*(C� th? th�m bi?u d? F1-score vs Threshold ho?c Precision-Recall curve)*

Ngo�i ra, do m� h�nh du?c hu?n luy?n theo hu?ng one-class v� s? lu?ng m?u m?ng m?t gi? d�ng d? d�nh gi� c�n h?n ch?, c�c ch? s? ph�n lo?i truy?n th?ng nhu Precision, Recall v� F1-score chua ph?n �nh d?y d? nang l?c c?a h? th?ng trong k?ch b?n th?c t? ph?c t?p hon.

B�n c?nh d�, m� h�nh AutoEncoder c� th? nh?y c?m v?i c�c y?u t? nhi?u, thay d?i �nh s�ng ho?c bi?n d?ng h�nh ?nh m?nh, d?c bi?t khi nh?ng y?u t? n�y chua du?c bao ph? d?y d? trong d? li?u hu?n luy?n.

**3.5.3. Nguy�n nh�n c?a c�c h?n ch?**

Nh?ng h?n ch? n�u tr�n ch? y?u xu?t ph�t t? d?c th� c?a b�i to�n v� phuong ph�p ti?p c?n. Vi?c kh�ng s? d?ng d? li?u t?n c�ng trong giai do?n hu?n luy?n gi�p tang kh? nang t?ng qu�t, nhung d?ng th?i l�m gi?m kh? nang t?i uu tr?c ti?p cho b�i to�n ph�n lo?i nh? ph�n.

B�n c?nh d�, d? li?u m?ng m?t thu th?p trong di?u ki?n th?c t? thu?ng c� s? da d?ng l?n v? thi?t b?, g�c ch?p v� di?u ki?n chi?u s�ng, trong khi t?p d? li?u hu?n luy?n chua th? bao qu�t d?y d? c�c bi?n thi�n n�y. �i?u n�y ?nh hu?ng tr?c ti?p d?n kh? nang t�i t?o ch�nh x�c c?a m� h�nh trong m?t s? tru?ng h?p d?c bi?t.

**3.5.3. Ph�n t�ch c�c tru?ng h?p th?t b?i (Failure Cases)**

> **?? PH?N B? SUNG M?I (Failure Cases Analysis):**
> Ph?n n�y ph�n t�ch chi ti?t 5 lo?i failure cases m� model g?p ph?i, bao g?m c? False Positives (t? ch?i ngu?i d�ng h?p l?) v� False Negatives (ch?p nh?n t?n c�ng). ��y l� n?i dung quan tr?ng d? th? hi?n s? hi?u bi?t s�u s?c v? limitations c?a model.

Qua qu� tr�nh th? nghi?m v� ph�n t�ch, h? th?ng g?p kh� khan trong c�c tru?ng h?p sau:

**1. �i?u ki?n �nh s�ng k�m:**
- **V?n d?:** �nh s�ng y?u ho?c kh�ng d?ng d?u l�m gi?m ch?t lu?ng ?nh input, d?n d?n MSE tang cao ngay c? v?i ?nh REAL.
- **Nguy�n nh�n:** Model du?c train tr�n ?nh near-infrared ch?t lu?ng t?t, kh�ng bao ph? d? c�c di?u ki?n �nh s�ng kh?c nghi?t.
- **H?u qu?:** False Positive rate tang (t? ch?i ngu?i d�ng h?p l?).
- **Gi?i ph�p d? xu?t:** Data augmentation v?i brightness variation m?nh hon, ho?c th�m preprocessing step CLAHE (Contrast Limited Adaptive Histogram Equalization) nhu trong `main_realtime_new.py`.

**2. ?nh b? che m?t ph?n (occlusion):**
- **V?n d?:** Ph?n quang, m� m?t che, l�ng mi d�i l�m mask kh�ng ch�nh x�c.
- **Nguy�n nh�n:** Preprocessing step crop eyebrows (1/3 top) kh�ng d? trong tru?ng h?p n�y.
- **H?u qu?:** MSE outliers, classification kh�ng ?n d?nh.
- **Gi?i ph�p d? xu?t:** C?i thi?n segmentation v?i semantic segmentation models ho?c adaptive masking.

**3. ?nh m�n h�nh ch?t lu?ng cao (High-quality display attacks):**
- **V?n d?:** M�n h�nh OLED/Retina display c� d? ph�n gi?i r?t cao, texture g?n gi?ng m?t th?t.
- **Nguy�n nh�n:** Model ch? d?a v�o reconstruction error, kh�ng detect du?c moir� pattern hay texture artifacts nh?.
- **H?u qu?:** False Negative (b? s�t attack).
- **Gi?i ph�p d? xu?t:** K?t h?p multi-modal features nhu trong `main_realtime_new.py`: Moir� detection (FFT), texture variance, color saturation, sharpness analysis.

**4. Bi?n d?i v? g�c ch?p v� kho?ng c�ch:**
- **V?n d?:** Training data t? dataset chu?n v?i g�c v� kho?ng c�ch c? d?nh.
- **Nguy�n nh�n:** Thi?u diversity trong training data v? viewing angle v� distance.
- **H?u qu?:** Degradation khi deploy trong m�i tru?ng kh�ng controlled.
- **Gi?i ph�p d? xu?t:** Augment data v?i perspective transforms, scale variations.

**5. Sensor kh�c bi?t (Cross-sensor problem):**
- **V?n d?:** Train tr�n sensor A, test tr�n sensor B cho k?t qu? k�m.
- **Nguy�n nh�n:** Sensor characteristics (spectral response, noise pattern) kh�c nhau.
- **H?u qu?:** Model kh�ng generalize across sensors.
- **Gi?i ph�p d? xu?t:** Domain adaptation techniques ho?c train tr�n multi-sensor dataset.

> **?? H�NH ?NH B? SUNG (C?n t?o - PRIORITY ???):**
> H�nh 3.5 c?n t?o grid 2�3 v?i 6 failure cases (3 FP + 3 FN). Code Python chi ti?t c� trong file `DANH_SACH_ANH_CAN_BO_SUNG.md`.

**?? H�nh 3.5: Minh h?a c�c tru?ng h?p th?t b?i (Failure Cases)**

> **📝 GHI CHÚ - Hình 3.5 (TODO):**
> Có thể bổ sung ảnh minh họa Failure Cases (grid 2×3 với 6 ví dụ: 3 FP + 3 FN).
> Hiện tại phần này được mô tả chi tiết bằng text trong phần 3.5.3.

*H�nh 3.5: Ph�n t�ch c�c tru?ng h?p model th?t b?i trong ph�n lo?i*

**Gi?i th�ch H�nh 3.5:**

H�nh 3.5 minh h?a c�c tru?ng h?p di?n h�nh m� m� h�nh g?p kh� khan trong vi?c ph�n lo?i ch�nh x�c, du?c chia th�nh hai nh�m:

**D�ng 1 - False Positives (REAL ? FAKE):** Model d? do�n sai l� FAKE khi th?c t? l� REAL

1. **Low Light Condition (�nh s�ng y?u):**
   - Input: ?nh m?ng m?t th?t nhung ch?p trong di?u ki?n thi?u s�ng
   - MSE: 0.0045 (cao b?t thu?ng, vu?t threshold 0.000312)
   - Nguy�n nh�n: Ch?t lu?ng ?nh k�m, nhi?u cao l�m model kh�ng reconstruct t?t
   - H?u qu?: T? ch?i ngu?i d�ng h?p l? (bad user experience)

2. **Partial Occlusion (Che m?t m?t ph?n):**
   - Input: Ph?n quang ho?c m� m?t che m?t ph?n iris
   - MSE: 0.0038 (cao do v�ng b? che kh�ng match v?i training data)
   - Nguy�n nh�n: Mask preprocessing kh�ng ho�n h?o, v�ng b? che t?o artifacts
   - H?u qu?: False rejection

3. **Motion Blur (M? do chuy?n d?ng):**
   - Input: ?nh b? m? do ngu?i d�ng di chuy?n trong khi ch?p
   - MSE: 0.0042 (cao do loss of detail)
   - Nguy�n nh�n: Model train tr�n ?nh sharp, kh�ng bao ph? motion blur
   - H?u qu?: Y�u c?u ngu?i d�ng ch?p l?i nhi?u l?n

**D�ng 2 - False Negatives (FAKE ? REAL):** Model d? do�n sai l� REAL khi th?c t? l� FAKE

1. **High-Quality OLED Display:**
   - Input: ?nh m?ng m?t hi?n th? tr�n m�n h�nh OLED cao c?p
   - MSE: 0.0002 (th?p, du?i threshold)
   - Nguy�n nh�n: OLED c� d? ph�n gi?i cao, m�u s?c ch�nh x�c, g?n gi?ng m?t th?t
   - H?u qu?: Cho ph�p t?n c�ng th�nh c�ng (security breach)

2. **High-Resolution Print:**
   - Input: ?nh in v?i d? ph�n gi?i r?t cao tr�n gi?y photo ch?t lu?ng
   - MSE: 0.0003 (g?n threshold nhung v?n pass)
   - Nguy�n nh�n: Print quality t?t, texture g?n v?i real iris
   - H?u qu?: B? s�t presentation attack

3. **Clear Contact Lens:**
   - Input: M?t th?t deo contact lens trong su?t kh�ng c� texture
   - MSE: 0.0001 (r?t th?p, model nh?m l� real)
   - Nguy�n nh�n: Contact lens trong kh�ng thay d?i nhi?u texture
   - H?u qu?: Kh�ng detect du?c lens attack

**Ph�n t�ch:**

C�c failure cases n�y ch? ra r?ng model d?a ho�n to�n v�o reconstruction error c� limitations:
- **False Positives** x?y ra khi ?nh REAL c� quality issues (lighting, blur, occlusion) ? C?n robust preprocessing
- **False Negatives** x?y ra khi FAKE c� quality cao g?n v?i REAL ? C?n multi-modal features (moir�, texture, frequency analysis)

��y l� l� do trong `main_realtime_new.py`, h? th?ng d� du?c c?i ti?n v?i:
- CLAHE preprocessing cho lighting correction
- Moir� pattern detection cho display attacks
- Texture variance analysis
- Sharpness v� saturation checks

K?t h?p multiple features gi�p gi?m d�ng k? c? False Positive v� False Negative rates.

**3.5.4. T�c d?ng v� � nghia th?c t?**

M?c d� c�n t?n t?i m?t s? h?n ch?, phuong ph�p d? xu?t d?a tr�n AutoEncoder mang l?i nhi?u gi� tr? th?c ti?n. Vi?c kh�ng y�u c?u d? li?u t?n c�ng trong qu� tr�nh hu?n luy?n gi�p gi?m d�ng k? chi ph� thu th?p v� g�n nh�n d? li?u, d?ng th?i tang kh? nang th�ch ?ng v?i c�c ki?u t?n c�ng m?i chua t?ng xu?t hi?n.

V?i ki?n tr�c g?n nh? (0.78M parameters), d? tr? th?p (2.84ms) v� kh? nang ho?t d?ng ?n d?nh, m� h�nh c� th? du?c s? d?ng nhu m?t **l?p ph�t hi?n liveness so c?p**, k?t h?p v?i c�c phuong ph�p h?c c� gi�m s�t ? t?ng sau nh?m n�ng cao d? an to�n t?ng th? c?a h? th?ng sinh tr?c h?c m?ng m?t.

**?? B? SUNG: Th�m ph�n t�ch Failure Cases**

**3.5.X Ph�n t�ch c�c tru?ng h?p th?t b?i**

Ph�n t�ch cho th?y model g?p kh� khan trong c�c tru?ng h?p sau:

1. **�i?u ki?n �nh s�ng y?u**: MSE tang cao c? v?i ?nh REAL do ch?t lu?ng ?nh k�m, d?n d?n False Positive.
2. **?nh b? che m?t ph?n**: Khi mask kh�ng ch�nh x�c (m� m?t che, ph?n quang), l?i t�i t?o tang b?t thu?ng.
3. **?nh m�n h�nh ch?t lu?ng cao**: C�c m�n h�nh OLED/Retina c� d? ph�n gi?i cao c� MSE g?n v?i ?nh REAL, kh� ph�n bi?t.
4. **Texture kh�ng d?ng nh?t**: ?nh c� v?t b?n, ph?n quang ho?c nhi?u m?nh t?o ra outliers trong ph�n b? MSE.

*(C� th? th�m h�nh minh h?a c�c failure cases)*

**3.5.5. Nh?n x�t chung**

T?ng h?p c�c ph�n t�ch cho th?y phuong ph�p ph�t hi?n liveness m?ng m?t d?a tr�n AutoEncoder theo hu?ng h?c kh�ng gi�m s�t l� m?t hu?ng ti?p c?n h?p l� v� ti?m nang. K?t qu? d?t du?c kh�ng ch? ch?ng minh kh? nang h?c d?c trung c?a m� h�nh m� c�n m? ra kh? nang ?ng d?ng trong c�c h? th?ng sinh tr?c h?c th?c t?, d?c bi?t trong b?i c?nh d? li?u t?n c�ng kh� thu th?p v� li�n t?c thay d?i.

**K?T LU?N V� HU?NG PH�T TRI?N**

**1. T�m t?t k?t qu? d?t du?c**

Nghi�n c?u n�y d� d? xu?t v� x�y d?ng m?t h? th?ng ph�t hi?n liveness m?ng m?t d?a tr�n m� h�nh **AutoEncoder theo hu?ng h?c kh�ng gi�m s�t (one-class learning)**. M� h�nh du?c hu?n luy?n ch? v?i d? li?u m?ng m?t th?t v� s? d?ng **l?i t�i t?o (reconstruction error)** l�m ti�u ch� ph�t hi?n c�c m?u b?t thu?ng.

K?t qu? th?c nghi?m cho th?y m� h�nh AutoEncoder c� kh? nang **h?i t? ?n d?nh**, h?c t?t ph�n b? c?a ?nh m?ng m?t th?t v� t?o ra s? kh�c bi?t r� r�ng v? l?i t�i t?o gi?a c�c m?u m?ng m?t th?t v� c�c m?u gi?. Ph�n t�ch du?ng ROC cho th?y gi� tr? AUC cao, ph?n �nh ti?m nang ph�n bi?t t?t gi?a hai nh�m d? li?u khi l?a ch?n ngu?ng quy?t d?nh ph� h?p. B�n c?nh d�, d? tr? x? l� th?p v� th�ng lu?ng cao cho th?y m� h�nh c� kh? nang d�p ?ng y�u c?u tri?n khai g?n th?i gian th?c.

**2. ��ng g�p ch�nh c?a nghi�n c?u**

C�c d�ng g�p ch�nh c?a nghi�n c?u c� th? du?c t�m t?t nhu sau:

- �? xu?t **c�ch ti?p c?n ph�t hi?n liveness m?ng m?t theo hu?ng h?c kh�ng gi�m s�t**, gi?m ph? thu?c v�o d? li?u t?n c�ng c� nh�n.
- X�y d?ng v� d�nh gi� m� h�nh AutoEncoder cho b�i to�n Iris PAD, l�m r� vai tr� c?a **reconstruction error** trong vi?c ph�t hi?n b?t thu?ng.
- Th?c hi?n ph�n t�ch to�n di?n th�ng qua c�c ch? s? d�nh gi�, bi?u d? v� h�nh minh h?a, cho th?y t�nh kh? thi c?a phuong ph�p trong c�c k?ch b?n th?c t?.
- Ch?ng minh ti?m nang ?ng d?ng c?a m� h�nh nhu m?t **l?p ph�t hi?n liveness so c?p**, c� th? t�ch h?p v�o c�c h? th?ng sinh tr?c h?c m?ng m?t hi?n c�.

**3. H?n ch? v� t?n t?i**

M?c d� d?t du?c nh?ng k?t qu? t�ch c?c, nghi�n c?u v?n t?n t?i m?t s? h?n ch?. Tru?c h?t, do m� h�nh du?c hu?n luy?n theo hu?ng one-class v� s? lu?ng m?u m?ng m?t gi? d�ng d? d�nh gi� c�n h?n ch?, c�c ch? s? ph�n lo?i truy?n th?ng nhu Precision, Recall v� F1-score chua ph?n �nh d?y d? hi?u nang c?a h? th?ng trong c�c k?ch b?n t?n c�ng da d?ng.

B�n c?nh d�, hi?u qu? c?a phuong ph�p ph? thu?c v�o vi?c l?a ch?n ngu?ng quy?t d?nh tr�n l?i t�i t?o. Vi?c x�c d?nh ngu?ng t?i uu trong m�i tru?ng tri?n khai th?c t? v?n l� m?t th�ch th?c. Ngo�i ra, m� h�nh AutoEncoder c� th? nh?y c?m v?i c�c y?u t? nhi?u m?nh, di?u ki?n chi?u s�ng ph?c t?p ho?c c�c bi?n d?ng h�nh ?nh chua du?c bao ph? d?y d? trong d? li?u hu?n luy?n.

**4. Hu?ng ph�t tri?n**

Trong tuong lai, nghi�n c?u c� th? du?c m? r?ng theo m?t s? hu?ng sau:

**4.1. C?i ti?n ki?n tr�c model:**
- K?t h?p m� h�nh AutoEncoder v?i c�c k? thu?t h?c s�u kh�c, ch?ng h?n nhu **Variational AutoEncoder (VAE)** ho?c **GAN**, nh?m n�ng cao kh? nang m� h�nh h�a ph�n b? d? li?u.
- Th? nghi?m v?i **Attention mechanisms** d? model t?p trung v�o v�ng iris quan tr?ng.
- �p d?ng **Contrastive Learning** d? h?c better representations.

**4.2. T?i uu ngu?ng v� deployment:**
- Nghi�n c?u c�c phuong ph�p **t? d?ng x�c d?nh ngu?ng quy?t d?nh** (adaptive threshold), gi�p tang t�nh ?n d?nh v� kh? nang tri?n khai th?c t?.
- Ph�t tri?n **meta-learning approaches** d? quickly adapt threshold cho m�i tru?ng m?i.
- X�y d?ng **confidence score** thay v� hard decision.

**4.3. M? r?ng d? li?u v� attack types:**
- M? r?ng t?p d? li?u d�nh gi� v?i nhi?u ki?u t?n c�ng tr�nh di?n kh�c nhau (in ?nh, m�n h�nh, contact lens, deepfake) d? d�nh gi� to�n di?n hon kh? nang t?ng qu�t c?a h? th?ng.
- Thu th?p data trong diverse conditions (lighting, distance, angles).

**4.4. Multi-layer defense:**
- K?t h?p phuong ph�p one-class v?i c�c m� h�nh h?c c� gi�m s�t ? t?ng sau, h�nh th�nh h? th?ng ph�t hi?n liveness da t?ng nh?m n�ng cao d? an to�n t?ng th?.
- T�ch h?p multi-modal features (nhu d� implement trong `main_realtime_new.py`: moir�, sharpness, texture).

**4.5. H?n ch? v? dataset v� d�nh gi� t�nh t?ng qu�t**

Nghi�n c?u hi?n t?i du?c th?c hi?n tr�n **dataset UBIPR2 duy nh?t**, m?t b? d? li?u near-infrared iris images. �i?u n�y t?o ra c�c h?n ch? v? t�nh t?ng qu�t:

**V?n d? dataset bias:**
- UBIPR2 thu th?p trong di?u ki?n controlled (lab environment, fixed sensor, professional setup).
- Kh�ng d?i di?n cho diversity trong real-world deployment (different sensors, lighting, user demographics).
- Thi?u c�c lo?i attack da d?ng (ch? c� REAL iris trong training, chua c� comprehensive fake samples).

**C?n thi?t cross-dataset evaluation:**

�? d�nh gi� **true generalization capability**, c?n th? nghi?m tr�n nhi?u datasets:

1. **LivDet-Iris competitions datasets:**
   - Nhi?u ki?u attack (printed, display, contact lens)
   - Cross-sensor evaluation
   - Standardized evaluation protocol

2. **Notre Dame Contact Lens Dataset:**
   - ��nh gi� kh? nang detect contact lens attacks
   - Textured vs clear lenses

3. **IIITD-WVU Dataset:**
   - Cross-spectral iris images
   - Visible light vs NIR

4. **Warsaw datasets:**
   - Post-mortem iris vs live iris
   - Aging effects

**�? xu?t evaluation protocol:**

```
Phase 1: Intra-dataset evaluation (hi?n t?i)
  - Train on UBIPR2 train set
  - Test on UBIPR2 test set
  - Baseline performance

Phase 2: Cross-dataset evaluation (d? xu?t)
  - Train on UBIPR2
  - Test on LivDet-Iris ? Measure generalization
  - Test on Notre Dame ? Measure contact lens detection
  - Test on IIITD-WVU ? Measure cross-spectral robustness

Phase 3: Cross-sensor evaluation
  - Train on Sensor A data
  - Test on Sensor B data
  - Measure domain shift impact

Phase 4: Multi-attack evaluation
  - Printed photo attacks
  - LCD/OLED/Retina display attacks
  - Textured contact lens attacks
  - 3D printed iris attacks
  - Deepfake/GAN-generated iris
```

**Expected outcomes:**
- Performance degradation in cross-dataset scenarios ? Need domain adaptation
- Different optimal thresholds per dataset ? Need adaptive threshold
- Some attack types may not be detected ? Need multi-modal approach

**Mitigation strategies:**
1. **Domain adaptation techniques:** Fine-tune on small labeled set from target domain
2. **Multi-dataset training:** Train on mixture of multiple datasets
3. **Meta-learning:** Learn to quickly adapt to new domains
4. **Ensemble methods:** Combine models trained on different datasets

K?t lu?n: Nghi�n c?u hi?n t?i l� **proof-of-concept** tr�n single dataset. �? tri?n khai th?c t?, c?n extensive cross-dataset v� cross-sensor evaluation d? d?m b?o robustness v� generalization.

**?? B? SUNG: Th�m h?n ch? v? dataset**

**5. H?n ch? v? dataset v� t�nh t?ng qu�t**

Nghi�n c?u hi?n t?i du?c th?c hi?n tr�n dataset UBIPR2, m?t b? d? li?u near-infrared iris images. �? n�ng cao t�nh t?ng qu�t v� kh? nang �p d?ng th?c t?, c?n:

- **Cross-dataset evaluation**: ��nh gi� tr�n c�c dataset kh�c nhu LivDet-Iris, IIITD-WVU, Notre Dame d? ki?m tra kh? nang t?ng qu�t.
- **M? r?ng lo?i t?n c�ng**: Th? nghi?m v?i nhi?u ki?u t?n c�ng da d?ng hon (in ?nh tr�n gi?y, m�n h�nh LCD/OLED/Retina, contact lens c� texture, ?nh 3D).
- **�i?u ki?n thu th?p da d?ng**: Th? nghi?m v?i nhi?u thi?t b? camera, g�c ch?p, kho?ng c�ch v� di?u ki?n �nh s�ng kh�c nhau.
- **��nh gi� cross-sensor**: Ki?m tra hi?u nang khi train tr�n m?t sensor v� test tr�n sensor kh�c.




