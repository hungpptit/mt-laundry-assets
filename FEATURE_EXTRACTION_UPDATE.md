# 4.2.2. Trích xuất Đặc trưng (Features) cho Naive Bayes - Phiên bản 2.0

## Tổng quan

Hệ thống sử dụng **HYBRID STRATEGY** với 2 loại model khác nhau:
- **Global Model** (weak_skill_model.pkl): Dùng khi user có ít data (attempts < 10)
- **Unified Model** (unified_model.pkl): Dùng khi user có đủ data (attempts ≥ 10)

Mỗi model có **feature set riêng** và **labeling rule riêng**.

---

## 1. Global Model (Weak Skill Model)

### 1.1 Trích xuất Đặc trưng (3 features)

Từ bảng `UserResults` và `QuestionSkills`, tính toán các đặc trưng như sau:

| Feature | Mô tả | Cách tính |
|---------|-------|----------|
| **attempts** | Số lần làm skill này | COUNT(*) từ UserResults grouped by skillId |
| **correct** | Số câu đúng skill này | SUM(isCorrect=1) grouped by skillId |
| **accuracy** | Tỷ lệ đúng = correct / attempts | Tính từ correct và attempts |

### 1.2 Gán Nhãn (Label - isWeak)

Rule cứng (fixed threshold):

```
isWeak = 1 nếu accuracy < 0.60 (dự đoán KỸ NĂNG YẾU)
isWeak = 0 nếu accuracy >= 0.60 (dự đoán KỸ NĂNG MẠNH)
```

**Lý do**: Khi user có ít data, không thể dùng rule cá nhân hoá → dùng threshold chung cho tất cả users.

### 1.3 Ví dụ Dữ liệu

| userId | skillName | attempts | correct | accuracy | isWeak |
|--------|-----------|----------|---------|----------|--------|
| 3 | Reading | 238 | 210 | 0.88 | 0 |
| 6 | Grammar | 25 | 12 | 0.48 | 1 |
| 6 | Vocabulary | 18 | 15 | 0.83 | 0 |
| 12 | Listening | 48 | 11 | 0.23 | 1 |
| 17 | Writing | 34 | 21 | 0.62 | 0 |

---

## 2. Unified Model

### 2.1 Trích xuất Đặc trưng (7 features)

Từ bảng `UserResults`, `QuestionSkills`, và `Skills`, tính toán các đặc trưng như sau:

#### 2.1.1 User Context Features (5 features)

| Feature | Mô tả | Cách tính |
|---------|-------|----------|
| **user_level** | Trình độ người dùng | `0 = Beginner (overall_accuracy < 0.5)` <br/> `1 = Intermediate (0.5 ≤ overall_accuracy < 0.7)` <br/> `2 = Advanced (overall_accuracy ≥ 0.7)` |
| **total_tests** | Tổng số bài test đã làm | COUNT(DISTINCT userTestId) từ UserResults |
| **total_questions** | Tổng số câu hỏi đã làm | COUNT(*) từ UserResults |
| **overall_accuracy** | Accuracy tổng quát của user | SUM(isCorrect=1) / COUNT(*) từ UserResults |
| **days_active** | Số ngày kể từ lần đầu làm bài | DATEDIFF(DAY, MIN(answeredAt), GETDATE()) |

#### 2.1.2 Skill Context Features (2 features)

| Feature | Mô tả | Cách tính |
|---------|-------|----------|
| **attempts** | Số lần làm skill này | COUNT(*) grouped by (userId, skillId) |
| **correct** | Số câu đúng skill này | SUM(isCorrect=1) grouped by (userId, skillId) |

#### 2.1.3 Lưu ý quan trọng

⚠️ **skill_accuracy** (correct/attempts) **KHÔNG** đưa vào feature vector!
- Dùng **CHỈ ĐỂ TÍNH NHÃN (isWeak)**
- Lý do: Tránh data leakage (model không nên biết kết quả trước khi predict)

### 2.2 Gán Nhãn (Label - isWeak)

Rule **động** (personalized per user) với fallback:

```python
# Bước 1: Kiểm tra có đủ dữ liệu để dùng rule cá nhân hoá?
nếu (attempts >= 5) AND (user_num_skills >= 3) AND (std_user_skill_acc > 0):
    # Bước 2: Tính ngưỡng cá nhân hoá
    dynamic_threshold = avg_user_skill_acc - 1.0 * std_user_skill_acc
    isWeak = 1 nếu skill_accuracy < dynamic_threshold
    isWeak = 0 nếu skill_accuracy >= dynamic_threshold
ngược lại:
    # Bước 3: Fallback về rule cứng
    isWeak = 1 nếu skill_accuracy < 0.60
    isWeak = 0 nếu skill_accuracy >= 0.60
```

**Giải thích**:
- `avg_user_skill_acc`: Trung bình accuracy của user trên **tất cả skills**
- `std_user_skill_acc`: Độ lệch chuẩn accuracy của user trên **tất cả skills**
- `k_std = 1.0`: Tính ngưỡng = TB - 1.0 * ĐLC (nếu thấp hơn 1 độ lệch chuẩn từ TB = yếu)

**Ví dụ cá nhân hoá**:
- User A: avg_acc = 0.70, std = 0.15
  - Threshold = 0.70 - 1.0 × 0.15 = **0.55**
  - Nếu skill_acc = 0.60 → **STRONG** (trên ngưỡng cá nhân)
  
- User B: avg_acc = 0.80, std = 0.05
  - Threshold = 0.80 - 1.0 × 0.05 = **0.75**
  - Nếu skill_acc = 0.60 → **WEAK** (dưới ngưỡng cá nhân)

### 2.3 Ví dụ Dữ liệu

| userId | skillId | skillName | user_level | total_tests | total_questions | overall_accuracy | days_active | attempts | correct | skill_accuracy | isWeak |
|--------|---------|-----------|-----------|-------------|-----------------|------------------|-------------|----------|---------|----------------|--------|
| 3 | 4 | Reading | 2 | 30 | 850 | 0.84 | 180 | 238 | 210 | 0.88 | 0 |
| 6 | 2 | Grammar | 1 | 4 | 120 | 0.65 | 90 | 25 | 12 | 0.48 | 1 |
| 6 | 3 | Vocabulary | 1 | 4 | 120 | 0.65 | 90 | 18 | 15 | 0.83 | 0 |
| 12 | 6 | Listening | 0 | 2 | 95 | 0.40 | 45 | 48 | 11 | 0.23 | 1 |
| 17 | 1 | Writing | 1 | 7 | 210 | 0.62 | 120 | 34 | 21 | 0.62 | 0 |

---

## 3. So sánh 2 loại Model

| Tiêu chí | Global Model | Unified Model |
|---------|--------------|----------------|
| **Số features** | 3 | 7 |
| **Khi nào dùng** | attempts < 10 | attempts ≥ 10 |
| **Labeling** | Rule cứng (threshold = 0.60) | Rule động (personalized) + fallback |
| **Personalization** | Thấp (chỉ dùng skill data) | Cao (bao gồm user context) |
| **Kích cỡ model file** | Nhỏ | Nhỏ (1 file dùng chung cho 10k users) |
| **Retrain time** | 1 phút | 2-3 phút |

---

## 4. Hybrid Strategy trong Prediction

```
IF attempts < 10:
    → Load global_model.pkl
    → Input: [attempts, correct, accuracy]
    → Output: isWeak ∈ {0, 1}
ELSE:
    → Load unified_model.pkl
    → Input: [user_level, total_tests, total_questions, 
              overall_accuracy, days_active, attempts, correct]
    → Output: isWeak ∈ {0, 1}
```

**Ưu điểm**:
- ✅ User mới (ít data): Dùng global model, dự đoán nhanh
- ✅ User cũ (đủ data): Dùng unified model, cá nhân hoá cao
- ✅ Scalable: 1 unified model cho 10k users (thay vì 10k personal models)
- ✅ Fast retrain: 2-3 phút thay vì 14 giờ

---

## 5. Query SQL để Extract Features

### 5.1 Global Model Query

```sql
SELECT 
    ur.userId,
    qs.skillId,
    s.name AS skillName,
    COUNT(*) AS attempts,
    SUM(CASE WHEN ur.isCorrect = 1 THEN 1 ELSE 0 END) AS correct
FROM UserResults ur
JOIN QuestionSkills qs ON ur.questionId = qs.questionId
JOIN Skills s ON qs.skillId = s.id
WHERE ur.userId IS NOT NULL
GROUP BY ur.userId, qs.skillId, s.name
```

### 5.2 Unified Model Query

```sql
WITH UserStats AS (
    SELECT 
        ur.userId,
        COUNT(DISTINCT ur.userTestId) AS total_tests,
        COUNT(*) AS total_questions,
        CAST(SUM(CASE WHEN ur.isCorrect = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) AS overall_accuracy,
        DATEDIFF(DAY, MIN(ur.answeredAt), GETDATE()) AS days_active
    FROM UserResults ur
    WHERE ur.userId IS NOT NULL
    GROUP BY ur.userId
),
SkillStats AS (
    SELECT 
        ur.userId,
        qs.skillId,
        COUNT(*) AS attempts,
        SUM(CASE WHEN ur.isCorrect = 1 THEN 1 ELSE 0 END) AS correct,
        CAST(SUM(CASE WHEN ur.isCorrect = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) AS skill_accuracy
    FROM UserResults ur
    JOIN QuestionSkills qs ON ur.questionId = qs.questionId
    WHERE ur.userId IS NOT NULL
    GROUP BY ur.userId, qs.skillId
)
SELECT 
    ss.userId,
    ss.skillId,
    us.total_tests,
    us.total_questions,
    us.overall_accuracy,
    us.days_active,
    ss.attempts,
    ss.correct,
    ss.skill_accuracy
FROM SkillStats ss
JOIN UserStats us ON ss.userId = us.userId
```

---

## 6. File liên quan

| File | Mục đích |
|------|---------|
| `train_model.py` | Train global model (weak_skill_model.pkl) |
| `train_unified_model.py` | Train unified model (unified_model.pkl) |
| `predict_hybrid_unified.py` | Dự đoán sử dụng hybrid strategy |
| `mlRetrainCron.js` | Tự động retrain model mỗi 6 giờ |
