# 🎯 CV-JD Matching API - Hướng dẫn Cài đặt & Tích hợp Dart/Flutter

## 📋 Mục lục

1. [Cài đặt & Chạy Server](#1-cài-đặt--chạy-server)
2. [Tích hợp Dart/Flutter](#2-tích-hợp-dartflutter)
3. [API Endpoints](#3-api-endpoints)
4. [Troubleshooting](#4-troubleshooting)

---

## 1. Cài đặt & Chạy Server

### 📁 Cấu trúc Project

```
52200142_DaoThuyBaoHan_MatchingJD/
├── app.py                      # FastAPI server (Hybrid: 70% BERT + 30% TF-IDF)
├── app_bert_only.py            # FastAPI server (100% BERT, NO TF-IDF)
├── download_model.py           # Script tải model về local
├── requirements.txt            # Dependencies
├── occupations_en.csv          # ESCO data (3,039 occupations)
├── esco_embeddings.npy         # ESCO embeddings (cached)
├── models/                     # Sentence-BERT model (local)
│   └── all-MiniLM-L6-v2/
└── data/                       # CV datasets
    ├── INFORMATION-TECHNOLOGY/
    ├── SALES/
    ├── HR/
    └── ...
```

### 🚀 Bước 1: Cài đặt Dependencies

```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 🚀 Bước 2: Tải Model về Local

```bash
python download_model.py
```

**Output:**
```
✅ SUCCESS! Model downloaded and saved locally!
📁 Location: D:\...\models\all-MiniLM-L6-v2
```

### 🚀 Bước 3: Chạy Server

**Option 1: Hybrid Server (70% BERT + 30% TF-IDF)**
```bash
python app.py
```
- **Server:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **Scoring:** 70% semantic (BERT) + 30% lexical (TF-IDF)

**Option 2: Pure AI Server (100% BERT - Recommended)**
```bash
python app_bert_only.py
```
- **Server:** http://localhost:8002
- **Swagger UI:** http://localhost:8002/docs
- **Scoring:** 100% semantic (BERT AI model)
- **Features:**
  - ✅ CV field analysis (email, phone, education, experience, skills)
  - ✅ Completeness checking
  - ✅ Debug endpoints for testing

> **💡 Khuyến nghị:** Dùng `app_bert_only.py` cho kết quả AI chính xác hơn!

---

## 2. Tích hợp Dart/Flutter

### Bước 1: Thêm Dependencies

File `pubspec.yaml`:

```yaml
dependencies:
  http: ^1.1.0
  file_picker: ^6.1.1
```

Chạy:
```bash
flutter pub get
```

### Bước 2: Tạo Data Models

Tạo file `lib/models/cv_match_models.dart`:

```dart
/// Model kết quả matching
class CVMatchResult {
  final String status;
  final String jdSummary;
  final int totalCvsUploaded;
  final int totalCvsProcessed;
  final List<String>? failedCvs;
  final List<TopMatch> topMatches;
  final String timestamp;

  const CVMatchResult({
    required this.status,
    required this.jdSummary,
    required this.totalCvsUploaded,
    required this.totalCvsProcessed,
    this.failedCvs,
    required this.topMatches,
    required this.timestamp,
  });

  factory CVMatchResult.fromJson(Map<String, dynamic> json) {
    return CVMatchResult(
      status: json['status'] as String,
      jdSummary: json['jd_summary'] as String,
      totalCvsUploaded: json['total_cvs_uploaded'] as int,
      totalCvsProcessed: json['total_cvs_processed'] as int,
      failedCvs: json['failed_cvs'] != null 
          ? List<String>.from(json['failed_cvs'] as List) 
          : null,
      topMatches: (json['top_matches'] as List)
          .map((match) => TopMatch.fromJson(match as Map<String, dynamic>))
          .toList(),
      timestamp: json['timestamp'] as String,
    );
  }
}

/// Model cho từng CV match
class TopMatch {
  final int rank;
  final String cvName;
  final double score;
  final double bertScore;  // Đổi từ baseScore -> bertScore
  final double escoBonus;
  final double matchPercentage;
  final String category;
  final int cvIndex;
  final FieldAnalysis? fieldAnalysis;  // Thêm field analysis

  const TopMatch({
    required this.rank,
    required this.cvName,
    required this.score,
    required this.bertScore,
    required this.escoBonus,
    required this.matchPercentage,
    required this.category,
    required this.cvIndex,
    this.fieldAnalysis,
  });

  factory TopMatch.fromJson(Map<String, dynamic> json) {
    return TopMatch(
      rank: json['rank'] as int,
      cvName: json['cv_name'] as String,
      score: (json['score'] as num).toDouble(),
      bertScore: (json['bert_score'] as num).toDouble(),  // Đổi key
      escoBonus: (json['esco_bonus'] as num).toDouble(),
      matchPercentage: (json['match_percentage'] as num).toDouble(),
      category: json['category'] as String,
      cvIndex: json['cv_index'] as int,
      fieldAnalysis: json['field_analysis'] != null
          ? FieldAnalysis.fromJson(json['field_analysis'] as Map<String, dynamic>)
          : null,
    );
  }

  // Helper getters
  bool get isExcellent => matchPercentage >= 80;
  bool get isGood => matchPercentage >= 70;
  bool get isFair => matchPercentage >= 60;
}

/// Model phân tích CV fields
class FieldAnalysis {
  final double completeness;
  final List<String> missingFields;
  final int filledFields;
  final int totalFields;

  const FieldAnalysis({
    required this.completeness,
    required this.missingFields,
    required this.filledFields,
    required this.totalFields,
  });

  factory FieldAnalysis.fromJson(Map<String, dynamic> json) {
    return FieldAnalysis(
      completeness: (json['completeness'] as num).toDouble(),
      missingFields: List<String>.from(json['missing_fields'] as List),
      filledFields: json['filled_fields'] as int,
      totalFields: json['total_fields'] as int,
    );
  }

  // Helper getters
  bool get isComplete => completeness >= 90;
  bool get needsImprovement => completeness < 70;
}
```

### Bước 3: Tạo API Service

Tạo file `lib/services/cv_matching_api_service.dart`:

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/cv_match_models.dart';

class CVMatchingApiService {
  final String baseUrl;

  CVMatchingApiService({
    this.baseUrl = 'http://localhost:8002',  // Port 8002 cho BERT-only server
  });

  /// Match nhiều CV với Job Description
  Future<CVMatchResult> matchCVs({
    required String jdText,
    required List<File> cvFiles,
    int topN = 5,
  }) async {
    try {
      final url = Uri.parse('$baseUrl/match');
      var request = http.MultipartRequest('POST', url);

      request.fields['jd_text'] = jdText;
      request.fields['top_n'] = topN.toString();

      for (var file in cvFiles) {
        final fileName = file.path.split(Platform.pathSeparator).last;
        request.files.add(
          await http.MultipartFile.fromPath(
            'cv_files',
            file.path,
            filename: fileName,
          ),
        );
      }

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body) as Map<String, dynamic>;
        return CVMatchResult.fromJson(jsonData);
      } else {
        throw Exception('API Error ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      print('❌ Error in matchCVs: $e');
      rethrow;
    }
  }

  /// Tính điểm cho 1 CV đơn lẻ
  Future<Map<String, dynamic>> scoreSingleCV({
    required String jdText,
    required File cvFile,
  }) async {
    try {
      final url = Uri.parse('$baseUrl/score-single');
      var request = http.MultipartRequest('POST', url);

      request.fields['jd_text'] = jdText;
      final fileName = cvFile.path.split(Platform.pathSeparator).last;
      request.files.add(
        await http.MultipartFile.fromPath(
          'cv_file',
          cvFile.path,
          filename: fileName,
        ),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        throw Exception('API Error ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      print('❌ Error in scoreSingleCV: $e');
      rethrow;
    }
  }

  /// Kiểm tra health của server
  Future<bool> checkHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/'),
      ).timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// Lấy thống kê server
  Future<Map<String, dynamic>> getStats() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/stats'),
      ).timeout(const Duration(seconds: 5));
      
      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        throw Exception('Failed to get stats: ${response.statusCode}');
      }
    } catch (e) {
      print('❌ Error getting stats: $e');
      rethrow;
    }
  }
}
```

### Bước 4: Tạo Flutter UI

Tạo file `lib/screens/cv_matching_screen.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import '../services/cv_matching_api_service.dart';
import '../models/cv_match_models.dart';

class CVMatchingScreen extends StatefulWidget {
  const CVMatchingScreen({Key? key}) : super(key: key);

  @override
  State<CVMatchingScreen> createState() => _CVMatchingScreenState();
}

class _CVMatchingScreenState extends State<CVMatchingScreen> {
  final CVMatchingApiService _apiService = CVMatchingApiService(
    baseUrl: 'http://localhost:8002',  // Dùng port 8002 cho BERT-only
  );
  
  final TextEditingController _jdController = TextEditingController();
  List<File> _selectedCVs = [];
  CVMatchResult? _matchResult;
  bool _isLoading = false;

  Future<void> _pickCVFiles() async {
    try {
      FilePickerResult? pickerResult = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
        allowMultiple: true,
      );

      if (pickerResult != null) {
        setState(() {
          _selectedCVs = pickerResult.paths
              .where((path) => path != null)
              .map((path) => File(path!))
              .toList();
        });
        _showSnackBar('✅ Đã chọn ${_selectedCVs.length} file CV');
      }
    } catch (e) {
      _showSnackBar('❌ Lỗi khi chọn file: $e', isError: true);
    }
  }

  Future<void> _submitMatching() async {
    if (_jdController.text.trim().isEmpty) {
      _showSnackBar('⚠️ Vui lòng nhập Job Description', isError: true);
      return;
    }

    if (_selectedCVs.isEmpty) {
      _showSnackBar('⚠️ Vui lòng chọn ít nhất 1 file CV', isError: true);
      return;
    }

    setState(() {
      _isLoading = true;
      _matchResult = null;
    });

    try {
      final result = await _apiService.matchCVs(
        jdText: _jdController.text,
        cvFiles: _selectedCVs,
        topN: 5,
      );

      setState(() {
        _matchResult = result;
        _isLoading = false;
      });

      _showSnackBar('✅ Hoàn thành! Xử lý ${result.totalCvsProcessed} CVs');
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showSnackBar('❌ Lỗi: $e', isError: true);
    }
  }

  void _showSnackBar(String message, {bool isError = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Colors.red : Colors.green,
      ),
    );
  }

  Color _getScoreColor(double percentage) {
    if (percentage >= 80) return Colors.green;
    if (percentage >= 70) return Colors.lightGreen;
    if (percentage >= 60) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🎯 CV-JD Matching'),
        backgroundColor: Colors.blue,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Job Description Input
            const Text(
              '📝 Job Description',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _jdController,
              maxLines: 8,
              decoration: InputDecoration(
                hintText: 'Nhập mô tả công việc...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                filled: true,
                fillColor: Colors.grey[50],
              ),
            ),
            const SizedBox(height: 24),

            // CV File Picker
            Text(
              '📄 CV Files (${_selectedCVs.length} đã chọn)',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: _pickCVFiles,
              icon: const Icon(Icons.upload_file),
              label: const Text('Chọn file PDF'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(16),
                backgroundColor: Colors.blueGrey,
              ),
            ),
            
            // Hiển thị danh sách CV
            if (_selectedCVs.isNotEmpty) ...[
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _selectedCVs.map((file) {
                  final fileName = file.path.split(Platform.pathSeparator).last;
                  return Chip(
                    label: Text(fileName),
                    deleteIcon: const Icon(Icons.close, size: 18),
                    onDeleted: () {
                      setState(() {
                        _selectedCVs.remove(file);
                      });
                    },
                  );
                }).toList(),
              ),
            ],
            const SizedBox(height: 24),

            // Submit Button
            ElevatedButton(
              onPressed: _isLoading ? null : _submitMatching,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Colors.blue,
              ),
              child: _isLoading
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    )
                  : const Text(
                      '🚀 Bắt đầu Matching',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
            ),
            const SizedBox(height: 32),

            // Results
            if (_matchResult != null) ...[
              Text(
                '🎯 Top ${_matchResult!.topMatches.length} Matches',
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              ...List.generate(_matchResult!.topMatches.length, (index) {
                final match = _matchResult!.topMatches[index];
                return Card(
                  elevation: 2,
                  margin: const EdgeInsets.only(bottom: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: ListTile(
                    contentPadding: const EdgeInsets.all(16),
                    leading: CircleAvatar(
                      radius: 24,
                      backgroundColor: _getScoreColor(match.matchPercentage),
                      child: Text(
                        '#${match.rank}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    title: Text(
                      match.cvName,
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const SizedBox(height: 8),
                        Text(
                          'Score: ${match.matchPercentage.toStringAsFixed(1)}%',
                          style: TextStyle(
                            color: _getScoreColor(match.matchPercentage),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text('📂 ${match.category}'),
                        
                        // Field Analysis
                        if (match.fieldAnalysis != null) ...[
                          const SizedBox(height: 4),
                          Text(
                            '✅ Completeness: ${match.fieldAnalysis!.completeness.toStringAsFixed(0)}%',
                            style: TextStyle(
                              color: match.fieldAnalysis!.isComplete
                                  ? Colors.green
                                  : match.fieldAnalysis!.needsImprovement
                                      ? Colors.orange
                                      : Colors.blue,
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                            ),
                          ),
                          if (match.fieldAnalysis!.missingFields.isNotEmpty)
                            Text(
                              '⚠️ Missing: ${match.fieldAnalysis!.missingFields.take(2).join(", ")}${match.fieldAnalysis!.missingFields.length > 2 ? "..." : ""}',
                              style: TextStyle(
                                color: Colors.grey[600],
                                fontSize: 11,
                              ),
                            ),
                        ],
                        
                        if (match.escoBonus > 0)
                          Container(
                            margin: const EdgeInsets.only(top: 4),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: Colors.green[50],
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              '🎯 ESCO Bonus: +${match.escoBonus.toStringAsFixed(3)}',
                              style: TextStyle(
                                color: Colors.green[700],
                                fontWeight: FontWeight.bold,
                                fontSize: 12,
                              ),
                            ),
                          ),
                      ],
                    ),
                    trailing: Icon(
                      match.isExcellent
                          ? Icons.star
                          : match.isGood
                              ? Icons.star_half
                              : Icons.star_border,
                      color: _getScoreColor(match.matchPercentage),
                      size: 32,
                    ),
                  ),
                );
              }),
            ],
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _jdController.dispose();
    super.dispose();
  }
}
```

### Bước 5: Sử dụng trong App

```dart
import 'package:flutter/material.dart';
import 'screens/cv_matching_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CV-JD Matching',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const CVMatchingScreen(),
    );
  }
}
```

---

## 3. API Endpoints

### POST /match
Match nhiều CV với JD

**Request:**
```
POST http://localhost:8002/match
Content-Type: multipart/form-data

jd_text: "Job description text..."
cv_files: [file1.pdf, file2.pdf, ...]
top_n: 5
```

**Response:**
```json
{
  "status": "success",
  "scoring_method": "100% BERT (no TF-IDF)",
  "top_matches": [
    {
      "rank": 1,
      "cv_name": "john_doe.pdf",
      "score": 0.85,
      "bert_score": 0.82,
      "esco_bonus": 0.03,
      "match_percentage": 85.0,
      "category": "INFORMATION-TECHNOLOGY",
      "field_analysis": {
        "completeness": 86.7,
        "missing_fields": ["other.has_certifications", "other.has_languages"],
        "filled_fields": 13,
        "total_fields": 15
      }
    }
  ],
  "total_cvs_processed": 10
}
```

### POST /score-single
Score 1 CV với JD

**Request:**
```
POST http://localhost:8002/score-single
Content-Type: multipart/form-data

jd_text: "Job description..."
cv_file: single.pdf
```

### POST /analyze-cv
Phân tích CV fields (email, phone, education, experience, skills)

**Request:**
```
POST http://localhost:8002/analyze-cv
Content-Type: multipart/form-data

cv_file: resume.pdf
```

**Response:**
```json
{
  "status": "success",
  "cv_name": "resume.pdf",
  "analysis": {
    "completeness_percentage": 86.7,
    "filled_fields": 13,
    "total_fields": 15,
    "missing_fields": [
      "other.has_certifications",
      "other.has_languages"
    ],
    "fields": {
      "contact": {"email": true, "phone": true, "address": true},
      "education": {"has_education": true, "has_dates": true},
      "experience": {"has_experience": true, "has_dates": true, "has_responsibilities": true},
      "skills": {"has_skills": true, "has_technical": true, "has_soft": true},
      "other": {"has_summary": true, "has_certifications": false, "has_languages": false, "has_references": true}
    }
  }
}
```

### POST /debug-cv
Debug CV text extraction và regex matching

**Request:**
```
POST http://localhost:8002/debug-cv
Content-Type: multipart/form-data

cv_file: resume.pdf
```

### GET /stats
Thông tin server và model

```
GET http://localhost:8002/stats
```

**Response:**
```json
{
  "model": {
    "loaded": true,
    "name": "all-MiniLM-L6-v2",
    "type": "Sentence-BERT (BERT-based)",
    "scoring": "100% Semantic (NO TF-IDF)"
  },
  "esco": {
    "loaded": true,
    "occupations_count": 3039
  }
}
```

---

## 4. Troubleshooting

### ❌ Model không tải được

**Giải pháp:**
```bash
# Tải lại model
python download_model.py
```

### ❌ Lỗi "Connection refused" trong Flutter

**Nguyên nhân:** Dùng `localhost` trên emulator/device

**Giải pháp:**
```dart
// Android emulator
CVMatchingApiService(baseUrl: 'http://10.0.2.2:8002')

// iOS simulator
CVMatchingApiService(baseUrl: 'http://localhost:8002')

// Physical device (cùng WiFi)
CVMatchingApiService(baseUrl: 'http://192.168.1.100:8002')
```

### ❌ Port 8002 đã bị sử dụng

**Giải pháp 1:** Kill process đang dùng port
```powershell
# Windows
$p = (Get-NetTCPConnection -LocalPort 8002).OwningProcess
Stop-Process -Id $p -Force
```

**Giải pháp 2:** Đổi port trong `app_bert_only.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8003)
```

### ❌ CORS errors

Server đã enable CORS mặc định. Nếu vẫn lỗi, check `app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 System Requirements

**Python Server:**
- Python 3.8+
- RAM: 2GB+ (khuyến nghị 4GB)
- Disk: ~500MB (model + dependencies)

**Flutter App:**
- Flutter 3.0+
- Dart 2.17+

---

## 🎯 Tổng kết

✅ **Server:** FastAPI với Sentence-BERT AI model  
✅ **Scoring:** 100% BERT semantic matching (pure deep learning)  
✅ **Model:** Local storage - không cần internet  
✅ **Tốc độ:** ~1-2s cho 10 CVs  
✅ **CV Analysis:** Email, phone, education, experience, skills detection  
✅ **Documentation:** Swagger UI tại http://localhost:8002/docs  
✅ **Flutter:** Complete data models, API service, và UI example với field analysis  

**Chạy server (BERT-only - Recommended):**
```bash
python app_bert_only.py
```
- Port: 8002
- Model load từ: `./models/all-MiniLM-L6-v2/`
- Features: CV field analysis, completeness checking, debug tools

**Hoặc chạy Hybrid server (70% BERT + 30% TF-IDF):**
```bash
python app.py
```
- Port: 8000
- Scoring: 70% semantic + 30% lexical

**Test API:**
- Swagger UI: http://localhost:8002/docs (BERT-only)
- Swagger UI: http://localhost:8000/docs (Hybrid)
- Health check: http://localhost:8002/ hoặc http://localhost:8000/

**Tích hợp Flutter:** 
1. Copy các file Dart ở Bước 2-4 vào project Flutter
2. Update `baseUrl` thành `http://localhost:8002` (hoặc IP máy bạn)
3. Test với Swagger UI trước khi tích hợp

**Lưu ý:**
- ✅ Model đã được tải sẵn trong `./models/all-MiniLM-L6-v2/`
- ✅ Server tự động load model từ local (không cần internet)
- ✅ ESCO embeddings được cache trong `esco_embeddings.npy`
- ✅ CV field analysis hoạt động với raw text (detect email, phone, dates chính xác)
