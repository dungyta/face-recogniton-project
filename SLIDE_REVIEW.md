# 📋 Nhận Xét & Góp Ý Slide Workshop — Face Recognition

> **File slide được review:** `Recognitiônverview.pdf` (27 trang)  
> **Người review:** Antigravity AI Assistant  
> **Ngày:** 2026-03-16  
> **Mục đích:** Workshop nội bộ về triển khai Face Recognition trên Indoor Advance Camera (Edge AI)

---

## ⭐ Điểm Mạnh Của Slide Hiện Tại

### ✅ Cấu trúc luồng tư duy tốt
Slide dẫn dắt khá mạch lạc theo đúng logic:
> Đặt vấn đề → CNN overview → Recognition algorithms → Chọn model → Pipeline training

Người xem không quen sẽ dễ theo dõi.

### ✅ Đặt vấn đề Edge AI rõ ràng (Slide 3)
Các constraint rất cụ thể:
- RAM < 256MB
- Flash < 16MB
- Latency < 50ms
- Model size 1–5MB

Đây là điểm **anchor tốt** cho cả bài thuyết trình — mọi quyết định sau đều nên quay về justify với những con số này.

### ✅ Giải thích được lý do chọn Cosine Similarity (Slide 9)
Việc phân biệt **Classification vs Learning Similarity** ở slide 8-9 là insight quan trọng, thể hiện hiểu biết về bài toán one-shot learning.

### ✅ Giải thích Width Multiplier rõ (Slide 15)
Chi tiết về Depthwise Separable Convolution và empirical validation với 0.25/0.5/0.75/1.0 là rất thực tế.

---

## ⚠️ Điểm Yếu Cần Cải Thiện

### 🔴 1. Slide 4 — "General Pipeline Inference" không có nội dung text

> **Vấn đề:** Slide 4 chỉ có title, không có diagram hoặc description nào được capture. Nếu đây là hình ảnh embedded → cần đảm bảo hình rõ ràng và đủ labels.

**Đề xuất:** Đây phải là slide **quan trọng nhất** của bài vì nó là toàn cảnh pipeline. Nên có:

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Camera    │───▶│    Face     │───▶│     Face     │───▶│   Feature   │
│   Frame     │    │  Detection  │    │  Alignment   │    │ Extraction  │
│  (Raw YUV)  │    │ (BBox x,y)  │    │  (112×112)   │    │  (NPU .bin) │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
                                                                  │
                   ┌──────────────────────────────────────────────▼
                   │    Face Database   ◀── Cosine Similarity ──── Embedding
                   │   (embeddings)          (Match/No match)      Vector
                   └────────────────
```

### 🔴 2. Slide 13 — "Các Backbone của MobileFaceNet" chỉ có title

> **Vấn đề:** Không có text, chỉ hình ảnh. Cần đảm bảo hình này render được khi present, hoặc thêm text mô tả.

**Đề xuất:** Thêm bảng so sánh backbone vào slide này:

| Backbone             | Params | Model Size | Accuracy (LFW) | Phù hợp       |
| -------------------- | ------ | ---------- | -------------- | ------------- |
| MobileNetV1 (α=0.25) | ~0.5M  | ~2MB       | ~97%           | ✅ Edge device |
| MobileNetV1 (α=0.5)  | ~1.5M  | ~5MB       | ~98.5%         | ✅ Camera AI   |
| MobileNetV1 (α=1.0)  | ~4M    | ~15MB      | ~99%           | ⚠️ Hơi nặng    |
| MobileNetV2          | ~3.5M  | ~13MB      | ~99.2%         | ⚠️ Hơi nặng    |

### 🔴 3. Slide 16 đến 26 — Hoàn toàn trống / chỉ có hình

> **Vấn đề:** 11 slide cuối phần "Pipeline Training", "Forward Pass", "Backward Pass", "Learning Similarity" và các slide 19-26 đều không có text nào — chỉ dựa vào hình ảnh. Nếu audience xem lại slide không có presenter giải thích → **hoàn toàn mất thông tin**.

**Đề xuất:** Mỗi slide nên có **ít nhất 2-3 bullet points** tóm tắt nội dung, dù hình ảnh là chính.

### 🟡 4. Thiếu slide về **Quantization / NNE Convert**

> **Vấn đề:** Slide dừng lại ở training pipeline nhưng **không có phần nào về quá trình convert ONNX → NNE .bin** — trong khi đây là điểm khó nhất và đặc thù nhất của dự án so với các bài toán face recognition thông thường.

**Đề xuất:** Thêm ít nhất 1-2 slide về:
- Toolchain NNE: ONNX → Convert → Quantize (INT8) → Compile → `.bin`
- Tại sao cần quantize: model size, inference speed
- Thách thức: accuracy drop sau quantize, cách mitigate

### 🟡 5. Thiếu **kết quả thực tế / số liệu**

> **Vấn đề:** Bài thuyết trình cần thuyết phục đây là approach đúng, nhưng chưa có số liệu benchmark thực tế.

**Đề xuất:** Thêm slide "Kết quả đạt được" với:
- Model size trước/sau quantize (MB)
- Accuracy trên test set (%)
- Inference latency (ms)
- So sánh với constraint ban đầu (slide 3)

### 🟡 6. Slide THANK YOU quá sớm (slide 27)

> **Vấn đề:** Bài kết thúc đột ngột ở slide 27 mà không có phần **Q&A**, **Kế hoạch tiếp theo** hay **Demo**.

---

## 📐 Cấu Trúc Slide Đề Xuất (Revised)

```
[CHƯƠNG 1] ĐẶT VẤN ĐỀ (2-3 slides)
  ├── Bối cảnh: Edge AI Camera
  ├── Constraints: RAM/Flash/Latency/Size  ← Đã có ✅
  └── Mục tiêu dự án

[CHƯƠNG 2] TỔNG QUAN HỆ THỐNG (2-3 slides)
  ├── General Pipeline (Detection → Align → Recognize)  ← Cần làm rõ hơn
  ├── Face Detection: Input/Output, method
  └── Face Alignment: Tại sao cần, phương pháp

[CHƯƠNG 3] MODEL & TRAINING (4-5 slides)
  ├── CNN Overview  ← Đã có ✅
  ├── Recognition: Classification vs Similarity  ← Đã có ✅
  ├── Face Representation: Embedding space  ← Đã có ✅
  ├── Lựa chọn model: MobileFaceNet + Backbone comparison  ← Cần thêm bảng
  ├── Width Multiplier  ← Đã có ✅
  └── Training Pipeline: Forward/Backward/Loss  ← Có hình nhưng cần text

[CHƯƠNG 4] TRIỂN KHAI TRÊN CAMERA (3-4 slides)  ← THIẾU HOÀN TOÀN
  ├── NNE Toolchain: ONNX → .bin
  ├── Quantization: INT8, accuracy tradeoff
  ├── NNE API Flow: ak_npu_init → Create_CH → Run → Destroy
  └── Feature Database: Xây dựng & deploy

[CHƯƠNG 5] KẾT QUẢ & DEMO (2-3 slides)  ← THIẾU
  ├── Bảng so sánh: Model size, Accuracy, Latency
  ├── Demo video/screenshot (nếu có)
  └── Kế hoạch tiếp theo

[PHẦN KẾT]
  └── Q&A
```

---

## 🔧 Góp Ý Chi Tiết Từng Slide

| Slide | Nội dung hiện tại                | Đánh giá            | Góp ý cụ thể                                            |
| ----- | -------------------------------- | ------------------- | ------------------------------------------------------- |
| 3     | Đặt vấn đề Edge AI               | ✅ Tốt               | OK, giữ nguyên                                          |
| 4     | General Pipeline (hình)          | ⚠️ Thiếu label/text  | Thêm labels vào hình, hoặc thêm flowchart text          |
| 5     | Face Detection                   | ✅ Tốt               | Thêm ví dụ BBox coordinates cụ thể                      |
| 6     | Face Alignment                   | ✅ Tốt               | Thêm hình before/after alignment                        |
| 7     | CNN Overview                     | ⚠️ Hơi lý thuyết     | OK nếu audience không biết CNN, rút gọn nếu họ biết rồi |
| 8     | Recognition Algorithms (Hướng 1) | ✅ Tốt               | Giữ nguyên, rõ hạn chế                                  |
| 9     | Learning Similarity              | ✅ Tốt               | Thêm hình embedding space 2D trực quan                  |
| 10    | Face Representation              | ✅ Tốt               | Thêm hình: cùng người → embedding gần nhau              |
| 11    | Recognition Algorithms (Hướng 2) | ✅ Tốt               | Thêm sơ đồ 1-N search                                   |
| 12    | Lựa chọn mô hình                 | ✅ Tốt               | Thêm bảng so sánh benchmark                             |
| 13    | Backbone comparison (hình)       | ❌ Chỉ có hình       | Thêm text summary bảng so sánh                          |
| 14    | MobileFaceNet + MobileNetV1      | ✅ Tốt               | Thêm diagram kiến trúc                                  |
| 15    | Width Multiplier                 | ✅ Tốt               | Thêm bảng số: params/accuracy vs α                      |
| 16    | Pipeline Training                | ❌ Trống             | Cần add text description                                |
| 17    | Forward Pass                     | ⚠️ Chỉ hình          | Thêm 2-3 bullet points giải thích                       |
| 18    | Backward Pass                    | ⚠️ Chỉ hình          | Thêm mention loss function cụ thể (ArcFace/CosFace)     |
| 19-26 | Nhiều slide trống                | ❌ Không rõ nội dung | Cần xác nhận lại các slide này có hình vẽ gì            |
| 27    | Thank You                        | ⚠️ Đột ngột          | Thêm slide "Next Steps" trước                           |

---

## 💡 Các Slide Nên Thêm Mới

### Slide mới A: "Embedding Space Visualization"
Hình ảnh trực quan nhất cho người xem hiểu Face Recognition:
```
     │ 
  +1 │  ●Alice ●Alice   ○Bob ○Bob
     │    ●Alice              ○Bob
   0 │────────────────────────────
     │         ◆Unknown
  -1 │
     └─────────────────────────────
       Embedding dimension visualized (t-SNE/PCA 2D)

  → Cùng người: cluster gần nhau
  → Khác người: cluster xa nhau
  → Unknown: không thuộc cluster nào
```

### Slide mới B: "Loss Function — ArcFace"
Giải thích tại sao chọn ArcFace loss thay vì Cross-Entropy thông thường:
- **Softmax thông thường:** Chỉ tối ưu để phân loại, không tối ưu để embedding space compact
- **ArcFace:** Thêm angular margin → embedding của cùng người **gần nhau hơn**, khác người **xa hơn**
- Kết quả: accuracy cao hơn với same model size

### Slide mới C: "NNE Toolchain — Deploy lên Camera"
```
PyTorch (.pth)
    │
    ▼  [onnx_export.py]
ONNX (.onnx)
    │
    ▼  [nne_convert]
Intermediate Format
    │
    ▼  [nne_quantize — INT8]   ← Giảm 4x memory, tăng tốc inference
Quantized Model
    │
    ▼  [nne_compile]
NNE Binary (.bin)              ← Deploy lên board
    │
    ▼  [C Code + NNE API]
Real-time Inference trên NPU
```

### Slide mới D: "Kết Quả Đạt Được"
|                       | Mục tiêu ban đầu | Thực tế đạt được |
| --------------------- | ---------------- | ---------------- |
| **Model size**        | < 5MB            | ___ MB           |
| **Accuracy (LFW)**    | > 95%            | ___%             |
| **Inference latency** | < 50ms           | ___ ms           |
| **RAM usage**         | < 256MB          | ___ MB           |

---

## 🎤 Lời Khuyên Khi Thuyết Trình

1. **Mở đầu mạnh:** Bắt đầu bằng 1 ví dụ thực tế — "Camera nhận ra mặt bạn trong <50ms, chạy hoàn toàn offline trên chip nhỏ hơn Raspberry Pi"

2. **Demo là vũ khí mạnh nhất:** Nếu có thể, mang camera demo live hoặc video demo — impact hơn 10 slide lý thuyết

3. **Nhấn mạnh điểm khó:** Phần NNE Quantize + Deploy là phần khó và đặc thù nhất, khác biệt với paper/tutorial thông thường — **đây là giá trị thực sự của dự án**

4. **Chuẩn bị số liệu:** Audience workshop thường hỏi về số cụ thể: *"Accuracy của bạn là bao nhiêu?"*, *"Tốc độ inference thực tế là bao lâu?"*

5. **Slide 19-26 cần clarify:** Những slide này có nhiều hình ảnh nhưng không rõ nội dung từ text extraction → cần đảm bảo khi present hình rõ ràng và có narration đủ
