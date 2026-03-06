# Hướng dẫn tối ưu hóa Model Face Recognition cho Indoor Advance Camera

## 📋 Mục lục

1. [Tổng quan kiến trúc repo](#1-tổng-quan-kiến-trúc-repo)
2. [Các kỹ thuật giảm kích thước model đã triển khai trong repo](#2-các-kỹ-thuật-giảm-kích-thước-model-đã-triển-khai-trong-repo)
3. [Hướng dẫn sử dụng từng kỹ thuật](#3-hướng-dẫn-sử-dụng-từng-kỹ-thuật)
4. [Pipeline tối ưu: từ training đến deployment](#4-pipeline-tối-ưu-từ-training-đến-deployment)
5. [Ý tưởng nghiên cứu cho Indoor Advance Camera](#5-ý-tưởng-nghiên-cứu-cho-indoor-advance-camera)
6. [Anti-Spoofing (Chống giả mạo khuôn mặt)](#6-anti-spoofing-chống-giả-mạo-khuôn-mặt)
7. [Tài liệu tham khảo](#7-tài-liệu-tham-khảo)

---

## 1. Tổng quan kiến trúc repo

### Cấu trúc file liên quan đến tối ưu hóa

```
face-recognition/
├── models/
│   ├── mobilenetv1.py       ← MobileNetV1 + width_mult (0.18, 0.25, 0.50, 1.0)
│   ├── mobilenetv2.py       ← MobileNetV2 + width_mult (0.25, 1.0)
│   ├── mobilenetv3.py       ← MobileNetV3 Small/Large (SE + Hardswish)
│   ├── sphereface.py        ← Sphere20/36/64 (backbone lớn, acc cao)
│   └── onnx_model.py        ← ONNX inference engine
├── utils/
│   ├── layers.py            ← DepthWise Separable Conv, GDC, SE block
│   ├── metrics.py           ← CosFace (MCP), SphereFace (AngleLinear) loss
│   └── general.py           ← Training utilities
├── train.py                 ← Training script (hỗ trợ chọn backbone + width_mult)
├── onnx_export.py           ← Chuyển đổi PyTorch → ONNX
├── convert_to_pth.py        ← Trích xuất backbone từ checkpoint
└── evaluate_onnx_benchmark.py ← Benchmark toàn diện
```

### So sánh các model có sẵn

| Model             | Params | Size (ONNX) | LFW (%) | CALFW (%) | Mục đích          |
| ----------------- | ------ | ----------- | ------- | --------- | ----------------- |
| Sphere20          | 24.5M  | ~95 MB      | 99.67   | 95.61     | Accuracy cao nhất |
| Sphere36          | 34.6M  | ~135 MB     | 99.72   | 95.64     | Nghiên cứu        |
| MobileNetV2       | 2.29M  | 8.67 MB     | 99.55   | 94.87     | Cân bằng tốt      |
| MobileNetV3_Large | 3.52M  | ~14 MB      | 99.53   | 94.56     | Cân bằng tốt      |
| MobileNetV3_Small | 1.25M  | ~5 MB       | 99.30   | 93.77     | Thiết bị nhúng    |
| MobileNetV1_0.25  | 0.36M  | 1.39 MB     | 98.76   | 92.02     | Nhúng nhẹ         |
| MobileNetV1_0.18  | 0.22M  | 0.87 MB     | ~94.5   | ~82.0     | Siêu nhẹ          |

---

## 2. Các kỹ thuật giảm kích thước model đã triển khai trong repo

### 2.1. Width Multiplier (Hệ số co kênh)

**File**: `models/mobilenetv1.py` dòng 11, `models/mobilenetv2.py` dòng 64

Đây là kỹ thuật **quan trọng nhất** trong repo. Tham số `width_mult` co nhỏ số kênh (channels) ở mọi tầng:

```python
# mobilenetv1.py
class MobileNetV1(nn.Module):
    def __init__(self, embedding_dim: int = 512, width_mult: float = 0.18):
        filters = [32, 64, 128, 256, 512, 1024]
        filters = [_make_divisible(filter * width_mult) for filter in filters]
```

**Nguyên lý**: Nếu tầng gốc có 512 kênh, với `width_mult=0.25` → chỉ còn 128 kênh → giảm ~16x params cho tầng đó.

| width_mult | Kênh ban đầu (512) | Kênh sau co | Giảm params (~) |
| ---------- | ------------------ | ----------- | --------------- |
| 1.0        | 512                | 512         | 0% (gốc)        |
| 0.50       | 512                | 256         | ~75%            |
| 0.25       | 512                | 128         | ~93.75%         |
| 0.18       | 512                | 96          | ~96.5%          |

**Hàm `_make_divisible`** (`utils/layers.py` dòng 6): Đảm bảo số kênh sau khi co chia hết cho 8, tối ưu cho tính toán trên GPU/NPU.

### 2.2. Depthwise Separable Convolution

**File**: `utils/layers.py` dòng 125-153

Thay thế Conv2D tiêu chuẩn bằng 2 bước: **Depthwise** (lọc theo không gian) + **Pointwise** (kết hợp kênh):

```python
class DepthWiseSeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride):
        layers = [
            Conv2dNormActivation(in_channels, in_channels, kernel_size=3,
                                 stride=stride, groups=in_channels),  # Depthwise
            Conv2dNormActivation(in_channels, out_channels, kernel_size=1)  # Pointwise
        ]
```

**Hiệu quả**: Giảm tính toán từ `K × K × C_in × C_out` xuống `K × K × C_in + C_in × C_out`. Với kernel 3×3:

```
Conv2D tiêu chuẩn:  9 × C_in × C_out  phép tính
Depthwise Separable: 9 × C_in + C_in × C_out  phép tính
→ Giảm ~8-9x phép tính!
```

### 2.3. Inverted Residual Block (MobileNetV2)

**File**: `models/mobilenetv2.py` dòng 11-57

Kiến trúc "mỏng → mở rộng → mỏng" (bottleneck) với skip connection:

```
Input (thin) → Expand (1×1 Conv) → Depthwise (3×3 DW) → Project (1×1 Conv) → Output (thin)
      └──────────────── Skip Connection ──────────────────────┘
```

**Ưu điểm**: Giữ skip connection trên tensor mỏng → tiết kiệm RAM cho activation maps.

### 2.4. Squeeze-and-Excitation (SE) Block

**File**: `utils/layers.py` dòng 87-122

Dùng trong MobileNetV3, giúp model tập trung vào kênh quan trọng:

```
Input → Global AvgPool → FC1 (squeeze) → ReLU → FC2 (excite) → Sigmoid → Scale Input
```

**Chi phí**: Rất nhỏ (~thêm 1-2% params) nhưng tăng accuracy đáng kể.

### 2.5. GDC (Global Depthwise Convolution)

**File**: `utils/layers.py` dòng 176-191

Thay thế Global Average Pooling + FC bằng cách dùng **Depthwise Conv với kernel = feature map size** (7×7):

```python
class GDC(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        self.features = nn.Sequential(
            LinearBlock(in_channels, in_channels, kernel_size=7,
                        stride=1, padding=0, groups=in_channels),  # Depthwise 7×7
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )
```

**Ưu điểm**: Bảo toàn thông tin không gian tốt hơn GAP, quan trọng cho face recognition.

### 2.6. CosFace / SphereFace Loss

**File**: `utils/metrics.py`

Sử dụng loss function chuyên biệt cho face recognition thay vì Softmax thông thường:

- **MarginCosineProduct (CosFace)**: `s × (cos(θ) - m)` — thêm margin vào cosine similarity
- **AngleLinear (SphereFace)**: Thêm margin góc vào decision boundary

**Ý nghĩa cho tối ưu**: Loss function tốt giúp model nhỏ vẫn đạt accuracy cao → **giảm kích thước model mà không hy sinh nhiều accuracy**.

### 2.7. ONNX Export Pipeline

**File**: `onnx_export.py`, `convert_to_pth.py`

Pipeline 2 bước để xuất model tối ưu:

```
Training (.ckpt chứa optimizer, scheduler)
    │
    ├─→ convert_to_pth.py: Trích backbone weights → .pth (nhẹ hơn ~10-50x)
    │
    └─→ onnx_export.py: Chuyển PyTorch → ONNX (tối ưu graph, constant folding)
```

| Dạng file | mobilenetv1_0.25 | Bao gồm                                                |
| --------- | ---------------- | ------------------------------------------------------ |
| `.ckpt`   | ~30 MB           | Model + Optimizer + Scheduler + Classifier head        |
| `.pth`    | ~1.5 MB          | Chỉ backbone weights                                   |
| `.onnx`   | ~1.4 MB          | Backbone tối ưu (constant folding, graph optimization) |

---

## 3. Hướng dẫn sử dụng từng kỹ thuật

### 3.1. Train model với width_mult nhỏ

```bash
conda activate face

# MobileNetV1 với width_mult=0.25 (0.36M params)
python train.py --root data/train/ms1m_112x112 --database MS1M \
    --network mobilenetv1 --classifier MCP --epochs 30

# MobileNetV2 với width_mult=0.25 (~0.5M params)
python train.py --root data/train/ms1m_112x112 --database MS1M \
    --network mobilenetv2_025 --classifier MCP --epochs 30

# MobileNetV1 với width_mult=0.50 (~1M params)
python train.py --root data/train/ms1m_112x112 --database MS1M \
    --network mobilenetv1_050 --classifier MCP --epochs 30
```

### 3.2. Trích xuất backbone từ checkpoint

```python
# convert_to_pth.py
import torch
from models import MobileNetV1

ckpt = torch.load("weights/mobilenetv1_MCP_best.ckpt", map_location="cpu")
model = MobileNetV1(embedding_dim=512, width_mult=0.18)
model.load_state_dict(ckpt["model"])
torch.save(model.state_dict(), "mobilenetv1_018.pth")
```

### 3.3. Export sang ONNX

```bash
# Export MobileNetV1_0.25
python onnx_export.py -w weights/mobilenetv1_0.25_mcp.pth -n mobilenetv1_0.25

# Export MobileNetV2
python onnx_export.py -w weights/mobilenetv2_mcp.pth -n mobilenetv2

# Export với dynamic batch size (cho server deployment)
python onnx_export.py -w weights/mobilenetv2_mcp.pth -n mobilenetv2 --dynamic
```

### 3.4. Benchmark so sánh

```bash
# Resource metrics (nhanh, ~10 giây)
python evaluate_onnx_benchmark.py --metrics-only

# Full benchmark (accuracy + resources)
python evaluate_onnx_benchmark.py --output results.csv
```

---

## 4. Pipeline tối ưu: từ training đến deployment

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────┐
│  Chọn model   │───→│   Training   │───→│   Convert    │───→│   Benchmark   │
│  + width_mult │    │  (MS1MV2)    │    │  .ckpt→.pth  │    │  (accuracy +  │
│               │    │              │    │  .pth→.onnx   │    │   resources)  │
└──────────────┘    └──────────────┘    └──────────────┘    └───────┬────────┘
                                                                    │
                                                        ┌───────────▼──────────┐
                                                        │  Đạt yêu cầu?       │
                                                        │  Acc > threshold?    │
                                                        │  Size < limit?       │
                                                        │  Latency < target?   │
                                                        └──────┬───────┬───────┘
                                                           YES │       │ NO
                                                    ┌──────────▼──┐  ┌─▼──────────────┐
                                                    │  Deploy lên │  │ Quay lại bước 1 │
                                                    │  Camera     │  │ chọn width_mult  │
                                                    └─────────────┘  │ nhỏ/lớn hơn    │
                                                                     └────────────────┘
```

### Bảng gợi ý chọn model theo phần cứng camera

| Phần cứng Camera            | RAM       | Storage     | Gợi ý model               | Gợi ý width_mult |
| --------------------------- | --------- | ----------- | ------------------------- | ---------------- |
| **SoC giá rẻ** (< $10)      | < 256MB   | < 8MB flash | MobileNetV1               | 0.18 - 0.25      |
| **SoC tầm trung** ($10-30)  | 256MB-1GB | 16-64MB     | MobileNetV1 hoặc V3_Small | 0.25 - 0.50      |
| **SoC cao cấp** (> $30)     | > 1GB     | > 64MB      | MobileNetV2 hoặc V3_Large | 0.50 - 1.0       |
| **Có AI accelerator** (NPU) | > 2GB     | > 128MB     | MobileNetV2/V3 ONNX       | 1.0              |

---

## 5. Ý tưởng nghiên cứu cho Indoor Advance Camera

### 5.1. Knowledge Distillation (Chưng cất tri thức)

**Ý tưởng**: Dùng Sphere36 (model lớn, acc cao) làm "Teacher", dạy MobileNetV1_0.25 (model nhỏ) làm "Student":

```
Teacher (Sphere36, 34.6M params, 99.72% LFW)
    │
    │  Soft labels / Feature matching
    ▼
Student (MobileNetV1_0.25, 0.36M params)
    → Accuracy tăng 1-3% so với train thông thường
```

**Cách thực hiện**:
1. Train Teacher (Sphere36) trước — đã có sẵn trong `weights/`
2. Thêm KD loss: `L = α × CE_loss + (1-α) × KL_div(student_logits, teacher_logits)`
3. Có thể matching embedding: `L_feat = MSE(student_embedding, teacher_embedding)`

### 5.2. Post-Training Quantization (Lượng tử hóa sau training)

**Ý tưởng**: Chuyển model từ FP32 → INT8, giảm ~4x kích thước và tăng tốc inference:

```bash
# Sử dụng ONNX Runtime quantization
python -m onnxruntime.quantization.preprocess \
    --input weights/mobilenetv1_0.25_mcp.onnx \
    --output weights/mobilenetv1_0.25_preprocessed.onnx

# Dynamic quantization
python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('weights/mobilenetv1_0.25_preprocessed.onnx',
                 'weights/mobilenetv1_0.25_int8.onnx',
                 weight_type=QuantType.QUInt8)
"
```

**Kỳ vọng**: Giảm ~2-4x kích thước, giảm ~2x latency, accuracy giảm < 1%.

### 5.3. Structured Pruning (Cắt tỉa có cấu trúc)

**Ý tưởng**: Loại bỏ các channel/filter ít quan trọng:

1. Train model bình thường
2. Đánh giá tầm quan trọng của từng filter (L1-norm, Taylor expansion)
3. Loại bỏ % filter ít quan trọng
4. Fine-tune lại model

**Kết hợp với repo**: Pruning + `width_mult` = double compression.

### 5.4. Model Architecture Search (NAS)

**Ý tưởng**: Tự động tìm kiến trúc tối ưu cho bài toán face recognition trên Indoor Camera:
- Tìm `width_mult` tối ưu cho từng tầng (không nhất thiết đồng nhất)
- Tìm cấu hình inverted residual (expand ratio, kernel size) tối ưu
- Constraint: latency < X ms trên target hardware

### 5.5. Multi-Task Learning

**Ý tưởng**: Train 1 model duy nhất cho nhiều tác vụ:

```
Shared Backbone (MobileNetV1/V2)
    │
    ├─→ Head 1: Face Embedding (512-D) → Recognition
    ├─→ Head 2: Liveness Score → Anti-spoofing
    ├─→ Head 3: Age/Gender → Demographics
    └─→ Head 4: Face Quality → Reject ảnh kém
```

**Ưu điểm**: 1 model thay vì 3-4 models → tiết kiệm tài nguyên.

---

## 6. Anti-Spoofing (Chống giả mạo khuôn mặt)

### Tại sao cần Anti-Spoofing cho Indoor Camera?

Indoor Advance Camera dễ bị tấn công bằng:
- **Photo attack**: Dùng ảnh in / ảnh trên màn hình điện thoại
- **Video replay**: Phát video khuôn mặt trên màn hình
- **3D mask attack**: Dùng mặt nạ 3D (khó hơn, hiếm gặp indoor)

### 6.1. Phương pháp dựa trên Texture Analysis

**Ý tưởng**: Phân biệt texture da thật vs ảnh in/màn hình

```
Input Image → LBP / HoG features → SVM/MLP → Real/Fake
```

**Ưu điểm**: Rất nhẹ, phù hợp embedded.
**Nhược điểm**: Dễ bị bypass bởi ảnh chất lượng cao.

### 6.2. Phương pháp dựa trên Depth Map

**Ý tưởng**: Khuôn mặt thật có chiều sâu 3D, ảnh phẳng thì không.

Có thể triển khai bằng:
- **IR Camera** (nếu camera hỗ trợ): So sánh ảnh IR và RGB
- **Structured Light**: Chiếu pattern ánh sáng và đo biến dạng
- **Monocular Depth**: Dùng CNN nhỏ ước tính depth map từ 1 ảnh RGB

### 6.3. Phương pháp dựa trên rPPG (Remote Photoplethysmography)

**Ý tưởng**: Phát hiện nhịp tim từ video khuôn mặt. Ảnh/video phát lại không có biến đổi màu da do nhịp tim.

```
Video (vài giây) → ROI tracking → Color signal extraction → FFT → Heart rate
    → Có nhịp tim? → Real face
    → Không có? → Fake (photo/video)
```

**Ưu điểm**: Rất khó giả mạo.
**Nhược điểm**: Cần video (không phải 1 frame), cần xử lý real-time.

### 6.4. Phương pháp dựa trên CNN nhẹ

**Paper gợi ý**: FeatherNet, MiniFASNet

```
Input (khuôn mặt cropped 112×112)
    │
    └─→ Lightweight CNN (~0.1M params)
            │
            └─→ Binary output: Real (1) hoặc Fake (0)
```

**Có thể tích hợp vào repo hiện tại**:
1. Thêm 1 model anti-spoofing nhỏ (< 0.5 MB)
2. Pipeline: Face Detection → Anti-Spoofing → Face Recognition
3. Nếu spoofing score < threshold → reject, không chạy recognition

### 6.5. Multi-Modal Liveness Detection (nếu camera hỗ trợ)

| Phương pháp            | Cần hardware              | Hiệu quả | Chi phí    |
| ---------------------- | ------------------------- | -------- | ---------- |
| RGB only (texture/CNN) | Camera thường             | ⭐⭐       | Thấp       |
| RGB + IR               | Camera có IR              | ⭐⭐⭐⭐     | Trung bình |
| RGB + Depth (ToF)      | Camera có ToF             | ⭐⭐⭐⭐⭐    | Cao        |
| RGB + rPPG             | Camera thường (cần video) | ⭐⭐⭐      | Thấp       |

### 6.6. Challenge-Response (Thách thức-Phản hồi)

**Ý tưởng cho Indoor Camera**:
1. Yêu cầu người dùng **nháy mắt** hoặc **quay đầu**
2. Detect hành động bằng Facial Landmark tracking
3. Ảnh in / video phát lại không thể thực hiện hành động theo yêu cầu

**Nhược điểm**: Cần tương tác người dùng, không phù hợp giám sát tự động.

---

## 7. Tài liệu tham khảo

### Papers về kiến trúc nhẹ
- **MobileNetV1**: [Howard et al., 2017](https://arxiv.org/abs/1704.04861) — Depthwise Separable Conv
- **MobileNetV2**: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381) — Inverted Residuals
- **MobileNetV3**: [Howard et al., 2019](https://arxiv.org/abs/1905.02244) — NAS + SE + H-Swish
- **MobileFaceNet**: [Chen et al., 2018](https://arxiv.org/abs/1804.07573) — GDC cho face recognition

### Papers về Loss Function
- **CosFace**: [Wang et al., 2018](https://arxiv.org/abs/1801.09414) — Margin Cosine Loss
- **SphereFace**: [Liu et al., 2017](https://arxiv.org/abs/1704.08063) — Angular Margin Loss
- **ArcFace**: [Deng et al., 2019](https://arxiv.org/abs/1801.07698) — Additive Angular Margin

### Papers về tối ưu hóa
- **Knowledge Distillation**: [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
- **Quantization**: [Jacob et al., 2018](https://arxiv.org/abs/1712.05877) — Quantization and Training of NN
- **Pruning**: [Li et al., 2017](https://arxiv.org/abs/1608.08710) — Filter Pruning

### Papers về Anti-Spoofing
- **MiniFASNet**: [Yu et al., 2020](https://arxiv.org/abs/2007.09273) — Multi-modal FAS
- **FeatherNet**: [Zhang et al., 2019](https://arxiv.org/abs/1904.09290) — Lightweight FAS
- **rPPG Liveness**: [Li et al., 2016](https://arxiv.org/abs/1601.05781)
- **CDCN**: [Yu et al., 2020](https://arxiv.org/abs/2003.04092) — Central Difference for FAS
