# 📊 Agenda Slide Reference — Face Recognition cho Indoor Camera

> **Mục đích:** Tài liệu chi tiết để làm slide trình bày nghiên cứu Face Recognition trên Indoor Camera.  
> **Dự án:** [face-recognition](https://github.com/yakhyo/face-recognition)  
> **Ngôn ngữ trình bày:** Tiếng Việt

---

## 📑 Mục Lục Slide

| #   | Phần Slide                                                                     | Mục đích                                     |
| --- | ------------------------------------------------------------------------------ | -------------------------------------------- |
| 1   | [Tổng quan bài toán](#1-tổng-quan-bài-toán)                                    | Giới thiệu context, vấn đề cần giải quyết    |
| 2   | [Kiến trúc mô hình](#2-kiến-trúc-mô-hình)                                      | Giải thích model architecture chi tiết       |
| 3   | [Data sử dụng](#3-data-sử-dụng)                                                | Dataset training & validation                |
| 4   | [Cách training](#4-cách-training)                                              | Pipeline training, hyperparameters, kỹ thuật |
| 5   | [Cách lựa chọn mô hình](#5-cách-lựa-chọn-mô-hình)                              | Tiêu chí so sánh và benchmark kết quả        |
| 6   | [Giảm model size & Trade-off Accuracy](#6-giảm-model-size--trade-off-accuracy) | Các kỹ thuật tối ưu và phân tích cân bằng    |
| 7   | [ONNX → BIN: Deploy lên Camera](#7-onnx--bin-deploy-lên-camera)                | Quy trình convert và triển khai thực tế      |

---

## 1. Tổng quan bài toán

### 1.1. Bối cảnh & Động lực

**Bài toán:** Nhận diện khuôn mặt (Face Recognition) trên camera indoor (advance camera) với các ràng buộc:

| Ràng buộc     | Chi tiết                                                             |
| ------------- | -------------------------------------------------------------------- |
| **Phần cứng** | SoC tích hợp trong camera, RAM giới hạn (256MB–2GB), có/không có NPU |
| **Storage**   | Flash memory giới hạn (8MB–128MB) → model phải rất nhỏ               |
| **Power**     | Camera hoạt động 24/7 → tiêu thụ năng lượng phải thấp                |
| **Latency**   | Real-time processing → inference phải nhanh (< 50ms/frame)           |
| **Accuracy**  | Đủ chính xác cho bài toán access control, attendance, surveillance   |

**Thách thức của Indoor Camera:**
- **Ánh sáng thay đổi:** Đèn nhân tạo, backlight, bóng tối ban đêm
- **Góc chụp đa dạng:** Người đi qua ở nhiều góc, không nhìn thẳng
- **Khoảng cách:** Người có thể đứng gần hoặc xa camera (1m–5m)
- **Anti-spoofing:** Chống giả mạo bằng ảnh/video phát lại
- **Resource cực hạn:** Model phải chạy trực tiếp trên chip camera, không có cloud

### 1.2. Pipeline Face Recognition hoàn chỉnh

```
┌─────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│  Frame   │───▶│ Face Detection│───▶│ Face Alignment │───▶│ CNN Backbone  │
│ từ Camera│    │  (RetinaFace) │    │  (112×112 crop)│    │  (MobileNet) │
└─────────┘    └───────────────┘    └────────────────┘    └──────┬───────┘
                                                                  │
                                                                  ▼
┌──────────┐    ┌───────────────────┐    ┌───────────────────────────────┐
│ Decision │◀───│ Cosine Similarity │◀───│ Embedding Vector (512-D)      │
│ Same/Diff│    │    threshold      │    │ [0.12, -0.34, 0.56, ...]     │
└──────────┘    └───────────────────┘    └───────────────────────────────┘
```

### 1.3. Giải thích từng bước trong Pipeline

| Bước                     | Mô tả                                                                                                                               | Input → Output                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| **1. Face Detection**    | Phát hiện vị trí khuôn mặt trong frame, trả về bounding box + 5 điểm landmark (mắt trái, mắt phải, mũi, 2 khóe miệng)               | Frame camera → Bounding box + 5 landmarks     |
| **2. Face Alignment**    | Dùng Similarity Transform ánh xạ 5 landmarks về 5 điểm tham chiếu chuẩn (ArcFace standard), xoay và cắt khuôn mặt thành ảnh 112×112 | Ảnh gốc + landmarks → Ảnh aligned 112×112     |
| **3. CNN Backbone**      | Mạng neural trích xuất đặc trưng khuôn mặt, biến ảnh 112×112×3 thành vector embedding                                               | Ảnh 112×112×3 → Feature map → Embedding 512-D |
| **4. Cosine Similarity** | So sánh embedding với database đã đăng ký, tính similarity score (-1 đến 1)                                                         | 2 embedding vectors → Similarity score        |
| **5. Decision**          | So sánh score với threshold để quyết định → cùng người hay khác người                                                               | Score + threshold → Same / Different          |

### 1.4. Tại sao chọn approach Embedding-based?

| Approach                     | Ưu điểm                                                     | Nhược điểm                    | Cho Camera Indoor     |
| ---------------------------- | ----------------------------------------------------------- | ----------------------------- | --------------------- |
| **Classification (Softmax)** | Training đơn giản                                           | Thêm người mới phải retrain   | ❌ Không linh hoạt     |
| **Embedding + Similarity**   | Thêm người chỉ cần thêm ảnh vào database, không cần retrain | Cần loss function chuyên biệt | ✅ **Lựa chọn tối ưu** |

**Kết luận cho slide:** Embedding-based cho phép camera thêm/xóa người dùng mà không cần update model → phù hợp triển khai thực tế.

---

## 2. Kiến trúc mô hình

### 2.1. Các model đã triển khai trong project

| Model                 | Loại        | Đặc điểm chính            | Params         | ONNX Size | Mục đích            |
| --------------------- | ----------- | ------------------------- | -------------- | --------- | ------------------- |
| **Sphere20**          | SphereFace  | 20 layer, residual blocks | 24.5M          | ~95 MB    | Accuracy cao nhất   |
| **Sphere36**          | SphereFace  | 36 layer, deeper          | 34.6M          | ~135 MB   | Nghiên cứu baseline |
| **MobileNetV1**       | Lightweight | Depthwise Separable Conv  | 0.36M (w=0.25) | 1.39 MB   | Thiết bị nhúng      |
| **MobileNetV2**       | Lightweight | Inverted Residual Block   | 2.29M (w=1.0)  | 8.67 MB   | Cân bằng tốt        |
| **MobileNetV2_025**   | Lightweight | V2 + width_mult=0.25      | 0.46M          | 1.80 MB   | Nhúng nhẹ           |
| **MobileNetV3 Small** | Lightweight | SE Block + H-Swish + NAS  | 1.25M          | 4.79 MB   | Nhúng tiên tiến     |
| **MobileNetV3 Large** | Lightweight | SE Block + H-Swish + NAS  | 3.52M          | ~14 MB    | Cân bằng cao        |

### 2.2. Kiến trúc MobileNetV2 chi tiết (Model chính cho camera)

```
Input: [1, 3, 112, 112] — Ảnh khuôn mặt đã align

  ▼ Conv2dNormActivation(3, 32, stride=2)      ← First Conv: RGB → 32 channels
  ▼ InvertedResidual blocks (t, c, n, s):
    ┌──────────────────────────────────────────────────────────────────┐
    │  t=1,  c=16,   n=1, s=1  — Không expand, 16 channels            │
    │  t=6,  c=24,   n=2, s=2  — Expand 6×, downsample → 56×56       │
    │  t=6,  c=32,   n=3, s=2  — Expand 6×, downsample → 28×28       │
    │  t=6,  c=64,   n=4, s=2  — Expand 6×, downsample → 14×14       │
    │  t=6,  c=96,   n=3, s=1  — Expand 6×, giữ nguyên 14×14         │
    │  t=6,  c=160,  n=3, s=2  — Expand 6×, downsample → 7×7         │
    │  t=6,  c=320,  n=1, s=1  — Expand 6×, output channels          │
    └──────────────────────────────────────────────────────────────────┘
  ▼ Conv2dNormActivation(320, 1280, k=1)       ← Last Conv: mở rộng channels
  ▼ GDC: Depthwise Conv 7×7 → Flatten → Linear(1280, 512)
     ← Global Depthwise Convolution → Embedding

Output: [1, 512] — Embedding vector
```

### 2.3. Inverted Residual Block — Cốt lõi của MobileNetV2

```
Input (thin, ít channels)
  │
  ├─── 1. Expansion (1×1 Conv): Mở rộng channels lên t lần
  │         Input [C] → [C × t]
  │
  ├─── 2. Depthwise (3×3 DW Conv): Lọc spatial cho từng channel riêng
  │         [C × t] → [C × t]  (groups = C × t)
  │
  ├─── 3. Projection (1×1 Conv, KHÔNG activation): Giảm channels về mỏng
  │         [C × t] → [C']
  │
  └─── + Residual Connection (nếu stride=1 và in_channels == out_channels)

Tại sao gọi "Inverted"?
  • ResNet:     Rộng → Hẹp → Rộng (bottleneck ở giữa)
  • MobileNetV2: Hẹp → Rộng → Hẹp (expansion ở giữa = inverted bottleneck)
```

### 2.4. Các Building Block quan trọng

| Block                        | File                    | Chức năng                          | Tại sao dùng                                                    |
| ---------------------------- | ----------------------- | ---------------------------------- | --------------------------------------------------------------- |
| **Conv2dNormActivation**     | `utils/layers.py`       | Conv + BatchNorm + PReLU           | Building block cơ bản, normalize + activation mặc định          |
| **DepthWiseSeparableConv2d** | `utils/layers.py`       | Depthwise + Pointwise              | Giảm params 8-9× so với Conv thường                             |
| **InvertedResidual**         | `models/mobilenetv2.py` | Expansion + DW + Projection + Skip | Cốt lõi MobileNetV2, hiệu quả cao                               |
| **GDC**                      | `utils/layers.py`       | Depthwise Conv 7×7 → Linear → BN   | Biến feature map sang embedding, tốt hơn Global Average Pooling |
| **SE Block**                 | `utils/layers.py`       | Squeeze-and-Excitation             | Dùng trong MobileNetV3, channel attention                       |

### 2.5. GDC — Global Depthwise Convolution (Điểm đặc biệt)

Thay vì dùng **Global Average Pooling** (mất thông tin spatial), dùng **Depthwise Conv với kernel = feature map size**:

```
Feature Map [B, C, 7, 7]
      │
      ▼ Depthwise Conv 7×7, groups=C   → [B, C, 1, 1]   ← Giữ thông tin spatial
      ▼ Flatten()                       → [B, C]
      ▼ Linear(C, 512) + BatchNorm1d   → [B, 512]       ← Embedding vector

Ưu điểm so với GAP:
  • GAP: Lấy trung bình → mất thông tin vị trí đặc trưng trên khuôn mặt
  • GDC: Mỗi channel có kernel riêng → bảo toàn thông tin vị trí mắt, mũi, miệng
```

### 2.6. Width Multiplier — Cơ chế co nhỏ model

```python
# Tham số α (width_mult) nhân với số channels ở mọi layer:
# width_mult = 1.0 → Full model
# width_mult = 0.25 → Model thu nhỏ 4× về channels

# Ví dụ: channels gốc = [32, 16, 24, 32, 64, 96, 160, 320]
# width_mult = 0.25 → [8,   8,  8,  8,  16, 24, 40,  80]

# Tại sao giảm nhiều hơn 4×?
# Params tỉ lệ bình phương channels:
#   Conv(64, 128, 3×3) = 64 × 128 × 9   = 73,728 params
#   Conv(16,  32, 3×3) = 16 ×  32 × 9   =  4,608 params (giảm 16×!)
```

| width_mult | Ý nghĩa            | Params ước lượng | ONNX Size |
| ---------- | ------------------ | ---------------- | --------- |
| 1.0        | Full model         | ~2.29M           | 8.67 MB   |
| 0.50       | Channels giảm 2×   | ~0.7M            | ~3 MB     |
| 0.25       | Channels giảm 4×   | ~0.46M           | 1.80 MB   |
| 0.18       | Channels giảm 5.5× | ~0.22M           | 0.87 MB   |

**Điểm quan trọng:** Embedding dimension (512) **KHÔNG bị ảnh hưởng** bởi width_mult. Chỉ các layer trung gian bị thu nhỏ → chất lượng embedding được bảo toàn.

---

## 3. Data sử dụng

### 3.1. Training Data

| Dataset                     | Identities | Images | Đặc điểm                                     | Nguồn       |
| --------------------------- | ---------- | ------ | -------------------------------------------- | ----------- |
| **CASIA-WebFace**           | 10,572     | 491K   | Dataset nhỏ, phù hợp thử nghiệm nhanh        | OpenSphere  |
| **VGGFace2**                | 8,631      | 3.1M   | Đa dạng pose, age, ethnicity                 | OpenSphere  |
| **MS1MV2 (MS-Celeb-1M v2)** | 85,742     | 5.8M   | Dataset lớn nhất, đã cleaned bởi InsightFace | InsightFace |

**Lựa chọn cho project:** Sử dụng **MS1MV2** để training (kết quả báo cáo trong bảng benchmark đều dùng MS1MV2).

### 3.2. Tiền xử lý Data

Tất cả ảnh đã được **pre-aligned và pre-cropped** về 112×112:

```
Ảnh gốc (tự nhiên, nhiều kích thước)
    │
    ▼ Face Detection (MTCNN/RetinaFace)  → Bounding box + 5 landmarks
    │
    ▼ Similarity Transform (5 landmarks → 5 tham chiếu chuẩn ArcFace)
    │
    ▼ Crop + Resize → Ảnh 112×112 RGB
    │
    ▼ Lưu theo cấu trúc folder:
        data/train/ms1m_112x112/
        ├── identity_00001/     ← Mỗi thư mục = 1 người
        │   ├── img_001.jpg
        │   ├── img_002.jpg
        │   └── ...
        ├── identity_00002/
        └── ... (85,742 thư mục)
```

### 3.3. Validation Data — 4 Benchmark Datasets

| Dataset      | Tên đầy đủ                | Pairs | Đặc điểm                                             | Độ khó        |
| ------------ | ------------------------- | ----- | ---------------------------------------------------- | ------------- |
| **LFW**      | Labeled Faces in the Wild | 6,000 | Benchmark cơ bản, ảnh tự nhiên                       | ⭐ Dễ          |
| **CALFW**    | Cross-Age LFW             | 6,000 | Cùng người nhưng **khác tuổi** (ví dụ 20 vs 50 tuổi) | ⭐⭐ Trung bình |
| **CPLFW**    | Cross-Pose LFW            | 6,000 | Cùng người nhưng **khác góc chụp** (profile, 45°)    | ⭐⭐⭐ Khó       |
| **AgeDB_30** | Age Database (gap 30 năm) | 6,000 | Khoảng cách tuổi 30 năm giữa 2 ảnh                   | ⭐⭐⭐ Khó       |

**Phương pháp đánh giá:** K-Fold Cross-Validation 10 folds
- Chia 6,000 pairs thành 10 phần
- 9 phần tìm threshold tối ưu, 1 phần test
- Lặp 10 lần, lấy trung bình accuracy

### 3.4. Data Augmentation khi Training

| Kỹ thuật                   | Mô tả                                                                    | Tại sao                            |
| -------------------------- | ------------------------------------------------------------------------ | ---------------------------------- |
| **Random Horizontal Flip** | Lật ngang ảnh 50% xác suất                                               | Khuôn mặt đối xứng, tăng diversity |
| **Normalize**              | ImageNet mean/std: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) | Chuẩn hóa pixel values             |
| **ToTensor**               | PIL Image → PyTorch Tensor [0, 1]                                        | Format phù hợp cho model           |

> **Lưu ý cho slide:** Ảnh đã pre-aligned → không cần heavy augmentation (random crop, rotation) vì khuôn mặt đã chuẩn vị trí.

---

## 4. Cách Training

### 4.1. Công nghệ & Framework

| Thành phần               | Công nghệ                                                      |
| ------------------------ | -------------------------------------------------------------- |
| **Framework**            | PyTorch                                                        |
| **Distributed Training** | `torchrun` + DDP (DistributedDataParallel)                     |
| **GPU**                  | Hỗ trợ multi-GPU (ví dụ: 2 GPU với `--nproc_per_node=2`)       |
| **Data Loading**         | Custom `ImageFolder` + `DataLoader` (multi-worker, pin_memory) |

### 4.2. Hyperparameters

| Hyperparameter     | Giá trị                          | Ý nghĩa                                         |
| ------------------ | -------------------------------- | ----------------------------------------------- |
| **Batch size**     | 512 (default), 256 cho model lớn | Số ảnh xử lý cùng lúc                           |
| **Epochs**         | 30                               | Số lần duyệt qua toàn bộ dataset                |
| **Learning Rate**  | 0.1 (initial)                    | Tốc độ học ban đầu                              |
| **LR Scheduler**   | MultiStepLR                      | Giảm LR tại các milestone epochs                |
| **Milestones**     | [10, 20, 25]                     | Epochs mà LR giảm 10×                           |
| **Gamma**          | 0.1                              | Hệ số giảm LR (LR × 0.1 tại mỗi milestone)      |
| **Optimizer**      | SGD                              | Stochastic Gradient Descent                     |
| **Momentum**       | 0.9                              | Gia tốc gradient                                |
| **Weight Decay**   | 5e-4                             | Regularization, chống overfitting               |
| **Early Stopping** | patience=10                      | Dừng nếu accuracy không cải thiện sau 10 epochs |

### 4.3. Learning Rate Schedule

```
LR
0.1  ──────────┐
               │ (epoch 10)
0.01 ──────────┼──────────┐
               │          │ (epoch 20)
0.001 ─────────┼──────────┼──────┐
               │          │      │ (epoch 25)
0.0001 ────────┼──────────┼──────┼──────
               10         20     25     30  → epoch
```

### 4.4. Loss Function — CosFace (MarginCosineProduct)

**Tại sao không dùng Softmax thông thường?**

Softmax Loss chỉ phân loại "ảnh này thuộc ai" nhưng **không ép** embeddings phải separable trong không gian feature. CosFace thêm **margin** vào cosine similarity:

```
                   Softmax Loss:    s × cos(θ)
                   CosFace Loss:    s × (cos(θ) - m)       ← margin m = 0.40
                   ArcFace Loss:    s × cos(θ + m)         ← margin m = 0.50

  • s = 30.0 (scale factor) — phóng đại gradient
  • m = 0.40 (margin) — "phạt" class đúng thêm margin m
  ──────────────────────────────────────────────────────────────
  Trực giác: Giống cuộc thi, người chiến thắng phải bơi thêm
  0.4 giây mới thắng → ép model học embedding cực chính xác.
```

**Công thức:**
```
output = s × (cosine_similarity - one_hot × m)
loss = CrossEntropyLoss(output, labels)
```

**So sánh Loss Functions:**

| Loss              | Margin           | Công thức      | Repo dùng     |
| ----------------- | ---------------- | -------------- | ------------- |
| Softmax           | Không            | s·cos(θ)       | ❌             |
| **CosFace (MCP)** | Cosine margin    | s·(cos(θ) − m) | ✅ **Chính**   |
| SphereFace (AL)   | Angular margin   | s·cos(m·θ)     | ✅ Có sẵn      |
| ArcFace           | Additive angular | s·cos(θ + m)   | ⬜ Có thể thêm |

### 4.5. Training Pipeline hoàn chỉnh

```
1. parse_arguments()           ← Đọc config từ command line
     │
2. main(params)
     ├── setup_seed()          ← Đảm bảo reproducibility
     │
     ├── Khởi tạo Model       ← MobileNetV2(embedding_dim=512, width_mult=0.25)
     │
     ├── Khởi tạo Head         ← MarginCosineProduct(512, 85742) — CosFace
     │     └── 85,742 = số identities trong MS1MV2
     │
     ├── Load Data             ← ImageFolder → DataLoader(batch=512, workers=8)
     │
     ├── Loss + Optimizer
     │     ├── criterion = CrossEntropyLoss()
     │     ├── optimizer = SGD(lr=0.1, momentum=0.9, wd=5e-4)
     │     └── scheduler = MultiStepLR(milestones=[10, 20, 25])
     │
     └── Training Loop (30 epochs)
           for epoch in range(30):
             ├── train_one_epoch():
             │     for images, labels in train_loader:
             │       ├── embeddings = model(images)          # [B, 512]
             │       ├── output = head(embeddings, labels)   # [B, 85742]
             │       ├── loss = criterion(output, labels)
             │       ├── loss.backward()
             │       └── optimizer.step()
             │
             ├── lr_scheduler.step()
             ├── save _last.ckpt
             ├── evaluate on LFW → accuracy
             └── if accuracy > best → save _best.ckpt
```

### 4.6. Distributed Training

```bash
# Multi-GPU (2 GPUs):
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 train.py \
    --root data/train/ms1m_112x112 \
    --database MS1M \
    --network mobilenetv2 \
    --classifier MCP \
    --batch-size 256

# Single GPU:
python train.py \
    --root data/train/ms1m_112x112 \
    --database MS1M \
    --network mobilenetv2_025 \
    --classifier MCP
```

### 4.7. Checkpoint Structure

```
weights/
├── mobilenetv2_025_MCP_last.ckpt     ← Checkpoint cuối cùng (~30 MB)
│     Chứa: model state_dict + optimizer + lr_scheduler + epoch
│
├── mobilenetv2_025_MCP_best.ckpt     ← Checkpoint accuracy cao nhất
│
├── mobilenetv2_025_mcp.pth           ← Chỉ backbone weights (~1.8 MB)
│     Trích từ .ckpt bằng convert_to_pth.py
│
└── mobilenetv2_025_mcp.onnx          ← ONNX format (~1.8 MB)
      Export từ .pth bằng onnx_export.py
```

---

## 5. Cách lựa chọn mô hình

### 5.1. Tiêu chí đánh giá

| Tiêu chí              | Metric              | Công cụ đo                                 | Tại sao quan trọng cho Camera    |
| --------------------- | ------------------- | ------------------------------------------ | -------------------------------- |
| **Accuracy**          | % trên 4 benchmarks | K-Fold 10 trên LFW, CALFW, CPLFW, AgeDB_30 | Chất lượng nhận diện             |
| **Model Size**        | MB (file ONNX)      | `os.path.getsize()`                        | Storage giới hạn trên camera     |
| **RAM Usage**         | MB khi chạy         | `psutil.Process().memory_info()`           | RAM giới hạn trên SoC            |
| **Inference Latency** | ms/frame            | Trung bình 100 lần (sau 10 warmup)         | Real-time FPS requirement        |
| **Parameters**        | Số params           | Đếm từ ONNX graph                          | Độ phức tạp model                |
| **FLOPs**             | Phép tính           | Từ Conv/Gemm nodes                         | Tương quan với power consumption |
| **Efficiency Score**  | Tổng hợp            | `Accuracy / (Size × Latency)^0.3`          | Chỉ số xếp hạng nhanh            |

### 5.2. Bảng Benchmark thực tế (từ kết quả thí nghiệm)

#### Accuracy Comparison (training trên MS1MV2):

| Model                 | Params | Size (MB) | LFW (%)   | CALFW (%) | CPLFW (%) | AgeDB_30 (%) | Avg Acc (%) |
| --------------------- | ------ | --------- | --------- | --------- | --------- | ------------ | ----------- |
| **Sphere20**          | 24.5M  | ~95       | **99.67** | **95.61** | 88.75     | **96.58**    | 95.15       |
| **Sphere36**          | 34.6M  | ~135      | **99.72** | **95.64** | **89.92** | **96.83**    | **95.53**   |
| **MobileNetV2**       | 2.29M  | 8.67      | 99.55     | 94.80     | 86.93     | 95.17        | 94.11       |
| **MobileNetV3 Small** | 1.25M  | 4.79      | 99.30     | 93.75     | 85.33     | 92.72        | 92.78       |
| **MobileNetV3 Large** | 3.52M  | ~14       | 99.53     | 94.56     | 86.79     | 95.13        | 94.00       |
| **MobileNetV1 0.25**  | 0.36M  | 1.39      | 98.75     | 91.97     | 82.38     | 90.05        | 90.79       |
| **MobileNetV2 0.25**  | 0.46M  | 1.80      | 95.17     | 83.48     | 71.65     | 79.41        | 82.43       |
| **MobileNetV1 0.18**  | 0.22M  | 0.87      | 94.48     | 82.02     | 73.01     | 79.79        | 82.33       |

#### Resource Metrics:

| Model                | Latency Avg (ms) | RAM Model (MB) | RAM Peak (MB) | FLOPs    | Efficiency Score |
| -------------------- | ---------------- | -------------- | ------------- | -------- | ---------------- |
| MobileNetV1 0.18     | 1.22–1.58        | 0.1–11.4       | 1.5–13.4      | 437K     | 74.8–80.8        |
| **MobileNetV1 0.25** | **1.23–2.40**    | **12.6–12.7**  | **14.4**      | **702K** | **63.3–77.4**    |
| MobileNetV2 0.25     | 3.07             | 2.3            | 7.0           | 906K     | 49.4             |
| MobileNetV3 Small    | 1.54–1.95        | 0.2            | 0.2–1.7       | 2.47M    | 47.5–50.9        |
| MobileNetV2          | 3.97–4.04        | 5.1–5.4        | 14.3–14.6     | 4.46M    | 32.4–32.6        |

### 5.3. Phân tích Trade-off

```
                    ĐỘ CHÍNH XÁC CAO (Avg ~95%)
                         ▲
                         │  ★ Sphere36 (135MB, 95.53%)
                         │  ★ Sphere20 (95MB, 95.15%)
                         │
                         │     ★ MobileNetV2 (8.67MB, 94.11%)  ← CÂN BẰNG TỐT
                         │     ★ MobileNetV3_L (14MB, 94.00%)
                         │
                         │  ★ MobileNetV3_S (4.79MB, 92.78%)
  ÍT TÀI NGUYÊN ◄───────┤
  (< 2MB model)         │
                         │  ★ MobileNetV1_0.25 (1.39MB, 90.79%)  ← NHÚNG TỐI ƯU
                         │
                         │  ★ MobileNetV2_025 (1.80MB, 82.43%)
                         │  ★ MobileNetV1_0.18 (0.87MB, 82.33%)  ← SIÊU NHẸ
                         ▼
                    ĐỘ CHÍNH XÁC THẤP (Avg ~82%)
```

### 5.4. Gợi ý lựa chọn theo phần cứng Camera

| Loại Camera                | RAM       | Storage     | Model đề xuất             | width_mult |
| -------------------------- | --------- | ----------- | ------------------------- | ---------- |
| **SoC giá rẻ** (< $10)     | < 256MB   | < 8MB flash | MobileNetV1               | 0.18–0.25  |
| **SoC tầm trung** ($10–30) | 256MB–1GB | 16–64MB     | MobileNetV1 hoặc V3_Small | 0.25–0.50  |
| **SoC cao cấp** (> $30)    | > 1GB     | > 64MB      | MobileNetV2 hoặc V3_Large | 0.50–1.0   |
| **Có NPU/AI accelerator**  | > 2GB     | > 128MB     | MobileNetV2/V3 ONNX       | 1.0        |

### 5.5. Quyết định cuối cùng cho Indoor Camera

> **Lựa chọn: MobileNetV1 0.25** (hoặc MobileNetV2 tùy phần cứng camera)
>
> **Lý do:**
> - Size 1.39 MB → fit vào flash memory camera
> - LFW 98.75% → đủ chính xác cho indoor access control
> - Latency ~1.2ms trên CPU → real-time capable
> - Efficiency Score cao nhất trong nhóm (~77)

---

## 6. Giảm Model Size & Trade-off Accuracy

### 6.1. Tổng quan các kỹ thuật đã áp dụng

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE TỐI ƯU MODEL SIZE                             │
│                                                                            │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────────┐ │
│  │ 1. Architecture │   │ 2. Width         │   │ 3. ONNX Optimization    │ │
│  │    Design       │   │    Multiplier    │   │    (constant folding,   │ │
│  │  (MobileNet +   │──▶│  (α = 0.25)     │──▶│     graph optimization) │ │
│  │   DW Sep Conv)  │   │                  │   │                         │ │
│  └─────────────────┘   └─────────────────┘   └──────────────────────────┘ │
│                                                                            │
│  Tùy chọn nâng cao (chưa triển khai trong project):                       │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────────┐ │
│  │ 4. Quantization │   │ 5. Pruning      │   │ 6. Knowledge            │ │
│  │  (FP32 → INT8)  │   │  (cắt filters)  │   │    Distillation         │ │
│  │  Giảm 2-4× size │   │  Giảm 20-60%    │   │  (Teacher → Student)   │ │
│  └─────────────────┘   └─────────────────┘   └──────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2. Kỹ thuật 1: Depthwise Separable Convolution

**So sánh chi phí:**

```
Normal Conv2D(64, 128, kernel_size=3):
  Params = 128 × 64 × 3 × 3 = 73,728
  FLOPs (input 56×56) ≈ 231 triệu

Depthwise Separable Conv(64, 128):
  Depthwise (64, 64, 3×3, groups=64):  Params = 64 × 1 × 3 × 3  = 576
  Pointwise (64, 128, 1×1):            Params = 128 × 64 × 1 × 1 = 8,192
  Tổng: 8,768 params

  ► Giảm 73,728 / 8,768 ≈ 8.4× params! 🚀
```

### 6.3. Kỹ thuật 2: Width Multiplier

**Phân tích chi tiết trade-off:**

| width_mult | Params | ONNX Size | LFW (%) | Avg Acc (%) | Giảm params | Giảm accuracy |
| ---------- | ------ | --------- | ------- | ----------- | ----------- | ------------- |
| 1.0 (V2)   | 2.29M  | 8.67 MB   | 99.55   | 94.11       | Baseline    | Baseline      |
| 0.25 (V2)  | 0.46M  | 1.80 MB   | 95.17   | 82.43       | **-80%**    | **-11.7%**    |
| 0.25 (V1)  | 0.36M  | 1.39 MB   | 98.75   | 90.79       | **-84%**    | **-3.3%**     |
| 0.18 (V1)  | 0.22M  | 0.87 MB   | 94.48   | 82.33       | **-90%**    | **-11.8%**    |

**Điểm nổi bật:** MobileNetV1 0.25 giảm **84% params** nhưng chỉ giảm **3.3% accuracy trung bình** → Trade-off cực tốt!

### 6.4. Kỹ thuật 3: ONNX Export Optimization

```
.ckpt (Training checkpoint)         →  ~30 MB
  │  Chứa: model + optimizer + scheduler + head
  │
  ▼ convert_to_pth.py (Trích backbone)
.pth (Backbone weights only)        →  ~1.5 MB
  │  Chứa: chỉ model state_dict
  │
  ▼ onnx_export.py (Graph optimization)
.onnx (Optimized inference model)   →  ~1.4 MB
  │  Constant folding, dead code elimination
  │  Opset version 16
  │
  ► Giảm 30 MB → 1.4 MB (~21× nhỏ hơn!)
```

### 6.5. Kỹ thuật nâng cao (Ý tưởng nghiên cứu thêm)

#### A. Post-Training Quantization (FP32 → INT8)

```
Kỳ vọng:
  • Model size giảm 2-4× (1.4 MB → 0.35-0.7 MB)
  • Inference latency giảm 2×
  • Accuracy giảm < 1%

Cách thực hiện:
  ONNX Runtime quantization hoặc NNE Quantize tool
```

#### B. Knowledge Distillation

```
Teacher: Sphere36 (34.6M params, 95.53% avg accuracy)
    │
    │  Soft labels + Feature matching
    ▼
Student: MobileNetV1_0.25 (0.36M params)
    → Accuracy kỳ vọng tăng 1-3% so với train thông thường
    → Từ 90.79% → có thể đạt ~92-93% avg accuracy
```

#### C. Structured Pruning

```
Quy trình:
  1. Train model bình thường
  2. Đánh giá tầm quan trọng từng filter (L1-norm)
  3. Cắt 20-30% filters ít quan trọng
  4. Fine-tune 5-10 epochs
  → Giảm thêm 20-30% params, accuracy giảm < 1%
```

### 6.6. Bảng tổng kết Trade-off

```
┌───────────────────────────────────────────────────────────────────┐
│           MODEL SIZE vs ACCURACY TRADE-OFF                        │
│                                                                   │
│  Accuracy                                                         │
│  (Avg %)                                                          │
│ 95% ─  ■ Sphere36 (135MB)                                        │
│        ■ MobileNetV2 (8.67MB)                                     │
│ 93% ─  ■ MobileNetV3_Small (4.79MB)                              │
│                                                                   │
│ 91% ─  ■ MobileNetV1_0.25 (1.39MB) ← SWEET SPOT 🎯              │
│                                                                   │
│ 85% ─                                                             │
│                                                                   │
│ 82% ─  ■ MobileNetV1_0.18 (0.87MB)                              │
│        ■ MobileNetV2_025 (1.80MB)                                 │
│        │                                                          │
│        └──────┬───────┬───────┬───────┬───────┬──── Size (MB)    │
│               1       2       5       10      50     100          │
└───────────────────────────────────────────────────────────────────┘
```

---

## 7. ONNX → BIN: Deploy lên Camera

### 7.1. Tổng quan Pipeline Deployment

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐
│ PyTorch  │───▶│  ONNX    │───▶│ NNE Convert  │───▶│ NNE Quantize│───▶│  .bin   │
│  .pth    │    │  .onnx   │    │  (→ NNE IR)  │    │ (FP32→INT8) │    │ (Camera)│
│          │    │          │    │              │    │             │    │         │
│ Training │    │ Portable │    │ Camera SDK   │    │ Tối ưu chip │    │ Deploy! │
└──────────┘    └──────────┘    └──────────────┘    └─────────────┘    └─────────┘
```

### 7.2. Bước 1: PyTorch → ONNX (Đã hoàn thành trong project)

```bash
# Export model sang ONNX
python onnx_export.py \
    -w weights/mobilenetv1_0.25_mcp.pth \   # Weights file
    -n mobilenetv1                           # Architecture name

# Hoặc với dynamic batch size (cho server):
python onnx_export.py \
    -w weights/mobilenetv2_mcp.pth \
    -n mobilenetv2 \
    --dynamic
```

**Cấu hình ONNX Export:**
| Tham số               | Giá trị          | Ý nghĩa                          |
| --------------------- | ---------------- | -------------------------------- |
| `opset_version`       | 16               | Phiên bản ONNX operators         |
| `do_constant_folding` | True             | Tối ưu graph tại export time     |
| `input_names`         | ['input']        | Tên input tensor                 |
| `output_names`        | ['output']       | Tên output tensor                |
| `input_shape`         | (1, 3, 112, 112) | Batch=1, 3 channels RGB, 112×112 |

### 7.3. Bước 2: ONNX Verification (Đảm bảo ONNX chính xác)

Trước khi convert sang BIN, verify ONNX model bằng benchmark tool:

```bash
# Đánh giá accuracy + resource metrics:
python evaluate_onnx_benchmark.py \
    --model weights/mobilenetv1_0.25_mcp.onnx

# Chỉ đo resource (nhanh, ~10 giây):
python evaluate_onnx_benchmark.py --metrics-only

# So sánh tất cả models:
python evaluate_onnx_benchmark.py --output results.csv
```

**Feature extraction trong ONNX:**
```
Ảnh 112×112 BGR
  ├─ Preprocess: pixel / 127.5 - 1.0 + BGR→RGB
  │    ├─ ONNX Inference → embedding_original (512-D)
  │
  ├─ Flip ngang
  │    ├─ ONNX Inference → embedding_flipped (512-D)
  │
  └─ Concat: [original, flipped] → feature_vector (1024-D)
       └─ Cosine Similarity → Decision
```

### 7.4. Bước 3: ONNX → BIN (NNE Compile Tool)

**NNE (Neural Network Engine)** là toolchain của camera SoC manufacturer, chuyển ONNX model sang format tối ưu cho NPU/AI accelerator trên camera.

```
NNE Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. NNE Convert                                                 │
│     Input:  model.onnx                                          │
│     Output: model.nne (NNE Intermediate Representation)         │
│     Chức năng: Chuyển ONNX graph → NNE graph format             │
│                                                                 │
│  2. NNE Quantize                                                │
│     Input:  model.nne + calibration_data (representative images)│
│     Output: model_int8.nne                                      │
│     Chức năng: FP32 → INT8, dùng calibration data để minimize   │
│               quantization error                                │
│                                                                 │
│  3. NNE Check                                                   │
│     Input:  model_int8.nne                                      │
│     Output: Validation report                                   │
│     Chức năng: Verify model compatibility với target hardware    │
│                                                                 │
│  4. NNE Simulate                                                │
│     Input:  model_int8.nne + test_image                         │
│     Output: Simulated inference result                          │
│     Chức năng: Mô phỏng inference trên PC trước khi deploy     │
│                                                                 │
│  5. Final Output                                                │
│     model_int8.bin → Flash vào camera SoC                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5. Quantization trong NNE

| Aspect              | FP32 (Original)       | INT8 (Quantized) | Thay đổi                   |
| ------------------- | --------------------- | ---------------- | -------------------------- |
| **Precision**       | 32-bit floating point | 8-bit integer    | 4× nhỏ hơn per value       |
| **Model Size**      | ~1.4 MB               | ~0.35 MB         | Giảm ~4×                   |
| **Inference Speed** | Baseline              | 2-4× nhanh hơn   | NPU tối ưu cho INT8        |
| **Accuracy**        | Baseline              | Giảm < 1-2%      | Chấp nhận được             |
| **Power**           | Baseline              | Giảm 2-3×        | Quan trọng cho camera 24/7 |

**Calibration Data:** Cần 100-500 ảnh representative (ảnh khuôn mặt aligned 112×112) để NNE tính toán optimal quantization parameters (scale, zero_point) cho từng layer.

### 7.6. Deployment Architecture trên Camera

```
┌──────────────────────────────────────────────────────────────────┐
│                    INDOOR ADVANCE CAMERA                         │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐│
│  │  Camera  │───▶│  ISP (Image  │───▶│  Main CPU (ARM)         ││
│  │  Sensor  │    │  Signal      │    │  ├─ Face Detection      ││
│  │  (CMOS)  │    │  Processor)  │    │  ├─ Face Alignment      ││
│  └──────────┘    └──────────────┘    │  └─ Pre/Post processing ││
│                                      └──────────┬──────────────┘│
│                                                  │               │
│                                                  ▼               │
│                                      ┌─────────────────────────┐│
│                                      │  NPU / AI Accelerator   ││
│                                      │  ├─ Load .bin model     ││
│                                      │  ├─ CNN Inference       ││
│                                      │  │    (INT8 optimized)  ││
│                                      │  └─ Output: Embedding   ││
│                                      │       512-D vector      ││
│                                      └──────────┬──────────────┘│
│                                                  │               │
│                                                  ▼               │
│                                      ┌─────────────────────────┐│
│                                      │  Main CPU (ARM)         ││
│                                      │  ├─ Cosine Similarity   ││
│                                      │  ├─ Database Matching   ││
│                                      │  └─ Decision: Allow/Deny││
│                                      └─────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Flash Memory: model.bin (~0.35 MB) + Face DB (~50 KB)     ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### 7.7. Tóm tắt toàn bộ Pipeline: Từ Training đến Camera

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  ① TRAIN    │───▶│  ② EXPORT   │───▶│  ③ BENCHMARK│───▶│  ④ CONVERT   │
│  PyTorch    │    │  .pth→.onnx │    │  Accuracy + │    │  ONNX → NNE  │
│  MS1MV2     │    │  Constant   │    │  Resource    │    │  → Quantize  │
│  CosFace    │    │  Folding    │    │  Metrics     │    │  → .bin      │
│  30 epochs  │    │  Opset 16   │    │  4 datasets  │    │  INT8        │
└─────────────┘    └─────────────┘    └──────┬───────┘    └──────┬───────┘
                                              │                   │
                                   ┌──────────▼───────────┐      │
                                   │  Đạt yêu cầu?       │      │
                                   │  • Acc > threshold    │ YES  │
                                   │  • Size < limit       │─────▶│
                                   │  • Latency < target   │      │
                                   └──────────┬───────────┘      │
                                          NO  │                   │
                                   ┌──────────▼───────────┐      │
                                   │ Điều chỉnh:          │      │
                                   │  • width_mult        │      │
                                   │  • backbone           │      │
                                   │  • training config    │      │
                                   └──────────────────────┘      │
                                                                  │
                                                    ┌─────────────▼───────┐
                                                    │  ⑤ DEPLOY lên Camera│
                                                    │  Flash .bin → SoC   │
                                                    │  NPU inference      │
                                                    │  Real-time FR       │
                                                    └─────────────────────┘
```

---

## 📌 Gợi ý cấu trúc Slide

### Slide 1: Title
- **Tiêu đề:** "Face Recognition cho Indoor Advance Camera"
- **Subtitle:** Từ Training đến Deployment trên SoC
- **Hình ảnh:** Camera indoor + khuôn mặt

### Slide 2-3: Tổng quan bài toán
- Bối cảnh indoor camera
- Pipeline Face Recognition (sơ đồ)
- Ràng buộc phần cứng

### Slide 4-5: Kiến trúc mô hình
- Bảng so sánh models
- MobileNet architecture diagram
- Inverted Residual Block
- GDC (điểm đặc biệt)

### Slide 6: Data sử dụng
- MS1MV2 training data stats
- 4 validation benchmarks
- Data preprocessing pipeline

### Slide 7-8: Cách Training
- Hyperparameters (bảng)
- CosFace loss explanation
- Training pipeline diagram
- LR schedule visualization

### Slide 9-10: Lựa chọn mô hình
- Bảng benchmark đầy đủ (accuracy + resources)
- Trade-off chart
- Efficiency ranking
- Quyết định cuối cùng + lý do

### Slide 11-12: Giảm Model Size
- 3 kỹ thuật chính (DW Sep Conv, Width Mult, ONNX Opt)
- Trade-off table: Size giảm vs Accuracy giảm
- Sweet spot analysis
- Kỹ thuật nâng cao (quantization, KD, pruning)

### Slide 13-14: ONNX → BIN Deploy
- Deployment pipeline diagram
- NNE tool workflow (Convert → Quantize → Check → Simulate)
- Camera architecture diagram
- Final specs (model size, latency trên target hardware)

### Slide 15: Kết luận & Kết quả
- Tóm tắt lựa chọn cuối cùng
- Bảng specs: model size, accuracy, latency
- Demo video (nếu có)
- Hướng phát triển: KD, quantization-aware training

---

## 📚 Tài liệu tham khảo

### Papers về kiến trúc
- **MobileNetV1:** Howard et al., 2017 — [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
- **MobileNetV2:** Sandler et al., 2018 — [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
- **MobileNetV3:** Howard et al., 2019 — [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
- **MobileFaceNet:** Chen et al., 2018 — [arXiv:1804.07573](https://arxiv.org/abs/1804.07573) (GDC)

### Papers về Loss Function
- **CosFace:** Wang et al., 2018 — [arXiv:1801.09414](https://arxiv.org/abs/1801.09414)
- **ArcFace:** Deng et al., 2019 — [arXiv:1801.07698](https://arxiv.org/abs/1801.07698)
- **SphereFace:** Liu et al., 2017 — [arXiv:1704.08063](https://arxiv.org/abs/1704.08063)

### Papers về tối ưu hóa
- **Knowledge Distillation:** Hinton et al., 2015 — [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- **Quantization:** Jacob et al., 2018 — [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
- **Pruning:** Li et al., 2017 — [arXiv:1608.08710](https://arxiv.org/abs/1608.08710)

### Datasets
- **MS1MV2:** [InsightFace / MS-Celeb-1M cleaned](https://github.com/deepinsight/insightface)
- **LFW:** [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
- **CALFW / CPLFW:** Cross-Age / Cross-Pose LFW variants

---

> 📝 **Tài liệu này dựa trên code thực tế và kết quả benchmark từ project [face-recognition](https://github.com/yakhyo/face-recognition).**
> Tất cả số liệu trong bảng benchmark được lấy từ kết quả thí nghiệm thực tế (`result1.csv`, `result2.csv`).
