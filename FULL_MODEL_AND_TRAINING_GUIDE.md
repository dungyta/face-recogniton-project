# 🧬 Tài Liệu Toàn Diện: Kiến Trúc Mô Hình & Quá Trình Training/Inference

> Tài liệu siêu chi tiết dựa trên mã nguồn thực tế của project, giải thích từng lớp mạng, cách model học, và sự khác biệt giữa Training vs Inference.

---

## Mục Lục

1. [Tổng quan kiến trúc & Tại sao chọn MobileNet](#1-tổng-quan-kiến-trúc)
2. [Chi tiết từng lớp bên trong MobileNetV1](#2-mobilenetv1-chi-tiết)
3. [Chi tiết từng lớp bên trong MobileNetV2](#3-mobilenetv2-chi-tiết)
4. [Các Building Block cốt lõi (layers.py)](#4-building-blocks)
5. [Pipeline Training hoàn chỉnh](#5-pipeline-training)
6. [Cách Model học được (Forward/Backward)](#6-cách-model-học)
7. [Pipeline Validation/Testing](#7-pipeline-testing)
8. [Sự khác biệt Training vs Inference](#8-training-vs-inference)

---

## 1. Tổng quan kiến trúc

### 1.1 Các model có trong project

| Model           | Kiến trúc lõi                    | Params     | Ưu điểm                   | Nhược điểm               |
| --------------- | -------------------------------- | ---------- | ------------------------- | ------------------------ |
| **Sphere20/36** | Standard Conv + Residual         | 24-34M     | Accuracy cao nhất (95%+)  | Quá nặng cho camera      |
| **MobileNetV1** | Depthwise Separable Conv         | 0.22-0.36M | Siêu nhẹ, inference nhanh | Không có skip connection |
| **MobileNetV2** | Inverted Residual Block          | 0.46-2.29M | Cân bằng tốt, có residual | Hơi phức tạp hơn V1      |
| **MobileNetV3** | Inverted Residual + SE + H-Swish | 1.25-3.52M | Kiến trúc NAS tối ưu      | Phức tạp nhất            |

### 1.2 Tại sao chọn MobileNet cho Camera Indoor?

```text
 ❌ Standard Conv (ResNet/VGG/Sphere):
    - 1 bộ lọc 3×3 chà lên TẤT CẢ kênh cùng lúc
    - Params = in_ch × out_ch × 3 × 3
    - VD: Conv(64→128) = 64 × 128 × 9 = 73,728 params  ← QUÁ NẶNG

 ✅ Depthwise Separable Conv (MobileNet):
    - Tách 2 bước: (1) lọc spatial riêng, (2) trộn channel sau
    - Params = in_ch × 9 + in_ch × out_ch
    - VD: DWSep(64→128) = 576 + 8,192 = 8,768 params   ← GIẢM 8.4 LẦN!
```

**Kết luận:** MobileNet sinh ra để chạy trên thiết bị yếu. Giảm 8-9× phép tính mà vẫn giữ khả năng "nhìn" tốt.

---

## 2. MobileNetV1 Chi Tiết

### 2.1 Kiến trúc tổng thể (từ `mobilenetv1.py`)

MobileNetV1 gồm **3 Stage** nối tiếp + 1 lớp output GDC:

```text
┌─────────────────────────── MobileNetV1 (width_mult=0.25) ───────────────────────────┐
│                                                                                      │
│  INPUT: Ảnh khuôn mặt [1, 3, 112, 112]                                              │
│         │                                                                            │
│  ╔══════╧════════════════════════ STAGE 1 ════════════════════════════════╗           │
│  ║  Conv2dNormActivation(3 → 8, k=3, s=1) + BN + PReLU                  ║           │
│  ║       │   Output: [1, 8, 112, 112]                                    ║           │
│  ║  DWSepConv(8 → 16, s=1)   Output: [1, 16, 112, 112]                  ║           │
│  ║  DWSepConv(16 → 32, s=2)  Output: [1, 32, 56, 56]    ← Giảm 2×      ║           │
│  ║  DWSepConv(32 → 32, s=1)  Output: [1, 32, 56, 56]                    ║           │
│  ║  DWSepConv(32 → 56, s=2)  Output: [1, 56, 28, 28]    ← Giảm 2×      ║           │
│  ║  DWSepConv(56 → 56, s=1)  Output: [1, 56, 28, 28]                    ║           │
│  ╚═══════════════════════════════════════════════════════════════════════╝           │
│         │                                                                            │
│  ╔══════╧════════════════════════ STAGE 2 ════════════════════════════════╗           │
│  ║  DWSepConv(56 → 112, s=2)  Output: [1, 112, 14, 14]  ← Giảm 2×      ║           │
│  ║  DWSepConv(112 → 112, s=1) × 5 lần                                   ║           │
│  ║       │   Output: [1, 112, 14, 14]                                    ║           │
│  ╚═══════════════════════════════════════════════════════════════════════╝           │
│         │                                                                            │
│  ╔══════╧════════════════════════ STAGE 3 ════════════════════════════════╗           │
│  ║  DWSepConv(112 → 224, s=2)  Output: [1, 224, 7, 7]   ← Giảm 2×      ║           │
│  ║  DWSepConv(224 → 224, s=1)  Output: [1, 224, 7, 7]                   ║           │
│  ╚═══════════════════════════════════════════════════════════════════════╝           │
│         │                                                                            │
│  ╔══════╧════════════════════════ OUTPUT (GDC) ═══════════════════════════╗           │
│  ║  LinearBlock: Conv2d(224, 224, k=7, groups=224) → [1, 224, 1, 1]      ║           │
│  ║  Flatten → [1, 224]                                                   ║           │
│  ║  Linear(224 → 512) + BatchNorm1d → [1, 512]                          ║           │
│  ╚═══════════════════════════════════════════════════════════════════════╝           │
│         │                                                                            │
│  OUTPUT: Embedding Vector [1, 512]                                                   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Width Multiplier (α) hoạt động ra sao?

```python
# Trong code thực tế (mobilenetv1.py dòng 14-15):
filters = [32, 64, 128, 256, 512, 1024]                    # Gốc
filters = [_make_divisible(f * width_mult) for f in filters] # Sau khi ép

# width_mult = 0.25 → filters = [8, 16, 32, 56, 112, 224]
# width_mult = 0.18 → filters = [8, 16, 24, 48, 96, 184]
```

| width_mult | filters thực tế               | Params | ONNX Size |
| ---------- | ----------------------------- | ------ | --------- |
| 1.0 (Full) | [32, 64, 128, 256, 512, 1024] | ~3.2M  | ~12 MB    |
| 0.25       | [8, 16, 32, 56, 112, 224]     | ~0.36M | 1.39 MB   |
| 0.18       | [8, 16, 24, 48, 96, 184]      | ~0.22M | 0.87 MB   |

---

## 3. MobileNetV2 Chi Tiết

### 3.1 Inverted Residual Block (từ `mobilenetv2.py`)

Khác V1, MobileNetV2 dùng **Inverted Residual** với 3 bước và **skip connection**:

```text
 Input (Mỏng, ít channels)
   │
   ├───── 1. Expansion (1×1 Conv): Phình channels lên t lần (t = expand_ratio = 6)
   │         [C] → [C × 6]
   │         + BatchNorm + PReLU
   │
   ├───── 2. Depthwise (3×3 DW Conv): Lọc spatial, groups = C×6
   │         [C × 6] → [C × 6]
   │         + BatchNorm + PReLU
   │
   ├───── 3. Projection (1×1 Conv Linear): Nén về mỏng, KHÔNG activation
   │         [C × 6] → [C']
   │         + BatchNorm (không PReLU!)
   │
   └───── + Residual Connection (if stride==1 AND in_ch == out_ch)
           output = input + conv(input)   ← Giữ thông tin gốc!
```

**Tại sao gọi "Inverted"?**
- ResNet thường: **Rộng → Hẹp → Rộng** (bottleneck ở giữa)
- MobileNetV2: **Hẹp → Rộng → Hẹp** (expansion ở giữa = inverted!)

### 3.2 Cấu hình 17 khối Inverted Residual

```text
┌─────────────── MobileNetV2 (width_mult=1.0) ────────────────┐
│                                                               │
│  Conv2dNormActivation(3 → 32, s=1) + BN + PReLU              │
│       │                                                       │
│  ┌────┴──── Inverted Residual Blocks (t, c, n, s) ─────────┐│
│  │  t=1, c=16,  n=1, s=1  (Không expand, 1 block)          ││
│  │  t=6, c=24,  n=2, s=2  (Expand 6×, 2 blocks, ↓56×56)   ││
│  │  t=6, c=32,  n=3, s=2  (Expand 6×, 3 blocks, ↓28×28)   ││
│  │  t=6, c=64,  n=4, s=2  (Expand 6×, 4 blocks, ↓14×14)   ││
│  │  t=6, c=96,  n=3, s=1  (Expand 6×, 3 blocks, 14×14)    ││
│  │  t=6, c=160, n=3, s=2  (Expand 6×, 3 blocks, ↓7×7)     ││
│  │  t=6, c=320, n=1, s=1  (Expand 6×, 1 block, 7×7)       ││
│  │            Tổng: 17 Inverted Residual Blocks              ││
│  └───────────────────────────────────────────────────────────┘│
│       │                                                       │
│  Conv2dNormActivation(320 → 512, k=1) + BN + PReLU           │
│       │                                                       │
│  GDC: Conv7×7(groups=512) → Flatten → Linear(512→512) + BN1d │
│       │                                                       │
│  OUTPUT: [1, 512] Embedding                                   │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Building Blocks (từ `layers.py`)

### 4.1 Conv2dNormActivation
Khối xây dựng cơ bản = **Conv2d + BatchNorm + PReLU** gộp lại:
```text
Input → [Conv2d] → [BatchNorm2d] → [PReLU] → Output
```
PReLU (Parametric ReLU): Cho phép gradient chảy cả khi giá trị âm (khác ReLU bị chặn ở 0).

### 4.2 DepthWiseSeparableConv2d
```text
Input [B, M, H, W]
  │
  ├─ Depthwise: Conv2d(M, M, k=3, groups=M) + BN + PReLU
  │     Mỗi channel có 1 bộ lọc 3×3 riêng (tổng M bộ lọc)
  │     Params = M × 9
  │
  └─ Pointwise: Conv2d(M, N, k=1) + BN + PReLU
        Trộn M channels thành N channels
        Params = M × N
  │
Output [B, N, H', W']
```

### 4.3 GDC (Global Depthwise Convolution) — Điểm đặc biệt
```text
Feature Map [B, C, 7, 7]     (VD: C=224 cho V1, C=512 cho V2)
  │
  ├─ LinearBlock: Conv2d(C, C, k=7, groups=C) + BN   ← Kernel = bằng feature map!
  │     Output: [B, C, 1, 1]    Nắm bắt VỊ TRÍ đặc trưng (mắt, mũi, miệng)
  │
  ├─ Flatten → [B, C]
  │
  └─ Linear(C, 512) + BatchNorm1d → [B, 512]   ← Vector Embedding cuối cùng

Tại sao KHÔNG dùng Global Average Pooling (GAP)?
  • GAP: Lấy trung bình → MẤT hết thông tin vị trí
  • GDC: Kernel 7×7 riêng cho từng channel → GIỮ thông tin mắt ở đâu, mũi ở đâu
```

---

## 5. Pipeline Training Hoàn Chỉnh

### 5.1 Sơ đồ tổng thể (dựa trên `train.py`)

```text
═══════════════════════════════════════════════════════════════════════════════════
                           TRAINING PIPELINE HOÀN CHỈNH
═══════════════════════════════════════════════════════════════════════════════════

 ① KHỞI TẠO                    ② NẠP DỮ LIỆU                ③ VÒNG LẶP TRAINING
 ──────────────                 ──────────────                ───────────────────

 parse_arguments()              ImageFolder(root)             for epoch in 0..29:
      │                              │                             │
 setup_seed()                   RandomHorizontalFlip()        ┌────┴────────────┐
      │                         ToTensor()                    │ train_one_epoch()│
 Tạo Model:                    Normalize(                     │                 │
 MobileNetV1/V2(               mean=[0.485,0.456,0.406]      │ for batch in    │
   embedding_dim=512,           std=[0.229,0.224,0.225])      │   dataloader:   │
   width_mult=α)                     │                        │                 │
      │                         DataLoader(                   │  Forward ──┐    │
 Tạo Head:                       batch=512,                   │  Loss    ──┤    │
 MarginCosineProduct(             workers=8,                  │  Backward──┤    │
   in=512,                        pin_memory=True)            │  Update  ──┘    │
   out=85742,                                                 │                 │
   s=30.0, m=0.40)                                            └────┬────────────┘
      │                                                            │
 Loss: CrossEntropyLoss()                                     lr_scheduler.step()
      │                                                            │
 Optimizer: SGD(                                              save _last.ckpt
   lr=0.1,                                                         │
   momentum=0.9,                                              evaluate → accuracy
   weight_decay=5e-4)                                              │
      │                                                       if acc > best:
 Scheduler: MultiStepLR(                                        save _best.ckpt
   milestones=[10,20,25],                                          │
   gamma=0.1)                                                 EarlyStopping(
      │                                                         patience=10)
 EarlyStopping(patience=10)

═══════════════════════════════════════════════════════════════════════════════════
```

### 5.2 Hyperparameters chi tiết

| Tham số        | Giá trị      | Tại sao?                                    |
| -------------- | ------------ | ------------------------------------------- |
| Batch size     | 512          | Đủ lớn cho SGD ổn định trên 5.8M ảnh        |
| Epochs         | 30           | Đủ hội tụ với LR schedule 3 bậc             |
| LR ban đầu     | 0.1          | Giá trị chuẩn cho SGD trên face recognition |
| Milestones     | [10, 20, 25] | Giảm 10× tại mỗi mốc → tinh chỉnh dần       |
| Momentum       | 0.9          | Tăng tốc hội tụ, vượt qua local minima      |
| Weight Decay   | 5e-4         | Regularization chống overfitting            |
| Early Stopping | patience=10  | Dừng nếu 10 epoch liên tiếp không cải thiện |

---

## 6. Cách Model Học Được (Chi tiết Forward & Backward)

### 6.1 Forward Pass — Một batch ảnh đi qua hệ thống

```text
  [256 ảnh × 3 × 112 × 112]
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │              BACKBONE (MobileNetV1/V2)              │
  │  Conv → DWSep → DWSep → ... → DWSep → GDC          │
  │  Mỗi lớp Conv: nhân kernel với input → feature map  │
  │  BatchNorm: chuẩn hóa → ổn định training            │
  │  PReLU: phi tuyến → model có thể học pattern phức    │
  └───────────────────────┬─────────────────────────────┘
                          │
                   [256 × 512]  ← 256 embedding vectors
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │           COSFACE HEAD (MarginCosineProduct)        │
  │                                                     │
  │  Bước 1: Normalize embedding & weight               │
  │    emb_norm = F.normalize(embedding)   → ‖e‖ = 1    │
  │    w_norm   = F.normalize(self.weight) → ‖w‖ = 1    │
  │                                                     │
  │  Bước 2: Tính cosine similarity                     │
  │    cosine = emb_norm @ w_norm.T   → [256, 85742]    │
  │    (Mỗi embedding so sánh với 85,742 class)         │
  │                                                     │
  │  Bước 3: Trừ margin cho class đúng                  │
  │    one_hot = F.one_hot(label, 85742)                │
  │    output = s × (cosine − one_hot × m)              │
  │           = 30 × (cosine − one_hot × 0.4)           │
  │    → Class đúng bị PHẠT thêm 0.4, khó hơn          │
  └───────────────────────┬─────────────────────────────┘
                          │
                   [256 × 85742]  ← Logits cho 85,742 người
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │              CROSS ENTROPY LOSS                     │
  │  loss = −log(softmax(output)[correct_class])        │
  │  → Đo "sai bao nhiêu" khi đoán nhãn                │
  └───────────────────────┬─────────────────────────────┘
                          │
                     Loss Value (VD: 2.5)
```

### 6.2 Backward Pass — Model "sửa sai"

```text
  Loss = 2.5
     │
     ▼  loss.backward()
  ┌─────────────────────────────────────────────────────┐
  │              TÍNH GRADIENT (Chain Rule)              │
  │                                                     │
  │  PyTorch tự động tính đạo hàm riêng ∂Loss/∂w       │
  │  cho MỌI weight trong toàn bộ mạng:                 │
  │                                                     │
  │  CosFace Head weights   ← ∂L/∂w_head               │
  │  GDC Linear weights     ← ∂L/∂w_gdc                │
  │  Stage 3 DWSep weights  ← ∂L/∂w_s3                 │
  │  Stage 2 DWSep weights  ← ∂L/∂w_s2                 │
  │  Stage 1 Conv weights   ← ∂L/∂w_s1                 │
  │                                                     │
  │  Gradient chảy NGƯỢC từ Loss → Output → ... → Input │
  └───────────────────────┬─────────────────────────────┘
                          │
                          ▼  optimizer.step()
  ┌─────────────────────────────────────────────────────┐
  │              CẬP NHẬT TRỌNG SỐ (SGD)               │
  │                                                     │
  │  w_new = w_old − lr × gradient − wd × w_old        │
  │                                                     │
  │  VD epoch 0:  w = w − 0.1 × grad − 5e-4 × w       │
  │  VD epoch 15: w = w − 0.01 × grad − 5e-4 × w      │
  │                    ↑ LR đã giảm 10× tại epoch 10   │
  │                                                     │
  │  Momentum: grad_new = 0.9 × grad_old + grad_now    │
  │  → Giúp vượt qua điểm cực tiểu cục bộ              │
  └─────────────────────────────────────────────────────┘
```

### 6.3 Tại sao CosFace giúp model học tốt hơn Softmax?

```text
  SOFTMAX thường:                    COSFACE (margin m=0.4):
  output = s × cos(θ)               output = s × (cos(θ) − 0.4)

  ┌──────────────────────┐           ┌──────────────────────┐
  │  Người A   Người B   │           │  Người A    Người B  │
  │     ●●●  ●●●         │           │   ●●       ●●       │
  │      ●●●●●           │           │    ●●     ●●        │
  │       ●●●            │ ← Xát     │         ↕           │ ← Cách xa
  │      ●●●●●           │   nhau    │    ●●  MARGIN ●●    │   bởi Margin
  │     ●●●  ●●●         │           │   ●●       ●●       │
  └──────────────────────┘           └──────────────────────┘
  → Dễ nhầm khi ánh sáng thay đổi   → An toàn, robust hơn nhiều
```

---

## 7. Pipeline Testing/Validation

### 7.1 Evaluation trong Training (từ `evaluate.py`)

```text
═══════════════════════════════════════════════════════════════
                   VALIDATION PIPELINE
═══════════════════════════════════════════════════════════════

 Sau mỗi epoch, chạy evaluate trên tập CALFW (6000 cặp ảnh)

 Với MỖI cặp ảnh (img1, img2):
 ┌─────────────────────────────────────────────────────────┐
 │  img1 (112×112)                                         │
 │    ├─ Original → model() → emb_original (512-D)         │
 │    └─ Flip ngang → model() → emb_flipped (512-D)        │
 │    → Concat: feature1 = [emb_orig, emb_flip] (1024-D)   │
 │                                                         │
 │  img2 (112×112) → Tương tự → feature2 (1024-D)         │
 │                                                         │
 │  similarity = feature1 · feature2 / (‖f1‖ × ‖f2‖)      │
 │             = Cosine Similarity ∈ [-1, 1]               │
 └───────────────────────────┬─────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────┐
 │              K-FOLD CROSS VALIDATION (K=10)             │
 │                                                         │
 │  6000 cặp chia thành 10 phần (mỗi phần 600 cặp)        │
 │                                                         │
 │  Lặp 10 lần (fold):                                     │
 │    9 phần (5400 cặp): Dò threshold tối ưu               │
 │      → Thử threshold từ -1.0 → +1.0 (bước 0.005)       │
 │      → Chọn threshold cho accuracy cao nhất              │
 │                                                         │
 │    1 phần (600 cặp): Áp threshold → tính accuracy       │
 │      → if similarity > threshold: SAME else DIFFERENT    │
 │                                                         │
 │  Kết quả: mean(10 accuracy) ± std                       │
 └─────────────────────────────────────────────────────────┘
```

### 7.2 Tại sao dùng Original + Flipped?

Model nhìn 2 góc của khuôn mặt (bình thường + lật gương), ghép 2 embedding lại thành 1024-D. Kỹ thuật này tăng accuracy khoảng 0.5-1% vì nó bù đắp sự bất đối xứng bên trái/phải khuôn mặt.

---

## 8. Sự Khác Biệt Training vs Inference

### 8.1 Bảng so sánh tổng quát

| Khía cạnh           | TRAINING                         | INFERENCE (Triển khai)              |
| ------------------- | -------------------------------- | ----------------------------------- |
| **Mục đích**        | Dạy model phân biệt 85,742 người | Nhận diện 1 người cụ thể từ DB      |
| **Input**           | 5.8M ảnh đã align sẵn 112×112    | Video stream từ camera real-time    |
| **Face Detection**  | KHÔNG CẦN (ảnh đã crop sẵn)      | CẦN (RetinaFace/InsightFace)        |
| **Face Alignment**  | KHÔNG CẦN (đã align sẵn)         | CẦN (5-point landmark → transform)  |
| **Augmentation**    | CÓ (Flip, Normalize)             | KHÔNG CÓ                            |
| **CosFace Head**    | CÓ (85,742-way so sánh)          | **KHÔNG CÓ** (Bỏ hoàn toàn!)        |
| **Loss + Backward** | CÓ (tính loss, cập nhật weight)  | **KHÔNG CÓ** (chỉ Forward)          |
| **Output**          | Loss value + accuracy            | Embedding 512-D → so với DB         |
| **BatchNorm**       | mode train (tính running stats)  | mode eval (dùng running stats)      |
| **Gradient**        | BẬT (torch cần nhớ graph)        | TẮT (torch.no_grad → tiết kiệm RAM) |
| **File Model**      | .ckpt (~30MB, chứa optimizer)    | .onnx/.bin (~1.4MB, chỉ backbone)   |

### 8.2 Sơ đồ so sánh trực quan

```text
════════════ TRAINING ════════════        ══════════ INFERENCE ══════════

    Ảnh 112×112 (có sẵn)                  Frame camera (bất kỳ kích thước)
         │                                         │
         ▼                                    Face Detection (RetinaFace)
    ┌─────────┐                                    │
    │Backbone │                              Face Alignment (112×112)
    │MobileNet│                                    │
    └────┬────┘                                    ▼
         │                                    ┌─────────┐
    [512-D emb]                               │Backbone │
         │                                    │MobileNet│ ← CÙNG WEIGHTS
         ▼                                    └────┬────┘
    ┌──────────┐                                   │
    │CosFace   │ ← BỎ khi deploy!            [512-D emb]
    │Head      │                                   │
    │(85742-D) │                                   ▼
    └────┬─────┘                              ┌──────────────┐
         │                                    │Cosine Sim vs │
    CrossEntropy                              │Database (N人) │
    Loss                                      └──────┬───────┘
         │                                           │
    Backward()                                  threshold
    (Cập nhật weights)                               │
         │                                     Same / Diff
    Lặp 30 epoch                              (Cho phép / Từ chối)
```

### 8.3 Inference thực tế (từ `inference.py`)

```text
 1. Load model backbone (KHÔNG có CosFace Head)
    model = MobileNetV1(embedding_dim=512)
    model.load_state_dict(weights)
    model.eval()                            ← Chuyển BatchNorm sang eval mode

 2. Detect khuôn mặt
    FaceAnalysis → 5 landmark points

 3. Align khuôn mặt
    face_alignment(img, landmark, size=112) → ảnh 112×112

 4. Extract embedding
    with torch.no_grad():                   ← TẮT gradient, tiết kiệm RAM
        features = model(tensor)            → [1, 512]
        features = F.normalize(features)    → L2 normalize

 5. So sánh
    similarity = cosine(feat1, feat2)
    if similarity > threshold: SAME PERSON
```

---

## 9. Tổng Kết Flow Hoàn Chỉnh

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ① CHUẨN BỊ          ② TRAINING              ③ EXPORT           ④ DEPLOY    │
│  ───────────          ──────────              ────────           ────────     │
│                                                                               │
│  Dataset MS1MV2       30 epochs ×             .ckpt (30MB)      Camera SoC   │
│  5.8M ảnh aligned     5.8M ảnh/epoch          ↓ strip           ↓ Flash      │
│  85,742 danh tính     SGD + CosFace           .pth (1.5MB)      .bin (350KB) │
│  ↓ augment           Loss backward            ↓ onnx_export     ↓ NPU       │
│  DataLoader           Update weights          .onnx (1.4MB)     Inference    │
│  batch=512            Save best ckpt          ↓ NNE Quantize    ↓ Embedding  │
│                       Eval K-Fold             INT8 .bin         ↓ Match DB   │
│                                               (350KB)           ↓ Decision   │
│                                                                               │
│  KHOẢNG CÁCH VỀ DUNG LƯỢNG:                                                 │
│  Checkpoint 30MB ──────────────────────────────────────────→ Binary 350KB    │
│                            GIẢM ~86 LẦN                                      │
└───────────────────────────────────────────────────────────────────────────────┘
```
