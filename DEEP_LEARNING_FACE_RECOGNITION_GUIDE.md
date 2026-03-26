# 📘 Hướng Dẫn Toàn Diện: Deep Learning cho Face Recognition

> **Đối tượng:** Lập trình viên biết Python & PyTorch cơ bản, muốn hiểu sâu kiến trúc model và cách tổ chức repo deep learning.
> **Ngôn ngữ:** Tiếng Việt, có ví dụ code thực tế từ repo `face-recognition`.

---

## Mục Lục

1. [Tổng quan hệ thống Face Recognition](#1-tổng-quan-hệ-thống-face-recognition)
2. [Neural Network Architecture](#2-neural-network-architecture)
3. [Convolutional Neural Network (CNN)](#3-convolutional-neural-network-cnn)
4. [Depthwise Separable Convolution](#4-depthwise-separable-convolution)
5. [Kiến trúc MobileNet](#5-kiến-trúc-mobilenet)
6. [Embedding trong Face Recognition](#6-embedding-trong-face-recognition)
7. [Cấu trúc Repository Deep Learning chuẩn](#7-cấu-trúc-repository-deep-learning-chuẩn)
8. [Cách Repo triển khai Model](#8-cách-repo-triển-khai-model)
9. [Data Pipeline trong Repo](#9-data-pipeline-trong-repo)
10. [Training Pipeline](#10-training-pipeline)
11. [Loss Function cho Face Recognition](#11-loss-function-cho-face-recognition)
12. [Inference Pipeline](#12-inference-pipeline)
13. [Các kỹ thuật giảm kích thước mô hình](#13-các-kỹ-thuật-giảm-kích-thước-mô-hình)
14. [Ví dụ Repo Face Recognition hoàn chỉnh](#14-ví-dụ-repo-face-recognition-hoàn-chỉnh)
15. [Luồng chạy toàn bộ hệ thống](#15-luồng-chạy-toàn-bộ-hệ-thống)
16. [Ví dụ thực tế với MobileNet](#16-ví-dụ-thực-tế-với-mobilenet)
17. [Tổng kết](#17-tổng-kết)

---

## 1. Tổng quan hệ thống Face Recognition

### 1.1 Pipeline hoàn chỉnh

Một hệ thống nhận diện khuôn mặt hoạt động theo chuỗi các bước sau:

```
┌─────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────┐
│  Image  │───▶│ Face Detection│───▶│ Face Alignment │───▶│ CNN Backbone  │
└─────────┘    └───────────────┘    └────────────────┘    └──────┬───────┘
                                                                 │
                                                                 ▼
┌──────────┐    ┌───────────────────┐    ┌───────────────────────────────┐
│ Decision │◀───│ Cosine Similarity │◀───│ Embedding Vector (512-d)      │
│ Same/Diff│    │    score = 0.85   │    │ [0.12, -0.34, 0.56, ...]     │
└──────────┘    └───────────────────┘    └───────────────────────────────┘
```

### 1.2 Giải thích từng bước

| Bước                  | Mô tả                                                                                                                                   | Ví dụ                                          |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Image**             | Ảnh đầu vào chứa khuôn mặt                                                                                                              | Ảnh chụp selfie, ảnh CCCD                      |
| **Face Detection**    | Phát hiện vị trí khuôn mặt trong ảnh, trả về bounding box + 5 điểm landmark (mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải) | InsightFace `FaceAnalysis`, RetinaFace, MTCNN  |
| **Face Alignment**    | Xoay & cắt khuôn mặt về tư thế chuẩn (mắt ngang hàng, mũi ở giữa) dựa trên 5 điểm landmark → ảnh 112×112                                | `face_alignment()` trong `utils/face_utils.py` |
| **CNN Backbone**      | Mạng neural trích xuất đặc trưng, biến ảnh 112×112×3 thành vector số                                                                    | MobileNetV1, MobileNetV2, SphereFace           |
| **Embedding Vector**  | Vector đặc trưng đại diện cho khuôn mặt, thường 512 chiều                                                                               | `[0.12, -0.34, 0.56, ...]`                     |
| **Cosine Similarity** | Đo độ tương đồng giữa 2 vector (-1 đến 1)                                                                                               | 0.85 = rất giống, 0.1 = khác nhau              |
| **Decision**          | So sánh similarity với threshold để quyết định                                                                                          | similarity > 0.35 → cùng người                 |

### 1.3 Code thực tế trong repo

```python
# Từ inference.py — Pipeline hoàn chỉnh
def extract_features(model, device, img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)

    # Bước 1: Face Detection — tìm khuôn mặt + 5 điểm landmark
    faces = face_app.get(img)
    landmark = faces[0].kps  # shape (5, 2)

    # Bước 2: Face Alignment — xoay & cắt về 112x112
    aligned = face_alignment(img, landmark, image_size=112)

    # Bước 3: Preprocessing — chuẩn hóa pixel
    aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(aligned)).unsqueeze(0).to(device)

    # Bước 4: CNN Backbone → Embedding Vector
    with torch.no_grad():
        features = model(tensor)
        features = F.normalize(features, dim=1)  # L2 normalize
        features = features.squeeze().cpu().numpy()

    return features  # Vector 512 chiều
```

### 1.4 Face Alignment chi tiết

Face Alignment sử dụng phép biến đổi **Similarity Transform** để ánh xạ 5 điểm landmark trên ảnh gốc về 5 điểm tham chiếu chuẩn:

```python
# Từ utils/face_utils.py
# 5 điểm tham chiếu chuẩn (ArcFace standard)
reference_alignment = np.array([
    [38.2946, 51.6963],   # Mắt trái
    [73.5318, 51.5014],   # Mắt phải
    [56.0252, 71.7366],   # Mũi
    [41.5493, 92.3655],   # Khóe miệng trái
    [70.7299, 92.2041]    # Khóe miệng phải
], dtype=np.float32)

def face_alignment(image, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)  # Ma trận biến đổi 2x3
    warped = cv2.warpAffine(image, M, (image_size, image_size))
    return warped  # Ảnh đã căn chỉnh 112x112
```

**Trực giác:** Giống như khi bạn scan CCCD — máy tự xoay & cắt cho thẳng hàng. Face alignment làm điều tương tự cho khuôn mặt.

---

## 2. Neural Network Architecture

### 2.1 Các khái niệm cơ bản

#### Layer (Tầng)
Một "bộ xử lý" trong mạng neural. Mỗi layer nhận input, xử lý, và trả output.

```python
# Các loại layer phổ biến trong face recognition
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)    # Tầng tích chập
bn = nn.BatchNorm2d(64)                               # Tầng chuẩn hóa
relu = nn.PReLU(64)                                    # Tầng kích hoạt
fc = nn.Linear(512, 512)                               # Tầng fully connected
```

#### Weight (Trọng số) & Parameter
- **Weight:** Các con số mà model "học" được trong quá trình training
- **Parameter:** Tổng hợp tất cả weights + biases của model

```python
# Đếm parameters
model = MobileNetV2(embedding_dim=512, width_mult=0.25)
total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng parameters: {total_params:,}")
# Ví dụ: 467,592 parameters
```

#### Feature Map
Output của một tầng Conv2d — là một "bản đồ đặc trưng" 3D (channels × height × width):

```
Input: [1, 3, 112, 112]     → 3 channels (RGB), 112x112 pixels
       ↓ Conv2d(3, 64)
Feature Map: [1, 64, 112, 112]  → 64 channels, mỗi channel chứa 1 đặc trưng
       ↓ Conv2d(64, 128, stride=2)
Feature Map: [1, 128, 56, 56]   → 128 channels, kích thước giảm một nửa
       ↓ ... (nhiều layers)
Feature Map: [1, 512, 7, 7]     → 512 channels, rất nhỏ nhưng giàu thông tin
       ↓ GDC (Global Depthwise Conv)
Embedding: [1, 512]             → Vector 512 chiều — "chữ ký" khuôn mặt
```

#### Embedding Vector
Vector cuối cùng đại diện cho một khuôn mặt. Hai khuôn mặt cùng một người sẽ có embedding gần nhau trong không gian 512 chiều.

### 2.2 Cách tính số Parameters và kích thước Model

```python
# Công thức tính parameters cho từng loại layer:

# Conv2d(in_ch, out_ch, kernel_size=k):
#   params = out_ch × in_ch × k × k + out_ch (bias)
#   Ví dụ: Conv2d(3, 64, 3) → 64 × 3 × 3 × 3 + 64 = 1,792

# BatchNorm2d(channels):
#   params = channels × 2 (gamma + beta)
#   Ví dụ: BatchNorm2d(64) → 64 × 2 = 128

# Linear(in_features, out_features):
#   params = in_features × out_features + out_features (bias)
#   Ví dụ: Linear(512, 512) → 512 × 512 + 512 = 262,656

# Kích thước model (MB):
#   size_mb = total_params × 4 / (1024 × 1024)  # float32 = 4 bytes
#   Ví dụ: 467,592 params → 467,592 × 4 / 1,048,576 ≈ 1.78 MB
```

---

## 3. Convolutional Neural Network (CNN)

### 3.1 Convolution — Trực giác

**Trực giác:** Hãy tưởng tượng bạn đang dùng một chiếc kính lúp nhỏ (kernel) để quét qua bức ảnh. Tại mỗi vị trí, kính lúp "tóm tắt" vùng ảnh đó thành 1 con số. Kết quả là một bản đồ cho biết "đặc trưng X có xuất hiện ở vị trí đó không".

```
Ảnh đầu vào (5×5)          Kernel (3×3)           Output (3×3)
┌───┬───┬───┬───┬───┐     ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │         ┌───┬───┬───┐
├───┼───┼───┼───┼───┤     ├───┼───┼───┤         │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 0 │ 1 │ 0 │  *  │ 0 │ 1 │ 0 │    =    ├───┼───┼───┤
├───┼───┼───┼───┼───┤     ├───┼───┼───┤         │ 2 │ 3 │ 2 │
│ 1 │ 0 │ 1 │ 0 │ 1 │     │ 1 │ 0 │ 1 │         ├───┼───┼───┤
├───┼───┼───┼───┼───┤     └───┴───┴───┘         │ 4 │ 3 │ 4 │
│ 0 │ 1 │ 0 │ 1 │ 0 │                            └───┴───┴───┘
├───┼───┼───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘
```

### 3.2 Các khái niệm quan trọng

| Khái niệm      | Mô tả                                   | Ảnh hưởng                                        |
| -------------- | --------------------------------------- | ------------------------------------------------ |
| **Kernel**     | Ma trận nhỏ (thường 3×3) trượt trên ảnh | Kernel khác nhau → phát hiện đặc trưng khác nhau |
| **Stride**     | Bước nhảy của kernel khi trượt          | stride=2 → output nhỏ đi 2×                      |
| **Padding**    | Thêm viền 0 xung quanh ảnh              | padding=1 → giữ nguyên kích thước                |
| **Activation** | Hàm phi tuyến sau conv (ReLU, PReLU)    | Giúp model học được pattern phức tạp             |
| **Pooling**    | Giảm kích thước feature map             | AvgPool, MaxPool, AdaptiveAvgPool                |

### 3.3 Công thức tính Output Shape

```
Output_Size = floor((Input_Size - Kernel_Size + 2 × Padding) / Stride) + 1
```

**Ví dụ:**
```
Input: 112×112, Kernel: 3×3, Stride=2, Padding=1
Output = floor((112 - 3 + 2×1) / 2) + 1 = floor(111/2) + 1 = 55 + 1 = 56
→ Output: 56×56
```

### 3.4 Code PyTorch

```python
import torch
import torch.nn as nn

# Ví dụ: Block Conv + BN + Activation (như trong repo)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Test
block = ConvBlock(3, 64, stride=2)
x = torch.randn(1, 3, 112, 112)
print(block(x).shape)  # torch.Size([1, 64, 56, 56])
```

Trong repo, class `Conv2dNormActivation` trong `utils/layers.py` thực hiện chính xác điều này:

```python
# Từ utils/layers.py
class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=None, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.PReLU, ...):
        if padding is None:
            padding = (kernel_size - 1) // 2  # Auto padding
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, groups=groups)]
        if norm_layer: layers.append(norm_layer(out_channels))
        if activation_layer: layers.append(activation_layer(out_channels))
        super().__init__(*layers)
```

---

## 4. Depthwise Separable Convolution

### 4.1 Vấn đề với Convolution thông thường

Convolution thông thường rất tốn tính toán. Ví dụ:

```
Normal Conv2d(64, 128, kernel_size=3):
  Parameters = 128 × 64 × 3 × 3 = 73,728
  FLOPs (với input 56×56) = 73,728 × 56 × 56 ≈ 231 triệu
```

→ Quá nặng cho thiết bị nhúng (điện thoại, Raspberry Pi, AI Box).

### 4.2 Giải pháp: Tách thành 2 bước

**Depthwise Separable Convolution** tách 1 conv lớn thành 2 conv nhỏ:

```
┌─────────────────────────────────────────────────────────┐
│                 NORMAL CONVOLUTION                       │
│  Input [64 ch] ──Conv2d(64,128,3×3)──▶ Output [128 ch] │
│  Params: 128 × 64 × 3 × 3 = 73,728                    │
└─────────────────────────────────────────────────────────┘

                        ↓ Tách thành ↓

┌─────────────────────────────────────────────────────────┐
│            DEPTHWISE SEPARABLE CONVOLUTION               │
│                                                          │
│  Bước 1: Depthwise Conv (lọc spatial cho từng channel)  │
│  Input [64 ch] ──Conv2d(64,64,3×3,groups=64)──▶ [64 ch]│
│  Params: 64 × 1 × 3 × 3 = 576                          │
│                                                          │
│  Bước 2: Pointwise Conv (trộn thông tin giữa channels) │
│  [64 ch] ──Conv2d(64,128,1×1)──▶ Output [128 ch]       │
│  Params: 128 × 64 × 1 × 1 = 8,192                      │
│                                                          │
│  Tổng params: 576 + 8,192 = 8,768                       │
│  Giảm: 73,728 / 8,768 ≈ 8.4× ít hơn! 🚀               │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Trực giác

- **Depthwise Conv:** Mỗi channel có 1 kernel riêng, chỉ xử lý channel đó. Giống như bạn lọc riêng kênh R, G, B.
- **Pointwise Conv:** Kernel 1×1, trộn thông tin giữa các channels. Giống như bạn pha trộn 3 kênh R, G, B lại với nhau.

### 4.4 Code trong repo

```python
# Từ utils/layers.py
class DepthWiseSeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride):
        layers = [
            # Depthwise: groups=in_channels → mỗi channel 1 kernel riêng
            Conv2dNormActivation(
                in_channels, in_channels,
                kernel_size=3, stride=stride,
                groups=in_channels  # ← Đây là key! groups = in_channels
            ),
            # Pointwise: kernel 1×1, trộn channels
            Conv2dNormActivation(
                in_channels, out_channels,
                kernel_size=1  # ← 1×1 convolution
            )
        ]
        super().__init__(*layers)
```

---

## 5. Kiến trúc MobileNet

### 5.1 MobileNetV1

MobileNetV1 xây dựng hoàn toàn từ Depthwise Separable Convolution:

```python
# Từ models/mobilenetv1.py — Cấu trúc thực tế trong repo
class MobileNetV1(nn.Module):
    def __init__(self, embedding_dim=512, width_mult=0.18):
        super().__init__()
        filters = [32, 64, 128, 256, 512, 1024]
        filters = [_make_divisible(f * width_mult) for f in filters]

        # Stage 1: Giảm dần từ 112×112 → 28×28
        self.stage1 = nn.Sequential(
            Conv2dNormActivation(3, filters[0], stride=1),           # 112→112
            DepthWiseSeparableConv2d(filters[0], filters[1], stride=1),  # 112→112
            DepthWiseSeparableConv2d(filters[1], filters[2], stride=2),  # 112→56
            DepthWiseSeparableConv2d(filters[2], filters[2], stride=1),  # 56→56
            DepthWiseSeparableConv2d(filters[2], filters[3], stride=2),  # 56→28
            DepthWiseSeparableConv2d(filters[3], filters[3], stride=1),  # 28→28
        )
        # Stage 2: 28×28 → 14×14
        self.stage2 = nn.Sequential(
            DepthWiseSeparableConv2d(filters[3], filters[4], stride=2),  # 28→14
            # ... 5 blocks thêm ở stride=1 ...
        )
        # Stage 3: 14×14 → 7×7
        self.stage3 = nn.Sequential(
            DepthWiseSeparableConv2d(filters[4], filters[5], stride=2),  # 14→7
            DepthWiseSeparableConv2d(filters[5], filters[5], stride=1),  # 7→7
        )
        # Output: 7×7 → embedding 512-d
        self.output_layer = GDC(filters[5], embedding_dim=embedding_dim)
```

### 5.2 MobileNetV2 — Inverted Residual Block

MobileNetV2 giới thiệu **Inverted Residual Block** với 3 bước:

```
Input (ít channels)
  │
  ├─── Expansion (1×1 Conv): Mở rộng channels lên t lần ──┐
  │                                                         │
  ├─── Depthwise (3×3 DW Conv): Lọc spatial ───────────────┤
  │                                                         │
  ├─── Projection (1×1 Conv): Giảm channels về ban đầu ────┤
  │                                                         │
  └─── + Residual Connection (nếu stride=1, in=out) ───────┘
```

**Tại sao gọi là "Inverted"?**
- ResNet truyền thống: Rộng → Hẹp → Rộng (bottleneck ở giữa)
- MobileNetV2: Hẹp → Rộng → Hẹp (inverted bottleneck — expansion ở giữa)

```python
# Từ models/mobilenetv2.py
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = (stride == 1 and in_planes == out_planes)

        layers = []
        if expand_ratio != 1:
            # Bước 1: Expansion — Pointwise 1×1 (mở rộng channels)
            layers.append(Conv2dNormActivation(in_planes, hidden_dim, kernel_size=1))

        layers.extend([
            # Bước 2: Depthwise — 3×3 conv riêng từng channel
            Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride,
                                 groups=hidden_dim),
            # Bước 3: Projection — Pointwise 1×1 (giảm channels, KHÔNG activation)
            nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)  # ← Residual connection
        return self.conv(x)
```

### 5.3 Cấu hình MobileNetV2 — Bảng t, c, n, s

```python
# Từ models/mobilenetv2.py
inverted_residual_setting = [
    # t (expand_ratio), c (output_channels), n (repeats), s (stride)
    [1,  16,  1, 1],   # Block 1: không expand, 16 ch
    [6,  24,  2, 2],   # Block 2: expand 6×, 24 ch, stride 2 → giảm kích thước
    [6,  32,  3, 2],   # Block 3: expand 6×, 32 ch
    [6,  64,  4, 2],   # Block 4: expand 6×, 64 ch
    [6,  96,  3, 1],   # Block 5: expand 6×, 96 ch, stride 1 → giữ nguyên
    [6, 160,  3, 2],   # Block 6: expand 6×, 160 ch
    [6, 320,  1, 1],   # Block 7: expand 6×, 320 ch
]
```

### 5.4 Width Multiplier — Tại sao `mobilenetv2_025` nhỏ hơn rất nhiều

**Width Multiplier** (α) nhân số channels ở mỗi layer với một hệ số < 1:

```python
# width_mult = 1.0 (full model)
filters = [32, 16, 24, 32, 64, 96, 160, 320, 512]

# width_mult = 0.25 (model thu nhỏ 4×)
filters = [8,  8,  8,  8,  16, 24, 40,  80,  512]
#          ↑ Mỗi giá trị được nhân với 0.25 rồi làm tròn về bội số 8
```

**Tại sao nhỏ đi cực nhiều?** Vì params tỉ lệ với bình phương channels:
- `Conv2d(64, 128, 3)` → 73,728 params
- `Conv2d(16, 32, 3)` → 4,608 params (giảm **16×** so với 4× channels!)

```python
# Ví dụ trong repo:
# width_mult=1.0:  ~3.4M params, ~14 MB
# width_mult=0.25: ~0.47M params, ~1.8 MB
model_full = MobileNetV2(embedding_dim=512, width_mult=1.0)
model_025  = MobileNetV2(embedding_dim=512, width_mult=0.25)
```

---

## 6. Embedding trong Face Recognition

### 6.1 Embedding Vector là gì?

Embedding là vector số thực N chiều (thường 512), đại diện cho một khuôn mặt trong **không gian đặc trưng** (feature space).

```
Khuôn mặt A → CNN → [0.12, -0.34, 0.56, 0.78, -0.23, ..., 0.45]  ← 512 số
Khuôn mặt B → CNN → [0.11, -0.33, 0.55, 0.77, -0.24, ..., 0.44]  ← Gần A
Khuôn mặt C → CNN → [-0.67, 0.89, -0.12, 0.34, 0.56, ..., -0.78] ← Xa A, B
```

**Trực giác:** Giống như "vân tay số" — mỗi người có một vector duy nhất, và vector của 2 ảnh cùng người sẽ nằm gần nhau trong không gian 512 chiều.

### 6.2 Cosine Similarity

```python
# Từ utils/face_utils.py
def compute_similarity(feat1, feat2):
    """Cosine similarity giữa 2 vectors."""
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return similarity
    # Kết quả: -1.0 (ngược hoàn toàn) đến 1.0 (giống hoàn toàn)
```

**Công thức toán học:**

```
                    A · B           Σ(aᵢ × bᵢ)
cos(θ) = ─────────────────── = ─────────────────────
              ‖A‖ × ‖B‖       √(Σaᵢ²) × √(Σbᵢ²)
```

### 6.3 Threshold — Tại sao có thể là 0.35 hoặc 0.7?

Threshold phụ thuộc vào:

| Yếu tố       | Threshold thấp (0.2-0.4)        | Threshold cao (0.5-0.7)        |
| ------------ | ------------------------------- | ------------------------------ |
| **Model**    | Model yếu, embedding chưa tốt   | Model mạnh, embedding rõ ràng  |
| **Ứng dụng** | Cần "nhận ra" nhiều (ít bỏ sót) | Cần "chắc chắn" (ít nhầm lẫn)  |
| **FAR**      | False Accept Rate cao           | False Accept Rate thấp         |
| **FRR**      | False Reject Rate thấp          | False Reject Rate cao          |
| **Ví dụ**    | Gợi ý tag bạn bè trên Facebook  | Mở khóa điện thoại, thanh toán |

```python
# Trong repo: threshold được tìm bằng K-Fold Cross-Validation
# Xem evaluate_onnx_benchmark.py, hàm k_fold_accuracy()
# Quét threshold từ -1.0 đến 1.0, bước 0.005
# Chọn threshold cho accuracy cao nhất trên tập train
thresholds = np.arange(-1.0, 1.0, 0.005)
```

---

## 7. Cấu trúc Repository Deep Learning chuẩn

### 7.1 Cấu trúc chung

```
project/
├── configs/          # File cấu hình (hyperparams, paths)
├── data/             # Dữ liệu training & validation
│   ├── train/        # Ảnh training, chia theo thư mục = class
│   └── val/          # Ảnh validation + annotation files
├── models/           # Định nghĩa kiến trúc model
│   ├── __init__.py   # Export tất cả models
│   ├── mobilenetv1.py
│   ├── mobilenetv2.py
│   └── sphereface.py
├── utils/            # Các hàm tiện ích
│   ├── dataset.py    # Custom Dataset class
│   ├── layers.py     # Custom layers (Conv blocks, etc.)
│   ├── metrics.py    # Loss functions (ArcFace, CosFace)
│   ├── face_utils.py # Face alignment, similarity
│   └── general.py    # Seed, logging, distributed training
├── weights/          # Model weights (.pth, .onnx, .ckpt)
├── train.py          # Script training chính
├── inference.py      # Script inference (so sánh khuôn mặt)
├── evaluate.py       # Script đánh giá trên benchmark
├── onnx_export.py    # Chuyển đổi PyTorch → ONNX
└── requirements.txt  # Dependencies
```

### 7.2 Chức năng từng folder

| Folder/File           | Chức năng                   | Ví dụ trong repo                                          |
| --------------------- | --------------------------- | --------------------------------------------------------- |
| `models/`             | Định nghĩa kiến trúc CNN    | `MobileNetV1`, `MobileNetV2`, `sphere20`                  |
| `utils/layers.py`     | Building blocks tái sử dụng | `Conv2dNormActivation`, `DepthWiseSeparableConv2d`, `GDC` |
| `utils/dataset.py`    | Load ảnh + label            | `ImageFolder` — đọc ảnh từ thư mục                        |
| `utils/metrics.py`    | Classification heads / Loss | `MarginCosineProduct` (CosFace), `AngleLinear`            |
| `utils/face_utils.py` | Xử lý khuôn mặt             | `face_alignment()`, `compute_similarity()`                |
| `train.py`            | Điều phối toàn bộ training  | Load data → Build model → Train → Save                    |
| `inference.py`        | So sánh 2 khuôn mặt         | Detect → Align → Extract → Compare                        |

---

## 8. Cách Repo triển khai Model

### 8.1 Sơ đồ phụ thuộc giữa các file

```
train.py / inference.py
    │
    ├── models/__init__.py ─── import MobileNetV1, MobileNetV2, sphere20, ...
    │       │
    │       ├── models/mobilenetv1.py ─── class MobileNetV1
    │       │       └── import từ utils/layers.py
    │       │
    │       ├── models/mobilenetv2.py ─── class MobileNetV2, InvertedResidual
    │       │       └── import từ utils/layers.py
    │       │
    │       └── models/sphereface.py ─── class sphere20, sphere36
    │
    ├── utils/layers.py ─── Conv2dNormActivation, DepthWiseSeparableConv2d, GDC
    ├── utils/dataset.py ─── ImageFolder
    ├── utils/metrics.py ─── MarginCosineProduct, AngleLinear
    └── utils/face_utils.py ─── face_alignment, compute_similarity
```

### 8.2 Cách model được build

```python
# Bước 1: utils/layers.py định nghĩa building blocks
class Conv2dNormActivation(nn.Sequential):  # Conv + BN + PReLU
class DepthWiseSeparableConv2d(nn.Sequential):  # DW + PW
class GDC(nn.Module):  # Global Depthwise Conv → Embedding

# Bước 2: models/mobilenetv2.py dùng building blocks để xây model
from utils.layers import Conv2dNormActivation, GDC
class InvertedResidual(nn.Module):  # Dùng Conv2dNormActivation
class MobileNetV2(nn.Module):       # Dùng InvertedResidual + GDC

# Bước 3: models/__init__.py export cho bên ngoài dùng
from models.mobilenetv2 import MobileNetV2

# Bước 4: train.py hoặc inference.py import và sử dụng
from models import MobileNetV2
model = MobileNetV2(embedding_dim=512, width_mult=0.25)
```

### 8.3 GDC — Từ Feature Map sang Embedding

`GDC` (Global Depthwise Convolution) là tầng cuối cùng, biến feature map 7×7 thành vector 512-d:

```python
# Từ utils/layers.py
class GDC(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(
            # Depthwise Conv 7×7: "nén" mỗi channel từ 7×7 → 1×1
            LinearBlock(in_channels, in_channels, kernel_size=7,
                        stride=1, padding=0, groups=in_channels),
            nn.Flatten()  # [B, C, 1, 1] → [B, C]
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, embedding_dim, bias=False),  # C → 512
            nn.BatchNorm1d(embedding_dim)  # Normalize embedding
        )

    def forward(self, x):
        # x: [B, 512, 7, 7] → features: [B, 512] → fc: [B, 512]
        x = self.features(x)
        x = self.fc(x)
        return x  # Embedding vector 512-d
```


---

## 9. Data Pipeline trong Repo

### 9.1 Cấu trúc dữ liệu Training

```
data/train/webface_112x112/
├── person_001/          # Mỗi thư mục = 1 người (class)
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── img_003.jpg
├── person_002/
│   ├── img_001.jpg
│   └── img_002.jpg
└── ...                  # 10,572 người (WebFace) hoặc 85,742 (MS1M)
```

### 9.2 Luồng chạy Data Pipeline

```
Thư mục ảnh (data/train/)
    │
    ▼
ImageFolder (utils/dataset.py)
    │  ← Quét thư mục, gán label tự động
    │     person_001/ → label 0
    │     person_002/ → label 1
    ▼
transforms.Compose([...])
    │  ← RandomHorizontalFlip, ToTensor, Normalize
    ▼
DataLoader (torch.utils.data)
    │  ← Chia thành batches, shuffle, multi-worker
    ▼
Model (MobileNetV2)
    │  ← Batch [B, 3, 112, 112] → Embeddings [B, 512]
    ▼
Classification Head (CosFace)
    │  ← Embeddings + Labels → Logits [B, num_classes]
    ▼
Loss (CrossEntropyLoss)
```

### 9.3 Code Dataset trong repo

```python
# Từ utils/dataset.py
class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = self._make_dataset(root)  # List of (path, label)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self._load_image(path)           # PIL Image
        if self.transform:
            image = self.transform(image)         # Tensor
        return image, label

    @staticmethod
    def _make_dataset(directory):
        """Quét thư mục, mỗi sub-folder = 1 class."""
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        instances = []
        for class_name, class_index in class_to_idx.items():
            class_dir = os.path.join(directory, class_name)
            for root, _, files in os.walk(class_dir):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        instances.append((os.path.join(root, f), class_index))
        return instances
```

### 9.4 Transforms trong Training

```python
# Từ train.py
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Augmentation: lật ngang ngẫu nhiên
    transforms.ToTensor(),               # PIL → Tensor [0, 1]
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),      # ImageNet mean
        std=(0.229, 0.224, 0.225)         # ImageNet std
    )
])
```

---

## 10. Training Pipeline

### 10.1 Luồng chạy khi `python train.py`

```
parse_arguments()          ← Đọc command line args
        │
main(params)
    │
    ├── setup_seed()       ← Đảm bảo reproducibility
    │
    ├── Model Selection ── MobileNetV2(embedding_dim=512, width_mult=0.25)
    │
    ├── Classification Head ── MarginCosineProduct(512, 10572)  # CosFace
    │
    ├── Dataset + DataLoader ── ImageFolder → DataLoader(batch=512)
    │
    ├── Loss + Optimizer
    │   ├── criterion = CrossEntropyLoss()
    │   ├── optimizer = SGD(lr=0.1, momentum=0.9)
    │   └── scheduler = MultiStepLR(milestones=[10, 20, 25])
    │
    └── Training Loop (30 epochs)
        │
        for epoch in range(30):
        │   ├── train_one_epoch()
        │   ├── lr_scheduler.step()
        │   ├── save checkpoint (_last.ckpt)
        │   ├── evaluate accuracy on LFW
        │   └── if best → save _best.ckpt
        │
        └── Training completed!
```

### 10.2 Training Loop chi tiết

```python
# Từ train.py — hàm train_one_epoch()
def train_one_epoch(model, classification_head, criterion, optimizer, data_loader, device, epoch, params):
    model.train()

    for batch_idx, (images, target) in enumerate(data_loader):
        images = images.to(device)         # [512, 3, 112, 112]
        target = target.to(device)         # [512] — labels

        optimizer.zero_grad()              # Reset gradients

        # Forward pass
        embeddings = model(images)         # [512, 512] — embedding vectors
        output = classification_head(embeddings, target)  # [512, 10572] — logits
        loss = criterion(output, target)   # Scalar — cross entropy loss

        # Backward pass
        loss.backward()                    # Tính gradients
        optimizer.step()                   # Cập nhật weights
```

### 10.3 Sơ đồ Forward Pass

```
images [512, 3, 112, 112]
        │
        ▼
   MobileNetV2.features        ← CNN Backbone
        │
        ▼
   feature maps [512, 512, 7, 7]
        │
        ▼
   GDC (output_layer)          ← Global Depthwise Conv → Linear → BN
        │
        ▼
   embeddings [512, 512]       ← Embedding vectors
        │
        ▼
   MarginCosineProduct          ← CosFace classification head
   (cosine similarity + margin)
        │
        ▼
   logits [512, 10572]         ← Score cho mỗi class
        │
        ▼
   CrossEntropyLoss             ← So sánh với labels thật
        │
        ▼
   loss (scalar)               ← Một con số, càng nhỏ càng tốt
```

---

## 11. Loss Function cho Face Recognition

### 11.1 Tại sao không dùng Softmax Loss thông thường?

Softmax Loss đơn giản chỉ phân loại "ảnh này thuộc người nào" nhưng **không ép** embeddings của cùng một người nằm gần nhau và embeddings của khác người nằm xa nhau.

**Giải pháp:** Thêm **margin** vào cosine similarity để ép model học phân biệt rõ ràng hơn.

### 11.2 CosFace (MarginCosineProduct) — Code thực tế

```python
# Từ utils/metrics.py
class MarginCosineProduct(nn.Module):
    """CosFace: Large Margin Cosine Loss"""
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.s = s    # Scale factor
        self.m = m    # Margin
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, label):
        # Bước 1: Tính cosine similarity giữa embedding và mỗi class center
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        # cosine: [B, num_classes] — similarity với từng class

        # Bước 2: Tạo one-hot label
        one_hot = F.one_hot(label.long(), num_classes=self.out_features).float()

        # Bước 3: Trừ margin m cho class đúng (làm khó hơn)
        output = self.s * (cosine - one_hot * self.m)
        # Ý nghĩa: class đúng bị "phạt" thêm margin m
        # → Model phải học embedding "chắc chắn" hơn
        return output
```

**Trực giác:** Giống như khi thi đấu bơi, người vô địch phải bơi thêm 0.4 giây so với đối thủ mới thắng. Điều này ép họ phải bơi cực nhanh → embeddings phải cực chính xác.

### 11.3 So sánh các Loss phổ biến

| Loss | Công thức | Margin áp dụng lên | Đặc điểm |
|------|-----------|---------------------|----------|
| **Softmax** | `s * cos(θ)` | Không có margin | Baseline, yếu nhất |
| **SphereFace** | `s * cos(m·θ)` | Góc θ (angular) | Nhân margin vào góc |
| **CosFace** | `s * (cos(θ) - m)` | Cosine trực tiếp | Trừ margin khỏi cosine |
| **ArcFace** | `s * cos(θ + m)` | Cộng margin vào góc | Mạnh nhất, phổ biến nhất |

### 11.4 ArcFace — Ví dụ code implement

```python
class ArcFace(nn.Module):
    """ArcFace: Additive Angular Margin Loss"""
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        one_hot = F.one_hot(labels.long(), num_classes=self.out_features).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return output
```

### 11.5 Triplet Loss (tham khảo)

```python
class TripletLoss(nn.Module):
    """Triplet Loss: anchor, positive, negative"""
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Khoảng cách anchor-positive phải nhỏ
        # Khoảng cách anchor-negative phải lớn
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()
```

---

## 12. Inference Pipeline

### 12.1 Luồng chạy `inference.py`

```
python inference.py
    │
    ├── load_model("sphere36", "weights/sphere36_mcp.pth")
    │       ├── get_network("sphere36")  ← Tạo model architecture
    │       └── model.load_state_dict()  ← Load trained weights
    │
    └── compare_faces(model, device, "img1.png", "img2.png", threshold=0.35)
            │
            ├── extract_features(model, device, "img1.png")
            │       ├── face_app.get(img)        ← Face Detection
            │       ├── face_alignment(img, kps)  ← Alignment 112×112
            │       ├── transform(img_pil)        ← Normalize
            │       ├── model(tensor)             ← CNN → Embedding
            │       └── F.normalize(features)     ← L2 normalize
            │
            ├── extract_features(model, device, "img2.png")
            │       └── ... (tương tự)
            │
            └── compute_similarity(feat1, feat2) → 0.85
                    └── similarity > threshold → "Same person!"
```

### 12.2 Code inference hoàn chỉnh

```python
# Từ inference.py
def compare_faces(model, device, img1_path, img2_path, threshold=0.35):
    feat1 = extract_features(model, device, img1_path)  # [512]
    feat2 = extract_features(model, device, img2_path)  # [512]
    similarity = compute_similarity(feat1, feat2)         # scalar
    is_same = similarity > threshold
    return similarity, is_same

# Sử dụng:
model = load_model("sphere36", "weights/sphere36_mcp.pth", device)
similarity, is_same = compare_faces(model, device, "img1.png", "img2.png")
print(f"Similarity: {similarity:.4f} - {'same' if is_same else 'different'}")
```

> **Lưu ý quan trọng:** Khi inference, `classification_head` (CosFace/ArcFace) **KHÔNG được dùng**. Nó chỉ dùng khi training. Khi inference, chỉ cần CNN backbone → embedding → cosine similarity.

---

## 13. Các kỹ thuật giảm kích thước mô hình

### 13.1 Architecture Design (MobileNet)

Đã giải thích ở Mục 5. Sử dụng Depthwise Separable Conv thay Normal Conv → giảm params 8-9×.

### 13.2 Width Multiplier

```python
# Giảm số channels ở mỗi layer
model_full  = MobileNetV2(width_mult=1.0)   # ~3.4M params
model_half  = MobileNetV2(width_mult=0.5)   # ~1.0M params
model_025   = MobileNetV2(width_mult=0.25)  # ~0.47M params
```

### 13.3 Pruning (Cắt tỉa)

Loại bỏ các weights có giá trị gần 0 (không quan trọng):

```python
import torch.nn.utils.prune as prune

# Cắt 30% weights có giá trị nhỏ nhất
model = MobileNetV2(embedding_dim=512)
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

### 13.4 Quantization (Lượng tử hóa)

Giảm precision từ float32 (4 bytes) → int8 (1 byte) → model nhỏ 4×, nhanh hơn:

```python
# Dynamic Quantization (đơn giản nhất)
import torch.quantization as quant

model_fp32 = MobileNetV2(embedding_dim=512)
model_int8 = quant.quantize_dynamic(
    model_fp32,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)
# Model size: ~1.8 MB → ~0.5 MB

# Static Quantization (chính xác hơn, cần calibration data)
model_fp32.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model_fp32)
# Chạy calibration data qua model
for data, _ in calibration_loader:
    model_prepared(data)
model_int8 = quant.convert(model_prepared)
```

### 13.5 Knowledge Distillation (Chưng cất tri thức)

Dùng model lớn (teacher) để dạy model nhỏ (student):

```python
# Teacher: MobileNetV2(width_mult=1.0)   — lớn, chính xác
# Student: MobileNetV2(width_mult=0.25)  — nhỏ, cần học

teacher = MobileNetV2(width_mult=1.0)
student = MobileNetV2(width_mult=0.25)

def distillation_loss(student_output, teacher_output, labels, alpha=0.5, T=4):
    # Soft loss: Student học phân phối xác suất của Teacher
    soft_loss = F.kl_div(
        F.log_softmax(student_output / T, dim=1),
        F.softmax(teacher_output / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    # Hard loss: Student vẫn học từ labels thật
    hard_loss = F.cross_entropy(student_output, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### 13.6 ONNX Export (Triển khai)

```python
# Từ onnx_export.py — Chuyển PyTorch → ONNX
dummy_input = torch.randn(1, 3, 112, 112)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'], output_names=['embedding'],
    dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}}
)
```

---

## 14. Ví dụ Repo Face Recognition hoàn chỉnh

### 14.1 Cấu trúc repo thực tế (từ repo face-recognition)

```
face-recognition/
├── models/                      # Kiến trúc model
│   ├── __init__.py              # Export: MobileNetV1, MobileNetV2, sphere20...
│   ├── mobilenetv1.py           # class MobileNetV1(nn.Module)
│   ├── mobilenetv2.py           # class MobileNetV2, InvertedResidual
│   ├── mobilenetv3.py           # class MobileNetV3
│   ├── sphereface.py            # class sphere20, sphere36, sphere64
│   └── onnx_model.py            # ONNX inference wrapper
│
├── utils/                       # Tiện ích
│   ├── layers.py                # Conv2dNormActivation, DepthWiseSeparableConv2d, GDC
│   ├── dataset.py               # ImageFolder (custom Dataset)
│   ├── metrics.py               # MarginCosineProduct (CosFace), AngleLinear
│   ├── face_utils.py            # face_alignment(), compute_similarity()
│   └── general.py               # setup_seed, AverageMeter, EarlyStopping, LOGGER
│
├── data/
│   ├── train/webface_112x112/   # Training data (10,572 identities)
│   └── val/                     # Validation data
│       ├── lfw/                 # LFW benchmark images
│       ├── lfw_ann.txt          # Annotation: 1/0 path1 path2
│       ├── calfw/
│       ├── cplfw/
│       └── agedb_30/
│
├── weights/                     # Saved models
│   ├── mobilenetv2_025_mcp.onnx
│   ├── sphere36_mcp.pth
│   └── *.ckpt                   # Training checkpoints
│
├── train.py                     # Training script
├── inference.py                 # Face comparison script
├── evaluate.py                  # PyTorch model evaluation
├── evaluate_onnx.py             # ONNX model evaluation
├── evaluate_onnx_benchmark.py   # Comprehensive benchmark
├── onnx_export.py               # PyTorch → ONNX conversion
└── requirements.txt
```

### 14.2 Sơ đồ kết nối giữa tất cả các file

```
                    ┌──────────────────────────┐
                    │       train.py            │
                    │  (Điều phối training)     │
                    └─────────┬────────────────┘
                              │ imports
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌──────────────────┐ ┌────────────────┐
│ models/         │ │ utils/dataset.py │ │ utils/metrics  │
│ mobilenetv2.py  │ │ ImageFolder      │ │ CosFace        │
│ mobilenetv1.py  │ └──────────────────┘ │ AngleLinear    │
│ sphereface.py   │                      └────────────────┘
└────────┬────────┘
         │ imports
         ▼
┌──────────────────┐
│ utils/layers.py  │
│ Conv2dNormAct    │
│ DepthWiseSep     │
│ GDC              │
└──────────────────┘

                    ┌──────────────────────────┐
                    │     inference.py          │
                    │  (So sánh khuôn mặt)     │
                    └─────────┬────────────────┘
                              │ imports
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌──────────────────┐ ┌────────────────┐
│ models/         │ │ utils/face_utils │ │  insightface   │
│ (same models)   │ │ face_alignment() │ │  FaceAnalysis  │
│                 │ │ compute_similar()│ │  (detection)   │
└─────────────────┘ └──────────────────┘ └────────────────┘
```

---

## 15. Luồng chạy toàn bộ hệ thống

### 15.1 Khi chạy `python train.py --network mobilenetv2_025 --classifier MCP`

```
 ① train.py::parse_arguments()
    └─ Đọc args: network="mobilenetv2_025", classifier="MCP", epochs=30, lr=0.1

 ② train.py::main(params)
    ├─ setup_seed()                                    ← utils/general.py
    │
    ├─ model = MobileNetV2(embedding_dim=512,          ← models/mobilenetv2.py
    │                       width_mult=0.25)
    │   └─ InvertedResidual blocks                     ← models/mobilenetv2.py
    │       └─ Conv2dNormActivation, GDC               ← utils/layers.py
    │
    ├─ classification_head = MarginCosineProduct(       ← utils/metrics.py
    │                         512, 10572)
    │
    ├─ train_dataset = ImageFolder("data/train/...")    ← utils/dataset.py
    │   └─ transform: RandomFlip + Normalize
    │
    ├─ train_loader = DataLoader(batch_size=512)
    │
    ├─ criterion = CrossEntropyLoss()
    ├─ optimizer = SGD([model.params, head.params])
    ├─ scheduler = MultiStepLR(milestones=[10,20,25])
    │
    └─ for epoch in range(30):
        │
        ├─ ③ train_one_epoch()                         ← train.py
        │   └─ for images, labels in train_loader:
        │       ├─ embeddings = model(images)           # [B,512]
        │       ├─ output = head(embeddings, labels)    # [B,10572]
        │       ├─ loss = criterion(output, labels)
        │       ├─ loss.backward()
        │       └─ optimizer.step()
        │
        ├─ lr_scheduler.step()
        │
        ├─ save_checkpoint("weights/mobilenetv2_025_MCP_last.ckpt")
        │
        ├─ ④ evaluate.eval(model, device)              ← evaluate.py
        │   └─ Đánh giá trên LFW → accuracy
        │
        └─ if accuracy > best → save _best.ckpt
```

### 15.2 Khi chạy `python inference.py`

```
 ① inference.py::__main__
    │
    ├─ ② load_model("sphere36", "weights/sphere36_mcp.pth")
    │       ├─ get_network("sphere36")                  ← Tạo model architecture
    │       ├─ torch.load("weights/sphere36_mcp.pth")   ← Load weights
    │       └─ model.load_state_dict(state_dict)        ← Gắn weights vào model
    │
    └─ ③ compare_faces(model, device, "img1.png", "img2.png", threshold=0.2255)
            │
            ├─ ④ extract_features(model, device, "img1.png")
            │       ├─ cv2.imread("img1.png")
            │       ├─ face_app.get(img)                ← InsightFace detection
            │       ├─ face_alignment(img, kps, 112)    ← utils/face_utils.py
            │       ├─ transform(img) → tensor          ← Resize + Normalize
            │       ├─ model(tensor)                    ← CNN → embedding [512]
            │       └─ F.normalize(features)            ← L2 normalize
            │
            ├─ ④ extract_features(model, device, "img2.png")
            │       └─ ... (tương tự)
            │
            ├─ ⑤ compute_similarity(feat1, feat2)       ← utils/face_utils.py
            │       └─ cosine similarity = 0.85
            │
            └─ 0.85 > 0.2255 → "Same person!"
```

---

## 16. Ví dụ thực tế với MobileNet

### 16.1 Phân tích `mobilenetv2_025`

| Metric | Giá trị |
|--------|---------|
| **Tên model** | MobileNetV2 với width_mult=0.25 |
| **Embedding dimension** | 512 |
| **Số parameters** | ~467K |
| **Model size (ONNX)** | ~1.8 MB |
| **Inference time (CPU)** | ~3-5 ms |
| **RAM usage** | ~10-15 MB |

### 16.2 Luồng dữ liệu qua model

```
Input: [1, 3, 112, 112]  ← Ảnh khuôn mặt đã align

  ▼ Conv2dNormActivation(3, 8, stride=1)     → [1, 8, 112, 112]
  ▼ InvertedResidual(8, 8, s=1, t=1)         → [1, 8, 112, 112]
  ▼ InvertedResidual(8, 8, s=2, t=6)         → [1, 8, 56, 56]
  ▼ InvertedResidual(8, 8, s=1, t=6)         → [1, 8, 56, 56]
  ▼ InvertedResidual(8, 8, s=2, t=6)         → [1, 8, 28, 28]
  ▼ ... (nhiều blocks khác)
  ▼ InvertedResidual(40, 80, s=2, t=6)       → [1, 80, 7, 7]
  ▼ Conv2dNormActivation(80, 512, k=1)       → [1, 512, 7, 7]
  ▼ GDC: LinearBlock(512, 512, k=7, groups=512) → [1, 512, 1, 1]
  ▼ Flatten()                                → [1, 512]
  ▼ Linear(512, 512) + BatchNorm1d           → [1, 512]

Output: [1, 512]  ← Embedding vector
```

### 16.3 Tại sao `width_mult=0.25` vẫn hoạt động tốt?

1. **Embedding dim vẫn giữ 512:** Layer cuối (`GDC → Linear → 512`) không bị ảnh hưởng bởi width_mult. Chỉ các layer trung gian bị thu nhỏ.
2. **Đủ capacity cho face recognition:** Nhận diện khuôn mặt không cần phân loại 1000 class như ImageNet. Chỉ cần trích xuất embedding tốt.
3. **Training data chất lượng:** WebFace/MS1M có hàng triệu ảnh, đủ để model nhỏ cũng học được.

---

## 17. Tổng kết

### 17.1 Kiến thức cốt lõi

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACE RECOGNITION SYSTEM                      │
│                                                                 │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌──────────────┐  │
│  │Detection│──▶│Alignment│──▶│   CNN   │──▶│  Embedding   │  │
│  │(Find)   │   │(Align)  │   │(Extract)│   │  (Compare)   │  │
│  └─────────┘   └─────────┘   └─────────┘   └──────────────┘  │
│                                                                 │
│  CNN Architecture:                                              │
│  • Conv2d → BatchNorm → Activation (building block cơ bản)    │
│  • Depthwise Separable Conv (giảm params 8×)                   │
│  • Inverted Residual Block (MobileNetV2)                       │
│  • Width Multiplier (thu nhỏ model)                            │
│  • GDC (feature map → embedding)                               │
│                                                                 │
│  Training:                                                      │
│  • Loss: CosFace/ArcFace (margin-based)                        │
│  • Optimizer: SGD + MultiStepLR                                │
│  • Data: ImageFolder + DataLoader                              │
│                                                                 │
│  Inference:                                                     │
│  • Detect → Align → Extract Embedding → Cosine Similarity     │
│  • Threshold quyết định Same/Different                         │
│                                                                 │
│  Model Compression:                                             │
│  • Architecture (MobileNet) + Width Multiplier                 │
│  • Pruning + Quantization + Knowledge Distillation             │
│  • ONNX Export cho production                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 17.2 Checklist khi xây dựng hệ thống Face Recognition

- [ ] **Chọn Backbone:** MobileNetV2 (nhẹ) hoặc ResNet50 (mạnh)
- [ ] **Chọn Loss:** ArcFace hoặc CosFace
- [ ] **Chuẩn bị Data:** Ảnh crop 112×112, chia thư mục theo person
- [ ] **Training:** Batch size lớn, SGD + MultiStepLR, 30+ epochs
- [ ] **Đánh giá:** LFW, CALFW, CPLFW, AgeDB_30
- [ ] **Tối ưu:** Width multiplier, Quantization, ONNX export
- [ ] **Deploy:** ONNX Runtime cho production, detect + align + embed + compare

### 17.3 Bảng tra cứu nhanh

| Khái niệm | File trong repo | Dòng code quan trọng |
|-----------|-----------------|---------------------|
| CNN Backbone | `models/mobilenetv2.py` | `class MobileNetV2` |
| Building blocks | `utils/layers.py` | `Conv2dNormActivation`, `GDC` |
| Depthwise Conv | `utils/layers.py` | `DepthWiseSeparableConv2d` |
| Loss Function | `utils/metrics.py` | `MarginCosineProduct` |
| Dataset | `utils/dataset.py` | `class ImageFolder` |
| Training | `train.py` | `train_one_epoch()`, `main()` |
| Inference | `inference.py` | `extract_features()`, `compare_faces()` |
| Face Alignment | `utils/face_utils.py` | `face_alignment()` |
| Similarity | `utils/face_utils.py` | `compute_similarity()` |
| Benchmark | `evaluate_onnx_benchmark.py` | `extract_onnx_features()` |
| ONNX Export | `onnx_export.py` | `torch.onnx.export()` |

---

> �� **Tài liệu này được tạo dựa trên code thực tế trong repo `face-recognition`.** Tất cả ví dụ code đều tham chiếu từ các file thật trong project.
