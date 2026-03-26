# 🧠 Giải Thích Chi Tiết Kiến Trúc MobileFaceNet

> **MobileFaceNet** là một mạng neural nhẹ (lightweight) được thiết kế đặc biệt cho bài toán **nhận diện khuôn mặt** trên các thiết bị edge/embedded. Kiến trúc này dựa trên backbone **MobileNetV1** với các cải tiến quan trọng để tối ưu cho face recognition.

---

## 📌 Tổng Quan Kiến Trúc

```
INPUT (Ảnh khuôn mặt)
    │
    ▼
┌─────────────────────┐
│      STAGE 1        │  ← Trích xuất đặc trưng cơ bản (cạnh, texture, màu sắc)
│  (6 lớp Conv)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│      STAGE 2        │  ← Trích xuất đặc trưng trung cấp (mắt, mũi, miệng)
│  (6 lớp Conv)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│      STAGE 3        │  ← Trích xuất đặc trưng cao cấp (cấu trúc khuôn mặt)
│  (2 lớp Conv)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   OUTPUT (GDC)      │  ← Nén toàn bộ thành embedding vector 512-D
│  Conv2d + Linear    │
└─────────────────────┘
         │
         ▼
   Embedding Vector
      [1, 512]
```

Kiến trúc gồm **3 stage chính** + **1 lớp output GDC** (Global Depthwise Convolution).

---

## 🔧 Các Khái Niệm Quan Trọng Cần Biết Trước

### 1. Depthwise Separable Convolution (DWSepConv)

Đây là **kỹ thuật cốt lõi** của MobileNet, giúp giảm đáng kể lượng tính toán so với convolution thông thường.

**Convolution thông thường:**
```
Input [C_in, H, W] → Conv2d(C_in, C_out, k) → Output [C_out, H', W']
Số phép tính: C_in × C_out × k × k × H' × W'
```

**Depthwise Separable Convolution (2 bước):**
```
Bước 1 - Depthwise Conv:   Mỗi kênh được conv riêng biệt (groups = C_in)
    Input [C_in, H, W] → DWConv(C_in, C_in, k, groups=C_in) → [C_in, H', W']
    Số phép tính: C_in × k × k × H' × W'

Bước 2 - Pointwise Conv:   Conv 1×1 để trộn thông tin giữa các kênh
    [C_in, H', W'] → Conv2d(C_in, C_out, 1×1) → [C_out, H', W']
    Số phép tính: C_in × C_out × H' × W'
```

**So sánh:**
|                          | Conv thường                 | DWSepConv                           | Tỷ lệ giảm |
| ------------------------ | --------------------------- | ----------------------------------- | ---------- |
| Ví dụ: 32→64, k=3, 56×56 | 32×64×3×3×56×56 ≈ **57.8M** | 32×9×56×56 + 32×64×56×56 ≈ **7.3M** | **~8 lần** |

> [!TIP]
> DWSepConv giảm khoảng **8-9 lần** lượng tính toán so với convolution thông thường, đây là lý do chính MobileFaceNet có thể chạy nhanh trên thiết bị nhúng!

### 2. Width Multiplier (α)

Width multiplier là một **hệ số nhân** áp dụng lên số kênh (channels) ở mỗi lớp:

```
Số kênh thực tế = α × Số kênh gốc
```

| Width Multiplier (α) | Số kênh gốc | Số kênh thực tế | Tác dụng                |
| -------------------- | ----------- | --------------- | ----------------------- |
| 1.0                  | 64          | 64              | Giữ nguyên (full model) |
| 0.75                 | 64          | 48              | Giảm 25% channels       |
| 0.5                  | 64          | 32              | Giảm 50% channels       |
| 0.25                 | 64          | 16              | Giảm 75% channels       |

> [!IMPORTANT]
> Trong kiến trúc MobileFaceNet ở hình, các con số channels (8, 16, 32, 56, 112, 224) đã được tính sẵn với một width multiplier cụ thể. Khi nhìn vào dãy `8 → 16 → 32 → 56 → 112 → 224`, ta thấy quy luật nhân đôi dần dần.

### 3. Stride (s) và Downsampling

- **s=1**: Giữ nguyên kích thước spatial (H, W không đổi)
- **s=2**: Giảm kích thước spatial đi **một nửa** (H/2, W/2) → Ký hiệu `/2` trong hình

```
Ví dụ: Input [1, 32, 56, 56] qua DWSepConv(s=2) → Output [1, 56, 28, 28]
                     ↑↑                                          ↑↑
              56×56 pixels                                 28×28 pixels (giảm 1/2)
```

### 4. Ký hiệu Tensor Shape `[B, C, H, W]`

```
[1,   3,   112, 112]
 │    │     │    │
 │    │     │    └── Width (chiều rộng ảnh)
 │    │     └─────── Height (chiều cao ảnh)
 │    └───────────── Channels (số kênh: RGB=3, hoặc feature maps)
 └────────────────── Batch size (số ảnh xử lý cùng lúc)
```

---

## 📥 INPUT — Đầu Vào

```
┌──────────────────────────────────┐
│  INPUT: Ảnh khuôn mặt           │
│  Shape: [1, 3, 112, 112]        │
└──────────────────────────────────┘
```

| Thành phần | Giá trị | Ý nghĩa                 |
| ---------- | ------- | ----------------------- |
| Batch size | 1       | 1 ảnh tại một thời điểm |
| Channels   | 3       | 3 kênh màu RGB          |
| Height     | 112     | 112 pixel chiều cao     |
| Width      | 112     | 112 pixel chiều rộng    |

> [!NOTE]
> Ảnh đầu vào đã được **crop và align** (cắt và căn chỉnh) chỉ chứa khuôn mặt. Kích thước 112×112 là tiêu chuẩn cho các mô hình face recognition nhẹ, cân bằng giữa chất lượng ảnh và tốc độ xử lý.

---

## 🟦 STAGE 1 — Trích Xuất Đặc Trưng Cơ Bản

Stage 1 là giai đoạn đầu tiên, chịu trách nhiệm trích xuất các **đặc trưng cấp thấp** như cạnh (edges), góc (corners), texture, và gradient màu sắc.

### Lớp 1: Conv2dNormActivation

```
Conv2dNormActivation (3 → 8, k=3, s=1)
Input:  [1, 3, 112, 112]
Output: [1, 8, 112, 112]
```

Đây là một lớp convolution **thông thường** (không phải depthwise separable), bao gồm 3 thao tác:

```
Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    → BatchNorm2d(8)
        → PReLU(8)   (hoặc ReLU)
```

| Tham số         | Giá trị | Giải thích            |
| --------------- | ------- | --------------------- |
| Input channels  | 3       | 3 kênh RGB            |
| Output channels | 8       | Tạo ra 8 feature maps |
| Kernel size (k) | 3×3     | Bộ lọc 3×3 pixel      |
| Stride (s)      | 1       | Không giảm kích thước |
| Padding         | 1       | Giữ nguyên H, W       |

> **Tại sao dùng Conv thường ở lớp đầu?** Vì input chỉ có 3 kênh, Depthwise Conv trên 3 kênh không hiệu quả (quá ít kênh để tách). Conv thường ở đây cũng không tốn nhiều tính toán.

### Lớp 2: DWSepConv (8 → 16)

```
DWSepConv (8 → 16, s=1)
Input:  [1, 8, 112, 112]
Output: [1, 16, 112, 112]
```

```
Depthwise Conv2d(8, 8, k=3, s=1, groups=8)    ← Mỗi kênh conv riêng
    → BatchNorm2d(8)
    → PReLU(8)
Pointwise Conv2d(8, 16, k=1)                   ← Trộn kênh, tăng từ 8→16
    → BatchNorm2d(16)
    → PReLU(16)
```

- Tăng gấp đôi số feature maps: 8 → 16
- Giữ nguyên kích thước spatial: 112×112

### Lớp 3: DWSepConv (16 → 32) với stride=2 ⬇️

```
DWSepConv (16 → 32, s=2) /2
Input:  [1, 16, 112, 112]
Output: [1, 32, 56, 56]        ← Kích thước giảm 1/2!
```

**Đây là lớp downsampling đầu tiên:**
- Channels tăng: 16 → 32 (×2)
- Spatial giảm: 112×112 → 56×56 (÷2)

> [!NOTE]
> Quy luật phổ biến trong CNN: Khi giảm spatial size đi 2 lần, ta tăng channels lên 2 lần. Điều này giúp duy trì tổng lượng thông tin qua các lớp.

### Lớp 4: DWSepConv (32 → 32)

```
DWSepConv (32 → 32, s=1)
Input:  [1, 32, 56, 56]
Output: [1, 32, 56, 56]
```

- Giữ nguyên cả channels lẫn spatial size
- Mục đích: **Tinh chỉnh và làm giàu thêm** các feature maps hiện có

### Lớp 5: DWSepConv (32 → 56) với stride=2 ⬇️

```
DWSepConv (32 → 56, s=2) /2
Input:  [1, 32, 56, 56]
Output: [1, 56, 28, 28]        ← Kích thước giảm 1/2!
```

- Channels tăng: 32 → 56 (gần ×2)
- Spatial giảm: 56×56 → 28×28 (÷2)

### Lớp 6: DWSepConv (56 → 56)

```
DWSepConv (56 → 56, s=1)
Input:  [1, 56, 28, 28]
Output: [1, 56, 28, 28]
```

- Lớp tinh chỉnh cuối cùng của Stage 1

### 📊 Tổng kết Stage 1

```
[1, 3, 112, 112] ──→ [1, 56, 28, 28]
```

| Metric               | Đầu vào            | Đầu ra            | Thay đổi     |
| -------------------- | ------------------ | ----------------- | ------------ |
| Channels             | 3                  | 56                | ×18.7        |
| Spatial              | 112×112            | 28×28             | ÷4           |
| Tổng pixels/features | 3×112×112 = 37,632 | 56×28×28 = 43,904 | ~tương đương |

---

## 🟧 STAGE 2 — Trích Xuất Đặc Trưng Trung Cấp

Stage 2 xử lý các **đặc trưng cấp trung**, bắt đầu nhận diện các bộ phận khuôn mặt: mắt, mũi, miệng, lông mày, đường viền khuôn mặt.

### Lớp 1: DWSepConv (56 → 112) với stride=2 ⬇️

```
DWSepConv (56 → 112, s=2) /2
Input:  [1, 56, 28, 28]
Output: [1, 112, 14, 14]       ← Kích thước giảm 1/2!
```

- Channels tăng gấp đôi: 56 → 112
- Spatial giảm: 28×28 → 14×14

### Lớp 2-6: DWSepConv (112 → 112) × 5 lần 🔄

```
DWSepConv (112 → 112, s=1) × 5 lần
Input:  [1, 112, 14, 14]
Output: [1, 112, 14, 14]       ← Lặp lại 5 lần, không đổi shape
```

> [!IMPORTANT]
> **5 lớp lặp lại** là điểm đặc biệt của Stage 2! Đây là nơi mạng "suy nghĩ sâu" nhất — với 5 lớp cùng kích thước, mạng có khả năng:
> - Học các đặc trưng phức tạp hơn qua nhiều bước chuyển đổi phi tuyến
> - Tinh chỉnh dần dần các feature maps mà không mất thông tin spatial
> - Tương tự như việc "nhìn đi nhìn lại" nhiều lần để hiểu rõ hơn

```
Lần 1: [1, 112, 14, 14] → [1, 112, 14, 14]  ← Phát hiện vùng mắt
Lần 2: [1, 112, 14, 14] → [1, 112, 14, 14]  ← Phân tích chi tiết mắt
Lần 3: [1, 112, 14, 14] → [1, 112, 14, 14]  ← Phát hiện mối quan hệ mắt-mũi
Lần 4: [1, 112, 14, 14] → [1, 112, 14, 14]  ← Tổng hợp đặc trưng cục bộ
Lần 5: [1, 112, 14, 14] → [1, 112, 14, 14]  ← Hoàn thiện biểu diễn
```

### 📊 Tổng kết Stage 2

```
[1, 56, 28, 28] ──→ [1, 112, 14, 14]
```

| Metric   | Đầu vào | Đầu ra | Thay đổi                        |
| -------- | ------- | ------ | ------------------------------- |
| Channels | 56      | 112    | ×2                              |
| Spatial  | 28×28   | 14×14  | ÷2                              |
| Số lớp   |         | 6      | (1 downsampling + 5 tinh chỉnh) |

---

## 🟥 STAGE 3 — Trích Xuất Đặc Trưng Cao Cấp

Stage 3 trích xuất các **đặc trưng cấp cao**, nắm bắt cấu trúc tổng thể và mối quan hệ giữa các bộ phận khuôn mặt — ví dụ: khoảng cách giữa hai mắt, tỷ lệ khuôn mặt, hình dáng tổng thể.

### Lớp 1: DWSepConv (112 → 224) với stride=2 ⬇️

```
DWSepConv (112 → 224, s=2) /2
Input:  [1, 112, 14, 14]
Output: [1, 224, 7, 7]         ← Kích thước giảm 1/2!
```

- Channels tăng gấp đôi: 112 → 224
- Spatial giảm: 14×14 → 7×7

### Lớp 2: DWSepConv (224 → 224)

```
DWSepConv (224 → 224, s=1)
Input:  [1, 224, 7, 7]
Output: [1, 224, 7, 7]
```

- Tinh chỉnh 224 feature maps cuối cùng
- Mỗi feature map 7×7 chứa thông tin tổng hợp từ toàn bộ khuôn mặt

### 📊 Tổng kết Stage 3

```
[1, 112, 14, 14] ──→ [1, 224, 7, 7]
```

| Metric   | Đầu vào | Đầu ra | Thay đổi                        |
| -------- | ------- | ------ | ------------------------------- |
| Channels | 112     | 224    | ×2                              |
| Spatial  | 14×14   | 7×7    | ÷2                              |
| Số lớp   |         | 2      | (1 downsampling + 1 tinh chỉnh) |

---

## 📤 OUTPUT — Lớp Đầu Ra (GDC + Linear)

Phần output chịu trách nhiệm **nén** tất cả thông tin từ 224 feature maps 7×7 thành một **embedding vector 512 chiều** duy nhất.

### Bước 1: LinearBlock (GDC) — Global Depthwise Convolution

```
LinearBlock (GDC): Conv2d(k=7, s=1, g=224)
Input:  [1, 224, 7, 7]
Output: [1, 224, 1, 1]
```

| Tham số         | Giá trị | Giải thích                           |
| --------------- | ------- | ------------------------------------ |
| Kernel size (k) | **7×7** | Bằng đúng kích thước spatial input!  |
| Stride (s)      | 1       |                                      |
| Groups (g)      | **224** | Mỗi kênh conv riêng biệt (depthwise) |

> [!IMPORTANT]
> **GDC (Global Depthwise Convolution)** là đặc trưng riêng của MobileFaceNet!
> 
> Thay vì dùng **Global Average Pooling (GAP)** như MobileNet gốc (lấy trung bình mỗi feature map), GDC dùng **convolution 7×7 depthwise** — nghĩa là mỗi kênh có bộ lọc 7×7 **học được** (learnable) thay vì trung bình cố định.
> 
> **Tại sao GDC tốt hơn GAP cho face recognition?**
> - GAP: Lấy trung bình → mất thông tin về **vị trí** đặc trưng trên khuôn mặt
> - GDC: Học **trọng số** cho từng vị trí → giữ được thông tin spatial quan trọng
> - Khuôn mặt có cấu trúc cố định (mắt ở trên, miệng ở dưới), nên vị trí rất quan trọng!

**Minh họa sự khác biệt:**
```
Feature map 7×7 (ví dụ kênh "mắt"):

GAP (Average Pooling):              GDC (Learnable weights):
┌─┬─┬─┬─┬─┬─┬─┐                   ┌────┬────┬────┬────┬────┬────┬────┐
│1│1│1│1│1│1│1│  ← Trọng số        │0.01│0.02│0.05│0.05│0.05│0.02│0.01│
├─┼─┼─┼─┼─┼─┼─┤     đều nhau      ├────┼────┼────┼────┼────┼────┼────┤
│1│1│1│1│1│1│1│     (1/49)         │0.02│0.10│0.30│0.30│0.30│0.10│0.02│ ← Tập trung
├─┼─┼─┼─┼─┼─┼─┤                   ├────┼────┼────┼────┼────┼────┼────┤     vào vùng
│1│1│1│1│1│1│1│                    │0.05│0.30│0.90│0.95│0.90│0.30│0.05│     mắt!
├─┼─┼─┼─┼─┼─┼─┤                   ├────┼────┼────┼────┼────┼────┼────┤
│1│1│1│1│1│1│1│                    │0.05│0.30│0.95│1.00│0.95│0.30│0.05│
├─┼─┼─┼─┼─┼─┼─┤                   ├────┼────┼────┼────┼────┼────┼────┤
│1│1│1│1│1│1│1│                    │0.05│0.30│0.90│0.95│0.90│0.30│0.05│
├─┼─┼─┼─┼─┼─┼─┤                   ├────┼────┼────┼────┼────┼────┼────┤
│1│1│1│1│1│1│1│                    │0.02│0.10│0.30│0.30│0.30│0.10│0.02│
├─┼─┼─┼─┼─┼─┼─┤                   ├────┼────┼────┼────┼────┼────┼────┤
│1│1│1│1│1│1│1│                    │0.01│0.02│0.05│0.05│0.05│0.02│0.01│
└─┴─┴─┴─┴─┴─┴─┘                   └────┴────┴────┴────┴────┴────┴────┘
→ Output: trung bình                → Output: tổng có trọng số (LEARNED!)
```

### Bước 2: Flatten

```
Flatten
Input:  [1, 224, 1, 1]
Output: [1, 224]
```

Bỏ 2 chiều spatial (đã là 1×1), chuyển tensor 4D thành vector 1D với 224 phần tử.

### Bước 3: Linear + BatchNorm1d

```
Linear (224 → 512) + BN1d
Input:  [1, 224]
Output: [1, 512]
```

```python
nn.Linear(224, 512)      # Fully connected: 224 → 512
nn.BatchNorm1d(512)      # Chuẩn hóa embedding
```

| Thành phần       | Vai trò                                               |
| ---------------- | ----------------------------------------------------- |
| Linear(224→512)  | Phép chiếu tuyến tính, mở rộng từ 224 lên 512 chiều   |
| BatchNorm1d(512) | Chuẩn hóa embedding vector, giúp training ổn định hơn |

> **Tại sao 512 chiều?** 512 là số chiều tiêu chuẩn trong face recognition, đủ lớn để biểu diễn sự khác biệt giữa hàng triệu khuôn mặt, nhưng đủ nhỏ để lưu trữ và so sánh nhanh.

### 🎯 Kết Quả Cuối Cùng

```
┌─────────────────────────────────────┐
│  OUTPUT: Embedding Vector           │
│  Shape: [1, 512]                    │
│                                     │
│  [0.023, -0.15, 0.87, ..., 0.42]   │
│   ← 512 số thực đại diện cho       │
│     danh tính khuôn mặt →          │
└─────────────────────────────────────┘
```

Vector 512 chiều này là **"chữ ký số"** của khuôn mặt:
- **Cùng 1 người** → 2 embedding vectors sẽ **gần nhau** (cosine similarity cao)
- **Khác người** → 2 embedding vectors sẽ **xa nhau** (cosine similarity thấp)

---

## 📏 Tổng Quan Biến Đổi Kích Thước Qua Toàn Bộ Mạng

```
Layer                              Shape              Channels   Spatial   Stride
─────────────────────────────────────────────────────────────────────────────────
INPUT                              [1, 3, 112, 112]      3       112×112    -
                                       │
═══════════════════════ STAGE 1 ═══════════════════════
Conv2dNormActivation (3→8, k=3)    [1, 8, 112, 112]      8       112×112   s=1
DWSepConv (8→16)                   [1, 16, 112, 112]    16       112×112   s=1
DWSepConv (16→32)             /2   [1, 32, 56, 56]      32        56×56    s=2 ⬇️
DWSepConv (32→32)                  [1, 32, 56, 56]      32        56×56    s=1
DWSepConv (32→56)             /2   [1, 56, 28, 28]      56        28×28    s=2 ⬇️
DWSepConv (56→56)                  [1, 56, 28, 28]      56        28×28    s=1
                                       │
═══════════════════════ STAGE 2 ═══════════════════════
DWSepConv (56→112)            /2   [1, 112, 14, 14]    112        14×14    s=2 ⬇️
DWSepConv (112→112) ×5             [1, 112, 14, 14]    112        14×14    s=1
                                       │
═══════════════════════ STAGE 3 ═══════════════════════
DWSepConv (112→224)           /2   [1, 224, 7, 7]      224         7×7     s=2 ⬇️
DWSepConv (224→224)                [1, 224, 7, 7]      224         7×7     s=1
                                       │
═══════════════════════ OUTPUT ════════════════════════
GDC Conv2d (k=7, g=224)           [1, 224, 1, 1]       224         1×1     s=1
Flatten                           [1, 224]              224          -       -
Linear(224→512) + BN1d            [1, 512]              512          -       -
─────────────────────────────────────────────────────────────────────────────────
                                       │
                                       ▼
                              Embedding [1, 512]
```

**Quan sát quan trọng:**
- Spatial giảm dần: `112 → 56 → 28 → 14 → 7 → 1` (mỗi lần ÷2)
- Channels tăng dần: `3 → 8 → 16 → 32 → 56 → 112 → 224 → 512`
- Tổng cộng **4 lần downsampling** (stride=2) trước GDC

---

## 🔑 Tại Sao MobileFaceNet Hiệu Quả?

### 1. Depthwise Separable Convolution
Giảm **~8-9 lần** lượng tính toán so với convolution thông thường, cho phép chạy trên thiết bị embedded với tài nguyên hạn chế.

### 2. Width Multiplier
Cho phép **điều chỉnh linh hoạt** kích thước mô hình theo tài nguyên phần cứng:
- Thiết bị mạnh → α = 1.0 (full model)
- Thiết bị yếu → α = 0.5 (half model, nhanh hơn ~4 lần)

### 3. GDC thay vì Global Average Pooling
Giữ được thông tin **vị trí** của đặc trưng trên khuôn mặt — rất quan trọng vì khuôn mặt có cấu trúc cố định.

### 4. Kiến trúc nhỏ gọn nhưng đủ sâu
Với tổng khoảng **14 lớp** convolution (không kể BN và activation), MobileFaceNet cân bằng tốt giữa:
- **Accuracy**: Đủ sâu để học các đặc trưng phức tạp
- **Speed**: Đủ nhỏ để inference nhanh
- **Size**: Model size chỉ khoảng **1-4 MB** (tùy width multiplier)

### So sánh với các mô hình khác

| Model                 | Params | Size    | Accuracy (LFW) | Phù hợp với |
| --------------------- | ------ | ------- | -------------- | ----------- |
| ResNet-100 (ArcFace)  | ~65M   | ~250 MB | 99.83%         | Server/GPU  |
| MobileFaceNet (α=1.0) | ~1M    | ~4 MB   | 99.55%         | Mobile/Edge |
| MobileFaceNet (α=0.5) | ~0.3M  | ~1.5 MB | 99.2%          | MCU/Camera  |

> [!TIP]
> MobileFaceNet chỉ giảm ~0.3% accuracy so với ResNet-100 nhưng **nhỏ hơn ~60 lần** về kích thước và **nhanh hơn ~50 lần** về inference speed. Đây là lý do nó được chọn cho các ứng dụng nhận diện khuôn mặt trên thiết bị nhúng!

---

## 🧮 Ví Dụ Code PyTorch (Đơn Giản Hóa)

```python
import torch
import torch.nn as nn

class DWSepConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Bước 1: Depthwise - mỗi kênh conv riêng biệt
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(in_ch)
        )
        # Bước 2: Pointwise - trộn thông tin giữa các kênh
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        # STAGE 1
        self.stage1 = nn.Sequential(
            # Conv thường đầu tiên
            nn.Conv2d(3, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.PReLU(8),
            # Depthwise Separable Conv blocks
            DWSepConv(8, 16, stride=1),      # [8→16, giữ 112×112]
            DWSepConv(16, 32, stride=2),     # [16→32, 112→56]  /2
            DWSepConv(32, 32, stride=1),     # [32→32, giữ 56×56]
            DWSepConv(32, 56, stride=2),     # [32→56, 56→28]   /2
            DWSepConv(56, 56, stride=1),     # [56→56, giữ 28×28]
        )

        # STAGE 2
        self.stage2 = nn.Sequential(
            DWSepConv(56, 112, stride=2),    # [56→112, 28→14]  /2
            *[DWSepConv(112, 112, stride=1) for _ in range(5)],  # ×5 lần
        )

        # STAGE 3
        self.stage3 = nn.Sequential(
            DWSepConv(112, 224, stride=2),   # [112→224, 14→7]  /2
            DWSepConv(224, 224, stride=1),   # [224→224, giữ 7×7]
        )

        # OUTPUT (GDC)
        self.output_layer = nn.Sequential(
            # GDC: Global Depthwise Convolution
            nn.Conv2d(224, 224, 7, 1, 0, groups=224, bias=False),
            nn.BatchNorm2d(224),
            nn.Flatten(),
            # Linear projection
            nn.Linear(224, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        # x: [1, 3, 112, 112]
        x = self.stage1(x)   # → [1, 56, 28, 28]
        x = self.stage2(x)   # → [1, 112, 14, 14]
        x = self.stage3(x)   # → [1, 224, 7, 7]
        x = self.output_layer(x)  # → [1, 512]
        return x


# Demo
model = MobileFaceNet(embedding_dim=512)
dummy_input = torch.randn(1, 3, 112, 112)
embedding = model(dummy_input)
print(f"Input shape:  {dummy_input.shape}")   # [1, 3, 112, 112]
print(f"Output shape: {embedding.shape}")     # [1, 512]
```

---

## 📝 Tóm Tắt

| Đặc điểm            | Chi tiết                                      |
| ------------------- | --------------------------------------------- |
| **Backbone**        | MobileNetV1                                   |
| **Kỹ thuật chính**  | Depthwise Separable Convolution               |
| **Số stage**        | 3 stage + 1 output GDC                        |
| **Input**           | Ảnh khuôn mặt 112×112×3                       |
| **Output**          | Embedding vector 512-D                        |
| **Đặc trưng riêng** | GDC thay Global Average Pooling               |
| **Tuning**          | Width Multiplier (α) điều chỉnh kích thước    |
| **Ưu điểm**         | Nhẹ, nhanh, accuracy cao cho face recognition |
| **Ứng dụng**        | Thiết bị embedded, mobile, camera AI          |
