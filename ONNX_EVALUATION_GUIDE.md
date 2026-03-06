# Hướng dẫn đánh giá model ONNX — Benchmark toàn diện

## 📋 Tổng quan

Script `evaluate_onnx_benchmark.py` là công cụ đánh giá **toàn diện** các model ONNX cho bài toán **Face Recognition** (nhận diện khuôn mặt). Script được thiết kế phục vụ bài toán **tối ưu hóa** trên thiết bị nhúng (embedded deployment), nơi cần cân bằng giữa:

- **Accuracy** (độ chính xác)
- **Power consumption** (tiêu thụ năng lượng)
- **Latency** (độ trễ xử lý)
- **Model size** (dung lượng mô hình)

---

## 📊 Các metrics đánh giá

### Bảng tổng hợp metrics

| Metric                           | Ý nghĩa                                                      | Cách đo                                                  | Tại sao quan trọng?                                                     |
| -------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Accuracy (%)**           | Độ chính xác xác minh khuôn mặt (face verification)     | K-Fold 10 trên 4 benchmark datasets                       | Đảm bảo chất lượng nhận diện đạt yêu cầu                      |
| **Model Size (MB)**        | Dung lượng file `.onnx` trên disk                         | `os.path.getsize()`                                      | Ảnh hưởng trực tiếp đến storage trên thiết bị nhúng            |
| **RAM Usage (MB)**         | Bộ nhớ RAM cần thiết để load model và chạy inference   | `psutil.Process().memory_info().rss` tại 3 thời điểm | Thiết bị nhúng thường có RAM giới hạn (256MB-2GB)                 |
| **Inference Latency (ms)** | Thời gian xử lý 1 ảnh khuôn mặt                          | Trung bình 100 lần inference (sau 10 lần warmup)        | Liên quan trực tiếp đến FPS real-time và power consumption          |
| **Num Parameters**         | Tổng số tham số (trọng số) của model                     | Đếm từ ONNX graph initializers                          | Thể hiện độ phức tạp — ít params = ít bộ nhớ + ít tính toán |
| **FLOPs**                  | Floating-Point Operations — số phép tính dấu phẩy động | Tính từ các node Conv, Gemm, BatchNorm trong ONNX graph | Tương quan trực tiếp với power consumption trên chip AI             |
| **Efficiency Score**       | Chỉ số tổng hợp để xếp hạng model                      | `Accuracy / (Size × Latency)^0.3`                       | Giúp so sánh nhanh model nào tối ưu nhất                            |

### Chi tiết từng metric

#### 1. Accuracy — Độ chính xác

Đánh giá trên **4 benchmark datasets** chuẩn quốc tế:

| Dataset            | Tên đầy đủ            | Mô tả                                                                  | Độ khó        |
| ------------------ | -------------------------- | ------------------------------------------------------------------------ | ---------------- |
| **LFW**      | Labeled Faces in the Wild  | Benchmark cơ bản, ảnh khuôn mặt trong tự nhiên                    | ⭐ Dễ           |
| **CALFW**    | Cross-Age LFW              | Cùng người nhưng**khác tuổi** (ví dụ 20 tuổi vs 50 tuổi) | ⭐⭐ Trung bình |
| **CPLFW**    | Cross-Pose LFW             | Cùng người nhưng**khác góc chụp** (nghiêng, xoay)          | ⭐⭐⭐ Khó      |
| **AgeDB_30** | Age Database (gap 30 năm) | Cùng người với khoảng cách tuổi tác 30 năm                      | ⭐⭐⭐ Khó      |

Mỗi dataset có ~6000 cặp ảnh (pairs), gồm cả cặp cùng người (positive) và khác người (negative).

**Phương pháp**: K-Fold cross-validation 10 folds — chia 6000 pairs thành 10 phần, mỗi lần dùng 9 phần tìm threshold tối ưu, 1 phần test. Lặp 10 lần rồi lấy trung bình.

#### 2. RAM Usage — Bộ nhớ

Đo tại 4 thời điểm:

- **Before Load**: RAM trước khi load model (baseline)
- **After Load**: RAM sau khi load model vào bộ nhớ
- **After Inference**: RAM sau khi chạy inference
- **Model Only**: = After Load - Before Load (RAM chỉ cho model)

#### 3. Inference Latency — Độ trễ

Quy trình đo:

1. **Warmup** 10 lần (để CPU/GPU cache ổn định)
2. **Đo** 100 lần inference liên tiếp
3. Tính: **Avg** (trung bình), **Min**, **Max**, **Std** (độ lệch chuẩn)

#### 4. Efficiency Score — Điểm hiệu suất tổng hợp

```
 Score = Avg_Accuracy / (Size_MB × Latency_ms) ^ 0.3
```

- **Score càng cao** = model càng tối ưu (chính xác mà vẫn nhẹ + nhanh)
- Bảng Efficiency Ranking sẽ xếp hạng: 🥇 Tốt nhất, 🥈 Nhì, 🥉 Ba

---

## 🚀 Hướng dẫn sử dụng

### Yêu cầu môi trường

```bash
# Kích hoạt môi trường conda
conda activate face

# Các thư viện cần thiết (đã có trong môi trường face)
# psutil, onnx, onnxruntime, numpy, opencv-python, tqdm
```

### Các lệnh chạy

#### Đánh giá tất cả model ONNX trong `weights/`

```bash
python evaluate_onnx_benchmark.py
```

Tự động tìm tất cả file `*.onnx` trong thư mục `weights/` và đánh giá từng model.

#### Đánh giá 1 model cụ thể

```bash
python evaluate_onnx_benchmark.py --model weights/mobilenetv1_0.25_mcp.onnx
```

#### Đánh giá nhiều models cùng lúc

```bash
python evaluate_onnx_benchmark.py --model weights/mobilenetv1_0.25_mcp.onnx weights/mobilenetv2_mcp.onnx weights/mobilenetv1_018.onnx
```

#### Chỉ đo resource metrics — bỏ qua accuracy (chạy rất nhanh, ~10 giây)

```bash
python evaluate_onnx_benchmark.py --metrics-only
```

Hữu ích khi chỉ muốn so sánh kích thước, RAM, latency mà không cần chờ đánh giá accuracy (~5 phút/model).

#### Xuất kết quả ra CSV

```bash
python evaluate_onnx_benchmark.py --output results.csv
```

File CSV có thể import vào Excel/Google Sheets để vẽ biểu đồ so sánh.

#### Tuỳ chỉnh số lần đo latency

```bash
# 20 lần warmup + 200 lần đo (chính xác hơn, lâu hơn)
python evaluate_onnx_benchmark.py --warmup 20 --runs 200
```

### Tham số dòng lệnh

| Tham số           | Mặc định                            | Mô tả                                     |
| ------------------ | -------------------------------------- | ------------------------------------------- |
| `--model`        | Tất cả `*.onnx` trong `weights/` | Đường dẫn đến file(s) ONNX model      |
| `--root`         | `data/val`                           | Thư mục chứa dữ liệu validation        |
| `--metrics-only` | `False`                              | Chỉ đo resource metrics, bỏ qua accuracy |
| `--output`       | Không lưu                            | Đường dẫn file CSV xuất kết quả      |
| `--warmup`       | `10`                                 | Số lần warmup trước khi đo latency     |
| `--runs`         | `100`                                | Số lần chạy để đo latency trung bình |

---

## 📈 Cách đọc kết quả

### Bảng 1: ACCURACY COMPARISON

So sánh % accuracy trên 4 benchmarks.

- **LFW cao** = model nhận diện cơ bản tốt
- **CALFW, CPLFW, AgeDB_30 cao** = model xử lý tốt các trường hợp khó (khác tuổi, khác góc)
- Chênh lệch lớn giữa LFW và các bộ khác → model chưa robust

### Bảng 2: RESOURCE METRICS

Tổng quan nhanh về tài nguyên.

- **Params nhỏ** → model nhẹ, ít tính toán
- **Size nhỏ** → dễ deploy trên thiết bị hạn chế storage
- **RAM thấp** → phù hợp thiết bị RAM nhỏ
- **Latency thấp** → xử lý nhanh, tiết kiệm điện
- **FLOPs thấp** → ít phép tính, ít tốn năng lượng

### Bảng 3: LATENCY DETAILS

Chi tiết thời gian inference.

- **Std thấp** = model ổn định, phù hợp real-time
- **Max cao bất thường** = có thể do tải hệ thống, không phải lỗi model

### Bảng 4: RAM USAGE DETAILS

Chi tiết bộ nhớ tại từng giai đoạn.

- **Model Only** = RAM thực sự cần cho model (quan trọng nhất)

### Bảng 5: EFFICIENCY RANKING

Xếp hạng tổng hợp cân bằng cả 3 yếu tố: **accuracy, model size, latency**.

- 🥇 Model đứng đầu là model **tối ưu nhất** cho embedded deployment
- Score cao = trade-off tốt nhất giữa chính xác và nhẹ/nhanh

---

## ⚖️ Phân tích Trade-off cho thiết bị nhúng

```
                    ĐỘ CHÍNH XÁC CAO
                         ▲
                         │
                         │    Sphere20, Sphere36
                         │    (lớn, chính xác)
                         │
                         │         MobileNetV2
                         │         (cân bằng tốt)
                         │
    ÍT TÀI NGUYÊN ◄─────┼─────► NHIỀU TÀI NGUYÊN
                         │
                         │    MobileNetV1_0.25
                         │    (rất nhẹ, acc khá)
                         │
                         │    MobileNetV1_0.18
                         │    (siêu nhẹ)
                         │
                         ▼
                    ĐỘ CHÍNH XÁC THẤP
```

### Gợi ý lựa chọn model theo mục đích

| Mục đích                                     | Tiêu chí chọn                         | Model phù hợp                  |
| ----------------------------------------------- | ---------------------------------------- | -------------------------------- |
| **Ưu tiên accuracy**                    | Accuracy cao nhất, bỏ qua size/latency | Model lớn (MobileNetV2, Sphere) |
| **Ưu tiên tốc độ + tiết kiệm pin** | Latency + FLOPs thấp nhất              | MobileNetV1_0.18 hoặc V1_0.25   |
| **Cân bằng tốt nhất**                 | Efficiency Score cao nhất               | Xem bảng Ranking 🥇             |
| **Thiết bị RAM < 512MB**                | RAM Model Only < 15MB                    | MobileNetV1 variants             |
| **Storage hạn chế**                     | Size < 2MB                               | MobileNetV1_0.18 (0.87MB)        |

### Mối liên hệ giữa các metrics và power consumption

```
FLOPs ↑  →  Số phép tính tăng     →  CPU/GPU hoạt động nhiều hơn  →  Power ↑
Latency ↑  →  Thời gian xử lý tăng  →  Chip hoạt động lâu hơn    →  Power ↑  
RAM ↑  →  Bộ nhớ chiếm nhiều     →  Chip nhớ tiêu tốn năng lượng →  Power ↑
Size ↑  →  Storage tăng           →  Ảnh hưởng load time          →  Power ↑
```

**Kết luận**: Để tối ưu power consumption → chọn model có **FLOPs thấp + Latency thấp + RAM thấp**, đồng thời đảm bảo accuracy đạt ngưỡng chấp nhận được.

---

## 📁 Cấu trúc Output CSV

Khi dùng `--output results.csv`, file CSV sẽ chứa các cột sau:

| Cột                 | Kiểu  | Mô tả                             |
| -------------------- | ------ | ----------------------------------- |
| `Model`            | string | Tên model (không có `.onnx`)   |
| `Num_Params`       | int    | Tổng số parameters                |
| `Size_MB`          | float  | Dung lượng file (MB)              |
| `RAM_Model_MB`     | float  | RAM chỉ riêng model (MB)          |
| `RAM_Peak_MB`      | float  | RAM peak khi inference (MB)         |
| `Latency_Avg_ms`   | float  | Latency trung bình (ms)            |
| `Latency_Min_ms`   | float  | Latency nhỏ nhất (ms)             |
| `Latency_Max_ms`   | float  | Latency lớn nhất (ms)             |
| `Latency_Std_ms`   | float  | Độ lệch chuẩn latency (ms)      |
| `FLOPs`            | int    | Floating-point operations           |
| `LFW_%`            | float  | Accuracy trên LFW (%)              |
| `CALFW_%`          | float  | Accuracy trên CALFW (%)            |
| `CPLFW_%`          | float  | Accuracy trên CPLFW (%)            |
| `AgeDB_30_%`       | float  | Accuracy trên AgeDB_30 (%)         |
| `Avg_Acc_%`        | float  | Accuracy trung bình 4 datasets (%) |
| `Efficiency_Score` | float  | Điểm hiệu suất tổng hợp       |

---

## 📂 Cấu trúc dữ liệu validation

```
data/val/
├── lfw_112x112/           # Ảnh khuôn mặt LFW đã crop 112×112
├── calfw_112x112/         # Ảnh khuôn mặt CALFW đã crop 112×112
├── cplfw_112x112/         # Ảnh khuôn mặt CPLFW đã crop 112×112
├── agedb_30_112x112/      # Ảnh khuôn mặt AgeDB_30 đã crop 112×112
├── lfw_ann.txt            # Annotation: label path1 path2
├── calfw_ann.txt
├── cplfw_ann.txt
└── agedb_30_ann.txt
```

**Lưu ý**: Tất cả ảnh đã được crop và align sẵn về kích thước 112×112 pixels. Script **không cần** dùng face detection — chỉ cần preprocess (normalize) rồi inference trực tiếp.

---

## 🔧 Giải thích kỹ thuật: Cách trích xuất features

Quá trình trích xuất feature vector cho mỗi ảnh:

```
Ảnh gốc (112×112 BGR)
    │
    ├─→ Preprocess: resize + normalize (pixel / 127.5 - 1.0) + BGR→RGB
    │       │
    │       └─→ ONNX Inference → embedding_original (512-D)
    │
    ├─→ Flip ngang (horizontal flip)
    │       │
    │       └─→ Preprocess + ONNX Inference → embedding_flipped (512-D)
    │
    └─→ Concat: [embedding_original, embedding_flipped] → feature_vector (1024-D)
```

Sau đó, tính **cosine similarity** giữa 2 feature vectors để xác định cùng người hay khác người.
