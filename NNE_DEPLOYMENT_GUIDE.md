# Tài Liệu Triển Khai Face Recognition trên AnyCloud39AV100 NNE SDK

**Đơn vị:** Camera AI Team  
**Phiên bản:** v1.0  
**Ngày:** 2026-03-16  
**Trạng thái:** ✅ Phase 1 (Model → .bin) Hoàn thành — 🔄 Phase 2 (Deployment) Đang triển khai

---

## Mục Lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc pipeline tổng thể](#2-kiến-trúc-pipeline-tổng-thể)
3. [NNE Compile Toolchain — Đã hoàn thành](#3-nne-compile-toolchain--đã-hoàn-thành)
4. [Kiến trúc NNE Runtime trên Board](#4-kiến-trúc-nne-runtime-trên-board)
5. [NNE API Call Flow](#5-nne-api-call-flow)
6. [Face Recognition Inference Pipeline](#6-face-recognition-inference-pipeline)
7. [Feature Database Design](#7-feature-database-design)
8. [Tích hợp Camera Pipeline](#8-tích-hợp-camera-pipeline)
9. [Kế hoạch Testing & Validation](#9-kế-hoạch-testing--validation)
10. [Kế hoạch triển khai (Roadmap)](#10-kế-hoạch-triển-khai-roadmap)

---

## 1. Tổng quan hệ thống

### 1.1 Mục tiêu

Triển khai hệ thống **Face Recognition real-time** trên camera AI sử dụng chip **AnyCloud39AV100** của Anyka. Hệ thống phải:

- Nhận diện khuôn mặt từ luồng video camera (real-time)
- Chạy inference neural network trực tiếp trên **NPU** (Neural Processing Unit) của chip
- Đạt tốc độ xử lý phù hợp cho ứng dụng nhúng (mục tiêu ≥ 10 FPS)
- Tiêu thụ tài nguyên tối thiểu (RAM, CPU)

### 1.2 Thành phần hệ thống

```mermaid
graph LR
    subgraph Hardware["🔧 AnyCloud39AV100 Hardware"]
        CAM["📷 Image Sensor\n(Camera Module)"]
        ISP["ISP\n(Image Signal Processor)"]
        NPU["🧠 NPU\n(Neural Processing Unit)"]
        CPU["💻 CPU\nARM Cortex-A"]
        RAM["💾 RAM / DMA"]
        OUT["📺 Output\n(Display / Network)"]
    end

    CAM --> ISP
    ISP --> NPU
    NPU --> CPU
    CPU --> RAM
    RAM --> NPU
    CPU --> OUT
```

### 1.3 Trạng thái hiện tại

| Giai đoạn       | Công việc                                           | Trạng thái   |
| --------------- | --------------------------------------------------- | ------------ |
| **Training**    | Train MobileFaceNet/MobileNetV1 trên custom dataset | ✅ Hoàn thành |
| **ONNX Export** | Convert PyTorch → ONNX                              | ✅ Hoàn thành |
| **NNE Compile** | ONNX → Quantize → `.bin` (NNE binary)               | ✅ Hoàn thành |
| **Feature DB**  | Xây dựng embedding database cho board               | 🔲 Chưa làm   |
| **C Inference** | Viết C code gọi NNE API, preprocess, postprocess    | 🔲 Chưa làm   |
| **Face Detect** | Tích hợp face detection vào pipeline                | 🔲 Chưa làm   |
| **Integration** | Ghép toàn bộ pipeline với camera                    | 🔲 Chưa làm   |
| **Testing**     | Validate accuracy, benchmark FPS                    | 🔲 Chưa làm   |

---

## 2. Kiến trúc Pipeline Tổng Thể

Toàn bộ pipeline chia làm **2 luồng chính**: luồng offline (chạy trên PC) và luồng online (chạy trên board).

```mermaid
flowchart TB
    subgraph OFFLINE["☁️ OFFLINE — Chạy trên PC (Đã/Sẽ làm)"]
        direction TB
        A1["📊 Dataset\n(LFW, Custom, VGGFace2)"] --> A2["🏋️ Training\nMobileFaceNet / MobileNetV1\n(PyTorch)"]
        A2 --> A3["📦 PyTorch Weights\n(.pth)"]
        A3 --> A4["🔄 ONNX Export\nonnx_export.py"]
        A4 --> A5["📄 ONNX Model\n(.onnx)"]
        A5 --> A6["⚙️ NNE Toolchain\nConvert → Quantize → Compile"]
        A6 --> A7["🗂️ NNE Binary\n(.bin)"]

        A8["📸 Enrollment Images\nMỗi người: 3-10 ảnh"] --> A9["🔢 Feature Extraction\n(ONNX inference trên PC)"]
        A9 --> A10["💾 Face Database\n(.bin embeddings)"]
    end

    subgraph ONLINE["📷 ONLINE — Chạy trên Board Camera"]
        direction TB
        B1["🎥 Camera Stream\n(YUV / RGB frames)"] --> B2["🔍 Face Detection\n(Detect bbox khuôn mặt)"]
        B2 --> B3["✂️ Face Crop & Align\n112×112 px"]
        B3 --> B4["⚡ NNE Inference\n(.bin model trên NPU)"]
        B4 --> B5["📐 L2 Normalize\n(Embedding vector)"]
        B5 --> B6["🔎 Cosine Similarity\n(So sánh với Database)"]
        B6 --> B7{{"Similarity\n≥ Threshold?"}}
        B7 -->|"Yes"| B8["✅ Recognized\n(Tên + Score)"]
        B7 -->|"No"| B9["❌ Unknown"]
    end

    A7 -->|"deploy lên board"| B4
    A10 -->|"deploy lên board"| B6

    style OFFLINE fill:#e8f4fd,stroke:#2196F3
    style ONLINE fill:#e8f5e9,stroke:#4CAF50
```

---

## 3. NNE Compile Toolchain — Đã Hoàn Thành

### 3.1 Các bước đã làm

```mermaid
flowchart LR
    S1["📄 ONNX Model\nmodel.onnx"] -->|"nne_convert"| S2["📋 Intermediate\nGraph Format"]
    S2 -->|"nne_quantize\n(với calibration data)"| S3["🔢 Quantized Model\nINT8"]
    S3 -->|"nne_check\n(validate)"| S4{{"✅ Check\nPassed?"}}
    S4 -->|"Yes"| S5["nne_simulate\n(PC simulation)"]
    S4 -->|"No (fix config)"| S2
    S5 -->|"nne_compile"| S6["🗂️ NNE Binary\nmodel.bin"]

    style S6 fill:#c8e6c9,stroke:#4CAF50,stroke-width:2px
    style S1 fill:#e3f2fd,stroke:#2196F3
```

### 3.2 Thông tin quan trọng của file `.bin`

File `.bin` đã compile chứa **struct header NNE_MODEL_HEADER** gồm:

| Trường                          | Ý nghĩa                             | Lưu ý khi dùng                         |
| ------------------------------- | ----------------------------------- | -------------------------------------- |
| `input_w`, `input_h`, `input_c` | Kích thước input (thường 112×112×3) | Preprocessing phải đúng kích thước này |
| `input_color`                   | RGB / BGR / YUV                     | Feed đúng color space                  |
| `input_norm`                    | Normalize mode (mean/std)           | **BẮT BUỘC khớp với C code**           |
| `quan_method`                   | INT8 / FP16                         | Biết để debug accuracy                 |
| Output blob info                | Shape tensor đầu ra                 | Dùng để đọc embedding vector           |

> ⚠️ **Điều quan trọng nhất:** Các thông số `input_norm` (mean, scale) trong config quantize khi compile `.bin` **phải được hardcode y hệt** vào C preprocessing code. Nếu sai → embedding sai hoàn toàn.

---

## 4. Kiến trúc NNE Runtime trên Board

### 4.1 Các thư viện NNE SDK

```mermaid
graph TD
    subgraph APP["Application Layer (C code của bạn)"]
        USR["face_recognition_app.c"]
    end

    subgraph NNE_LIB["NNE SDK Libraries"]
        LIB1["libplat_npu.so\n━━━━━━━━━━━━━━\nak_npu_init()\nak_npu_deinit()\nak_npu_reset_hw()\nak_npu_get_version()"]
        LIB2["libak_nne.so\n━━━━━━━━━━━━━━\nNNE_Create_CH()\nNNE_Destroy_CH()\nNNE_Run()\nNNE_Run_Interrupt()\nNNE_Get_CallBackFun()"]
    end

    subgraph KERNEL["Kernel Driver"]
        DRV["ak_npu.ko\n(NPU kernel module)"]
    end

    subgraph HW["Hardware"]
        NPU["NPU Core\n(Neural Processing Unit)"]
    end

    USR --> LIB1
    USR --> LIB2
    LIB1 --> DRV
    LIB2 --> DRV
    DRV --> NPU

    style APP fill:#fff9c4,stroke:#f9a825
    style NNE_LIB fill:#e3f2fd,stroke:#1976D2
    style KERNEL fill:#f3e5f5,stroke:#7B1FA2
    style HW fill:#e8f5e9,stroke:#388E3C
```

### 4.2 Memory Layout khi Inference

```mermaid
graph LR
    subgraph MEM["RAM / DMA Memory"]
        M1["Model Weights\n(từ .bin file)"]
        M2["DMA Input Buffer\n(preprocessed image)"]
        M3["DMA Output Buffer\n(embedding vector)"]
        M4["Face Database\n(embeddings DB)"]
    end

    subgraph NPU["NPU Core"]
        N1["Inference Engine"]
    end

    M1 -->|"load lúc Create_CH"| N1
    M2 -->|"input blob → NPU"| N1
    N1 -->|"output blob"| M3
    M3 -->|"cosine sim"| M4

    style NPU fill:#e8f5e9,stroke:#2e7d32
    style MEM fill:#e3f2fd,stroke:#1565c0
```

---

## 5. NNE API Call Flow

### 5.1 Luồng gọi API đầy đủ (theo tài liệu NNE API参考 V1.0.4)

```mermaid
sequenceDiagram
    participant APP as Application (C)
    participant PLAT as libplat_npu
    participant NNE as libak_nne
    participant NPU as NPU Hardware
    participant MEM as DMA Memory

    Note over APP,NPU: ═══ INITIALIZATION ═══
    APP->>PLAT: ak_npu_init()
    PLAT->>NPU: Initialize NPU hardware
    PLAT-->>APP: return 0 (OK)

    APP->>NNE: NNE_Get_CallBackFun()
    NNE-->>APP: return *cb_fun (malloc/free/DMA callbacks)

    APP->>NNE: NNE_Create_CH("model.bin", cb_fun)
    NNE->>MEM: Allocate DMA memory
    NNE->>NPU: Load model weights
    NNE-->>APP: return net_handle

    Note over APP,NPU: ═══ INFERENCE LOOP (mỗi frame) ═══
    loop Mỗi frame từ camera
        APP->>MEM: Fill input_blob_vectors (preprocessed image)
        APP->>NNE: NNE_Run_Interrupt(net_handle, input_blobs, output_blobs)
        NNE->>NPU: Trigger inference (interrupt mode)
        NPU-->>NNE: Inference complete (interrupt)
        NNE-->>APP: return 0 + output_blobs filled
        APP->>APP: nne_post_process(output_blobs) → embedding
        APP->>APP: cosine_similarity(embedding, database) → result
    end

    Note over APP,NPU: ═══ CLEANUP ═══
    APP->>NNE: NNE_Destroy_CH(net_handle)
    NNE->>MEM: Free DMA memory
    NNE-->>APP: return 0

    APP->>PLAT: ak_npu_deinit()
    PLAT->>NPU: Release NPU hardware
    PLAT-->>APP: return 0
```

### 5.2 Quan hệ giữa NNE_Run và NNE_Run_Interrupt

```mermaid
flowchart LR
    NEED["Cần chạy inference"] --> Q1{{"SDK hỗ trợ\ninterrupt mode?"}}
    Q1 -->|"Yes (kiểm tra ak_npu.ko)"| INT["NNE_Run_Interrupt()\n⚡ Nhanh hơn\n(CPU không bận chờ)"]
    Q1 -->|"No"| POLL["NNE_Run()\n🔄 Polling mode\n(CPU chờ NPU xong)"]

    INT -->|"Ưu tiên dùng"| DONE["✅ Output blobs"]
    POLL --> DONE

    style INT fill:#c8e6c9,stroke:#388E3C
    style POLL fill:#fff9c4,stroke:#f9a825
```

---

## 6. Face Recognition Inference Pipeline

### 6.1 Full inference pipeline mỗi frame

```mermaid
flowchart TD
    F0["📷 Raw Camera Frame\nYUV420 / RGB888\n(e.g., 1920×1080)"]

    F0 --> F1

    subgraph DETECT["🔍 Stage 1: Face Detection"]
        F1["Face Detector\n(SDK built-in / MTCNN / Haar)"]
        F1 --> F1A{{"Phát hiện\nkhuôn mặt?"}}
        F1A -->|"Không có mặt"| SKIP["⏭️ Skip frame"]
        F1A -->|"Có N mặt"| F2["Bounding Boxes\n[x, y, w, h] × N"]
    end

    subgraph ALIGN["✂️ Stage 2: Face Alignment"]
        F2 --> F3["Crop ROI theo BBox"]
        F3 --> F4["Resize → 112×112 px"]
        F4 --> F5["BGR → RGB\n(nếu cần)"]
    end

    subgraph PREPROCESS["🔧 Stage 3: Preprocessing"]
        F5 --> F6["Pixel Normalize\n(pixel - mean) / std"]
        F6 --> F7["HWC → CHW layout\nFill NNE_BLOB_VECTOR_S"]
    end

    subgraph INFERENCE["⚡ Stage 4: NPU Inference"]
        F7 --> F8["NNE_Run_Interrupt()\ntrên NPU"]
        F8 --> F9["Output NNE_BLOB_VECTOR_S\n(raw embedding, 128-512 dim)"]
    end

    subgraph POSTPROCESS["📐 Stage 5: Postprocessing"]
        F9 --> F10["L2 Normalize\nembedding = emb / ||emb||"]
        F10 --> F11["Cosine Similarity\nmatch với Face Database"]
        F11 --> F12{{"Score\n≥ Threshold??"}}
        F12 -->|"Yes"| F13["✅ Identity: [Tên]\nScore: 0.87"]
        F12 -->|"No"| F14["❌ UNKNOWN"]
    end

    F13 --> F15["🖥️ Vẽ kết quả lên frame\n(Tên + BBox + Score)"]
    F14 --> F15

    style DETECT fill:#e3f2fd,stroke:#1976D2
    style ALIGN fill:#f3e5f5,stroke:#7B1FA2
    style PREPROCESS fill:#fff9c4,stroke:#f9a825
    style INFERENCE fill:#e8f5e9,stroke:#388E3C
    style POSTPROCESS fill:#fce4ec,stroke:#C62828
```

### 6.2 Chi tiết Preprocessing — Tại sao quan trọng

```mermaid
flowchart LR
    subgraph QUANTIZE_CONFIG["Config Quantize (lúc compile .bin)"]
        QC["input_mean = 127.5\ninput_scale = 1/128.0\ncolor_format = RGB"]
    end

    subgraph C_PREPROCESS["C Code Preprocessing (trên board)"]
        CP["img_float = (pixel - 127.5) / 128.0\nbgr_to_rgb(img)\nhwc_to_chw(img)"]
    end

    QUANTIZE_CONFIG -->|"PHẢI KHỚP HOÀN TOÀN"| C_PREPROCESS

    MISMATCH["❌ Nếu không khớp"] --> WRONG["Embedding sai\nAccuracy = 0%"]
    MATCH["✅ Nếu khớp đúng"] --> RIGHT["Embedding đúng\nAccuracy như training"]

    style QUANTIZE_CONFIG fill:#e3f2fd,stroke:#1565c0
    style C_PREPROCESS fill:#fff9c4,stroke:#f9a825
    style MISMATCH fill:#ffcdd2,stroke:#c62828
    style MATCH fill:#c8e6c9,stroke:#2e7d32
```

---

## 7. Feature Database Design

### 7.1 Cách xây dựng Face Database

```mermaid
flowchart TB
    subgraph ENROLL["📸 Enrollment Phase (trên PC)"]
        E1["Thu thập ảnh\nMỗi người: 5-10 ảnh\nĐiều kiện ánh sáng khác nhau"]
        E2["Chạy ONNX model\n(pc inference)"]
        E3["Trích xuất embedding\n(128 / 256 / 512 dim)"]
        E4["L2 Normalize\nmỗi embedding"]
        E5["Tính Mean Embedding\nhoặc lưu tất cả"]
        E6["Lưu face_db.bin\n+ face_db.txt"]
        E1 --> E2 --> E3 --> E4 --> E5 --> E6
    end

    subgraph BOARD["🔧 Board Runtime"]
        B1["Load face_db.bin\nvào RAM"]
        B2["Query embedding\ntừ NNE inference"]
        B3["Cosine Similarity\nvới tất cả entries"]
        B4["Best match\n(max score)"]
        B1 --> B2 --> B3 --> B4
    end

    E6 -->|"Deploy lên board"| B1

    style ENROLL fill:#e3f2fd,stroke:#1976D2
    style BOARD fill:#e8f5e9,stroke:#388E3C
```

### 7.2 Cấu trúc Face Database

```
face_database/
├── face_db.bin          ← Float32 array [N_persons × embed_dim]
├── face_db.txt          ← Danh sách tên, 1 dòng / người
└── face_db_meta.json    ← Metadata: embed_dim, threshold, model_version
```

**Ví dụ `face_db_meta.json`:**
```json
{
  "model_version": "mobilefacenet_v1",
  "embed_dim": 128,
  "n_persons": 50,
  "threshold": 0.50,
  "normalize": "l2",
  "created_date": "2026-03-16"
}
```

### 7.3 Chiến lược lưu embedding

| Chiến lược                            | Ưu điểm    | Nhược điểm               | Khuyến nghị                   |
| ------------------------------------- | ---------- | ------------------------ | ----------------------------- |
| **Mean embedding** (trung bình N ảnh) | Nhỏ, nhanh | Mất thông tin biến thiên | ✅ Bắt đầu với cái này         |
| **Lưu tất cả** (N ảnh × embed)        | Robust hơn | Tốn RAM, chậm hơn        | Khi cần accuracy cao          |
| **Clustering** (đại diện cụm)         | Cân bằng   | Phức tạp hơn             | Khi database lớn (>100 người) |

---

## 8. Tích hợp Camera Pipeline

### 8.1 Kiến trúc module tổng thể trên board

```mermaid
graph TB
    subgraph CAMERA_SDK["Camera SDK (Anyka)"]
        C1["Camera Driver\n(ISP, Sensor)"]
        C2["Video Buffer\n(Circle Buffer)"]
        C3["Face Detect API\n(SDK có sẵn?)"]
    end

    subgraph APP_LAYER["Application (Code của bạn)"]
        A1["main.c\n(Orchestrator)"]
        A2["face_detect.c\n(BBox extraction)"]
        A3["face_align.c\n(Crop + Resize)"]
        A4["face_recognition_nne.c\n(NNE inference)"]
        A5["face_db.c\n(Database query)"]
        A6["output.c\n(Annotation + Display)"]
    end

    C1 --> C2
    C2 --> A1
    C3 --> A2
    A1 --> A2 --> A3 --> A4 --> A5 --> A6

    style CAMERA_SDK fill:#e8eaf6,stroke:#3F51B5
    style APP_LAYER fill:#fff9c4,stroke:#F57F17
```

### 8.2 Quyết định Face Detection

Trước khi viết code, cần quyết định dùng phương án nào:

```mermaid
flowchart TD
    Q1{{"Anyka SDK\ncó cung cấp\nFace Detection lib?"}}

    Q1 -->|"✅ Có"| OPT1["Option A ⭐ BEST\nDùng SDK Face Detect API\n► Đã optimize cho chip\n► Không cần compile thêm"]

    Q1 -->|"❌ Không"| Q2{{"Độ chính xác\nCần cao không?"}}

    Q2 -->|"Không cao lắm"| OPT2["Option B\nOpenCV Haar Cascade\n(chạy trên CPU)\n► Nhanh implement\n► Accuracy thấp"]

    Q2 -->|"Cần chính xác"| OPT3["Option C\nCompile MTCNN / YOLOFace\nthành .bin riêng\n► 2 model NNE chạy nối tiếp\n► Phức tạp hơn nhiều"]

    OPT1 --> NEXT["➡️ Tiếp tục viết Recognition code"]
    OPT2 --> NEXT
    OPT3 --> NEXT

    style OPT1 fill:#c8e6c9,stroke:#2e7d32
    style OPT2 fill:#fff9c4,stroke:#f9a825
    style OPT3 fill:#ffcdd2,stroke:#c62828
```

### 8.3 Cấu trúc thư mục project trên board

```
face_recognition_app/
│
├── CMakeLists.txt               ← Build config (cross-compile)
│
├── src/
│   ├── main.c                   ← Entry point, camera loop
│   ├── face_detect.c            ← Face detection wrapper
│   ├── face_align.c             ← Crop, resize, color convert
│   ├── face_recognition_nne.c   ← NNE API inference
│   └── face_db.c                ← Load database, cosine search
│
├── include/
│   ├── face_detect.h
│   ├── face_align.h
│   ├── face_recognition_nne.h
│   └── face_db.h
│
├── models/
│   └── face_recognition.bin     ← Model .bin đã compile
│
├── database/
│   ├── face_db.bin              ← Embeddings float32
│   ├── face_db.txt              ← Danh sách tên
│   └── face_db_meta.json        ← Metadata
│
└── libs/                        ← NNE SDK (từ vendor)
    ├── include/
    │   ├── ak_npu.h
    │   └── ak_nne_common.h
    └── lib/
        ├── libplat_npu.so
        └── libak_nne.so
```

---

## 9. Kế hoạch Testing & Validation

### 9.1 Strategy kiểm tra từng tầng

```mermaid
flowchart TB
    T1["🧪 Test 1: Preprocessing Validation\nSo sánh normalize output:\nPython ONNX vs C code"] --> T2

    T2["🧪 Test 2: Embedding Consistency\nCùng 1 ảnh → board output ≈ ONNX output\nKỳ vọng: Cosine ≥ 0.95"] --> T3

    T3["🧪 Test 3: Same Person Recognition\n2 ảnh khác nhau cùng người\nKỳ vọng: Cosine ≥ 0.50 (threshold)"] --> T4

    T4["🧪 Test 4: Different Person Separation\n2 ảnh khác người\nKỳ vọng: Cosine < 0.40"] --> T5

    T5["🧪 Test 5: End-to-end Accuracy\nTest set N người × M ảnh/người\nMetrics: Accuracy, FAR, FRR, FPS"]

    style T1 fill:#e3f2fd,stroke:#1976D2
    style T2 fill:#f3e5f5,stroke:#7B1FA2
    style T3 fill:#e8f5e9,stroke:#388E3C
    style T4 fill:#fff9c4,stroke:#f9a825
    style T5 fill:#fce4ec,stroke:#C62828
```

### 9.2 Metrics cần đo

| Metric                      | Công thức                                 | Mục tiêu |
| --------------------------- | ----------------------------------------- | -------- |
| **Accuracy**                | Đúng / Tổng                               | ≥ 95%    |
| **FAR** (False Accept Rate) | Unknown bị nhận sai / Total Unknown       | ≤ 1%     |
| **FRR** (False Reject Rate) | Người quen bị từ chối / Total Known       | ≤ 5%     |
| **Inference Latency**       | Thời gian 1 lần NNE_Run                   | ≤ 50ms   |
| **End-to-end FPS**          | Frame / giây (detect + align + recognize) | ≥ 10 FPS |

### 9.3 Quy trình validate preprocessing

```mermaid
sequenceDiagram
    participant PC as PC (Python)
    participant BOARD as Board (C)

    PC->>PC: Load test_face.jpg
    PC->>PC: Run ONNX inference
    PC->>PC: Output embedding_onnx[128]

    BOARD->>BOARD: Load test_face.jpg
    BOARD->>BOARD: Run NNE inference  
    BOARD->>BOARD: Output embedding_nne[128]
    BOARD->>PC: Send embedding_nne (via log / file)

    PC->>PC: cos_sim = dot(onnx, nne) / (|onnx| × |nne|)
    PC->>PC: Check: cos_sim ≥ 0.95?

    alt cos_sim >= 0.95
        PC->>PC: ✅ Preprocessing khớp, tiếp tục
    else cos_sim < 0.95
        PC->>PC: ❌ Kiểm tra lại mean/scale/color_format
    end
```

---

## 10. Kế hoạch Triển Khai (Roadmap)

### 10.1 Các giai đoạn công việc

```mermaid
gantt
    title Face Recognition Deployment Roadmap
    dateFormat YYYY-MM-DD
    section Phase 1 (DONE)
    Train Model          :done,    p1a, 2026-02-01, 2026-02-15
    Export ONNX          :done,    p1b, 2026-02-15, 2026-02-20
    NNE Compile (.bin)   :done,    p1c, 2026-03-01, 2026-03-15

    section Phase 2 (NOW)
    Build Face Database  :active,  p2a, 2026-03-16, 4d
    Viết C Inference Code :active, p2b, 2026-03-16, 7d
    Validate Preprocessing :        p2c, after p2b, 3d

    section Phase 3
    Face Detection Setup  :        p3a, after p2c, 5d
    CMake Build & Deploy  :        p3b, after p2c, 3d
    Integration Testing   :        p3c, after p3a p3b, 5d

    section Phase 4
    Accuracy Benchmark    :        p4a, after p3c, 3d
    Performance Tuning    :        p4b, after p4a, 4d
    Final Demo            :        p4c, after p4b, 2d
```

### 10.2 Thứ tự ưu tiên công việc ngay bây giờ

```mermaid
flowchart LR
    NOW1["🔴 NGAY BÂY GIỜ\n━━━━━━━━━━\n1. Kiểm tra config quantize\n   (mean/scale là bao nhiêu?)\n2. Chuẩn bị ảnh enrollment\n3. Chạy extract_embeddings.py"]

    NOW1 --> NEXT1["🟡 TIẾP THEO\n━━━━━━━━━━\n4. Hỏi vendor: SDK\n   có face detect lib?\n5. Xem SDK examples\n   (thường có sample C code)\n6. Setup cross-compile\n   toolchain"]

    NEXT1 --> LATER1["🟢 SAU ĐÓ\n━━━━━━━━━━\n7. Viết C wrapper\n   gọi NNE API\n8. Build & flash lên board\n9. Validate preprocessing\n10. Integration test"]

    style NOW1 fill:#ffcdd2,stroke:#c62828
    style NEXT1 fill:#fff9c4,stroke:#f9a825
    style LATER1 fill:#c8e6c9,stroke:#2e7d32
```

### 10.3 Checklist chi tiết

#### ✅ Đã hoàn thành
- [x] Train model `MobileFaceNet` / `MobileNetV1` trên custom dataset
- [x] Export sang ONNX (`onnx_export.py`)
- [x] Verify ONNX accuracy (`evaluate_onnx.py`)
- [x] Compile ONNX → `.bin` bằng NNE Toolchain

#### 🔲 Phase 2 — Feature Database (PC)
- [ ] Chuẩn bị thư mục ảnh enrollment: `data/enrollment/<tên>/*.jpg`
- [ ] Viết `extract_embeddings.py` — inference ONNX → lưu embeddings
- [ ] Viết `db_to_binary.py` — lưu embeddings sang `face_db.bin`
- [ ] Verify cosine similarity giữa ảnh cùng người (kỳ vọng ≥ 0.5)
- [ ] Copy `face_db.bin` + `face_db.txt` lên board

#### 🔲 Phase 2 — C Inference Code (Board)
- [ ] **Xác nhận normalize config** từ file config quantize `.cfg`/`.json`
- [ ] Kiểm tra SDK có sẵn face detection library không
- [ ] Xem qua code example trong SDK của vendor
- [ ] Viết `face_recognition_nne.c` với đúng NNE API call flow
- [ ] Viết `face_db.c` — load `.bin` database và cosine search
- [ ] Setup `CMakeLists.txt` với cross-compile toolchain

#### 🔲 Phase 3 — Build & Integration
- [ ] Build thành công: `cmake` + `make` (không lỗi)
- [ ] Copy binary lên board và chạy test 1 ảnh tĩnh
- [ ] Validate: embedding board ≈ embedding ONNX (cosine ≥ 0.95)
- [ ] Tích hợp với camera loop

#### 🔲 Phase 4 — Testing & Benchmark
- [ ] Test accuracy: ≥ 95%
- [ ] Test FAR: ≤ 1%
- [ ] Benchmark inference latency: ≤ 50ms/frame
- [ ] Benchmark FPS end-to-end: ≥ 10 FPS
- [ ] Demo final

---

## Phụ lục: Câu hỏi cần hỏi Vendor (Anyka)

Trước khi triển khai, nên xác nhận những điểm sau với vendor:

| #   | Câu hỏi                                                                       | Tại sao quan trọng                 |
| --- | ----------------------------------------------------------------------------- | ---------------------------------- |
| 1   | SDK có cung cấp **Face Detection library** không?                             | Quyết định approach face detection |
| 2   | Có **C code example** nào dùng NNE API không?                                 | Tiết kiệm thời gian implement      |
| 3   | Input tensor phải dùng **DMA buffer** hay regular malloc?                     | Ảnh hưởng đến `PhyAddr` trong blob |
| 4   | Có hỗ trợ **interrupt mode** (`NNE_Run_Interrupt`) trên version SDK hiện tại? | `NNE_Run_Interrupt` nhanh hơn      |
| 5   | Model `.bin` có thể load từ **SD card / flash** không?                        | Xác định đường dẫn model           |
| 6   | Board có bao nhiêu **RAM / DMA memory** khả dụng?                             | Ảnh hưởng đến database size        |

---

*Tài liệu này được sinh ra dựa trên: `AnyCloud39AV100 NNE API参考_V1.0.4.pdf`, `AnyCloud39AV100 NNE使用示例_V1.0.5.pdf`, `AnyCloud39AV100 NNE网络移植编程手册_V1.0.5.pdf` và cấu trúc repo hiện tại.*
