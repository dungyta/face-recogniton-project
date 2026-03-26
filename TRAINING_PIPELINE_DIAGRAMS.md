# 🖼️ Hình Ảnh Vẽ Pipeline: Training & Testing Face Recognition (Chi tiết Kiến trúc Mô hình)

Dưới đây là sơ đồ ASCII nâng cấp, kết hợp toàn cục (Macro) và vi mô (Micro) để biểu đồ hóa **CHÍNH XÁC cấu trúc Model MobileNetV1/V2 bên trong quá trình Training**. Bạn có thể dùng sơ đồ này copy paste thẳng vào Slide thuyết trình để thị phạm hình vẽ khối mạng nơ-ron thực tế.

---

## 1. Sơ Đồ Tổng Quan & Chi Tiết Lõi Mô Hình: Data ➔ Kiến trúc MobileNet ➔ Head ➔ Loss

Sơ đồ này cho thấy cách 1 bức ảnh chui qua các "khúc ruột" của MobileNet như thế nào (thể hiện rõ Width Multiplier và Depthwise Separable Conv) và quá trình so sánh với 85K người khác trong lúc train.

```text
========================================================================================================
                                     PIPELINE HUẤN LUYỆN (TRAINING PIPELINE)
========================================================================================================

 1. DATA INPUT (Pre-processed)       2. MẠNG CNN (BACKBONE - Dùng Trích Xuất Đặc Trưng)                 
 -----------------------------       --------------------------------------------------                    
                                  
   Ảnh Khớp Khuôn Mặt chuẩn                                       
      (112 x 112 x 3)            ╭─────────── [ MOBILE NET V1/V2 (width_mult = 0.25) ] ───────────╮   
             |                   │                                                                │   
      [ Augmentation ]           │  (A). Standard Conv (Lớp Mở Đầu)                               │   
   - Lật ngang (p=0.5)           │     (112x112x3) ➔ (56x56x8)      [Số kênh bị bóp đi 4 lần!]    │   
   - Chuẩn hóa màu sắc           │          |                                                     │   
             |                   │          v                                                     │   
             v                   │  (B). Inverted Residual / Depthwise Separable Blocks (Cốt Lõi) │   
     [ TENSOR BATCH ]            │   ----------------------------------------------------------   │   
  (256 ảnh một lúc vào GPU) ==== +> |  Bước 1: [Depthwise Conv 3x3] Lọc không gian từng kênh   |  │   
                                 │   |          (Ít tốn phép toán hơn 9 lần)                   |  │   
                                 │   |  Bước 2: [Pointwise Conv 1x1] Trộn vắt các kênh lại     |  │   
                                 │   ----------------------------------------------------------   │   
                                 │          |                                                     │   
                                 │          :  (Lặp lại 13-17 khối tương tự, kích thước ảnh       │   
                                 │          :   nhỏ dần (56 ➔ 28 ➔ 14 ➔ 7), kênh tăng lên 320)     │   
                                 │          v                                                     │   
                                 │  (C). Standard Conv (Lớp Chốt)                                 │   
                                 │     (7x7x160) ➔ (7x7x320)                                      │   
                                 │          |                                                     │   
                                 │          v                                                     │   
                                 │  (D). GDC (Global Depthwise Convolution) Layer                 │   
                                 │     [Conv 7x7x320] Thay vì lấy trung bình như GAP, GDC dùng    │   
                                 │     kernel 7x7 chà qua để giữ lại đặc trưng theo vị trí mũi/mắt│   
                                 │          |                                                     │   
                                 │          v                                                     │   
                                 │  (E). Linear Layer (Duỗi Thẳng)                                │   
                                 ╰────────────────────────────────────────────────────────────────╯   
                                                                   |
                                                                   v
 4. BỊ PHẠT VÀ SỬA TÍNH (BACKWARD)               3. ĐO MẶT VÀ CHẤM ĐIỂM (HEAD & LOSS)     
 --------------------------------                ------------------------------------           
                                                      
      [ Gradient Descent Cycle ]                      [ Vector Embedding (512-D) ]       
 Khởi động quá trình Backward, đi ngược                - Con số đại diện khuôn mặt:      
 lại toàn bộ quá trình trên. Sửa lại các ⟸===========    `[0.12, -0.4, 0.9, ...]`        
 "van nước" (Weights) bên trong các lớp                    |                             
 Conv lõi của MobileNet để lần sau                         v                             
 phán đoán đúng hơn! Giảm Loss xuống.                 [ CosFace Head ]                   
                                               So sánh Cosine (Góc) giữa Vector 512-D
                                               vừa sinh ra với ĐÁP ÁN của 85,742 người.  
                                                           |                             
                                                           v                             
                                              [ Margin Penalty (m = 0.4) ]        
                                               Kéo 2 mặt CÙNG MỘT NGƯỜI lại sát nhau.
                                               Đẩy 2 mặt KHÁC NGƯỜI ra xa nhau 0.4 góc.
                                                           |                             
                                                           v                             
                                                    [ HÀM TÍNH LOSS ]                       
                                                Model tính ra Điểm Phạt (Loss Value)
                                                do đoán sai hoặc 0 đủ chuẩn Margin.

========================================================================================================
```

---

## 2. Sơ Đồ Thu Nhỏ (Zoom-in): Bên trong "Trái Tim" MobileNet (Depthwise Separable Convolution)

Đây là hình vẽ "mổ xẻ" cái khối lõi của MobileNet (Khối B ở sơ đồ trên) giải thích làm sao để nó nhét vừa bức ảnh vào bộ nhớ camera 1.4MB. Sự kì diệu nằm ở việc toán học chia TÁCH khái niệm Lọc điểm ảnh (Không gian) và Lọc kênh màu (Chiều sâu).

```text
               STANDARD CONVOLUTION                             DEPTHWISE SEPARABLE CONVOLUTION
                (To kềnh càng)                                       (MobileNet - Bé tí teo)

               Khối Dữ Liệu Vào                                         Khối Dữ Liệu Vào
                 [ H x W x M ]                                            [ H x W x M ]
                       |                                                        |
         +-------------v-------------+                          +---------------v----------------+
         | Lấy 1 bộ lọc bự chà bá    |                          | BƯỚC 1: Depthwise Conv (3x3)   |
         | kích thước [3x3xM]        |                          | Lấy M bộ lọc [3x3x1] chà RIÊNG |
         | chà lên mọi kênh cùng lúc |                          | CHẼ vào một mình M kênh đó.    |
         | => Ra N khối mới          |                          | => Ra [H x W x M] ảnh mờmờ     |
         +-------------+-------------+                          +---------------+----------------+
                       |                                                        |
                       |                  <==== Tích Cực ====>                  v
                       v                       Thấy Không?      +---------------+----------------+
               Khối Dữ Liệu Ra                 Bên phải tách    | BƯỚC 2: Pointwise Conv (1x1)   |
                 [ H x W x N ]                 ra làm 2 bước    | Lấy bộ lọc [1x1xM] chà dọc     |
                                               nhưng số phép    | xuống để nhào nặn các kênh lại.|
               Tốn Kém Vô Địch:                nhân BỊ GIẢM     | => Ra N khối mới               |
            (H*W) * M * N * (3x3)              đi 8-9 lần!!     +---------------+----------------+
                (Quá Nặng)                                                      |
                                                                                v
                                                                          Khối Dữ Liệu Ra
                                                                            [ H x W x N ]

                                                                           Siêu Nhẹ Tiền:
                                                                (H*W)*M*(3x3) + (H*W)*M*N*(1x1)
```

---

## 3. Sơ Đồ Quá Trình Convert File (Deployment Architecture)

Làm sao nhét khối lõi PyTorch ở Sơ đồ 1 khít vào con chip yếu đuối của camera? Sơ đồ quá trình thay đổi Đuôi File để lượng tử hoá.

```text
 +---------------+       +---------------+        +---------------+         +-----------------+
 | File PyTorch  |       | File Chuẩn ĐH |        | File Phân Lớp |         |   File "Cục"    |
 | Dạng `.ckpt`  | =====>|  Dạng `.onnx` | =====> | Dạng `.nne IR`| ======> | Camera (`.bin`) |
 | (Khoảng 30MB) | Export| (Khoảng 1.4MB)| Convert|   Xác thực    | Quantize| (Khoảng ~350KB) |
 +---------------+       +---------------+        +---------------+         +-----------------+
         |                       |                        |                          |
   Cả lõi Model,         Cắt bỏ hết Loss, Optimizer, Dịch từ Toán chung       Biến các số thực Float32
   Loss, Optimizer,      chỉ giữ lại Trọng số/Toán   (ONNX) sang tập        gồm vài tỉ số lẻ 0.52352
   Lịch sử 30 vòng học.. Lưới MobileNet Cơ Bản.      lệnh của Camera Chip.  thành Số Nguyên INT8 (52).
```
