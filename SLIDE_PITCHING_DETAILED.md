# 📊 Kịch Bản Slide: Nhận Diện Khuôn Mặt Trên Camera (Chi Tiết Bán Kỹ Thuật)

> **Mục tiêu:** Dành cho buổi thuyết trình cần **độ sâu về kiến thức thuật toán và kiến trúc**, nhưng vẫn phải **trực quan, dễ hiểu** cho người mới bắt đầu tìm hiểu AI/Deep Learning.
> **Phương pháp:** Đưa ra lý thuyết, giải thích cơ chế hoạt động của lõi mô hình (MobileNet, CosFace, Depthwise Conv) bằng hình ảnh và con số cụ thể để tăng tính thuyết phục.

---

## 📑 Cấu trúc Bài Thuyết Trình (15 - 20 Phút)

| #   | Phần                          | Nội dung cốt lõi                                        |
| --- | ----------------------------- | ------------------------------------------------------- |
| 1   | **Bài toán & Thách thức**     | Đưa AI Face Recognition lên Edge Device (Camera nhỏ).   |
| 2   | **Hệ thống (Pipeline)**       | 4 bước: Detect ➔ Align ➔ Extract Feature ➔ Match.       |
| 3   | **Cách AI hiểu khuôn mặt**    | Khái niệm Vector Embedding 512 chiều. Mạng CNN.         |
| 4   | **Sức mạnh của MobileNet**    | Tại sao chọn MobileNet? Depthwise Separable Conv là gì? |
| 5   | **Bí quyết huấn luyện**       | Tại sao dùng CosFace Loss thay vì Softmax truyền thống? |
| 6   | **Tối ưu hóa dung lượng**     | Kỹ thuật Width Multiplier (α = 0.25). Trade-off.        |
| 7   | **Đưa xuống Camera (Deploy)** | ONNX và lượng tử hóa (Quantization FP32 ➔ INT8).        |

---

## 🗣️ Chi Tiết Từng Slide & Kịch Bản Thuyết Trình

### 🎯 Slide 1: Đặt vấn đề - Bài toán Edge AI
- **Tiêu đề:** Nhận diện khuôn mặt trên Edge Device: Khi giới hạn là phần cứng.
- **Visuals:** So sánh 2 server (Cloud) to lớn ngốn điện vs 1 chiếc Camera Indoor nhỏ bé. 
  - Server: RAM hàng chục GB, GPU hàng trăm triệu, Có internet.
  - Camera: Thiết bị nhúng (SoC), RAM < 256MB, Flash Storage cực ít (< 16MB), Yêu cầu chạy Offline & Real-time (<50ms).
- **Speaker Notes:** 
  > *"Chào các bạn. Nhận diện khuôn mặt không mới, nhưng chạy nó trên Cloud thì dễ, còn nhét nó vào một chiếc Camera an ninh bằng cái nắm tay lại là một thách thức kỹ thuật lớn. Camera bị giới hạn nghiêm trọng về RAM, bộ nhớ lưu trữ và sức mạnh tính toán. Bài toán của chúng ta là: Làm sao tạo ra một mô hình AI đủ nhỏ để nhét vừa Camera, nhưng vẫn đủ thông minh để không nhận diện sai người?"*

### ⚙️ Slide 2: Kiến trúc hệ thống (The Pipeline)
- **Tiêu đề:** Hệ thống hoạt động như thế nào? (End-to-End Pipeline)
- **Visuals:** Sơ đồ 4 khối có mũi tên: 
  `Face Detection (Tìm mặt)` ➔ `Face Alignment (Căn chỉnh)` ➔ `Feature Extraction (Trích xuất đặc trưng)` ➔ `Cosine Similarity (So sánh)`
- **Speaker Notes:** 
  > *"Để nhận diện ai đó, camera trải qua 4 bước: Đầu tiên, nó quét khung hình để móc ra vị trí khuôn mặt. Kế tiếp, nó xoay và cắt tỷ lệ cực chuẩn khuôn mặt đó (Alignment) về kích thước 112x112 pixel. Trái tim của hệ thống nằm ở bước 3: đưa ảnh này qua mạng nơ-ron sâu (CNN) để trích xuất đặc trưng. Cuối cùng, lấy đặc trưng đó đem so sánh với cơ sở dữ liệu để xem người này là ai."*

### 🧠 Slide 3: Cách AI biến ảnh thành "Mã Căn Cước" (Embedding)
- **Tiêu đề:** Biểu diễn khuôn mặt trong không gian toán học (Feature Vector)
- **Visuals:** Hình ảnh 1 khuôn mặt đi qua một cái phễu (CNN) ➔ chui ra một chuỗi số: `[0.12, -0.85, 0.44, ... ]` (Vector 512 chiều).
- **Speaker Notes:** 
  > *"Khác với các bài toán phân loại thông thường (Classification) yêu cầu đào tạo lại model mỗi khi có người mới. Chúng tôi dùng phương pháp Embedding. Mạng Nơ-ron (CNN) đóng vai trò như một cỗ máy mã hóa, biến bức ảnh lưới điểm ảnh thành 1 vector chứa đúng 512 con số thực. 512 con số này chính là 'mã ADN' đặc trưng của khuôn mặt đó. Khi camera thấy 1 người, nó tính ra 512 số này, gộp lại và so đối chiếu độ tương đồng (Cosine Similarity) với dữ liệu gốc."*

### 🏗️ Slide 4: Trái tim của hệ thống - Kiến trúc MobileNet
- **Tiêu đề:** Vượt qua giới hạn phần cứng với MobileNet
- **Visuals:** So sánh ResNet50 (25 Triệu tham số, quá nặng) vs MobileNetV2 (chỉ 2 Triệu tham số).
- **Kích thước:** Nhấn mạnh cấu trúc Inverted Residual Block.
- **Speaker Notes:** 
  > *"Để trích xuất 512 con số đó, chúng tôi cần một mạng Nơ-ron. Nếu dùng các mạng nổi tiếng như ResNet hay VGG, camera sẽ bị treo ngay lập tức vì quá nặng. Giải pháp kỹ thuật ở đây là dùng MobileNet. Đây là kiến trúc sinh ra dành riêng cho Mobile và thiết bị nhúng. Nó sử dụng một cơ chế gọi là Inverted Residual, giúp việc truyền luồng thông tin mượt mà hơn mà không cần nhiều bộ nhớ đệm."*

### 🔬 Slide 5: Phép thuật Toán Học - Depthwise Separable Convolution
- **Tiêu đề:** Bí mật giảm 8 lần khối lượng tính toán
- **Visuals:** Hình mô tả Standard Convolution (mạng lưới chằng chịt, nhân mọi thứ với nhau) VS Depthwise Separable Convolution (tách làm 2 bước: Depthwise và Pointwise).
  - Phép tính: `Conv(3x3) = 9x` ➔ chia làm 2 bước giảm computation.
- **Speaker Notes:** 
  > *"Tại sao MobileNet lại nhẹ? Trọng tâm lý thuyết nằm ở phép tính 'Tích chập tách rời theo chiều sâu' (Depthwise Separable Convolution). Thay vì dùng một bộ lọc 3D khổng lồ quét qua toàn bộ các kênh màu của ảnh (R,G,B) tốn rất nhiều phép nhân, kỹ thuật này chia làm 2 bước: Bước 1 lọc từng kênh màu riêng biệt, Bước 2 mới kết hợp chúng lại bằng bộ lọc siêu nhỏ 1x1. Toán học chứng minh kỹ thuật này giúp giảm khối lượng phép tính và số lượng tham số xuống từ 8 đến 9 lần so với truyền thống, trong khi vẫn giữ nguyên khả năng 'nhìn' của AI."*

### 🎯 Slide 6: Hàm Loss CosFace - Ép AI phân biệt người giống nhau
- **Tiêu đề:** Khắc phục nhược điểm của Softmax Loss với CosFace
- **Visuals:** Hai ranh giới (Decision Boundary). 
  - Ảnh 1: Softmax (các điểm dữ liệu xanh/đỏ nằm xát nhau ở biên).
  - Ảnh 2: CosFace (có một khoảng cách Margin rõ rệt ở giữa).
- **Speaker Notes:** 
  > *"Một rào cản học thuật lớn là nếu dùng hàm mất mát Softmax thông thường, AI sẽ dừng học ngay khi nó chớm phân biệt được 2 người. Trượt ra thực tế, với góc chụp tối hoặc đeo kính, AI liền nhận diện sai. Để khắc phục, chúng tôi huấn luyện model bằng hàm CosFace Loss. Nó bổ sung một khái niệm toán học gọi là 'Margin' (biên). Tức là, nó ép các bức ảnh của cùng 1 người phải co cụm chặt lại với nhau, và đẩy ranh giới giữa 2 người khác nhau ra xa hơn (Margin). Điều này tạo ra độ chính xác cực cao."*

### ✂️ Slide 7: Kỹ thuật Width Multiplier (Nhỏ gọn hơn nữa)
- **Tiêu đề:** Kỹ thuật Width Multiplier (α = 0.25)
- **Visuals:** Cấu trúc ống nước to (Full Width 1.0) thu thành ống nước nhỏ (Width 0.25). Bảng thông số kỹ thuật (Trade-off).
  - Baseline (MobileNetV1): 100% sức mạnh ➔ 8.6 MB
  - Optimizied (Alpha=0.25): 96% sức mạnh ➔ 1.4 MB
- **Speaker Notes:** 
  > *"Vẫn chưa đủ nhỏ cho chiếc Camera 16MB Storage. Chúng tôi dùng một hyperparameter tên là Width Multiplier (Ký hiệu là Alpha). Giống như việc bạn thu hẹp đường ống nước. Khi chúng tôi chỉnh Alpha = 0.25, số lượng kênh (chanel) ở mỗi lớp mạng giảm đi 4 lần. Kết quả rất đáng kinh ngạc: Số lượng tham số giảm đi 84% (từ hơn 8MB xuống chỉ còn 1.4MB), nhưng độ chính xác nhận diện chỉ sụt giảm hơn 3% (vẫn đạt xấp xỉ 99% trên tập test LFW). Một sự đánh đổi (trade-off) hoàn hảo!"*

### 🚀 Slide 8: Lượng tử hóa và đưa vào Camera (Deployment)
- **Tiêu đề:** Lượng tử hóa (Quantization) & Deploy với ONNX/NNE
- **Visuals:** Phép so sánh Float32 (Số thập phân cực dài: `0.123456789`) chuyển thành INT8 (Số nguyên ngắn: `12`). Khối `PyTorch` ➔ `ONNX` ➔ `.BIN (Camera)`.
- **Speaker Notes:** 
  > *"Cuối cùng, môi trường huấn luyện Python (PyTorch) không thể chạy trên Camera. Chúng tôi xuất mô hình ra chuẩn quốc tế ONNX, sau đó đưa qua trình biên dịch của camera (NNE Toolchain). Ở bước này, kỹ thuật Lượng Tử Hóa (Quantization) được áp dụng. AI giảm tính toán số thực dấu phẩy động (Float32) xuống số nguyên (INT8). Việc này giúp giảm thêm 4 lần giới hạn phần cứng nữa (model.bin cuối cùng chỉ còn vài trăm KB) và tăng tốc độ xử lý thêm 2-3 lần (đạt ngưỡng real-time). Đến đây, Camera của chúng ta đã có năng lực nhận diện siêu việt!"*

### 🙋 Slide 9: Review Kết quả & Q&A
- **Visuals:** Bắn 3 bullet points chính (Dung lượng siêu nhỏ - Suy luận siêu nhanh - Độ chính xác cao). Lời cảm ơn.
- **Lời kết:** Cảm ơn bạn giám khảo/khán giả đã lắng nghe.

---

## 💡 Mẹo thuyết trình để tạo sức hút (Pitching Tips)
1. **Dẫn dắt bằng con số ấn tượng:** Hãy liên tục nhắc đến việc dung lượng giảm **bao nhiêu lần**, tốc độ tăng **bao nhiêu mili-giây**. Dân kỹ thuật, nhà đầu tư rất thích con số thực tế thay vì nói "rất nhỏ", "rất nhanh".
2. **Body language với Slide số 5 (Depthwise Conv):** Khi giải thích việc bóc tách nhân chập, hãy dùng tay mô tả mặt phẳng 2D và nén lại thành 1D điểm, giúp người xem dễ hình dung toán học.
3. **Chuẩn bị Q&A:** Đối với dân nửa kỹ thuật/sếp công nghệ, họ sẽ hỏi xoáy vào Slide số 7: *"Tại sao giảm tham số nhiều thế mà accuracy chỉ rớt 3%?"*. Bạn cần chuẩn bị sẵn câu trả lời: *"Vì trong deep learning, các mạng nơ-ron cơ bản (overparameterized) chứa rất nhiều thông tin dư thừa. Việc giảm kích thước chỉ là loại bỏ phần dư thừa, giữ lại những bộ lọc cốt lõi nhất."*
