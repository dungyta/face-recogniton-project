# 🧑‍🏫 Hướng Dẫn Chi Tiết: Quá Trình Training & Testing (Face Recognition)

Tài liệu này giải thích **"tận răng"** cách chúng tôi huấn luyện (Train) AI nhận diện khuôn mặt và cách kiểm tra (Test/Validate) xem nó học có đúng không. 

Dùng tài liệu này để thuyết trình hoặc trả lời các câu hỏi cực khó từ chuyên gia/giám khảo về quy trình xử lý dữ liệu.

---

## 1. Dữ liệu đầu vào: Nguyên liệu nấu ăn (Training Dataset) 🥗

### 1.1. Tập dữ liệu siêu khổng lồ: MS1MV2
Để AI phân biệt được ai là ai, nó cần nhìn hàng chục triệu cái mặt trước. Chúng tôi cho nó học tập dữ liệu tên là **MS1MV2** (Mircrosoft Celeb).
- **Quy mô:** Bao gồm khoảng **85,700 người khác nhau** (Identities).
- **Tổng số ảnh:** Gần **5.8 triệu bức ảnh**.
- **Đặc điểm:** Ảnh chụp mọi góc độ, lúc cười, lúc mếu, lúc đeo kính, lúc thiếu sáng.

### 1.2. Sơ chế dữ liệu (Pre-processing) trước khi học
AI không thể học trực tiếp một bức ảnh to tướng hay cái đầu nằm lệch một bên tĩnh mạch. Do đó, tất cả 5.8 triệu ảnh phải trải qua 2 bước sơ chế:

1. **Face Detection & Alignment (Dò và Cắt gọt chuẩn):**
   - Dùng một mạng nhỏ tìm ra 5 điểm chính: *2 mắt, cái mũi, 2 khóe miệng*.
   - Khóa 5 điểm này lại và **Xoay, kéo giãn** mọi khuôn mặt về chung 1 góc nghiêng chuẩn duy nhất (Hai mắt nằm ngang hàng).
2. **Crop & Resize (Cắt và thu nhỏ):**
   - Vứt bỏ tóc tai, khung cảnh thừa đằng sau.
   - Cắt gọt lấy đúng hình vuông quanh cái mặt, và nén nó về kích thước siêu nhỏ: **112x112 Pixel**.

> **🌟 Điểm nhấn thuyết trình:** "Nếu chúng ta nhồi nhét cả cảnh vật và tóc tai vào cho AI học, nó sẽ học sai (ví dụ học cái áo đỏ thay vì cái mặt). Việc 'cắt tỉa' gắt gao thành 112x112px giúp model cực kỳ tập trung vào các đường nét sống mũi, hốc mắt."

---

## 2. Quá trình "Lên Lớp" (Training Pipeline) 🏋️‍♂️

### 2.1. Đưa dữ liệu vào lò (Data Loading & Augmentation)
Dù có 5.8 triệu ảnh, AI vẫn có thể học vẹt. Để tránh học vẹt, trong quá trình nhét ảnh vào Model, chúng tôi tung 'hỏa mù' thêm bằng cách:
- **Random Flip (Lật ngang ngẫu nhiên):** 50% xác suất bức ảnh bị lật ngược trái/phải. AI hỏng mắt phải vẫn phải nhận ra bằng mắt trái. 
- **Normalize:** Chuyển đổi màu sắc về các con số chuẩn cho máy tính dễ tiêu hóa (chia cho 255).

### 2.2. Học như thế nào? (Forward & Backward)
AI học trong **30 Vòng (Epochs)**. Mỗi vòng nó sẽ xem lại toàn bộ 5.8 triệu bức ảnh. Do số lượng quá lớn, các ảnh được bó lại thành từng bó (Batch) - mỗi bó 512 ảnh.

**Một bước học diễn ra trong chớp mắt như sau:**
1. Trích xuất (Forward): Mạng MobileNet đẩy hình 112x112 qua một hệ thống ống nước phức tạp (Convolution) để vắt ra **512 giọt nước cốt (Vector Embedding 512-D)**.
2. Thi Cử (Cosine Similarity & Loss): Model đoán 512 giọt đó thuộc về người số mấy trong 85.000 người kia. Nó đoán sai! 
3. Hàm Phạt (CosFace Loss): Nó bị hệ thống phạt cực kỳ nặng (Bởi cái gọi là Margin). 
4. Sửa sai (Backward & Gradient Descent): Bị phạt, AI lập tức chạy ngược lại đường ống, vặn lại các van (Trọng số/Weights) để lần sau đoán chuẩn hơn. 

> **🌟 Điểm nhấn thuyết trình:** "Cứ như thế hàng tỷ lần, AI của chúng ta bị ăn đấm liên tục trong suốt 3-4 ngày chạy liên tục trên các cỗ máy GPU đồ sộ của NVIDIA. Cuối cùng, cái van lưới ống nước (MobileNet) đạt đến độ hoàn hảo."

---

## 3. Cách ép AI học nhóm (Thuật toán CosFace Loss) 🧠

Đoạn này trả lời câu hỏi: *Làm sao AI biết đẩy 2 người giống nhau ra xa?*

Nếu thiết kế AI thông thường (Dùng thuật toán Softmax), nó giống như học sinh thi trắc nghiệm: Chỉ cần vượt qua 5 điểm là đậu. Nghĩa là AI chỉ cần hơi phân biệt được ông A và ông B là nó ngừng học. Khi đó mang ra ngoài sáng/tối nó sẽ nhận nhầm ngay lập tức!

**Thuật toán CosFace Loss (MarginCosineProduct):**
Chúng tôi bắt AI mang thêm tạ! Chúng tôi cài thêm thông số **Margin (m = 0.4)**.
Nghĩa là, AI không chỉ phải nhận ra ông A, mà nó phải đẩy ông A cách ông B một khoảng an toàn (Biên). Càng giống nhau càng phải đẩy chúng ra xa trên không gian toán học nhiều chiều. Bức ảnh nào cũng phải chụm sát lại thành từng cụm đặc, không có điểm mờ. Nhờ vậy, AI nhận dạng cực kỳ chính xác.

---

## 4. Quá trình Thi Cử Chấm Điểm (Testing / Validation) 📝

Không thể cho AI làm lại đề nó đã học. Để đo độ thông minh thực sự, chúng tôi cho nó thi 4 kỳ thi chuẩn quốc tế (Benchmark Datasets).

### 4.1. 4 Bộ đề thi hóc búa:
Tất cả bộ đề này, những người bên trong AI **chưa từng nhìn thấy bao giờ**.
1. **LFW (Dễ):** Bài kiểm tra cơ bản về người nổi tiếng.
2. **CALFW (Khó - Lệch Tuổi):** Đưa 2 bức ảnh, một lúc thanh niên 20 tuổi và một lúc già 70 tuổi (Cross-Age). Bắt AI đoán xem có phải 1 người không.
3. **CPLFW (Khó - Lệch Góc):** Một ảnh chụp thẳng mặt, một ảnh chụp ngoảnh mặt ngang 90 độ, chỉ thấy mỗi lỗ tai và cái mũi. Bắt đoán.
4. **AgeDB_30 (Siêu khó):** Giống CALFW nhưng khoảng cách 30 năm vô cùng khắt khe.

### 4.2. Phương pháp chấm điểm (K-Fold Cross-Validation)
Để đảm bảo điểm số là trung thực nhất, chúng tôi không chấm 1 lần. 
Mỗi tập thi có 6.000 cặp mặt người. Chúng tôi:
- Cắt 6000 cặp này thành **10 phần bằng nhau**.
- Lần lượt lấy 9 phần để AI tự mò mẫm tìm ra ranh giới ngưỡng (Threshold độ giống).
- Lấy ngưỡng đó áp dụng để chấm điểm vào 1 phần còn lại (Tập Test mù).
- Lặp lại đủ 10 lần gieo xúc xắc như vậy để loại bỏ yếu tố may mắn, và tính **Điểm Trung Bình Cuối Cùng**.

### 4.3. Kết quả đáng tự hào:
Mặc dù bị chúng tôi bắt "ép cân" đi 84% bộ não (bằng Width Multiplier 0.25), nhưng model vẫn vượt qua 4 kỳ thi trên với con số kinh ngạc:
- Bài kiểm tra LFW: Điểm **98.75%**.
- Bài kiểm tra góc nghiêng/tuổi tác khó nhất: Đều đạt trên 80-90%. 
- Trung bình 4 kỳ thi: Đạt ~90.79%.
- Dung lượng cầm về: Một khối mã vẻn vẹn **1.4 MB**.

> **🌟 Kết luận bài thi:** "Với dung lượng 1.4MB (bằng 1 chiếc đĩa mềm năm 1990), AI này thừa sức cài vào trong bóng đèn đuôi xoáy chứ đừng nói là một chiếc camera. Nó làm được điều phi thường chưa từng có trong kỷ nguyên Cloud AI."
