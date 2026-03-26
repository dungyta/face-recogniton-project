# 🌟 Hướng Dẫn Làm Slide Pitching (Dành Cho Khán Giả "Newbie")
> **Mục tiêu:** Trình bày dự án Face Recognition trên Camera Indoor sao cho **ai cũng hiểu được**, kể cả sếp, nhân sự, hay người ngoại đạo không có background về AI/Code.
> **Từ khóa chính:** Đơn giản hóa, dùng ẩn dụ (analogy), tập trung vào "Vấn đề" và "Kết quả" thay vì "Toán học" và "Code".

---

## 🎯 Cấu Trúc Slide (10-15 Phút)

| Slide # | Tiêu đề Gợi ý                                           | Nội dung chính (Kịch bản)                                                             | Thời gian |
| :-----: | ------------------------------------------------------- | ------------------------------------------------------------------------------------- | :-------: |
|    1    | **Tiêu đề & Chào hỏi**                                  | Tên dự án: "Nhỏ mà có võ - Mang AI nhận diện khuôn mặt vào Camera tại nhà".           |  1 phút   |
|    2    | **Khởi động: AI nhận diện khuôn mặt diễn ra thế nào?**  | Giới thiệu bằng ví dụ đời thực (như cách con người nhận ra người quen).               |  2 phút   |
|    3    | **Vấn đề: Đưa AI "khổng lồ" vào Camera bé xíu**         | Giải thích sự chênh lệch giữa AI cloud thông thường và hạn chế của camera.            |  2 phút   |
|    4    | **Giải pháp: Cấp "Căn cước công dân số" cho khuôn mặt** | Giải thích nôm na cách thuật toán (Embedding) biến mặt thành các con số.              |  2 phút   |
|    5    | **Bí kíp 1: Chọn "Vận động viên" phù hợp**              | Thay vì "Lực sĩ" (model to), ta chọn "Ninja" (model MobileNet nhỏ gọn, nhanh).        |  2 phút   |
|    6    | **Bí kíp 2: Chế độ "Ép cân" khắc nghiệt**               | Quá trình cắt giảm bộ nhớ (Giảm params) nhưng vẫn giữ được trí thông minh (Accuracy). |  2 phút   |
|    7    | **Kết quả thực tế**                                     | Khoe thành quả: Chạy nhanh ra sao? Chiếm ít dung lượng thế nào?                       |  2 phút   |
|    8    | **Hỏi & Đáp (Q&A)**                                     | Cảm ơn & sẵn sàng giải đáp.                                                           | Tùy chọn  |

---

## 📖 Chi Tiết Từng Slide & Lời Thoại (Speaker Notes)

### 🟢 Slide 1: Welcome & Mở bài
- **Visuals:** Hình ảnh một chiếc camera an ninh indoor, bên cạnh là icon một bộ não AI đang phát sáng. Tiêu đề to rõ. Tên người trình bày.
- **Kịch bản nói (Speaker Notes):** 
  > *"Chào mọi người. Hôm nay mình sẽ kể cho mọi người nghe câu chuyện về việc làm sao chúng mình nhét được một 'bộ não AI' thông minh vào trong một thiết bị cực kỳ nhỏ bé và yếu ớt: đó là những chiếc Camera an ninh ở trong nhà bạn."*

### 🟢 Slide 2: AI nhìn khuôn mặt chúng ta như thế nào?
- **Visuals:** So sánh: 1 bên là mắt người nhìn (hình 1 khuôn mặt), 1 bên là mắt máy tính nhìn (khuôn mặt biến thành 1 chuỗi các con số `[0.12, -0.4, 0.8...]`).
- **Kịch bản nói:**
  > *"Khi chúng ta tự động nhận ra một người quen, não bộ chụp lại rồi so sánh với trí nhớ. AI cũng vậy! Thay vì nhớ 'mắt to, mũi cao', AI biến khuôn mặt chúng ta thành một 'Căn Cước Công Dân' bằng số (dãy khoảng 500 con số). Khi ai đó bước qua camera, AI sẽ rút thẻ căn cước đó ra dò với danh sách trong nhà, khớp nhau thì mở cửa!"*

### 🔴 Slide 3: Vấn đề "Con voi và cái tủ lạnh"
- **Visuals:** Hình ảnh ẩn dụ. Một bên là con voi khổng lồ (ghi chữ: Mô hình AI thông thường), một bên là cái hộp/tủ lạnh bé xíu (ghi chữ: Bộ nhớ và chip của Camera).
- **Kịch bản nói:**
  > *"Nghe thì dễ, nhưng vấn đề lớn nhất ở đây là gì? Các hệ thống AI mạnh mẽ thường chạy trên những cỗ siêu máy tính khổng lồ, ngốn điện và đắt tiền (Cloud, Server). Trong khi đó, chiếc camera trong nhà giá chỉ vài trăm ngàn, bộ nhớ thua cả một chiếc điện thoại cục gạch ngày xưa (chỉ vài MB). Đưa AI vào camera không khác gì nhét một con voi vào một chiếc tủ lạnh mini!"*

### 🟡 Slide 4: Giải pháp 1 - Tìm "Võ sư tàng hình" thay vì "Lực sĩ Cử tạ" (Kiến trúc mô hình)
- **Visuals:** 
  - Bên trái: Lực sĩ to lớn (ghi "SphereFace - 135 MB").
  - Bên phải: Một Ninja nhỏ gọn, nhanh nhẹn (ghi "MobileNet - 2 MB").
- **Kịch bản nói:**
  > *"Thay vì bê nguyên một mô hình AI khổng lồ như các ông lớn công nghệ, chúng mình phải áp dụng kiến trúc đặc biệt có tên là MobileNet. Hãy tưởng tượng nó như một võ sư Ninja: không cần cơ bắp cuồn cuộn ngốn đồ ăn (bộ nhớ), nhưng ra đòn cực kỳ chuẩn xác và nhanh nhẹn. Ninja này sinh ra là để dành cho các thiết bị yếu!"*

### 🟡 Slide 5: Giải pháp 2 - Chế độ "Ép cân" khắc nghiệt (Tối ưu hóa dung lượng)
- **Visuals:** Hình ảnh quá trình vắt nước chiếc khăn, hoặc hình ảnh "ép mỡ". Biểu đồ đường thẳng tắp: Dung lượng giảm 80%, nhưng Độ thông minh (Accuracy) chỉ giảm 3%.
- **Kịch bản nói:**
  > *"Chọn được Ninja rồi, chúng mình còn bắt họ... ép cân sâu hơn nữa! Bằng các kỹ thuật gọt giũa thuật toán, loại bỏ các phép tính thừa thãi, chúng mình đã ép mô hình AI nhỏ đi tận 4 lần (từ 8MB xuống chưa tới 2MB). Điều kỳ diệu là: dù AI bị thu nhỏ đi 80% thể tích não, nó vẫn nhớ mặt người cực kỳ xuất sắc (độ chính xác vẫn đạt 99% trên tập dữ liệu chuẩn)."*

### 🟢 Slide 6: Kết quả - "Nhỏ nhưng có võ"
- **Visuals:** Các con số biết nói to oành: 
  - Dung lượng: **~1.4 MB** (chỉ bằng 1 bức ảnh chụp bằng điện thoại).
  - Tốc độ: **Chớp mắt** (chỉ 1-2 phần nghìn giây).
  - Ứng dụng: Nhận diện người nhà an toàn, chạy ngay trên thiết bị không cần mạng Internet!
- **Kịch bản nói:**
  > *"Và đây là kết quả cuối cùng. Tụi mình đóng gói được bộ não AI chỉ vỏn vẹn 1.4 MB - nhẹ bằng 1 bài hát MP3 chất lượng thấp. Camera có thể nhận diện bạn trong cái chớp mắt mà không cần tải dữ liệu lên mạng, đảm bảo 100% quyền riêng tư. Camera rẻ tiền giờ đã biến thành camera thông minh!"*

### 🟢 Slide 7: Chặng đường tiếp theo (Deploy)
- **Visuals:** Hình ảnh cuộn mã code bay vào một con chip điện tử.
- **Kịch bản nói:**
  > *"Bước tiếp theo của tụi mình là đưa nguyên khối óc nhạy bén này đóng gói vào 'con chip' của chiếc camera để nó có thể chạy ổn định 24/7 trong mọi nhà. Cảm ơn mọi người đã lắng nghe!"*

---

## 🎨 Lời khuyên khi làm Slide cho Newbie:
1. **Rule of three (Quy tắc số 3):** Không đưa quá nhiều chữ lên màn hình. Mỗi slide chỉ tối đa 3 ý chính, gạch đầu dòng siêu ngắn gọn.
2. **Hình ảnh > Chữ:** Slide cho newbie nên có thật nhiều hình minh họa, icon (dùng Flaticon, Freepik).
3. **Tuyệt đối không:** KHÔNG copy paste code lên màn hình, KHÔNG show bảng loss function, KHÔNG nói về đạo hàm hay ma trận. Thay từ "Cosine Similarity" bằng *"Phương pháp đối chiếu thẻ Căn cước"*. Thay "Feature Extraction/Embedded Layer" bằng *"Hệ thống quy đổi khuôn mặt ra con số"*.
4. **Kể chuyện (Storytelling):** Hãy biến bài thuyết trình thành câu chuyện về một người thợ thủ công cặm cụi gọt giũa một cỗ máy khổng lồ thành một công cụ tinh xảo bỏ túi.
