# Project Specification (PROJECT_SPEC.md)

## 1. Thông tin chung (General Information)
- **Tên dự án**: AI-Based Waste Recognition and Recycling Recommendation System
- **Mục tiêu**: Hỗ trợ người dùng nhận diện loại rác qua ảnh chụp/tải lên, cung cấp độ tin cậy và hướng dẫn phân loại/tái chế chính xác theo chuẩn môi trường.
- **Quy mô dự án**: Đồ án tốt nghiệp đại học (Capstone Project) cho nhóm 3 thành viên (Bảo, Huy, Minh).

---

## 2. Phạm vi và Yêu cầu Chức năng (Functional Scope)

### 2.1. Danh mục rác phân loại (Waste Categories)
Hệ thống dự kiến nhận diện 6 nhóm rác thải chính:
1. **Plastic (Nhựa)**: Chai nhựa, hộp nhựa, túi nilon, bao bì nhựa...
2. **Paper (Giấy)**: Bìa carton, sách báo, túi giấy, hóa đơn...
3. **Metal (Kim loại)**: Vỏ lon nhôm, hộp thiếc, nắp chai, kim loại phế liệu...
4. **Glass (Thủy tinh)**: Chai thủy tinh, lọ thủy tinh, mảnh vỡ thủy tinh...
5. **Organic (Hữu cơ)**: Thức ăn thừa, vỏ rau củ quả, lá cây, bã cà phê...
6. **Other (Khác)**: Rác thải nguy hại (pin, bóng đèn), rác không thể tái chế hoặc chưa xác định.

### 2.2. Luồng chức năng cốt lõi (Core Workflow)
1. **Nhập liệu**: Người dùng tải ảnh lên từ thiết bị hoặc chụp ảnh trực tiếp qua webcam/camera điện thoại.
2. **Nhận diện bằng AI**:
   - Tiền xử lý ảnh (resize, normalize theo chuẩn mô hình).
   - Đưa ảnh qua mô hình PyTorch để trích xuất đặc trưng và phân loại.
   - Trả về nhãn dự đoán (Predicted Label) và điểm tin cậy (Confidence Score, ví dụ: 94.2%).
   - Nếu điểm tin cậy dưới ngưỡng an toàn (ví dụ: < 50%), hệ thống sẽ cảnh báo hoặc xếp vào nhóm "Other / Cần kiểm tra lại".
3. **Cơ sở tri thức gợi ý tái chế (Disposal & Recycling Recommendation Knowledge Base)**:
   - **Tách biệt hoàn toàn logic nhận diện AI với cơ sở dữ liệu hướng dẫn xử lý**.
   - Cung cấp hướng dẫn rõ ràng:
     - Cách làm sạch (ví dụ: tráng sạch vỏ hộp sữa trước khi bỏ vào thùng giấy/nhựa).
     - Thùng rác tương ứng (màu sắc/ký hiệu thùng rác chuẩn).
     - Khả năng tái chế và các lưu ý an toàn (đặc biệt với pin, thủy tinh vỡ).

---

## 3. Yêu cầu Phi Chức năng (Non-Functional Requirements)

- **Thời gian phản hồi (Latency)**: Thời gian xử lý từ lúc gửi ảnh đến khi nhận kết quả gợi ý < 2 giây khi chạy với backend cục bộ.
- **Giao diện (UI/UX)**: Đơn giản, thân thiện trên trình duyệt web, responsive cho cả máy tính và điện thoại.
- **Khả năng mở rộng**: Tách rời module AI inference, API backend và Knowledge Base để dễ dàng cập nhật dữ liệu xử lý rác mà không cần train lại mô hình.
- **Bảo mật và an toàn dữ liệu**: Không lưu trữ trái phép dữ liệu cá nhân của người dùng; dữ liệu nhạy cảm được cấu hình qua biến môi trường (.env) và không đẩy lên Git.

---

## 4. Xác nhận Yêu cầu vs Giả định (Confirmed vs Assumptions)

| Nội dung | Trạng thái | Ghi chú |
| :--- | :--- | :--- |
| 6 danh mục rác (Plastic, Paper, Metal, Glass, Organic, Other) | **Đã xác nhận (Confirmed)** | Yêu cầu cốt lõi của đề tài |
| Tách biệt AI inference và Knowledge Base | **Đã xác nhận (Confirmed)** | Đảm bảo tính kiểm chứng và độ tin cậy |
| Công nghệ: React + Vite + FastAPI + PyTorch | **Đã đề xuất (Proposed)** | Đã kiểm tra tính tương thích trên máy trạm |
| Cơ sở dữ liệu lưu lịch sử / bài viết tái chế | **Chưa quyết định (TBD)** | Lựa chọn giữa PostgreSQL cục bộ hoặc Supabase |
| Dataset cụ thể và độ chính xác mô hình | **Chưa triển khai (Pending)** | Chưa tải dataset, chưa train, không tự bịa thông số |
