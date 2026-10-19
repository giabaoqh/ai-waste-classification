# Task Breakdown & Milestones (TASKS.md)

## 📌 Phân loại giai đoạn dự án (Project Phases)

Dự án được phân chia theo từng nhánh tính năng độc lập, giúp 3 thành viên có thể phối hợp nhịp nhàng mà không bị xung đột mã nguồn.

---

### Giai đoạn 1: Thiết lập nền tảng dự án (Repository & Environment Setup)
- [x] Khởi tạo Git repository cục bộ và cấu hình branch `main`.
- [x] Cấu hình file `.gitignore` toàn diện (bảo vệ secrets, dataset, model weights).
- [x] Soạn thảo bộ tài liệu kỹ thuật ban đầu (`README.md`, `docs/*`).
- [x] Kết nối remote repository GitHub (`giabaoqh/ai-waste-classification`).
- [ ] Thiết lập môi trường ảo Python (`.venv`) và PyTorch với hỗ trợ CUDA.
- [ ] Tạo các nhánh tính năng: `feature/frontend`, `feature/ai-model`, `feature/backend-api`.

---

### Giai đoạn 2: Phát triển theo từng nhánh tính năng (Feature Branches)

#### 🎨 Nhánh `feature/frontend` (Giao diện người dùng)
- [ ] Khởi tạo dự án React + Vite trong thư mục `frontend/`.
- [ ] Xây dựng hệ thống giao diện (UI) hiện đại, responsive:
  - Khung tải ảnh (Drag & Drop / File Input / Webcam Capture).
  - Vùng hiển thị ảnh xem trước và nút bấm phân tích.
  - Trạng thái loading và thông báo lỗi thân thiện.
  - Card hiển thị kết quả phân loại: nhãn, thanh tiến trình % tin cậy, thùng rác tương ứng.
  - Vùng khuyến nghị tái chế chi tiết và mẹo bảo vệ môi trường.
- [ ] Tích hợp gọi API `/api/predict` từ backend.
- [ ] Kiểm thử giao diện trên trình duyệt Desktop và Mobile.

#### 🧠 Nhánh `feature/ai-model` (Nghiên cứu & Huấn luyện AI)
- [ ] Khảo sát và lựa chọn tập dữ liệu (Dataset) phân loại rác phù hợp (ví dụ: TrashNet, Garbage Classification dataset).
- [ ] Xây dựng script tiền xử lý và tăng cường dữ liệu (Data Augmentation: xoay, lật, điều chỉnh độ sáng).
- [ ] Thiết lập baseline model (Transfer Learning với MobileNetV3 hoặc EfficientNet để tối ưu cho GPU 4GB VRAM).
- [ ] Xây dựng pipeline huấn luyện, validation và tính toán các chỉ số đánh giá (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- [ ] Tối ưu hóa mô hình và lưu trọng số tốt nhất (`best_model.pth` hoặc export sang ONNX).
- [ ] Viết module suy luận (Inference wrapper) nhận ảnh đầu vào và trả về class + probability.

#### ⚙️ Nhánh `feature/backend-api` (API Backend & Cơ sở tri thức)
- [ ] Khởi tạo khung ứng dụng FastAPI trong thư mục `backend/`.
- [ ] Cấu hình CORS middleware cho phép frontend kết nối.
- [ ] Xây dựng cơ sở tri thức phân loại & tái chế rác (`recycling_kb.json` hoặc database):
  - Định nghĩa quy tắc xử lý cho 6 nhóm rác: Plastic, Paper, Metal, Glass, Organic, Other.
- [ ] Xây dựng endpoint `POST /api/predict`:
  - Tiếp nhận file ảnh upload qua Multipart Form.
  - Gọi module AI inference để lấy kết quả phân loại.
  - Ghép thông tin khuyến nghị tương ứng từ Knowledge Base.
  - Trả về payload JSON hoàn chỉnh.
- [ ] Xây dựng endpoint kiểm tra trạng thái hệ thống `GET /api/health`.

---

### Giai đoạn 3: Tích hợp & Kiểm thử toàn diện (Integration & Evaluation)
- [ ] Tích hợp frontend với backend và AI Engine trên máy trạm cục bộ.
- [ ] Kiểm thử end-to-end với các mẫu ảnh rác thực tế ngoài đời sống.
- [ ] Đánh giá độ trễ xử lý (inference time + API roundtrip) và mức tiêu hao VRAM của GPU.
- [ ] Soạn thảo báo cáo kết quả và tài liệu hướng dẫn thuyết trình đồ án.
