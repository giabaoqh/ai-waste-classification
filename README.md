# AI-Based Waste Recognition and Recycling Recommendation System

> **Capstone Project**: Hệ thống nhận diện rác thải và gợi ý tái chế dựa trên Trí tuệ Nhân tạo.

---

## 📌 Giới thiệu dự án (Project Overview)

Dự án nhằm mục tiêu phát triển hệ thống hỗ trợ người dùng phân loại rác thải sinh hoạt thông qua hình ảnh. Hệ thống sẽ nhận diện loại rác, đánh giá độ tin cậy (confidence score) và đưa ra các hướng dẫn xử lý hoặc tái chế phù hợp theo tiêu chuẩn bảo vệ môi trường.

## 🌟 Tính Năng Nổi Bật Của Hệ Thống (Key Features)

### 1. Nhận Diện Rác Thải Đa Phương Thức (Multi-modal Input)
- **Tải ảnh từ máy (File Upload)**: Kéo thả trực quan, kiểm tra tự động định dạng và dung lượng tối đa 10 MB.
- **Chụp ảnh trực tiếp (Live Webcam / Camera)**: Tích hợp trực tiếp trên trình duyệt, phù hợp cho điện thoại và máy tính bảng.
- **Nhập liên kết ảnh (Predict by Image URL)**: Phân loại trực tiếp từ link ảnh công khai không cần tải về máy.
- **Phân loại hàng loạt (Batch Processing)**: Xử lý đồng thời tối đa 10 ảnh trong 1 lần gửi, xem kết quả theo danh sách.

### 2. Trực Quan Hóa Trí Tuệ Nhân Tạo (Explainable AI - Grad-CAM)
- Tích hợp phương pháp **Grad-CAM (Gradient-weighted Class Activation Mapping)** trên tầng tích chập cuối của `MobileNetV3-Large`.
- Cho phép người dùng và chuyên gia quan sát **Bản đồ nhiệt kích hoạt (Heatmap Overlay)**: Vùng màu Đỏ - Vàng thể hiện chính xác vị trí và đặc trưng mà mạng nơ-ron tập trung để đưa ra quyết định phân loại, loại bỏ hoàn toàn tính chất "hộp đen" của AI.

### 3. Vòng Lặp Phản Hồi Người Dùng (User Feedback Loop)
- Người dùng có thể đánh giá kết quả: **Chính xác (👍)** hoặc **Báo sai (👎)**.
- Khi báo sai, người dùng có thể chọn nhóm rác thực tế và nhập ghi chú. Dữ liệu này được lưu trữ để làm giàu tập dữ liệu tái huấn luyện mô hình (Active Learning).

### 4. Quản Lý & Xuất Dữ Liệu (History & CSV Export)
- Lưu trữ lịch sử phân loại trên SQLite (chỉ lưu metadata, không lưu nhị phân ảnh để bảo mật riêng tư).
- Hỗ trợ phân trang, lọc theo thời gian và nhóm rác.
- **Xuất báo cáo CSV chuẩn UTF-8 BOM** hiển thị tiếng Việt hoàn hảo trên Microsoft Excel.

### 5. Bảng Điều Khiển Đánh Giá & Hiệu Suất Mô Hình (Model Evaluation Dashboard)
- Hiển thị **Ma trận nhầm lẫn (Confusion Matrix 6x6)** của mô hình trên 713 ảnh tập kiểm thử độc lập.
- Chỉ số chi tiết từng lớp: **Precision, Recall, F1-Score, Support** (Độ chính xác tổng đạt **95.98%**).
- Đồ thị tiến trình huấn luyện qua 10 Epochs (Loss & Accuracy trên tập Train và Validation).

### 6. Danh Bạ Trạm Thu Gom & Tái Chế Xanh (Recycling Directory)
- Cung cấp danh mục các điểm thu gom rác tái chế và rác nguy hại (pin cũ, thiết bị điện tử, vỏ hộp giấy, nhựa) tại các đô thị lớn: **Hà Nội, TP. Hồ Chí Minh, Đà Nẵng, Cần Thơ**.
- Bộ lọc theo thành phố và loại vật liệu, hiển thị giờ mở cửa và số điện thoại liên hệ.

### 7. Trải Nghiệm Người Dùng Hiện Đại (UX & Dark Mode)
- Hỗ trợ chuyển đổi giao diện **Sáng / Tối (Dark / Light Mode)** mượt mà, lưu trạng thái tự động.
- Giao diện đáp ứng (Responsive) tối ưu cho cả Desktop, Tablet và Mobile.

---

## 👥 Thành viên nhóm & Phân công trách nhiệm (Team Members & Roles)

| Thành viên | Vai trò chính | Nhiệm vụ trọng tâm | Nhánh làm việc |
| :--- | :--- | :--- | :--- |
| **Huy** | **Team Leader** | Điều phối dự án, review PR, quản lý tích hợp và nhánh `main` | `main` |
| **Bảo** | **AI Model & Dataset** | Thẩm định dataset RealWaste, tiền xử lý, huấn luyện và đánh giá mô hình AI | `feature/ai-model` |
| **Minh** | **Backend, Frontend & Integration** | Phát triển API FastAPI, giao diện React/Vite và tích hợp end-to-end | `feature/backend-api`, `feature/frontend` |

### Chi tiết trách nhiệm:

#### 1. Huy — Team Leader
- Điều phối nhóm và theo dõi tiến độ tổng thể của dự án.
- Phân chia nhiệm vụ và điều phối tích hợp giữa mô hình AI, backend và frontend.
- Review pull requests, giải quyết xung đột tích hợp và giám sát, quản lý nhánh `main`.
- Điều phối kiểm thử toàn hệ thống và tổng hợp kết quả bàn giao của đồ án.

#### 2. Bảo — AI Model & Dataset Member
- Thẩm định và chuẩn bị tập dữ liệu RealWaste.
- Thực hiện tiền xử lý và phân chia tập dữ liệu (train/validation/test) đảm bảo tính tái lập.
- Huấn luyện, đánh giá và tối ưu hóa mô hình phân loại rác thải.
- Đóng gói và bàn giao mô hình đã huấn luyện kèm các chỉ số đánh giá (metrics) để tích hợp vào backend.

#### 3. Minh — Backend, Frontend & Integration Member
- Phát triển và duy trì REST API backend bằng FastAPI.
- Xây dựng giao diện người dùng React/Vite.
- Tích hợp tính năng tải ảnh, kết quả dự đoán, điểm tin cậy (confidence score) và các khuyến nghị xử lý/tái chế.
- Kiểm thử luồng hoạt động đầu-cuối (end-to-end) của toàn bộ ứng dụng.

---

## 🛠 Công nghệ dự kiến (Planned Tech Stack)

- **Frontend**: React + Vite (JavaScript/TypeScript, Vanilla CSS / Modern UI).
- **Backend API**: Python + FastAPI.
- **AI / Deep Learning**: PyTorch (Tăng tốc bằng NVIDIA CUDA).
- **Database**: PostgreSQL hoặc Supabase (sẽ quyết định sau).

---

## 🌿 Cấu trúc nhánh Git (Branching Strategy)

- `main`: Nhánh chính, chỉ chứa mã nguồn ổn định đã kiểm thử.
- `feature/frontend`: Phát triển giao diện người dùng React + Vite.
- `feature/ai-model`: Thu thập/xử lý dataset, huấn luyện và đánh giá mô hình phân loại rác.
- `feature/backend-api`: Xây dựng REST API bằng FastAPI, tích hợp inference mô hình và cơ sở tri thức tái chế.

---

## 📊 Trạng thái hiện tại (Current Status)

- [x] Khởi tạo cấu trúc repository và tài liệu thiết kế hệ thống.
- [x] Thiết lập môi trường ảo Python (`.venv`) và PyTorch với hỗ trợ GPU CUDA.
- [x] Chuẩn hóa và phân chia tập dữ liệu RealWaste (3,327 train / 712 val / 713 test).
- [x] Huấn luyện mô hình MobileNetV3-Large trên GPU RTX 3050.
- [x] Xây dựng REST API bằng FastAPI, nạp checkpoint và suy luận thời gian thực (`POST /api/predict`).
- [x] Xây dựng giao diện người dùng React + Vite hiển thị phân loại, phân bố xác suất Softmax và gợi ý tái chế tiếng Việt.
- [x] Kiểm thử tích hợp tự động và kiểm thử đầu-cuối (End-to-End).

---

## 🚀 Hướng dẫn khởi chạy hệ thống (Quickstart Guide)

### 1. Yêu cầu tiên quyết (Prerequisites)
- **Python**: Phiên bản 3.10 trở lên (khuyến nghị kèm NVIDIA GPU hỗ trợ CUDA).
- **Node.js**: Phiên bản 18 trở lên (kèm `npm`).
- **Trọng số mô hình (Checkpoint)**: Tệp `mobilenetv3_best.pth` phải được đặt tại:
  ```text
  checkpoints/mobilenetv3_best.pth
  ```

### 2. Cài đặt phụ thuộc (Dependencies)
- **Backend**:
  ```powershell
  python -m venv backend\.venv
  backend\.venv\Scripts\pip install -r backend\requirements.txt
  ```
- **Frontend**:
  ```powershell
  cd frontend
  npm install
  cd ..
  ```

### 3. Khởi chạy ứng dụng (Run Application)
- **Khởi động Backend (FastAPI)**:
  ```powershell
  backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
  - Kiểm tra trạng thái: `GET http://127.0.0.1:8000/api/health`
  - Swagger API Docs: `http://127.0.0.1:8000/docs`

- **Khởi động Frontend (React + Vite)**:
  ```powershell
  cd frontend
  npm run dev
  ```
  - Truy cập giao diện tại: `http://127.0.0.1:5173`

### 4. Chạy kiểm thử tự động (Running Tests)
Chạy bộ kiểm thử tích hợp mô hình và API:
```powershell
backend\.venv\Scripts\python.exe scratch\test_model_integration.py
```

---

## ⚠️ Giới hạn hiện tại của mô hình (Model Limitations)

1. **Phạm vi phân loại**: Mô hình được huấn luyện để phân loại đúng 6 nhóm rác mục tiêu (`plastic`, `paper`, `metal`, `glass`, `organic`, `other`).
2. **Không phân biệt chi tiết loại nhựa & mức độ nhiễm bẩn**: Mô hình không nhận diện cụ thể các phân nhóm polymer nhựa chi tiết (như PET, HDPE, PP, PVC) và không đánh giá định lượng độ nhiễm bẩn dầu mỡ/hóa chất của vật thể.
3. **Đặc tính nhóm 'Rác khác' (Other)**: Đây là danh mục tồn dư tổng hợp cho các vật thể hỗn hợp, cao su, vải vụn hoặc rác thải không thuộc 5 nhóm phân loại chuẩn. Khi gặp nhóm này, hệ thống khuyến cáo người dùng phân loại thủ công.
4. **Độ tin cậy & Quy định địa phương**: Điểm tin cậy (confidence score) thể hiện xác suất toán học qua hàm Softmax của mô hình AI, không phải cam kết tuyệt đối. Khuyến nghị xử lý rác mang tính tham khảo; người dùng cần tuân thủ các quy chuẩn và màu sắc thùng rác theo quy định tại từng địa phương.

---

## 📖 Tài liệu dự án (Documentation)

Chi tiết thiết kế và kế hoạch được lưu trữ trong thư mục `docs/`:
- [PROJECT_SPEC.md](docs/PROJECT_SPEC.md): Yêu cầu kỹ thuật và phạm vi dự án.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): Kiến trúc hệ thống và luồng dữ liệu.
- [TASKS.md](docs/TASKS.md): Danh sách công việc và phân công nhiệm vụ.
- [PROGRESS.md](docs/PROGRESS.md): Nhật ký tiến độ thực hiện.
- [DECISIONS.md](docs/DECISIONS.md): Các quyết định kiến trúc và công nghệ (ADR).
- [DATASET_RESEARCH.md](docs/DATASET_RESEARCH.md): Nghiên cứu và khảo sát tập dữ liệu rác thải.

