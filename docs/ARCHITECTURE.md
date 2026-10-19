# System Architecture (ARCHITECTURE.md)

## 1. Tổng quan Kiến trúc (System Overview)

Hệ thống được thiết kế theo kiến trúc phân lớp (Layered Architecture), tách bạch rõ ràng giữa tầng hiển thị (Presentation), tầng nghiệp vụ / API (Backend), tầng tính toán Trí tuệ Nhân tạo (AI Service) và cơ sở tri thức (Knowledge Base).

```
+-------------------------------------------------------------------+
|                        Client Layer (Web)                         |
|                   React + Vite (Modern Web App)                   |
+---------------------------------+---------------------------------+
                                  |
                   HTTP / REST API (Multipart Form)
                                  |
                                  v
+-------------------------------------------------------------------+
|                       API Gateway & Backend                       |
|                          Python + FastAPI                         |
|                                                                   |
|   +--------------------------+     +--------------------------+   |
|   |  Image Validation & Pre  |     |   Recycling Guideline    |   |
|   |        processing        |     |         Service          |   |
|   +------------+-------------+     +------------+-------------+   |
+----------------|--------------------------------|-----------------+
                 |                                |
                 v                                v
+--------------------------------+  +-------------------------------+
|         AI Engine Layer        |  |    Knowledge Base / Storage   |
|            (PyTorch)           |  |                               |
|                                |  |  - Recycling Rules DB/JSON     |
|  - CNN / Vision Transformer    |  |  - PostgreSQL / Supabase      |
|  - CUDA Accelerated (RTX 3050) |  |    (Lịch sử & tra cứu)        |
+--------------------------------+  +-------------------------------+
```

---

## 2. Các Thành phần Hệ thống (Components)

### 2.1. Frontend (Client Layer)
- **Công nghệ**: React (phiên bản mới nhất) chạy trên nền Vite.
- **Trách nhiệm**:
  - Giao diện người dùng thân thiện, cho phép kéo thả ảnh, chọn ảnh từ máy hoặc chụp từ camera.
  - Hiển thị ảnh xem trước (preview), trạng thái đang xử lý (loading state).
  - Trực quan hóa kết quả: Nhãn dự đoán, thanh đo độ tin cậy (Confidence Bar), biểu tượng thùng rác và danh sách các bước xử lý/tái chế.
  - Giao tiếp với backend qua RESTful API.

### 2.2. Backend API (FastAPI)
- **Công nghệ**: FastAPI (Python 3.13), Uvicorn.
- **Trách nhiệm**:
  - Tiếp nhận file ảnh upload qua endpoint `/api/predict`.
  - Kiểm tra tính hợp lệ của file (định dạng ảnh: JPEG, PNG; dung lượng tối đa).
  - Chuyển tiếp tensor ảnh tới AI Engine để lấy nhãn và xác suất.
  - Truy vấn Knowledge Base để lấy bộ chỉ dẫn tái chế tương ứng với nhãn rác.
  - Đóng gói dữ liệu trả về theo chuẩn JSON.

### 2.3. AI Engine (PyTorch)
- **Công nghệ**: PyTorch với CUDA runtime (tối ưu cho NVIDIA GeForce RTX 3050 Laptop GPU - 4GB VRAM).
- **Trách nhiệm**:
  - Load model weights đã huấn luyện vào GPU/CPU.
  - Thực hiện Image Preprocessing (Resize, Center Crop, ToTensor, Normalize).
  - Chạy Forward Pass (Inference) không tính gradient (`torch.no_grad()`).
  - Áp dụng hàm Softmax để tính độ tin cậy cho từng lớp rác.

### 2.4. Waste Recycling Knowledge Base
- **Đặc điểm**: Tách rời khỏi mô hình AI.
- **Lý do thiết kế**: Mô hình AI chỉ có nhiệm vụ thị giác máy tính ("Đây là cái gì?"). Còn quy tắc phân loại ("Cần xử lý như thế nào?") tuân theo quy định môi trường thực tế, có thể bổ sung, sửa đổi mà không phải train lại mô hình.
- **Cấu trúc dữ liệu**:
  - `category`: Tên danh mục (ví dụ: `plastic`).
  - `name_vi`: Tên tiếng Việt (ví dụ: `Nhựa`).
  - `bin_type`: Loại thùng rác và màu sắc quy định.
  - `cleaning_steps`: Các bước làm sạch trước khi vứt.
  - `recyclable`: Khả năng tái chế (Có / Không / Tùy điều kiện).
  - `tips`: Mẹo tái chế hoặc lưu ý an toàn.

---

## 3. Luồng Dữ Liệu Chi Tiết (Data Flow)

```
1. User chọn ảnh -> 
2. Frontend gửi POST /api/predict (FormData kèm ảnh) -> 
3. FastAPI nhận ảnh, kiểm tra kích thước & định dạng -> 
4. AI Service nạp tensor vào RTX 3050, thực hiện inference -> 
5. Model trả về: { label: "plastic", confidence: 0.94 } -> 
6. Service tra cứu Knowledge Base cho nhãn "plastic" -> 
7. FastAPI trả về response đầy đủ cho Client -> 
8. Frontend render giao diện kết quả và khuyến nghị tái chế.
```
