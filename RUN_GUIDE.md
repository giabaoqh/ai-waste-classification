# HƯỚNG DẪN KHỞI CHẠY VÀ SỬ DỤNG DỰ ÁN
## AI-Based Waste Recognition and Recycling Recommendation System

> **Hệ thống Nhận diện Rác thải & Gợi ý Tái chế ứng dụng Trí tuệ Nhân tạo (MobileNetV3-Large + FastAPI + React/Vite)**

---

## 📋 MỤC LỤC
1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Yêu cầu tiên quyết (Prerequisites)](#2-yêu-cầu-tiên-quyết-prerequisites)
3. [Cấu trúc thư mục dự án](#3-cấu-trúc-thư-mục-dự-án)
4. [Hướng dẫn cài đặt từ đầu (Setup Step-by-Step)](#4-hướng-dẫn-cài-đặt-từ-đầu-setup-step-by-step)
   - [Bước 1: Chuẩn bị mã nguồn và Checkpoint](#bước-1-chuẩn-bị-mã-nguồn-và-checkpoint)
   - [Bước 2: Cài đặt môi trường Backend (Python)](#bước-2-cài-đặt-môi-trường-backend-python)
   - [Bước 3: Cài đặt môi trường Frontend (Node.js)](#bước-3-cài-đặt-môi-trường-frontend-nodejs)
5. [Khởi chạy ứng dụng (Run Application)](#5-khởi-chạy-ứng-dụng-run-application)
   - [Khởi chạy Backend API (FastAPI)](#51-khởi-chạy-backend-api-fastapi)
   - [Khởi chạy Frontend Web (React + Vite)](#52-khởi-chạy-frontend-web-react--vite)
6. [Hướng dẫn sử dụng giao diện Web](#6-hướng-dẫn-sử-dụng-giao-diện-web)
7. [Các kịch bản kiểm thử & Huấn luyện (Dành cho AI/Backend)](#7-các-kịch-bản-kiểm-thử--huấn-luyện-dành-cho-aibackend)
8. [Xử lý sự cố thường gặp (Troubleshooting / FAQ)](#8-xử-lý-sự-cố-thường-gặp-troubleshooting--faq)

---

## 1. TỔNG QUAN HỆ THỐNG

Dự án gồm 2 khối chính hoạt động đồng bộ:
- **Backend API (`FastAPI` / Python)**:
  - Nạp mạng nơ-ron tích chập **MobileNetV3-Large** đã tinh chỉnh (fine-tuned) trên tập dữ liệu rác thải RealWaste.
  - Tự động nhận diện phần cứng GPU CUDA (RTX 3050...) hoặc tự động fallback sang CPU.
  - Cung cấp RESTful API:
    - `GET /api/health`: Báo cáo tình trạng server, kiến trúc mô hình, GPU/CPU và trạng thái nạp weights.
    - `POST /api/predict`: Tiếp nhận hình ảnh rác thải, tiền xử lý, tính xác suất Softmax cho 6 nhóm rác và sinh khuyến nghị tái chế chuẩn bằng tiếng Việt.
    - Swagger Interactive Docs tại `http://127.0.0.1:8000/docs`.
- **Frontend Web Client (`React` + `Vite` + Vanilla CSS)**:
  - Giao diện người dùng hiện đại, trực quan, hỗ trợ kéo thả ảnh (Drag & Drop) và xem trước (Preview).
  - Hiển thị nhãn rác tiếng Việt, thanh độ tin cậy (Confidence score), thẻ phân loại màu thùng rác chuẩn, hướng dẫn chi tiết và biểu đồ phân bố xác suất của cả 6 nhóm rác.
- **6 nhóm rác mục tiêu nhận diện**:
  1. `plastic` — **Nhựa (Plastic)**
  2. `paper` — **Giấy / Bìa carton (Paper)**
  3. `metal` — **Kim loại (Metal)**
  4. `glass` — **Thủy tinh (Glass)**
  5. `organic` — **Rác hữu cơ (Organic)**
  6. `other` — **Rác khác / Rác hỗn hợp (Other)**

---

## 2. YÊU CẦU TIÊN QUYẾT (PREREQUISITES)

Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt:

| Phần mềm / Công cụ | Phiên bản tối thiểu | Khuyến nghị | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Hệ điều hành** | Windows 10/11, macOS, Linux | Windows 11 (64-bit) | Dự án kiểm thử chính trên Windows Powershell |
| **Python** | Python 3.10 trở lên | Python 3.11 hoặc 3.12 | Tích hợp sẵn `pip` và `venv` |
| **Node.js** | Node.js v18.0.0 trở lên | Node.js v20.x LTS | Đã bao gồm trình quản lý gói `npm` |
| **NVIDIA GPU (Tùy chọn)** | Không bắt buộc | NVIDIA GPU hỗ trợ CUDA | Nếu không có GPU, hệ thống tự động suy luận bằng CPU |
| **Trọng số Checkpoint** | Bắt buộc | `mobilenetv3_best.pth` | Đặt tại thư mục `checkpoints/` |

---

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
D:\AI-Based Waste Recognition and Recycling Recommendation System
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI app, logic nạp model, tiền xử lý & gợi ý tái chế
│   ├── scripts/
│   │   ├── download_data.py          # Script tải dataset RealWaste từ GitHub
│   │   ├── prepare_dataset.py        # Tiền xử lý, gộp nhãn & phân chia train/val/test
│   │   └── train.py                  # Huấn luyện MobileNetV3-Large & xuất checkpoint
│   ├── requirements.txt              # Danh sách thư viện Python (PyTorch, FastAPI, Pillow...)
│   └── .venv/                        # Môi trường ảo Python (được tạo ở bước cài đặt)
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # Component React chính xử lý upload, API call & kết quả
│   │   ├── index.css                 # Hệ thống CSS hiện đại, responsive, dark/light accents
│   │   └── main.jsx                  # Entry point của React
│   ├── index.html                    # File HTML gốc
│   ├── package.json                  # Cấu hình dependency của Frontend (React, Vite...)
│   └── vite.config.js                # Cấu hình Vite dev server
├── checkpoints/
│   ├── mobilenetv3_best.pth          # Trọng số mô hình tốt nhất (Best Epoch 9, Val Acc ~95%)
│   ├── class_mapping.json            # Bảng tra cứu index -> tên lớp rác
│   └── test_evaluation.json          # Kết quả đánh giá chi tiết trên 713 ảnh test
├── data/                             # Dữ liệu ảnh rác thải (raw & processed)
├── docs/                             # Tài liệu kiến trúc, ADR và nghiên cứu dataset
├── README.md                         # Giới thiệu tổng quan dự án
└── RUN_GUIDE.md                      # TÀI LIỆU HƯỚNG DẪN NÀY
```

---

## 4. HƯỚNG DẪN CÀI ĐẶT TỪ ĐẦU (SETUP STEP-BY-STEP)

Mở cửa sổ **PowerShell** hoặc **Command Prompt** tại thư mục gốc của dự án:
```powershell
cd "D:\AI-Based Waste Recognition and Recycling Recommendation System"
```

### Bước 1: Chuẩn bị mã nguồn và Checkpoint
Đảm bảo tệp trọng số `mobilenetv3_best.pth` đã có sẵn tại đường dẫn:
```text
checkpoints/mobilenetv3_best.pth
```
*(Nếu bạn clone repo mới mà chưa có file checkpoint, hãy copy file checkpoint vào thư mục `checkpoints/`)*.

---

### Bước 2: Cài đặt môi trường Backend (Python)

1. **Tạo môi trường ảo Python (nếu chưa có)**:
   ```powershell
   python -m venv backend\.venv
   ```

2. **Kích hoạt môi trường ảo**:
   - Trên **PowerShell**:
     ```powershell
     .\backend\.venv\Scripts\Activate.ps1
     ```
     *(Nếu gặp lỗi UnauthorizedAccess trong PowerShell, chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi chạy lại lệnh trên).*
   - Trên **Command Prompt (CMD)**:
     ```cmd
     backend\.venv\Scripts\activate.bat
     ```

3. **Cài đặt các thư viện cần thiết**:
   ```powershell
   pip install --upgrade pip
   pip install -r backend\requirements.txt
   ```
   *Các thư viện chính bao gồm: `torch`, `torchvision`, `fastapi`, `uvicorn`, `pillow`, `pydantic`, `python-multipart`.*

---

### Bước 3: Cài đặt môi trường Frontend (Node.js)

1. **Di chuyển vào thư mục `frontend`**:
   ```powershell
   cd frontend
   ```

2. **Cài đặt các gói phụ thuộc (Dependencies)**:
   ```powershell
   npm install
   ```

3. **Quay lại thư mục gốc**:
   ```powershell
   cd ..
   ```

---

## 5. KHỞI CHẠY ỨNG DỤNG (RUN APPLICATION)

Để chạy trọn vẹn ứng dụng, bạn cần mở **2 cửa sổ Terminal** riêng biệt (một cho Backend và một cho Frontend).

### 5.1. Khởi chạy Backend API (FastAPI)

Mở **Terminal 1** tại thư mục gốc của dự án và chạy lệnh:

```powershell
backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Dấu hiệu khởi động thành công:**
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Starting up FastAPI application...
INFO:     Checkpoint loaded successfully from 'mobilenetv3_best.pth'. Device: cuda:0, Classes: ['plastic', 'paper', 'metal', 'glass', 'organic', 'other']
INFO:     AI model inference engine warmed up and ready.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Kiểm tra nhanh trên trình duyệt:**
- Kiểm tra trạng thái máy chủ: Mở [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
  - Phản hồi mong đợi:
    ```json
    {
      "status": "ok",
      "service": "waste-recognition-backend",
      "version": "0.2.0",
      "ai_model_loaded": true,
      "device": "cuda:0",
      "model_architecture": "mobilenet_v3_large",
      "checkpoint_loaded": true,
      "target_classes": ["plastic", "paper", "metal", "glass", "organic", "other"]
    }
    ```
- Tài liệu API tương tác (Swagger UI): Mở [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### 5.2. Khởi chạy Frontend Web (React + Vite)

Mở **Terminal 2** tại thư mục gốc của dự án và chạy lệnh:

```powershell
cd frontend
npm run dev
```

**Dấu hiệu khởi động thành công:**
```text
  VITE v6.x.x  ready in 240 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

Bây giờ, mở trình duyệt web và truy cập vào:
👉 **[http://localhost:5173](http://localhost:5173)** hoặc **[http://127.0.0.1:5173](http://127.0.0.1:5173)**

---

## 6. HƯỚNG DẪN SỬ DỤNG GIAO DIỆN WEB

1. **Kiểm tra trạng thái kết nối**:
   - Quan sát góc trên bên phải khung "Tải Ảnh Rác Thải", bạn sẽ thấy biểu tượng màu xanh lá:  
     🟢 `Model Sẵn Sàng (cuda:0)` hoặc `Model Sẵn Sàng (cpu)`.
2. **Chọn ảnh rác thải**:
   - Nhấp vào vùng khung nét đứt hoặc kéo thả trực tiếp một tệp ảnh (`.jpg`, `.jpeg`, `.png`, `.webp`) vào ô.
   - Ứng dụng sẽ hiển thị ngay ảnh xem trước kèm tên tệp và dung lượng.
   - *Gợi ý: Bạn có thể chọn thử các ảnh mẫu có sẵn trong thư mục `data/raw/realwaste/extracted/realwaste-main/RealWaste/Paper/` hoặc `Plastic/`.*
3. **Thực hiện nhận diện**:
   - Bấm nút **"Gửi Nhận Diện AI (POST /api/predict)"**.
   - Chờ trong giây lát (~0.1 - 0.5s tùy GPU/CPU).
4. **Xem kết quả phân tích**:
   - **Thẻ Kết Quả Chính**: Hiển thị biểu tượng và tên nhóm rác tiếng Việt (Ví dụ: 📦 Giấy / Bìa carton, 🧴 Nhựa...) cùng **Độ tin cậy** tính theo phần trăm (%).
   - **Badge Trạng Thái**: Hiển thị badge xanh `MobileNetV3-Large` chứng minh suy luận từ mô hình AI thật (không phải Mock).
   - **Thẻ Khuyến Nghị Tái Chế**:
     - Thùng rác tương ứng (Ví dụ: *Thùng rác tái chế Màu vàng/trắng* hoặc *Thùng rác hữu cơ Màu xanh lá*).
     - Hướng dẫn tổng quan chuẩn bảo vệ môi trường.
     - Các mẹo xử lý từng bước (làm sạch, bóp xẹp, phân loại phụ kiện...).
   - **Phân Bố Xác Suất 6 Nhóm (Softmax Probability Breakdown)**:
     - Biểu đồ thanh ngang chi tiết thể hiện xác suất của từng nhóm trong tổng số 6 nhóm rác (`plastic`, `paper`, `metal`, `glass`, `organic`, `other`).
5. **Đổi ảnh mới**:
   - Bấm nút **"Xóa"** hoặc nhấp vào ảnh xem trước để chọn ảnh khác.

---

## 7. CÁC KỊCH BẢN KIỂM THỬ & HUẤN LUYỆN (DÀNH CHO AI/BACKEND)

Nếu bạn muốn chạy lại các kịch bản kiểm thử, đánh giá hoặc huấn luyện:

### 7.1. Chạy kịch bản kiểm thử suy luận đối chứng
Kiểm tra khả năng phân loại đồng thời giữa mô hình PyTorch cục bộ và API endpoint:
```powershell
backend\.venv\Scripts\python.exe -c "import urllib.request, json; res = urllib.request.urlopen('http://127.0.0.1:8000/api/health'); print(json.loads(res.read().decode()))"
```

### 7.2. Tái tạo phân chia tập dữ liệu (Stratified Split)
Nếu cần phân chia lại tập dữ liệu RealWaste theo tỉ lệ 70% Train / 15% Val / 15% Test với seed cố định:
```powershell
backend\.venv\Scripts\python.exe backend\scripts\prepare_dataset.py
```

### 7.3. Huấn luyện lại mô hình (Training Baseline)
Huấn luyện lại mô hình MobileNetV3-Large 10 epochs trên GPU CUDA:
```powershell
backend\.venv\Scripts\python.exe backend\scripts\train.py --epochs 10 --batch-size 32 --lr 0.0003
```
*Kết quả checkpoint mới nhất và tốt nhất sẽ tự động lưu vào thư mục `checkpoints/`.*

---

## 8. XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING / FAQ)

### ❓ 1. Giao diện báo "Backend Offline" hoặc nút gửi báo lỗi kết nối?
- **Nguyên nhân**: Tiến trình FastAPI ở Terminal 1 chưa chạy hoặc bị tắt.
- **Cách khắc phục**:
  1. Kiểm tra lại Terminal 1 xem Uvicorn có đang chạy ở cổng `8000` không.
  2. Mở trình duyệt thử vào `http://127.0.0.1:8000/api/health`. Nếu không vào được, hãy chạy lại lệnh khởi động Backend ở mục 5.1.

### ❓ 2. Kết quả nhận diện luôn ra "Nhựa (Plastic)" hoặc hiển thị badge "Mock Response"?
- **Nguyên nhân**: Trình duyệt đang lưu cache của phiên kiểm thử Mock tự động trước đây (do hàm `window.fetch` bị can thiệp ở phiên làm việc cũ).
- **Cách khắc phục**:
  1. Nhấn tổ hợp phím **`Ctrl + F5`** (hoặc `Ctrl + Shift + R`) trên trình duyệt để **Hard Refresh** và xóa sạch cache JavaScript.
  2. Hoặc mở một cửa sổ Ẩn danh (Incognito) mới và truy cập `http://127.0.0.1:5173`.

### ❓ 3. Lỗi cổng 8000 hoặc 5173 bị xung đột (Address already in use)?
- **Cách xử lý**:
  - Tìm và tắt tiến trình đang chiếm cổng:
    ```powershell
    # Kiểm tra tiến trình chiếm cổng 8000
    netstat -ano | findstr :8000
    # Tắt tiến trình bằng PID (thay <PID> bằng số tương ứng)
    taskkill /PID <PID> /F
    ```
  - Hoặc khởi chạy Backend trên cổng khác (ví dụ 8080):
    ```powershell
    backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8080
    ```
    *(Lưu ý: Nếu đổi cổng backend, hãy cập nhật biến môi trường `VITE_BACKEND_URL` trong frontend).*

### ❓ 4. Backend báo "Checkpoint file not found at: ..."?
- **Nguyên nhân**: Thiếu file `mobilenetv3_best.pth` trong thư mục `checkpoints/`.
- **Cách xử lý**: Kiểm tra lại thư mục `checkpoints/` ở thư mục gốc dự án. Đảm bảo file `mobilenetv3_best.pth` nằm đúng vị trí.

### ❓ 5. Máy tính không có GPU NVIDIA thì có chạy được không?
- **Hoàn toàn chạy được**: Backend được lập trình với cơ chế tự phát hiện phần cứng thông minh:
  ```python
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ```
  Nếu máy không có GPU hoặc chưa cài driver CUDA, hệ thống sẽ tự động chuyển sang tính toán bằng CPU mà không phát sinh bất kỳ lỗi nào.

---

*Tài liệu được biên soạn phục vụ đồ án Capstone: **AI-Based Waste Recognition and Recycling Recommendation System**.*
