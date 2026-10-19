# Project Progress Log (PROGRESS.md)

Nhật ký ghi nhận tiến độ thực tế của dự án. Mọi mục đánh dấu hoàn thành đều phải được kiểm chứng bằng câu lệnh thực tế.

---

## 📅 Nhật ký tiến độ (Changelog)

### Ngày 19/10/2026

#### 1. Khảo sát môi trường hệ thống (Phase 1: Environment Inspection)
- [x] Kiểm tra phần cứng và hệ điều hành: Windows 11 (64-bit), NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), Driver `616.64`.
- [x] Kiểm tra các runtime đã cài đặt:
  - Python: `3.13.16` (tại `C:\Users\GIGABYTE\AppData\Local\Programs\Python\Python313\python.exe`) và `3.14.4`.
  - Node.js: `v24.15.0`, npm: `11.12.1`.
  - Git: `2.49.0.windows.1`.
- [x] Kiểm tra tính tương thích: Python 3.13 hoàn toàn tương thích với các bản PyTorch pre-built CUDA (`cu124` / `cu126`).
- [x] Kiểm tra trạng thái Git remote: `https://github.com/giabaoqh/ai-waste-classification.git` đã tồn tại, là repo public và hiện tại hoàn toàn trống (chưa có commit/nhánh nào).

#### 2. Khởi tạo cấu trúc & tài liệu dự án (Repository Setup)
- [x] Tạo file `.gitignore` toàn diện: Bảo vệ file môi trường `.env`, cấm commit model weights (`.pt`, `.pth`, `.onnx`), cấm commit dataset lớn (`data/`, `dataset/`), bỏ qua `.venv` và `node_modules`.
- [x] Tạo bộ tài liệu hướng dẫn và đặc tả kỹ thuật:
  - `README.md`: Giới thiệu tổng quan và trạng thái dự án.
  - `docs/PROJECT_SPEC.md`: Đặc tả 6 nhóm rác, luồng nghiệp vụ và phân tách AI với Knowledge Base.
  - `docs/ARCHITECTURE.md`: Sơ đồ kiến trúc 4 tầng và luồng dữ liệu chi tiết.
  - `docs/TASKS.md`: Bảng phân chia công việc theo 3 nhánh tính năng.
  - `docs/PROGRESS.md`: Nhật ký tiến độ hiện tại.
  - `docs/DECISIONS.md`: Ghi nhận các quyết định công nghệ quan trọng (ADR).

---

## ⏳ Các bước tiếp theo cần triển khai (Next Steps)
1. Khởi tạo Git cục bộ (`git init -b main`), thêm remote `origin` và tạo commit đầu tiên.
2. Đẩy nhánh `main` lên GitHub repository `ai-waste-classification`.
3. Tạo 3 nhánh cục bộ từ `main`:
   - `feature/frontend`
   - `feature/ai-model`
   - `feature/backend-api`
4. Chờ phê duyệt trước khi thiết lập môi trường Python (`.venv`) và cài đặt PyTorch CUDA.
