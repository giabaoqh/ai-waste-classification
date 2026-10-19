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
- [x] Khởi tạo Git cục bộ (`git init -b main`) và kết nối remote `origin` (`https://github.com/giabaoqh/ai-waste-classification.git`).
- [x] Tạo initial commit (`7bb1647`) và đẩy thành công nhánh `main` lên GitHub.
- [x] Tạo 3 nhánh tính năng cục bộ từ `main`:
  - `feature/frontend`
  - `feature/ai-model`
  - `feature/backend-api`
  *(Các nhánh tính năng được lưu cục bộ, chưa đẩy lên remote theo đúng yêu cầu)*.

---

## ⏳ Các bước tiếp theo cần triển khai (Next Steps)
1. Huy chấp nhận lời mời (accept invitation) để có quyền truy cập repo trên GitHub.
2. Chờ phê duyệt của nhóm trước khi tiến hành Phase tiếp theo:
   - Tạo môi trường ảo Python (`.venv`) bằng Python 3.13.16.
   - Cài đặt PyTorch với hỗ trợ NVIDIA CUDA (RTX 3050 Laptop GPU).

