# AI-Based Waste Recognition and Recycling Recommendation System

> **Capstone Project**: Hệ thống nhận diện rác thải và gợi ý tái chế dựa trên Trí tuệ Nhân tạo.

---

## 📌 Giới thiệu dự án (Project Overview)

Dự án nhằm mục tiêu phát triển hệ thống hỗ trợ người dùng phân loại rác thải sinh hoạt thông qua hình ảnh. Hệ thống sẽ nhận diện loại rác, đánh giá độ tin cậy (confidence score) và đưa ra các hướng dẫn xử lý hoặc tái chế phù hợp theo tiêu chuẩn bảo vệ môi trường.

### Nhóm danh mục rác dự kiến phân loại (Planned Categories):
1. **Plastic** (Nhựa)
2. **Paper** (Giấy)
3. **Metal** (Kim loại)
4. **Glass** (Thủy tinh)
5. **Organic** (Rác hữu cơ)
6. **Other** (Rác thải khác / rác nguy hại)

---

## 👥 Thành viên nhóm (Team Members)

| Thành viên | Vai trò dự kiến | Nhánh làm việc chính |
| :--- | :--- | :--- |
| **Bảo** (Leader) | Phụ trách chung / Tích hợp | `feature/bao` hoặc tính năng được phân công |
| **Huy** | Thành viên phát triển | `feature/...` theo phân công |
| **Minh** | Thành viên phát triển | `feature/...` theo phân công |

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

- [x] Khởi tạo cấu trúc repository và tài liệu thiết kế.
- [ ] Thiết lập môi trường ảo Python (`.venv`) và PyTorch với hỗ trợ GPU CUDA *(Chưa triển khai)*.
- [ ] Xây dựng khung giao diện React + Vite *(Chưa triển khai)*.
- [ ] Xây dựng khung API FastAPI *(Chưa triển khai)*.
- [ ] Thu thập dữ liệu và huấn luyện mô hình phân loại rác *(Chưa triển khai)*.

---

## 📖 Tài liệu dự án (Documentation)

Chi tiết thiết kế và kế hoạch được lưu trữ trong thư mục `docs/`:
- [PROJECT_SPEC.md](docs/PROJECT_SPEC.md): Yêu cầu kỹ thuật và phạm vi dự án.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): Kiến trúc hệ thống và luồng dữ liệu.
- [TASKS.md](docs/TASKS.md): Danh sách công việc và phân công nhiệm vụ.
- [PROGRESS.md](docs/PROGRESS.md): Nhật ký tiến độ thực hiện.
- [DECISIONS.md](docs/DECISIONS.md): Các quyết định kiến trúc và công nghệ (ADR).
