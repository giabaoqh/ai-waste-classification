# Architecture & Technical Decisions (DECISIONS.md)

Tài liệu này lưu trữ các Quyết định Kiến trúc & Kỹ thuật quan trọng (Architecture Decision Records - ADR) của dự án.

---

### ADR-001: Chiến lược phân nhánh Git theo tính năng (Feature-based Branching)
- **Bối cảnh**: Nhóm có 3 thành viên (Bảo, Huy, Minh) luân phiên sử dụng chung một máy tính Windows.
- **Quyết định**: Sử dụng cấu trúc nhánh theo tính năng thay vì đặt tên nhánh theo tên cá nhân:
  - `main`: Nhánh tích hợp mã nguồn ổn định.
  - `feature/frontend`: Giao diện React + Vite.
  - `feature/ai-model`: Mô hình PyTorch và xử lý dữ liệu.
  - `feature/backend-api`: FastAPI backend và Knowledge Base.
- **Lý do**:
  - Tránh nhầm lẫn khi một thành viên hỗ trợ sang mảng khác.
  - Phù hợp với chuẩn phát triển phần mềm chuyên nghiệp (Git Flow / GitHub Flow rút gọn).
  - Giữ lịch sử commit của từng mảng tính năng độc lập, dễ dàng code review và merge.

---

### ADR-002: Lựa chọn phiên bản Python 3.13 cho AI & Backend
- **Bối cảnh**: Máy tính đã cài đặt sẵn cả Python 3.13.16 và Python 3.14.4.
- **Quyết định**: Sử dụng Python 3.13.16 để tạo môi trường ảo `.venv` cho backend và AI.
- **Lý do**:
  - PyTorch chính thức đã phát hành bản pre-built hỗ trợ CUDA cho Python 3.13.
  - Python 3.14 là phiên bản quá mới, nhiều thư viện khoa học dữ liệu (Torchvision, OpenCV, Scikit-learn...) chưa có bản binary wheel ổn định trên Windows.

---

### ADR-003: Phân tách logic nhận diện AI và Cơ sở tri thức tái chế
- **Bối cảnh**: Hệ thống vừa phải nhận diện loại rác vừa phải đưa ra hướng dẫn phân loại và tái chế theo quy chuẩn môi trường.
- **Quyết định**: Mô hình AI chỉ dự đoán nhãn rác (ví dụ: `plastic`, `paper`, `glass`...) kèm điểm tin cậy. Toàn bộ logic hướng dẫn phân loại, làm sạch và mẹo tái chế được quản lý trong Knowledge Base riêng (dưới dạng JSON/Database).
- **Lý do**:
  - Quy chuẩn phân loại rác có thể thay đổi hoặc mở rộng theo từng địa phương mà không đòi hỏi phải huấn luyện lại mạng nơ-ron (tiết kiệm thời gian và tài nguyên tính toán).
  - Tách bạch trách nhiệm (Separation of Concerns): AI lo thị giác máy tính, Backend lo tra cứu tri thức và nghiệp vụ.

---

### ADR-004: Quản lý Dataset và Model Weights bên ngoài Git
- **Bối cảnh**: Repository dự án là Public trên GitHub. Giới hạn kích thước file của GitHub là 100MB và repo công khai không nên chứa dữ liệu thô quá nặng.
- **Quyết định**: Đưa toàn bộ file trọng số (`*.pt`, `*.pth`, `*.onnx`) và thư mục dữ liệu (`data/`, `dataset/`) vào `.gitignore`.
- **Lý do**:
  - Giữ cho repository nhẹ, thời gian clone/pull nhanh chóng.
  - Trọng số mô hình và dataset sẽ được chia sẻ nội bộ qua Google Drive hoặc lưu trữ đám mây chuyên dụng khi cần.

---

### ADR-005: Quản lý xác thực và danh tính tác giả trên máy dùng chung
- **Bối cảnh**: 3 thành viên dùng chung máy tính nhưng cần commit đúng tên mình và push bằng tài khoản GitHub cá nhân.
- **Quyết định**:
  - Không sửa `git config --global` bừa bãi. Khi đổi ca, thành viên cấu hình tên tác giả cục bộ trong repo:
    `git config user.name "Tên Tác Giả"` và `git config user.email "email@example.com"`.
  - Phân định rõ: `user.name` chỉ là chữ ký commit; quyền push lên GitHub dựa trên tài khoản đã được mời làm Collaborator trong repository settings.
