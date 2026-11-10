# Thiết lập máy phát triển

Repo được clone tại `D:\ai-waste-classification`, nhánh `main`, commit nguồn
`00d9524`. Các thay đổi thiết lập nằm ở máy cục bộ, chưa commit hoặc push.

## Môi trường

- Python 3.11.9; môi trường riêng: `backend/.venv`.
- Node.js 22.23.2 và npm 10.9.8.
- GPU: NVIDIA GeForce RTX 4050 Laptop, khoảng 6 GB VRAM, driver 572.83.
- PyTorch được cài từ kênh CUDA 12.6 chính thức. Phiên bản thực tế và kiểm tra
  phép tính GPU được hiển thị bằng lệnh chẩn đoán bên dưới.

Không cần kích hoạt venv hoặc thay đổi ExecutionPolicy. Chạy từ thư mục gốc:

```powershell
Set-Location D:\ai-waste-classification
backend\.venv\Scripts\python.exe scripts\check_setup.py
```

Mã thoát: `0` = môi trường và checkpoint nạp được; `1` = lỗi thư viện/môi trường;
`2` = môi trường hoạt động nhưng chưa có checkpoint. Kiểm tra bằng tensor giả chỉ
xác nhận kỹ thuật nạp model, không đánh giá độ chính xác nhận diện.

## Cài lại môi trường

```powershell
py -3.11 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
npm.cmd --prefix frontend ci --no-audit --no-fund
```

NumPy đã được khai báo trực tiếp trong `backend/requirements.txt` vì ứng dụng
import nó. `requirements-dev.txt` bổ sung pytest và HTTPX cho kiểm thử.
Không tự cập nhật driver hoặc cài CUDA Toolkit toàn hệ thống.

## Checkpoint và dữ liệu

**Cập nhật sau Task 01:** không khôi phục MobileNet theo quyết định người dùng.
Phần MobileNet bên dưới chỉ mô tả luồng legacy. Readiness mới chọn riêng từng
capability; xem `docs/API_CONTRACT.md`. Health 200/degraded và exit 2 của script
kiểm legacy không đồng nghĩa model pin thử nghiệm không dùng được.

Git không chứa model `.pth`, dataset hoặc lịch sử SQLite. Cần lấy bản model đã
huấn luyện của nhóm và đặt vào `checkpoints/mobilenetv3_best.pth`. Không thay bằng
trọng số ImageNet thuần hoặc model khởi tạo ngẫu nhiên. Dataset chỉ cần cho huấn
luyện/đánh giá, không bắt buộc để mở giao diện và chạy API ngoài phần inference.

Mã huấn luyện và `class_mapping.json` hiện dùng thứ tự:
`plastic`, `paper`, `metal`, `glass`, `organic`, `other`, `medical`.
Repo còn `class_mapping_6class.json` của phiên bản trước. Backend đọc danh sách
`target_classes` từ checkpoint; cần đối chiếu đúng phiên bản khi khôi phục file cũ.

`test_evaluation.json` hiện ghi accuracy `0.9456`; đây là kết quả được lưu sẵn,
chưa được tái đo trên máy này. README mô tả 6 lớp và số liệu khác, nên cần cập nhật
tài liệu sau khi xác minh checkpoint và tập test tương ứng.

Mô hình phát hiện nhiều vật thể tự tải pretrained SSDLite khi gọi lần đầu.
Gatekeeper ImageNet tìm file cache tại
`%USERPROFILE%\.cache\torch\hub\checkpoints\mobilenet_v3_large-5c1a4163.pth`;
nếu thiếu thì dùng chế độ heuristic. Đây là các model phụ, không thay thế checkpoint
phân loại rác của nhóm.

## Chạy ứng dụng

### YOLO nhóm vật thể — Task 02A

Backend đã có `POST /api/yolo/detect` (multipart `file`) và
`GET /api/yolo/capabilities`. Kiểm readiness bằng
`GET /api/readiness?capability=yolo_detector`; health/cờ model cũ vẫn thuộc legacy.
Swagger ở cổng 8000 hiển thị schema `DetectionResponse` và các lỗi có code.
Frontend chưa đổi sang endpoint này.

Mặc định dùng run có sẵn `runs/waste_pilot/yolo26n_multisource30_v1`, manifest
mapping/hash tại `backend/app/yolo_detector_manifest.json`; không cần dataset
để inference, không tự download. Backend hiện đã có Ultralytics 8.4.208.
`YOLO_DETECTOR_RUN` chỉ đổi nơi chứa bản sao cùng artifact đã pin hash;
`YOLO_CONFIDENCE_THRESHOLD`/`YOLO_UNCERTAINTY_THRESHOLD` mặc định 0.25/0.50,
phải thỏa `0 < confidence <= uncertainty <= 1`. Chi tiết: `docs/API_CONTRACT.md`.
Box là pixel xyxy theo ảnh sau EXIF transpose; chín nhóm vật thể không phải
bảy vật liệu. Model vẫn experimental, không quyết định pin/medical/phóng xạ,
không auto ghi history hoặc xuất ảnh overlay. Task 02B UI chưa triển khai.

### Sàng lọc pin và hướng dẫn nguy cơ (thử nghiệm 10/11/2026)

Môi trường backend trên máy này đã được bổ sung dependency tùy chọn:

```powershell
backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-safety.txt
```

Chạy backend/frontend như bên dưới, mở tab **Pin và rác có nguy cơ**. API:

- `GET /api/safety/capabilities`: phân biệt có artifact và đã nạp model.
- `POST /api/safety/scan`: multipart `file`, ảnh một vật tối đa 10 MB; model pin
  thực nằm tại `runs/battery_screen/yolo26n_cls_multisource_v2/weights/best.pt`. Chỉ sàng lọc
  hình dạng; chưa hỗ trợ phát hiện nhiều pin/ẩn hoặc đánh giá tình trạng pin.
- `POST /api/safety/assess`: JSON `item_type`, `battery_condition`, `city`,
  `ordinary_material`; hướng dẫn dựa trên lựa chọn, không phải kết quả model.
- `GET /api/safety/collection-contacts?city=Đà Nẵng&material=battery`: điểm từng
  công bố thu gom, còn cần xác nhận hiện tại; không có điểm y tế/phóng xạ được xác minh.

`BATTERY_SCREEN_RUN` có thể trỏ tới một thư mục run khác nhưng phải có report
đánh giá và SHA-256 khớp model cùng contract hai nhãn `battery`, `not_battery`.
Không tự tải checkpoint khi chạy API; thiếu/sai artifact trả 503. Runtime pin chạy
CPU; health của classifier MobileNet cũ vẫn có thể `degraded` do thiếu `.pth`.
Thay đổi này không khôi phục hoặc thay thế checkpoint MobileNet.

Nội dung nghiên cứu, tiêu chí nghiệm thu và giới hạn: `docs/HAZARD_ROADMAP_DANANG.md`.
Registry điểm thu gom nằm ở `backend/app/collection_contacts.json`; mở rộng tỉnh
thành bằng dữ liệu có nguồn và trạng thái xác nhận, không thêm địa chỉ suy đoán.

Terminal thứ nhất, tại repo root:

```powershell
backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal thứ hai, tại repo root:

```powershell
npm.cmd --prefix frontend run dev -- --host 127.0.0.1
```

- Web: <http://127.0.0.1:5173>
- Swagger: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/api/health>

Health có thể trả HTTP 200 ngay cả khi thiếu checkpoint. Chỉ coi nhận diện sẵn sàng
khi `ai_model_loaded` và `checkpoint_loaded` đều `true`. Backend vẫn mở được các
API không phụ thuộc model; nhận diện ảnh sẽ báo chưa sẵn sàng khi model thiếu.

`run_app.bat` có sẵn mở terminal và cho phép truy cập LAN. Hai lệnh trên phù hợp
cho phát triển cục bộ. Nếu cần chỉnh URL API, dùng mẫu `frontend/.env.example`;
chế độ mặc định dùng Vite proxy `/api` nên không cần tạo file môi trường.

## Kiểm tra

Sau Task 01, nhóm offline và full suite được chạy từ root như sau:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/tests tests -m "not integration and not model" -q
backend\.venv\Scripts\python.exe -m pytest backend/tests -v
backend\.venv\Scripts\python.exe -m pytest backend/tests tests --run-model-tests -v
```

Integration tự mở API cổng tạm và database tạm, không cần server cổng 8000.
Model tests opt-in; 20 case legacy thiếu checkpoint vẫn skip, không khôi phục.
Hai benchmark standalone có import side effects được loại khỏi pytest discovery.
Kết quả Task 01: 129 PASS/20 SKIP/0 FAIL + 4 subtest (full gồm dataset tests);
offline 115 PASS. Xem `PROJECT_STATUS.md` và `logs/task01_final_suite.log`.

```powershell
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend run lint
backend\.venv\Scripts\python.exe -m pip check
```

Nhóm unit test không cần checkpoint hoặc API đang chạy:

```powershell
backend\.venv\Scripts\python.exe -m pytest -q --capture=sys backend/tests/test_confidence_prediction.py backend/tests/test_prediction_history.py::TestDatabaseUnit backend/tests/test_analytics_dashboard.py::TestAnalyticsUnit backend/tests/test_disposal_knowledge_base.py::TestDisposalKnowledgeBase backend/tests/test_litter_map.py
backend\.venv\Scripts\python.exe -m pytest -q --capture=sys backend/tests/test_image_quality.py -k test_unit
```

`--capture=sys` tương thích với các module test tự cấu hình stdout trên Windows.
Lệnh đầy đủ theo quy định dự án:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -v --capture=sys
```

Suite trước Task 01 từng trộn unit/API/benchmark. Marker, fixture model và server
test biệt lập hiện nằm trong `backend/tests/conftest.py`. Benchmark vẫn giữ code
nhưng không chạy khi collect. Không tạo model giả hoặc tính các case skip là pass.

## Làm việc với Codex

Mở thư mục `D:\ai-waste-classification` làm project để dùng `AGENTS.md` và skill
`.agents/skills/ai-waste-dev/SKILL.md`. Có thể gọi `$ai-waste-dev` cho công việc
liên quan dự án. Nếu skill chưa xuất hiện, mở lại project hoặc khởi động lại Codex.

Skill được đặt trong repo theo [tài liệu OpenAI](https://learn.chatgpt.com/docs/build-skills).
Kênh wheel CUDA theo [hướng dẫn PyTorch](https://docs.pytorch.org/get-started/previous-versions/).

## Kết quả kiểm tra sau thiết lập — lịch sử trước Task 01

- Build frontend thành công; lint trả mã 0 nhưng còn cảnh báo có sẵn trong UI.
- `pip check`: không có xung đột dependency.
- Skill `ai-waste-dev` vượt qua trình kiểm tra skill chính thức.
- CUDA 12.6, tensor GPU và torchvision NMS hoạt động trên RTX 4050.
- 29 unit test về confidence, database, analytics, knowledge base và map đã qua;
  thêm 8 unit test chất lượng ảnh đã qua: tổng 37.
- Smoke test với database tạm: health, Swagger, metrics, history, analytics,
  recycling centers và litter reports đều trả HTTP 200. Health báo `degraded`
  đúng với tình trạng chưa có checkpoint; chưa kiểm thử suy luận ảnh thực tế.
- Chạy suite đầy đủ với `--maxfail=3` dừng ở 3 lỗi kết nối API cổng 8000 vì không
  có server chạy ngoài tiến trình (5 test đã qua trước khi dừng). Smoke test sau
  đó kiểm tra được các route trong tiến trình qua TestClient. Chưa có kết luận
  suite đầy đủ đạt; các test model vẫn cần checkpoint và dữ liệu.
- Không khởi chạy server thường trực, không huấn luyện, không commit/push.

## Công cụ Teacher–Student — Task 03A, không train

Đọc `docs/TEACHER_STUDENT_PIPELINE.md`, `docs/TEACHER_RESEARCH.md` và
`docs/TASK03A_REPORT.md`. Pilot 150 ảnh train đã có, chưa Teacher/approved labels.
Không dựng lại workspace đó hoặc chạy lệnh train từ các phần lịch sử.

```powershell
backend\.venv\Scripts\python.exe -m scripts.teacher_student serve --workspace data/teacher_student/task03a_pilot150 --port 8013
backend\.venv\Scripts\python.exe -m pytest tests/test_teacher_student.py -q
node.exe --test tests/teacher_review_ui.test.mjs
```

Review chỉ loopback `http://127.0.0.1:8013`, dùng dependency hiện có. Không gắn
vào API sản phẩm 8000/Vite5173 hoặc public endpoint. Không có nút gọi cloud hoặc
train; Teacher output được import JSON có provenance, chưa duyệt thì không export.
Kiểm process/cổng trước khởi động, không dừng process lạ. Teacher/env/weights mới
cần được người dùng duyệt trước khi cài/tải; Task 03A chưa thực hiện bước đó.
