# Task 01 — Chuẩn hóa API Contract & Backend Tests

Hoàn thành ngày **10/11/2026** trên working tree `D:\ai-waste-classification`.
Không train, download, cài lại môi trường, khôi phục MobileNet, xóa code cũ,
migration UI hoặc commit/push. Giữ nguyên các thay đổi local không thuộc task.

## Đã sửa

- Health giữ HTTP 200/liveness và các trường legacy, thêm trạng thái riêng cho
  legacy classifier, YOLO detector và battery classifier. Legacy vẫn degraded;
  YOLO chưa tích hợp không bị báo ready chỉ vì có file weights.
- Thêm readiness theo capability; xác minh artifact pin theo hash/threshold/nhãn/
  cờ cho phép runtime, cache gắn đúng artifact hiện tại, phản ánh lỗi loader.
- Decoder chung kiểm bytes, nội dung ảnh, EXIF và giới hạn 20 MP; đọc file tối đa
  10 MB + 1 byte. Chất lượng/ảnh lỗi được kiểm trước readiness, không bị che bằng
  503 thiếu legacy model. URL detector không còn đổi lỗi 413 thành 400.
- Batch giữ envelope tương thích, thêm HTTP status từng item.
- Sửa ba case cũ: health dựa trạng thái model thật; ảnh hỏng trả 400; analytics
  assertion kiểm ý nghĩa disclaimer thay vì đòi đúng chữ hoa. Cả ba được chạy và PASS.
- Tách offline/integration/model bằng marker. HTTP tests dùng API/database tạm;
  model tests opt-in, không tạo weights giả. Bổ sung fixture còn thiếu; giữ hai
  benchmark standalone ngoài pytest collection để tránh import tải mạng/model.
- Regression không cần weights bao phủ readiness, request validation, URL/batch,
  cache khác đường dẫn. Opt-in kiểm model pin thật, report sai/hash/loader lỗi.

## File thay đổi

| Nhóm | File |
|---|---|
| Runtime sửa | `backend/app/main.py`, `backend/app/battery_screen.py`, `backend/app/safety_routes.py` |
| Runtime mới | `backend/app/image_input.py`, `backend/app/model_status.py` |
| Test/config mới | `pytest.ini`, `backend/tests/conftest.py`, `backend/tests/test_api_contract.py` |
| Test sửa | `backend/tests/test_api_integration.py`, `backend/tests/test_analytics_dashboard.py`, `backend/tests/test_safety_api.py` |
| Tài liệu | `PROJECT_STATUS.md`, `CHATGPT_HANDOVER.md`, `AGENTS.md`, `docs/LOCAL_SETUP.md`, `docs/API_CONTRACT.md`, báo cáo này |

Các file frontend, dataset, checkpoint và thay đổi từ các task trước được giữ
nguyên. Snapshot một số file trước task nằm ở `logs/task01_before/` để phân biệt
diff task này với working tree đã có sẵn; không lấy toàn bộ Git diff làm diff Task 01.

## Test PASS / FAIL / SKIP

| Lệnh / kiểm tra | Kết quả |
|---|---|
| `backend\.venv\Scripts\python.exe -m pytest backend/tests tests --run-model-tests -v --tb=short` | **129 PASS / 0 FAIL / 20 SKIP**, thêm 4 subtest PASS; 3 warnings |
| `backend\.venv\Scripts\python.exe -m pytest backend/tests tests -m "not integration and not model" -q` | **115 PASS / 0 FAIL**, 34 deselected; thêm 4 subtest PASS; 3 warnings |
| Ba test lỗi ban đầu | Đều PASS; không được skip |
| API pin thật sau restart local | 4 ảnh Kaggle/RecyBat và filename invariance PASS |
| Live health/readiness/request-error | PASS: health 200/degraded; legacy 503; YOLO 503; pin 200; ảnh hỏng 400 |
| Hash checkpoint pin | Không đổi, khớp report |
| `git diff --check` | PASS |

20 SKIP là các test inference legacy thiếu checkpoint, không phục hồi MobileNet
để làm chúng chạy. 34 deselected là nhóm không thuộc offline; không cộng các nhóm
test chồng nhau. Không tính hai script benchmark thành pass. 3 warnings gồm
Starlette/httpx và hai lần dùng tên hằng HTTP 413 cũ; không nâng dependency.
Không sửa frontend nên không chạy lại build/lint trong task này; baseline audit
trước đó build qua và lint còn 45 cảnh báo.

Bằng chứng: `logs/task01_final_suite.log`, `task01_offline_units.log`,
`task01_live_contract.json`, `task01_real_battery_api.log`. Toàn bộ case thu thập
của full superset đã kết thúc, không dùng `--maxfail` để dừng sớm.

## API contract sau sửa

- `GET /api/health`: 200 cho service sống; các cờ model cũ thuộc legacy. Thêm
  `contract_version=1.0`, `service_alive` và `capabilities` cho ba model.
- `GET /api/readiness?capability=legacy_classifier|yolo_detector|battery_classifier`:
  200/503 theo `runtime_ready`; thiếu/sai selector 422.
- Pin phân biệt `unavailable`, `invalid_artifact`, `blocked`,
  `experimental_not_loaded`, `experimental_loaded`, `load_failed`. Lazy readiness
  là đủ điều kiện thử inference, không phải probe accuracy; GET không nạp model.
- `production_ready=false` cho các model hiện tại, không biến pin thử nghiệm
  thành detector rác hoặc chứng nhận an toàn.
- Upload: lỗi nội dung/chất lượng 400; quá byte/pixel limit 413; schema 422;
  ảnh hợp lệ nhưng model chưa sẵn sàng 503. Batch envelope vẫn 200 và có
  `items[].status_code`. Safety scan đổi lỗi ảnh từ 422 sang 400 có chủ đích.

Contract đầy đủ và ví dụ: `docs/API_CONTRACT.md`.

## Vấn đề còn tồn tại

- YOLO detector chưa tích hợp endpoint/UI; chất lượng detector vẫn thấp.
- 20 test legacy chưa kiểm định inference; không có task khôi phục MobileNet.
- Pin vẫn là classifier ảnh một vật, thiếu kiểm định độc lập Đà Nẵng; điểm số
  mô hình giữ nguyên, không có phép đo accuracy mới trong Task 01.
- SSRF/private IP/redirect, auth/phân quyền, provenance bản đồ/vote và CI/frontend
  tests chưa được giải quyết trong task này. Không coi full suite pass là đã an
  toàn để triển khai công khai.
- README/model metrics legacy còn cần xử lý khi migration; không làm lại model
  hoặc thay taxonomy để hợp thức hóa số liệu cũ.

## Đề xuất task tiếp theo

**Task 02 — tích hợp luồng YOLO theo contract mới**, sau khi người dùng giao:
chốt output chín nhóm, adapter/model version/quality/uncertainty và test endpoint;
chuyển UI có kiểm soát theo code thực tế. Không làm lại Task 01, không khôi phục
MobileNet, không tuyên bố model pin hoặc detector đã được nghiệm thu.
