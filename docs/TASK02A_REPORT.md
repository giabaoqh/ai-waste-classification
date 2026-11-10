# Task 02A — YOLO26n Multisource Backend

Hoàn thành ngày **10/11/2026**. Repo `D:\ai-waste-classification`, working tree
có sẵn nhiều thay đổi local. Task 01 đã hoàn thành, không triển khai lại nó.

## Công việc đã triển khai

- Kiểm tra artifact run `yolo26n_multisource30_v1`: checkpoint task detect,
  head nc=9, names đúng thứ tự; đối chiếu metadata per_class và dataset YAML
  đúng SHA-256 training, cùng taxonomy script. Không suy đoán nhãn từ legacy.
- Lưu manifest đã xác minh: architecture, run/version, checksum checkpoint và
  metadata, training library version, input size, mapping ID/name/label_vi.
- Adapter chỉ dùng file local, xác minh trước nạp, CPU/lazy-load/lock, cache theo
  artifact hiện tại. Nạp thật kiểm lại names/head/task/version. Tắt integrations
  sync, không train hoặc dùng fallback auto-download khi thiếu file.
- Endpoint detection/capabilities riêng; readiness/health lấy runtime thực của
  adapter. Không thay endpoint legacy/pin hoặc frontend.
- Dùng nguyên validation ảnh Task 01; error 400/413/422 trước model. Model thiếu,
  artifact sai, config/dependency/load lỗi trả 503; forward/output lỗi trả 500.
- Bbox/scores được kiểm finite/range/class ID/geometry; clip theo ảnh sau EXIF,
  không tạo box hay detection để thay lỗi. Không có detection vẫn bất định.
- Có cấu hình filter/review threshold, unit/API/fault tests và model smoke thật.

## Artifact và mapping

- Weights: `runs/waste_pilot/yolo26n_multisource30_v1/weights/best.pt`.
- SHA-256: `dc8f481be1668c960b78264db70c5f48c44d42b8fcb2fab93d23625532225efc`.
- Metadata: `pilot_summary.json`, SHA-256
  `4b8aa49fcc845ce6432a1b188b7c767ee99e3daaf81d84c2ad3f053f5c556b9e`.
- Checkpoint và backend runtime Ultralytics: **8.4.208**; imgsz 640.
- Model version trả về: `yolo26n_multisource30_v1@dc8f481be166`.
- Mapping đúng thứ tự ID 0..8: **bottle_jar, can, cup, carton, cap_lid,
  bag_film_wrapper, paper, cigarette, other_litter**.

Đây là taxonomy nhóm vật thể, không phải bảy vật liệu cũ. Không có nhãn pin,
medical hoặc phóng xạ. Model weights/run metadata nguồn giữ nguyên hash sau task.

## Endpoint và response schema

**POST `/api/yolo/detect`**, multipart field `file`, dùng file JPEG/PNG/WebP,
tối đa 10 MB và 20 megapixel, kiểm chất lượng như Task 01. Không có URL/batch.

Response 200 có:

- `success`, `inference_status=completed`, `detection_status`, `total_objects`.
- `detections[]`: class_id, class_name, label_vi, confidence, bounding_box,
  is_uncertain, uncertainty_note.
- Box **xyxy pixel**, kích thước image_width/image_height trên ảnh sau EXIF
  transpose; schema ghi rõ bounding_box_format và coordinate_space.
- `model`: ID, architecture, version/hash, task/taxonomy, library versions,
  input_size và mapping đủ chín nhóm đã xác minh.
- Quality, device CPU, adapter processing time, thresholds, thông tin bất định.
- `experimental=true`, `production_ready=false`, global `is_uncertain=true`,
  `hazard_assessment=not_assessed`, `confirmed_safe=false`, hướng dẫn tiếng Việt.

Không ghi DB/history, không tạo overlay/base64/URL ảnh, không trả khuyến nghị
thùng rác từ nhãn vật thể. Confidence không phải xác suất đã hiệu chỉnh.
5xx có `detail.code/status/capability/inference_status/message`; 503
`inference_status=not_started`, 500 `inference_status=failed`.

**GET `/api/yolo/capabilities`** trả trạng thái, readiness, artifact verification,
model info và ngưỡng. **GET `/api/readiness?capability=yolo_detector`** dùng chung
contract Task 01 (200/503). Tài liệu đầy đủ: `docs/API_CONTRACT.md`.

## Model readiness

Trạng thái: unavailable, invalid_artifact, invalid_configuration,
experimental_not_loaded, experimental_loaded, load_failed, inference_error.
GET không nạp model hay probe accuracy; chỉ hai trạng thái experimental cho
phép thử runtime. Model đã nạp chỉ khi cache khớp artifact hiện tại. Lỗi runtime
có thể retry, hash/mapping/config sai phải sửa đúng. Production luôn false.

Live sau restart API local: YOLO **experimental_loaded**, readiness **200**;
legacy vẫn **503**, health **200/degraded** đúng contract cũ. Pin vẫn độc lập.

Biến môi trường: `YOLO_DETECTOR_RUN` (bản sao cùng artifact),
`YOLO_CONFIDENCE_THRESHOLD` mặc định **0.25**,
`YOLO_UNCERTAINTY_THRESHOLD` mặc định **0.50**, ràng buộc
`0 < confidence <= uncertainty <= 1`. Không tuyên bố đây là ngưỡng an toàn/accuracy.

## Test PASS / FAIL / SKIP

| Nhóm | Kết quả |
|---|---|
| Full backend + dataset, `--run-model-tests -v --tb=short` | **172 PASS / 0 FAIL / 20 SKIP**, 4 subtest PASS, 3 warnings |
| Offline `not integration and not model` | **154 PASS**, 38 deselected, 4 subtest PASS |
| Task 01 regression | Health/readiness, invalid image, analytics disclaimer, URL/batch và pin vẫn PASS |
| YOLO cases | Mapping/head/version, missing/invalid artifact, thresholds, malformed outputs, empty/low-score, EXIF box, cache switch, load/inference failure/retry PASS |
| Real-model opt-in | YOLO checkpoint thật và hash/metadata tampering trên bản sao test PASS |
| Live API smoke | 2 ảnh val có sẵn: WaRP trả empty, HF TACO trả 1 detection; schema/box/readiness và ảnh hỏng 400 PASS |
| Bảo toàn | Hash frontend, image_input, battery/safety code, conftest Task 01 và source weights/metadata không đổi |
| `git diff --check` | PASS |

20 SKIP vẫn do checkpoint legacy thiếu; không khôi phục MobileNet. Không cộng
nhóm test chồng nhau, không tính fixture detections thành kết quả model. Test
contract dùng fixture có ghi rõ; các score/box live trong log đến từ inference thật.
Smoke không phải phép đo accuracy. Chỉ kiểm inference, không đánh giá lại toàn bộ
dataset. Frontend không đổi nên không chạy lại build/lint. 3 warnings dependency/
tên hằng 413 đã có từ Task 01; không nâng/cài môi trường.

Bằng chứng: `logs/task02a_artifact_inspection.log`, `task02a_full_suite.log`,
`task02a_offline_units.log`, `task02a_live_contract.json`.

## Các file thay đổi riêng Task 02A

- Mới: `backend/app/yolo_detector.py`, `yolo_routes.py`,
  `yolo_detector_manifest.json`, `backend/tests/test_yolo_detector.py`.
- Sửa: `backend/app/main.py` (đăng ký router), `model_status.py` (YOLO readiness).
- Regression: `backend/tests/test_api_contract.py`, `test_api_integration.py`
  bỏ giả định YOLO chưa tích hợp, thêm endpoint vào validation tests hiện có.
- Tài liệu: `PROJECT_STATUS.md`, `CHATGPT_HANDOVER.md`, `AGENTS.md`,
  `docs/API_CONTRACT.md`, `docs/LOCAL_SETUP.md`, báo cáo này.

Snapshot `logs/task02a_before/` phân biệt task này với các thay đổi có sẵn.
Không train/download/xóa checkpoint hoặc dataset, không sửa frontend, không
khôi phục MobileNet, không commit/push hoặc tạo ảnh AI.

## Vấn đề còn tồn tại

- Model còn yếu: kết quả lượt train trước mAP50 validation hỗn hợp khoảng
  26,04%, recall 27,87%; chưa có bộ test Đà Nẵng. Task này không cải thiện weights.
- Không được suy ra material, pin, medical, phóng xạ hoặc điều kiện an toàn từ
  output chín nhóm. `other_litter` có thể bao gồm vật nguy hại.
- Chưa chuyển UI, chưa thêm URL/batch/history/feedback riêng cho YOLO.
- Legacy routes còn nguyên và unavailable; metrics/dashboard cũ không áp cho
  detector này. SSRF/auth/provenance bản đồ và CI vẫn ngoài phạm vi.

## Có đủ điều kiện chuyển Task 02B không?

**Có, ở mức kỹ thuật cho UI thử nghiệm có giám sát.** Task 02B nên đọc readiness,
gọi endpoint mới, render đúng mapping/bbox/EXIF, hiển thị experimental/uncertainty
và xử lý 4xx/5xx. Không giao lại adapter/contract/mapping đã hoàn thành; không
auto gán chín nhóm thành vật liệu hoặc dùng metrics legacy cho model mới.

**Chưa đủ điều kiện triển khai công khai hoặc tự động phân loại/xử lý rác.**
Test pass chứng minh contract/runtime, không chứng minh model đủ chính xác.
