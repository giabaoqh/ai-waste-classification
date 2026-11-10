# API contract — Task 01 + Task 02A + frontend Task 02B, ngày 10/11/2026

Contract trạng thái version **1.0**, bổ sung trên API 0.4.0. Không train, tải dữ
liệu, khôi phục MobileNet hoặc chuyển UI sang YOLO trong hai task này.
Task 02A bổ sung endpoint YOLO riêng và cập nhật readiness; không thay endpoint legacy.
Task 02B dùng contract này cho UI chính; không đổi API/schema backend.

## Health và readiness

`GET /api/health` trả **200** khi service trả lời được. Giữ các trường cũ:
`status`, `ai_model_loaded`, `checkpoint_loaded`, `model_architecture`,
`target_classes`, `uncertainty_threshold`, `error`.

**Các trường model cũ chỉ nói về legacy MobileNet**. `status=degraded` khi legacy
chưa sẵn sàng; đây không có nghĩa classifier pin cũng không sẵn sàng. Không đổi
thành `ok` giả để test pass. Trường mới:

- `contract_version="1.0"`, `service_alive=true`.
- `capabilities.legacy_classifier`: `legacy_unavailable` hoặc `legacy_loaded`;
  có code tích hợp legacy nhưng không có task khôi phục model này.
- `capabilities.yolo_detector`: từ Task 02A đã tích hợp endpoint thử nghiệm;
  `experimental_not_loaded`/`experimental_loaded` khi artifact hợp lệ, các trạng
  thái unavailable/invalid/load/inference error khi không sẵn sàng. Không coi
  artifact tồn tại là model đã nạp hoặc đủ chất lượng triển khai.
- `capabilities.battery_classifier`: trạng thái classifier pin đang cấu hình.
- Mỗi capability có `model_loaded`, `runtime_ready`, `production_ready`,
  `experimental`, `integrated`, `artifact_files_present`, `artifact_verified`, `reason`.

`GET /api/readiness?capability=...` yêu cầu chọn đúng một trong ba capability
trên. Trả **200** nếu `runtime_ready=true`, **503** nếu chưa sẵn sàng; payload:

```json
{
  "contract_version": "1.0",
  "capability": "yolo_detector",
  "ready": true,
  "model": {
    "status": "experimental_not_loaded",
    "model_loaded": false,
    "runtime_ready": true,
    "production_ready": false,
    "experimental": true,
    "integrated": true,
    "artifact_files_present": true,
    "artifact_verified": true,
    "reason": "YOLO nhóm vật thể đã tích hợp để thử nghiệm; chưa đủ điều kiện triển khai không giám sát."
  }
}
```

`artifact_files_present` tùy máy; ví dụ có file không chứng minh model đã nạp.
Thiếu/sai tên capability trả **422**, không tự chọn một model để báo ready.
Readiness ở đây có phạm vi **suy luận của từng capability**, không phải kiểm định
chất lượng model, tính sẵn sàng toàn bộ database hay chứng nhận triển khai.
`production_ready=false` cho các model hiện tại.

Ví dụ trên ứng với artifact YOLO đã được xác minh ở máy hiện tại, trả HTTP 200
cho thử inference. Máy thiếu artifact sẽ trả 503, không tự tải model.

## YOLO detector — mới ở Task 02A

`POST /api/yolo/detect`, multipart field **file**, endpoint riêng; không đổi
`/api/predict`, `/api/detect` hoặc `/api/safety/scan`. `GET /api/yolo/capabilities`
trả capability chi tiết, model identity/mapping và ngưỡng hiện tại.

Artifact mặc định: `runs/waste_pilot/yolo26n_multisource30_v1/weights/best.pt`
và `pilot_summary.json`; manifest đi kèm code:
`backend/app/yolo_detector_manifest.json`. Không cần dataset để phục vụ inference.
`YOLO_DETECTOR_RUN` cho phép đổi thư mục chứa **bản sao cùng artifact đã xác minh**;
weights hoặc metadata khác hash sẽ bị từ chối. Không tự chọn checkpoint khác.

Mapping đã đối chiếu checkpoint names/head nc=9, per_class metadata, dataset YAML
khớp hash training và taxonomy script. Không suy ra từ bảy nhãn vật liệu cũ:

| class_id | class_name |
|---:|---|
| 0 | bottle_jar |
| 1 | can |
| 2 | cup |
| 3 | carton |
| 4 | cap_lid |
| 5 | bag_film_wrapper |
| 6 | paper |
| 7 | cigarette |
| 8 | other_litter |

Model: YOLO26n, task detect, run version `yolo26n_multisource30_v1`, weights SHA-256
`dc8f481be1668c960b78264db70c5f48c44d42b8fcb2fab93d23625532225efc`.
Checkpoint và runtime Ultralytics đã xác minh là **8.4.208**; khác phiên bản runtime
thì báo unavailable, không tự cài lại. Run metadata cũng được pin hash trong manifest.
Khi nạp thật còn kiểm task/names/head/version; không chỉ dựa vào report JSON.

Response 200 được định nghĩa bởi `DetectionResponse` trong OpenAPI:

| Trường | Ý nghĩa |
|---|---|
| `success`, `inference_status` | true, completed — forward pass hoàn tất |
| `detection_status` | detections_found hoặc no_detections_above_threshold |
| `detections[]` | class_id, class_name, label_vi, confidence, bounding_box, is_uncertain, uncertainty_note |
| `bounding_box_format` | xyxy_pixels: [x1,y1,x2,y2], clip trong kích thước ảnh, không box suy đoán |
| `coordinate_space`, `image_width`, `image_height` | tọa độ pixel trên ảnh **sau EXIF transpose**, dùng kích thước này khi vẽ UI |
| `model` | model_id, architecture, model_version dạng run@hash, task, taxonomy, checksum, versions, input_size, class_mapping 9 nhãn |
| `quality_assessment` | kết quả validator model-independent của Task 01 |
| `confidence_threshold`, `uncertainty_threshold` | ngưỡng đã dùng cho request |
| `device`, `processing_time_ms` | CPU; thời gian adapter gồm kiểm artifact/chờ lock/nạp/inference, không gồm upload/decoder |
| `experimental`, `production_ready` | true / false |
| `is_uncertain`, `uncertainty_note` | toàn response luôn bất định do model còn thử nghiệm; detection riêng dưới ngưỡng review có cờ/note riêng |
| `hazard_assessment`, `confirmed_safe` | not_assessed / false; không xác định pin, medical hay phóng xạ |
| `label_vi`, `guidance`, `tips` | giải thích tiếng Việt; không tạo hướng dẫn bỏ thùng theo nhóm vật thể |

`confidence` là score model, `confidence_is_calibrated_probability=false`.
Không dùng score để tuyên bố accuracy hoặc ngưỡng an toàn. Không có detection
vẫn là inference completed nhưng không khẳng định không có rác hoặc vật nguy hại.
Không ghi ảnh/prediction vào DB/history, không xuất annotated image/base64 trong
endpoint này. Task 02B đã render box phía client, không tự ghép taxonomy cũ.

Cấu hình môi trường (không override bằng input của người upload):

- `YOLO_CONFIDENCE_THRESHOLD`: mặc định **0.25**, lọc detection.
- `YOLO_UNCERTAINTY_THRESHOLD`: mặc định **0.50**, đánh dấu score cần review.
- Phải hữu hạn và `0 < confidence <= uncertainty <= 1`. Cấu hình sai báo 503
  `invalid_configuration`, không âm thầm dùng lại default. Các ngưỡng là lựa chọn
  kỹ thuật thử nghiệm, chưa hiệu chỉnh/xác nhận trên dữ liệu Đà Nẵng.
- Input size 640 theo run; CPU, max_det=100; không train/download khi inference.

Trạng thái YOLO: unavailable, invalid_artifact, invalid_configuration,
experimental_not_loaded, experimental_loaded, load_failed, inference_error.
`runtime_ready` true chỉ ở hai trạng thái experimental; `model_loaded` gắn với
artifact đang cấu hình. GET chỉ kiểm artifact/config/dependency, không nạp model.
Lỗi runtime có thể retry bằng request mới; GET phản ánh lỗi đến khi retry thành
công hoặc artifact đổi. Mapping/weights/config sai phải được sửa đúng trước.

HTTP 400/413/422 theo Task 01 được kiểm **trước** model readiness. HTTP 503 cho
model/artifact/config/load unavailable; HTTP 500 cho forward pass/output lỗi,
không có detection giả. Hai lỗi 5xx có schema `YoloErrorResponse`:

```text
detail: {
  code: yolo_model_unavailable | yolo_invalid_artifact | yolo_invalid_configuration |
        yolo_dependency_unavailable | yolo_load_failed | yolo_inference_error,
  status: trạng thái tương ứng,
  capability: yolo_detector,
  inference_status: not_started | failed,
  message: giải thích tiếng Việt
}
```

Chưa thêm URL/batch/ghi history/feedback cho YOLO; không dùng endpoint legacy để
ngầm chuyển sang taxonomy chín nhóm. Chất lượng detector còn thấp và test pass
chỉ chứng minh hành vi phần mềm/contract, không chứng nhận triển khai model.

## Battery capabilities

`GET /api/safety/capabilities` giữ các trường trước đây, thêm
`artifact_verified`, `runtime_ready`, `production_ready`, `experimental`,
`readiness_status`, `readiness_reason`.

| Trạng thái | Ý nghĩa | Runtime ready |
|---|---|---|
| `unavailable` | Thiếu weights/report hoặc thư viện suy luận | false |
| `invalid_artifact` | JSON/threshold/nhãn/hash không hợp lệ | false |
| `blocked` | Report chưa cho phép `runtime_enabled=true` | false |
| `experimental_not_loaded` | Xác minh file/report/nhãn/hash/dependency, nạp lười khi scan | true |
| `experimental_loaded` | Đã nạp và suy luận thành công với đúng artifact đang cấu hình | true |
| `load_failed` | Lần nạp/suy luận hiện tại thất bại | false |

`runtime_ready=true` cho pin nghĩa là đủ điều kiện thử suy luận, không phải probe
inference/accuracy khi gọi GET. Lần scan thật vẫn có thể trả 503; sau lỗi đó
readiness phản ánh `load_failed`. GET không import/nạp model hay tải weights.
Scan kế tiếp có thể thử lại; đổi artifact/report sẽ tạo danh tính cache mới.

`ai_model_loaded` gắn với **đường dẫn + mtime weights/report hiện tại**, không dựa
vào việc cache từng chứa một model khác. Hash weights phải khớp report, nhãn đúng
`battery`, `not_battery`. Không suy ra an toàn từ một kết quả âm tính.

## Thứ tự validation và HTTP status

File được đọc tối đa 10 MB + 1 byte để phát hiện quá giới hạn; đóng UploadFile
sau khi đọc. Kiểm bytes → định dạng/nội dung ảnh → giới hạn điểm ảnh/chất lượng
→ readiness model → inference. MIME/đuôi tên tệp không thay thế kiểm nội dung.

| Trường hợp | HTTP |
|---|---|
| Thiếu field bắt buộc, schema không hợp lệ | 422 |
| File rỗng/hỏng, định dạng không hỗ trợ, kích thước nhỏ/ánh sáng bị từ chối | 400 |
| Vượt 10 MB hoặc vượt 20 megapixel | 413 |
| Ảnh hợp lệ nhưng legacy/model pin chưa sẵn sàng | 503 |
| Ảnh hợp lệ, inference thành công | 200 |

Áp dụng cùng decoder cho `/api/predict`, `/api/detect`, `/api/safety/scan`.
Các URL routes kiểm URL, giới hạn tải và ảnh trước readiness; giữ nguyên 413 thay
vì bắt `HTTPException` rồi đổi thành 400. Lỗi tải URL trả 400 như trước.
SSRF/private IP/redirect hardening vẫn là việc chưa làm, không được coi task này
đã giải quyết bảo mật URL.

Batch giữ envelope **200** và `successful/failed/items` như trước, thêm
`items[].status_code` (200/400/413/503/500) để không che lỗi từng ảnh. Không đổi
toàn bộ batch thành 503 nếu có một item thiếu model.

Thay đổi status có chủ đích: safety scan trước trả 422 cho file/chất lượng sai,
nay trả **400** thống nhất với upload khác; 422 dành cho schema. Frontend hiện
đọc lỗi mọi HTTP non-2xx nên không cần migration UI. Legacy payload thành công
và các trường uncertainty/quality/hướng dẫn được giữ.

## Test và phạm vi được xác minh

Chạy từ root bằng backend venv; không cần API cổng 8000 hay GPU cho nhóm offline:

```powershell
backend\.venv\Scripts\python.exe -m pytest backend/tests tests -m "not integration and not model" -q
backend\.venv\Scripts\python.exe -m pytest backend/tests -v
backend\.venv\Scripts\python.exe -m pytest backend/tests tests --run-model-tests -v
```

- `integration`: tự khởi động API localhost cổng tạm, database tạm; không chạm
  history/service đang chạy của người dùng. Dọn subprocess sau suite.
- `model`: opt-in `--run-model-tests`, chỉ dùng artifact/ảnh có sẵn. Thiếu legacy
  checkpoint thì skip có lý do, không tạo weights thay thế hoặc khôi phục model.
- Unit/real-model tests chặn tải qua urllib/Torch hub. News unit kiểm logic bằng
  fallback offline, không xác minh tin tức ngoài Internet.
- Hai file `test_gatekeeper_benchmark.py`, `test_ood_evaluation.py` là script
  đánh giá độc lập, không có test function pytest; loại khỏi discovery để tránh
  model/network side effects khi import. Không xóa nội dung và không tính là pass.
- Regression bắt buộc không cần weights: health degraded, readiness từng model,
  malformed/oversized/quality/URL/batch và cache model khác đường dẫn.
- Test opt-in dùng weights pin thật: inference, đổi đường dẫn cache, hash report
  sai, report bị chặn/hỏng, fault injection lỗi loader. Không tạo score giả.

Ba lỗi trước đã xử lý theo contract: health assertion dựa trạng thái legacy thật;
ảnh hỏng trả 400 trước readiness; disclaimer kiểm ý nghĩa hạn chế thay chữ hoa.
Không thay đổi disclaimer sản phẩm để phục vụ một assertion nguyên văn.

## Frontend consumer — Task 02B

- Tab chính gọi GET `/api/yolo/capabilities`, POST `/api/yolo/detect` (multipart file).
  Không fallback sang `/api/predict` hoặc `/api/detect` khi lỗi YOLO.
- Client kiểm MIME/10 MB/64 px/20 MP, chuẩn hóa EXIF thành JPEG, preview và upload
  cùng bytes. Backend tiếp tục dùng validator chung; client không thay validation server.
- Chỉ vẽ response inference completed, task detect/taxonomy object_groups, mapping
  9 nhóm và dimensions/xyxy/coordinate_space khớp ảnh; phản hồi sai bị từ chối.
- SVG scale theo ảnh; box/card dùng cùng index và class mapping từ response.
  Không tạo phân loại vật liệu hoặc hướng dẫn theo class YOLO.
- User report riêng gửi JSON `/api/safety/assess`; label vật liệu có ghi nguồn
  người dùng/chưa xác minh. Không tự chạy `/api/safety/scan` trên ảnh detector.
- Abort + sequence guard bỏ request cũ khi đổi/xóa ảnh/hủy/unmount; timeout 30 s.
  500/503 hiển thị message lỗi. Missing/invalid artifact khóa phân tích; khi
  `artifact_verified=true` và status `load_failed`/`inference_error`, cho phép
  POST retry với cảnh báo not-ready vì GET refresh không tự xóa lỗi runtime.
- Chưa thêm history/URL/batch/webcam/feedback YOLO; giữ contract/backend legacy.
