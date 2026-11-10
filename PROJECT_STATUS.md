# PROJECT_STATUS — AI Waste Classification

Cập nhật: **10/11/2026**. Mốc audit gốc: rà soát working tree tại
`D:\ai-waste-classification`, nhánh `main`, HEAD `00d9524`.
Repo có nhiều thay đổi local chưa commit từ công việc trước bước rà soát này.
Audit ban đầu chỉ đọc và ghi tài liệu. **Task 01 sau đó đã sửa API contract/backend
tests theo yêu cầu người dùng; không train, download, cài môi trường, khôi phục
MobileNet, commit/push hoặc migration UI.**

**Quyết định mới của người dùng ngày 10/11/2026:** không khôi phục MobileNet.
Đã bỏ việc này khỏi backlog ưu tiên; tiếp tục theo hướng YOLO và sàng lọc pin.
Code MobileNet hiện còn là legacy, chưa bị xóa trong lần cập nhật tài liệu này.
Bàn giao cho ChatGPT: `CHATGPT_HANDOVER.md`.

## Snapshot Git được người dùng yêu cầu

Ngày 10/11/2026 người dùng yêu cầu commit/push trạng thái hiện tại lên nhánh khác.
Nhánh snapshot: `codex/yolo-teacher-student-pipeline`, tách từ `main`/`00d9524`.
Phạm vi là source, tests, cấu hình và tài liệu của các công việc đã hoàn tất;
không đưa weights/dataset/database/log/workspace duyệt nhãn lên GitHub.
Các ghi chú “không commit/push” trong báo cáo task phía dưới là ràng buộc lịch sử,
đã được yêu cầu Git mới này cho phép riêng cho snapshot. Không suy thành quyền
train/download, merge vào main hoặc tự commit các task tương lai.

## Trạng thái mới nhất — Task 03A: Teacher–Student preparation (10/11/2026)

- Audit read-only 1.988 ảnh / 6.744 box, giữ nguyên 1.655 train / 318 val / 15 test.
  Train class counts: bottle_jar 2873, can 318, cup 89, carton 408, cap_lid 150,
  bag_film_wrapper 367, paper 60, cigarette 233, other_litter 1097.
- 1.988 nhãn/raw hashes khớp nguồn; không lỗi cấu trúc, exact duplicate hoặc
  identity/group cross-split collision qua kiểm đã chạy. Không chứng minh hết scene leakage.
  Có 39 ảnh tiny box, 1 conflict candidate, 58 unverified negatives, 346 thiếu
  license ref. dHash≤8 có 3.170 cặp (3.169 train/train, 1 train/val distance 8);
  giữ nguyên eval, loại train candidate khỏi pilot, chưa kết luận nhãn sai/leakage.
- Mới `scripts/teacher_student/`: CLI audit/pilot/import-teacher/serve/export,
  UI sửa class/box/vẽ/xóa, class/IoU/score QA, issues và provenance/revision.
  Unknown giữ ở draft/hold; approve/export bắt buộc nhãn hợp lệ và người duyệt,
  checklist, quyền ảnh, không issue mở, HMAC/revision và source fingerprint không đổi.
  Không có auto-approval, cloud call hoặc train trong tool.
- Adapter `propose-local` đã viết: local-files-only, pin artifact, safetensors,
  chặn network, giữ raw box/score/versions/timings và không auto-import/approve.
  `teacher-status` xác nhận thiếu Transformers; artifact preflight trả exit 2,
  không tạo output. Chưa smoke runtime/model Teacher thật; không coi unit pass
  là đã chạy Teacher.
- Pilot `data/teacher_student/task03a_pilot150`: 150 train-only (HF TACO 80/WaRP 55/
  Street 15), 1.378 box gốc; **150 pending / 0 Teacher / 0 approved**. Real export
  zero-approved bị chặn, không tạo output. Không dùng val/test để tạo train labels.
- Đề xuất Grounding DINO-T local; chưa có dependency/artifact Teacher ở roots đã
  kiểm, chưa có Teacher inference. Cần duyệt env riêng/weights ~689 MB; Qwen4B BF16
  weights ~8,88 GB vượt VRAM 6 GB. SAM2/VLM cloud là tùy chọn, chưa tải/cài/gửi ảnh/chi tiền.
- Kiểm thử: **45 pipeline tests PASS**; full **217 PASS / 0 FAIL / 20 SKIP +
  4 subtests PASS**; JS geometry 2 PASS, lint tool exit 0. Browser test sửa/lưu/reload,
  draw/delete/reset/layers/dirty guard và chặn approval trong workspace test_only riêng.
  Teacher overlay chỉ có schema/unit tests; chưa có output model thật để smoke.
- App/backend source, checkpoints, mapping, source snapshot và DB giữ nguyên.
  Review thật cổng 8013 giữ mở; cổng 8014 chỉ dùng test và được dọn sau test.
  Backend 8000/frontend 5173 không đổi. Không train/download/commit/push hoặc restore.
- Báo cáo `docs/TASK03A_REPORT.md`, audit `docs/DATASET_AUDIT_TASK03A.md`, research
  `docs/TEACHER_RESEARCH.md`, protocol `docs/TEACHER_STUDENT_PIPELINE.md`,
  `docs/TEACHER_PROMPT.md`; evidence/progress `logs/task03a/`.

**Tiếp theo:** xin duyệt Teacher/env/artifact rồi smoke 5–10 ảnh train, đo VRAM/
latency/annotation QA; người thật review pilot trước khi cân nhắc train. Taxonomy
pin/y tế/material/radiation-warning mới chỉ đề xuất, không sửa mapping 9. Không
làm lại Task 01/02A/02B hoặc tự chạy lệnh lịch sử.

## Task 02B — mốc UI đã hoàn tất (10/11/2026)

- Tab chính **Phân Loại Rác** đã dùng `/api/yolo/detect`, preview ảnh chuẩn hóa
  EXIF và mọi box có số/tên tiếng Việt/confidence/màu; chọn box ↔ focus/highlight card.
  Có 0/1/nhiều detection, confidence thấp, loading/hủy/timeout/request cũ và API lỗi.
- Vật liệu/xử lý mặc định **Chưa xác định**. Hướng dẫn có nguồn lấy từ API safety
  theo khai báo của người dùng, không suy từ 9 nhóm YOLO. Pin vẫn ở luồng ảnh một vật.
- Model version/ngưỡng/bất định hiển thị rõ. Runtime YOLO experimental_loaded,
  readiness 200; health degraded/legacy 503; production_ready=false.
- Frontend **10 PASS / 0 FAIL / 0 SKIP**, build PASS, lint exit 0 với **45 warning
  cũ**. Full backend regression **172 PASS / 0 FAIL / 20 SKIP + 4 subtests PASS**.
  Không sửa backend trong task này; ba regression Task 01 vẫn qua.
- Browser IAB đã kiểm ảnh validation thật 0/1/2 detection, box/card hai chiều,
  EXIF 6, viewport 390 px, khai báo vật liệu/nguy hại, pin positive/negative,
  ảnh hỏng/quá lớn; fault proxy riêng kiểm 500/503/unavailable/timeout/hủy/stale.
- Weight hashes và số dòng DB giữ nguyên. Không train/download/restore/commit/push.
  Cổng chính 5173/8000 giữ cho người dùng; proxy 8012/Vite 5174 đã dọn sau kiểm thử.
- Chưa kiểm điện thoại thật/Safari/Firefox, mọi EXIF, cảnh nhiều box chồng nhau;
  không nghiệm thu toàn bộ map/history. YOLO chưa có URL/batch/webcam/history/feedback;
  các tab legacy được giữ riêng, không dùng taxonomy vật liệu cho output mới.
- Báo cáo/file cụ thể/bằng chứng: `docs/TASK02B_REPORT.md`, `logs/task02b/`.
  Khi mở chat mới đọc phần này trước các mốc Task 01/02A bên dưới.

**Tiếp theo:** review UI và bổ sung browser regression bền vững; thiết kế history
YOLO có model/hash/taxonomy nếu cần; thu test độc lập tại Đà Nẵng trước đánh giá
chất lượng/triển khai. Không bắt đầu train/download khi chưa được giao.

## Task 02A — đã hoàn thành, backend YOLO riêng

- Đối chiếu checkpoint, head nc=9/task detect, per_class run metadata, dataset
  YAML khớp hash và taxonomy script. Thứ tự chính xác: `bottle_jar`, `can`, `cup`,
  `carton`, `cap_lid`, `bag_film_wrapper`, `paper`, `cigarette`, `other_litter`.
  Không dùng mapping vật liệu bảy lớp cũ. Pin hash/mapping/version trong
  `backend/app/yolo_detector_manifest.json`.
- Artifact `runs/waste_pilot/yolo26n_multisource30_v1/weights/best.pt` SHA-256
  `dc8f481be1668c960b78264db70c5f48c44d42b8fcb2fab93d23625532225efc`.
  Checkpoint và runtime Ultralytics 8.4.208; không thay weights/metadata nguồn.
- Thêm adapter local-only CPU/lazy load/lock và endpoint **POST /api/yolo/detect**,
  **GET /api/yolo/capabilities**. Dùng validator ảnh Task 01, không phụ thuộc
  readiness MobileNet, không thay route legacy/pin hoặc frontend.
- Response có class ID/name/label_vi, confidence, box xyxy pixel theo ảnh đã EXIF
  transpose, model version/hash/mapping, quality và inference/detection status.
  Không có detection vẫn bất định; không suy ra material/recycling/hazard.
- Ngưỡng môi trường: `YOLO_CONFIDENCE_THRESHOLD=0.25`,
  `YOLO_UNCERTAINTY_THRESHOLD=0.50`, kiểm `0 < confidence <= uncertainty <= 1`.
  Đây là cấu hình thử nghiệm, không phải accuracy/ngưỡng an toàn đã xác nhận.
- Readiness YOLO nay integrated=true: experimental_not_loaded/loaded khi artifact
  hợp lệ; missing/invalid/config/load/inference error phản ánh runtime và trả
  503/500 phù hợp. `production_ready=false`. Không fake health legacy.

Kết quả Task 02A ngày 10/11/2026:

| Kiểm tra | Kết quả |
|---|---|
| Full backend + dataset: `backend\.venv\Scripts\python.exe -m pytest backend/tests tests --run-model-tests -v --tb=short` | **172 PASS / 0 FAIL / 20 SKIP**, 4 subtest PASS, 3 warnings |
| Offline: `-m "not integration and not model"` | **154 PASS**, 38 deselected, 4 subtest PASS |
| Regression Task 01 | Ba lỗi cũ vẫn PASS; health assertion YOLO cập nhật theo capability thật, không ép trạng thái not_integrated cũ |
| Optional model checks | Pin thật vẫn qua; YOLO thật, hash/metadata tamper trên bản sao test và mapping/head/version checks qua |
| Live API 8000 | 2 ảnh val có sẵn từ WaRP/HF TACO: một empty, một có detection; response 200/schema/box hợp lệ. Ảnh hỏng 400 |
| Runtime sau smoke | YOLO experimental_loaded, readiness 200; legacy 503; production_ready=false |
| Bảo toàn | Hash weights/run metadata khớp manifest; frontend, image_input, battery/safety code và conftest Task 01 không đổi |

20 SKIP vẫn do checkpoint legacy thiếu, không phục hồi. Không cộng các nhóm
chồng nhau hoặc coi smoke là phép đo accuracy. 3 warnings dependency/hằng 413
như Task 01. Không sửa frontend nên không chạy lại build/lint. Không train,
download, cài lại môi trường, xóa artifact/data, commit/push hoặc tạo ảnh AI.
Log: `logs/task02a_artifact_inspection.log`, `task02a_full_suite.log`,
`task02a_offline_units.log`, `task02a_live_contract.json`.

File riêng Task 02A: mới `backend/app/yolo_detector.py`, `yolo_routes.py`,
`yolo_detector_manifest.json`, `backend/tests/test_yolo_detector.py`; sửa đăng ký
router trong `main.py`, nhánh YOLO trong `model_status.py`, regression health/
validation ở `test_api_contract.py`, `test_api_integration.py`; cập nhật tài liệu.

**Kết luận tại mốc 02A (02B đã hoàn tất ở phần mới nhất phía trên):** dùng endpoint/mapping mới,
vẽ box theo ảnh đã định hướng, hiển thị bất định và xử lý 4xx/5xx. Chưa đủ điều
kiện public hoặc tự động xử lý rác: chất lượng detector thấp, chưa có kiểm định
Đà Nẵng; endpoint mới chưa có URL/batch/history/feedback và không thay taxonomy cũ.

## Task 01 — đã hoàn thành

- Giữ health HTTP 200 cho liveness và trường legacy cũ; thêm trạng thái riêng
  `legacy_classifier`, `yolo_detector`, `battery_classifier`. Không báo YOLO ready
  khi chưa tích hợp, không lấy pin loaded để làm legacy loaded.
- Thêm `GET /api/readiness?capability=...`, 200/503 theo capability; thiếu/sai
  selector 422. Pin phân biệt artifact đã xác minh, chưa nạp, đã nạp, bị chặn,
  artifact hỏng và lỗi nạp/suy luận. Tất cả còn `production_ready=false`.
- Dùng decoder chung kiểm nội dung/10 MB/20 MP, EXIF/chất lượng trước readiness.
  Ảnh sai trả 400/413, ảnh hợp lệ nhưng legacy unavailable mới trả 503. Batch
  giữ envelope 200, thêm `items[].status_code`; URL route giữ đúng lỗi 413.
- Safety scan đổi lỗi ảnh/chất lượng từ 422 sang 400; 422 chỉ dùng schema.
  Payload thành công và các trường cũ vẫn giữ. Contract chi tiết:
  `docs/API_CONTRACT.md`.
- Sửa cả ba test lỗi cũ theo contract. Test HTTP tự mở API cổng tạm/database tạm,
  không dùng lịch sử ở cổng 8000. Unit không gọi mạng; model tests opt-in và không
  tải weights. Hai script benchmark không có pytest test được giữ ngoài discovery.
- Regression gồm readiness/missing model, ảnh sai/quá lớn/chất lượng/URL/batch,
  hash/report sai, cache model đường dẫn khác và fault injection loader.

Kết quả cuối ngày 10/11/2026 (không cộng các nhóm chồng nhau):

| Lệnh / kiểm tra | Kết quả |
|---|---|
| `backend\.venv\Scripts\python.exe -m pytest backend/tests tests --run-model-tests -v --tb=short` | **129 passed, 20 skipped, 0 failed, 4 subtests passed**, 3 warnings |
| `backend\.venv\Scripts\python.exe -m pytest backend/tests tests -m "not integration and not model" -q` | **115 passed, 34 deselected, 0 failed, 4 subtests passed**, 3 warnings |
| Ba case health/invalid image/analytics disclaimer trước đây | Đều PASS, đã chạy thật; không bị skip |
| Pin thật qua API tại cổng 8000 sau restart | 4 ảnh từ hai nguồn + đổi filename: PASS; đây là smoke API, không đo accuracy mới |
| Live health/readiness | Health 200/degraded; legacy 503; YOLO 503/not_integrated; pin 200/experimental_loaded sau scan |
| Checkpoint pin v2 | Hash vẫn khớp report, không thay weights |

20 SKIP đều là case cần checkpoint legacy đang thiếu, không phải test đã được
chứng minh chạy đúng. Không khôi phục model để chạy chúng. 3 warnings: một cảnh
báo Starlette/httpx và hai cảnh báo tên hằng HTTP 413 cũ; chưa cài/nâng dependency.
Frontend không thay đổi trong Task 01; số build/lint bên dưới là baseline audit.
Bằng chứng: `logs/task01_final_suite.log`, `task01_offline_units.log`,
`task01_live_contract.json`, `task01_real_battery_api.log`.

File code/test thay đổi riêng Task 01: `backend/app/main.py`, `battery_screen.py`,
`safety_routes.py`; mới `image_input.py`, `model_status.py`; `pytest.ini`,
`backend/tests/conftest.py`, `test_api_contract.py`, `test_api_integration.py`,
`test_analytics_dashboard.py`, `test_safety_api.py`. Thay đổi local khác đã tồn
tại trước Task 01 và được giữ nguyên.

## Kết luận hiện tại

Dự án là ứng dụng thử nghiệm cục bộ, có giao diện/API/dữ liệu lịch sử tương đối
đầy đủ. Hướng phát triển hiện tại là YOLO và sàng lọc pin. Chưa thể coi là hệ thống
đã nghiệm thu: YOLO chín nhóm đã tích hợp backend và UI chính thử nghiệm ở Task 02B,
còn 20 test legacy
không chạy được vì thiếu artifact, dữ liệu bản đồ có nguồn đặt sẵn trong code và model pin chưa được kiểm
định độc lập tại Đà Nẵng. Thiếu checkpoint MobileNet là tình trạng legacy, không
phải task cần khôi phục theo quyết định mới.

## Kiến trúc và công nghệ được xác minh

| Thành phần | Hiện trạng |
|---|---|
| Frontend | React 19.3.1, Vite 8.3.4, JavaScript/JSX, CSS, Oxlint; có package-lock. `frontend/src/App.jsx` điều phối phần lớn state, API và tab. |
| Backend | FastAPI, Pydantic, Uvicorn; `backend/app/main.py` chứa schema, startup, inference và nhiều API. |
| Lưu trữ | SQLite, SQL tham số hóa, WAL, lưu metadata thay vì ảnh nhị phân. Chưa có phân tách tài khoản người dùng. PostgreSQL/Supabase trong README là dự kiến cũ. |
| AI sản phẩm cũ | PyTorch/torchvision MobileNetV3-Large; chuẩn hóa ảnh, temperature/margin uncertainty, Grad-CAM. SSDLite COCO đề xuất box rồi classifier phân loại crop. |
| AI thử nghiệm mới | YOLO26n detector chín nhóm riêng; YOLO26n-cls pin hai nhãn riêng. Không được thay chéo checkpoint hoặc taxonomy. |
| Hướng dẫn | Knowledge base JSON tách khỏi model; router safety phân biệt gợi ý thị giác và thông tin người dùng tự báo. |
| Bản đồ | Leaflet qua `window.L`, nhiều nguồn tile và OSRM/Google Maps bên ngoài; báo cáo điểm rác lưu SQLite. Không bảo đảm hoạt động offline. |
| Môi trường máy | Python 3.11.9, torch 2.14.1+cu126, torchvision 0.29.1, RTX 4050 Laptop. Backend venv và môi trường YOLO riêng. |

Luồng chính: React → FastAPI → kiểm ảnh/inference → tra hướng dẫn → SQLite
metadata/response. Các endpoint safety chạy classifier pin CPU riêng; health
MobileNet và readiness pin là hai năng lực khác nhau.

## Tính năng: có code không đồng nghĩa đã nghiệm thu

| Nhóm | Đã có | Còn thiếu / giới hạn |
|---|---|---|
| Nhận diện cũ (legacy) | File, URL, batch, webcam UI, uncertainty, Grad-CAM | Vẫn phụ thuộc checkpoint MobileNet đang thiếu; người dùng không yêu cầu khôi phục. Cần chuyển luồng phù hợp sang hướng YOLO khi triển khai tiếp. |
| Nhiều vật | UI chính YOLO chín nhóm: preview/box/card và hướng dẫn theo khai báo; SSDLite legacy giữ ở tab riêng | Chưa có YOLO history/URL/batch/webcam. Chất lượng còn thấp; không thay taxonomy vật liệu hay nhận diện pin/medical/phóng xạ. |
| Lịch sử/feedback | CRUD, filter, CSV, báo sai, analytics SQLite; unit test qua | Metadata feedback chưa tạo thành vòng active learning tự động có QA/nhãn mới. |
| Chấm điểm model | API/dashboard đọc evaluation/training JSON | Metadata cũ và thông số đặt sẵn; chưa gắn chặt checkpoint hash/split/model version đang phục vụ. |
| Điểm tái chế | Directory/filter, registry pin Đà Nẵng có nguồn và trạng thái | Chưa xác nhận hoạt động/tiếp nhận hiện tại; không có danh sách cơ sở y tế/phóng xạ được xác minh. |
| Bản đồ rác | CRUD, nearby, vote, status, tin RSS + nguồn curated | Nguồn curated/vote đặt sẵn và geocode suy từ địa danh; chưa chứng minh là dữ liệu xác minh trực tiếp. |
| Pin | V2 đa nguồn đã train; API thực nạp được, kiểm hash, không lưu ảnh upload | Ảnh một vật, không có box; không tìm pin ẩn, hóa học, tình trạng pin hay chứng nhận an toàn. |
| Rác y tế/phóng xạ | Hướng dẫn phòng ngừa theo thông tin người dùng cung cấp | Chưa train/evaluate medical detector mới. Ảnh RGB không đo bệnh truyền nhiễm/liều bức xạ. |
| Mở rộng địa phương | Ưu tiên Đà Nẵng, có kế hoạch và quy trình thu thập ảnh | Người dùng xác nhận chưa có ảnh Đà Nẵng; chưa có test Bắc/Nam. |
| Vận hành | Logging Python/Uvicorn, script kiểm môi trường, log train/audit | Chưa thấy CI, frontend test runner, auth/RBAC, quản lý artifact sản phẩm và quan sát vận hành đầy đủ. |

## Baseline audit trước Task 01 — lịch sử, không phải kết quả mới nhất

Log bằng chứng ở `logs/project_status_*`. Không dùng kết quả JSON lịch sử làm
phép đo mới. Các test có thể trùng phạm vi; không cộng 7 pass của full suite vào
65 test độc lập bên dưới để tăng tổng số.

| Kiểm tra | Kết quả |
|---|---|
| `npm.cmd --prefix frontend run build` | PASS, exit 0. |
| `npm.cmd --prefix frontend run lint` | Exit 0 nhưng **45 cảnh báo**; chủ yếu ở App/SmartLitterMap, gồm unused và React hooks/state. Không coi là lint sạch. |
| Standalone unit + safety + dataset tests | **57 passed**, thêm **4 subtests passed**; 1 cảnh báo Starlette/httpx. |
| Image quality unit subset | **8 passed**, 7 deselected. Tổng hai nhóm độc lập: **65 passed**. |
| Full backend, API đang chạy, `--maxfail=3` | **3 failed, 7 passed**, dừng sớm; không kết luận các test còn lại đạt. |
| `scripts/check_setup.py` | Exit **2**: CUDA/tensor/torchvision hoạt động, thiếu checkpoint MobileNet; không phải môi trường inference đầy đủ. |
| Backend `pip check` | PASS, không có dependency hỏng trong môi trường hiện tại. |
| GET health | HTTP 200 nhưng `status=degraded`, `checkpoint_loaded=false`, `ai_model_loaded=false`. |
| GET safety capabilities | HTTP 200, model pin đã nạp; `experimental_development_only`, `independent_external_validated=false`. |
| GET history/analytics | HTTP 200. Nội dung lịch sử cá nhân không chép vào báo cáo audit. |

Ba lỗi full suite từng tái hiện (đã xử lý trong Task 01, xem phía trên):

1. `TestApiAnalyticsIntegration::test_api_get_dashboard_analytics_structure`:
   so khớp nguyên văn disclaimer, thiếu chuỗi `KHÔNG đại diện` dù response có
   nội dung tương đương. Đây là lệch assertion/contract, chưa sửa trong bước này.
2. `TestApiIntegration::test_health_endpoint`: test đòi `ok`, thực tế `degraded`
   vì thiếu checkpoint. Không được làm health báo sẵn sàng giả để test xanh.
3. `TestApiIntegration::test_predict_invalid_image_bytes`: test đòi HTTP 400,
   API trả 503 khi model chưa sẵn sàng; cần thống nhất thứ tự validation/readiness.

Hạn chế ở baseline audit: benchmark có thể thực thi khi import và cần cache/checkpoint;
`test_multi_object_detection.py` có hàm cần các đối số model nhưng không khai báo
pytest fixtures tương ứng. Có đường dẫn máy cũ, test gọi cổng 8000 cố định. Chưa
có kiểm thử frontend tự động/coverage được xác minh. Không báo coverage giả.

## Kết quả AI từ lượt train trước bước audit

Đây là artifact thực vừa tạo trong phiên làm việc 10/11/2026, **không chạy lại
training/evaluation model trong bước audit**. Xem `docs/MULTISOURCE_BATTERY_RESULTS.md`
và các run tương ứng để có checkpoint hash, nguồn/split và điều kiện đo.

- Detector `yolo26n_multisource30_v1`: 30 epoch, 1.655 train/318 val/15 test.
  Trên cùng validation hỗn hợp: mAP50 pilot 2,87% → mới 26,04%; mAP50–95 18,77%,
  recall 27,87%. Test TACO chỉ 15 ảnh: mAP50 16,66%, chưa đại diện thực tế.
- Pin baseline v1: accuracy test nội bộ 95,59%, nhưng recall ngoài miền RecyBat
  **0/2.834** tại ngưỡng 0,95. Đã giữ baseline và chặn runtime mặc định của bản này.
- Pin v2: Kaggle + RecyBat, tách 110 ID viên pin RecyBat theo dạng + ID; 20 epoch,
  2.404 train/646 val/760 development holdout. Ngưỡng 0,95 từ validation.
  Holdout gộp: precision 99,83%, recall 98,13%, accuracy 98,42%.
  **Kaggle riêng vẫn recall 91,13%**, 11/124 ảnh pin bị bỏ sót; RecyBat holdout
  nhận 465/465 ảnh positive nhưng không có negative độc lập cùng miền.
- RecyBat đã dùng chẩn đoán v1 rồi chọn làm nguồn phát triển v2; không được gọi
  holdout v2 là test ngoài miền độc lập hoặc accuracy tại Đà Nẵng. Ảnh nhiều mặt
  cùng viên tương quan; Kaggle thiếu ID vật. Chưa đủ điều kiện dùng không giám sát.
- API pin đã được smoke test trên bốn ảnh thật từ hai nguồn ở bước trước; score
  không đổi khi thay tên tệp. Đây là kiểm API, không phải benchmark accuracy mới.

## Vấn đề, rủi ro và technical debt

| Mức | Vấn đề và bằng chứng | Tác động / việc cần làm |
|---|---|---|
| P1 — chất lượng trước public | Task 02B đã có UI chính YOLO thử nghiệm; detector còn yếu, chưa có test Đà Nẵng độc lập | Rà nhãn/test thực tế và kiểm định trước public; không chuyển score thành chứng nhận an toàn. Không khôi phục MobileNet. |
| P1 — tính trung thực dữ liệu | `news_crawler.py` khởi tạo từ `CURATED_VERIFIED_NEWS_REPORTS`, có tọa độ/lượt vote cố định; RSS còn gán vote cố định | Chưa đủ căn cứ gọi là tin/vote xác thực. Không chạy nút sync để kiểm audit vì sẽ ghi DB; cần phân biệt dữ liệu mẫu/curated/nguồn live và lưu provenance xác minh. |
| P1 — model contract/metrics | README nói sáu lớp và 95,98%; `class_mapping.json` có bảy lớp. `/api/model/metrics` có `num_classes=6` nhưng `classes_order` bảy phần tử, latency/thông số đặt sẵn | Cần thống nhất version, class order, checkpoint hash, tập đo; không gán số liệu cũ cho model đang chạy. |
| P1 — trước khi public | Hai URL prediction route chỉ thấy kiểm scheme/netloc rồi `urlopen`, chưa có chặn IP private/loopback/redirect | Rủi ro SSRF từ phân tích code; chưa thực hiện khai thác. Cần kiểm DNS/IP đích và từng redirect, timeout/giới hạn tài nguyên. |
| P1 — trước khi public | Không thấy auth/authorization; history delete/status/vote/sync có thể truy cập không qua vai trò; CORS rộng, Vite `allowedHosts=true` | Chỉ phù hợp phạm vi local được kiểm soát; cần giới hạn truy cập/host/origin và quyền thao tác trước mở LAN/Internet. |
| P1 — độ chính xác nguy hại | Baseline domain failure, recall từng nguồn chưa đạt mục tiêu; chưa có dữ liệu địa phương | Cần test theo miền/vật thể và cơ chế chuyển người xem; không dùng score âm tính xác nhận an toàn. |
| P2 — test/reproducibility | Task 01 đã tách marker/fixture, server tạm và benchmark; 20 test legacy còn thiếu artifact, chưa có CI | Giữ regression mới, thêm CI offline sau; không báo các case skip là đã kiểm định inference legacy. |
| P2 — kiến trúc | App.jsx và main.py lớn, trộn schema/UI/state/inference/networking; IO đồng bộ trong async URL routes | Khó bảo trì và dễ chặn request; tách dần sau khi khóa hành vi bằng test, chưa refactor trong audit. |
| P2 — dependency/docs | Backend requirements phần lớn dùng `>=`, frontend README còn template; RESUME từng ghi trạng thái đang train | Cần tái lập môi trường bằng lock/constraints có CUDA phù hợp và cập nhật tài liệu nguồn chân lý. |

Đây không phải pentest hoặc chứng nhận bảo mật/pháp lý. Rủi ro từ code được
phân biệt với lỗi tái hiện qua test. Không xác nhận địa điểm/rủi ro lây nhiễm chỉ
bằng tên cơ sở, filename hoặc một ảnh.

## AGENTS, rules, skills và logging

- **Đã đủ nền tảng hướng dẫn**: AGENTS.md, `docs/handover/AGENT_RULES.md` và skill
  `.agents/skills/ai-waste-dev/SKILL.md` đã có, áp dụng đúng repo. Không cần thêm
  skill/rules trùng lặp hay cài plugin trong bước này. Cập nhật tiến trình là cần
  thiết để tránh agent sau khởi động lại lượt train đã hoàn tất.
- **Test đã bổ sung trong Task 01**: readiness, validation, cache/hash/failure và
  tách suite. Bước tiếp theo còn contract detector/lớp/metrics, SSRF/redirect,
  quyền thao tác, provenance news/votes và smoke frontend khi thực hiện migration.
- **Logging đã có**: Python logging theo module, exception, Uvicorn access log,
  log training/evaluation. Chưa cần dựng ELK/service riêng. Khi chuẩn bị vận hành,
  thêm request ID, model version/hash, latency và mã lỗi; rotation/retention phù
  hợp; không log ảnh, token, nội dung lịch sử hay vị trí cá nhân không cần thiết.

## Backlog và ba ưu tiên

| ID | Task | Trạng thái / tiêu chí hoàn tất |
|---|---|---|
| DONE-01 | Đọc repo, README, config, hướng dẫn agent | Hoàn tất audit hiện tại. |
| DONE-02 | Build/lint/unit/full test và readiness | Audit ban đầu có ba lỗi, Task 01 đã xử lý; regression mới nhất 172 PASS/20 SKIP, không còn FAIL. |
| DONE-03 | Train đa nguồn/pin và báo cáo trước audit | Hoàn tất các lượt đã nêu; không còn process train/download được quan sát. |
| DONE-04 | **Task 01: API contract và backend tests** | Hoàn tất: health/readiness theo capability, validation trước model, ba lỗi cũ PASS; full superset 129 PASS/20 SKIP. |
| DONE-05 | **Task 02A: adapter/endpoint YOLO backend** | Hoàn tất: artifact/mapping verified, endpoint/schema/readiness/errors và test; full 172 PASS/20 SKIP. |
| DONE-06 | **Task 02B: UI YOLO thử nghiệm** | Hoàn tất preview/box/card/khai báo/readiness/error; frontend 10 PASS, backend 172 PASS/20 SKIP. Chưa đủ chất lượng production. |
| DONE-07 | **Task 03A: Teacher–Student preparation** | Audit, pilot150, QA/review/import/export gates; 45 pipeline tests, full217 PASS/20 SKIP. Teacher chưa chạy, 0 approved, không train. |
| NEXT-01 | **Teacher local smoke và người duyệt pilot** | Cần phê duyệt env/artifact trước tải/cài; chạy 5–10 train-only, đo VRAM/latency và QA rồi human review. Chưa được train. |
| NEXT-04 | **Review UI, browser regression và history YOLO** | Backlog giữ lại: chốt UI, test trình duyệt bền vững và thiết kế history có model/hash/taxonomy. Không ghép taxonomy cũ. |
| NEXT-02 | **Đưa suite đã chuẩn hóa vào CI tối thiểu** | Task 01 đã hoàn tất tách suite và sửa ba lỗi; chưa tạo CI. Không giao lại các việc đã xong. |
| NEXT-03 | **Chốt provenance dữ liệu và rào bảo vệ trước public** | Ưu tiên 3: loại bỏ việc coi seed/vote đặt sẵn là xác minh; xác nhận điểm thu gom; xử lý URL SSRF, auth/quyền và cấu hình truy cập. |

Không tự train lại, tải thêm dataset hoặc sửa logic từ báo cáo audit này. Mọi thay
đổi tiếp theo cần bám nhiệm vụ được người dùng giao; giữ nguyên checkpoint/data,
không commit/push/đổi nhánh nếu chưa được yêu cầu.
