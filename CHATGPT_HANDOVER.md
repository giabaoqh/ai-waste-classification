# Báo cáo bàn giao cho ChatGPT — AI Waste Classification

Cập nhật ngày **10/11/2026, sau Task 03A**. File này giúp một cuộc chat mới hiểu mục tiêu,
công việc đã thực hiện và điểm tiếp tục, không cần đọc lại toàn bộ hội thoại.

Người dùng sau đó đã yêu cầu commit/push snapshot lên nhánh riêng
`codex/yolo-teacher-student-pipeline` từ `main`/`00d9524`. Source/tests/docs được
đưa vào snapshot; model/data/database/log/review workspace giữ local ngoài Git.
Kiểm `git log`, upstream và Git status khi tiếp tục. Quyền commit/push lần này
không phải quyền train/download hay merge main.

## 1. Người dùng muốn làm gì?

Phát triển dự án AI phân loại rác, nâng độ chính xác, nhận diện pin, hướng dẫn
phân loại/tái chế và xử lý rác có nguy cơ. Ưu tiên **Đà Nẵng**, sau đó mở rộng
miền Bắc/miền Nam. Người dùng đã xác nhận **chưa có ảnh chụp thực tế Đà Nẵng**.
Không tạo ảnh AI vì người dùng muốn tiết kiệm token.

**Quyết định mới nhất: KHÔNG khôi phục MobileNet.** Bỏ task khôi phục model đó
khỏi danh sách ưu tiên, tiếp tục hướng YOLO và sàng lọc pin. Backend YOLO riêng
và UI chính thử nghiệm đã tích hợp. Giữ code MobileNet/SSDLite legacy riêng, không khôi phục.
Đừng tự diễn giải quyết định này thành lệnh xóa code hoặc checkpoint.

### Mốc mới nhất: Task 03A đã chuẩn bị pipeline, chưa chạy Teacher/train

- Audit read-only 1.988 ảnh, giữ 1.655/318/15 split; 1.988 labels/raw hashes khớp
  nguồn, không phát hiện lỗi cấu trúc/exact leakage. Bottle/paper lệch 47,88 lần;
  39 ảnh tiny, 1 overlap/class candidate, 58 unverified negatives, 346 thiếu license ref.
  dHash≤8 có 3.170 cặp, chỉ 1 cross-split (distance 8); train candidate loại khỏi
  pilot. Không sửa val/test hoặc coi heuristic là leakage đã xác nhận.
- Pilot thật `data/teacher_student/task03a_pilot150`: 150 train-only (TACO 80,
  WaRP 55, Street 15), 1.378 box gốc, **150 pending / 0 Teacher / 0 approved**;
  export zero-approved bị chặn.
- Mới `scripts/teacher_student/`: audit/pilot/import JSON Teacher/QA/review UI/
  approved-only export. Sửa/vẽ/xóa box, class/IoU/score, issues/provenance,
  revision/HMAC/frozen-split gates. Unknown giữ draft/hold; không auto-map/approve.
- 45 pipeline tests PASS; full 217 PASS / 0 FAIL / 20 SKIP + 4 subtests PASS;
  JS geometry 2 PASS, lint tool exit 0. Browser kiểm trên workspace test_only riêng,
  không tự duyệt nhãn thật. App, weights, source dataset, DB không đổi.
- Teacher deps/artifacts chưa có trong roots/venvs đã kiểm. Đề xuất Grounding DINO-T
  (HF revision/hash pin trong research, weights 689 MB); VLM/SAM 2 tùy chọn.
  Không download/cài/paid API/gửi ảnh/cloud inference/train/restore/commit/push.
- Adapter `local_teacher.py`/CLI `propose-local` đã có, offline/local-files-only,
  pin checksum/revision và safetensors. Chưa real-model smoke vì thiếu artifact/deps;
  preflight trả exit2, không tạo bundle. Đọc protocol trước khi dùng sau phê duyệt.
- Review thật cổng **8013** giữ cho người dùng; cổng 8014 test được dọn sau test;
  5173/8000 giữ nguyên. Kiểm process trước tái sử dụng. Đọc `docs/TASK03A_REPORT.md`,
  `docs/DATASET_AUDIT_TASK03A.md`, `docs/TEACHER_RESEARCH.md`,
  `docs/TEACHER_STUDENT_PIPELINE.md`, `docs/TEACHER_PROMPT.md`, `logs/task03a/PROGRESS.md`.
- Tiếp theo cần phê duyệt Teacher/env/weights rồi smoke 5–10 train-only, đo nguồn lực
  và annotation QA, sau đó human review. Chưa đủ train vì 0 approved; không tự chạy
  lệnh train lịch sử hoặc làm lại 01/02A/02B. Mapping 9 đang phục vụ không đổi.

### Task 02B đã hoàn tất — mốc trước đó

- Tab Phân Loại Rác dùng `/api/yolo/detect`, preview chuẩn hóa EXIF, mọi box
  số/tên Việt/confidence/màu và card phía dưới; chọn box ↔ card hai chiều.
- Không suy vật liệu/hazard từ 9 nhóm. Vật liệu mặc định Chưa xác định;
  user report → `/api/safety/assess`; pin classifier chỉ chạy riêng với ảnh một vật.
- Loading/abort/timeout/stale/API errors và explicit retry lỗi tạm thời đã có.
  Backend Task 02A không sửa lại; weights và số dòng DB giữ nguyên.
- Frontend **10 PASS**, build/lint exit 0 (45 warning cũ); full backend
  **172 PASS / 0 FAIL / 20 SKIP + 4 subtests PASS**. Không coi là model accuracy.
- IAB thực: 0/1/2 detection, box/card, EXIF 6, mobile 390 px, hướng dẫn khai báo,
  pin thật Kaggle/RecyBat + negative; fault proxy riêng kiểm 500/503/timeout/stale.
  Chưa Safari/Firefox/điện thoại thật, mọi EXIF, crowded scene, full map/history.
- YOLO readiness 200/experimental_loaded, legacy 503, production_ready=false.
  Dùng `docs/TASK02B_REPORT.md`, `logs/task02b/PROGRESS.md` và evidence logs.
- Main app 5173/backend 8000 giữ cho người dùng; môi trường fault test 5174/8012
  đã dọn sau test (xem processes_after.json). Kiểm process lại ở chat sau,
  không dừng process lạ.

## 2. Dự án và môi trường

- Repo: `https://github.com/giabaoqh/ai-waste-classification.git`.
- Thư mục: `D:\ai-waste-classification`; nhánh `main`, HEAD `00d9524` lúc audit.
- Có nhiều thay đổi local chưa commit. Đọc Git status trước khi sửa; không ghi
  đè thay đổi, commit/push/đổi nhánh nếu người dùng chưa yêu cầu.
- Stack: React/Vite + FastAPI/Pydantic + PyTorch + SQLite metadata.
- Python backend: `backend\.venv\Scripts\python.exe`.
- Môi trường YOLO riêng: `experiments\yolo\.venv\Scripts\python.exe`.
- Máy: RTX 4050 Laptop, khoảng 6 GB VRAM; CUDA hoạt động. Không cần cài lại môi
  trường hoặc tải lại dataset đã có. Weights/dataset nằm ngoài Git.

## 3. Đã làm tới đâu?

### A. Detector rác đa nguồn — đã train và tích hợp backend riêng ở Task 02A

- Chuẩn bị Kaggle WaRP, Hugging Face TACO mirror và ảnh negative Street-garbage.
  TACO mirror vẫn là cùng nguồn TACO, không tính thành nguồn độc lập mới.
- Dataset `data/processed/waste_multisource_v3`: **1.655 train / 318 val / 15 test**.
  Manifest/audit/mapping có sẵn. Thư mục v1/v2 dở dang không dùng để train.
- YOLO26n đã chạy **30 epoch**, khoảng **23 phút**.
- Run: `runs/waste_pilot/yolo26n_multisource30_v1/`.
- So sánh cùng validation hỗn hợp: mAP50 pilot **2,87% → 26,04%**;
  model mới mAP50–95 **18,77%**, recall **27,87%**.
- Test TACO giữ riêng chỉ 15 ảnh: mAP50 **16,66%**. Chất lượng còn thấp, không
  mô tả là detector đã đạt độ chính xác triển khai.
- Chín nhóm: `bottle_jar`, `can`, `cup`, `carton`, `cap_lid`,
  `bag_film_wrapper`, `paper`, `cigarette`, `other_litter`.
  Đây là nhóm vật thể, không phải taxonomy bảy vật liệu cũ; chưa có nhãn pin/medical.

**Task 02A đã xong:** adapter `backend/app/yolo_detector.py`, router
`yolo_routes.py`, manifest pin hash/mapping/version. Endpoint
`POST /api/yolo/detect` (file) và `GET /api/yolo/capabilities`; dùng validation
Task 01. Response class/confidence/box xyxy pixel trên ảnh sau EXIF transpose,
model identity/hash/mapping, quality và trạng thái inference/bất định.
`YOLO_CONFIDENCE_THRESHOLD=0.25`, `YOLO_UNCERTAINTY_THRESHOLD=0.50` chỉ là cấu hình
thử nghiệm, không phải ngưỡng an toàn/accuracy. CPU/lazy-load, chỉ artifact có sẵn.
Readiness YOLO đã integrated, chuyển experimental_not_loaded → experimental_loaded
sau inference thật; invalid artifact/config/load/inference failures có status/error.
Không thay `/api/detect` legacy hoặc safety pin. Sau đó Task 02B đã chuyển UI chính
sang endpoint YOLO; các tab/route legacy riêng giữ nguyên.
Không có URL/batch/history/feedback mới cho YOLO. Không gợi ý thùng rác theo
chín nhóm; hazard_assessment=not_assessed, confirmed_safe=false.

### B. Pin — đã train hai bản, API đang dùng bản v2 thử nghiệm

1. **Baseline Kaggle-only v1:** 17 epoch, test nội bộ accuracy **95,59%** nhưng
   trên RecyBat ngoài miền nhận **0/2.834 ảnh pin** tại ngưỡng 0,95. Đã đối chiếu
   score CPU API/GPU batch trên cùng ảnh, chưa thấy lỗi pipeline giải thích thất
   bại. Giữ baseline để truy vết, gắn `blocked_known_domain_failure`.
2. **V2 Kaggle + RecyBat:** 20 epoch; **2.404 train / 646 val / 760 holdout phát triển**.
   RecyBat tách theo **110 viên pin, dạng + ID**, tất cả mặt/ánh sáng cùng viên
   thuộc một split. Train cân bằng 1.202 pin và 1.202 negative. Giữ split Kaggle cũ.
3. Ngưỡng **0,95** chọn từ validation. Holdout gộp: precision **99,83%**, recall
   **98,13%**, accuracy **98,42%**, TP=578, FP=1, FN=11, TN=170.
4. **Phải báo riêng:** Kaggle recall vẫn **91,13%**, bỏ sót 11/124 ảnh pin;
   RecyBat holdout nhận 465/465 ảnh pin nhưng phần nguồn này chỉ có positive.
   Không lấy điểm gộp cao che việc bỏ sót ở Kaggle.
5. RecyBat đã được xem khi chẩn đoán v1 rồi dùng phát triển v2, nên holdout v2
   **không phải kiểm định ngoài miền độc lập**, càng không phải accuracy Đà Nẵng.
   Kaggle thiếu ID vật, các ảnh nhiều mặt RecyBat có tương quan.

Artifact v2: `runs/battery_screen/yolo26n_cls_multisource_v2/` gồm
`weights/best.pt`, `battery_report.json`, `heldout_predictions.json`, `results.csv`.
SHA-256 weights:
`d49244d539b00f3f6c9265be66963925b2b0a7d4f05f0b3fb09d5f46b9688b4e`.

API `/api/safety/scan` đã thử bằng ảnh thật từ cả Kaggle và RecyBat; không dùng
tên tệp để quyết định nhãn, không lưu ảnh upload, kiểm hash report/weights trước
khi nạp. Đây là **classifier ảnh cận cảnh một vật**, chưa có box, chưa tìm pin ẩn,
chưa nhận hóa học/tình trạng hỏng/cháy. Trạng thái `experimental_development_only`.

### C. Hướng dẫn và điểm thu gom — đã có bản thử nghiệm

- Tab **Pin và rác có nguy cơ** có sàng lọc pin và lựa chọn quan sát của người dùng.
- Hướng dẫn tách pin nguyên vẹn, hỏng, nóng/bốc khói; rác y tế/sắc nhọn; vật nghi
  phóng xạ; cùng giấy, nhựa, kim loại, thủy tinh, hữu cơ và bao bì hỗn hợp.
- Hướng dẫn dựa trên EPA/WHO/IAEA và nguồn Việt Nam được ghi trong tài liệu.
- **Chưa train medical detector mới. Không xác nhận nhiễm bệnh hay đo/loại trừ
  bức xạ bằng RGB.** Medical/phóng xạ hiện là hướng dẫn theo thông tin người dùng báo.
- FEC từng công bố thu gom pin tại 108 Phan Châu Trinh và 808 Ngô Quyền,
  Đà Nẵng. Registry ghi **cần xác nhận hiện trạng**, không bảo đảm còn nhận,
  không có tọa độ/giờ mở cửa bịa. Không mặc định nhận pin hỏng, rác y tế/phóng xạ.
- Bản đồ/tin cũ có dữ liệu curated và vote đặt sẵn trong code; chưa được coi là
  nguồn live đã xác minh. Audit chỉ ghi nhận, chưa sửa logic này.

## 4. Kiểm tra gần nhất và vấn đề còn lại

Các lệnh kiểm tra đã chạy ngày 10/11/2026; không phải phép đo mới của một máy khác:

| Kiểm tra | Kết quả |
|---|---|
| Frontend build | Qua |
| Frontend lint | Exit 0, còn **45 cảnh báo** |
| Nhóm offline `not integration and not model` | **154 PASS, 38 deselected, 4 subtest qua** |
| Full backend + dataset, `--run-model-tests` | **172 PASS, 20 SKIP, 0 FAIL, 4 subtest qua**, không dừng sớm |
| pip check | Không xung đột dependency |

Ba lỗi cũ **đã sửa và cả ba case đều PASS**: disclaimer kiểm ý nghĩa hạn chế;
health kiểm trạng thái legacy thực thay vì buộc `ok`; ảnh hỏng trả 400 trước lỗi
thiếu model. 20 SKIP là test cần legacy checkpoint đang thiếu, không phục hồi nó.
API HTTP tests tự mở server/database tạm; unit offline; model tests opt-in, không
download. Các script benchmark giữ ngoài pytest discovery, không xóa code.

Contract mới tại `docs/API_CONTRACT.md`: health 200/liveness + capabilities riêng;
readiness bắt buộc chọn `legacy_classifier`, `yolo_detector`, `battery_classifier`,
trả 200/503 theo model. Legacy unavailable; YOLO đã integrated experimental từ
Task 02A; pin vẫn experimental. Tất cả production_ready=false.
Ảnh sai 400, quá giới hạn 413, schema sai 422, ảnh hợp lệ nhưng model thiếu 503.
Safety scan cũng dùng 400 thay 422 cho lỗi ảnh; batch giữ envelope 200, thêm
status_code từng item. Cache pin theo artifact hiện tại, kiểm hash và lỗi loader.
Không khôi phục MobileNet, train/download hoặc migration UI trong Task 01.

Các rủi ro khác: README/dashboard metrics còn lệch sáu/bảy lớp và số liệu cũ;
20 test legacy chưa kiểm định inference, chưa có CI/frontend test;
URL fetch chưa thấy chặn private IP/redirect, thiếu phân quyền khi public;
`App.jsx`/`main.py` lớn. Xem `PROJECT_STATUS.md` để có bằng chứng và backlog.

Lúc viết bàn giao, cổng local 8000 và 5173 đang lắng nghe; không quan sát thấy
process train/download. Đây là ảnh chụp trạng thái, cần kiểm lại khi mở chat mới.
Health legacy vẫn có thể degraded dù model pin hoạt động; dùng
`GET /api/safety/capabilities` để kiểm năng lực pin riêng.

## 5. Tiếp theo nên làm gì?

1. **Teacher pilot:** Task 03A đã có audit/pilot/review/export, không giao lại.
   Cần phê duyệt artifact/env, smoke 5–10 ảnh train và human review; chưa train.
2. **Review UI/browser regression/history YOLO:** Task 02B đã làm UI chính;
   không giao lại adapter/backend/mapping hoặc box/card. Chốt trải nghiệm với nhóm,
   tạo browser regression bền vững và thiết kế lưu detection gắn model/hash/taxonomy
   nếu cần history/feedback. URL/batch/webcam/Grad-CAM YOLO chưa có contract mới.
3. **CI:** tests Task 01/02A/02B và 45 pipeline tests Task 03A đã có;
   đưa nhóm offline/frontend/pipeline checks vào CI.
4. **Dữ liệu thực tế và độ tin cậy:** thu thập test Đà Nẵng theo quy trình đã viết,
   bổ sung negative khó/ngoài miền, xác minh điểm nhận; xử lý provenance bản đồ
   và bảo vệ URL/quyền truy cập trước public. Chưa có ảnh địa phương để làm việc này.

Đây là việc tiếp theo đề xuất, không phải thông báo đã triển khai. Nhiệm vụ gần
nhất của người dùng là hoàn tất Task 03A và báo cáo; không tự train lại hoặc tải thêm chỉ vì
thấy lệnh cũ trong tài liệu. Không tạo ảnh AI. Không xóa checkpoint/dataset.

## 6. Các file nên đọc khi tiếp tục

- `AGENTS.md`, `docs/handover/AGENT_RULES.md`, `.agents/skills/ai-waste-dev/SKILL.md`.
- `PROJECT_STATUS.md`: audit/backlog; `docs/RESUME_MULTISOURCE.md`: tiến trình lưu lại.
- `docs/MULTISOURCE_BATTERY_RESULTS.md`: báo cáo train/đánh giá đầy đủ.
- `docs/HAZARD_ROADMAP_DANANG.md`, `docs/DANANG_CAPTURE_PROTOCOL.md`.
- `backend/app/battery_screen.py`, `safety_routes.py`, `safety_guidance.py`,
  `collection_contacts.json`; `frontend/src/components/SafetyPanel.jsx`.
- `backend/app/yolo_detector.py`, `yolo_routes.py`, `yolo_detector_manifest.json`,
  `backend/tests/test_yolo_detector.py` và `logs/task02a_*`.
- `frontend/src/components/YoloDetectionPanel.jsx`, `WasteObjectCard.jsx`,
  `YoloDetectionPanel.css`, `frontend/src/utils/yoloView.mjs`, `frontend/tests/`;
  `docs/TASK02B_REPORT.md`, `logs/task02b/`.
- `scripts/teacher_student/`, `tests/test_teacher_student.py`,
  `tests/teacher_review_ui.test.mjs`, các docs Task 03A và `logs/task03a/`.
- `scripts/train_battery_multisource.py`, `prepare_multisource_waste.py`,
  `evaluate_multisource_waste.py`, `verify_battery_api.py`.
- Bằng chứng audit: `logs/project_status_*`; artifact model/dataset nằm trong
  `runs/` và `data/processed/`, không có trong một bản clone Git thuần.

**Mốc dừng:** đã train, đánh giá, tích hợp pin thử nghiệm, audit và hoàn tất
**Task 01, Task 02A — YOLO Backend, Task 02B — UI chính và Task 03A — Teacher pipeline
preparation**. Chờ duyệt Teacher/env và người duyệt pilot; không tự train/download.
Không làm lại contract/test/backend adapter hoặc box/card UI đã xong. Các route/tab
legacy riêng chưa migration; không coi kết quả YOLO là taxonomy vật liệu hay an toàn.
