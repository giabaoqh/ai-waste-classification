# Task 03A — Teacher–Student pipeline, 10/11/2026

**Đã chuẩn bị pipeline và pilot; chưa chạy Teacher hoặc train Student.**

1. **Chất lượng dataset:** audit 1.988 ảnh, giữ 1.655/318/15 ảnh train/val/test;
   có 5.595/1.006/143 box. Không phát hiện lỗi cấu trúc hoặc exact-identity
   leakage; 1.988 labels và raw hashes khớp nguồn. Bottle/paper lệch 47,88 lần;
   39 ảnh có tiny box, 1 ảnh nghi box/class xung đột, 58 negative chưa xác minh,
   346 ảnh thiếu license reference. Có 1 cặp train–val dHash distance 8, chưa
   kết luận leakage; mẫu train đã loại khỏi pilot. Chưa đo nhãn sai ngữ nghĩa
   hoặc vật bị bỏ sót khi chưa có Teacher/người duyệt. Chi tiết:
   `DATASET_AUDIT_TASK03A.md`, `logs/task03a/dataset_audit_final.json`.

2. **Teacher đề xuất:** Grounding DINO-T cho box; VLM hỗ trợ ngữ nghĩa tùy chọn,
   SAM 2 chỉ khi cần mask. Accuracy trên 9 nhóm chưa biết; không dùng COCO AP
   làm accuracy dự án. Hai venv hiện thiếu Teacher/deps, chưa có inference thật.
   [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
   [SAM 2](https://github.com/facebookresearch/sam2),
   [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).

3. **Pipeline đã xây:** `scripts/teacher_student/` có audit, pilot, import Teacher
   JSON, QA/review UI và approved-only export. So class/IoU/score; sửa nhãn,
   tọa độ, vẽ/xóa box, mark issue. Unknown giữ trong draft/hold, không ép mapping.
   Human checklist, căn cứ quyền ảnh, issue resolution, revision/HMAC và split/hash
   gates bắt buộc. Không đổi app/backend/checkpoint. Pilot 150 ảnh train-only:
   TACO 80, WaRP 55, Street 15; 1.378 box gốc, **0 Teacher / 0 approved**.
   Có adapter DINO offline/local-only; thiếu artifact bị chặn, chưa smoke model thật.

4. **Kiểm thử thực tế:** **45 test pipeline PASS** trong full suite
   **217 PASS / 0 FAIL / 20 SKIP + 4 subtests PASS**; skips vì thiếu artifact legacy.
   JS geometry **2 PASS**, lint tool exit 0. Browser đã sửa/lưu/reload, vẽ/xóa/reset,
   layers, dirty guard, chặn box ngoài ảnh và approval workspace test. Real export
   zero-approved bị chặn, không tạo dataset. Snapshot, weights, app source và DB
   giữ nguyên. Chưa smoke Teacher thật/cloud, chưa kiểm overlay bằng output model
   Teacher thật; chưa Safari/Firefox/điện thoại thật hoặc nghiệm thu bảo mật public.

5. **Tài nguyên/chi phí:** RTX 4050 có 6.141 MiB. Cần duyệt môi trường Teacher
   riêng và weights Grounding DINO khoảng 689 MB, revision/hash đã ghi;
   VRAM/latency chưa đo. Qwen 4B có khoảng 8,88 GB weights, vượt 6 GB trước
   activations. API tùy chọn cho 150 ảnh: Pro khoảng $2,40–4,20; Flash-Lite
   $0,30–0,53 theo giả định token, **chưa phát sinh chi phí**. Cần duyệt gửi ảnh
   và budget; nguồn/công thức trong `TEACHER_RESEARCH.md` và
   [bảng giá chính thức](https://ai.google.dev/gemini-api/docs/pricing).

6. **Điều kiện train:** hiện **chưa đủ** vì 0 approved. Cần người duyệt toàn ảnh,
   provenance/rights; giải quyết leakage/nguy cơ/issues, export hoàn chỉnh,
   fingerprint còn đúng, khóa config/baseline/evaluation và có yêu cầu train riêng.
   Kế hoạch mAP50, mAP50-95, P/R từng class/source và hazard FN nằm trong
   `TEACHER_STUDENT_PIPELINE.md`. Không có GT hazard thì không tính hazard recall.
   Taxonomy mở rộng mới là thiết kế, chưa sửa 9 class đang phục vụ.

7. **Task tiếp theo:** phê duyệt Teacher local artifact/env, smoke 5–10 ảnh train
   để đo tài nguyên và kiểm output; sau đó người duyệt pilot, đo lợi ích/chi phí
   gán nhãn trước quyết định train. Không tự train/download từ báo cáo này.

Review thật: `http://127.0.0.1:8013` (giữ mở); app 5173/API 8000 giữ nguyên.
Server UI test 8014 được dọn sau test. Bằng chứng `logs/task03a/`; hướng dẫn
`TEACHER_STUDENT_PIPELINE.md`, prompt protocol `TEACHER_PROMPT.md`.

File mới: `scripts/teacher_student/`, `tests/test_teacher_student.py`,
`tests/teacher_review_ui.test.mjs` và các tài liệu Task 03A.
Đã cập nhật `PROJECT_STATUS.md`, `CHATGPT_HANDOVER.md`, `docs/LOCAL_SETUP.md`.
