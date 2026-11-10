# Task 02B — YOLO visual detection & waste information

Ngày 10/11/2026. Hoàn tất ở phạm vi **UI thử nghiệm cục bộ**, chưa đủ điều kiện public.

## Đã triển khai

- Tab chính Phân Loại Rác gọi `POST /api/yolo/detect`; không gọi legacy predict/detect.
- Preview cùng ảnh JPEG đã chuẩn hóa EXIF gửi đến API, không crop/resize nội dung;
  SVG dùng đúng `image_width`, `image_height`, `xyxy_pixels`. Kiểm contract trước khi vẽ.
- Vẽ mọi detection trả về: số thứ tự, tên tiếng Việt từ API, confidence, màu riêng.
  Click hình chữ nhật/nhãn box hoặc Enter/Space chọn và focus card; click card
  highlight/focus box. Kết quả nằm dưới ảnh, mỗi detection có card riêng.
- Hiển thị class, mô tả nhóm vật thể, confidence thấp, bất định thử nghiệm,
  version model và ngưỡng thực tế. Score không phải xác suất đã hiệu chỉnh.
- Vật liệu/xử lý mặc định **Chưa xác định**. Người dùng khai báo loại/vật liệu/
  tình trạng pin rồi lấy hướng dẫn hiện có qua `/api/safety/assess`, có nguồn.
  Ghi rõ thông tin do người dùng cung cấp, chưa xác minh. Không tự ghép taxonomy vật liệu.
- Hướng dẫn pin, y tế/sắc nhọn, điện tử, nghi phóng xạ là luồng khai báo riêng.
  Classifier pin không tự chạy trên ảnh nhiều vật; nút dẫn sang tab ảnh một vật.
- Có loading, đổi/xóa ảnh, hủy, timeout 30 giây, bỏ kết quả request cũ,
  lỗi API/model và trạng thái không có detection. Không kết luận ảnh an toàn.
- Cho phép chủ động retry lỗi nạp/inference tạm thời khi artifact vẫn verified;
  giữ cảnh báo not-ready. Thiếu/hỏng artifact không được xem là ready.
- Điều chỉnh navigation xuống dòng trên màn hình hẹp; đánh dấu tab đa vật thể cũ
  là legacy và chỉ rõ tab chính đã dùng YOLO. Giữ nguyên code legacy.

## Kiểm thử thực tế

Backend `http://127.0.0.1:8000`, frontend `http://127.0.0.1:5173`, từ source local.

| Kiểm tra | Kết quả |
|---|---|
| Node frontend tests | **10 PASS / 0 FAIL / 0 SKIP**; contract, geometry, colors, error, abort/timeout, retry |
| Frontend build | PASS |
| Frontend lint | Exit 0; **45 cảnh báo cũ**, không có cảnh báo trong component/utility mới |
| Full backend regression, opt-in model | **172 PASS / 0 FAIL / 20 SKIP**, 4 subtests PASS; skip do thiếu artifact legacy |
| Git diff check | PASS |
| IAB desktop + viewport mobile 390 px | 2 box/2 card; image và SVG cùng kích thước, offset 0; sai số vị trí rect đo được <0.001 CSS px |
| Box → card / card → box | Đã click rect thật, click card; đúng selection và focus. Keyboard Enter cũng kiểm tra được |
| Một vật | `hftaco_1312.jpg`: 1 box/1 card |
| Nhiều vật + confidence thấp | `hftaco_1365.jpg`: 2 box/2 card, 30.3% bag_film_wrapper và 29.2% other_litter |
| Không detection | `warp_prepared_data_all_MGS-27-Oct_23-54-07.jpg`: 0 box/card, hiện giới hạn và không khẳng định an toàn |
| EXIF orientation 6 | Bản sao ảnh thật dưới logs: lưu 480×640, preview/API 640×480, 2 box; đã quan sát ảnh xoay và overlay |
| Hướng dẫn theo khai báo | Ordinary/paper, pin, sắc nhọn y tế, nghi y tế, nghi phóng xạ có hướng dẫn từ KB; vật liệu không tự suy từ class |
| Pin riêng | Kaggle `kg_battery532.jpg` và RecyBat `rb_train_2_4_F_2_pri.jpg`: nghi có pin, UI score 1.000; negative `kg_biological433.jpg`: score 0.037, chưa đủ bằng chứng |
| Ảnh lỗi / >10 MB | Chặn ở client, báo lỗi rõ, không còn box cũ |
| HTTP 500/503, unavailable, timeout | Đã kiểm tra trên proxy fault injection riêng, không thay model thật. Timeout 30 giây; capability unavailable khóa nút; transient error cho phép retry |
| Loading / đổi ảnh / hủy | Request delay 4 giây: chọn ảnh khác hoặc hủy, kết quả cũ không xuất hiện; normal mode phục hồi inference được |
| Console | Không có warning/error được công cụ console capture trong tab test; không xuất HAR/network trace toàn trình duyệt |

Ảnh upload được chuẩn hóa lại thành JPEG nên score có thể khác đôi chút so với
gửi byte ảnh gốc. Kết quả trên là smoke test chức năng, **không phải đánh giá accuracy**.
Mobile đo trên tab Vite phụ cùng source/proxy normal và kiểm lại trực tiếp tab
5173/8000; cả hai đều viewport 390 px. Desktop/EXIF kiểm trực tiếp cổng chính.

## Runtime và bảo toàn dữ liệu

- YOLO readiness 200, `experimental_loaded`, `production_ready=false`.
- Health 200/degraded do legacy unavailable; legacy readiness 503, không khôi phục.
- Safety capabilities 200; pin vẫn experimental. Không tự kết luận nguy cơ/an toàn từ detector.
- Hash weights YOLO và pin khớp mốc Task 02A; **không thay backend source trong Task 02B**.
- SQLite vẫn prediction_history=0, litter_reports=6 như baseline; không gọi endpoint ghi/xóa lịch sử.
- Giữ backend 8000 và frontend 5173 cho người dùng; dọn proxy 8012/Vite 5174 sau test.
- Không train/download/cài dependency/commit/push; giữ thay đổi local trước task.

## File thay đổi riêng Task 02B

- `frontend/src/App.jsx`, `frontend/package.json`.
- Mới: `frontend/src/components/YoloDetectionPanel.jsx`, `YoloDetectionPanel.css`,
  `WasteObjectCard.jsx`, `frontend/src/utils/yoloView.mjs`, `frontend/tests/yoloView.test.mjs`.
- Tài liệu: `PROJECT_STATUS.md`, `CHATGPT_HANDOVER.md`, `docs/API_CONTRACT.md`,
  `docs/TASK02B_REPORT.md`. Bằng chứng/fixtures/proxy tạm nằm `logs/task02b/` (Git ignored).
- Không đổi lockfile hoặc dependency. App.jsx chỉ sửa phần liên quan, giữ newline/code cũ khác.

## Giới hạn và việc tiếp theo

- **P1 hiện hữu:** model còn yếu/chưa có test Đà Nẵng độc lập. Các test phần mềm
  không chứng nhận model đủ tốt để triển khai hoặc tự động xử lý rác.
- **P2 còn lại:** URL/batch/webcam/Grad-CAM/history/feedback YOLO chưa có contract;
  tab legacy và analytics/XAI cũ chưa migration. Kết quả YOLO mới không ghi history.
- **P3 hiện hữu:** 45 lint warnings trong code cũ; chưa giải quyết ngoài phạm vi.
- Chưa kiểm thử Safari/Firefox/điện thoại thật, mọi EXIF orientation, ảnh >2 detection
  dày/chồng box, stress/performance. Không nghiệm thu toàn bộ map/history trong Task 02B.
- Screenshot viewport đã lưu; full-page screenshot một số lần không capture được
  bằng công cụ. Console capture trống không thay thế một bản HAR đầy đủ.
- Tiếp theo: (1) review UI/nhãn với nhóm và tạo regression browser tự động có fixture;
  (2) thiết kế history YOLO gắn model/hash/taxonomy nếu cần lưu kết quả;
  (3) thu test Đà Nẵng độc lập, rà nhãn/điểm thu gom trước quyết định train/public tiếp.
  Không tự bắt đầu train hoặc download từ đề xuất này.

## Bằng chứng để tiếp tục

`logs/task02b/PROGRESS.md`, `frontend_tests.log`, `build.log`, `lint.log`,
`backend_regression.log`, `browser_images.json`, `browser_geometry.json`,
`browser_selection.json`, `browser_guidance.json`, `browser_battery.json`,
`browser_errors.json`, `browser_retry.json`, `browser_stale_request.txt`,
`browser_console.json`, `fault_requests.jsonl`, `runtime_after.json` và screenshots.
