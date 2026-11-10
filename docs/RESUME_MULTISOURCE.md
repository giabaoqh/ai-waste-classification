# Tiến trình mở rộng dữ liệu và train YOLO

Trạng thái hiện tại: **ĐÃ HOÀN TẤT LƯỢT RESUME ngày 10/11/2026**.
Quyết định mới nhất: **không khôi phục MobileNet**, ưu tiên hướng YOLO + pin.
Đọc `CHATGPT_HANDOVER.md` ở root để tiếp nối đúng mục tiêu và mốc công việc.
Dataset v3 đã hoàn tất (1.655 train, 318 val, 15 test); không chạy lại lệnh dựng v3.
Lượt `yolo26n_multisource30_v1` đã hoàn tất 30 epoch và so sánh domain/test.
Classifier pin `yolo26n_cls_v1` hoàn tất 17 epoch nhưng thất bại ngoài miền
(0/2.834 ảnh RecyBat ở ngưỡng 0,95); đối chiếu CPU/GPU xác nhận không phải lỗi
pipeline tên tệp/kênh ảnh. Bản cải tiến `yolo26n_cls_multisource_v2` hoàn tất
20 epoch, Kaggle + RecyBat tách theo dạng pin + ID viên. Holdout phát triển:
precision 99,83%, recall 98,13%; Kaggle riêng recall vẫn 91,13%. Không dùng
RecyBat làm phép đo ngoài miền độc lập cho v2 nữa. Không còn lượt train/download
chạy hoặc được lên lịch. Không chạy lại các lệnh lịch sử bên dưới.

Đã tải WaRP, Kaggle Garbage Classification, HF TACO/negative Street, RecyBat bản
cũ (record 13126689) và bản gốc v3 (15226091, 263.476.594 bytes). Hash archive
RecyBat đều khớp metadata nguồn. Bản cũ có augmentation; không trộn hai phiên bản.
Pin v1 dùng nhãn image-level Kaggle; RecyBat v3 dùng chẩn đoán v1 rồi phát triển
v2. Người dùng xác nhận chưa có ảnh Đà Nẵng. Pin v2 là classifier một vật thử nghiệm,
không phải detector pin/medical hay phép đo nhiễm bệnh/bức xạ.
Hướng dẫn nguy cơ/API/UI và kế hoạch Đà Nẵng: `docs/HAZARD_ROADMAP_DANANG.md`.
Kết quả đầy đủ: `docs/MULTISOURCE_BATTERY_RESULTS.md`. Audit và việc tiếp theo:
`PROJECT_STATUS.md`. API/frontend cục bộ đang mở; MobileNet cũ vẫn thiếu checkpoint.

## Ảnh chụp lúc tạm dừng trước đây — chỉ là lịch sử

Các phần bên dưới mô tả lần dừng để chờ quota, không phải trạng thái hiện tại.

## Công việc đã hoàn thành

- Repo: `D:\ai-waste-classification`, nhánh main, nhiều thay đổi local chưa commit.
  Không commit/push hoặc đổi nhánh khi tiếp tục nếu chưa được yêu cầu.
- Pilot cũ 100 ảnh / 9 nhóm: cả YOLO26n và YOLO11n đã train 10 epoch.
  Kết quả và weights nằm trong `runs/waste_pilot/` và thư mục outputs của chat.
- Môi trường GPU riêng: `experiments/yolo/.venv`; CUDA hoạt động.
- Đã tải đầy đủ Kaggle WaRP: `data/raw/warp_kaggle/warp.zip`, 844.977.310 bytes.
  Có detection, classification và segmentation; chỉ dùng phần `Warp-D/train`.
  522 ảnh official test của WaRP vẫn để riêng, không dùng train/validation.
- Đã kiểm tra Hugging Face Street-garbage: train có 55 ảnh không annotation,
  valid có 3 ảnh không annotation. Nhãn positive chỉ là garbage tổng quát nên
  chỉ lấy các ảnh negative vào bộ 9 nhóm. Phần tải ảnh negative chưa được chạy.
- Đã thêm script chuẩn bị dữ liệu và script đánh giá theo domain.
- 9 unit test dữ liệu và 4 subtest đã qua; `pip check` không có xung đột.

## Trạng thái chính xác lúc dừng

Đã dừng tiến trình Python chuẩn bị dữ liệu. **Chưa bắt đầu lượt train dữ liệu
mở rộng**, chưa có checkpoint hay metrics mới từ Kaggle/Hugging Face.

`data/processed/waste_multisource_v2` là thư mục **chưa hoàn tất**:

- train: 1.251 ảnh = 1.000 WaRP + 251 TACO từ HF đã được chấp nhận.
- val: 315 ảnh = 200 WaRP + 100 TACO HF + 15 validation cũ giữ nguyên.
- test: 15 ảnh test pilot cũ giữ nguyên.
- Script đang tải TACO HF, mục tiêu train HF là 600 ảnh. Chưa chạy phần negative
  Street-garbage và chưa ghi `manifest.json`, `audit.json`, `dataset.yaml` cuối.
- Cache `data/raw/hf_multisource`: 366 file, khoảng 1.140 MB lúc dừng.
  Có thể chứa file tải xong nhưng chưa được thêm vào dataset; cache không đồng
  nghĩa số ảnh đã duyệt/chấp nhận. Giữ cả file `.part` nếu có để truy vết.
- `logs/prepare_multisource_v2.log` chứa log đến mốc HF train 251/600.
- `waste_multisource_v1` là một lần chuẩn bị dở trước đó, gặp dòng label trống;
  lỗi đã được sửa. Không dùng v1/v2 dở dang để train.

## Nguồn và phiên bản

1. Kaggle `parohod/warp-waste-recycling-plant-dataset`.
   File zip đã nằm trên ổ D; không cần tải lại. Nhãn 28 lớp được gộp có ghi mapping:
   bottle-/glass- → bottle_jar; cans → can; cardboard → carton; còn lại other_litter.
   Canister không phải can. Dataset khai báo CC0 trên Kaggle; vẫn giữ provenance.
2. HF `pictograph/taco`, commit `bfac5b10c53ae66922a252bd00a4eacb95a43591`.
   Dùng ảnh mirror nhưng annotation từ bản TACO gốc đã pin; không tin class names
   trên card mirror nếu chúng không khớp. Đây vẫn là TACO, không phải nguồn độc lập.
3. HF `Shynuaa/Street-garbage`, commit `9d80bca6a178b53864737a8967faa350cafba4f3`.
   Annotation JSON đã lưu ở `data/raw/hf_street/`. Card khai báo MIT. Chỉ dùng
   cảnh tác giả gán không có garbage; việc duyệt lại bằng người vẫn pending.

## Cách tiếp tục mà không mất dữ liệu đã tải

Chạy từ root repo. Script `prepare_multisource_waste.py` tái sử dụng ảnh trong
cache, nhưng từ chối output đã có nội dung. Vì v2 dở dang, dựng phiên bản mới v3:

```powershell
experiments\yolo\.venv\Scripts\python.exe scripts\prepare_multisource_waste.py --output data\processed\waste_multisource_v3
```

Lệnh này dựng lại file processed từ cache; không cần tải lại zip WaRP hoặc ảnh
HF đã được lưu hoàn chỉnh. Bản raw được giữ. Không xóa v1/v2 hoặc cache để tiết
kiệm dung lượng khi chưa cần và chưa kiểm tra đường dẫn.

Sau khi báo `DATASET_COMPLETE`, cần kiểm tra:

1. Tồn tại đủ dataset.yaml, manifest.json, audit.json, mapping và nhãn YOLO.
2. Hash ảnh, box hợp lệ, số lượng theo nguồn/split, số ảnh bị loại, mapping 9 lớp.
3. Không có ảnh train từ các batch TACO đã dành cho val/test; WaRP tách theo ngày
   chụp; negative HF giữ split tác giả. Hash/dHash chỉ là heuristic chống trùng.
4. Duyệt nhãn của người vẫn pending; không mô tả bộ này là đã QA thủ công đầy đủ.
5. Kiểm tra RAM trống; lần trước khoảng 3,2 GiB. Đề xuất workers=0, cache=False,
   batch=8 nếu chạy ổn; giảm batch=4 nếu OOM. Không chạy hai model trên GPU cùng lúc.

Lượt huấn luyện dự kiến, chưa chạy:

```powershell
experiments\yolo\.venv\Scripts\python.exe scripts\train_yolo_pilot.py --data data\processed\waste_multisource_v3\dataset.yaml --model yolo26n.pt --epochs 30 --batch 8 --workers 0 --name yolo26n_multisource30_v1
```

Script có early stopping patience 10, nạp pretrained chính thức, không ghi đè
checkpoint MobileNet của backend. Kiểm tra log/args/results.csv để báo số epoch
thực. Theo dõi tới khi run hoàn tất nếu người dùng yêu cầu tiếp tục train.

## Đánh giá và báo cáo còn phải làm

- `scripts/evaluate_multisource_waste.py` hiện còn hardcode dataset v2. **Trước khi
  gọi phải bổ sung tham số dataset hoặc đổi sang v3**, vì v2 chưa hoàn tất.
- Đánh giá model mới và model pilot cũ trên cùng tập val mới; báo riêng WaRP,
  TACO HF và tập val cũ 15 ảnh. Không so aggregate mới với 10,30% cũ như cùng test.
- 3 negative validation của Street-garbage chỉ là chẩn đoán, chưa đủ đo FPR đáng tin.
- Không mở tập test để tinh chỉnh; không công bố năng lực medical/polymer hoặc
  quyết định xử lý rác từ taxonomy 9 nhóm vật thể.
- Lưu checkpoint, hash, môi trường, nguồn dữ liệu, kết quả thực và hướng dẫn chạy
  vào outputs. Không tạo ảnh AI; người dùng đã yêu cầu tiết kiệm token.

## File cần đọc khi quay lại

- `AGENTS.md`, `docs/handover/AGENT_RULES.md`, `.agents/skills/ai-waste-dev/SKILL.md`.
- `scripts/prepare_multisource_waste.py` — tạo dữ liệu, tải có cache, gộp nhãn.
- `scripts/train_yolo_pilot.py` — đã thêm `--workers`, giới hạn tối đa 30 epoch.
- `scripts/evaluate_multisource_waste.py` — đánh giá theo domain, cần sửa path v3.
- `tests/test_multisource_waste.py`, `tests/test_taco_pilot.py`.
- `docs/YOLO_PILOT_RESULTS.md` — kết quả cũ, chưa phải train mở rộng.

Không có tiến trình train được lên lịch, không có automation tự chạy lại.
