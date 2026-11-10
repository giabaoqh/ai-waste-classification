# Audit dataset detector — đo mới ngày 10/11/2026

Nguồn: `data/processed/waste_multisource_v3`, giữ nguyên **1655 train / 318 val /
15 test**. Audit read-only, không huấn luyện. Chi tiết máy đọc tại
`logs/task03a/dataset_audit_final.json`, class_distribution.csv và near_duplicate_candidates.csv.

| Class ID / nhóm vật thể | Train boxes | Val boxes | Test boxes |
|---|---:|---:|---:|
| 0 bottle_jar | 2873 | 401 | 13 |
| 1 can | 318 | 64 | 7 |
| 2 cup | 89 | 13 | 5 |
| 3 carton | 408 | 71 | 5 |
| 4 cap_lid | 150 | 46 | 10 |
| 5 bag_film_wrapper | 367 | 48 | 18 |
| 6 paper | 60 | 20 | 11 |
| 7 cigarette | 233 | 89 | 27 |
| 8 other_litter | 1097 | 254 | 47 |
| Tổng | **5595** | **1006** | **143** |

Bottle_jar/paper train lệch **47.88 lần** theo box; không phải số ảnh độc lập.
WaRP chỉ đóng góp 4 nhóm theo mapping đã có; những nhóm khác phụ thuộc nguồn TACO.
Test 15 ảnh chỉ có TACO, không negative; negative val chỉ 3 ảnh. Không suy kết quả
gộp thành độ tin cậy địa phương/hazard.

## Kết quả có thể xác nhận bằng file

- 1988 processed-image hashes khớp manifest; **0 lỗi hình học/class/format**,
  0 orphan image/label TXT.
- 1988 nhãn khớp transcription/taxonomy từ annotation nguồn đã pin, kể cả empty
  annotations. 1988 raw-image hashes khớp source_sha256; 0 missing raw và 0 mismatch.
  Hash TACO annotations khớp pin; WaRP archive hash khớp audit lúc build.
- 0 duplicate group theo file SHA, decoded-pixel SHA, source SHA hoặc source ID;
  0 cross-split collision theo các identity này và capture_date/TACO batch hiện có.
  Đây là **không phát hiện qua kiểm đã chạy**, không chứng minh hết leakage theo
  scene/vật thể vì thiếu scene IDs và cùng camera/background.
- dHash≤8 có **3170 cặp nghi tương tự**: 3169 train/train, 1 train/val; không tự
  xóa/gộp. Cặp cross-split distance8:
  `images/train/warp_prepared_data_all_MGS-28-Oct_01-05-19.jpg` và
  `images/val/warp_prepared_data_all_MGS_19-Oct_22-45-31.jpg`.
  Chưa xác nhận là cùng ảnh/vật thể; mẫu train đã loại khỏi pilot, val giữ nguyên.
- **39 ảnh** có box cạnh<3 px hoặc area<16 px²: cần review khả năng nhìn thấy,
  tọa độ/resize và định nghĩa nhãn; không tự coi tiny box là sai.
- **1 ảnh** có cặp box khác class IoU≥0.95: `images/train/hftaco_460.jpg`,
  box indices 4/8 (0-based, tương ứng G5/G9 trong UI), IoU **0.96378**.
  Đây là conflict candidate, chưa kết luận nhãn sai.
- **58 ảnh nhãn trống** (55 train,3val), đều Street-garbage: zero annotations
  không chứng minh không có rác; chưa được người xác minh, scene/group unknown.
- **346 ảnh TACO** không có license reference; 144 ghi “CC”,240 ghi ODBL
  OpenLitterMap & Contributors. Không coi các chuỗi này là xác minh quyền ảnh đầy đủ.
  Nguồn WaRP khai báo CC0, Street khai báo MIT; vẫn cần kiểm nguồn/quyền từng case.
- **1988/1988** metadata human_review=pending. Chưa có Teacher, nên **chưa đo
  semantic mislabel hoặc số vật thể bỏ sót thực**. Khớp annotation nguồn không
  khắc phục được nhãn nguồn thiếu/sai.

## Pilot đã tạo

`data/teacher_student/task03a_pilot150`: **150 train-only**, seed42; HF TACO80,
WaRP55, Street-negative15; **1378 box gốc**, phủ9nhóm. Số ảnh có class tương ứng:
bottle85/can45/cup24/carton47/cap37/bag48/paper21/cigarette31/other112; một ảnh
có thể chứa nhiều class nên không cộng các số này thành số ảnh.

Ưu tiên rare class, scene diversity, crowding/tiny-box heuristics; đây không phải
random đại diện population hoặc lỗi Teacher đã xác nhận. Mẫu near-cross-split
đã quarantine khỏi pilot. Eval SHA intersection=0,0ảnh val/test được chọn.
Real pilot **150 pending / 0 Teacher / 0 approved**; export thực tế bị chặn (exit2,
không tạo output). Browser sửa nháp chỉ trên workspace test_only riêng.

Fingerprint nguồn trước/sau:
`039e46d6b73cbbc18479f1d667af829348110c7e199b695dcfc3be245025654e`.
Xác nhận snapshot/source app/weights/database: `logs/task03a/final_verification.json`.
