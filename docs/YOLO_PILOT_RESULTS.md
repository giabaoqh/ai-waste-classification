# Kết quả pilot YOLO trên TACO

Chạy trên RTX 4050 Laptop 6 GB, ngày 10/11/2026. Đây là kết quả đo thực tế của pilot kỹ thuật; chưa phải model đủ chất lượng để triển khai.

## Dữ liệu

- Dataset: `D:/ai-waste-classification/data/processed/taco_pilot_v3`.
- 100 ảnh, chia 70 train / 15 validation / 15 test; 9 nhóm vật thể.
- Số annotation: 771 train / 182 validation / 143 test, tổng 1.096.
- Annotation gốc TACO; mapping nhóm, EXIF, chuẩn hóa JPEG và kiểm tra hash được lưu lại.
- Giữ batch nguồn riêng từng split; dHash chỉ là kiểm tra gần trùng theo heuristic.
- Chọn mẫu ưu tiên phủ nhóm nhãn, chưa đại diện phân bố thực tế.
- Duyệt nhãn theo định nghĩa dự án còn pending; thiếu tập negative độc lập và ảnh Việt Nam.
- Test được giữ riêng, chưa dùng để đánh giá hay chọn model.

## So sánh validation

Cùng 15 ảnh / 182 annotation, 10 epoch, batch 4, input 640, seed 42, AdamW. Mỗi model chọn checkpoint tốt nhất bằng validation. Recipe chung chưa phải hyperparameter tối ưu từng model.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Train (giây) | Peak PyTorch allocation (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| yolo26n.pt | 10.30% | 7.52% | 36.15% | 15.85% | 79.1 | 741.8 |
| yolo11n.pt | 9.37% | 6.21% | 24.32% | 11.25% | 68.3 | 670.2 |

Đây là mAP của detection, không phải accuracy. Precision/recall tổng hợp theo evaluator không phải phép đo an toàn ở một ngưỡng production đã chốt. Chênh lệch hai model trên 15 ảnh và một seed chưa đủ kết luận hơn kém thống kê.

## Kết quả theo nhóm

| Nhóm | Instance val | YOLO26n AP50 | YOLO11n AP50 |
|---|---:|---:|---:|
| bottle_jar | 18 | 7.03% | 23.26% |
| can | 18 | 59.12% | 10.74% |
| cup | 7 | 8.20% | 5.92% |
| carton | 6 | 9.28% | 7.50% |
| cap_lid | 21 | 0.00% | 4.20% |
| bag_film_wrapper | 10 | 7.35% | 25.36% |
| paper | 12 | 0.00% | 2.97% |
| cigarette | 31 | 0.05% | 0.36% |
| other_litter | 59 | 1.64% | 4.04% |

## Tốc độ chẩn đoán

| Model | Median predict | p95 predict |
|---|---:|---:|
| yolo26n.pt | 15.58 ms | 16.67 ms |
| yolo11n.pt | 14.68 ms | 15.53 ms |

30 lượt đo sau 5 lượt warm-up, batch 1, đọc ảnh validation và gọi predict mặc định FP32. Chưa phải benchmark nghiệm thu 200 lượt; không gồm upload mạng, OCR, Grad-CAM hoặc điều khiển thiết bị. Peak memory là allocation PyTorch, không phải tổng VRAM của hệ thống.

## Kiểm chứng và giới hạn

- Smoke test một epoch trên bản v3 hoàn tất; hash của cả 100 ảnh giữ nguyên sau training.
- Hai model chính đều chạy đủ 10 epoch; checkpoint tải lại, hash và manifest được kiểm tra.
- 7 kiểm thử mới cho pipeline dữ liệu và 29 regression unit test hiện có đã qua: 36 test, thêm 4 subtest.
- Đã ghi prediction thực trên 15 ảnh val của YOLO26n tại conf=0.25; trả trạng thái REVIEW vì model pilot chưa được duyệt. Không gán nhãn xử lý rác hoặc vật liệu từ các kết quả này.
- Script mới không sửa backend/UI. Bộ test backend đầy đủ vẫn cần môi trường API, checkpoint và dữ liệu như ghi trong tài liệu thiết lập.
- Chưa đạt mục tiêu dự thảo mAP50 80% hoặc mAP50:95 50%; chưa đo false-positive rate trên tập negative.
- Chưa chứng minh năng lực nhận diện medical, polymer, nhiễm bẩn hay quyết định tái chế.
- Không thay `mobilenetv3_best.pth`, không commit/push, không mở dịch vụ cloud. Backend phân loại cũ vẫn cần checkpoint riêng.

## Chạy lại và dùng checkpoint

Xem `D:/ai-waste-classification/docs/YOLO_PILOT.md`. Môi trường: `experiments/yolo/.venv`. Mỗi lần chạy lại cần tên run mới.

Hai checkpoint đính kèm chỉ dùng với YOLO và taxonomy 9 nhóm TACO, không dùng làm weights MobileNet của backend. Chi tiết config, epoch log, environment và weights gốc nằm trong `runs/waste_pilot/<run>/`.

## Việc tiếp theo dựa trên kết quả

1. Duyệt 100 ảnh pilot và mapping; kiểm tra đặc biệt giấy, đầu lọc thuốc lá, nắp nhỏ và nhãn other_litter.
2. Mở rộng dataset có nguồn lên quy mô lớn hơn, bổ sung ảnh Việt Nam và 300 ảnh negative; giữ test độc lập.
3. Sau QA mới tăng số epoch/ngân sách và chạy nhiều seed, dùng val để chọn ngưỡng; chưa suy ra model lớn hơn sẽ tự giải quyết thiếu dữ liệu.
4. Tích hợp adapter/Decision Engine sau khi recall và negative test đạt tiêu chí.

Nguồn annotation: [TACO](https://github.com/pedropro/TACO/tree/70479ad9adc3da971a8ae75ac97eb1026910eec7).
