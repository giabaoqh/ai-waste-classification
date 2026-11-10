# Báo cáo train và mở rộng Đà Nẵng — 10/11/2026

Đã hoàn tất train và đo trên máy RTX 4050. **Các model mới còn ở mức thử nghiệm;
chưa đủ căn cứ dùng tự động để quyết định xử lý chất thải nguy hại.** Không thay
checkpoint MobileNet cũ. Không tạo ảnh AI, không commit/push.

Phát hiện chính: pin v1 accuracy nội bộ 95,59% nhưng recall ngoài miền RecyBat
0%. Vì vậy đã train v2 đa nguồn; holdout phát triển v2 đạt precision
99.83%, recall 98.13%. Chưa có kiểm định
độc lập tại Đà Nẵng; không gọi các số này là độ chính xác sử dụng ngoài thực tế.

## 1. Detector rác đa nguồn

- YOLO26n, seed 42, ảnh 640, batch 8, workers 0; thực chạy **30 epoch**.
- Thời gian train: **23.03 phút**. Best model chọn bằng
  validation; test chỉ dùng báo cáo sau khi khóa checkpoint.
- Dữ liệu: **1.655 train / 318 validation / 15 test**. WaRP: 1.000/200; HF TACO:
  600/100; Street negative: 55/3; giữ 15 val + 15 test TACO cũ. TACO mirror không
  phải nguồn độc lập với TACO gốc. 522 ảnh official test WaRP chưa dùng.
- Hash/dHash chống trùng là heuristic; duyệt nhãn thủ công vẫn pending. Tách WaRP
  theo ngày chụp và TACO theo batch nguồn; không có bộ test địa phương Đà Nẵng.

So sánh hai checkpoint trên **cùng** từng tập, không so các aggregate khác nhau:

| Tập | Pilot cũ mAP50 | Mới mAP50 | Thay đổi | Mới mAP50–95 | Mới recall |
|---|---:|---:|---:|---:|---:|
| TACO validation cũ (15 ảnh) | 10.30% | 25.74% | +15.45 điểm % | 17.95% | 23.20% |
| TACO mirror HF (100 ảnh) | 5.79% | 26.86% | +21.07 điểm % | 20.34% | 27.61% |
| WaRP validation (200 ảnh) | 2.26% | 33.41% | +31.15 điểm % | 23.82% | 36.94% |
| Validation hỗn hợp (318 ảnh) | 2.87% | 26.04% | +23.17 điểm % | 18.77% | 27.87% |
| Test TACO giữ riêng (15 ảnh) | 7.12% | 16.66% | +9.54 điểm % | 10.15% | 21.31% |

mAP là metric phát hiện box, không phải accuracy phân loại bảy vật liệu.
Test chỉ 15 ảnh nên chưa đại diện ngoài thực tế. Negative diagnostic: model mới
có detection ở 0/3 ảnh (conf 0,25);
cỡ mẫu này không đủ kết luận false-positive rate.

Từng nhóm trên validation 318 ảnh:

| Nhóm vật thể | Precision | Recall | AP50 |
|---|---:|---:|---:|
| bottle_jar | 60.52% | 62.59% | 64.37% |
| can | 34.46% | 25.47% | 21.49% |
| cup | 44.77% | 30.77% | 36.27% |
| carton | 21.75% | 39.44% | 22.69% |
| cap_lid | 43.08% | 23.91% | 24.86% |
| bag_film_wrapper | 30.25% | 50.00% | 38.98% |
| paper | 65.51% | 5.00% | 7.04% |
| cigarette | 100.00% | 2.12% | 9.61% |
| other_litter | 31.49% | 10.24% | 9.25% |

Từng lớp lấy từ lượt validation sau train (batch 8); bảng so sánh domain dùng
batch 4. Sai khác số học nhỏ giữa hai lượt được giữ nguyên trong JSON nguồn.

Thời gian suy luận thử nghiệm GPU: median 15.83 ms,
p95 16.73 ms, 30 lần batch 1 gồm đọc ảnh, sau warmup.
Đây không phải benchmark thiết bị đích. Chưa tích hợp detector này thay luồng
MobileNet/SSDLite sản phẩm; chín nhóm không bao gồm pin hay nhãn medical.

## 2. Sàng lọc hình dạng pin

YOLO26n-cls hai nhãn `battery`, `not_battery`, **17 epoch**,
224 px, batch 16. Nguồn Kaggle Garbage Classification. Sau lọc gần trùng:
1202 train / 256 validation /
295 test. Không có ID vật/cảnh để bảo đảm tách theo vật thật.

Ngưỡng **0.95** chọn từ validation, không chọn theo test. Mục tiêu
validation recall ≥90%, precision ≥80%: **đạt trong miền**.

| Tập | TP | FP | FN | TN | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 109 | 0 | 8 | 139 | 100.00% | 93.16% | 96.88% |
| Test nội bộ | 113 | 2 | 11 | 169 | 98.26% | 91.13% | 95.59% |

Kiểm tra ngoài miền với RecyBat24 v3 **chỉ có ảnh pin**: nhận dương
0/2834, bỏ sót ở mức ảnh
2834/2834; recall ảnh **0.00%**.
Không có ảnh negative nên **không đo được precision, specificity hoặc FPR** ở đây.
Nhiều ảnh là các mặt/ánh sáng của cùng viên pin, không coi là các vật độc lập.
Không tinh chỉnh model/ngưỡng theo tập này. Chưa kiểm tra pin ẩn hoặc ảnh Đà Nẵng.

Đã thêm API và tab **Pin và rác có nguy cơ** để dùng classifier thật trên ảnh cận
cảnh một vật. Không có bounding box pin; không xác định hóa học, hư hỏng, nguy cơ
cháy, nhiễm bệnh hoặc bức xạ. Score chưa hiệu chỉnh; âm tính không xác nhận an toàn.

### Bản cải tiến v2 sau thất bại ngoài miền

V1 bị loại khỏi runtime mặc định sau kết quả ngoài miền 0%; không hạ ngưỡng theo
tập chẩn đoán. V2 train lại từ pretrained với Kaggle + RecyBat, bổ sung negative
Kaggle chưa dùng vào train và cân bằng hai nhãn trong train. RecyBat tách theo
**110 ID viên pin (kèm dạng pin)**, gộp tất cả mặt/ánh sáng của cùng viên vào một
split, tối đa tám ảnh mỗi viên ở train. Các split Kaggle cũ được giữ nguyên.

V2: 20 epoch; 2404 train /
646 validation / 760 development
holdout. Ngưỡng 0.95 chọn trên validation với mục tiêu recall ≥95%,
precision ≥90%: **đạt trong tập phát triển**.

Mục tiêu này tính trên tập gộp. **Chưa đạt recall ≥95% cho từng nguồn**:
Kaggle validation chỉ 91.45%.
Không dùng nhiều ảnh pin RecyBat để che bỏ sót pin Kaggle.

| Tập v2 | TP | FP | FN | TN | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 497 | 0 | 10 | 139 | 100.00% | 98.03% | 98.45% |
| Development holdout | 578 | 1 | 11 | 170 | 99.83% | 98.13% | 98.42% |

Trên phần Kaggle của holdout v2: precision 99.12%,
recall 91.13%. Trên ảnh pin RecyBat
holdout v2: recall 100.00%; phần nguồn này
chỉ có positive nên precision ở đây không dùng kết luận năng lực báo nhầm.

**Đây là kiểm tra phát triển, chưa phải kiểm định ngoài miền độc lập**: RecyBat
đã được xem trong chẩn đoán v1 rồi được chọn làm nguồn phát triển v2. Không dùng
sự cải thiện để tuyên bố hiệu quả trên ảnh Đà Nẵng. Vẫn thiếu negative băng chuyền
độc lập, ID vật Kaggle và bộ ảnh địa phương. Trạng thái runtime: `experimental_development_only`;
chỉ phục vụ thử nghiệm có giám sát khi vượt mục tiêu validation.

## 3. Hướng dẫn xử lý và địa điểm

- Hướng dẫn có nguồn, tách lựa chọn của người dùng với kết quả thị giác. Pin nóng/
  bốc khói không được hướng dẫn đóng gói/vận chuyển tới điểm pin công cộng.
- Kim tiêm/rác y tế: cảnh báo phòng ngừa và liên hệ chuyên trách; chưa train model
  nhận diện medical mới. Không chẩn đoán bệnh cụ thể từ ảnh.
- Nghi phóng xạ: giữ tránh xa, báo cơ quan chức năng; cần thiết bị đo/chuyên môn.
  Không đo hay loại trừ bức xạ từ RGB, không có hướng dẫn xử lý phóng xạ tại nhà.
- Hướng dẫn riêng cho giấy, nhựa, lon/kim loại, chai lọ, hữu cơ, bao bì hỗn hợp.
- FEC từng công bố 108 Phan Châu Trinh và 808 Ngô Quyền, Đà Nẵng. Cả hai ghi
  **cần xác nhận hiện trạng**, không có bảo đảm đang nhận, giờ mở cửa, pin hỏng
  hoặc rác y tế/phóng xạ. Danh mục địa điểm cũ cũng ghi chưa xác minh.

Nguồn và kế hoạch nghiệm thu chi tiết: [Lộ trình Đà Nẵng](HAZARD_ROADMAP_DANANG.md).

## 4. Kiểm thử phần mềm và cách chạy

**65 test độc lập đã qua**, thêm 4 subtest dữ liệu. Build frontend và lint
trả mã 0; lint còn cảnh báo cũ. `pip check` không có xung đột.

Kết quả từng lệnh, cảnh báo lint và lỗi suite đầy đủ được lưu nguyên văn trong
`MODEL_SCORECARD.json` và thư mục `verification/`. Không đổi các lỗi cũ thành pass.
Suite đầy đủ có lỗi kỳ vọng health `ok` dù thiếu MobileNet, yêu cầu HTTP 400 thay
503 khi model chưa sẵn sàng, và so khớp nguyên văn disclaimer. Xem log cuối để
biết số pass/fail thực tế; các benchmark cần checkpoint/ảnh local còn hạn chế.

Chạy từ `D:\ai-waste-classification`:

```powershell
backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
npm.cmd --prefix frontend run dev -- --host 127.0.0.1
```

Mở http://127.0.0.1:5173, chọn tab **Pin và rác có nguy cơ**. Health MobileNet
vẫn `degraded` do thiếu `mobilenetv3_best.pth`; dùng `/api/safety/capabilities` và
thử suy luận thực để kiểm tra classifier pin. Hai năng lực được báo riêng.

Artifact đã chép vào `waste_detector/` và `battery_screen/`, gồm weights, báo cáo,
môi trường và hash. Dữ liệu/weights vẫn ngoài Git. Để dùng bản sao pin, đặt
`BATTERY_SCREEN_RUN` tới thư mục `battery_screen` có report và `weights/best.pt`.

## 5. Bước tiếp theo

Ưu tiên duyệt nhãn và thu thập test Đà Nẵng, negative khó, pin cúc áo/che khuất,
ảnh nhiều vật; sau đó train detector pin từ box gốc RecyBat + ảnh địa phương.
Không tiếp tục tăng dữ liệu không tương thích. SAM có thể hỗ trợ gán nhãn sau
khi người duyệt xác nhận, chưa được chạy trong lượt này. Khi mở rộng Bắc/Nam,
phải có test theo miền và registry điểm nhận được xác minh.

**Chưa đạt nghiệm thu dùng không giám sát.** Cần cải thiện recall, kiểm định
ngoài miền, chuyên gia duyệt hướng dẫn và xác nhận điểm xử lý trước triển khai.
