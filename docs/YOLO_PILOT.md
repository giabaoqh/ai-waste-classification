# Pilot YOLO trên TACO — chạy cục bộ

Pilot dùng annotation có sẵn của tác giả TACO để kiểm tra pipeline dữ liệu,
huấn luyện và validation trên RTX 4050 6 GB. Dataset đích hiện tại:
`data/processed/taco_pilot_v3`. Đây chưa phải dataset địa phương đã duyệt thủ công
hoặc model dùng cho sản phẩm. Xem kết quả đo thực tế trong `YOLO_PILOT_RESULTS.md`
sau khi các lượt chạy hoàn tất.

## Phạm vi và nhãn

9 nhóm vật thể: `bottle_jar`, `can`, `cup`, `carton`, `cap_lid`,
`bag_film_wrapper`, `paper`, `cigarette`, `other_litter`.
Mapping từng nhãn gốc được lưu trong `taxonomy.json`; không thay mapping vật liệu
7 lớp của backend. `other_litter` bao gồm những lớp hiếm và có thể nguy hại, nên
không được dùng để khuyến nghị bỏ vào thùng rác thường.

Annotation được cố định theo TACO commit
`70479ad9adc3da971a8ae75ac97eb1026910eec7` và SHA-256 trong `audit.json`.
Lấy thumbnail khi có, nếu không thì dùng ảnh gốc. Giữ nguyên bytes nguồn trong
`data/raw/taco/thumbnails/` (tên thư mục có tính lịch sử; có cả ảnh full-size).
Ảnh train là bản dẫn xuất: EXIF transpose, RGB, JPEG quality 95, không thay đổi
nội dung ảnh bằng AI. Lưu cả hash nguồn và hash bản dẫn xuất trong manifest.

Bản v3 xử lý EXIF và chuẩn hóa JPEG trước khi train để tránh Ultralytics tự sửa
file khi kiểm tra dataset. Hai bản dữ liệu kỹ thuật trước được giữ để truy vết;
so sánh hai model chính phải dùng cùng v3 và cùng manifest hash.

## Tách tập và QA

- 100 ảnh: 70 train, 15 val, 15 test; seed 42.
- Giữ các batch nguồn ở riêng một split. Batch chỉ là proxy, chưa chứng minh
  mọi cảnh hoặc vật thể trong thực tế độc lập.
- Chọn ảnh để tăng phủ nhóm nhãn; tập này thiên về ảnh có nhiều vật và chưa
  đại diện phân bố sử dụng thực tế.
- Loại trùng bằng SHA-256 và dHash; dHash không chứng minh không còn gần trùng.
- Kiểm tra ảnh giải mã được, EXIF, tỷ lệ ảnh, box hợp lệ và annotation ID.
- Nhãn gốc là annotation tác giả; việc duyệt lại theo taxonomy của dự án vẫn
  `pending`. Chưa hoàn thành bước người dùng duyệt 100 ảnh pilot.
- Ghi license từng ảnh khi nguồn cung cấp; trường license trống cần rà soát trước
  khi phân phối ảnh. Không đính kèm ảnh nguồn vào gói kết quả.
- Script huấn luyện chỉ đọc train/val; giữ test riêng, chưa dùng để chọn model.

## Môi trường riêng

Chạy từ root `D:\ai-waste-classification`. Môi trường thử nghiệm là
`experiments/yolo/.venv`, tách khỏi `backend/.venv`.

```powershell
py -3.11 -m venv experiments\yolo\.venv
experiments\yolo\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126
experiments\yolo\.venv\Scripts\python.exe -m pip install -r experiments\yolo\requirements.txt
```

Config của các script ở `logs/ultralytics_config`, integrations đồng bộ bên ngoài
được tắt khi chạy train. Toàn bộ package versions được lưu trong `environment.txt`
của từng run. Không cần tài khoản cloud hoặc dịch vụ train trả phí.

## Tạo lại dataset vào một phiên bản mới

```powershell
experiments\yolo\.venv\Scripts\python.exe scripts\prepare_taco_pilot.py --count 100 --seed 42 --output data\processed\taco_pilot_rebuild
```

Script từ chối ghi vào output đã có nội dung. URL nguồn có thể thay đổi; dùng
cache cùng hash gốc để tái lập đúng bản ảnh đã sử dụng. Seed cố định không bảo
đảm chọn cùng ảnh nếu nguồn ảnh tải được khác đi.

## Chạy thử và so sánh

```powershell
experiments\yolo\.venv\Scripts\python.exe scripts\train_yolo_pilot.py --data data\processed\taco_pilot_v3\dataset.yaml --model yolo26n.pt --epochs 1 --name yolo26n_smoke_v3
experiments\yolo\.venv\Scripts\python.exe scripts\train_yolo_pilot.py --data data\processed\taco_pilot_v3\dataset.yaml --model yolo26n.pt --epochs 10 --name yolo26n_pilot10_v1
experiments\yolo\.venv\Scripts\python.exe scripts\train_yolo_pilot.py --data data\processed\taco_pilot_v3\dataset.yaml --model yolo11n.pt --epochs 10 --name yolo11n_pilot10_v1
```

Mỗi lệnh cần tên run mới nếu chạy lại; không ghi đè run cũ. Hai run 10 epoch dùng
cùng data, batch 4, input 640, seed 42, AdamW và ngân sách epoch. Đây là so sánh
pilot với recipe chung, chưa phải model đã tuning tối ưu hay kết luận hơn kém.
Nếu trainer early-stop, kiểm tra `results.csv` để biết số epoch thực.

Kết quả nằm trong `runs/waste_pilot/<name>/`: `weights/best.pt`, `weights/last.pt`,
`args.yaml`, `results.csv`, `pilot_summary.json` và `environment.txt`.
Weights pretrained được lưu riêng trong `checkpoints/yolo_pretrained/`.

Script kiểm tra hash ảnh trước khi train. Latency chỉ là phép đo chẩn đoán 30
lượt sau 5 lượt warm-up trên val, batch 1, mặc định FP32. Nó bao gồm lời gọi
`predict` với đọc file, nhưng chưa phải benchmark nghiệm thu 200 lượt và chưa
gồm upload mạng, Grad-CAM, OCR hay luồng xử lý rác. VRAM báo peak allocation của
PyTorch, không phải tổng VRAM mọi tiến trình.

## Kiểm thử script dữ liệu

```powershell
backend\.venv\Scripts\python.exe -m pytest -q tests\test_taco_pilot.py
```

Các kiểm thử bảo vệ chuyển đổi tọa độ, clip box, split theo batch, mapping lớp
hiếm, tải ảnh khi không có thumbnail và chuẩn hóa JPEG mà không sửa bản nguồn.
Không sửa backend hoặc nạp weights pilot thay cho `mobilenetv3_best.pth`.

Nguồn: [TACO chính thức](https://github.com/pedropro/TACO),
[train YOLO](https://docs.ultralytics.com/modes/train/).
