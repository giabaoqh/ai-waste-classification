# Phương án cải tiến AI phân loại và phát hiện rác

> Bổ sung thiết kế đa tầng ngày 10/11/2026: xem
> [AI_SYSTEM_DESIGN.md](AI_SYSTEM_DESIGN.md) cho taxonomy vật thể/vật liệu,
> Decision Engine, thí nghiệm RF-DETR có điều kiện và lộ trình từ RGB tới cảm biến.
> Tài liệu này tiếp tục giữ kế hoạch dữ liệu và baseline ban đầu.

> Triển khai pilot: xem [YOLO_PILOT.md](YOLO_PILOT.md) và
> [YOLO_PILOT_RESULTS.md](YOLO_PILOT_RESULTS.md). Đây là thử nghiệm detector riêng,
> chưa thay thế model production hoặc hoàn thành quy trình duyệt dataset.

Phạm vi: đề xuất cho repo `giabaoqh/ai-waste-classification`, dựa trên mã nguồn
commit `00d9524` và máy RTX 4050 Laptop 6 GB. Các chỉ tiêu bên dưới là mục tiêu
nghiệm thu do đề xuất này đặt ra, chưa phải kết quả thực nghiệm. Chưa triển khai
YOLO/SAM hoặc huấn luyện lại. Chưa tìm được checkpoint cũ trong Thùng rác có thể
truy cập; việc này không chứng minh file đã mất ở mọi bản sao khác.

## 1. Quyết định kiến trúc đề xuất

Chọn **YOLO26n fine-tune trên dữ liệu rác** làm ứng viên phát hiện chính; dùng
**YOLO11n** làm đối chứng. Chỉ thử bản small sau khi bản nano đã chạy ổn và lỗi
phân tích cho thấy cần tăng năng lực mô hình. YOLO26 và YOLO11 đều được Ultralytics
hỗ trợ; tài liệu hiện khuyến nghị YOLO26 cho dự án mới và YOLO11 là lựa chọn đã
trưởng thành. Đây là lý do chọn ứng viên, chưa chứng minh model nào tốt hơn trên
dữ liệu của nhóm. [Nguồn mô hình](https://docs.ultralytics.com/models/).

| Phương án | Đầu ra và vai trò | Đánh giá cho dự án |
|---|---|---|
| MobileNetV3 hiện tại | Nhãn cho toàn ảnh hoặc một crop | Giữ để tái lập baseline ảnh đơn và Grad-CAM |
| SSDLite COCO + MobileNet hiện tại | Đề xuất hộp theo COCO rồi phân loại crop | Baseline hai tầng; vật rác không thuộc COCO có thể bị bỏ sót ngay bước đầu |
| YOLO26n / YOLO11n detect | Hộp, nhãn, điểm cho từng vật thể | Ưu tiên cho ảnh nhiều rác và camera |
| YOLO26n-seg | Hộp và mask từng vật thể | Giai đoạn sau nếu cần phân tách vật chồng lấn hoặc hiển thị vùng rác |
| SAM 2.1 Tiny | Mask theo điểm/hộp gợi ý | Hỗ trợ gán nhãn; thử tách vùng theo hộp khi người dùng cần |
| SAM 3 / 3.1 | Phân vùng theo văn bản hoặc gợi ý thị giác | Thử nghiệm gán nhãn offline; chưa chọn làm đường suy luận mặc định |

SAM 2 là mô hình phân vùng theo gợi ý, không tự cung cấp bộ nhãn vật liệu của dự
án. SAM 3 có thể phát hiện/phân vùng bằng mô tả văn bản, nên không đúng khi nói
mọi SAM đều không nhận diện được lớp. Tuy nhiên, prompt như “medical waste” không
thay thế đánh giá vật liệu và độ tin cậy trên dữ liệu địa phương. SAM 3 công bố
848 triệu tham số, yêu cầu Python >=3.12 và checkpoint cần xin quyền truy cập;
backend mới thiết lập đang dùng Python 3.11. Đề xuất môi trường riêng cho SAM,
đo VRAM/latency trước; chưa có bằng chứng chạy toàn pipeline SAM 3 ổn trên GPU
6 GB này. [SAM 2](https://github.com/facebookresearch/sam2),
[SAM 3](https://github.com/facebookresearch/sam3).

Luồng sản phẩm mục tiêu:

```text
Ảnh → kiểm tra chất lượng → detector đã fine-tune
    → từng hộp + nhãn + điểm + quyết định chấp nhận/từ chối
    → cơ sở tri thức xử lý rác → giao diện tiếng Việt

Gán nhãn offline: SAM gợi ý vùng → con người sửa và xác nhận → dataset phiên bản mới
```

Phát hiện nhiều vật thể bằng YOLO trước; bổ sung mask khi đã chứng minh có ích.
Mask không trực tiếp cho biết vật liệu, độ nhiễm bẩn, khối lượng hoặc khả năng
tái chế. Nếu muốn ước lượng diện tích thì cần định nghĩa bài toán riêng; diện
tích pixel không phải khối lượng thực.

## 2. Những điểm cần sửa từ trạng thái hiện tại

1. **Thống nhất hợp đồng nhãn**: mã huấn luyện và mapping hiện là 7 lớp gồm
   `medical`; README và nhiều tài liệu cũ là 6 lớp. Gắn nhãn lớp, preprocessing,
   checkpoint, split và metrics vào cùng một phiên bản.
2. **Tái lập baseline**: số `accuracy=0.9456` nằm trong JSON lưu sẵn. Không còn
   checkpoint/split đã xác minh thì không dùng nó làm kết quả tái lập hoặc so
   trực tiếp với mAP detector. Tìm bản weights từ các thành viên trước; nếu không
   có thì huấn luyện baseline mới khi nhóm quyết định thực hiện.
3. **Huấn luyện detector với ảnh rác thực tế**: SSDLite hiện dùng pretrained COCO,
   chưa phải một detector chuyên rác. Giảm ngưỡng đơn thuần có thể tăng cả bắt
   đúng lẫn báo nhầm; cần đường precision–recall để chọn ngưỡng.
4. **Tách unknown khỏi other**: `other` là lớp rác tồn dư có ví dụ huấn luyện;
   `unknown/uncertain` là quyết định từ chối vì thiếu bằng chứng. Không gọi mọi
   vật ngoài tập nhãn là “rác khác”.
5. **Tách dữ liệu demo trên bản đồ khỏi quan sát thật**: database tự seed báo cáo
   và crawler có danh sách tin viết sẵn. Bổ sung nguồn và trạng thái xác minh
   cho từng mục, tránh xem tất cả điểm có sẵn là kết quả AI phát hiện.

## 3. Dataset nên lấy ở đâu và dùng thế nào

| Nguồn chính thức | Nhãn và vai trò | Giới hạn khi đưa vào dự án |
|---|---|---|
| [RealWaste — UCI](https://archive.ics.uci.edu/dataset/908/realwaste) | 4.752 ảnh, 9 vật liệu; tái lập classifier, học ảnh rác nền thực | Nhãn toàn ảnh, chưa có box/mask dùng ngay cho YOLO; không có lớp medical |
| [TACO — repo tác giả](https://github.com/pedropro/TACO) | Rác trong bối cảnh thực, có annotation phục vụ detection/segmentation | Phải kiểm tra phiên bản, mức lệch lớp, mapping vật liệu và tình trạng URL ảnh |
| [TrashNet — repo tác giả](https://github.com/garythung/trashnet) | 2.527 ảnh, 6 lớp; bổ sung ảnh đơn để đối chứng classifier | Nền trắng, không có organic/medical; không đại diện ảnh đường phố |
| [ZeroWaste — repo tác giả](https://github.com/dbash/zerowaste/) | Nghiên cứu phân vùng rác công nghiệp lộn xộn | Khác miền ảnh gia đình/đường phố; phù hợp tập thử thách hoặc mở rộng có chủ đích |
| Ảnh nhóm tự thu thập có quyền sử dụng | Nguồn quan trọng để đánh giá và thích nghi bối cảnh Việt Nam | Cần box đầy đủ, phân nhóm cảnh/vật thể và người kiểm tra nhãn |

Theo bài báo TACO năm 2020, phiên bản được công bố có 1.500 ảnh và 4.784
annotation; không mặc định bản tải hôm nay có đúng số này. Đếm ảnh, annotation và
phân bố lớp sau khi tải, ghi hash annotation.
[Bài báo TACO](https://arxiv.org/abs/2003.06975).

RealWaste có CC BY 4.0 trên UCI. Với các nguồn khác, lưu license và quy định từng
bản ảnh; license mã nguồn công cụ không mặc nhiên là license mọi ảnh đi kèm.
Kaggle/Roboflow có thể hỗ trợ tìm thêm dữ liệu, nhưng phải lần về tác giả gốc,
kiểm tra trùng lặp và chất lượng annotation. Chưa chọn một dataset medical công
khai nào làm nguồn đã được xác minh cho kế hoạch này.

Mapping đề xuất cho RealWaste: Cardboard + Paper → paper; Food Organics +
Vegetation → organic; Textile Trash + Miscellaneous Trash → other; giữ plastic,
metal, glass. Đây là quyết định nhãn của dự án, không phải bộ nhãn gốc. Đối với
TACO, duyệt từng category; bao bì ghép nhiều vật liệu không tự động thành paper.
Không đổi mọi khẩu trang thành medical chỉ dựa vào hình dáng; thống nhất chính
sách nhãn dựa trên bằng chứng quan sát được.

Đích dữ liệu giai đoạn đầu, điều chỉnh sau thử gán nhãn 100 ảnh:

- Khoảng 2.500–4.000 ảnh detection đã kiểm tra từ TACO và ảnh tự thu thập;
  hướng tới ít nhất 500 instance train cho mỗi lớp thường gặp.
- Thêm 300 ảnh kiểm thử địa phương được giữ riêng và 300 ảnh không có rác thuộc
  phạm vi mục tiêu: người, thú nuôi, nội thất, cảnh tự nhiên, đồ đang sử dụng.
- Có các lát đánh giá riêng: ảnh đơn, 2–5 vật thể, nhiều vật chồng lấn, ánh sáng
  yếu, vật nhỏ, nhựa trong và vật bẩn/bẹp. Không cố đánh nhãn vật liệu khi người
  gán nhãn cũng không nhìn rõ; quy định ignore/out-of-scope nhất quán.
- `medical` chỉ được công bố là lớp đã kiểm chứng khi có dữ liệu đủ chất lượng
  và test riêng. Nếu chưa đủ, giữ API tương thích nhưng hiển thị hạn chế và yêu
  cầu kiểm tra thủ công; không lấy một vài ảnh lặp lại để chứng minh recall cao.

Không dùng box bao toàn ảnh chỉ để chuyển dataset classification thành detection.
Chỉ dùng box toàn ảnh khi vật thể thực sự lấp đầy ảnh và box đã được kiểm tra.

## 4. Quy trình dữ liệu trước huấn luyện

1. Viết quy tắc box, vật bị che, vật nhiều thành phần, nhãn medical/other và ảnh
   không thuộc phạm vi. Gán nhãn thử 100 ảnh trước để sửa quy tắc còn mơ hồ.
2. Lưu raw bất biến, nguồn, quyền sử dụng, ngày thu thập, `scene_id`, `object_id`,
   session/video và thiết bị. Tạo hash chính xác và perceptual hash để tìm trùng.
3. Chia train/val/test theo nhóm cảnh, vật thể hoặc video trước khi tạo augmentation.
   Dùng 70/15/15 làm khởi điểm nếu dataset chưa có split chuẩn. Với benchmark có
   split chính thức, giữ split đó; ảnh cùng phiên chụp không sang hai tập.
4. Giữ tập local 300 ảnh độc lập để kiểm thử domain shift. Không đưa ảnh test,
   crop, ảnh gần trùng hoặc pseudo-label của chúng vào train hay chỉnh ngưỡng.
5. Gán nhãn bằng công cụ như CVAT. SAM có thể đề xuất mask, sau đó người gán nhãn
   sửa box/lớp và xác nhận. Kiểm tra chéo ít nhất 10–20% train/val; nhãn test được
   duyệt kỹ toàn bộ. [Công cụ CVAT](https://docs.cvat.ai/docs/getting_started/overview/).
6. Xuất YOLO format: class ID và tọa độ box chuẩn hóa; kiểm tra box nằm trong ảnh,
   kích thước dương, ảnh giải mã được và ID đúng mapping. Ảnh negative phải được
   xác nhận không có vật thuộc tập nhãn. SAM không được dùng làm đáp án chuẩn của
   chính test dùng để đánh giá model.
   [Định dạng YOLO](https://docs.ultralytics.com/datasets/detect/).

Đầu ra: manifest có nguồn + hash + split, mapping phiên bản, báo cáo phân bố lớp,
ảnh mẫu có box overlay và danh sách trường hợp bị loại. Một dataset tốt cần truy
được nguồn và tái tạo split, không chỉ có một thư mục ảnh lớn.

## 5. Phương pháp huấn luyện phù hợp RTX 4050 6 GB

Fine-tune pretrained weights thay vì train từ đầu. Tạo môi trường thử YOLO riêng
để giữ backend đã chạy ổn. Chạy 1 epoch với 50–100 ảnh kiểm tra dữ liệu, loss,
box overlay, RAM/VRAM và thời gian; lần chạy này chưa phải kết quả mô hình.

Cấu hình khởi điểm đề xuất: nano, ảnh 640, batch 4, AMP, 80–100 epochs,
early stopping patience 15–20, workers 0 trên Windows để kiểm tra ban đầu; sau đó
thử workers 2. Nếu OOM, giảm batch xuống 2; chỉ giảm resolution khi đã đánh giá
ảnh hưởng vật nhỏ. Nếu còn dư bộ nhớ và có ích trên val thì thử batch 8. Không
cam kết thời gian train hoặc FPS trước khi đo.

Ví dụ lệnh dự kiến, **chưa chạy**; chỉ hợp lệ sau khi tạo dataset YAML và môi
trường có phiên bản Ultralytics/PyTorch đã kiểm tra:

```powershell
yolo detect train model=yolo26n.pt data=data/waste_detection/dataset.yaml epochs=100 imgsz=640 batch=4 device=0 amp=True workers=0 patience=20 seed=42 deterministic=True close_mosaic=10 project=runs/waste name=yolo26n_v1
```

Train mode hỗ trợ batch cố định/tự chọn, AMP, seed và tắt mosaic cuối quá trình;
các giá trị trên là lựa chọn khởi điểm cho máy này.
[Tham số chính thức](https://docs.ultralytics.com/modes/train/).

Với classifier baseline, dùng ảnh đơn/crop có nhãn vật liệu, ImageNet transfer
learning, cross entropy; nếu lệch lớp, thử weighted sampler hoặc weighted loss
riêng rẽ và chọn bằng val. Không chỉnh cả hai ngay khi chưa có đối chứng.

Với detector: bắt đầu bằng recipe có sẵn, augmentation màu/độ sáng/scale vừa
phải, flip và mosaic. Xem ảnh augmented để tránh làm mất vật nhỏ. So sánh YOLO11n
và YOLO26n trên cùng dữ liệu, resolution, split và ngân sách thử; lưu recipe thực
tế thay vì giả định optimizer của hai phiên bản giống nhau.

Sau baseline, thêm 200–500 ảnh train được chọn từ lỗi thực: đồ nhựa trong, kim
loại phản chiếu, bìa bẩn, đồ dùng chưa bị vứt, vật nhỏ, ảnh nhiều vật. Gán nhãn lại
và train phiên bản v2. Thu thập hard negatives từ dữ liệu ngoài test; không đào
lỗi trên test rồi tiếp tục báo kết quả đó là kiểm thử độc lập.

Chọn model và ngưỡng trên val; dành riêng một phần val cho calibration hoặc dùng
cross-validation nếu dữ liệu ít. Thử 2–3 seed cho ứng viên cuối khi đủ thời gian.
Nếu chưa ổn định giữa seed, ưu tiên bổ sung dữ liệu và báo độ dao động.

## 6. Ma trận thí nghiệm và tiêu chí đo

| Run | Mục đích | So sánh đúng |
|---|---|---|
| A: MobileNetV3 tái lập | Baseline ảnh đơn | Macro-F1, per-class F1 trên tập classification cố định |
| B: SSDLite + A | Baseline hệ thống hiện tại | Chấm hộp và nhãn đầu-cuối trên cùng tập detection |
| C: YOLO11n | Detector đối chứng | mAP/precision/recall trên cùng box test |
| D: YOLO26n | Ứng viên chính | So với C và B, cùng tập và quy tắc matching |
| E: D + dữ liệu lỗi | Kiểm chứng hiệu quả dữ liệu | Cùng bộ kiểm thử cố định; báo cả thay đổi dữ liệu và model |
| F: YOLO26n-seg, tùy chọn | Kiểm chứng lợi ích mask | Mask AP/IoU, độ trễ và mức tăng hữu ích thực tế |

Không so accuracy 94,56% của classifier với mAP của YOLO như cùng một chỉ số.
Thời gian end-to-end cần gồm decode/preprocess/inference/postprocess; đo Grad-CAM
hoặc mask riêng nếu chúng bật. Đo batch 1, cùng kích thước ảnh và chế độ precision,
warm-up 20 lần rồi tối thiểu 200 lần; báo median, p95, VRAM peak, GPU/CPU và cấu
hình nguồn điện laptop. Chỉ ghi tốc độ thực trên máy; không lấy benchmark T4 từ
tài liệu làm FPS của RTX 4050.

Mục tiêu nghiệm thu ban đầu, cần xem lại sau baseline và khó độ dữ liệu:

| Hạng mục | Mục tiêu đề xuất | Điều kiện công bố |
|---|---|---|
| Classifier | Macro-F1 >=0,90 nội bộ, >=0,80 local | Cùng định nghĩa lớp; báo F1/support từng lớp |
| Detection | mAP@0.5 >=0,80; mAP@0.5:0.95 >=0,50 trên test nội bộ | Đánh giá hộp và lớp; báo riêng local/dense, không gộp che lỗi |
| Ảnh nhiều vật địa phương | Recall >=0,80 ở IoU 0.5 và ngưỡng chốt từ val | Báo precision cùng recall và support; matching 1-1 |
| Ảnh không có rác mục tiêu | Tỷ lệ ảnh có ít nhất một false positive <=5% | 300 ảnh negative độc lập; báo số lỗi và khoảng tin cậy 95% |
| medical nếu đủ dữ liệu | Recall >=0,90 tại precision >=0,80 | Tối thiểu 100 instance test độc lập, báo khoảng tin cậy và giới hạn; nếu thiếu thì chưa nghiệm thu lớp |
| Độ trễ detector | p95 <=250 ms/ảnh trên RTX 4050 | Batch 1, input 640, quy trình đo cố định; không gồm upload mạng |
| Bộ nhớ | Peak dưới khoảng 5,5 GB | Đo thực để chừa tài nguyên cho tiến trình khác |
| Từ chối dự đoán | Báo accuracy/precision trên phần được chấp nhận kèm coverage | Không đạt điểm đẹp bằng cách từ chối hầu hết ảnh |

Các mục tiêu medical chỉ là đánh giá một lớp thị giác trong phạm vi dataset, không
chứng nhận an toàn y tế. Không gán nhãn chắc chắn khi chỉ dựa vào filename. Điểm
confidence của detector chưa phải xác suất đã hiệu chuẩn; đo calibration nếu
định hiển thị như xác suất, đồng thời báo đường risk–coverage.

## 7. Lộ trình 6 tuần và điều kiện qua bước

Ước lượng cho nhóm nhỏ; sửa lịch sau khi đo tốc độ gán nhãn 100 ảnh. Đây không
phải cam kết thời gian train hay lịch đã thống nhất của các thành viên.

| Thời gian | Công việc | Sản phẩm và điều kiện chuyển bước |
|---|---|---|
| Tuần 1 | Chốt phạm vi lớp, kiểm kê model, lấy dữ liệu, gán nhãn thử | Spec nhãn, manifest nguồn, 100 ảnh mẫu được duyệt, split nhóm không rò rỉ |
| Tuần 2 | Hoàn thiện box và negative, kiểm tra nhãn | Dataset v1, thống kê từng lớp, local test và negative test đóng băng; chỉ train khi box audit đạt |
| Tuần 3 | Tái lập A/B, fine-tune C/D | Checkpoint theo phiên bản, config, log, bảng val, đo VRAM; chọn nano theo chất lượng và tốc độ |
| Tuần 4 | Phân tích lỗi, hard negatives, chạy E | Dataset v2, bảng so sánh trên val, ngưỡng/calibration chốt; model đủ mạnh mới được mở test cuối |
| Tuần 5 | Đánh giá cuối và segmentation tùy chọn | Test report có hỗ trợ từng lớp/nhóm cảnh; thử F chỉ khi nhu cầu mask rõ và đủ thời gian |
| Tuần 6 | Tích hợp API/UI, kiểm thử hồi quy, đóng gói | Demo ảnh đơn/đa vật/negative/uncertain; API ổn định, model card, checksum và hướng dẫn tái lập |

Nếu chỉ còn 2–3 tuần, giữ dataset audit + YOLO nano + negative/uncertain + bảng
đánh giá, lùi SAM runtime/segmentation. Nếu model chưa đạt mục tiêu, bàn giao kết
quả thật, nhóm lỗi chính và dữ liệu cần thêm; không điều chỉnh test hoặc che lớp
yếu để báo đạt.

## 8. Tích hợp cụ thể vào repo

- `backend/app/detector.py`: adapter YOLO và kết quả object chuẩn hóa; giữ đường
  SSDLite để benchmark cho tới khi có quyết định thay thế.
- `backend/app/main.py`: `/api/detect` dùng backend được cấu hình; endpoint ảnh
  đơn classifier có thể giữ cho Grad-CAM. Health trả model version, task, lớp và
  trạng thái nạp. Không tự dùng random model khi thiếu weights.
- Schema từng object: box, class, `detector_confidence`, cờ uncertain, model
  version và mask nếu có. Score detector không phải phân bố softmax của 7 lớp;
  không điền giả vào trường probabilities cũ. Cập nhật frontend/test theo hợp
  đồng API đã chọn trước khi thay đường nhận diện.
- `backend/app/knowledge_base.py`: giữ quy tắc xử lý tách khỏi thị giác. Vật nhiều
  chất liệu và medical cần thông báo phù hợp khi mô hình chưa đủ bằng chứng.
- UI: hộp từng vật, chọn vật xem hướng dẫn, hiển thị rõ ảnh không có vật mục tiêu,
  ảnh không chắc chắn và model chưa nạp. Feedback sửa nhãn đi vào hàng đợi duyệt,
  chưa tự động trở thành training data.
- Tests: tách unit/API/model/benchmark, bỏ tác vụ mạng/model chạy lúc import,
  chuẩn hóa fixture và DB tạm. Đảm bảo build và các unit test hiện có vẫn đạt.

Grad-CAM classifier không mặc nhiên áp dụng trực tiếp cho detector mới. Giữ XAI
cũ cho endpoint classifier hoặc thiết kế kiểm chứng riêng; mask của SAM cũng
không phải bằng chứng giải thích lý do phân loại vật liệu.

## 9. Bàn giao, tái lập và bảo toàn công sức

Mỗi model bàn giao kèm: weights, SHA-256, mapping lớp, preprocessing, phiên bản
package, config/seed, dataset/split hashes, commit code, báo cáo test theo lớp,
latency/VRAM và model card giới hạn sử dụng. Dùng định danh riêng cho classifier
và detector, không ghi đè `mobilenetv3_best.pth` bằng YOLO `.pt`.

Weights/dataset tiếp tục nằm ngoài Git; lưu model vào kho có phiên bản mà nhóm
được phép dùng, thêm một bản sao độc lập và thử tải lại xác minh hash. Code dùng
nhánh/PR theo quy trình nhóm; không coi một máy hoặc Thùng rác là bản backup.
Việc push/upload cần nằm trong hành động người dùng yêu cầu, chưa thực hiện ở
lần thiết lập này.

Ultralytics công bố YOLO26 theo AGPL-3.0 hoặc Enterprise; ghi nhận điều kiện
license của code/weights khi chọn cách phân phối dự án.
[Nguồn license mô hình](https://docs.ultralytics.com/models/yolo26/).

Ưu tiên hành động đầu tiên: chốt chính sách 6/7 lớp và tiêu chí medical, tạo
100 ảnh có box được kiểm tra, rồi đo một lần train thử nano. Chỉ sau đó mới chốt
khối lượng dữ liệu, lịch và các mục tiêu cuối.
