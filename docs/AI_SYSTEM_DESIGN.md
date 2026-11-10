# Thiết kế AI phân loại rác đa tầng và kế hoạch kiểm chứng

Ngày rà soát: 10/11/2026. Repo: `giabaoqh/ai-waste-classification`, nguồn
`00d9524`; phần cứng đã kiểm tra: RTX 4050 Laptop 6 GB. Đây là bản thiết kế từ
nội dung nghiên cứu người dùng cung cấp, chưa phải hệ thống được huấn luyện hoặc
triển khai. Không tạo thêm ảnh, cài thêm mô hình hoặc mua phần cứng trong bước này.

## 1. Quyết định cho phiên bản tiếp theo

Xây MVP camera RGB với một detector, đánh giá vật liệu quan sát được, cơ chế từ
chối và cơ sở tri thức. SAM/DINO dùng offline khi có lợi ích đo được. Nhánh cảm
biến và điều khiển thiết bị là các giai đoạn sau, dùng chung hợp đồng bằng chứng.

| Thành phần | Đề xuất triển khai | Điều kiện mở rộng |
|---|---|---|
| Detector | YOLO26n trước, YOLO11n đối chứng | Thử YOLO26s khi nano đã ổn và còn đủ tài nguyên |
| RF-DETR | Nano/Small là thí nghiệm bổ sung | Chạy thử bộ nhớ trước; không mặc định train vừa 6 GB |
| Phân vùng | Annotation có sẵn hoặc SAM hỗ trợ gán nhãn | Chạy segmentation online khi cải thiện rõ lỗi chồng lấn |
| Vật liệu | Tái lập MobileNetV3, thử classifier ROI | Giữ nhánh riêng chỉ khi tốt hơn detector trực tiếp |
| OCR | Tùy chọn cho nhãn/ký hiệu nhìn thấy | Đo mức đọc đúng và tỷ lệ không đọc được |
| DINOv3 | Embedding offline, tìm ảnh gần giống và mẫu đa dạng | So với hash/embedding đơn giản trước; không tự coi khoảng cách là bằng chứng OOD |
| NIR/SWIR | Chưa đưa vào MVP | Có cảm biến, dữ liệu phổ và ground truth vật liệu |
| Decision Engine | Quy tắc độc lập với framework AI | Có trạng thái thiếu dữ liệu, mâu thuẫn và kiểm tra thủ công |
| Robot/băng chuyền | Chưa điều khiển trong MVP | Qua nghiệm thu thiết bị, đồng bộ và cơ chế dừng |

RF-DETR hỗ trợ detection/instance segmentation. Repo hiện phân biệt license theo
variant: Nano/Small/Medium/Large detection là Apache 2.0; không suy rộng sang mọi
size. Tài liệu có cấu hình tiết kiệm bộ nhớ lấy ví dụ 8 GB; đó không phải mức tối
thiểu tuyệt đối, cũng không chứng minh chạy được trên 6 GB. Nếu thử, dùng môi
trường riêng, batch vật lý nhỏ, gradient checkpointing/accumulation khi cần và
đo cả bước validation. Giữ ngân sách thử có giới hạn, không cắt nhỏ ảnh đến mức
mất mục tiêu chỉ để báo chạy được.
[RF-DETR](https://github.com/roboflow/rf-detr),
[bộ nhớ](https://rfdetr.roboflow.com/latest/learn/train/advanced/),
[tham số train](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/).

DINOv3 có bản ViT-S/16 21 triệu tham số trong danh mục chính thức. Có thể thử
backbone nhỏ để trích embedding với weights cố định rồi huấn luyện linear probe
hoặc classifier nhẹ; không cần tự pretrain một foundation model. Tách quyền
truy cập weights và khả năng sử dụng thực tế khỏi đề xuất kiến trúc.
[DINOv3 chính thức](https://github.com/facebookresearch/dinov3/blob/main/README.md).

## 2. Tách taxonomy thành các trường độc lập

Các nhãn trong YAML người dùng gửi đang trộn kiểu vật thể, vật liệu và nhóm
chức năng. Chúng là bản nháp phù hợp để thảo luận, chưa đủ để xác nhận polymer,
khả năng tái chế hay nguy hại. Trước khi train, chốt schema cùng hướng dẫn gán nhãn:

| Trường | Ví dụ | Bằng chứng và giới hạn |
|---|---|---|
| `object_type` | bottle, can, bag, food_container, battery | Box/mask và nhãn người kiểm duyệt |
| `material_group` | plastic, metal, glass, paper, organic, mixed | Có thể là ước lượng thị giác; ghi nguồn và confidence riêng |
| `polymer` | PET, HDPE, PP hoặc null | Không suy ra chắc chắn từ hình dạng chai; cần chứng cứ phù hợp |
| `visible_contamination` | observed, not_observed, not_assessed | Chỉ tình trạng nhìn thấy; không đồng nghĩa sạch hóa học/vi sinh |
| `hazard_status` | suspected, no_visible_cue, not_assessed | Không thấy dấu hiệu nguy hại không có nghĩa đã xác nhận an toàn |
| `route` | recyclable, organic, residual, special_collection, manual_review | Quy tắc xử lý có phiên bản, địa phương và điều kiện đầu vào |

Chốt 8–10 lớp vật thể có đủ ví dụ sau pilot 100 ảnh; thêm lớp khác bằng phiên bản
taxonomy mới. Giải quyết các trường hợp chồng nghĩa như hộp thức ăn/bao bì hỗn
hợp trong hướng dẫn gán nhãn trước khi giao nhiều người annotate.

Giữ nguyên ID của 7 lớp classifier cũ khi tái lập checkpoint. Dataset vật thể mới
là taxonomy khác; không đổi `medical` thành `battery` hoặc tái sử dụng ID mà không
đổi version. Không tự thêm một lớp `unknown` rồi kỳ vọng học được mọi thứ ngoài
dataset: từ chối là chính sách đầu ra cần kiểm thử riêng.

Một chiếc chai còn đang sử dụng có thể giống chai bị bỏ đi. Phải xác định phạm vi
ảnh đầu vào và ngữ cảnh gán nhãn; nhận diện được chai không tự chứng minh đó là rác.

## 3. Luồng xử lý và trách nhiệm từng tầng

Luồng online:

1. Kiểm tra ảnh: định dạng, kích thước, ánh sáng, độ nét; ảnh lỗi yêu cầu chụp lại.
2. Detector trả box, nhãn vật thể, score và model version. Không có box là một
   kết quả hợp lệ; không ép classifier toàn ảnh tạo ra nhãn thay thế.
3. Router chọn nhánh cần dùng: classifier ROI, OCR hoặc kiểm tra dấu hiệu nguy
   hại. Chỉ chạy mô-đun cần thiết; không để mọi vật đi qua mọi model.
4. Fusion giữ từng bằng chứng, nguồn, chất lượng và phần còn thiếu. Chưa có dữ
   liệu đa phương thức thì dùng quy tắc minh bạch; chưa cần mạng fusion học sâu.
5. Decision Engine áp dụng thứ tự ưu tiên bên dưới rồi trả lời khuyến nghị tiếng
   Việt và lý do. Detector không trực tiếp điều khiển servo/motor.

Luồng offline: feedback/ảnh mới có quyền sử dụng → hàng đợi duyệt → gán nhãn và
QA → dataset version → train → chọn bằng validation → test cuối → model registry
→ triển khai có kiểm thử hồi quy. Feedback không tự cập nhật model đang phục vụ.

SAM hỗ trợ hình học annotation; người duyệt chịu trách nhiệm lớp và vùng cuối.
DINOv3 có thể ưu tiên mẫu đa dạng. Khi chọn ảnh bằng active learning, kết hợp
uncertainty, đa dạng embedding, độ hiếm lớp và loại trùng để tránh hàng trăm ảnh
gần giống nhau. So với chọn ngẫu nhiên với cùng số phút gán nhãn, không chỉ cùng
số ảnh.

## 4. Decision Engine và dữ liệu thiếu

| Ưu tiên | Tình huống | Đầu ra MVP |
|---|---|---|
| 1 | Lỗi đầu vào hoặc model chưa nạp | Báo chưa sẵn sàng/chụp lại; không tạo dự đoán |
| 2 | Có bằng chứng nghi nguy hại | `HAZARD_SUSPECTED`, khuyến nghị thu gom chuyên biệt/kiểm tra |
| 3 | Bằng chứng mâu thuẫn hoặc không đủ | `REVIEW`, chỉ rõ trường chưa biết |
| 4 | Đủ bằng chứng cho chính sách đang dùng | `RECOMMEND`, nêu luồng xử lý và điều kiện |
| 5 | Không phát hiện vật thuộc phạm vi | `NO_TARGET`, không diễn giải thành “không có rác ở mọi dạng” |

Điểm detector, classifier và OCR không phải xác suất của cùng một biến. Không
lấy trung bình hoặc nhân trực tiếp các score không được hiệu chuẩn để tạo một
confidence tổng. Chỉ gọi một score là xác suất đã hiệu chuẩn khi đã kiểm chứng.

Không có cảm biến thì trường đó giữ `not_assessed`/null; không mặc định polymer,
độ sạch hoặc `hazard_suspected=false`. Filename/barcode chỉ là gợi ý metadata,
chưa phải phép đo vật liệu. Repo hiện có override dựa trên filename; cần hiển thị
rõ nguồn của override và kiểm thử ảnh giữ nguyên nhưng đổi tên file.

Ví dụ cấu trúc trả lời sau đây chỉ mô tả schema, không chứa kết quả inference:

```json
{
  "example_only": true,
  "schema_version": "waste-observation-v2-draft",
  "object_id": "example-object",
  "object": {"label": null, "score": null, "model_version": null},
  "material": {"group": null, "polymer": null, "source": "not_measured"},
  "visible_contamination": {"status": "not_assessed"},
  "hazard": {"status": "not_assessed", "evidence": []},
  "decision": {"action": "REVIEW", "route": "manual_review", "reasons": ["insufficient_evidence"]},
  "policy_version": "draft"
}
```

MVP trả khuyến nghị; `SORT` là hành động riêng chỉ xuất hiện ở giai đoạn phần
cứng. Khi đó lệnh cần ID vật thể, timestamp, thời hạn hiệu lực và trạng thái thiết
bị. Cảm biến phải khớp đúng vật thể/thời gian; không gán phổ của vật A cho box B.
Chưa có cơ khí thì không báo tỷ lệ rác “đã vào đúng thùng” từ số lần dự đoán.

## 5. Dữ liệu và thí nghiệm tối thiểu

Nguồn tải và kế hoạch dữ liệu chi tiết nằm trong
[AI_IMPROVEMENT_PLAN.md](AI_IMPROVEMENT_PLAN.md). Ưu tiên TACO + ảnh địa phương cho
box/mask, RealWaste/TrashNet cho baseline classifier. Dữ liệu classification cần
annotation bổ sung trước khi dùng như detection; không tạo box giả toàn ảnh.

ZeroWaste-f có 4.503 ảnh; trang dự án cung cấp nguồn tải và ghi license dữ liệu
CC BY-NC 4.0. Giữ riêng lineage và điều kiện sử dụng khi phối trộn.
[Nguồn ZeroWaste](https://ai.bu.edu/zerowaste/).

Pilot dữ liệu phải tạo được: manifest nguồn/quyền sử dụng, ảnh overlay, taxonomy
version, nhóm cảnh/vật thể/video và báo cáo QA. Chia theo nhóm trước augmentation;
giữ tập local và negative độc lập. Với dữ liệu phổ tương lai, nhiều lần quét cùng
một mẫu vật cũng phải cùng split, không chia ngẫu nhiên từng vector phổ.

| Thí nghiệm | Câu hỏi cần trả lời | Cách giữ so sánh công bằng |
|---|---|---|
| YOLO11n vs YOLO26n | Detector nào phù hợp miền rác? | Cùng dữ liệu/split/evaluator, ghi recipe và ngân sách train |
| YOLO26n vs YOLO26s | Model lớn hơn có đáng chi phí? | Đo chất lượng, p95 và VRAM cùng chế độ inference |
| RF-DETR Nano/Small | Transformer có cải thiện đáng kể? | Chạy khi đủ bộ nhớ; báo resolution và chi phí khác biệt |
| Detector vật liệu trực tiếp vs ROI classifier | Nhánh chuyên biệt có ích? | So end-to-end cả lỗi bỏ sót box và lỗi phân loại |
| Không mask vs có mask | Mask có giảm lỗi vật liệu/chồng lấn? | Cùng ảnh test; mask ground truth và mask dự đoán đánh giá riêng |
| Active learning vs chọn ngẫu nhiên | Có tiết kiệm annotation? | Cùng thời gian gán nhãn và số vòng, không dùng test để chọn mẫu |
| Đổi filename/ẩn metadata | Model dựa vào ảnh hay metadata? | Giữ nguyên ảnh, tách kết quả thị giác và hướng dẫn theo metadata |

Chọn cấu hình trên validation; dùng test cố định để báo cáo kết quả cuối sau khi
chốt model/ngưỡng. Nếu cần đánh giá nhiều model trên test cho báo cáo nghiên cứu,
không tiếp tục dùng các kết quả đó để tinh chỉnh rồi vẫn gọi test là độc lập.

Train YOLO nano bắt đầu với batch 4, ảnh 640, AMP; giảm batch nếu OOM. Lệnh batch
16 cho YOLO26s trong nội dung gốc chưa được kiểm tra trên laptop này. RF-DETR cần
thử riêng; gradient accumulation tăng batch danh nghĩa qua nhiều lượt, không làm
một lượt forward vừa bộ nhớ khi bản thân lượt đó đã quá lớn.

## 6. Mốc triển khai và tiêu chí dừng

Lịch 6 tuần là ước lượng cho MVP RGB, không bao gồm hoàn thiện cảm biến/robot:

| Mốc | Đầu ra | Điều kiện qua bước |
|---|---|---|
| Tuần 1 | Taxonomy, schema, 100 ảnh pilot, quy tắc annotate | Lớp không mâu thuẫn; split nhóm và quyền dùng truy được |
| Tuần 2 | Dataset v1 và tập local/negative đóng băng | Box/mask QA đạt; lớp thiếu dữ liệu được đánh dấu |
| Tuần 3 | Baseline + hai YOLO; RF-DETR nếu khả thi | Có log/config/version và phép đo bộ nhớ, không chỉ loss giảm |
| Tuần 4 | Dữ liệu lỗi v2, router/Decision Engine được đặc tả | Nhánh mới chứng minh giá trị trên val; ngưỡng và cơ chế từ chối chốt |
| Tuần 5 | Test cuối, model card, báo cáo lỗi | Báo đủ support, lớp yếu, coverage và khoảng tin cậy |
| Tuần 6 | API/UI tích hợp và gói tái lập | Hợp đồng score đúng; model thiếu/ảnh lỗi/unknown đều xử lý rõ |

Mục tiêu dự thảo giữ từ kế hoạch trước: mAP50 >=0,80, mAP50:95 >=0,50 trên test
nội bộ; false-positive image rate <=5% trên 300 ảnh negative; p95 detector <=250 ms
trên GPU đang dùng. Chúng chưa được đo, không phải mức tối thiểu chắc chắn đạt.
Báo riêng ảnh local/dense, per-class precision/recall/AP, VRAM và risk–coverage.
Lớp nguy hại có gate riêng trong kế hoạch dữ liệu; thiếu mẫu thì chưa công bố đạt.

Mask cần mask AP/IoU phù hợp tác vụ; không dùng mIoU duy nhất thay cho chất lượng
từng instance. Giai đoạn cơ khí bổ sung purity/recovery theo count hoặc mass đã
định nghĩa, tỷ lệ vật vào đúng thùng, tỷ lệ nghi nguy hại bị lọt và thời gian toàn
chu trình. Chưa có cân thì không công bố purity theo khối lượng.

Lùi SAM online, NIR hoặc robot nếu dataset/recall/negative test chưa ổn. Nếu cần
điểm mới nghiên cứu, ưu tiên kiểm chứng lợi ích từ chối và active learning trước
khi tăng số framework và model chạy đồng thời.

## 7. Những thay đổi code cần thực hiện sau bước thiết kế

| Thứ tự | Phạm vi dự kiến | Nghiệm thu |
|---|---|---|
| P0 | Chuẩn taxonomy, manifest và kiểm tra nhãn | Tái tạo split; phát hiện ID sai, box lỗi và nhóm trùng |
| P1 | Detector adapter trong `backend/app/detector.py` | Phân biệt unavailable/no-target/uncertain, nạp đúng version |
| P2 | Schema v2 và Decision Engine trong backend | Không tạo xác suất classifier giả từ score detector; kiểm thử thiếu/mâu thuẫn bằng chứng |
| P3 | Material ROI/OCR adapter có điều kiện | Bỏ nhánh nếu không cải thiện chất lượng/chi phí theo tiêu chí đã chốt |
| P4 | UI từng object, trạng thái REVIEW và hàng đợi feedback | Người dùng thấy nguồn nhận định, sửa nhãn không tự cập nhật model |
| P5 | Model registry và export/benchmark | Checksum, môi trường, config, dataset version và rollback rõ |

Giữ `/api/predict` cũ trong thời gian chuyển đổi, thiết kế version cho hợp đồng
đa vật thể trước khi thay schema. SQLite metadata vẫn phù hợp MVP cục bộ; chỉ
chuyển PostgreSQL khi có nhu cầu nhiều người dùng/đồng thời cụ thể. MQTT/ROS 2 chỉ
đưa vào khi xuất hiện bài toán thiết bị. Chưa sửa backend/UI trong lần rà soát này.

## 8. Kiểm chứng các nguồn trong bản nghiên cứu người dùng gửi

| Nội dung | Trạng thái sau rà soát | Cách dùng |
|---|---|---|
| RF-DETR hỗ trợ detection/segmentation, license phụ thuộc variant | Xác minh từ repo chính thức | Chọn đúng variant và pin phiên bản |
| DINOv3 có backbone nhỏ để trích đặc trưng | Xác minh danh mục Meta | Đề xuất thử offline, chưa khẳng định OOD tốt trên rác |
| ZeroWaste-f 4.503 ảnh, CC BY-NC 4.0 | Xác minh trang Boston University | Dữ liệu thử thách công nghiệp có lineage riêng |
| Edge AI tại Việt Nam: 21.329 ảnh, 6.171 ảnh Hà Nội, QAT mAP50 0,886 và 34,5 FPS | Xác minh abstract công bố | Dùng làm tài liệu tham khảo; chưa thấy link tải dataset trong trang này |
| NIST ngày 30/10/2026 về AI/robot phân loại rác | Chưa xác minh được nguồn cụ thể | Không đưa ngày/số liệu này thành luận cứ đã xác nhận |
| RGB + NIR đạt 99,1% polymer | Chưa tìm được bài/DOI khớp từ thông tin đã gửi | Chờ DOI hoặc bài gốc; không dùng làm mục tiêu cam kết |
| SWIR + AutoEncoder-gated XGBoost | Tìm được định danh DOI 10.1016/j.measurement.2026.122077; trang nhà xuất bản bị 403 | Chưa đọc được nội dung phương pháp/kết quả; không suy ra hiệu năng |

Bài Edge AI Việt Nam dùng YOLOv8s và Jetson Orin Nano; các số trên thuộc điều kiện
của nghiên cứu đó, không phải benchmark YOLO26 trên RTX 4050, và một bài báo có
dataset không đồng nghĩa dataset đã được phát hành công khai.
[Trang bài gốc](https://docs.lib.purdue.edu/cib-conferences/vol2/iss1/48/).
[Định danh nghiên cứu SWIR](https://doi.org/10.1016/j.measurement.2026.122077).

Đóng góp nghiên cứu đề xuất: hệ thống ra quyết định có nguồn bằng chứng và khả
năng từ chối, cộng với active learning được đo bằng ngân sách annotation. Các
nhánh cảm biến sau này phải chứng minh cải thiện so với RGB trên cùng mẫu vật,
và báo cả tỷ lệ có phép đo hợp lệ, trường hợp thiếu cảm biến và mâu thuẫn dữ liệu.
