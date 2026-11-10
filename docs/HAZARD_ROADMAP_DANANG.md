# Pin, rác có nguy cơ và lộ trình Đà Nẵng

Ngày cập nhật: 10/11/2026. Đây là thiết kế và tiêu chí phát triển; các ngưỡng bên
dưới là **mục tiêu**, không phải kết quả đo được. Số đo thực được ghi trong báo cáo run.

## Phân biệt các năng lực

| Năng lực | Phương án | Giới hạn |
|---|---|---|
| Rác nhiều vật | YOLO26n, taxonomy 9 nhóm vật thể TACO/WaRP | Không suy ra polymer, pin, medical từ 9 nhóm |
| Pin ảnh cận cảnh | YOLO26n-cls binary, Kaggle + RecyBat, thử nghiệm phát triển | Không có box, không thấy pin ẩn, chưa xác thực độc lập/Đà Nẵng |
| Pin nhiều vật | Bước kế tiếp: detection bằng box gốc RecyBat24 + ảnh địa phương | Chưa hoàn tất detector riêng; không tạo box giả từ nhãn classification |
| Kim tiêm/rác y tế | Hiện có hướng dẫn khi người dùng báo nghi ngờ | Chưa có model thị giác mới được đánh giá cho nhãn này |
| Nghi phóng xạ | Cảnh báo theo quan sát/ngữ cảnh do người dùng cung cấp | Không có phép đo bức xạ; không xác nhận hay loại trừ từ RGB |
| Nơi xử lý | Registry có nguồn, địa phương và trạng thái xác minh | Có địa chỉ không đồng nghĩa còn hoạt động hay nhận mọi loại rác |

API mới `/api/safety/scan` chỉ dùng checkpoint pin thực, đối chiếu SHA-256 với
báo cáo đánh giá trước khi nạp. Chạy CPU để tránh tranh GPU với training. Không
lưu ảnh upload, không dùng tên tệp làm bằng chứng thị giác. Thiếu model trả 503.
`is_uncertain` luôn true vì chưa có kiểm định ngoài miền và hiệu chỉnh xác suất.

Baseline Kaggle-only v1 thất bại ngoài miền: 0/2.834 ảnh RecyBat vượt ngưỡng 0,95.
Đã kiểm tra đối chiếu CPU API và GPU batch trên cùng ảnh, score gần khớp; không
thấy lỗi tên tệp/đổi kênh ảnh giải thích thất bại. Giữ baseline và tập chẩn đoán;
v2 dùng RecyBat cho phát triển, nên không còn gọi RecyBat là kiểm định ngoài miền
độc lập cho v2. RecyBat v2 tách theo dạng pin + ID viên, không trộn hai mặt/phiên
ánh sáng của cùng viên sang các split. Kaggle thiếu ID vật nên vẫn là giới hạn.

`/api/safety/assess` dựa trên lựa chọn của người dùng. Phân biệt pin nguyên vẹn,
hỏng, nóng/bốc khói. Không gợi ý điểm pin công cộng cho pin hỏng/nóng, rác y tế hay
vật nghi nguồn phóng xạ. Có hướng dẫn theo sáu nhóm vật liệu thông thường.

## Nguồn khoa học và dữ liệu

- [TACO, tác giả, 2020](https://arxiv.org/abs/2003.06975): annotation vật thể trong
  bối cảnh. Mirror Hugging Face không làm tăng số nguồn độc lập; giữ ID/hash nguồn.
- [WaRP, Kaggle](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset):
  bối cảnh băng chuyền. Đánh giá riêng miền này và miền rác ngoài đường.
- [Garbage Classification, Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification):
  nhãn image-level có pin, phù hợp classifier thử nghiệm; không có box để train detection.
- [RecyBat24, Scientific Data 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12098665/):
  box ba dạng pin. Có giới hạn môi trường chụp và thiếu pin cúc áo. Cần tách theo
  **ID viên pin** khi đánh giá tổng quát hóa; ảnh hai mặt cùng viên không độc lập.
  [Zenodo v3](https://zenodo.org/records/15226091) phân biệt bản gốc và augmentation.
- [XBAT+, Scientific Data 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13545264/):
  RGB nhận dạng thiết bị, X-ray phục vụ nhận pin bên trong; không chuyển thành tuyên
  bố camera điện thoại nhìn xuyên thiết bị. Không tự đề xuất vận hành nguồn X-ray.
- [SAM, tài liệu chính thức](https://docs.ultralytics.com/models/sam/): có thể hỗ trợ
  gán mask/nhãn; người gán nhãn phải duyệt. Chưa chạy SAM trong lượt này; không
  xem mask tự sinh là ground truth hoặc bằng chứng nhận dạng chất nguy hại.
- [Medical Waste 4.0, tác giả CNR](https://openportal.isti.cnr.it/doc?id=people______%3A%3A003c0d7646b0b4aa92250ada3f4407d5),
  [Zenodo](https://doi.org/10.5281/zenodo.7643417): nguồn để khảo sát glove/gauze/
  test tube/urine bag; không mặc định có kim tiêm chỉ từ tên dataset. Chưa train
  trong lượt này. Nhãn hình dạng dụng cụ không phải nhãn chứa tác nhân lây nhiễm.
- [Nghiên cứu Deep MW, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8828884/):
  phân biệt tám nhóm đồ y tế, nhưng kết quả tác giả không phải kết quả dự án này;
  phải xác minh quyền truy cập dữ liệu và annotation trước sử dụng.
- [Dataset mô phỏng lấy máu, tác giả, 2026](https://arxiv.org/abs/2602.04624): có
  syringe và annotation polygon, phù hợp nghiên cứu hình dạng. Ảnh từ video thao
  tác mô phỏng, cần chia theo video/phiên và kiểm tra cảnh ngoài miền, không gộp
  tất cả frame rồi chia ngẫu nhiên để báo khả năng phát hiện rác y tế ngoài đường.

Giấy phép và điều kiện của từng phiên bản dữ liệu/model cần được lưu cùng manifest
khi chia sẻ sản phẩm. Không gộp các tập có nhãn không tương thích chỉ để tăng số ảnh.

## Hướng dẫn và điểm thu gom

Nguồn hướng dẫn: [EPA về pin](https://www.epa.gov/recycle/frequent-questions-lithium-ion-batteries),
[EPA về pin hỏng](https://www.epa.gov/electronics-batteries-management/battery-collection-tip-sheet-end-life-household-small-format),
[WHO về rác y tế](https://www.who.int/news-room/fact-sheets/detail/health-care-waste),
[IAEA về nguồn phóng xạ kín](https://www.iaea.org/sites/default/files/sealedradsource1013.pdf),
[hướng dẫn phân loại của Chính phủ](https://xaydungchinhsach.chinhphu.vn/huong-dan-ky-thuat-ve-phan-loai-chat-thai-ran-sinh-hoat-119231107185909363.htm).
Nguồn nước ngoài cung cấp nguyên tắc kỹ thuật, không thay thế quy định/tiếp nhận địa phương.

[FEC](https://fec.edu.vn/tin-tuc/thu-gom-pin-cu-bao-ve-moi-truong) từng công bố
108 Phan Châu Trinh và 808 Ngô Quyền. Nguồn còn truy cập ngày 10/11/2026 nhưng chưa
xác nhận trực tiếp chiến dịch còn mở. Registry ghi `needs_confirmation`, không
có tọa độ/giờ mở cửa giả. Các chiến dịch Cocoon 29/4–29/5/2026 và OCB tới
31/5/2026 đã hết thời gian công bố, không đưa vào danh sách điểm đang nhận.

Để một điểm được coi là xác nhận: người quản lý xác nhận địa chỉ, luồng nhận,
loại pin/hóa chất được nhận, điều kiện từ chối, giờ và ngày xác nhận; xác định
đơn vị tiếp nhận cuối theo loại chất thải. Điểm pin không mặc nhiên nhận rác y tế.
Không có dữ liệu cơ sở y tế/phóng xạ được xác nhận hiện tại để chỉ đường giao nộp.

## Quy trình nâng chất lượng và mốc nghiệm thu đề xuất

1. **Kiểm toán dữ liệu**: duyệt nhãn, box, ảnh gần trùng; thống kê theo nguồn,
   lớp, thiết bị, cảnh và ID vật. Giữ một test độc lập; augmentation chỉ ở train.
   Không đưa kết quả test trở lại vòng chọn hyperparameter.
2. **Bổ sung ảnh Đà Nẵng**: mục tiêu 1.500–3.000 ảnh đa cảnh, ánh sáng và thiết bị,
   gồm nhiều vật, vật nhỏ, che khuất, ảnh không có rác; ít nhất 200 vật pin khác
   nhau và negative khó. Đây là mục tiêu thu thập, chưa có bộ ảnh này. Không thu
   gom hoặc dàn dựng pin hỏng, kim tiêm đã dùng, vật nghi phóng xạ để chụp.
3. **Train theo miền**: baseline cố định, ablation TACO/WaRP/địa phương, cân bằng
   minibatch thay vì để băng chuyền lấn át ảnh ngoài đường. Thử 3 seed sau khi
   chất lượng nhãn ổn; báo trung bình/độ dao động và chi phí thực đo.
4. **Chấm điểm độc lập**: box mAP50 và mAP50–95, precision/recall từng lớp; pin
   báo TP/FP/FN/TN và interval, số ảnh cần người xem; không dùng accuracy chung để
   che bỏ sót pin. Mục tiêu khởi điểm: recall pin ≥95%, precision ≥90% trên test
   địa phương; cần cỡ mẫu/interval phù hợp, chưa phải ngưỡng an toàn công nghiệp.
5. **Chọn ngưỡng bằng validation**: hiệu chỉnh confidence và đánh giá abstention.
   Bất định/ảnh xấu chuyển người xem; âm tính không có nghĩa vật thể an toàn.
6. **Kiểm định hướng dẫn**: tất cả ca pin nóng/hỏng, sharps, nghi phóng xạ phải
   tránh thùng tái chế thường; không sinh chẩn đoán bệnh/liều bức xạ. Đơn vị chuyên
   môn duyệt nội dung trước sử dụng thực tế; unit test không thay đánh giá chuyên gia.
7. **Pilot có giám sát tại Đà Nẵng**: đo latency p50/p95 trên thiết bị triển khai,
   lỗi theo bối cảnh và tỷ lệ chuyển người xem, xác minh điểm thu gom trước khi
   chỉ đường. Không tự động ra lệnh cho robot/thiết bị phân loại nguy hại.
8. **Mở rộng Bắc/Nam**: bổ sung registry có nguồn theo địa phương và test riêng
   từng miền; kiểm tra lại mức tiếp nhận, dữ liệu mới và sai lệch, giữ model version
   có thể rollback. Chưa có địa chỉ được xác minh cho các tỉnh thành khác.

Các câu hỏi còn cần nhóm dự án quyết định sau baseline: người duyệt nhãn tại
Đà Nẵng, đối tác thu gom có thể xác nhận tiếp nhận, thiết bị đích và yêu cầu latency.
Không cần tài khoản dịch vụ trả phí để hoàn tất baseline cục bộ này.
