# Bộ kiểm tra Đà Nẵng — danh sách cần thu thập

Người dùng xác nhận **chưa có ảnh Đà Nẵng** ngày 10/11/2026. Chưa có số đo địa
phương. Danh sách này là đề xuất thu thập tiếp theo, không phải dữ liệu đã có.

## Cảnh cần có

- Chai/lọ, lon, cốc, carton, nắp, túi/màng, giấy, đầu lọc thuốc lá và rác khác;
  ảnh một vật và nhiều vật ở sân trường, đường phố, khu dân cư, bãi biển phù hợp.
- Pin nguyên vẹn được phép chụp an toàn: nhiều kích cỡ, pin cúc áo, màu nhãn,
  góc nhìn, khoảng cách và nền; không dùng cùng một viên pin làm cả train và test.
- Negative khó: bút, bút xóa, ống nhỏ, nắp chai, đồ chơi, linh kiện, lon nhỏ,
  đồ điện tử có thể có pin ẩn. Nhãn thị giác âm tính không có nghĩa không có pin ẩn.
- Ban ngày/thiếu sáng vừa phải, bóng đổ, phản sáng, nền bẩn, vật nhỏ/che khuất,
  ảnh rung hoặc quá tối để kiểm tra khả năng từ chối. Không lại gần vật nguy hại
  chỉ để lấy ảnh rõ hơn.

Mục tiêu phát triển 1.500–3.000 ảnh ở nhiều phiên/thiết bị. Giữ một tập kiểm tra
riêng có cả pin và negative, đủ vật thể/cảnh khác nhau; số ảnh lớn của cùng một
vật hoặc video không thay thế số mẫu độc lập. Chốt ngân sách ảnh/nhãn trước khi
thu thập toàn bộ; đợt thử nhỏ dùng kiểm tra quy trình, không dùng công bố độ tin cậy.

## Giới hạn khi chụp

Không thu gom, mở, chọc thủng, dàn dựng pin hỏng/nóng; không sờ/nhặt kim tiêm,
chất thải lâm sàng, hóa chất hoặc vật nghi nguồn phóng xạ để làm dataset. Không
vào khu xử lý/cơ sở y tế khi chưa được phép. Với đồ y tế, ưu tiên ảnh được cấp
quyền hoặc mô phỏng an toàn do người có chuyên môn chuẩn bị; ghi rõ miền mô phỏng.
Không gán nhãn nhiễm bệnh/không nhiễm bệnh hoặc mức bức xạ bằng quan sát ảnh.

Tránh khuôn mặt, biển số, tên bệnh nhân/đơn thuốc, địa chỉ nhà và vị trí nhạy cảm.
Ghi quyền sử dụng, xử lý dữ liệu nhận dạng/EXIF trước khi chia sẻ; ảnh chỉ lưu trong
bộ dữ liệu được kiểm soát, không đưa lịch sử/ảnh người dùng lên URL công khai.

## Metadata và nhãn bắt buộc

Mỗi ảnh có `image_id`, `physical_object_id`, `scene_id`, `capture_session_id`,
`device_id` đã ẩn danh, ngày thu thập, quyền sử dụng, miền thực/mô phỏng, split,
SHA-256 và trạng thái duyệt. Không cần GPS chính xác để huấn luyện nhận dạng.

- Box theo nhóm vật thể; gán đầy đủ các vật thuộc lớp mục tiêu, kiểm box nằm trong ảnh.
- Vật liệu và tình trạng pin là trường riêng; chưa xác minh thì `unknown`.
- `infection_status=not_determined`, `radiation_status=not_measured` khi không có
  phép xác minh chuyên môn, không thay bằng nhãn giả.
- Một ảnh có thể có nhiều vật. Không tạo box toàn ảnh từ nhãn classification.
- Ít nhất một người gán và một người duyệt các ca pin/vật y tế; ghi bất đồng và
  bỏ khỏi benchmark xác nhận nếu chưa giải quyết, vẫn giữ hồ sơ truy vết.

## Chia tập và sử dụng

Chia theo nhóm **vật thể + cảnh + phiên/video**, giữ tất cả góc chụp/frame/ảnh
augmentation của cùng nhóm trong một split. Khóa test trước train. Chọn ngưỡng
trên validation; khi đã dùng test để quyết định thay đổi thì cần bộ test mới cho
nghiệm thu. So sánh mAP/recall theo nguồn và nhóm khó, không gộp điểm miền khác
để tuyên bố độ chính xác ở Đà Nẵng.

Trước khi đưa ảnh vào pipeline hiện tại, tạo phiên bản dataset mới, audit nhãn,
trùng ảnh và mapping; không ghi đè `waste_multisource_v3` hay checkpoint đã đo.
