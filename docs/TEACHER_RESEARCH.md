# Lựa chọn Teacher — khảo sát ngày 10/11/2026

Đây là lựa chọn để thử nghiệm sau phê duyệt, không phải kết quả inference tại máy.
Không có Teacher/dependency trong hai venv đã kiểm (`transformers`, `groundingdino`,
`sam2`, `qwen_vl_utils`, `llama_cpp` đều absent); không tìm thấy weights tương ứng
trong roots model/cache đã kiểm. Không chứng minh rằng mọi nơi trên máy đều không có model.
GPU đo bằng nvidia-smi: RTX 4050 Laptop **6141 MiB**, lúc kiểm tra GPU dùng 0 MiB.

## Khuyến nghị

**Grounding DINO-T là Teacher box đầu tiên; VLM là bước hỗ trợ ngữ nghĩa tùy chọn;
SAM 2 chỉ thêm khi cần mask hoặc tinh chỉnh biên.** Teacher có thể sai và không
phải ground truth. Chưa thể ước lượng accuracy trên rác Đà Nẵng/9 nhóm từ benchmark khác.

| Lựa chọn | Vai trò và giới hạn | RTX 4050 6 GB / điều kiện |
|---|---|---|
| Grounding DINO-T / Swin-T | Grounding theo prompt, xuất box và detector score; hợp tác vụ định vị. Class từ phrase chỉ ánh xạ theo bảng explicit, phrase ngoài bảng giữ unknown để review. Không dùng từ “litter” gán tất cả thành other_litter. | 172,277,902 parameters theo metadata; weights 689,359,096 bytes (~657 MiB). Là ứng viên hợp lý để thử batch=1/ảnh 640, nhưng chưa đo peak VRAM/latency; không cam kết vừa bộ nhớ. Dùng CPU fallback nếu cần, chạy riêng Teacher và Student. |
| Qwen3-VL-4B-Instruct | VLM hỗ trợ mô tả, OCR/nhãn bao bì và gợi ý vật bị bỏ sót. Tọa độ/score tự báo cần xác minh; không dùng mô tả để suy hóa học pin, bệnh truyền nhiễm hay bức xạ. | Hai safetensors tổng **8,875,719,344 bytes (~8.27 GiB)** đã vượt VRAM chỉ riêng weights. Cần quantization/offload hoặc máy VRAM lớn hơn; mức dùng sau quantization chưa đo. Không đề xuất cài bản BF16 để chạy toàn bộ GPU 6 GB. |
| SAM 2.1 tiny | Segmentation theo prompt point/box; có thể tinh chỉnh mask rồi lấy box. Không tự đặt tên vật thể và không thay detector/semantic Teacher. | Bản tiny 38.9M parameters; cần cài riêng và weights. Trên Windows tài liệu tác giả khuyến nghị WSL Ubuntu; chưa cần thay môi trường hiện tại cho pilot detection. |
| VLM API (Gemini 3.1 Pro Preview; Flash-Lite làm phương án giá thấp) | Review ngữ nghĩa có cấu trúc sau phê duyệt ảnh/nhà cung cấp/ngân sách. Output vẫn là đề xuất, VLM self-confidence không là xác suất hiệu chỉnh. Không gửi ảnh từ tool hiện tại. | Không cần GPU local; cần explicit approval, kiểm quyền ảnh và chính sách dữ liệu. Model alias có thể đổi: lưu model version thực, request ID và raw response receipt. |

Nguồn và metadata trực tiếp:
[Grounding DINO chính thức](https://github.com/IDEA-Research/GroundingDINO),
[HF implementation](https://huggingface.co/docs/transformers/model_doc/grounding-dino),
[Grounding DINO-T metadata](https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny?blobs=true),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct),
[Qwen metadata](https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct?blobs=true),
[SAM 2](https://github.com/facebookresearch/sam2).
Đánh giá khả năng vừa VRAM ở trên là suy luận tài nguyên, không phải benchmark laptop.
Grounding DINO-T được tác giả báo COCO zero-shot AP 48.4; số đó **không phải
accuracy kỳ vọng của bộ rác này**. FPS SAM công bố đo trên A100, không áp cho RTX 4050.

## Artifact cần phê duyệt trước khi tải/cài

Ưu tiên bản HF `IDEA-Research/grounding-dino-tiny`:

- Revision metadata đã đọc: `a2bb814dd30d776dcf7e30523b00659f4f141c71`.
- `model.safetensors`: SHA-256 công bố LFS
  `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`.
- Tải weights ~689 MB, cộng config/tokenizer/processor; chưa tải byte weights nào.
- Cần môi trường Teacher riêng và pin dependency sau kiểm compatibility; không
  cài đè backend/yolo venv. Ưu tiên implementation Transformers có sẵn của HF;
  native repo có bước CUDA extension cần kiểm riêng trên Windows.
- Sau phê duyệt: local-only inference 5–10 ảnh **train trong pilot**, batch=1,
  ghi VRAM/latency/versions/checksums và output thô. Không dùng Student hiện tại
  giả làm Teacher. Nếu OOM, giảm resolution có ghi transform hoặc CPU; không đổi
  tọa độ mà không chiếu lại ảnh nguồn. Đo trước khi chạy cả 150 ảnh.
- Adapter `propose-local` đã viết với offline/pinned-artifact gates, nhưng chưa
  cài dependency/nạp model/chạy inference thật. Preflight thiếu artifact bị chặn,
  không tạo output. Pipeline cũng nhận bundle JSON bằng `import-teacher`,
  không có đường gọi cloud tự động. Xem protocol về manifest local trước khi dùng.

Qwen revision metadata: `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
Nếu chọn bản quantized phải duyệt artifact/nguồn/license và hash riêng; không coi
hash BF16 là xác minh cho một bản quantized khác.

## Chi phí API tham khảo, chưa phát sinh

Đọc bảng giá chính thức ngày 10/11/2026; bản markdown gốc lưu tại
`logs/task03a/google_pricing_source.txt` vì công cụ web không parse content-type này.

| Model / standard tier | Input text/image / 1M tokens | Output gồm thinking / 1M tokens | Ước tính 150 ảnh |
|---|---:|---:|---:|
| Gemini 3.1 Pro Preview, mỗi request ≤200k input tokens | $2 | $12 | **$2.40–4.20** |
| Gemini 3.1 Flash-Lite | $0.25 | $1.50 | **$0.30–0.53** |

Giả định tính toán, **không phải usage đo**: mỗi ảnh một request, tổng input
2,000–5,000 tokens (ảnh + prompt), tổng output gồm thinking 1,000–1,500 tokens.
Công thức `150 * (input_tokens * input_rate + output_tokens * output_rate) / 1e6`.
Độ phân giải/resolution và model ảnh hưởng token; chưa bao gồm retry, thuế,
currency conversion hoặc công cụ search. Cần estimate/count tokens rồi đặt budget
hard cap trước khi gửi. Cả free tier cũng không được dùng nếu chưa duyệt gửi ảnh.
[Bảng giá Google](https://ai.google.dev/gemini-api/docs/pricing).

Teacher local không có phí API nhưng có dung lượng, điện và thời gian xử lý.
Chi phí chính có thể là người kiểm duyệt: giả định 2–5 phút/ảnh là **5–12.5 giờ**
cho 150 ảnh, chưa tính sửa dày và người kiểm thứ hai. Đây là dự toán nguồn lực,
không phải thời gian đã đo.

## Kỳ vọng chất lượng và quyết định dùng Teacher

Chưa có kết quả Teacher thật nên không đưa ra precision/recall hoặc lời hứa tăng mAP.
Pilot trước tiên đo: tỷ lệ proposal đúng class + IoU≥0.5 theo người duyệt, số vật
bị bỏ sót, số box sửa/xóa/thêm, thời gian review và tỷ lệ hold/reject, chia theo
class/source/size. Score giữa DINO và VLM không được gộp hoặc xem là cùng xác suất.

Ưu tiên một nhóm tham chiếu nhỏ được người gán nhãn độc lập trước khi xem Teacher,
double-review bất đồng/unknown và mọi nghi nguy hại. Teacher phải chứng minh tiết
kiệm công sức mà không tăng lỗi nghiêm trọng; đây là điều kiện quyết định mở rộng,
không phải một ngưỡng an toàn đã được xác nhận. Không coi TACO mirror là nguồn độc lập,
không loại trừ được pretraining overlap nếu nhà cung cấp không công bố đủ dữ liệu.
