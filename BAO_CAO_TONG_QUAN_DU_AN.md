# 📘 BÁO CÁO TOÀN DIỆN VỀ DỰ ÁN AI-WASTE (DÀNH CHO BÁO CÁO & BẢO VỆ ĐỒ ÁN)

> **Tên đề tài:** Hệ Thống Nhận Diện Rác Thải Và Gợi Ý Tái Chế Thông Minh Dựa Trên Trí Tuệ Nhân Tạo  
> **Tên tiếng Anh:** AI-Based Waste Recognition and Recycling Recommendation System  
> **Mục đích tài liệu:** Tài liệu tra cứu cốt lõi gom toàn bộ kiến thức, kiến trúc công nghệ, dữ liệu, giải thuật, số liệu thực nghiệm và kịch bản trả lời câu hỏi phản biện của giảng viên.

---

## 📑 MỤC LỤC
1. [Tổng Quan & Ý Nghĩa Thực Tiễn Của Dự Án](#1-tổng-quan--ý-nghĩa-thực-tiễn)
2. [Kiến Trúc Hệ Thống & Tech Stack (Sử Dụng Những Gì?)](#2-kiến-trúc-hệ-thống--tech-stack)
3. [Dữ Liệu Huấn Luyện (Data Ở Đâu & Có Bao Nhiêu?)](#3-dữ-liệu-huấn-luyện-data)
4. [Các Trụ Cột Công Nghệ & Giải Thuật Cốt Lõi (AI Pipeline)](#4-các-trụ-cột-công-nghệ--giải-thuật-cốt-lõi)
   - 4.1. Mô hình phân loại 6 nhóm rác (MobileNetV3-Large)
   - 4.2. Phát hiện & khoanh vùng đa vật thể (SSDLite320 + Two-Stage + Scene Fallback)
   - 4.3. Hiệu chuẩn độ tin cậy (Confidence Calibration & Temperature Scaling)
   - 4.4. Cơ chế lọc vật thể không phải rác (Semantic Non-Waste Gatekeeper)
   - 4.5. Giải thích quyết định AI (Explainable AI - Grad-CAM)
   - 4.6. Cơ sở tri thức phân loại rác Việt Nam (Luật BVMT 2020 & TCVN)
5. [Kết Quả Thực Nghiệm & Đo Lường Định Lượng](#5-kết-quả-thực-nghiệm--đo-lường-định-lượng)
6. [Quy Trình Vận Hành & Khởi Động Dự Án (Demo Sống)](#6-quy-trình-vận-hành--khởi-động-dự-án)
7. [Bộ Câu Hỏi Phản Biện Của Giảng Viên & Cách Trả Lời Điểm 10](#7-bộ-câu-hỏi-phản-biện-của-giảng-viên)

---

## 1. TỔNG QUAN & Ý NGHĨA THỰC TIỄN

### 1.1. Bài toán thực tế
* **Thực trạng:** Luật Bảo vệ Môi trường 2020 của Việt Nam (có hiệu lực bắt buộc phân loại rác tại nguồn từ cuối năm 2024) yêu cầu người dân phân loại rác thành các nhóm riêng biệt. Tuy nhiên, đại đa số người dân gặp khó khăn trong việc nhận diện chất liệu (nhựa PET hay nhựa tạp chất, lon nhôm hay giấy tráng kim loại) và cách xử lý an toàn (rác nguy hại như pin, bóng đèn).
* **Giải pháp của dự án:** Xây dựng ứng dụng AI toàn diện giúp người dùng chỉ cần chụp ảnh bằng điện thoại hoặc tải ảnh lên:
  1. Tự động nhận diện chính xác chất liệu rác thải (đơn vật thể hoặc đa vật thể cùng lúc).
  2. Gợi ý thùng rác phù hợp theo quy định pháp luật Việt Nam.
  3. Cung cấp hướng dẫn sơ chế (rửa sạch, tháo nắp, làm bẹp) và đánh giá khả năng cháy / giá trị kinh tế tái chế.
  4. Cảnh báo an toàn tuyệt đối với rác nguy hại (pin, ắc quy, thiết bị điện tử hỏng).

---

## 2. KIẾN TRÚC HỆ THỐNG & TECH STACK

Dự án được xây dựng theo mô hình kiến trúc Client-Server hiện đại, phân tách độc lập giữa AI Engine, Backend API và Web/Mobile Client:

```
[Camera Điện Thoại / Trình Duyệt Web]
                 │
                 ▼  (HTTP RESTful / Multi-part Upload)
       [FastAPI Backend - Port 8000]
                 │
  ┌──────────────┼──────────────────────────────┐
  │              │                              │
  ▼              ▼                              ▼
[SSDLite320]   [MobileNetV3-Large]     [Knowledge Base Engine]
Object Proposal  Classifier & Grad-CAM   Luật BVMT 2020 & TCVN
  │              │                              │
  └──────────────┼──────────────────────────────┘
                 ▼
       [SQLite Database] ──► Lưu lịch sử, Active Learning Feedback
```

### Chi tiết các công nghệ sử dụng:
| Tầng hệ thống | Công nghệ / Thư viện | Vai trò cụ thể trong dự án |
|:---|:---|:---|
| **AI Framework** | **PyTorch 2.14 + Torchvision** (CUDA 12.6) | Xây dựng, huấn luyện và chạy suy luận (inference) mô hình nơ-ron sâu trên GPU RTX 3050 Laptop. |
| **Phân loại ảnh** | **MobileNetV3-Large** | Backbone trích xuất đặc trưng và phân loại 6 nhóm rác. Tối ưu cho tốc độ cao và dung lượng nhẹ (~16 MB). |
| **Phát hiện vật thể**| **SSDLite320-MobileNetV3** | Định vị bounding box đa vật thể, crop từng vùng đối tượng phục vụ pipeline Two-Stage. |
| **Trực quan hóa XAI**| **Grad-CAM** | Sinh bản đồ nhiệt (heatmap) làm sáng tỏ vùng không gian mà AI tập trung nhìn vào. |
| **Backend API** | **FastAPI + Uvicorn** | Cung cấp RESTful API hiệu năng cao, bất đồng bộ (async), tự động sinh tài liệu Swagger UI (`/docs`). |
| **Cơ sở dữ liệu** | **SQLite 3 + WAL Mode** | Lưu trữ lịch sử nhận diện, hình ảnh base64, phản hồi người dùng (correct/incorrect) phục vụ Active Learning. |
| **Frontend Web** | **React 18 + Vite** | Giao diện Single Page Application (SPA), hỗ trợ camera thời gian thực, responsive trên máy tính & điện thoại. |
| **Khởi động một chạm**| **Windows Batch Script (`run_app.bat`)** | Tự động dò IP mạng LAN, chạy ngầm FastAPI & Vite, mở trình duyệt tự động. |

---

## 3. DỮ LIỆU HUẤN LUYỆN (DATA)

### 3.1. Nguồn gốc dữ liệu
Dự án tích hợp và chuẩn hóa từ 2 nguồn dữ liệu học thuật quốc tế uy tín:
1. **RealWaste Dataset:** Tập dữ liệu rác thải ngoài đời thực được thu thập và gán nhãn bởi các nhà nghiên cứu Đại học Nam Úc (University of South Australia), chụp trên nền trắng sạch với nhiều góc độ tự nhiên.
2. **Kaggle Garbage Classification 12 Classes:** Tập dữ liệu ảnh phong phú trên internet bao gồm đa dạng các biến thể bao bì đồ uống, hộp thực phẩm, đồ gia dụng.

### 3.2. Quy mô và phân chia tập dữ liệu (Đã làm sạch rò rỉ)
Toàn bộ dữ liệu được chuẩn hóa về **6 danh mục rác tiêu chuẩn** và chia theo tỷ lệ phân tầng (Stratified Split) nghiêm ngặt:
* **Tập Train (Huấn luyện):** **9.236 mẫu ảnh** (đã loại bỏ 6 ảnh trùng lặp SHA256 chống học vẹt).
* **Tập Validation (Thẩm định):** **1.978 mẫu ảnh** (dùng chọn epoch tốt nhất và tối ưu tham số hiệu chuẩn nhiệt độ $T^*$).
* **Tập Test Độc Lập:** **1.988 mẫu ảnh** (khóa cố định, dùng đánh giá năng lực thực tế khách quan).
* **Tổng quy mô ảnh:** **13.202 mẫu ảnh**.

### 3.3. Bảng ánh xạ 6 nhóm danh mục rác:
| STT | Tên lớp (English) | Tên tiếng Việt | Ví dụ vật thể đại diện trong tập dữ liệu |
|:---:|:---|:---|:---|
| 1 | `plastic` | Nhựa (Plastic) | Chai nước khoáng PET, ly nhựa trà sữa, nắp chai, can nhựa, khay thức ăn. |
| 2 | `paper` | Giấy (Paper) | Bìa carton, giấy báo, tạp chí, cốc giấy, túi giấy kraft, tài liệu văn phòng. |
| 3 | `metal` | Kim loại (Metal) | Lon nước ngọt (nhôm), hộp cá hộp (thiếc), nắp khoen, đinh ốc, phế liệu sắt. |
| 4 | `glass` | Thủy tinh (Glass) | Chai bia, chai rượu vang, hũ thủy tinh thực phẩm (trắng, xanh lá, nâu). |
| 5 | `organic` | Hữu cơ (Organic) | Vỏ chuối, cùi táo, bã cà phê, rau củ quả hư hỏng, thức ăn thừa. |
| 6 | `other` | Rác thải khác (Other) | Túi nilon phức hợp, vải vụn, xốp EPS, gốm sứ, rác tổng hợp không phân loại. |

---

## 4. CÁC TRỤ CỘT CÔNG NGHỆ & GIẢI THUẬT CỐT LÕI

### 4.1. Mô hình Phân Loại Rác (MobileNetV3-Large)
* **Vì sao chọn MobileNetV3?** Thay vì dùng ResNet-50 hay VGG quá nặng nề (hàng chục triệu tham số, chậm trên CPU), MobileNetV3 sử dụng kiến trúc **Inverted Residuals** kết hợp khối **Squeeze-and-Excitation (SE)** và hàm kích hoạt **Hard-Swish**. Mô hình chỉ có khoảng **5.4 triệu tham số**, kích thước file vỏn vẹn **16.8 MB**, đạt tốc độ suy luận siêu tốc (~40ms trên GPU, ~120ms trên CPU văn phòng).
* **Cơ chế Transfer Learning:** Khởi tạo từ trọng số tiền huấn luyện ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT`), thay thế lớp Linear cuối cùng bằng `nn.Linear(in_features=960, out_features=6)`.
* **Cân bằng trọng số lớp (Class Weights):** Tự động tính toán theo công thức nghịch đảo tần suất mẫu:
  $$w_c = \frac{N}{C \times N_c}$$
  nhằm chống lệch dự đoán về các lớp chiếm số lượng lớn.

---

### 4.2. Phát Hiện & Khoanh Vùng Đa Vật Thể (Two-Stage Detection Pipeline)
Hệ thống cho phép quét một bức ảnh chứa nhiều vật thể rác nằm cạnh nhau:
1. **Giai đoạn 1 (Proposal):** Mạng `SSDLite320-MobileNetV3` quét toàn bộ ảnh, đề xuất các bounding box của 29 danh mục ứng viên rác thải.
2. **Giai đoạn 2 (Non-Maximum Suppression):** Áp dụng NMS với ngưỡng IoU = 0.45 để loại bỏ các hộp chồng lấn, chỉ giữ lại các hộp tối ưu nhất.
3. **Giai đoạn 3 (Safe Cropping & Padding):** Cắt từng vùng ảnh với tỉ lệ đệm (padding) 6%, kẹp toạ độ trong khung viền `[0, W]` và `[0, H]`, loại bỏ nhiễu nhỏ hơn 16px.
4. **Giai đoạn 4 (Fine-Grained Classification):** Đưa từng crop ảnh vào mạng MobileNetV3-Large để xác định chất liệu và độ tin cậy.
5. **Cơ chế Fallback Bãi Phế Liệu Quy Mô Lớn (Bulk Scrapyard Scene Fallback):**
   * *Vấn đề thực tế:* Khi chụp đống sắt vụn hoặc bãi phế liệu, detector thông thường bị ảo giác gán nhãn `boat` do vỏ kim loại cong.
   * *Giải pháp độc quyền:* Khi phát hiện cảnh hỗn độn không có bóng người/thú cưng, hệ thống tự động gọi Classifier đánh giá toàn cảnh. Nếu xác suất phế liệu kim loại hoặc rác tổng hợp chiếm ưu thế, hệ thống tự động khoanh vùng tổng thể và định danh là **"Bãi phế liệu sắt thép / Rác tổng hợp quy mô lớn"**, tuyệt đối không báo nhầm "không có rác".

---

### 4.3. Hiệu Chuẩn Độ Tin Cậy (Confidence Calibration & Temperature Scaling)
* **Hiện tượng Overconfidence:** Mạng nơ-ron sâu thường đưa ra độ tin cậy Softmax cao vọt (99-100%) ngay cả khi dự đoán sai.
* **Giải pháp Temperature Scaling:** Đưa tham số nhiệt độ $T > 0$ vào hàm Softmax:
  $$\hat{p}_i = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$
  * Đã tối ưu hóa $T^*$ trên tập Validation độc lập bằng thuật toán L-BFGS, đạt giá trị tối ưu:
    $$T^* = 1.6348$$
  * **Hiệu quả:** Kéo giảm sai số kỳ vọng **ECE (Expected Calibration Error) từ 2.12% xuống còn 1.19%** (giảm 43.9% độ lệch ảo).
* **Cơ chế Lọc Bất Định (Top-2 Margin Uncertainty):**
  * Tính khoảng cách chênh lệch giữa xác suất cao nhất và xác suất thứ nhì:
    $$\Delta = p_{\text{top1}} - p_{\text{top2}}$$
  * Nếu $\Delta < 0.15$ hoặc độ tin cậy $< 65\%$, hệ thống tự động kích hoạt cờ `is_uncertain = True`, hiển thị nhãn vàng *"⚠️ Cần kiểm tra lại"* để tránh người dùng vứt rác sai lầm.

---

### 4.4. Cơ Chế Lọc Vật Thể Phi Rác (Semantic Non-Waste Gatekeeper)
* Khi người dùng vô tình chụp mặt người, bàn tay, thú cưng (chó, mèo) hoặc nội thất phòng khách (sofa, giường), hệ thống sử dụng mạng semantic gatekeeper ngăn chặn ngay từ đầu, từ chối phân loại sai và yêu cầu chụp lại ảnh rác thật.

---

### 4.5. Giải Thích Quyết Định AI (Explainable AI - Grad-CAM)
* Sử dụng đạo hàm của lớp đặc trưng cuối cùng (`features[-1]`) để tạo nhiệt đồ màu (Jet Colormap):
  * Vùng màu **Đỏ / Vàng:** Nơi nơ-ron mạng tập trung cao độ (nhãn hiệu chai, nắp kim loại, mép rách giấy).
  * Vùng màu **Xanh lam / Tối:** Vùng nền xung quanh bị mạng bỏ qua.
* Giúp người dùng và giảng viên kiểm chứng AI nhìn vào chi tiết thật của rác chứ không phải nhìn vào nền ảnh.

---

### 4.6. Cơ Sở Tri Thức Phân Loại Rác Việt Nam (Decoupled Knowledge Base)
Hệ thống **tách rời hoàn toàn** giữa AI dự đoán và tri thức xử lý tại file [`waste_disposal_kb.json`](file:///d:/AI-Based%20Waste%20Recognition%20and%20Recycling%20Recommendation%20System/backend/app/waste_disposal_kb.json), bám sát:
1. **Luật Bảo vệ Môi trường 2020** và **Nghị định 08/2022/NĐ-CP**.
2. **TCVN 13952:2024** về quy chuẩn quản lý và tái chế chất thải nhựa.
3. **Cơ chế ghi đè an toàn rác nguy hại (Hazardous Safety Overrides):**
   * Khi phát hiện từ khóa hoặc dấu hiệu của **Pin, ắc quy (`battery`)** hay **Rác điện tử (`e_waste`)**: Hệ thống cưỡng bức chuyển thùng rác sang **Điểm thu gom rác nguy hại đặc thù**, cảnh báo nguy cơ cháy nổ, tuyệt đối không cho phép bỏ vào thùng rác sinh hoạt thông thường.

---

## 5. KẾT QUẢ THỰC NGHIỆM & ĐO LƯỜNG ĐỊNH LƯỢNG

Kết quả kiểm thử độc lập trên toàn bộ **1.988 ảnh Test chưa từng xuất hiện trong quá trình huấn luyện**:

### 5.1. Bảng số liệu chi tiết từng lớp
| Danh mục | Số mẫu Test (Support) | Precision | Recall | F1-Score | Đánh giá |
|:---|:---:|:---:|:---:|:---:|:---|
| **Giấy (`paper`)** | 421 | 93.75% | 96.20% | **94.96%** | Nhận diện xuất sắc bề mặt carton và chữ in. |
| **Kim loại (`metal`)** | 235 | 89.92% | 94.89% | **92.34%** | Nhận diện tốt vỏ lon, nắp hộp kim loại. |
| **Thủy tinh (`glass`)** | 366 | 96.40% | 95.08% | **95.74%** | Rất chuẩn xác với chai lọ màu trong suốt. |
| **Hữu cơ (`organic`)** | 276 | 95.73% | 97.46% | **96.59%** | Tỷ lệ nhận diện thức ăn thừa, hoa quả rất cao. |
| **Khác (`other`)** | 421 | 97.25% | 92.40% | **94.76%** | Gom các loại rác tạp chất, xốp, cao su. |
| **Nhựa (`plastic`)** | 269 | 90.23% | 89.22% | **89.72%** | Lớp khó nhất do nhựa trong suốt phản quang giống thủy tinh. |
| **TOÀN HỆ THỐNG** | **1.988** | **93.88%** | **94.21%** | **94.02%** | **Test Accuracy: 94.27%** *(Baseline: 95.98%)* |

### 5.2. Ma trận nhầm lẫn thực nghiệm (Confusion Matrix)
```
                DỰ ĐOÁN (PREDICTED)
           plastic   paper   metal   glass  organic   other
plastic  |   240       5       9       9       3       3
paper    |     6     405       5       0       3       2
metal    |     7       3     223       2       0       0
glass    |     6       2       5     348       1       4
organic  |     1       4       0       0     269       2
other    |     6      13       6       2       5     389
```

---

## 6. QUY TRÌNH VẬN HÀNH & KHỞI ĐỘNG DỰ ÁN

Hệ thống được đóng gói với kịch bản khởi động thông minh 1 chạm [`run_app.bat`](file:///d:/AI-Based%20Waste%20Recognition%20and%20Recycling%20Recommendation%20System/run_app.bat):
1. **Kiểm tra môi trường:** Tự động đối soát `backend\.venv\Scripts\python.exe` và `frontend\node_modules`.
2. **Dò IP LAN tự động:** Trích xuất địa chỉ IP nội bộ của card mạng Wi-Fi (ví dụ: `192.168.1.x`).
3. **Chạy song song 2 dịch vụ:**
   * Backend API: Chạy uvicorn tại cổng `8000` (`http://localhost:8000/docs`).
   * Frontend Client: Chạy Vite với cờ `--host` tại cổng `5173` (`http://localhost:5173`).
4. **Truy cập trên điện thoại:** Người dùng và thầy cô có thể dùng camera điện thoại kết nối cùng mạng Wi-Fi truy cập `http://<IP_MAY_TINH>:5173` để quét rác trực tiếp ngoài đời thực.

---

## 7. BỘ CÂU HỎI PHẢN BIỆN CỦA GIẢNG VIÊN & CÁCH TRẢ LỜI ĐIỂM 10

### ❓ Câu 1: Tại sao em chọn MobileNetV3-Large mà không dùng ResNet-50 hay YOLOv8?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, mục tiêu của dự án là một hệ thống hỗ trợ người dân và triển khai thực tế trên môi trường web/mobile với tài nguyên biên hạn chế. ResNet-50 có kích thước tới gần 100 MB và ~25 triệu tham số, tốc độ suy luận chậm và tốn RAM. Trong khi đó, MobileNetV3-Large chỉ có ~5.4 triệu tham số, dung lượng nhẹ 16 MB nhưng đạt độ chính xác tương đương (>94-95% trên test set). Đối với bài toán phát hiện đa vật thể, nhóm kết hợp SSDLite320 (cũng phát triển trên MobileNet) để tạo pipeline Two-Stage, vừa đảm bảo tốc độ real-time ~50ms, vừa giúp phân tách rõ ràng giữa khâu định vị (Detector) và khâu nhận diện chất liệu (Classifier)."*

### ❓ Câu 2: Làm sao hệ thống biết một bức ảnh chụp người hay chụp cảnh mà từ chối, không gắn nhãn rác bừa bãi?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, hệ thống sử dụng cơ chế Semantic Gatekeeper ở tầng tiền xử lý. Bằng việc phân tích các lớp ngữ nghĩa thông qua bộ detector COCO và mạng lọc Out-of-Distribution, nếu ảnh chứa các đối tượng như con người (`person`), thú cưng (`dog`, `cat`), hoặc nội thất sinh hoạt sạch sẽ, hệ thống sẽ chặn ngay từ đầu, trả về cờ cảnh báo đối tượng phi rác và hướng dẫn người dùng chụp đúng trọng tâm vật thể rác."*

### ❓ Câu 3: Làm thế nào em xử lý hiện tượng mô hình AI quá tự tin (Overconfidence)?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, mạng nơ-ron phân loại bằng hàm Softmax tiêu chuẩn luôn gặp nhược điểm overconfidence (cho ra xác suất 99% kể cả khi phán đoán sai). Dự án đã giải quyết vấn đề này bằng kỹ thuật **Temperature Scaling**. Nhóm đã tối ưu hóa tham số nhiệt độ $T^* = 1.6348$ trên tập validation độc lập thông qua giải thuật L-BFGS, giúp kéo giảm sai số kỳ vọng ECE từ 2.12% xuống 1.19%. Ngoài ra, hệ thống áp dụng cơ chế **Top-2 Margin Thresholding** ($\Delta < 0.15$): nếu khoảng cách giữa lớp thứ nhất và lớp thứ nhì quá hẹp, hệ thống lập tức gắn cờ cảnh báo không chắc chắn (`is_uncertain = True`) để nhắc người dùng kiểm tra lại."*

### ❓ Câu 4: Nếu người dùng chụp một bãi phế liệu sắt vụn hoặc đống rác hỗn hợp lớn thì hệ thống phản hồi ra sao?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, trước đây các mô hình detector COCO thông thường sẽ bị ảo giác gán nhãn đống sắt gỉ là 'thuyền' (`boat`) và bị bộ lọc loại bỏ. Nhóm đã nghiên cứu và phát triển thành công cơ chế **Scene-Level Bulk Waste Fallback**. Khi detector gặp cảnh phức tạp có độ tương phản cao nhưng không có vật thể đơn lẻ, hệ thống tự động kích hoạt bộ phân loại toàn cảnh MobileNetV3. Nếu xác nhận đây là bãi phế liệu hoặc cụm rác lớn, hệ thống sẽ tự động khoanh vùng tổng thể, định danh đúng là 'Bãi phế liệu sắt thép / Rác tổng hợp' và xuất đầy đủ khuyến nghị thu gom phế liệu quy mô lớn."*

### ❓ Câu 5: Căn cứ vào đâu mà hệ thống đưa ra hướng dẫn bỏ vào thùng rác nào và các bước sơ chế?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, nhóm tuyệt đối không hardcode cứng các quy tắc vào trọng số mô hình AI mà xây dựng riêng một module **Knowledge Base** độc lập (`waste_disposal_kb.json`). Toàn bộ quy chuẩn về màu thùng rác, bước sơ chế (rửa sạch chai, tháo nắp nhựa, làm bẹp lon) đều được đối soát và trích xuất trực tiếp từ **Luật Bảo vệ Môi trường 2020 (Nghị định 08/2022/NĐ-CP)** và tiêu chuẩn quốc gia **TCVN 13952:2024**. Đặc biệt, hệ thống có quy tắc ghi đè an toàn: phát hiện pin hoặc rác điện tử sẽ lập tức hướng dẫn mang đến điểm thu gom nguy hại riêng chứ không được bỏ vào thùng rác thường."*

### ❓ Câu 6: Dự án làm thế nào để tự cải thiện trong tương lai khi có thêm dữ liệu người dùng?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, hệ thống đã tích hợp sẵn cơ chế **Active Learning Loop**. Mỗi khi người dùng nhận kết quả và bấm phản hồi 'Báo sai' kèm nhãn thực tế trên giao diện, thông tin này được lưu tự động vào cơ sở dữ liệu SQLite. Quản trị viên có thể bấm nút xuất file CSV các ca lỗi thực tế để xem xét và bổ sung đúng vào tập huấn luyện theo bản đặc tả dữ liệu mục tiêu (`targeted_collection_spec.json`), giúp các vòng huấn luyện tiếp theo nâng cao độ chính xác một cách có chủ đích."*

---

*Tài liệu được biên soạn và bảo chứng tự động bởi hệ thống AI-Waste Engineering.*
