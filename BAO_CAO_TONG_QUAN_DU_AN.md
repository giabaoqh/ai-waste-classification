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
   - 4.1. Mô hình phân loại 7 nhóm rác (MobileNetV3-Large)
   - 4.2. Phát hiện & khoanh vùng đa vật thể (SSDLite320 + Two-Stage + Scene Fallback)
   - 4.3. Hiệu chuẩn độ tin cậy (Confidence Calibration & Temperature Scaling)
   - 4.4. Cơ chế lọc vật thể không phải rác (Semantic Non-Waste Gatekeeper)
   - 4.5. Giải thích quyết định AI (Explainable AI - Grad-CAM)
   - 4.6. Cơ sở tri thức phân loại rác Việt Nam (Luật BVMT 2020 & TCVN)
   - 4.7. Bản đồ báo cáo điểm nóng rác thải đô thị (Crowdsourced Smart City Litter Map)
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
| **Phân loại ảnh** | **MobileNetV3-Large** | Backbone trích xuất đặc trưng và phân loại 7 nhóm rác (kèm Rác thải Y tế nguy hại). Tối ưu cho tốc độ cao và dung lượng nhẹ (~16 MB). |
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

### 4.1. Mô hình Phân Loại Rác 7 Lớp (MobileNetV3-Large)
* **Vì sao chọn MobileNetV3?** Thay vì dùng ResNet-50 hay VGG quá nặng nề (hàng chục triệu tham số, chậm trên CPU), MobileNetV3 sử dụng kiến trúc **Inverted Residuals** kết hợp khối **Squeeze-and-Excitation (SE)** và hàm kích hoạt **Hard-Swish**. Mô hình chỉ có khoảng **5.4 triệu tham số**, kích thước file vỏn vẹn **16.8 MB**, đạt tốc độ suy luận siêu tốc (~40ms trên GPU, ~120ms trên CPU văn phòng).
* **Cơ chế Transfer Learning:** Khởi tạo từ trọng số tiền huấn luyện ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT`), thay thế lớp Linear cuối cùng bằng `nn.Linear(in_features=960, out_features=7)`.
* **Cân bằng trọng số lớp (Class Weights):** Tự động tính toán theo công thức nghịch đảo tần suất mẫu:
  $$w_c = \frac{N}{C \times N_c}$$
  nhằm chống lệch dự đoán về các lớp chiếm số lượng lớn.

---

### 4.2. Phát Hiện & Khoanh Vùng Đa Vật Thể Nâng Cao (Multi-Scale Slicing Two-Stage Pipeline)
Hệ thống cho phép quét một bức ảnh chứa đống rác phức tạp, dày đặc hoặc nhiều vật thể rác nằm đè lên nhau:
1. **Multi-Scale Slicing Inference (Mẫu kiến trúc SAHI - Slicing Aided Hyper Inference):**
   * *Vấn đề thực tế:* Khi mạng nơ-ron thu nhỏ ảnh gốc (1000px+) về 320x320, các vỏ lon bị bẹp, túi nilon nhăn, mẩu rác nhỏ chỉ còn 10-15px, bị mất chi tiết và trôi khỏi trường tiếp nhận (receptive field).
   * *Giải pháp:* Hệ thống chia ảnh thành 6 cửa sổ trượt phân giải cao (Toàn cảnh + 4 góc phần tư gối đầu 16% + Vùng trung tâm trọng điểm), chạy detector trên từng slice rồi quy đổi tọa độ về không gian gốc.
2. **Giai đoạn 1 (Proposal Generation & Threshold Thích ứng):** Thu nhận đề xuất từ 33 danh mục ứng viên rác thải COCO với ngưỡng nhạy 0.10 (tăng gấp 3 lần độ nhạy đối với rác bị biến dạng/vùi lấp).
3. **Giai đoạn 2 (Class-Agnostic Non-Maximum Suppression):** Áp dụng NMS toàn cục với ngưỡng IoU = 0.38 để hợp nhất các box trùng giữa các góc slice nhưng vẫn tách bạch các món rác nằm sát cạnh nhau.
4. **Giai đoạn 3 (Safe Cropping & Padding):** Cắt từng vùng ảnh trực tiếp từ ảnh gốc chất lượng cao với tỉ lệ đệm viền 6%, kẹp toạ độ trong viền [0, W] và [0, H], loại bỏ nhiễu < 16px.
5. **Giai đoạn 4 (Fine-Grained 7-Class Classification):** Đưa từng crop vào MobileNetV3-Large để phân loại chính xác 7 nhóm rác (nhựa, giấy, kim loại, thủy tinh, hữu cơ, y tế, khác). Giới hạn tối đa 35 vật thể rõ ràng nhất, sắp xếp theo độ tự tin kết hợp.
6. **Tiện ích Người dùng:** Hỗ trợ kéo thả ảnh trực tiếp (Drag & Drop) từ bất kỳ thư mục hay trình duyệt nào vào trang web mà không cần mở hộp thoại chọn tệp, đồng thời hỗ trợ dán ảnh chụp màn hình ngay lập tức qua phím tắt Ctrl + V.
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

### 4.5. Bản Chất Nhận Diện Của AI Qua Lớp Phủ Nhiệt & Heatmap (Explainable AI - Grad-CAM)

#### a. Bản chất cốt lõi: Lớp phủ nhiệt (Heatmap) thực chất là gì?
* **Không phải cảm biến nhiệt vật lý:** Lớp phủ nhiệt trong hệ thống không đo nhiệt độ độ C, mà là kết quả của kỹ thuật **Trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI)** dựa trên giải thuật **Grad-CAM (Gradient-weighted Class Activation Mapping)**.
* **Xóa bỏ định kiến "Hộp đen" (Black-Box):** Bình thường, mạng nơ-ron sâu đưa ra nhãn dự đoán kèm xác suất (ví dụ `plastic 98%`) mà không cho biết lý do. Heatmap chiếu sáng trực tiếp các vùng pixel không gian trong bức ảnh đã kích hoạt mạnh nhất các nơ-ron ra quyết định, biến AI thành **"Hộp kính trong suốt" (Glass-Box)**.

#### b. Ý nghĩa các dải màu trên Heatmap (Bảng màu Jet Colormap)
Thuật toán ánh xạ cường độ kích hoạt qua bảng màu nhiệt Jet và hòa trộn đè lên ảnh gốc (Alpha Blending tỷ lệ 45%):
```
[Xanh lam đậm / Tối] ──► [Xanh lục] ──► [Vàng rực] ──► [Đỏ rực / Đỏ tươi]
(Hoàn toàn bỏ qua)    (Ít quan trọng)  (Chú ý cao)   (Quyết định cốt lõi)
```
* **Vùng Đỏ rực / Vàng rực (Cường độ cao nhất):** Nơi mạng nơ-ron tập trung năng lượng cao độ để đưa ra quyết định (ví dụ: mép vặn nắp chai nhựa, khoen mở lon nhôm, cùi cuống quả táo). Nếu che hoặc xóa vùng này, điểm số dự đoán của AI sẽ giảm sút nghiêm trọng.
* **Vùng Xanh lục (Cường độ trung bình):** Các chi tiết hỗ trợ nhận diện ngữ cảnh xung quanh vật thể.
* **Vùng Xanh lam đậm / Tối đen (Cường độ = 0):** Vùng AI hoàn toàn "bỏ qua" và không sử dụng làm căn cứ (mặt bàn, nền đất, tường nhà, bóng đổ).

#### c. Bản chất toán học 4 bước của giải thuật Grad-CAM (trong file `backend/app/gradcam.py`):
1. **Trích xuất bản đồ đặc trưng không gian ($A^k$):**  
   Tại tầng tích chập sâu nhất của MobileNetV3-Large (`model.features[-1]`), mạng lưu giữ thông tin ngữ nghĩa hình ảnh phong phú nhất với **960 kênh đặc trưng (channels)** kích thước không gian $7 \times 7$ pixel. Mỗi kênh được ký hiệu là $A^k$.
2. **Tính đạo hàm lan truyền ngược (Backpropagation Gradients):**  
   Tính đạo hàm riêng của điểm số logit $y^c$ (cho lớp rác $c$) theo từng phần tử của bản đồ đặc trưng:
   $$\frac{\partial y^c}{\partial A^k_{i,j}}$$
   *Ý nghĩa:* Đạo hàm này cho biết nếu pixel tại tọa độ $(i, j)$ của kênh $k$ thay đổi nhẹ, điểm số phân loại rác $c$ sẽ tăng hay giảm bao nhiêu.
3. **Tính trọng số quan trọng $\alpha_k^c$ (Global Average Pooling):**  
   Lấy trung bình cộng đạo hàm trên toàn bộ không gian chiều cao $H$ và chiều rộng $W$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A^k_{i,j}}$$
   *Ý nghĩa:* $\alpha_k^c$ định lượng độ đóng góp sống còn của toàn bộ kênh thứ $k$ đối với quyết định dự đoán lớp $c$.
4. **Tổ hợp tuyến tính và áp dụng hàm kích hoạt ReLU:**  
   $$L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k=1}^{960} \alpha_k^c A^k \right)$$
   *Vì sao bắt buộc dùng hàm ReLU?* Ta chỉ giữ lại các đặc trưng **đóng góp tích cực (làm tăng điểm số)** cho lớp rác $c$. Những đặc trưng thuộc về các lớp khác (làm giảm điểm số) sẽ có giá trị âm và bị hàm ReLU triệt tiêu về $0$.

Bản đồ $L^c_{\text{Grad-CAM}}$ sau đó được **nội suy song tuyến tính (Bilinear Interpolation)** lên kích thước gốc ($224 \times 224$), chuẩn hóa về $[0, 1]$ và phủ màu lên ảnh ban đầu.

#### d. Ba giá trị thực tiễn cốt lõi khi báo cáo giảng viên:
1. **Phát hiện và chứng minh không học vẹt (Chống Clever Hans):** Chứng minh AI phân loại chai nước vì nó thấy vỏ nhựa và nắp chai, chứ không phải vì cái bàn màu trắng bên dưới.
2. **Giải thích các ca nhầm lẫn thị giác:** Giải thích vì sao chai nhựa trong suốt bị nhầm thành thủy tinh (do vùng đỏ rực tập trung vào các vệt sáng phản quang chói lóa).
3. **Minh bạch hóa & Tạo dựng niềm tin (Trustworthiness):** Người dân và các chuyên gia môi trường hoàn toàn yên tâm vì có thể "nhìn thấy" cơ sở logic thị giác của AI trước khi phân loại rác.

---

### 4.6. Cơ Sở Tri Thức Phân Loại Rác Việt Nam (Decoupled Knowledge Base)
Hệ thống **tách rời hoàn toàn** giữa AI dự đoán và tri thức xử lý tại file [`waste_disposal_kb.json`](file:///d:/AI-Based%20Waste%20Recognition%20and%20Recycling%20Recommendation%20System/backend/app/waste_disposal_kb.json), bám sát:
1. **Luật Bảo vệ Môi trường 2020** và **Nghị định 08/2022/NĐ-CP**.
2. **TCVN 13952:2024** về quy chuẩn quản lý và tái chế chất thải nhựa.
3. **Cơ chế ghi đè an toàn rác nguy hại (Hazardous Safety Overrides):**
   * Khi phát hiện từ khóa hoặc dấu hiệu của **Pin, ắc quy (`battery`)** hay **Rác điện tử (`e_waste`)**: Hệ thống cưỡng bức chuyển thùng rác sang **Điểm thu gom rác nguy hại đặc thù**, cảnh báo nguy cơ cháy nổ, tuyệt đối không cho phép bỏ vào thùng rác sinh hoạt thông thường.

---

### 4.7. Bản Đồ Báo Cáo Điểm Nóng Rác Thải Đô Thị (Crowdsourced Smart City Litter Map)
Hệ thống mở rộng từ một công cụ nhận diện cá nhân thành **nền tảng quản lý môi trường đô thị thông minh (Smart City Platform)**:
1. **Thu thập dữ liệu cộng đồng (Crowdsourcing):**
   * Cho phép người dân chụp ảnh hiện trường các bãi rác tự phát, thùng rác công cộng quá tải hoặc rác thải y tế vứt bừa bãi.
   * Định vị tọa độ địa lý chính xác qua HTML5 Geolocation API và ghim trực tiếp lên bản đồ OpenStreetMap (Leaflet).
2. **AI Tự động thẩm định & Phân cấp khẩn cấp:**
   * Ngay khi ảnh hiện trường được tải lên, pipeline AI chạy phân tích thành phần rác thải.
   * Nếu phát hiện có rác y tế (`medical`) hay rác nguy hại, hệ thống tự động gắn cờ **Khẩn cấp (`critical`)** với viền phát sáng đỏ nhấp nháy trên bản đồ để ưu tiên điều phối xử lý trước.
3. **Cơ chế xác thực cộng đồng & Điều phối xử lý:**
   * Người dân khu vực lân cận có thể bấm **"👍 Xác nhận điểm nóng (+1 upvote)"** để tăng độ tin cậy và mức độ ưu tiên của phản ánh.
   * Đơn vị thu gom môi trường hoặc đội tình nguyện viên có thể cập nhật trạng thái theo chu trình: **Chờ xử lý (Pending) ⏳ ➔ Đang dọn dẹp (In Progress) 🚚 ➔ Đã dọn sạch (Resolved) ✅**.
4. **Gắn liền với Gamification (Điểm Sinh Thái):**
   * Người dân khi báo cáo một điểm nóng rác thải chính xác được cộng ngay **+50 Điểm Sinh Thái (Eco Points)**, khuyến khích tinh thần công dân xanh.

---

## 5. KẾT QUẢ THỰC NGHIỆM & ĐO LƯỜNG ĐỊNH LƯỢNG (7 LỚP)

Kết quả kiểm thử độc lập trên toàn bộ **2.223 ảnh Test độc lập (chưa từng xuất hiện trong quá trình huấn luyện)** sau 10 Epochs trên GPU NVIDIA GeForce RTX 3050 Laptop:

### 5.1. Bảng số liệu chi tiết từng lớp
| Danh mục | Số mẫu Test (Support) | Precision | Recall | F1-Score | Đánh giá năng lực thực tế |
|:---|:---:|:---:|:---:|:---:|:---|
| 💉 **Rác y tế (`medical`)** | **235** | **97.05%** | **97.87%** | **97.46%** | **Cao nhất toàn hệ thống (230/235 mẫu đúng). Bắt trúng kim tiêm, khẩu trang, găng tay.** |
| 🍎 **Hữu cơ (`organic`)** | 276 | 95.74% | 97.83% | **96.77%** | Nhận diện cực nhạy rau củ, thức ăn thừa, cành lá. |
| 🍾 **Thủy tinh (`glass`)** | 366 | 96.94% | 95.36% | **96.14%** | Phân biệt xuất sắc chai lọ thủy tinh màu và chai trong. |
| 📦 **Giấy (`paper`)** | 421 | 95.52% | 96.20% | **95.86%** | Bắt trúng bìa carton và tài liệu in. |
| 🗑️ **Rác khác (`other`)** | 421 | 96.77% | 92.40% | **94.53%** | Phân loại ổn định pin, đồ vải và rác tổng hợp. |
| 🧴 **Nhựa (`plastic`)** | 269 | 92.28% | 88.85% | **90.53%** | Cải thiện độ chính xác so với bản cũ (+0.81%). |
| 🥫 **Kim loại (`metal`)** | 235 | 84.94% | 93.62% | **89.07%** | Recall đạt 93.62%, bắt trúng vỏ lon và mảnh kim loại. |
| **TOÀN HỆ THỐNG** | **2.223** | **94.18%** | **94.59%** | **94.34%** | **Test Accuracy: 94.56%** *(Val Acc: 94.94%)* |

### 5.2. Ma trận nhầm lẫn thực nghiệm 7 lớp (Confusion Matrix)
```
                DỰ ĐOÁN (PREDICTED)
           plastic   paper   metal   glass  organic   other  medical
plastic  |   239       2      17       6       1       3       1
paper    |     2     405       5       0       2       5       2
metal    |     7       4     220       1       1       1       1
glass    |     9       1       5     349       0       2       0
organic  |     0       2       1       1     270       2       0
other    |     1       8      11       2       7     389       3
medical  |     1       2       0       1       1       0     230  ★ (Đúng 230/235)
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

### ❓ Câu 7: Nhóm xử lý bài toán nhận diện Rác thải y tế lây nhiễm (`medical`) như thế nào và căn cứ pháp lý là gì?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, rác thải y tế (khẩu trang, găng tay phẫu thuật, bơm kim tiêm nhọn, vỉ thuốc quá hạn) là danh mục nguy hại sinh học đặc biệt, nếu phân loại nhầm sang rác tái chế nhựa hay kim loại sẽ đe dọa trực tiếp đến tính mạng công nhân vệ sinh và nguy cơ bùng phát dịch bệnh. 
  > Nhóm đã tích hợp **1.563 ảnh rác y tế thực tế từ tập dữ liệu TrashBox** và mở rộng mô hình lên **7 nhóm phân loại chuẩn**, đạt độ chính xác kiểm thử trên nhóm y tế lên tới **F1-Score: 97.46% (chính xác 230/235 mẫu test)**. 
  > Về mặt quy chuẩn, hệ thống căn cứ trực tiếp theo **Thông tư 20/2021/TT-BYT** và **QCVN 02:2012/BTNMT**: hướng dẫn bắt buộc bỏ vào **thùng rác màu vàng (Biohazard)**, vật sắc nhọn kim tiêm phải bỏ vào **hộp kháng thủng**, đồng thời kích hoạt **Cơ chế Ghi đè An toàn (Hazardous Override)** và cộng **+25 Điểm Xanh** để khuyến khích thói quen bảo vệ an toàn sinh học cộng đồng."*

---


*Tài liệu được biên soạn và bảo chứng tự động bởi hệ thống AI-Waste Engineering.*


### 💡 Câu 8: Dự án này ngoài việc cho người dân quét rác thì có thể tích hợp với các hệ thống quản lý đô thị thông minh (Smart City) như thế nào?
* **Cách trả lời chuẩn:**
  > *"Thưa thầy/cô, dự án đã phát triển sẵn module **Bản đồ Điểm nóng Rác thải Đô thị (Smart City Litter Map)** theo mô hình Crowdsourcing kết hợp Trí tuệ Nhân tạo. Người dân khi bắt gặp bãi rác tự phát hay thùng rác quá tải có thể chụp ảnh và gửi tọa độ GPS lên hệ thống. AI sẽ tự động phân tích chất liệu rác trong ảnh, phát hiện nếu có rác y tế hoặc chất độc hại để tự động nâng cấp mức độ khẩn cấp. Dữ liệu này được hiển thị trực tiếp theo thời gian thực trên bản đồ số, cho phép các cơ quan quản lý đô thị hoặc công ty vệ sinh môi trường điều phối xe rác tối ưu lộ trình và người dân theo dõi được tiến độ dọn dẹp sạch sẽ qua các trạng thái minh bạch."*
