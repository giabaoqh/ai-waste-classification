# Báo Cáo Nghiên Cứu và Lựa Chọn Tập Dữ Liệu (DATASET_RESEARCH.md)

> **Dự án**: AI-Based Waste Recognition and Recycling Recommendation System  
> **Mục tiêu**: Phân loại hình ảnh rác thải vào 6 nhóm mục tiêu: **Plastic (Nhựa), Paper (Giấy), Metal (Kim loại), Glass (Thủy tinh), Organic (Hữu cơ), Other (Khác / Rác nguy hại)**.  
> **Phần cứng mục tiêu**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).  
> **Tài liệu**: Báo cáo khảo sát các tập dữ liệu công khai, đánh giá tính pháp lý, ánh xạ nhãn, chiến lược chia dữ liệu và khuyến nghị kỹ thuật.

---

## 1. Yêu Cầu Cốt Lõi Về Dữ Liệu Của Đề Tài

Hệ thống yêu cầu nhận diện chính xác 6 nhóm rác thải sinh hoạt phổ biến:
1. **Plastic**: Chai nhựa, hộp xốp, túi nilon, đồ dùng nhựa dùng một lần.
2. **Paper**: Bìa carton, giấy báo, sách vở, cốc giấy sạch.
3. **Metal**: Vỏ lon nhôm, hộp thiếc, nắp chai kim loại, phế liệu sắt thép nhỏ.
4. **Glass**: Chai lọ thủy tinh trong suốt, xanh, nâu, mảnh vỡ thủy tinh.
5. **Organic**: Vỏ hoa quả, thức ăn thừa, rau củ hỏng, lá cây, bã cà phê.
6. **Other**: Rác thải hỗn hợp không tái chế được, rác nguy hại (pin, bóng đèn, đồ điện tử nhỏ), quần áo cũ, giày dép hỏng.

> [!IMPORTANT]
> **Nguyên tắc phân tách**: Mô hình học sâu (Deep Learning) chỉ chịu trách nhiệm dự đoán nhãn thị giác (`category`) kèm xác suất (`confidence`). Cơ sở tri thức hướng dẫn phân loại và quy định tái chế được quản lý độc lập tại tầng backend/database để dễ dàng cập nhật theo quy định địa phương mà không phải huấn luyện lại mạng nơ-ron.

---

## 2. Khảo Sát & Đánh Giá Các Tập Dữ Liệu Công Khai (Candidate Datasets)

Dưới đây là 4 tập dữ liệu thị giác máy tính về rác thải phổ biến nhất được cộng đồng nghiên cứu quốc tế sử dụng:

| Tiêu chí | 1. RealWaste (Khuyên dùng) | 2. Garbage Classification (12 classes) | 3. TrashNet (Stanford) | 4. TACO (Trash in Context) |
| :--- | :--- | :--- | :--- | :--- |
| **Nguồn phát hành** | UCI ML Repository & MDPI *Information* (2023) | Kaggle (Mostafa Mohamed / Abla) | Stanford University CS229 (Gary Thung & Mindy Yang, 2016) | Tacodataset.org / GitHub (Pedro Proença & Pedro Simões) |
| **Giấy phép (License)** | **CC BY 4.0** (Tự do học tập, nghiên cứu và thương mại khi trích dẫn) | **ODbL** (Open Database License) / © Original Authors | **MIT License** | **CC BY 4.0** |
| **Tổng số ảnh** | **4,752 ảnh** | **~15,150 ảnh** | **2,527 ảnh** | **1,500 ảnh** (4,784 bounding boxes) |
| **Số lớp nguyên bản** | 9 lớp | 12 lớp | 6 lớp | 60 lớp (Instance Segmentation) |
| **Bối cảnh ảnh** | Rác thực tế tại bãi phân loại và thực địa (ánh sáng thực, góc chụp tự nhiên) | Đa dạng: nền trắng kết hợp ảnh thu thập thực tế | Nền trắng đồng nhất (studio), mỗi ảnh 1 vật thể | Rác rơi vương vãi ngoài môi trường (đường phố, bãi biển, rừng) |
| **Hỗ trợ đủ 6 lớp?** | **CÓ (100%)** | **CÓ (100%)** | **KHÔNG** (Thiếu hoàn toàn lớp Organic) | Cần crop bounding box; phân bố lớp rất lệch (long-tail) |
| **Mức độ tương thích VRAM 4GB** | **Rất cao** (Huấn luyện nhanh, không nghẽn RAM/Disk) | **Tốt** (Cần batch size 16-32) | **Rất cao** (nhưng thiếu lớp) | **Thấp** (Thiết kế cho Object Detection, không phải Image Classification) |

---

### Chi Tiết Từng Ứng Viên

#### Ứng viên 1: RealWaste Dataset (UCI Machine Learning Repository)
- **Nguồn chính thức**: [UCI Machine Learning Repository - RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste)  
- **Bài báo gốc**: *"RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning"* (MDPI *Information*, 2023).
- **Giấy phép**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Phân bố số lượng ảnh thực tế theo lớp**:
  - `Cardboard`: 461 ảnh
  - `Paper`: 500 ảnh
  - `Glass`: 420 ảnh
  - `Metal`: 790 ảnh
  - `Plastic`: 921 ảnh
  - `Food Organics`: 411 ảnh
  - `Vegetation`: 436 ảnh
  - `Miscellaneous Trash`: 495 ảnh
  - `Textile Trash`: 318 ảnh
  - **Tổng cộng**: 4,752 ảnh màu.
- **Ưu điểm**:
  - Tỷ lệ các lớp tương đối cân bằng (từ ~400 đến 900 ảnh/lớp).
  - Có đầy đủ thành phần rác hữu cơ thực phẩm (`Food Organics`) và thực vật (`Vegetation`).
  - Hình ảnh phản ánh bối cảnh rác thực tế, chất lượng cao, có bài báo khoa học thẩm định bình duyệt.
- **Nhược điểm**: Một số ảnh có độ bẩn hoặc biến dạng cao của rác thực tế, cần tăng cường độ tương phản khi tiền xử lý.

---

#### Ứng viên 2: Garbage Classification (12 classes) - Kaggle
- **Nguồn chính thức**: [Kaggle - Garbage Classification (12 classes) by Mostafa Mohamed](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Giấy phép**: Open Database License (ODbL).
- **Quy mô**: ~15,150 ảnh với 12 danh mục:
  - `biological` (~985 ảnh)
  - `paper` (~1,050 ảnh), `cardboard` (~890 ảnh)
  - `plastic` (~865 ảnh)
  - `metal` (~769 ảnh)
  - `green-glass`, `brown-glass`, `white-glass` (~2,000 ảnh tổng)
  - `batteries`, `clothes`, `shoes`, `trash` (~8,000+ ảnh)
- **Ưu điểm**: Số lượng ảnh lớn (~15k ảnh), có lớp `biological` đại diện cho rác hữu cơ, có lớp `batteries` cho rác nguy hại.
- **Nhược điểm**: Được tổng hợp từ web scraping và gộp nhiều dataset mở; tồn tại một số ảnh trùng lặp gần (near-duplicates) và mất cân bằng số lượng giữa các nhóm.

---

#### Ứng viên 3: TrashNet (Stanford CS229)
- **Nguồn chính thức**: [GitHub - garythung/trashnet](https://github.com/garythung/trashnet)
- **Giấy phép**: MIT License.
- **Quy mô**: 2,527 ảnh (Paper: 594, Glass: 501, Plastic: 482, Metal: 410, Cardboard: 403, Trash: 137).
- **Ưu điểm**: Dataset kinh điển, ảnh chụp trên nền phông trắng sạch, dễ hội tụ khi mới bắt đầu thử nghiệm mô hình baseline.
- **Hạn chế nghiêm trọng**: **Không có nhóm rác hữu cơ (Organic)**. Lớp `trash` chỉ có 137 ảnh (quá ít). Không thể dùng đơn độc để giải bài toán 6 nhóm của đồ án.

---

#### Ứng viên 4: TACO (Trash Annotations in Context)
- **Nguồn chính thức**: [tacodataset.org](http://tacodataset.org) / [GitHub - pedropro/TACO](https://github.com/pedropro/TACO)
- **Giấy phép**: CC BY 4.0.
- **Đặc điểm**: 1,500 ảnh chứa 4,784 nhãn bounding box/polygon dành cho bài toán Object Detection / Instance Segmentation ngoài thực địa.
- **Hạn chế**: Không phù hợp trực tiếp cho bài toán Image Classification cơ bản (1 ảnh chụp 1 món rác). Để dùng được, phải viết pipeline crop từng bounding box, nhiều box kích thước quá nhỏ hoặc bị che khuất một phần.

---

## 3. Quy Tắc Ánh Xạ Nhãn Sang 6 Nhóm Mục Tiêu (Label Mapping Strategy)

Để tránh hiện tượng gán nhãn sai lệch (silent mislabeling), quy tắc gom nhóm từ dataset gốc sang 6 nhóm mục tiêu phải được định nghĩa tường minh bằng mã nguồn (code-level mapping dictionary):

### Bảng Ánh Xạ Chuẩn Cho RealWaste (Khuyên Dùng Chính):

```python
REALWASTE_MAPPING = {
    # 1. Nhựa
    "Plastic": "plastic",
    
    # 2. Giấy (Gộp cả giấy thường và bìa carton)
    "Paper": "paper",
    "Cardboard": "paper",
    
    # 3. Kim loại
    "Metal": "metal",
    
    # 4. Thủy tinh
    "Glass": "glass",
    
    # 5. Hữu cơ (Gộp thức ăn thừa và phụ phẩm thực vật)
    "Food Organics": "organic",
    "Vegetation": "organic",
    
    # 6. Rác khác / Hỗn hợp / Nguy hại
    "Miscellaneous Trash": "other",
    "Textile Trash": "other"
}
```

**Thống kê số lượng sau ánh xạ trên RealWaste (4,752 ảnh)**:
- `plastic`: 921 ảnh (19.4%)
- `paper`: 961 ảnh (20.2%)
- `metal`: 790 ảnh (16.6%)
- `glass`: 420 ảnh (8.8%)
- `organic`: 847 ảnh (17.8%)
- `other`: 813 ảnh (17.1%)

> [!TIP]
> **Đánh giá độ cân bằng**: Sau khi ánh xạ, 5 trong 6 nhóm đều có từ 790 đến 961 ảnh. Nhóm `glass` có 420 ảnh (vẫn chiếm gần 9%), hoàn toàn xử lý được thông qua kỹ thuật cân bằng trọng số hàm mất mát (`Class-weighted CrossEntropyLoss`) hoặc Data Augmentation.

---

## 4. Chiến Lược Dữ Liệu Khuyến Nghị Cho Laptop RTX 3050 (4 GB VRAM)

### 4.1. Khuyến nghị tập dữ liệu chính (Primary Choice)
- **Lựa chọn**: Sử dụng **RealWaste Dataset** làm tập dữ liệu huấn luyện và đánh giá cốt lõi.
- **Lý do kỹ thuật**:
  1. **Quy mô hoàn hảo cho GPU 4GB VRAM**: 4,752 ảnh kích thước 224x224 chỉ chiếm ~1.5 GB RAM khi nạp theo batch, huấn luyện Transfer Learning (MobileNetV3 hoặc EfficientNet-B0) với batch size 16–32 chỉ tiêu tốn khoảng **1.8 GB – 2.4 GB VRAM** (nằm an toàn dưới trần 4.0 GB VRAM của card RTX 3050 Laptop).
  2. **Thời gian huấn luyện hợp lý**: Mỗi epoch chỉ mất khoảng 20–35 giây trên RTX 3050. Toàn bộ quá trình huấn luyện 25 epoch chỉ mất khoảng 10–15 phút, giúp nhóm thử nghiệm và tinh chỉnh siêu tham số (hyperparameters) nhanh chóng.
  3. **Độ tin cậy học thuật**: Dataset có bài báo khoa học xuất bản năm 2023, giấy phép CC BY 4.0 chuẩn mực, không lo ngại về vấn đề bản quyền khi báo cáo đồ án tốt nghiệp.

### 4.2. Phương án mở rộng (Optional Extension)
- Nếu sau khi huấn luyện baseline với RealWaste, nhóm muốn bổ sung thêm dữ liệu rác sạch chụp nền trắng (như chai nhựa, lon nước ngọt còn mới): Nhóm có thể bổ sung thêm subset từ **TrashNet** (chỉ lấy Plastic, Metal, Paper, Glass) để tăng độ đa dạng cho mô hình khi người dùng chụp vật thể trong nhà.

---

## 5. Phân Chia Tập Dữ Liệu & Chống Rò Rỉ Dữ Liệu (Data Leakage Prevention)

### 5.1. Tỷ lệ phân chia chuẩn (Reproducible Split)
- **Tỷ lệ**: **70% Train - 15% Validation - 15% Test** (hoặc 80/10/10).
- **Kỹ thuật phân chia**: Bắt buộc sử dụng phân tầng (**Stratified Sampling**) theo 6 lớp mục tiêu và cố định `random_state=42` để bảo đảm tỷ lệ từng lớp ở cả 3 tập là đồng nhất:
  - **Train set (~3,326 ảnh)**: Dùng để cập nhật gradient mạng nơ-ron.
  - **Validation set (~713 ảnh)**: Dùng để chọn checkpoint tốt nhất (Early Stopping, Model Checkpoint).
  - **Test set (~713 ảnh)**: Được khóa kín, **chỉ đánh giá 1 lần duy nhất** sau khi đã chọn xong mô hình tối ưu.

### 5.2. Biện pháp chống rò rỉ dữ liệu (Anti-Leakage Measures)
1. **Lọc ảnh trùng lặp (Deduplication via Perceptual Hash - pHash)**:
   - Trước khi chia tập, chạy thuật toán tính mã băm hình ảnh (MD5 và pHash với Hamming Distance < 3) để loại bỏ các ảnh giống nhau chụp liên tiếp từ cùng một góc hoặc cùng một vật thể.
   - Đảm bảo không có trường hợp một vật thể rác xuất hiện ở cả tập Train lẫn tập Test.
2. **Quy trình Augmentation nghiêm ngặt**:
   - Các phép biến đổi dữ liệu (Random Crop, Random Rotation, ColorJitter) **chỉ được áp dụng trên tập Train**.
   - Tập Validation và Test tuyệt đối chỉ áp dụng: `Resize(256)` -> `CenterCrop(224)` -> `ToTensor()` -> `Normalize()`.

---

## 6. Đánh Giá Toàn Diện Vấn Đề Giấy Phép & Rủi Ro Pháp Lý Của RealWaste

### 6.1. Chi tiết khác biệt giấy phép giữa các kênh phát hành (License Discrepancy)
Khi khảo sát nguồn gốc của RealWaste, có một điểm cần làm rõ về mặt pháp lý:
- **Kênh UCI Machine Learning Repository (Chính thức)**: Phân phối dưới giấy phép **CC BY 4.0** (Creative Commons Attribution 4.0 International). Giấy phép này cho phép chia sẻ, tái sử dụng và chuyển thể cho mọi mục đích (kể cả thương mại) miễn là có ghi nhận tác giả hợp lý.
- **Kênh GitHub tác giả gốc (AILab-UOW / University of Wollongong)**: Chỉ định giấy phép **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International). Giấy phép này giới hạn cho mục đích phi thương mại (NonCommercial) và yêu cầu chia sẻ tương tự (ShareAlike).

> [!IMPORTANT]
> **Nguyên tắc an toàn tối đa (Conservative Principle)**:
> Để đảm bảo an toàn pháp lý tuyệt đối cho đồ án tốt nghiệp đại học, nhóm chúng ta sẽ tuân thủ theo phiên bản có điều kiện nghiêm ngặt hơn: **CC BY-NC-SA 4.0**.

---

### 6.2. Phân tích tuân thủ cho Đồ Án Tốt Nghiệp (Capstone Project)

| Điều khoản Creative Commons | Ý nghĩa pháp lý | Mức độ tuân thủ của Đồ Án | Hướng xử lý cụ thể |
| :--- | :--- | :--- | :--- |
| **Attribution (BY)** | Phải ghi nhận tác giả gốc và trích dẫn bài báo khoa học. | **Bắt buộc tuân thủ 100%** | Ghi rõ trích dẫn tác giả (*Sam Single, Saeid Iranmanesh, Raad Raad*) trong `README.md`, tài liệu báo cáo đồ án và trang thông tin của hệ thống. |
| **Non-Commercial (NC)** | Không được sử dụng tập dữ liệu cho mục đích thương mại hoặc kiếm tiền trực tiếp. | **Tuân thủ 100%** | Đồ án tốt nghiệp là công trình nghiên cứu học thuật phi lợi nhuận của sinh viên đại học. Không bán phần mềm, không thu phí dịch vụ. |
| **Share-Alike (SA)** | Nếu tạo và phân phối tập dữ liệu phái sinh, phải áp dụng giấy phép tương đương. | **Tuân thủ 100%** | Nhóm không tự ý tái phân phối (redistribute) tập dữ liệu. Nếu nhóm công bố tập dữ liệu đã lọc sạch (preprocessed dataset), tập đó cũng sẽ được gán nhãn CC BY-NC-SA 4.0. |

---

### 6.3. Chiến lược lưu trữ an toàn đối với GitHub Repository (Public Repo)

Repository của nhóm trên GitHub (`giabaoqh/ai-waste-classification`) hiện đang ở chế độ **Public**:
1. **Tuyệt đối KHÔNG đưa tập ảnh thô lên GitHub**:
   - File `.gitignore` đã chặn triệt để thư mục `data/` và `dataset/`.
   - Toàn bộ dữ liệu RealWaste khi tải về sẽ chỉ nằm ở ổ cứng cục bộ (`D:\...\data\realwaste`), không bao giờ bị commit hay push lên GitHub.
2. **Đảm bảo tính tái lập (Reproducibility)**:
   - Thay vì lưu trữ lại file ảnh trên repo công khai, nhóm sẽ cung cấp một file script tải dữ liệu tự động (`download_data.py`) lấy trực tiếp từ nguồn chính thức của UCI/Authors kèm kiểm tra mã băm MD5/SHA256. Bất kỳ ai clone repo đều có thể tự tải dữ liệu về máy của họ một cách hợp pháp.

---

### 6.4. Các rủi ro kỹ thuật khác & Biện pháp giảm thiểu

1. **Rủi ro rác hỗn hợp (Ambiguous / Multi-label waste)**:
   - Trong thực tế, nhiều món rác có cấu tạo từ nhiều chất liệu (ví dụ: chai thủy tinh có nắp kim loại, hộp sữa giấy tráng nhôm/nhựa).
   - *Biện pháp giảm thiểu*: Gán nhãn theo vật liệu chiếm diện tích/khối lượng lớn nhất; đồng thời thiết lập ngưỡng tin cậy (Confidence Threshold). Nếu Softmax không có lớp nào đạt > 50%, hệ thống sẽ xếp vào nhóm **Other / Cần kiểm tra thủ công**.
2. **Rủi ro chênh lệch bối cảnh chụp (Domain Shift)**:
   - Ảnh trong dataset có thể chụp dưới điều kiện ánh sáng công nghiệp hoặc nền bãi rác, trong khi ảnh người dùng cuối chụp qua webcam/điện thoại có thể bị mờ hoặc ngược sáng.
   - *Biện pháp giảm thiểu*: Tích hợp Data Augmentation đa dạng (Gaussian Blur, Random Perspective, ColorJitter) trong pipeline huấn luyện PyTorch.

---

## 7. Trích Dẫn Nguồn Chính Thức (Official Citations)

Nhóm cam kết trích dẫn đầy đủ nguồn tác giả gốc theo định dạng chuẩn:

```bibtex
@article{single2023realwaste,
  title={RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning},
  author={Single, Sam and Iranmanesh, Saeid and Raad, Raad},
  journal={Information},
  volume={14},
  number={12},
  pages={659},
  year={2023},
  publisher={MDPI},
  doi={10.3390/info14120659}
}
```
- **Nguồn chính thức**: [UCI Machine Learning Repository - RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste)
- **Kho mã nguồn tác giả**: [GitHub - AILab-UOW/RealWaste](https://github.com/AILab-UOW/RealWaste)
- **Giấy phép áp dụng**: CC BY-NC-SA 4.0 / CC BY 4.0.

---

## 8. Kết Luận & Kế Hoạch Tiếp Theo

1. **Kết luận pháp lý**: Việc sử dụng tập dữ liệu **RealWaste** cho đồ án tốt nghiệp đại học là **hoàn toàn hợp pháp, minh bạch và an toàn** khi nhóm tuân thủ tiêu chí phi thương mại (NC), ghi nhận tác giả đầy đủ (BY) và không tải ảnh thô lên GitHub Public.
2. **Cam kết an toàn**: Hiện tại **chưa tải bất kỳ dữ liệu nào về máy**, chưa chỉnh sửa mã nguồn ứng dụng, chưa thay đổi Git branch hay tạo commit/push.
3. **Đề xuất bước tiếp theo (sau khi bạn xác nhận)**:
   - Soạn thảo script chuẩn bị môi trường tải dữ liệu `download_data.py` (chỉ tải về thư mục `data/` được `.gitignore` bảo vệ, có cơ chế xác thực checksum).
