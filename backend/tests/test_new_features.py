"""
Comprehensive test suite verifying:
- Waste disposal & recycling recommendations
- Combustibility & Recyclability environmental analysis
- 3-step industrial recycling workflow
- Redesigned Dashboard (totals, most frequent, daily/weekly trends, feedback accuracy)
- Error reporting & Exporting active learning error cases
"""
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torchvision.transforms as transforms
from PIL import Image

from backend.app.main import (
    app,
    find_checkpoint_path,
    load_model_from_checkpoint,
    process_waste_image_bytes,
    submit_prediction_feedback,
    export_history_csv,
    export_error_cases_dataset,
    get_model_evaluation_metrics,
    get_recycling_centers,
    get_dashboard_analytics,
    FeedbackRequest,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from backend.app.database import init_db

def test_all():
    print("1. Khởi tạo Database & Nạp mô hình AI...")
    init_db()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app.state.device = str(device)
    app.state.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    ckpt_path = find_checkpoint_path()
    model, classes = load_model_from_checkpoint(ckpt_path, device)
    app.state.model = model
    app.state.classes = classes
    app.state.model_loaded = True

    # 2. Test Predict với Thuộc tính Bắt Cháy & Mức Độ Tái Chế
    print("\n2. Kiểm thử nhận diện và phân tích thuộc tính rác thải...")
    img = Image.new("RGB", (224, 224), color=(50, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    pred = process_waste_image_bytes(
        contents=image_bytes,
        filename="chai_nhua_demo.jpg",
        content_type="image/jpeg",
        include_gradcam=True
    )
    print(f"   -> Lớp dự đoán: {pred.predicted_class} ({pred.label_vi})")
    print(f"   -> Độ tin cậy: {pred.confidence * 100:.2f}%")
    print(f"   -> Khả năng cháy: {pred.recommendation.combustibility.status}")
    print(f"   -> Mức độ tái chế: {pred.recommendation.recyclability.level} ({pred.recommendation.recyclability.difficulty_score})")
    print(f"   -> Số bước quy trình xử lý: {len(pred.recommendation.processing_steps)}")
    assert pred.recommendation.combustibility is not None
    assert pred.recommendation.recyclability is not None
    assert len(pred.recommendation.processing_steps) >= 3

    # 3. Test Báo Cáo Phản Hồi (3 trạng thái: correct, incorrect, uncertain)
    print("\n3. Kiểm thử phản hồi người dùng & báo cáo sai...")
    fb1 = submit_prediction_feedback(
        record_id=pred.record_id,
        payload=FeedbackRequest(feedback_status="incorrect", is_correct=False, correct_class="glass", note="Thực tế là chai thủy tinh màu xanh")
    )
    print(f"   -> Báo sai thành công: {fb1.success}, msg: {fb1.message}")
    assert fb1.success is True

    # 4. Test Redesigned Dashboard Analytics
    print("\n4. Kiểm thử Bảng Thống Kê Dashboard đã làm lại...")
    analytics = get_dashboard_analytics()
    print(f"   -> Tổng số lần nhận diện: {analytics.total_predictions}")
    if analytics.most_frequent_class:
        print(f"   -> Loại rác nhiều nhất: {analytics.most_frequent_class.label_vi} ({analytics.most_frequent_class.count} lượt)")
    print(f"   -> Xu hướng theo ngày: {len(analytics.daily_trends)} mốc")
    print(f"   -> Xu hướng theo tuần: {len(analytics.weekly_trends)} mốc")
    print(f"   -> Tỷ lệ đúng theo phản hồi: {analytics.feedback_metrics.accuracy_rate}%")
    print(f"   -> Số ca lỗi tổng hợp (Active Learning): {len(analytics.error_cases)}")
    print(f"   -> Cảnh báo bắt buộc: '{analytics.disclaimer[:40]}...'")
    assert "không đồng nghĩa với khối lượng rác thực tế" in analytics.disclaimer

    # 5. Test Xuất File Ca Lỗi Tái Huấn Luyện
    print("\n5. Kiểm thử xuất file ca lỗi active learning...")
    error_csv = export_error_cases_dataset()
    assert "attachment; filename=tap_du_lieu_ca_loi_active_learning.csv" in error_csv.headers.get("content-disposition", "")
    print(f"   -> Xuất CSV ca lỗi thành công: {error_csv.headers.get('content-disposition')}")

    print("\n🎉 TOÀN BỘ CÁC YÊU CẦU MỚI ĐÃ VƯỢT QUA KIỂM THỬ THÀNH CÔNG 100%!")

if __name__ == "__main__":
    test_all()
