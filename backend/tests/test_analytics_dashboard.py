"""
Automated Test Suite for Waste Recognition Analytics Dashboard (Phase 4).

Covers:
- Database aggregation logic (get_analytics_summary)
- Empty database behavior
- Multi-class distribution and confidence buckets calculations
- Daily trends aggregation
- Date range filtering
- SQL injection safety on date parameters
- Absence of misleading environmental or carbon claims
- API Integration endpoint (/api/analytics/dashboard)
"""

import io
import json
import tempfile
import unittest
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

from backend.app.database import (
    init_db,
    save_prediction_record,
    get_analytics_summary,
    get_db_connection
)

BASE_URL = "http://127.0.0.1:8000"


class TestAnalyticsUnit(unittest.TestCase):
    """Kiểm thử đơn vị tầng phân tích dữ liệu SQLite Dashboard."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_analytics.db"
        init_db(self.test_db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_analytics_empty_database(self):
        """Khi cơ sở dữ liệu rỗng: Trả về cấu trúc chuẩn xác với các chỉ số bằng 0."""
        data = get_analytics_summary(db_path=self.test_db_path)
        self.assertEqual(data["total_predictions"], 0)
        self.assertEqual(data["average_confidence"], 0.0)
        self.assertEqual(data["uncertain_count"], 0)
        self.assertEqual(data["uncertain_percentage"], 0.0)

        # Đủ 6 lớp rác thải
        self.assertEqual(len(data["class_distribution"]), 6)
        self.assertTrue(all(item["count"] == 0 for item in data["class_distribution"]))

        # Đủ 4 dải độ tin cậy
        self.assertEqual(len(data["confidence_buckets"]), 4)
        self.assertTrue(all(item["count"] == 0 for item in data["confidence_buckets"]))

        self.assertEqual(len(data["daily_trends"]), 0)
        self.assertEqual(len(data["recent_records"]), 0)

    def test_analytics_aggregation_calculations(self):
        """Tính toán tổng hợp: phân bố lớp, độ tin cậy trung bình, dải tin cậy và xu hướng ngày."""
        now = datetime.now(timezone.utc)
        day1 = (now - timedelta(days=2)).strftime("%Y-%m-%dT10:00:00Z")
        day2 = (now - timedelta(days=1)).strftime("%Y-%m-%dT12:00:00Z")
        day3 = now.strftime("%Y-%m-%dT14:00:00Z")

        records_to_insert = [
            # Day 1: 3 records
            {"created_at": day1, "filename": "p1.jpg", "predicted_class": "plastic", "label_vi": "Nhựa", "confidence": 0.95, "is_uncertain": False, "probabilities": {}, "top_predictions": []},
            {"created_at": day1, "filename": "p2.jpg", "predicted_class": "plastic", "label_vi": "Nhựa", "confidence": 0.91, "is_uncertain": False, "probabilities": {}, "top_predictions": []},
            {"created_at": day1, "filename": "m1.jpg", "predicted_class": "metal", "label_vi": "Kim loại", "confidence": 0.85, "is_uncertain": False, "probabilities": {}, "top_predictions": []},
            # Day 2: 2 records
            {"created_at": day2, "filename": "pa1.jpg", "predicted_class": "paper", "label_vi": "Giấy", "confidence": 0.72, "is_uncertain": False, "probabilities": {}, "top_predictions": []},
            {"created_at": day2, "filename": "o1.jpg", "predicted_class": "organic", "label_vi": "Hữu cơ", "confidence": 0.50, "is_uncertain": True, "probabilities": {}, "top_predictions": []},
            # Day 3: 1 record
            {"created_at": day3, "filename": "g1.jpg", "predicted_class": "glass", "label_vi": "Thủy tinh", "confidence": 0.88, "is_uncertain": False, "probabilities": {}, "top_predictions": []},
        ]

        for r in records_to_insert:
            save_prediction_record(r, self.test_db_path)

        data = get_analytics_summary(db_path=self.test_db_path)

        # 1. Tổng số
        self.assertEqual(data["total_predictions"], 6)

        # 2. Điểm tin cậy trung bình: (0.95+0.91+0.85+0.72+0.50+0.88)/6 = 4.81/6 ≈ 0.8017
        self.assertAlmostEqual(data["average_confidence"], 0.8017, places=2)

        # 3. Tỷ lệ không chắc chắn: 1/6 ≈ 16.7%
        self.assertEqual(data["uncertain_count"], 1)
        self.assertAlmostEqual(data["uncertain_percentage"], 16.7, places=1)

        # 4. Phân bố theo lớp: Plastic nhiều nhất (2), Paper (1), Metal (1), Glass (1), Organic (1), Other (0)
        class_map = {item["class_name"]: item["count"] for item in data["class_distribution"]}
        self.assertEqual(class_map["plastic"], 2)
        self.assertEqual(class_map["metal"], 1)
        self.assertEqual(class_map["paper"], 1)
        self.assertEqual(class_map["glass"], 1)
        self.assertEqual(class_map["organic"], 1)
        self.assertEqual(class_map["other"], 0)

        # 5. Dải độ tin cậy:
        # >= 90%: p1 (0.95), p2 (0.91) -> 2
        # 80-89%: m1 (0.85), g1 (0.88) -> 2
        # 65-79%: pa1 (0.72) -> 1
        # < 65%: o1 (0.50) -> 1
        bucket_map = {b["range_key"]: b["count"] for b in data["confidence_buckets"]}
        self.assertEqual(bucket_map["very_high"], 2)
        self.assertEqual(bucket_map["high"], 2)
        self.assertEqual(bucket_map["medium"], 1)
        self.assertEqual(bucket_map["low"], 1)

        # 6. Xu hướng ngày: Day 1 (3), Day 2 (2), Day 3 (1)
        self.assertEqual(len(data["daily_trends"]), 3)
        self.assertEqual(data["daily_trends"][0]["count"], 3)
        self.assertEqual(data["daily_trends"][1]["count"], 2)
        self.assertEqual(data["daily_trends"][2]["count"], 1)

    def test_analytics_date_filtering(self):
        """Kiểm tra bộ lọc khoảng thời gian start_date và end_date."""
        now = datetime.now(timezone.utc)
        t1 = (now - timedelta(days=10)).strftime("%Y-%m-%dT10:00:00Z")
        t2 = (now - timedelta(days=5)).strftime("%Y-%m-%dT10:00:00Z")
        t3 = now.strftime("%Y-%m-%dT10:00:00Z")

        save_prediction_record({"created_at": t1, "filename": "1.jpg", "predicted_class": "plastic", "label_vi": "Nhựa", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)
        save_prediction_record({"created_at": t2, "filename": "2.jpg", "predicted_class": "paper", "label_vi": "Giấy", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)
        save_prediction_record({"created_at": t3, "filename": "3.jpg", "predicted_class": "metal", "label_vi": "Kim loại", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)

        # Lọc trong khoảng từ 7 ngày trước đến 3 ngày trước -> chỉ có t2
        start_str = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        end_str = (now - timedelta(days=3)).strftime("%Y-%m-%d")

        filtered = get_analytics_summary(start_date=start_str, end_date=end_str, db_path=self.test_db_path)
        self.assertEqual(filtered["total_predictions"], 1)
        self.assertEqual(filtered["recent_records"][0]["filename"], "2.jpg")

    def test_analytics_no_misleading_environmental_claims(self):
        """Đảm bảo hệ thống KHÔNG đưa ra số liệu ngụy tạo về carbon savings hay khối lượng rác tái chế."""
        data = get_analytics_summary(db_path=self.test_db_path)
        self.assertNotIn("carbon_saved_kg", data)
        self.assertNotIn("co2_reduction", data)
        self.assertNotIn("weight_recycled_kg", data)
        self.assertNotIn("trees_saved", data)

    def test_analytics_sql_injection_safety(self):
        """Kiểm tra các chuỗi SQL injection vào start_date và end_date được xử lý an toàn."""
        malicious = "' OR '1'='1"
        data = get_analytics_summary(start_date=malicious, end_date=malicious, db_path=self.test_db_path)
        self.assertEqual(data["total_predictions"], 0)


class TestApiAnalyticsIntegration(unittest.TestCase):
    """Kiểm thử tích hợp qua HTTP API /api/analytics/dashboard tới server thật."""

    def test_api_get_dashboard_analytics_structure(self):
        """GET /api/analytics/dashboard trả về mã 200 và cấu trúc phản hồi đầy đủ."""
        req = urllib.request.Request(f"{BASE_URL}/api/analytics/dashboard")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

            self.assertIn("total_predictions", data)
            self.assertIn("average_confidence", data)
            self.assertIn("uncertain_count", data)
            self.assertIn("uncertain_percentage", data)
            self.assertIn("class_distribution", data)
            self.assertIn("confidence_buckets", data)
            self.assertIn("daily_trends", data)
            self.assertIn("quality_metrics", data)
            self.assertIn("recent_records", data)
            self.assertIn("disclaimer", data)

            # Kiểm tra tuyên bố minh bạch
            self.assertIn("KHÔNG đại diện", data["disclaimer"])

    def test_api_dashboard_date_filter_query(self):
        """Kiểm tra truyền tham số start_date và end_date qua API."""
        req = urllib.request.Request(f"{BASE_URL}/api/analytics/dashboard?start_date=2026-01-01&end_date=2026-12-31")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["filter_applied"]["start_date"], "2026-01-01")
            self.assertEqual(data["filter_applied"]["end_date"], "2026-12-31")

    def test_api_dashboard_future_date_empty_state(self):
        """Truy vấn ngày trong tương lai trả về 0 lượt dự đoán an toàn."""
        req = urllib.request.Request(f"{BASE_URL}/api/analytics/dashboard?start_date=2099-01-01&end_date=2099-12-31")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["total_predictions"], 0)
            self.assertEqual(len(data["daily_trends"]), 0)


if __name__ == "__main__":
    unittest.main()
