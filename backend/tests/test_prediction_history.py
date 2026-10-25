"""
Automated Test Suite for Prediction History and Database (Phase 3).

Covers:
- Database initialization and schema creation
- CRUD operations (Create, Read with Pagination, Filter by Class, Filter by Date Range, Delete, Clear)
- Parameterized queries and SQL injection safety
- Zero storage of image binaries
- API Integration tests (/api/history, /api/history/stats, /api/history/{id}, DELETE /api/history/{id})
- Automatic record creation upon successful prediction (/api/predict)
"""

import io
import json
import os
import tempfile
import unittest
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from PIL import Image

from backend.app.database import (
    init_db,
    save_prediction_record,
    get_history_records,
    get_history_record_by_id,
    delete_history_record_by_id,
    clear_all_history,
    get_history_statistics,
    get_db_connection
)

BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(r"D:\AI-Based Waste Recognition and Recycling Recommendation System")


def send_image_upload(url: str, img_bytes: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
    boundary = "----WebKitFormBoundaryHistoryTest123"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


class TestDatabaseUnit(unittest.TestCase):
    """Kiểm thử đơn vị tầng cơ sở dữ liệu SQLite (sử dụng isolated test DB)."""

    def setUp(self):
        # Tạo tệp DB tạm biệt lập cho mỗi lần kiểm thử
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_history.db"
        init_db(self.test_db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_and_indexes_exist(self):
        """Kiểm tra bảng và các chỉ mục (indexes) được tạo đầy đủ."""
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            # Kiểm tra bảng
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_history';")
            self.assertIsNotNone(cursor.fetchone())

            # Kiểm tra các chỉ mục
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='prediction_history';")
            index_names = [row[0] for row in cursor.fetchall()]
            self.assertIn("idx_pred_history_created_at", index_names)
            self.assertIn("idx_pred_history_class", index_names)

    def test_save_and_retrieve_record(self):
        """Lưu bản ghi thành công và truy xuất lại đầy đủ dữ liệu cấu trúc."""
        sample_data = {
            "filename": "plastic_bottle.jpg",
            "predicted_class": "plastic",
            "label_vi": "Nhựa (Plastic)",
            "confidence": 0.9521,
            "is_uncertain": False,
            "uncertainty_note": None,
            "probabilities": {"plastic": 0.9521, "paper": 0.02, "metal": 0.01, "glass": 0.01, "organic": 0.005, "other": 0.0029},
            "top_predictions": [
                {"class_name": "plastic", "label_vi": "Nhựa", "probability": 0.9521},
                {"class_name": "paper", "label_vi": "Giấy", "probability": 0.02},
                {"class_name": "metal", "label_vi": "Kim loại", "probability": 0.01}
            ],
            "quality_assessment": {
                "is_acceptable": True,
                "has_warnings": False,
                "brightness_score": 142.5,
                "sharpness_score": 620.0,
                "warnings": []
            },
            "recommendation": {
                "label_vi": "Nhựa (Plastic)",
                "bin": "Thùng rác tái chế",
                "bin_color": "#eab308",
                "guidance": "Rửa sạch và làm bẹp trước khi tái chế",
                "tips": ["Bóp xẹp", "Rửa sạch"]
            },
            "image_dimensions": {"width": 640, "height": 480},
            "image_format": "JPEG",
            "image_size_bytes": 102400
        }

        record = save_prediction_record(sample_data, self.test_db_path)
        self.assertIsNotNone(record["id"])
        self.assertEqual(record["predicted_class"], "plastic")
        self.assertEqual(record["confidence"], 0.9521)
        self.assertFalse(record["is_uncertain"])
        self.assertEqual(record["image_dimensions"], {"width": 640, "height": 480})
        self.assertEqual(record["top_predictions"][0]["class_name"], "plastic")
        self.assertEqual(record["recommendation"]["bin"], "Thùng rác tái chế")

        # Xác thực KHÔNG có trường nhị phân ảnh được lưu trong bảng
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(prediction_history);")
            columns = [row[1] for row in cursor.fetchall()]
            self.assertNotIn("image_binary", columns)
            self.assertNotIn("image_data", columns)
            self.assertNotIn("image_bytes", columns)

    def test_pagination(self):
        """Kiểm tra phân trang: 25 bản ghi, kích thước trang 10 -> 3 trang chuẩn xác."""
        for i in range(25):
            save_prediction_record({
                "filename": f"item_{i}.jpg",
                "predicted_class": "paper" if i % 2 == 0 else "metal",
                "label_vi": "Giấy" if i % 2 == 0 else "Kim loại",
                "confidence": 0.8 + (i * 0.005),
                "is_uncertain": False,
                "probabilities": {"paper": 0.8, "metal": 0.2},
                "top_predictions": []
            }, self.test_db_path)

        # Trang 1: 10 phần tử
        items_p1, total = get_history_records(page=1, page_size=10, db_path=self.test_db_path)
        self.assertEqual(total, 25)
        self.assertEqual(len(items_p1), 10)

        # Trang 2: 10 phần tử
        items_p2, _ = get_history_records(page=2, page_size=10, db_path=self.test_db_path)
        self.assertEqual(len(items_p2), 10)
        self.assertNotEqual(items_p1[0]["id"], items_p2[0]["id"])

        # Trang 3: 5 phần tử còn lại
        items_p3, _ = get_history_records(page=3, page_size=10, db_path=self.test_db_path)
        self.assertEqual(len(items_p3), 5)

    def test_filter_by_class(self):
        """Lọc theo danh mục rác: Chỉ trả về các bản ghi thuộc lớp chỉ định."""
        save_prediction_record({"filename": "a.jpg", "predicted_class": "plastic", "label_vi": "Nhựa", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)
        save_prediction_record({"filename": "b.jpg", "predicted_class": "paper", "label_vi": "Giấy", "confidence": 0.85, "probabilities": {}, "top_predictions": []}, self.test_db_path)
        save_prediction_record({"filename": "c.jpg", "predicted_class": "plastic", "label_vi": "Nhựa", "confidence": 0.92, "probabilities": {}, "top_predictions": []}, self.test_db_path)

        plastic_items, plastic_total = get_history_records(class_name="plastic", db_path=self.test_db_path)
        self.assertEqual(plastic_total, 2)
        self.assertTrue(all(item["predicted_class"] == "plastic" for item in plastic_items))

        paper_items, paper_total = get_history_records(class_name="paper", db_path=self.test_db_path)
        self.assertEqual(paper_total, 1)
        self.assertEqual(paper_items[0]["predicted_class"], "paper")

    def test_filter_by_date_range(self):
        """Lọc theo khoảng ngày ISO-8601."""
        now = datetime.now(timezone.utc)
        yesterday = (now - timedelta(days=1)).isoformat()
        today = now.isoformat()
        tomorrow = (now + timedelta(days=1)).isoformat()

        save_prediction_record({"created_at": yesterday, "filename": "y.jpg", "predicted_class": "glass", "label_vi": "Thủy tinh", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)
        save_prediction_record({"created_at": today, "filename": "t.jpg", "predicted_class": "organic", "label_vi": "Hữu cơ", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)

        # Lọc chỉ ngày hôm nay
        start = now.strftime("%Y-%m-%d")
        items, total = get_history_records(start_date=start, db_path=self.test_db_path)
        self.assertEqual(total, 1)
        self.assertEqual(items[0]["filename"], "t.jpg")

    def test_delete_single_and_clear_all(self):
        """Kiểm tra xóa một bản ghi và xóa sạch toàn bộ lịch sử."""
        r1 = save_prediction_record({"filename": "1.jpg", "predicted_class": "metal", "label_vi": "Kim loại", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)
        r2 = save_prediction_record({"filename": "2.jpg", "predicted_class": "glass", "label_vi": "Thủy tinh", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)

        # Xóa r1
        deleted = delete_history_record_by_id(r1["id"], self.test_db_path)
        self.assertTrue(deleted)
        self.assertIsNone(get_history_record_by_id(r1["id"], self.test_db_path))

        # r2 vẫn còn
        self.assertIsNotNone(get_history_record_by_id(r2["id"], self.test_db_path))

        # Clear all
        count = clear_all_history(self.test_db_path)
        self.assertEqual(count, 1)
        items, total = get_history_records(db_path=self.test_db_path)
        self.assertEqual(total, 0)

    def test_sql_injection_safety(self):
        """Kiểm tra câu lệnh tham số hóa (parameterized query) chống lại chuỗi SQL injection."""
        save_prediction_record({"filename": "normal.jpg", "predicted_class": "plastic", "label_vi": "Nhựa", "confidence": 0.9, "probabilities": {}, "top_predictions": []}, self.test_db_path)

        # Thử tiêm SQL injection vào tham số class_name
        malicious_input = "' OR '1'='1"
        items, total = get_history_records(class_name=malicious_input, db_path=self.test_db_path)
        # Vì được tham số hóa an toàn, không tìm thấy class nào tên là "' OR '1'='1"
        self.assertEqual(total, 0)
        self.assertEqual(len(items), 0)


class TestApiHistoryIntegration(unittest.TestCase):
    """Kiểm thử tích hợp đầu cuối qua HTTP API tới backend server đang chạy."""

    def test_api_predict_automatically_creates_history(self):
        """Gửi ảnh qua /api/predict: Phân loại thành công và tự động lưu bản ghi vào SQLite."""
        img_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Paper" / "Paper_1.jpg"
        if not img_path.exists():
            self.skipTest(f"Không tìm thấy ảnh tại: {img_path}")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        status, pred_data = send_image_upload(f"{BASE_URL}/api/predict", img_bytes, "Paper_1.jpg")
        self.assertEqual(status, 200)
        self.assertIn("record_id", pred_data)
        self.assertIsNotNone(pred_data["record_id"])
        record_id = pred_data["record_id"]

        # Truy vấn lại bản ghi đó qua GET /api/history/{id}
        req = urllib.request.Request(f"{BASE_URL}/api/history/{record_id}")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            hist_item = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(hist_item["id"], record_id)
            self.assertEqual(hist_item["predicted_class"], "paper")
            self.assertIn("giấy", hist_item["label_vi"].lower())
            self.assertAlmostEqual(hist_item["confidence"], pred_data["confidence"], delta=0.001)

    def test_api_list_history_and_stats(self):
        """Kiểm tra GET /api/history phân trang và GET /api/history/stats."""
        # 1. Stats
        req_stats = urllib.request.Request(f"{BASE_URL}/api/history/stats")
        with urllib.request.urlopen(req_stats) as resp:
            self.assertEqual(resp.status, 200)
            stats = json.loads(resp.read().decode("utf-8"))
            self.assertIn("total_predictions", stats)
            self.assertIn("class_counts", stats)

        # 2. List
        req_list = urllib.request.Request(f"{BASE_URL}/api/history?page=1&page_size=5")
        with urllib.request.urlopen(req_list) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("items", data)
            self.assertIn("total", data)
            self.assertIn("page", data)
            self.assertIn("total_pages", data)
            self.assertEqual(data["page"], 1)
            self.assertLessEqual(len(data["items"]), 5)

    def test_api_not_found_and_delete(self):
        """Kiểm tra xử lý lỗi 404 khi ID không tồn tại và xóa bản ghi."""
        # 1. 404 Not Found
        try:
            req = urllib.request.Request(f"{BASE_URL}/api/history/99999999")
            urllib.request.urlopen(req)
            self.fail("Mong đợi HTTPError 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

        # 2. Xóa qua API
        # Lấy 1 bản ghi hiện có
        req_list = urllib.request.Request(f"{BASE_URL}/api/history?page=1&page_size=1")
        with urllib.request.urlopen(req_list) as resp:
            items = json.loads(resp.read().decode("utf-8"))["items"]

        if items:
            target_id = items[0]["id"]
            del_req = urllib.request.Request(f"{BASE_URL}/api/history/{target_id}", method="DELETE")
            with urllib.request.urlopen(del_req) as del_resp:
                self.assertEqual(del_resp.status, 200)
                del_body = json.loads(del_resp.read().decode("utf-8"))
                self.assertTrue(del_body["success"])
                self.assertEqual(del_body["deleted_id"], target_id)


if __name__ == "__main__":
    unittest.main()
