"""
Unit Tests for Litter News Crawler and GPS Proximity Filtering
"""

import tempfile
import unittest
from pathlib import Path

from backend.app.database import (
    init_db,
    create_litter_report,
    get_litter_reports,
    calculate_haversine_km
)
from backend.app.news_crawler import (
    extract_litter_metadata_from_text,
    sync_litter_news_and_social,
    CURATED_VERIFIED_NEWS_REPORTS
)
from backend.app.main import api_get_litter_reports, api_get_nearby_litter_reports, api_sync_litter_news


class TestNewsCrawlerAndProximity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_news_litter.db"
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_haversine_distance_calculation(self):
        # Khoảng cách giữa Hà Nội (Hoàn Kiếm 21.0285, 105.8542) và Long Biên (21.0562, 105.9085) ~ 6.4 km
        dist = calculate_haversine_km(21.0285, 105.8542, 21.0562, 105.9085)
        self.assertGreater(dist, 5.0)
        self.assertLess(dist, 10.0)

    def test_nlp_extraction_medical_waste(self):
        title = "Phát hiện túi rác y tế, kim tiêm đã qua sử dụng ven đường Đống Đa"
        text = "Các bơm kim tiêm và khẩu trang bị vứt bừa bãi đe dọa người đi đường."
        classes, severity, geo = extract_litter_metadata_from_text(title, text)
        
        self.assertIn("medical", classes)
        self.assertEqual(severity, "critical")
        self.assertIsNotNone(geo)
        self.assertEqual(geo[2], "Q. Đống Đa, Hà Nội")

    def test_nlp_extraction_plastic_coastal_waste(self):
        title = "Rác thải nhựa và chai nilon phủ kín bờ biển Sơn Trà sau bão"
        text = "Hàng tấn chai nhựa, phao xốp trôi dạt vào bãi cát."
        classes, severity, geo = extract_litter_metadata_from_text(title, text)

        self.assertIn("plastic", classes)
        self.assertIn(severity, ["high", "medium"])
        self.assertIsNotNone(geo)
        self.assertEqual(geo[2], "Bán đảo Sơn Trà, Đà Nẵng")

    def test_sync_litter_news_to_database(self):
        result = sync_litter_news_and_social(db_path=self.db_path)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["new_reports_added"], len(CURATED_VERIFIED_NEWS_REPORTS))

        # Chạy lần 2 để kiểm tra deduplication (không được thêm trùng)
        result_second = sync_litter_news_and_social(db_path=self.db_path)
        self.assertTrue(result_second["success"])
        self.assertEqual(result_second["new_reports_added"], 0)
        self.assertGreater(result_second["skipped_existing"], 0)

    def test_gps_proximity_filter(self):
        # Đồng bộ tin tức vào db
        sync_litter_news_and_social(db_path=self.db_path)

        # Đặt tọa độ người dùng tại Quận 1, TP.HCM (10.7769, 106.7009)
        # Các điểm rác gần (Mai Chí Thọ, Gò Vấp, Bình Thạnh) < 15km
        nearby_reports = get_litter_reports(
            user_lat=10.7769,
            user_lng=106.7009,
            radius_km=15.0,
            sort_by="distance",
            db_path=self.db_path
        )
        self.assertGreater(len(nearby_reports), 0)
        for r in nearby_reports:
            self.assertIsNotNone(r["distance_km"])
            self.assertLessEqual(r["distance_km"], 15.0)

        # Kiểm tra sắp xếp tăng dần theo khoảng cách
        distances = [r["distance_km"] for r in nearby_reports]
        self.assertEqual(distances, sorted(distances))


if __name__ == "__main__":
    unittest.main()
