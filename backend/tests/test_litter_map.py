"""
Unit test suite for Smart City Crowdsourced Litter Map module.
"""

import unittest
from pathlib import Path
import tempfile
import json

from backend.app.database import (
    init_db,
    get_litter_reports,
    create_litter_report,
    get_litter_report_by_id,
    update_litter_report_status,
    upvote_litter_report,
    get_litter_reports_stats
)


class TestSmartLitterMap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_litter.db"
        init_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_seed_data_loaded(self):
        reports = get_litter_reports(db_path=self.db_path)
        self.assertGreaterEqual(len(reports), 5)
        # Verify critical medical reports are seeded and prioritized
        critical_reports = [r for r in reports if r["severity"] == "critical"]
        self.assertGreaterEqual(len(critical_reports), 1)
        self.assertIn("medical", critical_reports[0]["detected_waste_classes"])

    def test_create_and_fetch_report(self):
        payload = {
            "latitude": 21.0123,
            "longitude": 105.8456,
            "address": "Đường Lê Duẩn, Hà Nội",
            "description": "Đống rác chai lọ nhựa bừa bãi",
            "severity": "medium",
            "reported_by": "Nguyễn Văn A",
            "detected_waste_classes": ["plastic", "glass"],
            "image_base64": "data:image/jpeg;base64,sample123"
        }
        report_id = create_litter_report(payload, db_path=self.db_path)
        self.assertIsInstance(report_id, int)

        fetched = get_litter_report_by_id(report_id, db_path=self.db_path)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["address"], "Đường Lê Duẩn, Hà Nội")
        self.assertEqual(fetched["status"], "pending")
        self.assertEqual(fetched["upvotes"], 1)
        self.assertEqual(fetched["detected_waste_classes"], ["plastic", "glass"])

    def test_upvote_report(self):
        reports = get_litter_reports(db_path=self.db_path)
        first = reports[0]
        initial_votes = first["upvotes"]

        new_votes = upvote_litter_report(first["id"], db_path=self.db_path)
        self.assertEqual(new_votes, initial_votes + 1)

    def test_update_status_to_resolved(self):
        reports = get_litter_reports(status="pending", db_path=self.db_path)
        self.assertGreater(len(reports), 0)
        target = reports[0]

        ok = update_litter_report_status(
            target["id"],
            status="resolved",
            resolved_note="Đội tình nguyện đã dọn sạch",
            db_path=self.db_path
        )
        self.assertTrue(ok)

        updated = get_litter_report_by_id(target["id"], db_path=self.db_path)
        self.assertEqual(updated["status"], "resolved")
        self.assertIsNotNone(updated["resolved_at"])
        self.assertEqual(updated["resolved_note"], "Đội tình nguyện đã dọn sạch")

    def test_stats_calculation(self):
        stats = get_litter_reports_stats(db_path=self.db_path)
        self.assertIn("total_reports", stats)
        self.assertIn("pending_reports", stats)
        self.assertIn("resolved_reports", stats)
        self.assertIn("resolution_rate", stats)
        self.assertGreater(stats["total_reports"], 0)
        self.assertGreaterEqual(stats["resolution_rate"], 0.0)
        self.assertLessEqual(stats["resolution_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()

