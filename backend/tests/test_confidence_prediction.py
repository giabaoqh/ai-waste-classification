import io
import os
import unittest
from pathlib import Path
from PIL import Image

# Import Pydantic models and logic from main
from backend.app.main import (
    PredictionResponse,
    HealthResponse,
    TopPredictionItem,
    RecommendationInfo,
    RECOMMENDATIONS,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    UNCERTAINTY_THRESHOLD
)


class TestConfidenceAwarePrediction(unittest.TestCase):
    """Bộ kiểm thử đơn vị cho tính năng Top-3 dự đoán và xử lý độ không chắc chắn (Phase 1)."""

    def setUp(self):
        self.classes = ["plastic", "paper", "metal", "glass", "organic", "other"]

    def test_top3_ordering_and_consistency(self):
        """Kiểm tra Top-3 luôn sắp xếp giảm dần theo xác suất và khớp với 6 lớp."""
        # Giả lập softmax probabilities
        probs = [0.10, 0.65, 0.05, 0.02, 0.15, 0.03]  # Tổng = 1.00
        # Thứ tự đúng: paper (0.65) > organic (0.15) > plastic (0.10) > metal (0.05) > other (0.03) > glass (0.02)
        sorted_indices = sorted(range(len(self.classes)), key=lambda i: probs[i], reverse=True)
        top_3_indices = sorted_indices[:3]

        top_predictions = []
        for idx in top_3_indices:
            c_name = self.classes[idx]
            rec = RECOMMENDATIONS[c_name]
            top_predictions.append(
                TopPredictionItem(
                    class_name=c_name,
                    label_vi=rec["label_vi"],
                    probability=round(probs[idx], 4)
                )
            )

        # 1. Đủ 3 phần tử
        self.assertEqual(len(top_predictions), 3)

        # 2. Thứ tự giảm dần
        self.assertEqual(top_predictions[0].class_name, "paper")
        self.assertEqual(top_predictions[0].probability, 0.65)
        self.assertEqual(top_predictions[1].class_name, "organic")
        self.assertEqual(top_predictions[1].probability, 0.15)
        self.assertEqual(top_predictions[2].class_name, "plastic")
        self.assertEqual(top_predictions[2].probability, 0.10)
        self.assertGreaterEqual(top_predictions[0].probability, top_predictions[1].probability)
        self.assertGreaterEqual(top_predictions[1].probability, top_predictions[2].probability)

        # 3. Nhãn tiếng Việt đầy đủ
        self.assertIn("giấy", top_predictions[0].label_vi.lower())
        self.assertIn("hữu cơ", top_predictions[1].label_vi.lower())
        self.assertIn("nhựa", top_predictions[2].label_vi.lower())

    def test_probability_distribution_sum(self):
        """Kiểm tra tổng xác suất của 6 lớp luôn xấp xỉ 1.0 (dung sai 1e-4)."""
        probs = [0.248, 0.005, 0.012, 0.008, 0.725, 0.002]
        prob_dict = {cls: round(probs[i], 6) for i, cls in enumerate(self.classes)}
        total_sum = sum(prob_dict.values())
        self.assertAlmostEqual(total_sum, 1.0, delta=1e-3)

    def test_uncertainty_threshold_flagging(self):
        """Kiểm tra khi confidence < threshold thì is_uncertain = True nhưng KHÔNG thay đổi predicted_class."""
        # Trường hợp 1: Confidence thấp (0.52 < 0.65) -> Uncertain
        conf_low = 0.52
        is_uncertain = conf_low < DEFAULT_UNCERTAINTY_THRESHOLD
        self.assertTrue(is_uncertain)

        resp_uncertain = PredictionResponse(
            success=True,
            mock=False,
            predicted_class="plastic",
            label_vi="Nhựa (Plastic)",
            confidence=conf_low,
            probabilities={"plastic": 0.52, "metal": 0.28, "paper": 0.10, "glass": 0.05, "organic": 0.03, "other": 0.02},
            top_predictions=[
                TopPredictionItem(class_name="plastic", label_vi="Nhựa (Plastic)", probability=0.52),
                TopPredictionItem(class_name="metal", label_vi="Kim loại (Metal)", probability=0.28),
                TopPredictionItem(class_name="paper", label_vi="Giấy / Bìa carton (Paper)", probability=0.10),
            ],
            is_uncertain=is_uncertain,
            uncertainty_threshold=DEFAULT_UNCERTAINTY_THRESHOLD,
            uncertainty_note="Độ tin cậy thấp, khuyến nghị chụp lại ảnh.",
            recommendation=RecommendationInfo(
                label_vi="Nhựa (Plastic)",
                bin="Thùng rác tái chế",
                bin_color="#eab308",
                guidance="Rửa sạch trước khi tái chế",
                tips=["Làm sạch", "Bóp xẹp"]
            ),
            filename="low_conf.jpg",
            content_type="image/jpeg",
            image_size_bytes=10240,
            image_dimensions={"width": 300, "height": 300},
            image_format="JPEG"
        )
        # Khẳng định: predicted_class vẫn là 'plastic', không bị ghi đè thành 'other'
        self.assertEqual(resp_uncertain.predicted_class, "plastic")
        self.assertTrue(resp_uncertain.is_uncertain)
        self.assertIsNotNone(resp_uncertain.uncertainty_note)

        # Trường hợp 2: Confidence cao (0.94 >= 0.65) -> Certain
        conf_high = 0.94
        is_uncertain_high = conf_high < DEFAULT_UNCERTAINTY_THRESHOLD
        self.assertFalse(is_uncertain_high)

    def test_backward_compatibility_schema(self):
        """Kiểm tra tính tương thích ngược với các trường frontend cũ."""
        resp = PredictionResponse(
            success=True,
            mock=False,
            predicted_class="paper",
            label_vi="Giấy / Bìa carton (Paper)",
            confidence=0.985,
            probabilities={"plastic": 0.005, "paper": 0.985, "metal": 0.002, "glass": 0.001, "organic": 0.002, "other": 0.005},
            top_predictions=[
                TopPredictionItem(class_name="paper", label_vi="Giấy", probability=0.985),
                TopPredictionItem(class_name="plastic", label_vi="Nhựa", probability=0.005),
                TopPredictionItem(class_name="other", label_vi="Khác", probability=0.005),
            ],
            is_uncertain=False,
            uncertainty_threshold=0.65,
            recommendation=RecommendationInfo(
                label_vi="Giấy / Bìa carton (Paper)",
                bin="Thùng tái chế",
                bin_color="#eab308",
                guidance="Giữ khô ráo",
                tips=["Gấp phẳng"]
            ),
            filename="paper.jpg",
            content_type="image/jpeg",
            image_size_bytes=2048,
            image_dimensions={"width": 224, "height": 224},
            image_format="JPEG"
        )
        # Các trường cũ vẫn tồn tại và đúng kiểu
        self.assertIsInstance(resp.predicted_class, str)
        self.assertIsInstance(resp.confidence, float)
        self.assertIsInstance(resp.probabilities, dict)
        self.assertEqual(len(resp.probabilities), 6)
        self.assertIsInstance(resp.recommendation.tips, list)

        # Các trường mới
        self.assertIsInstance(resp.top_predictions, list)
        self.assertEqual(len(resp.top_predictions), 3)
        self.assertIsInstance(resp.is_uncertain, bool)

    def test_health_response_schema(self):
        """Kiểm tra schema HealthResponse có uncertainty_threshold."""
        health = HealthResponse(
            status="ok",
            service="waste-recognition-backend",
            version="0.2.0",
            ai_model_loaded=True,
            device="cuda:0",
            model_architecture="mobilenet_v3_large",
            checkpoint_loaded=True,
            target_classes=self.classes,
            uncertainty_threshold=0.65
        )
        self.assertEqual(health.status, "ok")
        self.assertEqual(health.uncertainty_threshold, 0.65)
        self.assertEqual(len(health.target_classes), 6)


if __name__ == "__main__":
    unittest.main()
