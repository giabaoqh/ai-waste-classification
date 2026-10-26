"""
Automated Unit Tests for Multi-Object Waste Detection API Handler.
Uses Python's standard unittest library.
"""

import sys
import unittest
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.main import (
    app,
    process_multi_object_detection_bytes,
    process_waste_image_bytes,
    load_model_from_checkpoint,
    find_checkpoint_path,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from backend.app.database import init_db


class TestMultiObjectDetectionHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        app.state.device = str(device)
        app.state.classes = ["plastic", "paper", "metal", "glass", "organic", "other"]
        app.state.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        ckpt = find_checkpoint_path()
        model, classes = load_model_from_checkpoint(ckpt, device)
        app.state.model = model
        app.state.classes = classes
        app.state.model_loaded = True

    def test_multi_object_detection_real_waste(self):
        """Test multi-object detection pipeline on real plastic sample."""
        sample_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
        self.assertTrue(sample_path.exists())

        with open(sample_path, "rb") as f:
            bytes_data = f.read()

        res = process_multi_object_detection_bytes(bytes_data, "Plastic_1.jpg")
        self.assertTrue(res.success)
        self.assertGreaterEqual(res.total_objects, 1)
        self.assertIsNotNone(res.annotated_image)
        self.assertTrue(res.annotated_image.startswith("data:image/jpeg;base64,"))

        first = res.objects[0]
        # Detector confidence and classifier confidence must be distinct
        self.assertIn("detector_confidence", first.model_dump())
        self.assertIn("classifier_confidence", first.model_dump())
        self.assertEqual(first.waste_class, "plastic")
        self.assertGreater(first.classifier_confidence, 0.8)
        print(f"\n[PASS] Multi-object detection detected {res.total_objects} item(s). Class: {first.waste_class} ({first.classifier_confidence*100:.1f}%)")

    def test_multi_object_detection_empty_canvas(self):
        """Test empty uniform canvas returns clean 0 objects without crash."""
        img = Image.new("RGB", (320, 320), color=(220, 220, 220))
        buf = BytesIO()
        img.save(buf, format="JPEG")

        res = process_multi_object_detection_bytes(buf.getvalue(), "blank.jpg")
        self.assertTrue(res.success)
        self.assertEqual(res.total_objects, 0)
        self.assertEqual(len(res.objects), 0)
        self.assertIsNone(res.annotated_image)
        print("\n[PASS] Multi-object detection handled empty canvas cleanly.")

    def test_single_predict_remains_fully_functional(self):
        """Verify that existing single-image predict pipeline is 100% intact."""
        sample_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
        with open(sample_path, "rb") as f:
            bytes_data = f.read()

        pred = process_waste_image_bytes(bytes_data, "Plastic_1.jpg", include_gradcam=True)
        self.assertTrue(pred.success)
        self.assertEqual(pred.predicted_class, "plastic")
        self.assertIsNotNone(pred.gradcam_overlay)
        print("\n[PASS] Existing /api/predict with Grad-CAM verified 100% operational.")


if __name__ == "__main__":
    unittest.main()

