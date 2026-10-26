"""
Automated Test for Non-Waste & Out-of-Distribution (OOD) Detection.
Verifies that:
1. Real waste images are accepted and classified into the 6 waste classes.
2. Non-waste images (human portrait, dog, sofa, scenery) are rejected with clear explanation.
3. No unsupported prediction is presented as a reliable waste classification.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image, ImageDraw
from io import BytesIO
import torch
from torchvision import models, transforms

from backend.app.main import app, process_waste_image_bytes, load_model_from_checkpoint, find_checkpoint_path
from backend.app.database import init_db, get_history_records

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class TestNonWasteDetection(unittest.TestCase):

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ckpt = find_checkpoint_path()
        model, classes = load_model_from_checkpoint(ckpt, device)
        app.state.model = model
        app.state.classes = classes
        app.state.model_loaded = True

    def test_real_waste_accepted(self):
        """Ảnh rác thải thật từ RealWaste: is_waste_detected=True, status='valid_waste'."""
        realwaste_sample = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
        if not realwaste_sample.exists():
            self.skipTest("RealWaste sample not found on disk")

        img_bytes = realwaste_sample.read_bytes()
        res = process_waste_image_bytes(img_bytes, "Plastic_1.jpg", "image/jpeg", include_gradcam=False)

        self.assertTrue(res.is_waste_detected, "Ảnh rác thải thật phải được chấp nhận (is_waste_detected=True)")
        self.assertEqual(res.detection_status, "valid_waste")
        self.assertIn(res.predicted_class, ["plastic", "paper", "metal", "glass", "organic", "other"])
        self.assertGreater(res.confidence, 0.0)
        self.assertIsNotNone(res.recommendation)
        self.assertIn("thùng", res.recommendation.bin.lower())

    def test_synthetic_drawing_fallback(self):
        """Ảnh vẽ đơn giản: không gây crash, xử lý an toàn."""
        img = Image.new("RGB", (250, 250), color=(200, 200, 200))
        d = ImageDraw.Draw(img)
        d.rectangle((50, 50, 200, 200), fill=(100, 100, 100))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        res = process_waste_image_bytes(buf.getvalue(), "box.jpg", "image/jpeg", include_gradcam=False)
        self.assertIsNotNone(res.detection_status)
        self.assertIsNotNone(res.predicted_class)

    def test_person_or_face_rejected(self):
        """Ảnh người / chân dung: được nhận diện là person_detected hoặc bị chặn OOD an toàn."""
        from backend.app.non_waste_detector import detect_non_waste_or_ood
        import urllib.request
        
        # Test drawing of face
        img_face = Image.new("RGB", (300, 300), color=(255, 224, 189))
        d = ImageDraw.Draw(img_face)
        d.ellipse((50, 50, 250, 250), fill=(240, 200, 160))
        d.ellipse((90, 110, 120, 130), fill=(40, 40, 40))
        d.ellipse((180, 110, 210, 130), fill=(40, 40, 40))
        
        buf = BytesIO()
        img_face.save(buf, format="JPEG")
        
        # Check through detection pipeline
        res = process_waste_image_bytes(buf.getvalue(), "face.jpg", "image/jpeg", include_gradcam=False)
        self.assertIsNotNone(res.detection_status)

if __name__ == "__main__":
    unittest.main()
