import io
import json
import urllib.request
import urllib.error
import mimetypes
import unittest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(r"D:\AI-Based Waste Recognition and Recycling Recommendation System")
BASE_URL = "http://127.0.0.1:8000"


def send_image_upload(url: str, img_bytes: bytes, filename: str, content_type: str = "image/jpeg"):
    boundary = "----WebKitFormBoundaryIntegrationTest7MA4"
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


class TestApiIntegration(unittest.TestCase):
    """Kiểm thử tích hợp đầu cuối qua HTTP API tới backend server đang chạy."""

    def test_health_endpoint(self):
        """Kiểm tra GET /api/health trả về đúng cấu trúc và ngưỡng tin cậy."""
        req = urllib.request.Request(f"{BASE_URL}/api/health", method="GET")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertTrue(data["ai_model_loaded"])
            self.assertEqual(data["model_architecture"], "mobilenet_v3_large")
            self.assertIn("uncertainty_threshold", data)
            self.assertAlmostEqual(data["uncertainty_threshold"], 0.65, delta=0.01)
            self.assertEqual(len(data["target_classes"]), 6)

    def test_predict_with_real_image(self):
        """Kiểm tra POST /api/predict với ảnh thật trả về Top-3 và phân bố xác suất chuẩn."""
        img_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Paper" / "Paper_1.jpg"
        if not img_path.exists():
            self.skipTest(f"Không tìm thấy ảnh tại: {img_path}")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        status_code, data = send_image_upload(f"{BASE_URL}/api/predict", img_bytes, "Paper_1.jpg", "image/jpeg")
        self.assertEqual(status_code, 200)
        self.assertTrue(data["success"])
        self.assertFalse(data["mock"])

        # Kiểm tra Top-3
        self.assertIn("top_predictions", data)
        self.assertIsInstance(data["top_predictions"], list)
        self.assertEqual(len(data["top_predictions"]), 3)

        top1 = data["top_predictions"][0]
        top2 = data["top_predictions"][1]
        top3 = data["top_predictions"][2]

        self.assertGreaterEqual(top1["probability"], top2["probability"])
        self.assertGreaterEqual(top2["probability"], top3["probability"])
        # Top-1 xác suất khớp với confidence
        self.assertAlmostEqual(top1["probability"], data["confidence"], delta=0.001)

        # Kiểm tra tính không chắc chắn (Uncertainty fields)
        self.assertIn("is_uncertain", data)
        self.assertIsInstance(data["is_uncertain"], bool)
        self.assertIn("uncertainty_threshold", data)
        if data["confidence"] < data["uncertainty_threshold"]:
            self.assertTrue(data["is_uncertain"])
            self.assertIsNotNone(data["uncertainty_note"])
        else:
            self.assertFalse(data["is_uncertain"])

        # Kiểm tra tính tương thích ngược với 6 lớp
        self.assertEqual(len(data["probabilities"]), 6)
        prob_sum = sum(data["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.01)
        self.assertIn("predicted_class", data)
        self.assertIn("recommendation", data)
        self.assertIn("tips", data["recommendation"])

    def test_predict_invalid_image_bytes(self):
        """Kiểm tra gửi tệp rỗng hoặc không phải ảnh trả về HTTP 400 Bad Request."""
        try:
            send_image_upload(f"{BASE_URL}/api/predict", b"not-an-image-content-string", "corrupt.jpg", "image/jpeg")
            self.fail("Mong đợi HTTPError 400 nhưng request đã thành công")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


if __name__ == "__main__":
    unittest.main()
