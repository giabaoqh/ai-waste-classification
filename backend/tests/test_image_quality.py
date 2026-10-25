"""
Automated Test Suite for Input Image Quality Validation (Phase 2).

Tests both unit level (assess_image_quality) and integration level (POST /api/predict)
covering:
- Normal valid images (pass without warnings)
- Extremely small images (<64px: hard rejection HTTP 400)
- Extremely dark images (<6.0: hard rejection HTTP 400)
- Extremely bright images (>252.0: hard rejection HTTP 400)
- Boundary cases (e.g. exactly 64x64 minimum threshold)
- Quality warnings for low resolution (64px <= dim < 160px: warning, HTTP 200)
- Quality warnings for low lighting (6.0 <= brightness < 40.0: warning, HTTP 200)
- Quality warnings for overexposure (230.0 < brightness <= 252.0: warning, HTTP 200)
- Quality warnings for blurry images (sharpness < 45.0: warning, HTTP 200)
- Corrupt and non-image files (HTTP 400)
- Preservation of AI inference and prediction response structure
"""

import io
import json
import unittest
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

from backend.app.main import (
    assess_image_quality,
    ImageQualityInfo,
    MIN_IMAGE_DIMENSION,
    REJECT_DARK_BRIGHTNESS,
    REJECT_BRIGHT_BRIGHTNESS,
    WARN_LOW_RES_DIMENSION,
    WARN_DARK_BRIGHTNESS,
    WARN_BRIGHT_BRIGHTNESS,
    WARN_BLUR_SHARPNESS,
)

BASE_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(r"D:\AI-Based Waste Recognition and Recycling Recommendation System")


def create_test_image_bytes(
    width: int = 256,
    height: int = 256,
    mode: str = "RGB",
    color: int = 128,
    pattern: str = "solid",
    blur_radius: float = 0.0,
    fmt: str = "JPEG"
) -> bytes:
    """Helper to generate in-memory synthetic images for edge case testing."""
    if pattern == "solid":
        if mode == "RGB":
            img = Image.new("RGB", (width, height), (color, color, color))
        else:
            img = Image.new(mode, (width, height), color)
    elif pattern == "checkerboard":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        block_size = max(4, min(width, height) // 8)
        for y in range(height):
            for x in range(width):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    arr[y, x] = [230, 230, 230]
                else:
                    arr[y, x] = [20, 20, 20]
        img = Image.fromarray(arr, "RGB")
    elif pattern == "noise":
        rng = np.random.RandomState(42)
        arr = rng.randint(80, 180, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
    else:
        img = Image.new("RGB", (width, height), (128, 128, 128))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def send_image_upload(url: str, img_bytes: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
    boundary = "----WebKitFormBoundaryQualityTest9XYZ"
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


class TestImageQualityValidation(unittest.TestCase):
    """Kiểm thử tính năng xác thực chất lượng ảnh (Phase 2)."""

    # =========================================================================
    # 1. UNIT TESTS: assess_image_quality
    # =========================================================================

    def test_unit_valid_normal_image(self):
        """Ảnh rõ nét, kích thước chuẩn, độ sáng tự nhiên: Không bị từ chối và không có cảnh báo."""
        # Checkerboard 256x256: sắc nét (sharpness cao), độ sáng trung bình ~125
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                arr[y, x] = [220, 220, 220] if ((x // 16) + (y // 16)) % 2 == 0 else [40, 40, 40]
        img = Image.fromarray(arr, "RGB")

        is_valid, reject_reason, quality_info = assess_image_quality(img)
        self.assertTrue(is_valid)
        self.assertIsNone(reject_reason)
        self.assertTrue(quality_info.is_acceptable)
        self.assertFalse(quality_info.has_warnings)
        self.assertEqual(len(quality_info.warnings), 0)
        self.assertGreater(quality_info.sharpness_score, WARN_BLUR_SHARPNESS)
        self.assertGreaterEqual(quality_info.brightness_score, WARN_DARK_BRIGHTNESS)
        self.assertLessEqual(quality_info.brightness_score, WARN_BRIGHT_BRIGHTNESS)

    def test_unit_reject_tiny_dimensions(self):
        """Ảnh dưới MIN_IMAGE_DIMENSION (64px) phải bị từ chối ngay lập tức."""
        img = Image.new("RGB", (48, 48), (128, 128, 128))
        is_valid, reject_reason, quality_info = assess_image_quality(img)
        self.assertFalse(is_valid)
        self.assertIsNotNone(reject_reason)
        self.assertIn("quá nhỏ", reject_reason.lower())
        self.assertIn("64", reject_reason)
        self.assertFalse(quality_info.is_acceptable)

    def test_unit_boundary_dimension_64px(self):
        """Boundary case: Đúng 64px không bị reject, nhưng có cảnh báo độ phân giải nhỏ (<160px)."""
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        is_valid, reject_reason, quality_info = assess_image_quality(img)
        self.assertTrue(is_valid)
        self.assertIsNone(reject_reason)
        self.assertTrue(quality_info.is_acceptable)
        self.assertTrue(quality_info.has_warnings)
        self.assertTrue(any("độ phân giải" in w.lower() for w in quality_info.warnings))

    def test_unit_reject_pitch_black(self):
        """Ảnh tối đen hoàn toàn (brightness < REJECT_DARK_BRIGHTNESS) phải bị reject."""
        img = Image.new("RGB", (200, 200), (2, 2, 2))
        is_valid, reject_reason, quality_info = assess_image_quality(img)
        self.assertFalse(is_valid)
        self.assertIn("tối", reject_reason.lower())
        self.assertFalse(quality_info.is_acceptable)

    def test_unit_reject_completely_white(self):
        """Ảnh trắng xóa/cháy sáng hoàn toàn (> REJECT_BRIGHT_BRIGHTNESS) phải bị reject."""
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        is_valid, reject_reason, quality_info = assess_image_quality(img)
        self.assertFalse(is_valid)
        self.assertIn("cháy sáng", reject_reason.lower())
        self.assertFalse(quality_info.is_acceptable)

    def test_unit_warning_low_lighting(self):
        """Ảnh thiếu sáng nhưng chưa đen (6.0 <= brightness < 40.0): Chấp nhận với cảnh báo."""
        img = Image.new("RGB", (200, 200), (25, 25, 25))
        is_valid, reject_reason, quality_info = assess_image_quality(img)
        self.assertTrue(is_valid)
        self.assertTrue(quality_info.is_acceptable)
        self.assertTrue(quality_info.has_warnings)
        self.assertTrue(any("thiếu sáng" in w.lower() for w in quality_info.warnings))

    def test_unit_warning_overexposed(self):
        """Ảnh rất sáng/chói nhưng chưa trắng xóa (230.0 < brightness <= 252.0): Chấp nhận với cảnh báo."""
        img = Image.new("RGB", (200, 200), (240, 240, 240))
        is_valid, reject_reason, quality_info = assess_image_quality(img)
        self.assertTrue(is_valid)
        self.assertTrue(quality_info.is_acceptable)
        self.assertTrue(quality_info.has_warnings)
        self.assertTrue(any("độ sáng rất cao" in w.lower() for w in quality_info.warnings))

    def test_unit_warning_blurry_image(self):
        """Ảnh bị mờ (Gaussian blur bán kính lớn khiến Laplacian variance < 45.0): Cảnh báo mờ nét."""
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        for y in range(200):
            for x in range(200):
                arr[y, x] = [200, 200, 200] if (x // 20) % 2 == 0 else [50, 50, 50]
        sharp_img = Image.fromarray(arr, "RGB")
        # Áp dụng làm mờ mạnh
        blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=8))

        is_valid, reject_reason, quality_info = assess_image_quality(blurry_img)
        self.assertTrue(is_valid)
        self.assertTrue(quality_info.is_acceptable)
        self.assertTrue(quality_info.has_warnings)
        self.assertLess(quality_info.sharpness_score, WARN_BLUR_SHARPNESS)
        self.assertTrue(any("mờ" in w.lower() for w in quality_info.warnings))

    # =========================================================================
    # 2. INTEGRATION TESTS: HTTP API /api/predict
    # =========================================================================

    def test_api_real_valid_image_inference(self):
        """Ảnh rác thật từ dataset đạt chuẩn: Trả về HTTP 200, quality_assessment đầy đủ và phân loại thành công."""
        img_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Paper" / "Paper_1.jpg"
        if not img_path.exists():
            self.skipTest(f"Không tìm thấy ảnh tại: {img_path}")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        status, data = send_image_upload(f"{BASE_URL}/api/predict", img_bytes, "Paper_1.jpg")
        self.assertEqual(status, 200)
        self.assertTrue(data["success"])
        self.assertIn("quality_assessment", data)

        qa = data["quality_assessment"]
        self.assertIsNotNone(qa)
        self.assertTrue(qa["is_acceptable"])
        self.assertIsInstance(qa["has_warnings"], bool)
        self.assertIsInstance(qa["brightness_score"], (int, float))
        self.assertIsInstance(qa["sharpness_score"], (int, float))
        self.assertIsInstance(qa["warnings"], list)

        # Đảm bảo phân loại vẫn đúng nhóm rác
        self.assertEqual(data["predicted_class"], "paper")

    def test_api_reject_tiny_image_400(self):
        """Gửi ảnh 32x32 qua API: Trả về HTTP 400 cùng thông báo tiếng Việt rõ ràng."""
        tiny_bytes = create_test_image_bytes(width=32, height=32, color=128)
        try:
            send_image_upload(f"{BASE_URL}/api/predict", tiny_bytes, "tiny.jpg")
            self.fail("Mong đợi HTTPError 400 nhưng request thành công")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            err_body = json.loads(e.read().decode("utf-8"))
            self.assertIn("quá nhỏ", err_body["detail"].lower())

    def test_api_reject_black_image_400(self):
        """Gửi ảnh đen qua API: Trả về HTTP 400 với khuyến nghị bật đèn/chụp nơi đủ sáng."""
        black_bytes = create_test_image_bytes(width=200, height=200, color=0)
        try:
            send_image_upload(f"{BASE_URL}/api/predict", black_bytes, "black.jpg")
            self.fail("Mong đợi HTTPError 400 nhưng request thành công")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            err_body = json.loads(e.read().decode("utf-8"))
            self.assertIn("tối", err_body["detail"].lower())

    def test_api_reject_white_image_400(self):
        """Gửi ảnh trắng xóa qua API: Trả về HTTP 400 với khuyến nghị giảm ánh sáng chói."""
        white_bytes = create_test_image_bytes(width=200, height=200, color=255)
        try:
            send_image_upload(f"{BASE_URL}/api/predict", white_bytes, "white.jpg")
            self.fail("Mong đợi HTTPError 400 nhưng request thành công")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            err_body = json.loads(e.read().decode("utf-8"))
            self.assertIn("cháy sáng", err_body["detail"].lower())

    def test_api_warning_dark_image_proceeds_to_inference(self):
        """Ảnh thiếu sáng nhẹ (brightness ~25): Vẫn cho phép suy luận (HTTP 200) nhưng có warning."""
        # Tạo ảnh 200x200 có đốm sáng nhỏ đủ để trung bình ~25
        arr = np.full((200, 200, 3), 22, dtype=np.uint8)
        arr[80:120, 80:120] = 50  # Trung bình khoảng ~23.5
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        dark_bytes = buf.getvalue()

        status, data = send_image_upload(f"{BASE_URL}/api/predict", dark_bytes, "low_light.jpg")
        self.assertEqual(status, 200)
        self.assertTrue(data["success"])
        qa = data["quality_assessment"]
        self.assertTrue(qa["has_warnings"])
        self.assertTrue(any("thiếu sáng" in w.lower() for w in qa["warnings"]))

    def test_api_warning_blurry_image_proceeds_to_inference(self):
        """Ảnh bị mờ nét (blurry): Vẫn suy luận trả về HTTP 200 nhưng cảnh báo người dùng giữ chắc tay."""
        blurry_bytes = create_test_image_bytes(width=256, height=256, pattern="checkerboard", blur_radius=9.0)
        status, data = send_image_upload(f"{BASE_URL}/api/predict", blurry_bytes, "blurry.jpg")
        self.assertEqual(status, 200)
        self.assertTrue(data["success"])
        qa = data["quality_assessment"]
        self.assertTrue(qa["has_warnings"])
        self.assertTrue(any("mờ" in w.lower() for w in qa["warnings"]))

    def test_api_corrupted_file_400(self):
        """Tệp hỏng hoặc không phải định dạng ảnh: Trả về HTTP 400 an toàn."""
        corrupt_bytes = b"This is purely invalid corrupted non-image content."
        try:
            send_image_upload(f"{BASE_URL}/api/predict", corrupt_bytes, "corrupt.txt", content_type="text/plain")
            self.fail("Mong đợi HTTPError 400 nhưng request thành công")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


if __name__ == "__main__":
    unittest.main()
