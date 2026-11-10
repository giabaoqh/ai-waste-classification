"""Guard coordinate conversion and split provenance before using external annotations."""

import math
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.prepare_taco_pilot import CLASSES, assign_groups, category_mapping, download_image, normalized_box, write_training_image


class TestTacoPilot(unittest.TestCase):
    def test_box_conversion_preserves_position(self):
        self.assertEqual(normalized_box([20, 10, 40, 20], 100, 50), (.4, .4, .4, .4))

    def test_partial_box_is_clipped_before_normalization(self):
        self.assertEqual(normalized_box([-10, 10, 30, 20], 100, 50), (.1, .4, .2, .4))

    def test_malformed_boxes_are_rejected(self):
        for bbox in ([0, 0, 0, 2], [0, 0, -1, 2], [110, 0, 5, 5], [math.nan, 0, 2, 2]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                normalized_box(bbox, 100, 100)

    def test_source_groups_never_split_between_partitions(self):
        images = [{"id": i * 2 + j, "file_name": f"batch_{i}/{j}.jpg"} for i in range(15) for j in range(2)]
        splits = assign_groups(images, 42)
        self.assertEqual(set(splits.values()), {"train", "val", "test"})
        self.assertEqual(splits, assign_groups(list(reversed(images)), 42))
        self.assertEqual(len(splits), 15)

    def test_rare_hazardous_items_are_not_assigned_safe_material_labels(self):
        mapped = category_mapping([{"id": 1, "name": "Battery"}, {"id": 6, "name": "Glass bottle"}])
        self.assertEqual(CLASSES[mapped[1]], "other_litter")
        self.assertEqual(CLASSES[mapped[6]], "bottle_jar")

    def test_download_accepts_source_with_no_thumbnail_url(self):
        image = {"id": 348, "width": 30, "height": 40, "flickr_640_url": None, "flickr_url": "https://example.org/source.jpg"}
        buffer = BytesIO()
        Image.new("RGB", (30, 40), "green").save(buffer, format="JPEG")
        with tempfile.TemporaryDirectory() as temp:
            with patch("urllib.request.urlopen", return_value=BytesIO(buffer.getvalue())) as fetch:
                result = download_image(image, Path(temp))
            self.assertNotIn("error", result)
            self.assertEqual(result["url"], image["flickr_url"])
            self.assertEqual((result["width"], result["height"]), (30, 40))
            self.assertEqual(fetch.call_args.args[0].full_url, image["flickr_url"])

    def test_canonical_jpeg_preserves_raw_source_and_has_eoi(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source.jpg", Path(temp) / "derived.jpg"
            Image.new("RGB", (30, 40), "red").save(source)
            original = source.read_bytes() + b"trailing source metadata"
            source.write_bytes(original)
            write_training_image(source, target)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(target.read_bytes()[-2:], b"\xff\xd9")
            with Image.open(target) as derived:
                self.assertEqual(derived.size, (30, 40))


if __name__ == "__main__":
    unittest.main()
