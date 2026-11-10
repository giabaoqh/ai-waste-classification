from io import BytesIO
import unittest
from PIL import Image
from scripts.prepare_multisource_waste import canonical


class TestCanonicalDatasetImage(unittest.TestCase):
    def test_resize_keeps_aspect_ratio_for_normalized_boxes(self):
        source=BytesIO();Image.new('RGB',(2048,1024),'blue').save(source,'JPEG')
        raw=source.getvalue()
        data,_,size=canonical(raw)
        self.assertEqual(size,(2048,1024))
        self.assertEqual(Image.open(BytesIO(data)).size,(640,320))
        self.assertEqual(data[-2:],b'\xff\xd9')
        self.assertEqual(source.getvalue(),raw)

    def test_exif_orientation_is_applied_before_dimension_check(self):
        source=BytesIO();im=Image.new('RGB',(80,40),'red')
        exif=im.getexif();exif[274]=6;im.save(source,'JPEG',exif=exif)
        data,_,size=canonical(source.getvalue())
        self.assertEqual(size,(40,80))
        out=Image.open(BytesIO(data))
        self.assertEqual(out.size,(40,80))
        self.assertNotIn(274,out.getexif())
