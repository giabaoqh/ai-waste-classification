"""Model-independent upload validation; request errors precede readiness errors."""
import io

from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def decode_image(contents: bytes, filename: str = "image"):
    if not contents:
        raise HTTPException(400, "Tệp ảnh tải lên rỗng (0 bytes).")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(413, "Dung lượng tệp vượt quá giới hạn cho phép 10 MB.")
    try:
        with Image.open(io.BytesIO(contents)) as original:
            image_format = original.format
            if image_format not in ALLOWED_FORMATS:
                raise HTTPException(400, "Định dạng ảnh không được hỗ trợ. Hỗ trợ: JPEG, PNG, WebP.")
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Ảnh vượt quá giới hạn 20 megapixel.")
            original.verify()
        with Image.open(io.BytesIO(contents)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        return image, image_format
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "Ảnh vượt quá giới hạn số điểm ảnh.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(400, f"Tệp '{filename}' không phải là hình ảnh hợp lệ hoặc bị hỏng.") from exc
