"""Separate experimental detection endpoint; legacy routes remain intact."""
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from .image_input import MAX_UPLOAD_SIZE_BYTES
from . import yolo_detector

router = APIRouter(prefix='/api/yolo', tags=['Experimental YOLO detection'])
logger = logging.getLogger(__name__)


@router.get('/capabilities', response_model=yolo_detector.YoloCapabilities)
def capabilities():
    return yolo_detector.capabilities()


@router.post('/detect', response_model=yolo_detector.DetectionResponse,
             responses={400: {'description': 'Invalid image'}, 413: {'description': 'Image too large'},
                        503: {'model': yolo_detector.YoloErrorResponse, 'description': 'Model/artifact/configuration unavailable'},
                        500: {'model': yolo_detector.YoloErrorResponse, 'description': 'Inference failed; no fabricated detections'}})
def detect(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(400, 'Không có tên tệp ảnh được đính kèm.')
        try:
            contents = file.file.read(MAX_UPLOAD_SIZE_BYTES+1)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, 'Không đọc được tệp ảnh tải lên.') from exc
        # Task 01's pure image/content/quality validator, without any legacy readiness check.
        from .main import validate_legacy_image
        image, _, quality = validate_legacy_image(contents, file.filename)
        try:
            return yolo_detector.detect(image, quality.model_dump())
        except yolo_detector.YoloServiceError as exc:
            logger.exception('YOLO detection failed: %s', exc.code)
            detail = yolo_detector.YoloErrorDetail(code=exc.code, status=exc.status, message=exc.message,
                                                  inference_status='failed' if exc.http_status == 500 else 'not_started')
            raise HTTPException(exc.http_status, detail=detail.model_dump()) from exc
        finally:
            image.close()
    finally:
        file.file.close()
