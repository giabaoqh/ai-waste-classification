import logging
import io
from fastapi import APIRouter, Query, File, HTTPException, UploadFile
from .battery_screen import MAX_UPLOAD_BYTES, capabilities, prepare_image, screen_image
from .safety_guidance import SafetyRequest,assess_safety,collection_contacts

router=APIRouter(prefix='/api/safety',tags=['Safety guidance'])
logger = logging.getLogger(__name__)

@router.get('/capabilities')
def model_capabilities():
    return capabilities()

@router.post('/scan')
def scan(file: UploadFile = File(...)):
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Ảnh vượt quá 10 MB.')
        try:
            image, quality = prepare_image(io.BytesIO(content))
        except ValueError as exc:
            raise HTTPException(400, 'Ảnh không hợp lệ hoặc chất lượng không phù hợp. ' + str(exc)) from exc
        try:
            result = screen_image(image, quality)
        except Exception as exc:
            logger.exception('Battery screen unavailable')
            raise HTTPException(503, 'Mô hình pin chưa sẵn sàng. Bạn vẫn có thể chọn loại vật để xem hướng dẫn.') from exc
        guidance = assess_safety(SafetyRequest(item_type='battery' if result['battery_suspected'] else 'unknown'))
        guidance['evidence_source'] = 'image_classifier'
        result['guidance'] = guidance
        return result
    finally:
        file.file.close()

@router.post('/assess')
def assess(request:SafetyRequest):
    return assess_safety(request)

@router.get('/collection-contacts')
def contacts(city:str=Query('Đà Nẵng',max_length=100),material:str=Query('battery',max_length=50)):
    return {'items':collection_contacts(city,material),'notice':'Đây là điểm tham khảo cần xác nhận hiện trạng, không phải danh sách cơ sở được bảo đảm tiếp nhận mọi chất thải nguy hại.'}
