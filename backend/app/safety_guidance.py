"""Precautionary guidance with evidence provenance; never a disease/radiation assay."""
import unicodedata
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

SOURCES={
 'battery':'https://www.epa.gov/recycle/frequent-questions-lithium-ion-batteries',
 'damaged':'https://www.epa.gov/electronics-batteries-management/battery-collection-tip-sheet-end-life-household-small-format',
 'medical':'https://www.who.int/news-room/fact-sheets/detail/health-care-waste',
 'radiation':'https://www.iaea.org/sites/default/files/sealedradsource1013.pdf',
 'varans':'https://www.varans.vn/',
 'danang':'https://cttdt.danangportal.gov.vn/vi/web/dng/w/huong-dan-phan-loai-chat-thai-ran-sinh-hoat-khac-nguy-hai-tai-nguon-i',
 'fec':'https://fec.edu.vn/tin-tuc/thu-gom-pin-cu-bao-ve-moi-truong'}
SOURCES.update(
    recycling='https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables',
    compost='https://www.epa.gov/recycle/composting-home',
    vietnam='https://xaydungchinhsach.chinhphu.vn/huong-dan-ky-thuat-ve-phan-loai-chat-thai-ran-sinh-hoat-119231107185909363.htm')

MATERIAL_STEPS = {
    'paper': ['Giữ giấy sạch, khô; tách giấy dính thức ăn, dầu mỡ hoặc hóa chất.', 'Gấp phẳng bìa carton; hỏi nơi thu gom về giấy tráng nhựa và hộp giấy nhiều lớp.'],
    'plastic': ['Đổ hết phần còn lại và làm sạch cặn thông thường khi an toàn; để ráo.', 'Xác nhận nơi thu gom nhận đúng dạng nhựa; ký hiệu nhựa không bảo đảm họ nhận.', 'Túi/màng nhựa và bao bì nhiều lớp cần luồng riêng nếu được tiếp nhận; không tự đốt hoặc nấu chảy.'],
    'metal': ['Chỉ áp dụng cho lon/hộp thông thường đã rỗng, không nhiễm hóa chất.', 'Pin, bình khí/bình xịt còn áp suất phải tách riêng; không đục, ép hoặc đốt.'],
    'glass': ['Tách chai/lọ khỏi gốm sứ, bóng đèn và kính chuyên dụng; hỏi nơi nhận trước.', 'Không đập nhỏ; tránh chạm mảnh vỡ, cảnh báo người thu gom nếu có vật sắc.'],
    'organic': ['Tách thức ăn và phụ phẩm thực vật khỏi bao bì; giữ trong vật chứa kín phù hợp, bàn giao đúng lịch.', 'Nếu ủ tại nhà: ưu tiên rau quả, lá khô; tránh thịt, dầu mỡ, phân thú nuôi và vật nghi ô nhiễm; xem hướng dẫn ủ phù hợp.'],
    'mixed': ['Không mặc định bao bì nhiều lớp là giấy/nhựa tái chế được.', 'Hỏi đơn vị thu gom về dòng tiếp nhận; không tự tách bằng hóa chất hoặc đốt.'],
}

class SafetyRequest(BaseModel):
    item_type: Literal['unknown','battery','e_waste','medical_sharp','medical_suspected','radiation_suspected','ordinary']='unknown'
    battery_condition: Literal['not_assessed','intact_cool','damaged','hot_or_smoking']='not_assessed'
    city: str=Field(default='Đà Nẵng',max_length=100)
    ordinary_material: Literal['unknown','paper','plastic','metal','glass','organic','mixed']='unknown'

def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKD',value.lower().replace('đ','d')) if not unicodedata.combining(c))

def collection_contacts(city='Đà Nẵng',material='battery'):
    """Only source-backed candidates; no fabricated acceptance, coordinates or opening hours."""
    if material!='battery':return []
    registry=json.loads(Path(__file__).with_name('collection_contacts.json').read_text(encoding='utf-8'))
    region=next((r for r in registry['cities'] if normalized(city).strip() in r['aliases']),None)
    if region is None:return []
    return [{'id':point['id'],'name':'Anh ngữ FEC — điểm từng công bố thu gom pin','city':region['name'],'address':point['address'],
             'source_url':region['source_url'],'source_checked_on':region['source_checked_on'],'verification_status':region['verification_status'],
             'acceptance_confirmed_current':False,'accepted_materials_reported':['battery'],
             'damaged_batteries_acceptance':'not_confirmed','medical_waste_acceptance':'not_confirmed','radioactive_waste_acceptance':'not_confirmed',
             'notes':'Nguồn đơn vị tổ chức có nêu địa chỉ, chưa xác minh trực tiếp hoạt động hiện tại. Liên hệ trước, không tự mang pin hỏng/rác y tế/vật nghi phóng xạ tới.'}
            for point in region['battery_candidates']]

def assess_safety(request: SafetyRequest):
    item=request.item_type
    result={'assessment_scope':'precautionary_guidance','evidence_source':'user_report',
      'status':'REVIEW','infection_status':'not_determinable_from_image','radiation_status':'not_measured',
      'radiation_dose':None,'diagnosis':None,'confirmed_safe':False,'can_use_general_recycling_bin':False,
      'city':request.city,'title':'Cần xác minh loại vật thể','risks':[],'steps':['Không đủ dữ liệu để xác nhận vật thể an toàn. Chụp rõ hoặc nhờ người có chuyên môn kiểm tra.'],
      'avoid':[],'sources':[],'collection_contacts':[],
      'limitations':['Ảnh và score mô hình không đo được tác nhân gây bệnh, liều bức xạ hay thành phần hóa học.']}
    if item=='battery':
        result.update(status='HAZARD_SUSPECTED',title='Pin — thu gom riêng',risks=['Đoản mạch, cháy, rò rỉ hóa chất'],sources=[SOURCES['battery'],SOURCES['damaged']],avoid=['Không nghiền, đốt, chọc thủng hoặc tháo bộ pin.','Không bỏ pin lithium vào thùng rác sinh hoạt hoặc thùng tái chế thông thường.'])
        if request.battery_condition=='intact_cool':
            result['steps']=['Chỉ khi pin nguyên vẹn, nguội và việc thao tác an toàn: cách ly đầu cực bằng băng keo không dẫn điện hoặc để riêng từng viên.','Giữ nhãn nhận dạng, tránh va đập và để xa trẻ em.','Xác nhận đơn vị tiếp nhận nhận đúng loại pin; bàn giao theo hướng dẫn của họ.']
            result['collection_contacts']=collection_contacts(request.city)
        elif request.battery_condition=='hot_or_smoking':
            result.update(status='URGENT_REVIEW',steps=['Rời xa pin, tránh khói và giữ người khác tránh xa.','Liên hệ lực lượng ứng cứu địa phương; không tự đóng gói hoặc vận chuyển pin đang nóng/bốc khói.'])
        else:
            result['steps']=['Không chạm chất rò rỉ, ép, tháo hoặc cố sửa pin.','Hỏi đơn vị xử lý chất thải nguy hại/nhà sản xuất về phương án bàn giao trước khi di chuyển.','Chưa biết tình trạng thì không áp dụng hướng dẫn đóng gói dành cho pin nguyên vẹn.']
    elif item in ['medical_sharp','medical_suspected']:
        result.update(status='HAZARD_SUSPECTED',title='Vật y tế nghi nguy hại — cần xử lý chuyên biệt',risks=['Vật sắc nhọn có thể gây đâm/cắt.','Có thể có nguy cơ lây nhiễm khi có tác nhân gây bệnh và đường phơi nhiễm; ảnh không xác nhận được điều này.'],steps=['Giữ trẻ em và người không có nhiệm vụ tránh xa; không nhặt kim/vật sắc nhọn bằng tay.','Không đậy lại nắp kim, bẻ kim, tự đốt hoặc tái chế tại nhà.','Liên hệ cơ sở y tế/đơn vị chuyên trách để được hướng dẫn thu gom đúng loại; không mặc định nhà thuốc hoặc điểm pin có nhận.','Nếu bị kim đâm hoặc tiếp xúc dịch cơ thể, tìm trợ giúp y tế kịp thời để đánh giá phơi nhiễm.'],avoid=['Không kết luận nhiễm HIV, viêm gan hoặc bệnh cụ thể từ ảnh.'],sources=[SOURCES['medical']])
    elif item=='radiation_suspected':
        result.update(status='HAZARD_SUSPECTED',title='Nghi nguồn phóng xạ — không tiếp cận',risks=['Bức xạ không thể xác nhận hoặc loại trừ chỉ bằng ảnh.'],steps=['Không chạm, mở, tháo, vận chuyển hoặc mang tới điểm tái chế.','Rời xa vật nghi ngờ, giữ người khác tránh xa và thông báo cơ quan chức năng địa phương/cơ quan an toàn bức xạ.','Cần thiết bị đo phù hợp và người có chuyên môn xác minh.'],avoid=['Không tự chế che chắn, tháo nguồn hoặc xử lý phóng xạ tại nhà.'],sources=[SOURCES['radiation'],SOURCES['varans']])
    elif item=='e_waste':
        result.update(status='REVIEW',title='Thiết bị điện tử — chuyển đơn vị chuyên trách',steps=['Không tự tháo pin ẩn hoặc linh kiện khi chưa có hướng dẫn phù hợp.','Xóa dữ liệu cá nhân nếu có thể thực hiện an toàn.','Xác nhận chương trình thu hồi của nhà sản xuất/địa phương nhận đúng thiết bị; báo rõ pin hỏng nếu nghi có.'],sources=[SOURCES['battery']])
    elif item=='ordinary':
        result.update(title='Rác thông thường — cần xác định vật liệu và điều kiện thu gom',steps=['Tách pin, vật sắc nhọn hoặc dấu hiệu nguy hại nếu nhận biết được và việc thao tác an toàn.','Giấy/nhựa/kim loại sạch, rác thực phẩm và bao bì hỗn hợp có thể thuộc các luồng khác nhau.','Xác nhận quy định và loại vật liệu được đơn vị địa phương tiếp nhận. Không tự đốt rác.'],sources=[SOURCES['danang']])
        result['steps'].extend(MATERIAL_STEPS.get(request.ordinary_material, []))
        result['sources'].extend([SOURCES['vietnam'], SOURCES['compost'] if request.ordinary_material=='organic' else SOURCES['recycling']])
    return result
