import csv
import io
import json
import logging
import os
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from backend.app.database import (
    init_db,
    save_prediction_record,
    get_history_records,
    get_history_record_by_id,
    delete_history_record_by_id,
    clear_all_history,
    get_history_statistics,
    get_analytics_summary,
    save_user_feedback,
    get_all_history_for_export,
    get_error_cases_for_export
)
from backend.app.gradcam import generate_gradcam_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("waste_recognition_backend")

# Constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Vietnamese Waste Disposal, Recycling & Environmental Analysis
RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "plastic": {
        "label_vi": "Nhựa (Plastic)",
        "bin": "Thùng rác tái chế (Màu vàng/trắng)",
        "bin_color": "#eab308",
        "guidance": "Tráng sạch cặn chất lỏng hoặc dầu mỡ thừa, bóp xẹp chai/hộp nhựa để tiết kiệm thể tích trước khi bỏ vào thùng rác tái chế. Tháo rời nắp và nhãn mác nếu có thể.",
        "tips": [
            "Đổ hết nước và chất lỏng còn đọng lại bên trong",
            "Tráng sơ qua nước sạch nếu còn dính dầu mỡ/thức ăn",
            "Bóp xẹp chai hoặc cán phẳng hộp để giảm 70% thể tích chứa",
            "Phân loại riêng nắp chai (nhựa HDPE) và thân chai (nhựa PET)"
        ],
        "combustibility": {
            "is_combustible": True,
            "status": "Cháy được (Cảnh báo khí độc hại)",
            "badge_color": "#ea580c",
            "analysis": "Nhựa có nhiệt trị cao (tương đương dầu mỏ) nhưng TUYỆT ĐỐI KHÔNG tự đốt hở tại nhà vì sinh ra khí cực độc như Dioxin, Furan, khí Clo và muội than đen gây hại hệ hô hấp. Chỉ được thiêu đốt trong các nhà máy điện rác (WtE - Waste to Energy) có hệ thống lọc khí thải tiêu chuẩn.",
            "warning": "Không đốt hở ngoài trời. Khí thải Dioxin từ nhựa cháy có nguy cơ gây ung thư."
        },
        "recyclability": {
            "level": "Tái chế phổ biến / Giá trị kinh tế cao",
            "difficulty_score": "Dễ",
            "badge_color": "#16a34a",
            "analysis": "Nhựa nhiệt dẻo như PET (chai nước) và HDPE (hộp sữa, can xà phòng) rất dễ tái chế cơ học. Nhựa nhiều lớp (túi snack, màng bọc nilon mỏng) khó tái chế hơn và thường cần ép viên nhiên liệu.",
            "economic_value": "Rất cao — Được các nhà máy tái sinh hạt nhựa thu mua liên tục."
        },
        "processing_steps": [
            "1. Tại nguồn: Rửa sạch, để ráo và bóp xẹp vỏ chai.",
            "2. Thu gom: Đưa về trạm ép kiện nhựa tập trung.",
            "3. Nhà máy: Băm nhỏ thành mảnh vảy (flakes), giặt nhiệt tẩy keo và kéo hạt nhựa tái sinh."
        ],
        "environmental_impact": {
            "decomposition_time": "450 - 500 năm trong tự nhiên",
            "carbon_saving": "Tái chế 1 tấn nhựa tiết kiệm ~1.5 tấn CO2 so với sản xuất nhựa nguyên sinh."
        }
    },
    "paper": {
        "label_vi": "Giấy / Bìa carton (Paper)",
        "bin": "Thùng rác tái chế (Màu vàng/trắng)",
        "bin_color": "#3b82f6",
        "guidance": "Giữ giấy khô ráo và sạch sẽ, gấp phẳng các thùng bìa carton. Nếu giấy bị dính dầu mỡ thức ăn hoặc ẩm ướt nặng, vui lòng bỏ vào thùng rác sinh hoạt thông thường.",
        "tips": [
            "Giữ giấy khô ráo, tránh để dính nước hoặc ẩm mốc",
            "Gấp phẳng thùng carton để tiết kiệm diện tích thu gom",
            "Loại bỏ băng dính dán trên hộp carton nếu có thể",
            "Giấy thấm dầu, hộp giấy dính mỡ thức ăn phải bỏ rác sinh hoạt"
        ],
        "combustibility": {
            "is_combustible": True,
            "status": "Dễ bắt cháy (Nhiên liệu sinh khối RDF)",
            "badge_color": "#f97316",
            "analysis": "Giấy khô cực kỳ dễ bắt cháy, nhiệt độ bốc cháy thấp (~230°C). Giấy sạch nên ưu tiên tái chế bột giấy; giấy loại dơ không thể tái chế có thể dùng làm mồi lửa hoặc ép thành viên nhiên liệu đốt sinh khối công nghiệp.",
            "warning": "Cần lưu trữ nơi khô ráo, tránh xa nguồn nhiệt để phòng ngừa hỏa hoạn."
        },
        "recyclability": {
            "level": "Dễ tái chế cao",
            "difficulty_score": "Dễ",
            "badge_color": "#16a34a",
            "analysis": "Sợi cellulose trong giấy có thể tái chế từ 5 đến 7 chu kỳ trước khi sợi bị đứt gãy quá ngắn. Giấy carton sóng có tỷ lệ thu hồi tái chế cao nhất.",
            "economic_value": "Cao — Là nguồn nguyên liệu chính cho ngành bao bì carton."
        },
        "processing_steps": [
            "1. Tại nguồn: Xếp phẳng, giữ khô ráo, bó thành kiện.",
            "2. Thu gom: Vận chuyển đến các xưởng xeo giấy.",
            "3. Nhà máy: Đánh bột thuỷ lực, tẩy mực, sàng lọc xơ sợi và xeo thành giấy bao bì mới."
        ],
        "environmental_impact": {
            "decomposition_time": "2 - 6 tuần (trong điều kiện hiếu khí)",
            "carbon_saving": "Tái chế 1 tấn giấy cứu sống ~17 cây xanh và tiết kiệm 4.000 kWh điện."
        }
    },
    "metal": {
        "label_vi": "Kim loại (Metal)",
        "bin": "Thùng rác tái chế (Màu vàng/trắng)",
        "bin_color": "#ef4444",
        "guidance": "Đổ sạch phần thức ăn hoặc đồ uống còn lại bên trong lon/hộp kim loại; tráng sơ qua nước. Cẩn thận các cạnh sắc nhọn của nắp lon khi bỏ vào thùng rác tái chế.",
        "tips": [
            "Làm rỗng hoàn toàn nước ngọt/bia trong vỏ lon",
            "Tráng sạch cặn thức ăn thừa trong hộp thiếc",
            "Gập mép nắp sắc nhọn vào trong để đảm bảo an toàn cho nhân viên thu gom",
            "Dẫm dẹp vỏ lon nhôm để giảm thể tích"
        ],
        "combustibility": {
            "is_combustible": False,
            "status": "Không thể cháy",
            "badge_color": "#0284c7",
            "analysis": "Kim loại (nhôm, sắt, thép, thiếc) là vật liệu vô cơ không cháy trong điều kiện sinh hoạt thông thường. Khi bị đưa vào lò đốt rác, kim loại không sinh nhiệt mà làm tiêu tốn năng lượng lò và sót lại dưới dạng xỉ đáy lò.",
            "warning": "Vỏ bình xịt khí nén (aerosol) có nguy cơ phát nổ nếu đưa vào nguồn nhiệt."
        },
        "recyclability": {
            "level": "Tái chế vô hạn lần (100% Recyclable)",
            "difficulty_score": "Rất dễ",
            "badge_color": "#10b981",
            "analysis": "Kim loại có thể nấu chảy và tái chế tuần hoàn vô hạn lần mà không làm suy giảm tính chất cơ lý tính. Tái chế nhôm tiết kiệm đến 95% năng lượng so với luyện nhôm từ quặng bauxite.",
            "economic_value": "Rất cao — Kim loại phế liệu luôn được săn đón và thu mua giá cao."
        },
        "processing_steps": [
            "1. Tại nguồn: Rửa sạch cặn, làm dẹp lon nhôm.",
            "2. Thu gom: Phân loại sắt từ bằng nam châm và tách nhôm bằng dòng điện xoáy (Eddy current).",
            "3. Nhà máy: Nung chảy ở 660°C (nhôm) hoặc 1500°C (sắt thép) và đúc thành phôi kim loại."
        ],
        "environmental_impact": {
            "decomposition_time": "200 - 500 năm đối với lon nhôm",
            "carbon_saving": "Tái chế 1 tấn nhôm ngăn phát thải ~9 tấn khí nhà kính CO2."
        }
    },
    "glass": {
        "label_vi": "Thủy tinh (Glass)",
        "bin": "Thùng rác tái chế / Trạm thu gom thủy tinh",
        "bin_color": "#06b6d4",
        "guidance": "Làm rỗng và tráng sạch chai lọ thủy tinh. Tháo bỏ nắp nhựa hoặc nắp kim loại để phân loại riêng. Nếu là mảnh vỡ thủy tinh, bọc kỹ bằng giấy báo hoặc bìa cứng để đảm bảo an toàn.",
        "tips": [
            "Tráng sạch cặn bẩn hoặc dầu giấm trong chai lọ",
            "Tháo nắp nhựa hoặc nắp kim loại để phân loại vào nhóm tương ứng",
            "Nếu bị vỡ: Gom mảnh vỡ và bọc chặt bằng giấy báo hoặc hộp các-tông",
            "Ghi chú chữ 'Thủy tinh vỡ nguy hiểm' bên ngoài bọc"
        ],
        "combustibility": {
            "is_combustible": False,
            "status": "Không thể cháy (Nhiệt độ nóng chảy cực cao)",
            "badge_color": "#0284c7",
            "analysis": "Thủy tinh có thành phần chủ yếu là Silicat (SiO2), trơ hoàn toàn với lửa và không bắt cháy. Thủy tinh nóng chảy ở nhiệt độ rất cao (1.400 - 1.600°C), vượt xa nhiệt độ lò thiêu hủy rác sinh hoạt thông thường.",
            "warning": "Không đưa thủy tinh vào lò đốt rác vì làm hư hại ghi lò."
        },
        "recyclability": {
            "level": "Tái chế vô hạn lần",
            "difficulty_score": "Dễ",
            "badge_color": "#10b981",
            "analysis": "Thủy tinh tinh khiết có thể tái sinh vô hạn lần mà không giảm chất lượng hoặc độ trong suốt. Tuy nhiên cần phân loại kỹ theo màu sắc (thủy tinh trắng, xanh, nâu) và không lẫn gốm sứ/pha lê.",
            "economic_value": "Trung bình đến Cao — Phù hợp mô hình tái sử dụng vỏ chai (Refill)."
        },
        "processing_steps": [
            "1. Tại nguồn: Rửa sạch, để khô ráo, giữ nguyên vẹn vỏ chai.",
            "2. Thu gom: Phân loại theo 3 màu: Trắng trong, Xanh lá cây, Nâu hổ phách.",
            "3. Nhà máy: Đập vụn thành miểng (cullet), trộn cát thạch anh và nấu chảy đúc chai mới."
        ],
        "environmental_impact": {
            "decomposition_time": "Hơn 1.000.000 năm (Gần như không bao giờ phân hủy tự nhiên)",
            "carbon_saving": "Dùng 1 tấn miểng thủy tinh tái chế giảm phát thải 315 kg CO2."
        }
    },
    "organic": {
        "label_vi": "Rác hữu cơ (Organic)",
        "bin": "Thùng rác hữu cơ (Màu xanh lá)",
        "bin_color": "#22c55e",
        "guidance": "Bao gồm thức ăn thừa, cuống rau củ quả, bã trà, bã cà phê và lá cây. Thích hợp dùng làm phân bón hữu cơ ủ (compost) hoặc bỏ vào thùng rác hữu cơ dễ phân hủy sinh học.",
        "tips": [
            "Tách hoàn toàn khỏi bao bì nilon, tăm tre, đồ nhựa dùng một lần",
            "Gạn hết nước canh hoặc nước sốt thừa để giảm mùi hôi và chảy nước",
            "Tự ủ phân compost tại nhà nếu có sân vườn hoặc thùng ủ vi sinh",
            "Bỏ vào thùng rác hữu cơ màu xanh lá"
        ],
        "combustibility": {
            "is_combustible": False,
            "status": "Khó cháy (Độ ẩm cao 70% - 85%)",
            "badge_color": "#16a34a",
            "analysis": "Rác thực phẩm và phụ phẩm hữu cơ chứa lượng nước rất lớn (70-85%). Đốt rác hữu cơ tiêu tốn nhiệt lượng của lò thiêu đốt, hạ thấp nhiệt độ buồng đốt và sinh ra khói độc chứa muội ẩm.",
            "warning": "Không nên đốt. Đốt rác ướt gây lãng phí năng lượng và ô nhiễm khói mù."
        },
        "recyclability": {
            "level": "Tái sinh sinh học (Ủ Phân Vi Sinh / Biogas)",
            "difficulty_score": "Rất dễ",
            "badge_color": "#10b981",
            "analysis": "Rác hữu cơ là nguồn tài nguyên dinh dưỡng tuyệt vời để ủ thành phân bón mùn hữu cơ (compost) giàu đạm/khoáng cho cây trồng, hoặc ủ kỵ khí sinh khí metan sạch (Biogas) phục vụ đun nấu và phát điện.",
            "economic_value": "Rất thiết thực — Cải tạo đất trồng trọt và giảm phụ thuộc phân hóa học."
        },
        "processing_steps": [
            "1. Tại nguồn: Chắt ráo nước, loại bỏ rác vô cơ lẫn vào.",
            "2. Thu gom: Vận chuyển hàng ngày về nhà máy chế biến phân vi sinh.",
            "3. Nhà máy: Ủ hiếu khí 45 ngày với chế phẩm vi sinh bản địa hoặc ủ biogas sinh điện năng."
        ],
        "environmental_impact": {
            "decomposition_time": "1 - 4 tuần (Phân hủy hoàn toàn thành mùn đất dinh dưỡng)",
            "carbon_saving": "Phân loại rác hữu cơ tránh phát thải khí nhà kính Methane (CH4) tại bãi rác lộ thiên."
        }
    },
    "other": {
        "label_vi": "Rác khác / Rác nguy hại (Other)",
        "bin": "Thùng rác sinh hoạt / Điểm thu gom nguy hại riêng",
        "bin_color": "#64748b",
        "guidance": "Bao gồm rác tồn dư khó phân loại, cao su, vải vụn và các loại rác nguy hại (pin, bóng đèn huỳnh quang, rác điện tử). Cần tách riêng rác nguy hại đưa về trạm thu gom chuyên dụng.",
        "tips": [
            "TUYỆT ĐỐI KHÔNG vứt pin cũ, bóng đèn, nhiệt kế thủy ngân vào thùng rác chung",
            "Gom pin cũ vào hộp riêng và gửi đến điểm thu gom nguy hại của siêu thị/phường",
            "Vải vụn, da giày hỏng cần gói kín bỏ thùng rác sinh hoạt",
            "Tham khảo quy định phân loại chất thải nguy hại của chính quyền địa phương"
        ],
        "combustibility": {
            "is_combustible": True,
            "status": "Nguy cơ cháy nổ cao (Đối với Pin / Rác điện tử)",
            "badge_color": "#dc2626",
            "analysis": "Nếu chứa Pin Lithium-ion, bình ắc quy hoặc rác điện tử: cực kỳ dễ chập cháy và phát nổ khi gặp nhiệt độ cao hoặc bị va đập trong máy ép rác. Nếu là giẻ lau dầu mỡ hoặc cao su: cháy sinh khói đen khét lẹt chứa hợp chất lưu huỳnh độc hại.",
            "warning": "CẢNH BÁO NGUY HIỂM: Pin và rác điện tử có nguy cơ gây cháy nổ chết người nếu đem thiêu đốt!"
        },
        "recyclability": {
            "level": "Khó tái chế / Cần xử lý công nghệ đặc thù",
            "difficulty_score": "Khó / Đặc thù",
            "badge_color": "#dc2626",
            "analysis": "Vật liệu hỗn hợp (composite) hoặc cao su lưu hóa rất khó phân tách để tái sinh. Rác điện tử chứa kim loại quý (vàng, bạc, đồng) nhưng cũng chứa chì, thủy ngân, cadimi đòi hỏi dây chuyền bóc tách công nghệ cao đạt chuẩn an toàn.",
            "economic_value": "Đặc thù — Cần chi phí xử lý và thu hồi kim loại chuyên biệt."
        },
        "processing_steps": [
            "1. Tại nguồn: Cách ly pin và bóng đèn ra khỏi dòng rác sinh hoạt thông thường.",
            "2. Thu gom: Chuyển giao cho đơn vị có giấy phép xử lý chất thải nguy hại (Hazardous Waste).",
            "3. Nhà máy: Bóc tách hóa học thu hồi kim loại nặng hoặc chôn lấp cô lập hóa học an toàn."
        ],
        "environmental_impact": {
            "decomposition_time": "Vải sợi tổng hợp: 50-100 năm; Pin rò rỉ: Ô nhiễm nguồn nước vĩnh viễn",
            "carbon_saving": "Thu gom đúng 1 viên pin cúc áo bảo vệ được 500 lít nước và 1m3 đất khỏi nhiễm độc thủy ngân."
        }
    }
}

# Cấu hình ngưỡng tin cậy (Uncertainty Threshold)
DEFAULT_UNCERTAINTY_THRESHOLD = 0.65
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAIN_CONFIDENCE_THRESHOLD", str(DEFAULT_UNCERTAINTY_THRESHOLD)))

# Cấu hình ngưỡng kiểm tra chất lượng hình ảnh
MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", "64"))
REJECT_DARK_BRIGHTNESS = float(os.getenv("REJECT_DARK_BRIGHTNESS", "6.0"))
REJECT_BRIGHT_BRIGHTNESS = float(os.getenv("REJECT_BRIGHT_BRIGHTNESS", "252.0"))

WARN_LOW_RES_DIMENSION = int(os.getenv("WARN_LOW_RES_DIMENSION", "160"))
WARN_DARK_BRIGHTNESS = float(os.getenv("WARN_DARK_BRIGHTNESS", "40.0"))
WARN_BRIGHT_BRIGHTNESS = float(os.getenv("WARN_BRIGHT_BRIGHTNESS", "230.0"))
WARN_BLUR_SHARPNESS = float(os.getenv("WARN_BLUR_SHARPNESS", "45.0"))


# =============================================================================
# Pydantic Schemas
# =============================================================================

class CombustibilityInfo(BaseModel):
    is_combustible: bool
    status: str
    badge_color: str
    analysis: str
    warning: Optional[str] = None

class RecyclabilityInfo(BaseModel):
    level: str
    difficulty_score: str
    badge_color: str
    analysis: str
    economic_value: Optional[str] = None

class RecommendationInfo(BaseModel):
    label_vi: str
    bin: str
    bin_color: str
    guidance: str
    tips: List[str]
    combustibility: Optional[CombustibilityInfo] = None
    recyclability: Optional[RecyclabilityInfo] = None
    processing_steps: Optional[List[str]] = None
    environmental_impact: Optional[Dict[str, str]] = None

class TopPredictionItem(BaseModel):
    class_name: str = Field(..., description="Mã định danh lớp rác thải (vd: plastic, paper, metal, glass, organic, other)")
    label_vi: str = Field(..., description="Tên hiển thị tiếng Việt của danh mục")
    probability: float = Field(..., description="Xác suất phân loại Softmax của lớp (0.0 đến 1.0)")

class ImageQualityInfo(BaseModel):
    is_acceptable: bool = Field(default=True, description="Ảnh đáp ứng đủ điều kiện chất lượng để suy luận")
    has_warnings: bool = Field(default=False, description="Có cảnh báo về chất lượng ánh sáng, độ phân giải hoặc độ nét")
    brightness_score: float = Field(..., description="Độ sáng trung bình của ảnh (thang đo 0 đến 255)")
    sharpness_score: float = Field(..., description="Độ sắc nét ước tính qua phương sai Laplacian (càng cao càng sắc nét)")
    warnings: List[str] = Field(default_factory=list, description="Danh sách các lưu ý cải thiện chất lượng ảnh")

class PredictionResponse(BaseModel):
    success: bool = True
    mock: bool = False
    record_id: Optional[int] = Field(default=None, description="Mã định danh bản ghi lịch sử trong SQLite")
    predicted_class: str
    label_vi: str
    confidence: float
    probabilities: Dict[str, float]
    top_predictions: List[TopPredictionItem] = Field(default_factory=list, description="Top-3 lớp có xác suất cao nhất")
    is_uncertain: bool = Field(default=False, description="Cảnh báo độ tin cậy thấp nếu confidence < ngưỡng cài đặt")
    uncertainty_threshold: float = Field(default=DEFAULT_UNCERTAINTY_THRESHOLD, description="Ngưỡng tin cậy đang áp dụng")
    uncertainty_note: Optional[str] = Field(default=None, description="Khuyến nghị cho người dùng khi ảnh có độ tin cậy thấp")
    quality_assessment: Optional[ImageQualityInfo] = Field(default=None, description="Kết quả đánh giá chất lượng kỹ thuật ảnh")
    recommendation: RecommendationInfo
    filename: str
    content_type: str
    image_size_bytes: int
    image_dimensions: Dict[str, int]
    image_format: str
    gradcam_overlay: Optional[str] = Field(default=None, description="Ảnh trực quan hóa Grad-CAM (XAI) Data URI Base64")
    gradcam_heatmap: Optional[str] = Field(default=None, description="Bản đồ nhiệt Grad-CAM độc lập Data URI Base64")

class ImageUrlRequest(BaseModel):
    image_url: str = Field(..., description="Đường dẫn URL công khai của hình ảnh rác thải cần phân loại")

class BatchPredictionItem(BaseModel):
    filename: str
    success: bool
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    total: int
    successful: int
    failed: int
    items: List[BatchPredictionItem]

class FeedbackRequest(BaseModel):
    feedback_status: str = Field("correct", description="Trạng thái đánh giá: 'correct' (Đúng), 'incorrect' (Sai), 'uncertain' (Không chắc chắn)")
    is_correct: Optional[bool] = Field(None, description="Tương thích ngược: True nếu đúng, False nếu sai")
    correct_class: Optional[str] = Field(None, description="Nhãn đúng thực tế nếu người dùng sửa lại")
    note: Optional[str] = Field(None, description="Góp ý hoặc mô tả thêm từ người dùng")

class FeedbackResponse(BaseModel):
    success: bool = True
    message: str
    record_id: int

class HistoryRecordItem(BaseModel):
    id: int
    created_at: str
    filename: str
    predicted_class: str
    label_vi: str
    confidence: float
    is_uncertain: bool
    uncertainty_note: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    top_predictions: Optional[List[TopPredictionItem]] = None
    quality_assessment: Optional[ImageQualityInfo] = None
    recommendation: Optional[RecommendationInfo] = None
    image_dimensions: Optional[Dict[str, int]] = None
    image_format: Optional[str] = None
    image_size_bytes: Optional[int] = None
    feedback_is_correct: Optional[bool] = None
    feedback_status: Optional[str] = None
    feedback_correct_class: Optional[str] = None
    feedback_note: Optional[str] = None
    feedback_at: Optional[str] = None

class HistoryListResponse(BaseModel):
    items: List[HistoryRecordItem]
    total: int
    page: int
    page_size: int
    total_pages: int

class HistoryDeleteResponse(BaseModel):
    success: bool = True
    message: str
    deleted_id: Optional[int] = None
    deleted_count: Optional[int] = None

class HistoryStatsResponse(BaseModel):
    total_predictions: int
    uncertain_predictions: int
    class_counts: Dict[str, int]

# Schemas for Analytics Dashboard
class ClassDistributionItem(BaseModel):
    class_name: str
    label_vi: str
    count: int
    percentage: float
    avg_confidence: float

class ConfidenceBucketItem(BaseModel):
    range_key: str
    label: str
    count: int
    percentage: float

class DailyTrendItem(BaseModel):
    date: str
    count: int

class WeeklyTrendItem(BaseModel):
    week: str
    count: int

class QualityMetricsSummary(BaseModel):
    images_with_warnings: int
    clean_images_count: int
    warning_rate_percentage: float

class FeedbackMetricsSummary(BaseModel):
    total_feedbacks: int
    agreed_count: int
    disagreed_count: int
    uncertain_feedback_count: int
    accuracy_rate: float

class FilterAppliedInfo(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class DashboardAnalyticsResponse(BaseModel):
    total_predictions: int
    average_confidence: float
    uncertain_count: int
    uncertain_percentage: float
    most_frequent_class: Optional[ClassDistributionItem] = None
    class_distribution: List[ClassDistributionItem]
    confidence_buckets: List[ConfidenceBucketItem]
    daily_trends: List[DailyTrendItem]
    weekly_trends: List[WeeklyTrendItem]
    quality_metrics: QualityMetricsSummary
    feedback_metrics: Optional[FeedbackMetricsSummary] = None
    error_cases: List[HistoryRecordItem] = Field(default_factory=list)
    recent_records: List[HistoryRecordItem]
    filter_applied: FilterAppliedInfo
    disclaimer: str = (
        "Lưu ý: Số lần nhận diện phản ánh lượt suy luận của mô hình AI, "
        "không đồng nghĩa với khối lượng rác thực tế đã được thu gom hay tái chế."
    )

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    ai_model_loaded: bool
    device: str
    model_architecture: str
    checkpoint_loaded: bool
    target_classes: List[str]
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD
    error: Optional[str] = None

class RecyclingCenterItem(BaseModel):
    id: int
    name: str
    city: str
    district: str
    address: str
    accepted_materials: List[str]
    contact: Optional[str] = None
    opening_hours: Optional[str] = None
    notes: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================

def find_checkpoint_path() -> Path:
    """Tìm đường dẫn tệp checkpoint từ project root hoặc thư mục làm việc."""
    root_candidate = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "mobilenetv3_best.pth"
    if root_candidate.exists():
        return root_candidate
    
    cwd_candidate = Path.cwd() / "checkpoints" / "mobilenetv3_best.pth"
    if cwd_candidate.exists():
        return cwd_candidate

    return root_candidate


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Nạp trọng số MobileNetV3-Large từ checkpoint đã huấn luyện."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    target_classes = checkpoint.get("target_classes", ["plastic", "paper", "metal", "glass", "organic", "other"])

    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(target_classes))

    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        f"Checkpoint loaded successfully from '{ckpt_path.name}'. Device: {device}, Classes: {target_classes}"
    )
    return model, target_classes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời FastAPI: Nạp mô hình AI một lần duy nhất lúc khởi động."""
    logger.info("Starting up FastAPI application...")

    try:
        init_db()
    except Exception as db_init_err:
        logger.error(f"Failed to initialize SQLite database: {db_init_err}", exc_info=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app.state.device = str(device)
    app.state.model = None
    app.state.classes = ["plastic", "paper", "metal", "glass", "organic", "other"]
    app.state.model_loaded = False
    app.state.model_error = None

    app.state.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    ckpt_path = find_checkpoint_path()
    try:
        model, classes = load_model_from_checkpoint(ckpt_path, device)
        app.state.model = model
        app.state.classes = classes
        app.state.model_loaded = True

        dummy_input = torch.zeros(1, 3, 224, 224, device=device)
        with torch.inference_mode():
            _ = model(dummy_input)
        logger.info("AI model inference engine warmed up and ready.")
    except Exception as e:
        logger.error(f"Failed to load AI model checkpoint from {ckpt_path}: {e}", exc_info=True)
        app.state.model_loaded = False
        app.state.model_error = str(e)

    yield

    logger.info("Shutting down FastAPI application...")
    if app.state.model is not None:
        del app.state.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="AI-Based Waste Recognition & Recycling Recommendation API",
    description="Backend API for waste classification, Grad-CAM XAI, combustibility & recyclability analysis.",
    version="0.4.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, summary="Service & Model Health Check")
async def health_check():
    """Kiểm tra tình trạng hoạt động của backend service và trạng thái nạp mô hình AI."""
    is_loaded = getattr(app.state, "model_loaded", False)
    status_str = "ok" if is_loaded else "degraded"
    
    return HealthResponse(
        status=status_str,
        service="waste-recognition-backend",
        version="0.4.0",
        ai_model_loaded=is_loaded,
        device=getattr(app.state, "device", "unknown"),
        model_architecture="mobilenet_v3_large",
        checkpoint_loaded=is_loaded,
        target_classes=getattr(app.state, "classes", []),
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        error=getattr(app.state, "model_error", None)
    )


def assess_image_quality(img: Image.Image) -> Tuple[bool, Optional[str], ImageQualityInfo]:
    """Đánh giá chất lượng kỹ thuật của ảnh đầu vào trước khi đưa vào mạng nơ-ron."""
    width, height = img.size
    min_dim = min(width, height)

    if min_dim < MIN_IMAGE_DIMENSION:
        return (
            False,
            f"Kích thước ảnh quá nhỏ ({width} × {height} px). "
            f"Độ phân giải tối thiểu hợp lệ là {MIN_IMAGE_DIMENSION} × {MIN_IMAGE_DIMENSION} px.",
            ImageQualityInfo(
                is_acceptable=False,
                has_warnings=True,
                brightness_score=0.0,
                sharpness_score=0.0,
                warnings=["Kích thước ảnh dưới ngưỡng tối thiểu"]
            )
        )

    gray = img.convert("L")
    gray_arr = np.array(gray, dtype=np.float32)
    brightness = float(np.mean(gray_arr))

    if brightness < REJECT_DARK_BRIGHTNESS:
        return (
            False,
            f"Ảnh quá tối hoặc gần như đen hoàn toàn (độ sáng: {brightness:.1f}/255). "
            "Không thể quan sát vật thể rác thải. Vui lòng chụp lại nơi có đủ ánh sáng.",
            ImageQualityInfo(
                is_acceptable=False,
                has_warnings=True,
                brightness_score=round(brightness, 1),
                sharpness_score=0.0,
                warnings=["Ảnh tối đen hoàn toàn"]
            )
        )

    if brightness > REJECT_BRIGHT_BRIGHTNESS:
        return (
            False,
            f"Ảnh bị cháy sáng hoặc trắng xóa hoàn toàn (độ sáng: {brightness:.1f}/255). "
            "Không thể nhận diện chi tiết vật thể. Vui lòng chụp lại giảm bớt ánh sáng chói.",
            ImageQualityInfo(
                is_acceptable=False,
                has_warnings=True,
                brightness_score=round(brightness, 1),
                sharpness_score=0.0,
                warnings=["Ảnh cháy sáng hoàn toàn"]
            )
        )

    if gray_arr.shape[0] > 4 and gray_arr.shape[1] > 4:
        laplacian = (
            gray_arr[:-2, 1:-1] + gray_arr[2:, 1:-1] +
            gray_arr[1:-1, :-2] + gray_arr[1:-1, 2:] -
            4 * gray_arr[1:-1, 1:-1]
        )
        sharpness = float(np.var(laplacian))
    else:
        sharpness = 0.0

    warnings: List[str] = []

    if min_dim < WARN_LOW_RES_DIMENSION:
        warnings.append(
            f"Độ phân giải ảnh tương đối nhỏ ({width} × {height} px). "
            f"Khuyến nghị tải ảnh từ {WARN_LOW_RES_DIMENSION}px trở lên để mô hình nhận diện rõ chi tiết."
        )

    if brightness < WARN_DARK_BRIGHTNESS:
        warnings.append(
            f"Ảnh chụp trong điều kiện thiếu sáng (độ sáng: {brightness:.1f}/255). "
            "Khuyến nghị bật đèn flash hoặc chụp ở nơi có ánh sáng tốt hơn."
        )
    elif brightness > WARN_BRIGHT_BRIGHTNESS:
        warnings.append(
            f"Ảnh có độ sáng rất cao / có thể bị chói lóa (độ sáng: {brightness:.1f}/255). "
            "Khuyến nghị tránh chụp ngược sáng hoặc bóng đèn chiếu thẳng."
        )

    if sharpness < WARN_BLUR_SHARPNESS:
        warnings.append(
            f"Ảnh có dấu hiệu bị mờ hoặc mất nét (độ sắc nét: {sharpness:.1f}). "
            "Khuyến nghị giữ chắc tay và lấy nét rõ vào vật thể khi chụp."
        )

    quality_info = ImageQualityInfo(
        is_acceptable=True,
        has_warnings=len(warnings) > 0,
        brightness_score=round(brightness, 1),
        sharpness_score=round(sharpness, 1),
        warnings=warnings
    )

    return (True, None, quality_info)


def process_waste_image_bytes(
    contents: bytes,
    filename: str,
    content_type: Optional[str] = None,
    include_gradcam: bool = True
) -> PredictionResponse:
    """Hàm xử lý suy luận rác thải cốt lõi kèm phân tích thuộc tính cháy & tái chế."""
    if not getattr(app.state, "model_loaded", False) or app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI inference engine is unavailable or failed to load. Please check server logs."
        )

    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp ảnh tải lên rỗng (0 bytes)."
        )

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dung lượng tệp vượt quá giới hạn cho phép {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    try:
        image_stream = io.BytesIO(contents)
        pil_image = Image.open(image_stream)
        pil_image.verify()
        img_format = pil_image.format
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tệp '{filename}' không phải là hình ảnh hợp lệ hoặc bị hỏng."
        )

    if not img_format or img_format.upper() not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Định dạng ảnh '{img_format}' không được hỗ trợ. Hỗ trợ: JPEG, PNG, WebP."
        )

    image_stream.seek(0)
    with Image.open(image_stream) as img:
        img_dimensions = {"width": img.width, "height": img.height}
        rgb_image = img.convert("RGB")

        is_valid, reject_reason, quality_info = assess_image_quality(rgb_image)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=reject_reason
            )

        tensor_input = app.state.transform(rgb_image).unsqueeze(0)

    device = torch.device(app.state.device)
    tensor_input = tensor_input.to(device)
    model = app.state.model
    classes = app.state.classes

    try:
        with torch.inference_mode():
            logits = model(tensor_input)
            probs = torch.softmax(logits, dim=1)[0]
    except Exception as e:
        logger.error(f"Error during model inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Đã xảy ra lỗi trong quá trình suy luận mô hình AI."
        )

    probs_list = probs.cpu().tolist()
    pred_idx = int(torch.argmax(probs).item())
    predicted_class = classes[pred_idx]
    confidence = float(probs_list[pred_idx])

    probabilities: Dict[str, float] = {}
    for i, cls_name in enumerate(classes):
        probabilities[cls_name] = round(float(probs_list[i]), 6)

    sorted_class_indices = sorted(range(len(classes)), key=lambda i: probs_list[i], reverse=True)
    top_predictions: List[TopPredictionItem] = []
    for idx in sorted_class_indices[:3]:
        c_name = classes[idx]
        c_rec = RECOMMENDATIONS.get(c_name, RECOMMENDATIONS["other"])
        top_predictions.append(
            TopPredictionItem(
                class_name=c_name,
                label_vi=c_rec["label_vi"],
                probability=round(probs_list[idx], 4)
            )
        )

    is_uncertain = confidence < UNCERTAINTY_THRESHOLD
    uncertainty_note = None
    if is_uncertain:
        uncertainty_note = (
            f"Độ tin cậy dự đoán ({confidence * 100:.1f}%) thấp hơn ngưỡng an toàn ({UNCERTAINTY_THRESHOLD * 100:.0f}%). "
            "Hình ảnh có thể bị mờ, thiếu sáng, chụp góc nghiêng hoặc chứa nhiều vật liệu lẫn lộn. "
            "Khuyến nghị bạn kiểm tra lại hoặc chụp cận cảnh rõ nét hơn."
        )

    rec_data = RECOMMENDATIONS.get(predicted_class, RECOMMENDATIONS["other"])
    recommendation = RecommendationInfo(
        label_vi=rec_data["label_vi"],
        bin=rec_data["bin"],
        bin_color=rec_data["bin_color"],
        guidance=rec_data["guidance"],
        tips=rec_data["tips"],
        combustibility=CombustibilityInfo(**rec_data["combustibility"]) if rec_data.get("combustibility") else None,
        recyclability=RecyclabilityInfo(**rec_data["recyclability"]) if rec_data.get("recyclability") else None,
        processing_steps=rec_data.get("processing_steps"),
        environmental_impact=rec_data.get("environmental_impact")
    )

    gradcam_overlay_b64 = None
    gradcam_heatmap_b64 = None
    if include_gradcam:
        try:
            gradcam_overlay_b64, gradcam_heatmap_b64 = generate_gradcam_overlay(
                model=model,
                input_tensor=tensor_input,
                class_idx=pred_idx,
                original_pil_img=rgb_image,
                alpha=0.45
            )
        except Exception as xai_err:
            logger.warning(f"Grad-CAM generation failed for '{filename}': {xai_err}")

    history_record_id: Optional[int] = None
    try:
        saved_record = save_prediction_record({
            "filename": filename,
            "predicted_class": predicted_class,
            "label_vi": rec_data["label_vi"],
            "confidence": round(confidence, 4),
            "is_uncertain": is_uncertain,
            "uncertainty_note": uncertainty_note,
            "probabilities": probabilities,
            "top_predictions": [item.model_dump() for item in top_predictions],
            "quality_assessment": quality_info.model_dump() if quality_info else None,
            "recommendation": recommendation.model_dump(),
            "image_dimensions": img_dimensions,
            "image_format": img_format,
            "image_size_bytes": file_size
        })
        history_record_id = saved_record.get("id")
    except Exception as db_save_err:
        logger.error(f"Failed to save prediction record to database: {db_save_err}", exc_info=True)

    return PredictionResponse(
        success=True,
        mock=False,
        record_id=history_record_id,
        predicted_class=predicted_class,
        label_vi=rec_data["label_vi"],
        confidence=round(confidence, 4),
        probabilities=probabilities,
        top_predictions=top_predictions,
        is_uncertain=is_uncertain,
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        uncertainty_note=uncertainty_note,
        quality_assessment=quality_info,
        recommendation=recommendation,
        filename=filename,
        content_type=content_type or f"image/{img_format.lower()}",
        image_size_bytes=file_size,
        image_dimensions=img_dimensions,
        image_format=img_format,
        gradcam_overlay=gradcam_overlay_b64,
        gradcam_heatmap=gradcam_heatmap_b64
    )


# =============================================================================
# Prediction Endpoints
# =============================================================================

@app.post("/api/predict", response_model=PredictionResponse, summary="Classify Single Waste Image")
async def predict_waste(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(True, description="Kèm bản đồ nhiệt giải thích Grad-CAM XAI")
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không có tệp ảnh nào được đính kèm."
        )

    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lỗi khi đọc luồng tệp tải lên: {e}"
        )

    return process_waste_image_bytes(
        contents=contents,
        filename=file.filename,
        content_type=file.content_type,
        include_gradcam=include_gradcam
    )


@app.post("/api/predict/url", response_model=PredictionResponse, summary="Classify Waste from Public Image URL")
async def predict_waste_by_url(
    payload: ImageUrlRequest,
    include_gradcam: bool = Query(True, description="Kèm bản đồ nhiệt giải thích Grad-CAM XAI")
):
    url = payload.image_url.strip()
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Đường dẫn URL không hợp lệ (phải bắt đầu bằng http:// hoặc https://)."
        )

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AIWasteClassifier/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10.0) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Ảnh từ URL vượt quá dung lượng tối đa 10 MB."
                )
            contents = response.read(MAX_UPLOAD_SIZE_BYTES + 1)
            if len(contents) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Ảnh từ URL vượt quá dung lượng tối đa 10 MB."
                )
            content_type = response.headers.get("Content-Type", "image/jpeg")
    except HTTPException:
        raise
    except Exception as net_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể tải ảnh từ URL cung cấp: {net_err}"
        )

    path_name = Path(parsed.path).name or "web_image.jpg"
    return process_waste_image_bytes(
        contents=contents,
        filename=path_name,
        content_type=content_type,
        include_gradcam=include_gradcam
    )


@app.post("/api/predict/batch", response_model=BatchPredictionResponse, summary="Batch Waste Classification")
async def predict_waste_batch(
    files: List[UploadFile] = File(...),
    include_gradcam: bool = Query(False, description="Tắt mặc định Grad-CAM trong batch để tối ưu tốc độ")
):
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vui lòng cung cấp ít nhất 1 tệp ảnh để phân loại hàng loạt."
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Số lượng ảnh trong 1 lần phân loại hàng loạt tối đa là 10 ảnh."
        )

    items: List[BatchPredictionItem] = []
    success_count = 0

    for file in files:
        try:
            contents = await file.read()
            res = process_waste_image_bytes(
                contents=contents,
                filename=file.filename or "unknown.jpg",
                content_type=file.content_type,
                include_gradcam=include_gradcam
            )
            items.append(BatchPredictionItem(filename=file.filename or "unknown.jpg", success=True, result=res))
            success_count += 1
        except HTTPException as he:
            items.append(BatchPredictionItem(filename=file.filename or "unknown.jpg", success=False, error=he.detail))
        except Exception as e:
            items.append(BatchPredictionItem(filename=file.filename or "unknown.jpg", success=False, error=str(e)))

    return BatchPredictionResponse(
        total=len(files),
        successful=success_count,
        failed=len(files) - success_count,
        items=items
    )


# =============================================================================
# Prediction History, Feedback & Export Endpoints
# =============================================================================

@app.get("/api/history", response_model=HistoryListResponse, summary="List Prediction History with Pagination & Filters")
def list_history(
    page: int = Query(1, ge=1, description="Số trang (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Số bản ghi mỗi trang (1-100)"),
    class_name: Optional[str] = Query(None, description="Lọc theo nhóm rác thải"),
    start_date: Optional[str] = Query(None, description="Lọc từ ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Lọc đến ngày kết thúc (YYYY-MM-DD)")
):
    items, total = get_history_records(
        page=page,
        page_size=page_size,
        class_name=class_name,
        start_date=start_date,
        end_date=end_date
    )
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return HistoryListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@app.get("/api/history/export", summary="Export Prediction History as CSV")
def export_history_csv(
    class_name: Optional[str] = Query(None, description="Lọc theo nhóm rác thải"),
    start_date: Optional[str] = Query(None, description="Lọc từ ngày bắt đầu (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Lọc đến ngày kết thúc (YYYY-MM-DD)")
):
    records = get_all_history_for_export(class_name=class_name, start_date=start_date, end_date=end_date)
    output = io.StringIO()
    output.write("\ufeff")  # BOM UTF-8 for Excel
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Thời Gian (UTC)",
        "Tên Tệp",
        "Nhóm Dự Đoán (Mã)",
        "Danh Mục Tiếng Việt",
        "Độ Tin Cậy (%)",
        "Cảnh Báo Không Chắc Chắn",
        "Đánh Giá Người Dùng (Feedback)",
        "Nhãn Thực Tế Sửa Lại",
        "Ghi Chú Đóng Góp"
    ])

    for r in records:
        status_map = {"correct": "Đúng", "incorrect": "Báo sai", "uncertain": "Không chắc chắn"}
        fb_status = status_map.get(r.get("feedback_status")) or ("Đúng" if r.get("feedback_is_correct") is True else ("Sai" if r.get("feedback_is_correct") is False else "Chưa đánh giá"))
        writer.writerow([
            r.get("id"),
            r.get("created_at"),
            r.get("filename"),
            r.get("predicted_class"),
            r.get("label_vi"),
            f"{r.get('confidence', 0.0) * 100:.2f}%",
            "Có" if r.get("is_uncertain") else "Không",
            fb_status,
            r.get("feedback_correct_class") or "",
            r.get("feedback_note") or ""
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=lich_su_phan_loai_rac.csv"}
    )


@app.get("/api/analytics/errors/export", summary="Export Error Cases Dataset for Model Retraining")
def export_error_cases_dataset():
    """Xuất danh sách các ca lỗi / không chắc chắn để nhóm phát triển tải về huấn luyện lại mô hình (Active Learning)."""
    errors = get_error_cases_for_export()
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Thời Gian",
        "Tên Tệp Ảnh",
        "AI Dự Đoán",
        "Độ Tin Cậy (%)",
        "Mô Hình Báo Không Chắc Chắn",
        "Phản Hồi Người Dùng",
        "Nhãn Đúng Thực Tế",
        "Ghi Chú Chi Tiết"
    ])

    for e in errors:
        writer.writerow([
            e.get("id"),
            e.get("created_at"),
            e.get("filename"),
            e.get("predicted_class"),
            f"{e.get('confidence', 0.0) * 100:.2f}%",
            "Có" if e.get("is_uncertain") else "Không",
            e.get("feedback_status") or ("Đúng" if e.get("feedback_is_correct") is True else "Báo sai"),
            e.get("feedback_correct_class") or "",
            e.get("feedback_note") or ""
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=tap_du_lieu_ca_loi_active_learning.csv"}
    )


@app.post("/api/history/{record_id}/feedback", response_model=FeedbackResponse, summary="Submit User Feedback on Prediction")
def submit_prediction_feedback(record_id: int, payload: FeedbackRequest):
    record = get_history_record_by_id(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bản ghi lịch sử với ID {record_id} không tồn tại."
        )

    success = save_user_feedback(
        record_id=record_id,
        is_correct=payload.is_correct,
        correct_class=payload.correct_class,
        note=payload.note,
        feedback_status=payload.feedback_status
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể lưu đánh giá phản hồi vào cơ sở dữ liệu."
        )

    if payload.feedback_status == "correct":
        msg = "Cảm ơn bạn đã xác nhận AI dự đoán chuẩn xác!"
    elif payload.feedback_status == "uncertain":
        msg = "Đã ghi nhận ca không chắc chắn để nhóm phát triển thẩm định!"
    else:
        msg = "Đã lưu ca báo sai kèm nhãn thực tế để bổ sung vào tập dữ liệu cải thiện mô hình!"

    return FeedbackResponse(success=True, message=msg, record_id=record_id)


@app.get("/api/history/stats", response_model=HistoryStatsResponse, summary="Get Prediction History Statistics")
def get_history_stats():
    stats = get_history_statistics()
    return HistoryStatsResponse(**stats)


@app.get("/api/history/{record_id}", response_model=HistoryRecordItem, summary="Get Single Prediction History Record")
def get_history_record(record_id: int):
    record = get_history_record_by_id(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bản ghi lịch sử với ID {record_id} không tồn tại."
        )
    return record


@app.delete("/api/history/{record_id}", response_model=HistoryDeleteResponse, summary="Delete Single Prediction History Record")
def delete_history_record(record_id: int):
    success = delete_history_record_by_id(record_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bản ghi lịch sử với ID {record_id} không tồn tại."
        )
    return HistoryDeleteResponse(
        success=True,
        message=f"Bản ghi lịch sử ID {record_id} đã được xóa thành công.",
        deleted_id=record_id
    )


@app.delete("/api/history", response_model=HistoryDeleteResponse, summary="Clear All Prediction History")
def clear_history():
    deleted_count = clear_all_history()
    return HistoryDeleteResponse(
        success=True,
        message=f"Đã xóa thành công toàn bộ {deleted_count} bản ghi lịch sử.",
        deleted_count=deleted_count
    )


# =============================================================================
# Analytics & Model Evaluation Endpoints
# =============================================================================

@app.get("/api/analytics/dashboard", response_model=DashboardAnalyticsResponse, summary="Get Full Analytics Dashboard Metrics")
def get_dashboard_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Truy vấn dữ liệu tổng hợp phân tích thực tế từ SQLite database cho Dashboard:
    - Tổng số lần nhận diện
    - Tỷ lệ từng loại rác
    - Loại rác được nhận diện nhiều nhất
    - Biểu đồ theo ngày hoặc tuần
    - Thống kê kết quả đúng/sai từ phản hồi người dùng
    - Danh sách các ca lỗi / không chắc chắn để nhóm phát triển cải thiện mô hình
    """
    data = get_analytics_summary(start_date=start_date, end_date=end_date)
    return DashboardAnalyticsResponse(**data)


@app.get("/api/model/metrics", summary="Get Trained Model Evaluation Metrics & Training History")
def get_model_evaluation_metrics():
    project_root = Path(__file__).resolve().parent.parent.parent
    eval_file = project_root / "checkpoints" / "test_evaluation.json"
    history_file = project_root / "checkpoints" / "training_history.json"

    eval_data = {}
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read test_evaluation.json: {e}")

    history_data = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read training_history.json: {e}")

    model_spec = {
        "architecture": "MobileNetV3-Large",
        "pretrained_backbone": "ImageNet-1K (weights=None fine-tuned)",
        "dataset": "RealWaste (4,752 images total)",
        "train_split": 3327,
        "val_split": 712,
        "test_split": 713,
        "num_classes": 6,
        "input_resolution": "224x224 RGB",
        "parameters_count": "5.48 Million Parameters",
        "model_file_size": "17.0 MB (.pth)",
        "hardware_used": "NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)",
        "average_inference_latency_ms": 14.5,
        "classes_order": ["plastic", "paper", "metal", "glass", "organic", "other"]
    }

    return {
        "model_spec": model_spec,
        "evaluation": eval_data,
        "training_history": history_data
    }


# =============================================================================
# Recycling Centers Directory Endpoint
# =============================================================================

RECYCLING_CENTERS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Lagom Vietnam - Trạm Thu Gom Hộp Sữa & Vỏ Lon",
        "city": "Hà Nội",
        "district": "Cầu Giấy",
        "address": "Số 10 ngõ 84 Trần Thái Tông, Dịch Vọng Hậu, Cầu Giấy, Hà Nội",
        "accepted_materials": ["paper", "metal", "plastic"],
        "contact": "0987 654 321",
        "opening_hours": "08:30 - 17:30 (Thứ 2 - Thứ 7)",
        "notes": "Chuyên thu gom và tái chế vỏ hộp sữa giấy, lon nhôm, nắp chai nhựa HDPE."
    },
    {
        "id": 2,
        "name": "Green Life - Điểm Thu Gom Rác Tái Chế & Pin Cũ Đổi Cây",
        "city": "Hà Nội",
        "district": "Cầu Giấy",
        "address": "Số 5 ngõ 106 Hoàng Quốc Việt, Nghĩa Đô, Cầu Giấy, Hà Nội",
        "accepted_materials": ["plastic", "paper", "metal", "glass", "other"],
        "contact": "0912 345 678",
        "opening_hours": "09:00 - 18:00 (Hàng ngày)",
        "notes": "Tiếp nhận pin cũ, rác thải điện tử nhỏ, giấy báo cũ, chai lọ nhựa và thủy tinh."
    },
    {
        "id": 3,
        "name": "Điểm Thu Hồi Pin Cũ & Rác Nguy Hại - UBND Phường Tràng Tiền",
        "city": "Hà Nội",
        "district": "Hoàn Kiếm",
        "address": "Số 2 Cổ Tân, Tràng Tiền, Hoàn Kiếm, Hà Nội",
        "accepted_materials": ["other"],
        "contact": "024 3824 5555",
        "opening_hours": "08:00 - 17:00 (Thứ 2 - Thứ 6)",
        "notes": "Thùng thu gom chuyên dụng cho pin cũ, bóng đèn huỳnh quang hỏng."
    },
    {
        "id": 4,
        "name": "Limart Zero Waste - Trạm Thu Gom Chai Nhựa & Mỹ Phẩm Rỗng",
        "city": "TP. Hồ Chí Minh",
        "district": "Quận 3",
        "address": "396/1 Lê Văn Sỹ, Phường 14, Quận 3, TP. Hồ Chí Minh",
        "accepted_materials": ["plastic", "glass"],
        "contact": "0898 123 456",
        "opening_hours": "09:00 - 21:00 (Hàng ngày)",
        "notes": "Chương trình đổi vỏ chai mỹ phẩm, chai nhựa sạch lấy quà tặng xanh."
    },
    {
        "id": 5,
        "name": "Ve Chai Chú Hỏa / Green Connect - Trạm Xử Lý Rác Tái Chế & Hữu Cơ",
        "city": "TP. Hồ Chí Minh",
        "district": "Gò Vấp",
        "address": "125/44 Lê Đức Thọ, Phường 17, Gò Vấp, TP. Hồ Chí Minh",
        "accepted_materials": ["organic", "plastic", "paper", "metal"],
        "contact": "0909 888 999",
        "opening_hours": "07:30 - 18:00 (Hàng ngày)",
        "notes": "Hệ thống thu gom rác hữu cơ ủ phân vi sinh và rác vô cơ tái chế công nghệ cao."
    },
    {
        "id": 6,
        "name": "Điểm Thu Gom Pin Cũ - Chuỗi Siêu Thị Co.opmart Cống Quỳnh",
        "city": "TP. Hồ Chí Minh",
        "district": "Quận 1",
        "address": "189C Cống Quỳnh, Phường Nguyễn Cư Trinh, Quận 1, TP. Hồ Chí Minh",
        "accepted_materials": ["other", "paper"],
        "contact": "028 3832 5283",
        "opening_hours": "08:00 - 22:00 (Hàng ngày)",
        "notes": "Thùng thu hồi pin cũ và vỏ hộp sữa giấy đặt ngay lối vào sảnh chính."
    },
    {
        "id": 7,
        "name": "Lagom Vietnam Cơ Sở Phía Nam",
        "city": "TP. Hồ Chí Minh",
        "district": "Quận 10",
        "address": "284/41/2 Lý Thường Kiệt, Phường 14, Quận 10, TP. Hồ Chí Minh",
        "accepted_materials": ["paper", "metal", "plastic"],
        "contact": "0934 567 890",
        "opening_hours": "08:30 - 17:30 (Thứ 2 - Thứ 7)",
        "notes": "Trạm tái chế vỏ hộp giấy Tetrapak và nhựa sạch các loại."
    },
    {
        "id": 8,
        "name": "Trung Tâm Dịch Vụ Môi Trường Đô Thị Đà Nẵng",
        "city": "Đà Nẵng",
        "district": "Hải Châu",
        "address": "471 Núi Thành, Hòa Cường Nam, Hải Châu, Đà Nẵng",
        "accepted_materials": ["plastic", "paper", "metal", "glass", "other"],
        "contact": "0236 3622 222",
        "opening_hours": "07:30 - 17:00 (Thứ 2 - Thứ 6)",
        "notes": "Điểm tập kết và phân loại rác tái chế & thiết bị điện tử gia dụng cũ."
    },
    {
        "id": 9,
        "name": "Trạm Thu Gom Rác Tái Chế Cần Thơ Xanh",
        "city": "Cần Thơ",
        "district": "Ninh Kiều",
        "address": "Đường 30 Tháng 4, Phường Hưng Lợi, Ninh Kiều, Cần Thơ",
        "accepted_materials": ["plastic", "paper", "metal"],
        "contact": "0292 3838 838",
        "opening_hours": "08:00 - 17:00 (Thứ 2 - Thứ 7)",
        "notes": "Tiếp nhận rác thải nhựa nông nghiệp, bao bì tái chế và kim loại."
    }
]

@app.get("/api/recycling-centers", response_model=List[RecyclingCenterItem], summary="Get Recycling & Disposal Centers Directory")
def get_recycling_centers(
    city: Optional[str] = Query(None, description="Lọc theo thành phố"),
    material: Optional[str] = Query(None, description="Lọc theo vật liệu")
):
    results = RECYCLING_CENTERS
    if city:
        city_lower = city.lower()
        results = [c for c in results if city_lower in c["city"].lower()]
    if material:
        mat_lower = material.lower()
        results = [c for c in results if mat_lower in [m.lower() for m in c["accepted_materials"]]]
    return results
