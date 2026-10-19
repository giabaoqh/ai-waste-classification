import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("waste_recognition_backend")

# Constraints
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Vietnamese Waste Disposal & Recycling Recommendations
RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    "plastic": {
        "label_vi": "Nhựa (Plastic)",
        "bin": "Thùng rác tái chế (Màu vàng/trắng)",
        "bin_color": "#eab308",
        "guidance": "Tráng sạch cặn chất lỏng hoặc dầu mỡ thừa, bóp xẹp chai/hộp nhựa để tiết kiệm thể tích trước khi bỏ vào thùng rác tái chế. Tháo rời nắp và nhãn mác nếu có thể. (Lưu ý: Quy định thu gom tái chế có thể khác nhau tùy địa phương).",
        "tips": [
            "Đổ hết nước và chất lỏng còn đọng lại",
            "Tráng sơ qua nước sạch nếu còn dính bẩn",
            "Bóp xẹp chai hoặc cán phẳng hộp để giảm thể tích",
            "Bỏ vào thùng rác tái chế theo quy định"
        ]
    },
    "paper": {
        "label_vi": "Giấy / Bìa carton (Paper)",
        "bin": "Thùng rác tái chế (Màu vàng/trắng)",
        "bin_color": "#eab308",
        "guidance": "Giữ giấy khô ráo và sạch sẽ, gấp phẳng các thùng bìa carton. Nếu giấy bị dính dầu mỡ thức ăn hoặc ẩm ướt nặng, vui lòng bỏ vào thùng rác sinh hoạt thông thường. (Lưu ý: Quy định thu gom tái chế có thể khác nhau tùy địa phương).",
        "tips": [
            "Giữ giấy khô ráo và sạch sẽ",
            "Gấp phẳng thùng carton để tiết kiệm diện tích",
            "Giấy dính dầu mỡ thực phẩm phải bỏ vào rác sinh hoạt",
            "Bỏ vào thùng rác tái chế"
        ]
    },
    "metal": {
        "label_vi": "Kim loại (Metal)",
        "bin": "Thùng rác tái chế (Màu vàng/trắng)",
        "bin_color": "#eab308",
        "guidance": "Đổ sạch phần thức ăn hoặc đồ uống còn lại bên trong lon/hộp kim loại; tráng sơ qua nước. Cẩn thận các cạnh sắc nhọn của nắp lon khi bỏ vào thùng rác tái chế. (Lưu ý: Quy định thu gom tái chế có thể khác nhau tùy địa phương).",
        "tips": [
            "Làm rỗng lon/hộp kim loại",
            "Tráng sạch cặn thừa bên trong",
            "Gập mép nắp sắc nhọn vào trong an toàn",
            "Bỏ vào thùng rác tái chế"
        ]
    },
    "glass": {
        "label_vi": "Thủy tinh (Glass)",
        "bin": "Thùng rác tái chế / Điểm thu gom thủy tinh riêng",
        "bin_color": "#3b82f6",
        "guidance": "Làm rỗng và tráng sạch chai lọ thủy tinh. Tháo bỏ nắp nhựa hoặc nắp kim loại để phân loại riêng. Nếu là mảnh vỡ thủy tinh, bọc kỹ bằng giấy báo hoặc bìa cứng để đảm bảo an toàn cho nhân viên thu gom. (Lưu ý: Quy định thu gom tái chế có thể khác nhau tùy địa phương).",
        "tips": [
            "Tráng sạch cặn bẩn trong chai lọ",
            "Tháo nắp nhựa hoặc nắp kim loại phân loại riêng",
            "Bọc kỹ mảnh thủy tinh vỡ bằng giấy báo",
            "Bỏ vào thùng tái chế hoặc điểm thu gom thủy tinh"
        ]
    },
    "organic": {
        "label_vi": "Rác hữu cơ (Organic)",
        "bin": "Thùng rác hữu cơ (Màu xanh lá)",
        "bin_color": "#22c55e",
        "guidance": "Bao gồm thức ăn thừa, cuống rau củ quả, bã trà, bã cà phê và lá cây. Thích hợp dùng làm phân bón hữu cơ ủ (compost) hoặc bỏ vào thùng rác hữu cơ dễ phân hủy sinh học. (Lưu ý: Quy định thu gom rác hữu cơ có thể khác nhau tùy địa phương).",
        "tips": [
            "Tách riêng khỏi bao bì ni lông và rác vô cơ",
            "Gạn bớt nước thừa để tránh mùi hôi",
            "Ủ phân compost tại nhà nếu có điều kiện",
            "Bỏ vào thùng rác hữu cơ màu xanh lá"
        ]
    },
    "other": {
        "label_vi": "Rác khác / Rác hỗn hợp (Other)",
        "bin": "Thùng rác không thể tái chế / rác sinh hoạt (Màu xám/cam)",
        "bin_color": "#64748b",
        "guidance": "Nhóm 'Rác khác' là nhóm rác tồn dư tổng hợp gồm các vật liệu hỗn hợp, khó phân loại, cao su, vải vụn hoặc vật phẩm không thuộc 5 nhóm rác chuẩn của mô hình. Cần kiểm tra kỹ quy tắc phân loại rác tại địa phương hoặc phân loại thủ công. Lưu ý: Pin, bóng đèn, rác điện tử phải đưa về điểm thu gom rác nguy hại riêng.",
        "tips": [
            "Kiểm tra quy định phân loại rác tại địa phương",
            "Vật liệu hỗn hợp hoặc không chắc chắn cần phân loại thủ công",
            "Pin, thiết bị điện tử, bóng đèn phải chuyển đến điểm thu gom nguy hại riêng",
            "Gói kín và bỏ vào thùng rác sinh hoạt không tái chế"
        ]
    }
}

# Pydantic Schemas
class RecommendationInfo(BaseModel):
    label_vi: str
    bin: str
    bin_color: str
    guidance: str
    tips: List[str]

class PredictionResponse(BaseModel):
    success: bool = True
    mock: bool = False
    predicted_class: str
    label_vi: str
    confidence: float
    probabilities: Dict[str, float]
    recommendation: RecommendationInfo
    filename: str
    content_type: str
    image_size_bytes: int
    image_dimensions: Dict[str, int]
    image_format: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    ai_model_loaded: bool
    device: str
    model_architecture: str
    checkpoint_loaded: bool
    target_classes: List[str]
    error: Optional[str] = None


def find_checkpoint_path() -> Path:
    """Tìm đường dẫn tệp checkpoint từ project root hoặc thư mục làm việc."""
    # 1. Thử từ file location: backend/app/main.py -> parent x 3 = root
    root_candidate = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "mobilenetv3_best.pth"
    if root_candidate.exists():
        return root_candidate
    
    # 2. Thử từ current working directory
    cwd_candidate = Path.cwd() / "checkpoints" / "mobilenetv3_best.pth"
    if cwd_candidate.exists():
        return cwd_candidate

    return root_candidate


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Nạp trọng số MobileNetV3-Large từ checkpoint checkpoint đã huấn luyện."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    target_classes = checkpoint.get("target_classes", ["plastic", "paper", "metal", "glass", "organic", "other"])

    # Khởi tạo kiến trúc MobileNetV3-Large
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(target_classes))

    # Nạp state_dict
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    app.state.device = str(device)
    app.state.model = None
    app.state.classes = ["plastic", "paper", "metal", "glass", "organic", "other"]
    app.state.model_loaded = False
    app.state.model_error = None

    # Khởi tạo transforms tiền xử lý ảnh (giống tập validation/test trong quá trình train)
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

        # Warmup model 1 lần
        dummy_input = torch.zeros(1, 3, 224, 224, device=device)
        with torch.inference_mode():
            _ = model(dummy_input)
        logger.info("AI model inference engine warmed up and ready.")
    except Exception as e:
        logger.error(f"Failed to load AI model checkpoint from {ckpt_path}: {e}", exc_info=True)
        app.state.model_loaded = False
        app.state.model_error = str(e)

    yield

    # Cleanup nếu cần
    logger.info("Shutting down FastAPI application...")
    if app.state.model is not None:
        del app.state.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="AI-Based Waste Recognition & Recycling Recommendation API",
    description="Backend API for waste classification and recycling guidance using fine-tuned MobileNetV3-Large.",
    version="0.2.0",
    lifespan=lifespan
)

# CORS middleware
ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse, summary="Service & Model Health Check")
async def health_check():
    """
    Kiểm tra tình trạng hoạt động của backend service và trạng thái nạp mô hình AI.
    """
    is_loaded = getattr(app.state, "model_loaded", False)
    status_str = "ok" if is_loaded else "degraded"
    
    return HealthResponse(
        status=status_str,
        service="waste-recognition-backend",
        version="0.2.0",
        ai_model_loaded=is_loaded,
        device=getattr(app.state, "device", "unknown"),
        model_architecture="mobilenet_v3_large",
        checkpoint_loaded=is_loaded,
        target_classes=getattr(app.state, "classes", []),
        error=getattr(app.state, "model_error", None)
    )


@app.post("/api/predict", response_model=PredictionResponse, summary="Classify Waste Image & Recommend Disposal")
async def predict_waste(file: UploadFile = File(...)):
    """
    Nhận diện rác thải từ hình ảnh tải lên và đưa ra khuyến nghị tái chế/xử lý rác.
    
    Yêu cầu và xác thực:
    - Kiểm tra tính tồn tại của tệp tải lên.
    - Giới hạn kích thước tối đa 10 MB (HTTP 413).
    - Xác thực định dạng ảnh hợp lệ bằng Pillow (JPEG, PNG, WebP) (HTTP 400).
    - Chạy suy luận MobileNetV3-Large với `torch.inference_mode()`.
    - Tính xác suất Softmax cho tất cả 6 lớp rác thải (tổng xấp xỉ 1.0).
    - Đính kèm khuyến nghị phân loại rác bằng tiếng Việt.
    - Không lưu trữ vĩnh viễn ảnh của người dùng trên ổ đĩa.
    """
    # 1. Kiểm tra mô hình đã nạp
    if not getattr(app.state, "model_loaded", False) or app.state.model is None:
        logger.error("Predict endpoint called but AI model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI inference engine is unavailable or failed to load. Please check server logs."
        )

    # 2. Kiểm tra tệp đính kèm
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided in the upload request."
        )

    # 3. Đọc dữ liệu vào bộ nhớ và kiểm tra dung lượng
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error reading upload stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded file stream."
        )

    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes)."
        )

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # 4. Xác thực tính toàn vẹn và định dạng ảnh bằng Pillow
    try:
        image_stream = io.BytesIO(contents)
        pil_image = Image.open(image_stream)
        pil_image.verify()  # Kiểm tra tính hợp lệ của header ảnh
        img_format = pil_image.format
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f"Uploaded file '{file.filename}' is not a valid image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid or readable image."
        )

    # Kiểm tra định dạng được hỗ trợ
    if not img_format or img_format.upper() not in ALLOWED_FORMATS:
        logger.warning(f"Unsupported image format '{img_format}' for file '{file.filename}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{img_format}'. Supported formats: JPEG, PNG, WebP."
        )

    # Mở lại để lấy kích thước và chuyển đổi màu sắc (verify() đóng hoặc hủy stream)
    try:
        image_stream.seek(0)
        with Image.open(image_stream) as img:
            img_dimensions = {"width": img.width, "height": img.height}
            rgb_image = img.convert("RGB")
            # 5. Tiền xử lý ảnh theo pipeline chuẩn
            tensor_input = app.state.transform(rgb_image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error processing image data for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode or process image content."
        )

    # 6. Chạy suy luận mô hình (Inference Mode, không tính Gradient)
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
            detail="An error occurred during AI model inference."
        )

    # 7. Tính xác suất và lớp dự đoán cao nhất
    probs_list = probs.cpu().tolist()
    pred_idx = int(torch.argmax(probs).item())
    predicted_class = classes[pred_idx]
    confidence = float(probs_list[pred_idx])

    # Tạo từ điển xác suất theo từng lớp (đảm bảo tổng xấp xỉ 1.0 và hữu hạn)
    probabilities: Dict[str, float] = {}
    for i, cls_name in enumerate(classes):
        val = float(probs_list[i])
        probabilities[cls_name] = round(val, 6)

    # 8. Gợi ý xử lý rác thải tiếng Việt
    rec_data = RECOMMENDATIONS.get(predicted_class, RECOMMENDATIONS["other"])
    recommendation = RecommendationInfo(
        label_vi=rec_data["label_vi"],
        bin=rec_data["bin"],
        bin_color=rec_data["bin_color"],
        guidance=rec_data["guidance"],
        tips=rec_data["tips"]
    )

    logger.info(
        f"Prediction complete for '{file.filename}': class='{predicted_class}', confidence={confidence:.4f}, size={file_size}B"
    )

    return PredictionResponse(
        success=True,
        mock=False,
        predicted_class=predicted_class,
        label_vi=rec_data["label_vi"],
        confidence=round(confidence, 4),
        probabilities=probabilities,
        recommendation=recommendation,
        filename=file.filename,
        content_type=file.content_type or f"image/{img_format.lower()}",
        image_size_bytes=file_size,
        image_dimensions=img_dimensions,
        image_format=img_format
    )
