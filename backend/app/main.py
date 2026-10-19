import io
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waste_recognition_backend")

# Upload size constraint: 10 MB maximum
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="AI-Based Waste Recognition & Recycling Recommendation API",
    description="Backend API for waste classification and recycling guidance. Currently in initial skeleton / mock phase.",
    version="0.1.0"
)

# Configure CORS strictly for local Vite development origin
ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health", summary="Service Health Check")
async def health_check():
    """
    Check backend service health.
    NOTE: Real AI model is NOT loaded in this phase.
    """
    return {
        "status": "ok",
        "service": "waste-recognition-backend",
        "version": "0.1.0",
        "ai_model_loaded": False,
        "note": "AI inference engine has not been connected yet."
    }


@app.post("/api/predict", summary="Mock Waste Recognition Prediction")
async def predict_waste(file: UploadFile = File(...)):
    """
    Mock inference endpoint accepting an uploaded waste image.
    
    Validations:
    - Verifies file presence and content.
    - Enforces maximum file size limit (10 MB).
    - Validates image integrity using PIL (without relying solely on extension/MIME).
    
    Returns:
    - Mock response clearly indicating that real AI inference has not yet been executed.
    - No permanent image files are stored on disk.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided in the upload request."
        )

    # Read file content into memory to check size and validate
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

    # Validate image bytes using Pillow
    try:
        image_stream = io.BytesIO(contents)
        with Image.open(image_stream) as img:
            img.verify()  # Verifies the image headers / integrity
            img_format = img.format
            img_size = img.size  # (width, height)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f"Uploaded file is not a valid image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid or readable image."
        )

    logger.info(
        f"Received valid image '{file.filename}': format={img_format}, size={img_size}, bytes={file_size}"
    )

    # Return explicit mock response (no fake predictions or confidence scores)
    return {
        "mock": True,
        "status": "success",
        "message": "Real AI inference has not been implemented yet.",
        "filename": file.filename,
        "content_type": file.content_type,
        "image_size_bytes": file_size,
        "image_dimensions": {
            "width": img_size[0],
            "height": img_size[1]
        },
        "image_format": img_format,
        "supported_categories": [
            "plastic",
            "paper",
            "metal",
            "glass",
            "organic",
            "other"
        ]
    }
