"""Experimental close-up classifier. Scores never establish absence of hazards."""
import hashlib
import json
import os
import importlib.util
from functools import lru_cache
from pathlib import Path
from threading import Lock

from PIL import ImageStat
from .image_input import decode_image

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'logs/ultralytics_config'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'logs/matplotlib'))
RUN = ROOT / "runs/battery_screen/yolo26n_cls_multisource_v2"
INFERENCE_LOCK = Lock()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_loaded_artifact_key = None
_failed_artifact_key = None


def artifact_key(weights, report):
    return (str(weights), str(report), weights.stat().st_mtime_ns, report.stat().st_mtime_ns)


def artifact_paths():
    run = Path(os.getenv("BATTERY_SCREEN_RUN", str(RUN))).resolve()
    return run / "weights/best.pt", run / "battery_report.json"


def capabilities():
    weights, report = artifact_paths()
    present = weights.is_file() and report.is_file()
    verified = False
    key = None
    state, reason = 'unavailable', 'Thiếu artifact model pin hoặc báo cáo đánh giá.'
    metadata = {}
    try:
        if present:
            key = artifact_key(weights, report)
            metadata = json.loads(report.read_text(encoding='utf-8'))
            if not isinstance(metadata, dict):
                raise ValueError('Report must be an object')
            if metadata.get('runtime_enabled') is not True:
                state, reason = 'blocked', 'Model pin bị chặn theo kết quả đánh giá.'
            elif not 0.5 <= float(metadata['threshold']) <= 0.95:
                state, reason = 'invalid_artifact', 'Ngưỡng trong báo cáo không hợp lệ.'
            elif hashlib.sha256(weights.read_bytes()).hexdigest() != metadata.get('checkpoint_sha256'):
                state, reason = 'invalid_artifact', 'Hash checkpoint không khớp báo cáo đánh giá.'
            elif set(metadata.get('class_names', {}).values()) != {'battery', 'not_battery'}:
                state, reason = 'invalid_artifact', 'Nhãn checkpoint pin không đúng contract.'
            elif importlib.util.find_spec('ultralytics') is None:
                state, reason = 'unavailable', 'Chưa có thư viện suy luận pin trong môi trường này.'
            else:
                verified = True
                state = 'experimental_loaded' if key == _loaded_artifact_key else 'experimental_not_loaded'
                reason = 'Artifact pin hợp lệ cho suy luận thử nghiệm; chưa được kiểm định độc lập.'
                if key == _failed_artifact_key:
                    state, reason = 'load_failed', 'Lần nạp/suy luận model pin gần nhất thất bại.'
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        metadata = {}
        state, reason = 'invalid_artifact', 'Không đọc hoặc xác minh được artifact pin.'
    ready = verified and state in {'experimental_loaded', 'experimental_not_loaded'}
    return {
        "artifact_files_present": present,
        "artifact_verified": verified,
        "runtime_ready": ready,
        "production_ready": False,
        "experimental": True,
        "readiness_status": state,
        "readiness_reason": reason,
        "runtime_status": metadata.get('runtime_status', 'awaiting_evaluation'),
        "independent_external_validated": False,
        "task": "experimental_image_classification",
        "label_vi": "Sàng lọc hình dạng pin — thử nghiệm",
        "ai_model_loaded": bool(verified and key == _loaded_artifact_key and key != _failed_artifact_key),
        "limitations": [
            "Dùng ảnh cận cảnh một vật; chưa xác thực trên ảnh điện thoại tại Đà Nẵng.",
            "Không định vị pin, tìm pin ẩn, đo bức xạ hoặc xác định nhiễm bệnh.",
            "Score thấp không loại trừ pin hay chất thải nguy hại.",
        ],
    }


@lru_cache(maxsize=1)
def _load_model(weights_path, report_path, weights_mtime, report_mtime):
    # Metadata and weights must belong to the same measured experiment.
    weights = Path(weights_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get('runtime_enabled') is not True:
        raise RuntimeError('Model blocked after evaluation failure')
    if hashlib.sha256(weights.read_bytes()).hexdigest() != report.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint/report mismatch")
    threshold = float(report["threshold"])
    if not 0.5 <= threshold <= 0.95:
        raise RuntimeError("Invalid validation threshold")
    if set(report.get('class_names', {}).values()) != {'battery', 'not_battery'}:
        raise RuntimeError('Unexpected report class contract')
    from ultralytics import YOLO, settings
    settings.update({key: False for key in ('sync', 'wandb', 'comet', 'clearml', 'mlflow', 'dvc', 'tensorboard') if key in settings})
    model = YOLO(str(weights), task="classify")
    if model.task != "classify" or set(model.names.values()) != {"battery", "not_battery"}:
        raise RuntimeError("Unexpected model class contract")
    return model, threshold


def prepare_image(stream):
    """Validate real image content before allocation/inference, independent of filenames."""
    image, _ = decode_image(stream.read(MAX_UPLOAD_BYTES + 1))
    width, height = image.size
    if min(width, height) < 64:
        raise ValueError("Ảnh cần tối thiểu 64 px mỗi chiều.")
    brightness = ImageStat.Stat(image.convert("L")).mean[0]
    if brightness < 6 or brightness > 252:
        raise ValueError("Ảnh quá tối hoặc quá sáng; hãy chụp lại khi có thể làm an toàn.")
    return image, {"width": width, "height": height, "brightness": brightness,
                   "scope": "basic_input_checks", "is_acceptable": True}


def screen_image(image, quality):
    global _loaded_artifact_key, _failed_artifact_key
    weights, report = artifact_paths()
    if not weights.is_file() or not report.is_file():
        raise RuntimeError("Battery checkpoint/evaluation not available")
    with INFERENCE_LOCK:
        key = artifact_key(weights, report)
        try:
            model, threshold = _load_model(*key)
            prediction = model.predict(image, imgsz=224, device="cpu", verbose=False, save=False)[0]
            index = next(i for i, name in model.names.items() if name == "battery")
            score = float(prediction.probs.data[index])
        except Exception:
            _failed_artifact_key = key
            raise
        _loaded_artifact_key = key
        _failed_artifact_key = None
    positive = score >= threshold
    return {
        "battery_suspected": positive, "battery_score": score, "threshold": threshold,
        "score_is_calibrated_probability": False,
        "label_vi": "Nghi có pin" if positive else "Chưa đủ bằng chứng hình ảnh để nhận diện pin",
        "is_uncertain": True,
        "uncertainty_note": "Kết quả phát triển chưa được kiểm định độc lập ngoài nguồn dữ liệu; có thể bỏ sót pin, không xác nhận an toàn.",
        "quality_assessment": quality,
        "evidence_source": "image_classifier", "bounding_boxes": None,
        "confirmed_safe": False,
    }
