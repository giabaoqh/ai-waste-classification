"""Pinned, local-only YOLO object-group inference; no material/hazard classification."""
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'runs/waste_pilot/yolo26n_multisource30_v1'
MANIFEST_PATH = Path(__file__).with_name('yolo_detector_manifest.json')
MAX_DETECTIONS = 100
INFERENCE_LOCK = Lock()
_loaded_model = None
_loaded_key = None
_failure = None


class ClassMapping(BaseModel):
    class_id: int
    class_name: str
    label_vi: str


class ModelInformation(BaseModel):
    model_id: str
    architecture: str
    model_version: str
    task: Literal['detect'] = 'detect'
    taxonomy: Literal['object_groups'] = 'object_groups'
    checkpoint_sha256: str
    training_ultralytics_version: str
    runtime_ultralytics_version: str
    input_size: int
    class_mapping: list[ClassMapping]


class Detection(BaseModel):
    class_id: int
    class_name: str
    label_vi: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: list[float] = Field(min_length=4, max_length=4)
    is_uncertain: bool
    uncertainty_note: Optional[str] = None


class DetectionResponse(BaseModel):
    contract_version: str = '1.0'
    success: bool = True
    inference_status: Literal['completed'] = 'completed'
    detection_status: Literal['detections_found', 'no_detections_above_threshold']
    detections: list[Detection]
    total_objects: int
    model: ModelInformation
    image_width: int
    image_height: int
    bounding_box_format: Literal['xyxy_pixels'] = 'xyxy_pixels'
    coordinate_space: Literal['exif_oriented_image'] = 'exif_oriented_image'
    confidence_threshold: float
    uncertainty_threshold: float
    confidence_is_calibrated_probability: bool = False
    device: Literal['cpu'] = 'cpu'
    processing_time_ms: float
    experimental: bool = True
    production_ready: bool = False
    is_uncertain: bool = True
    uncertainty_note: str = 'Model thử nghiệm chưa được kiểm định độc lập tại Đà Nẵng; cần người kiểm tra kết quả.'
    hazard_assessment: Literal['not_assessed'] = 'not_assessed'
    confirmed_safe: bool = False
    quality_assessment: dict
    label_vi: str = 'Nhận diện nhóm vật thể rác — thử nghiệm'
    guidance: str = 'Chỉ gợi ý nhóm vật thể, không xác định vật liệu, khả năng tái chế, pin, rác y tế hay phóng xạ.'
    tips: list[str] = Field(default_factory=lambda: [
        'Không có detection không chứng minh ảnh không có rác hoặc không có vật nguy hại.',
        'Không dùng nhãn other_litter để quyết định bỏ vào thùng rác thông thường.',
    ])


class YoloCapabilities(BaseModel):
    status: str
    model_loaded: bool
    runtime_ready: bool
    production_ready: bool = False
    experimental: bool = True
    integrated: bool = True
    artifact_files_present: bool
    artifact_verified: bool
    reason: str
    confidence_threshold: Optional[float] = None
    uncertainty_threshold: Optional[float] = None
    model: Optional[ModelInformation] = None


class YoloErrorDetail(BaseModel):
    code: str
    status: str
    capability: Literal['yolo_detector'] = 'yolo_detector'
    inference_status: Literal['not_started', 'failed']
    message: str


class YoloErrorResponse(BaseModel):
    detail: YoloErrorDetail


class YoloServiceError(Exception):
    def __init__(self, code, status, message, http_status=503):
        super().__init__(message)
        self.code, self.status, self.message, self.http_status = code, status, message, http_status


@dataclass(frozen=True)
class Artifact:
    weights: Path
    summary: Path
    manifest: dict
    key: tuple
    runtime_version: str


@dataclass(frozen=True)
class InferenceConfig:
    confidence: float
    uncertainty: float


def configuration():
    try:
        confidence = float(os.getenv('YOLO_CONFIDENCE_THRESHOLD', '0.25'))
        uncertainty = float(os.getenv('YOLO_UNCERTAINTY_THRESHOLD', '0.50'))
        if not (math.isfinite(confidence) and math.isfinite(uncertainty) and 0 < confidence <= uncertainty <= 1):
            raise ValueError('Expected 0 < confidence <= uncertainty <= 1')
        return InferenceConfig(confidence, uncertainty)
    except ValueError as exc:
        raise YoloServiceError('yolo_invalid_configuration', 'invalid_configuration',
                               'Ngưỡng YOLO không hợp lệ: cần 0 < confidence <= uncertainty <= 1.') from exc


def artifact_paths():
    run = Path(os.getenv('YOLO_DETECTOR_RUN', str(DEFAULT_RUN))).resolve()
    return run/'weights/best.pt', run/'pilot_summary.json', MANIFEST_PATH.resolve()


def validate_manifest(manifest):
    """No guessed mapping: the checked-in manifest records the verified artifact identity."""
    if not isinstance(manifest, dict):
        raise ValueError('Manifest must be an object')
    if manifest['manifest_version'] != 1 or manifest['task'] != 'detect' or manifest['taxonomy'] != 'object_groups':
        raise ValueError('Unsupported manifest contract')
    mapping = manifest['class_mapping']
    for row in mapping:
        ClassMapping.model_validate(row)
    if not isinstance(manifest['model_id'], str) or not manifest['model_id'] or not isinstance(manifest['training_ultralytics_version'], str):
        raise ValueError('Invalid model identity/version')
    if manifest['experimental'] is not True or manifest['production_ready'] is not False:
        raise ValueError('Only experimental manifest is supported')
    if len(mapping) != 9 or [r['class_id'] for r in mapping] != list(range(9)):
        raise ValueError('Class IDs must be ordered 0..8')
    if len({r['class_name'] for r in mapping}) != 9 or any(not r['class_name'] or not r['label_vi'] for r in mapping):
        raise ValueError('Invalid class names')
    for key in ['checkpoint_sha256', 'summary_sha256', 'dataset_yaml_sha256']:
        if len(manifest[key]) != 64 or any(c not in '0123456789abcdef' for c in manifest[key]):
            raise ValueError('Invalid artifact digest')
    if manifest['imgsz'] != 640 or manifest['architecture'] != 'YOLO26n':
        raise ValueError('Unsupported model architecture/input size')


def inspect_artifact():
    weights, summary_path, manifest_path = artifact_paths()
    if not all(p.is_file() for p in (weights, summary_path, manifest_path)):
        raise YoloServiceError('yolo_model_unavailable', 'unavailable', 'Thiếu checkpoint YOLO, metadata run hoặc manifest mapping.')
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        validate_manifest(manifest)
        if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint digest mismatch')
        summary_bytes = summary_path.read_bytes()
        if hashlib.sha256(summary_bytes).hexdigest() != manifest['summary_sha256']:
            raise ValueError('Run metadata digest mismatch')
        summary = json.loads(summary_bytes)
        names = [r['class_name'] for r in sorted(summary['per_class'], key=lambda r: r['class_id'])]
        if ([r['class_id'] for r in sorted(summary['per_class'], key=lambda r: r['class_id'])] != list(range(9))
                or names != [r['class_name'] for r in manifest['class_mapping']]
                or summary['best_checkpoint_sha256'] != manifest['checkpoint_sha256']
                or summary['run'] != manifest['model_id'] or summary['model'] != 'yolo26n.pt'
                or summary['dataset_yaml_sha256'] != manifest['dataset_yaml_sha256']
                or summary['packages']['ultralytics'] != manifest['training_ultralytics_version']
                or summary['imgsz'] != manifest['imgsz']):
            raise ValueError('Run/model/mapping contract mismatch')
        key = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in (weights, summary_path, manifest_path))
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        raise YoloServiceError('yolo_invalid_artifact', 'invalid_artifact', 'Checkpoint hoặc metadata/mapping YOLO không khớp manifest đã xác minh.') from exc
    try:
        if importlib.util.find_spec('ultralytics') is None:
            raise ImportError('ultralytics missing')
        runtime_version = importlib.metadata.version('ultralytics')
        if runtime_version != manifest['training_ultralytics_version']:
            raise ImportError('Ultralytics runtime version differs from verified version')
    except (ImportError, ValueError) as exc:
        raise YoloServiceError('yolo_dependency_unavailable', 'unavailable', 'Thiếu hoặc khác phiên bản thư viện YOLO đã xác minh; không tự cài/tải.') from exc
    return Artifact(weights, summary_path, manifest, key, runtime_version)


def model_information(artifact):
    m = artifact.manifest
    return ModelInformation(model_id=m['model_id'], architecture=m['architecture'],
                            model_version=m['model_id']+'@'+m['checkpoint_sha256'][:12],
                            checkpoint_sha256=m['checkpoint_sha256'],
                            training_ultralytics_version=m['training_ultralytics_version'],
                            runtime_ultralytics_version=artifact.runtime_version,
                            input_size=m['imgsz'], class_mapping=m['class_mapping'])


def capabilities():
    paths = artifact_paths()
    config = None
    try:
        config = configuration()
        artifact = inspect_artifact()
        loaded = _loaded_key == artifact.key and _loaded_model is not None
        state = 'experimental_loaded' if loaded else 'experimental_not_loaded'
        reason = 'YOLO nhóm vật thể đã tích hợp để thử nghiệm; chưa đủ điều kiện triển khai không giám sát.'
        if _failure is not None and _failure[0] == artifact.key:
            state, reason = _failure[1], _failure[2]
        return YoloCapabilities(status=state, model_loaded=loaded, runtime_ready=state in {'experimental_loaded', 'experimental_not_loaded'},
                                artifact_files_present=True, artifact_verified=True, reason=reason,
                                confidence_threshold=config.confidence, uncertainty_threshold=config.uncertainty,
                                model=model_information(artifact)).model_dump()
    except YoloServiceError as exc:
        return YoloCapabilities(status=exc.status, model_loaded=False, runtime_ready=False,
                                artifact_files_present=all(p.is_file() for p in paths), artifact_verified=False,
                                reason=exc.message, confidence_threshold=config.confidence if config else None,
                                uncertainty_threshold=config.uncertainty if config else None).model_dump()


def validate_checkpoint_model(model, manifest):
    expected = {r['class_id']: r['class_name'] for r in manifest['class_mapping']}
    try:
        valid = (model.task == 'detect' and model.names == expected and model.model.model[-1].nc == 9
                 and model.ckpt.get('version') == manifest['training_ultralytics_version'])
    except (AttributeError, IndexError, TypeError):
        valid = False
    if not valid:
        raise YoloServiceError('yolo_invalid_artifact', 'invalid_artifact', 'Nhãn/head/task hoặc phiên bản checkpoint YOLO khác metadata đã xác minh.')


def _load_model(artifact):
    # Existence/hash checks happen before the library constructor; never use an asset name fallback.
    if not artifact.weights.is_file():
        raise YoloServiceError('yolo_model_unavailable', 'unavailable', 'Checkpoint YOLO không còn tồn tại.')
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'logs/ultralytics_config'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'logs/matplotlib'))
    from ultralytics import YOLO, settings
    settings.update({k: False for k in ('sync', 'wandb', 'comet', 'clearml', 'mlflow', 'dvc', 'tensorboard') if k in settings})
    model = YOLO(str(artifact.weights), task='detect')
    validate_checkpoint_model(model, artifact.manifest)
    return model


def parse_detections(xyxy, classes, confidences, mapping, width, height, config):
    if not (len(xyxy) == len(classes) == len(confidences)) or len(xyxy) > MAX_DETECTIONS:
        raise ValueError('Invalid number of boxes/classes/scores')
    detections = []
    for box, class_value, score in zip(xyxy, classes, confidences):
        if not math.isfinite(float(class_value)) or not float(class_value).is_integer() or not 0 <= int(class_value) < len(mapping):
            raise ValueError('Unknown predicted class ID')
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Invalid box/score')
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid box geometry')
        clipped = [max(0., min(width, x1)), max(0., min(height, y1)), max(0., min(width, x2)), max(0., min(height, y2))]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError('Box outside image')
        if score < config.confidence:
            continue
        label = mapping[int(class_value)]
        uncertain = score < config.uncertainty
        detections.append(Detection(**label, confidence=score, bounding_box=clipped, is_uncertain=uncertain,
                                    uncertainty_note='Score thấp hơn ngưỡng cần người kiểm tra.' if uncertain else None))
    return detections


def detect(image, quality):
    global _loaded_model, _loaded_key, _failure
    started = time.perf_counter()
    config = configuration()
    artifact = inspect_artifact()
    with INFERENCE_LOCK:
        # Revalidate after waiting, so artifact switches cannot reuse a different cached model.
        artifact = inspect_artifact()
        if _loaded_key != artifact.key or _loaded_model is None:
            try:
                model = _load_model(artifact)
            except Exception as exc:
                failure = exc if isinstance(exc, YoloServiceError) else YoloServiceError('yolo_load_failed', 'load_failed', 'Không nạp được checkpoint YOLO để suy luận.')
                _failure = (artifact.key, failure.status, failure.message)
                if failure is exc:
                    raise
                raise failure from exc
            _loaded_model, _loaded_key = model, artifact.key
        try:
            results = _loaded_model.predict(image, imgsz=artifact.manifest['imgsz'], conf=config.confidence,
                                            max_det=MAX_DETECTIONS, device='cpu', verbose=False, save=False)
            if len(results) != 1 or results[0].boxes is None:
                raise ValueError('Expected one detection result')
            boxes = results[0].boxes
            detections = parse_detections(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                                          artifact.manifest['class_mapping'], image.width, image.height, config)
        except Exception as exc:
            _failure = (artifact.key, 'inference_error', 'Suy luận YOLO thất bại; không trả detection giả.')
            raise YoloServiceError('yolo_inference_error', 'inference_error', _failure[2], 500) from exc
        _failure = None
    response = DetectionResponse(detection_status='detections_found' if detections else 'no_detections_above_threshold',
                                 detections=detections, total_objects=len(detections), model=model_information(artifact),
                                 image_width=image.width, image_height=image.height,
                                 confidence_threshold=config.confidence, uncertainty_threshold=config.uncertainty,
                                 processing_time_ms=round((time.perf_counter()-started)*1000, 3), quality_assessment=quality)
    if not detections:
        response.uncertainty_note = 'Không có detection vượt ngưỡng; không kết luận ảnh không có rác, pin hoặc vật nguy hại.'
    return response
