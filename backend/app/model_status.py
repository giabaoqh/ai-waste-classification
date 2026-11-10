"""Capability-specific readiness, without loading models or downloading artifacts."""
from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel

from .battery_screen import capabilities as battery_capabilities
from .yolo_detector import capabilities as yolo_capabilities

CONTRACT_VERSION = "1.0"
CapabilityName = Literal["legacy_classifier", "yolo_detector", "battery_classifier"]
ROOT = Path(__file__).resolve().parents[2]


class ModelCapability(BaseModel):
    status: str
    model_loaded: bool
    runtime_ready: bool
    production_ready: bool = False
    experimental: bool
    integrated: bool
    artifact_files_present: bool
    artifact_verified: Optional[bool] = None
    reason: str


class ReadinessResponse(BaseModel):
    contract_version: str = CONTRACT_VERSION
    capability: CapabilityName
    ready: bool
    model: ModelCapability


def model_capabilities(state) -> Dict[str, ModelCapability]:
    legacy_loaded = bool(getattr(state, "model_loaded", False) and getattr(state, "model", None) is not None)
    yolo = yolo_capabilities()
    battery = battery_capabilities()
    return {
        "legacy_classifier": ModelCapability(
            status="legacy_loaded" if legacy_loaded else "legacy_unavailable",
            model_loaded=legacy_loaded, runtime_ready=legacy_loaded, experimental=False,
            integrated=True, artifact_files_present=(ROOT / "checkpoints/mobilenetv3_best.pth").is_file(),
            reason="Luồng legacy; không phải hướng phát triển mới." if legacy_loaded else "Model legacy chưa sẵn sàng; không khôi phục MobileNet trong hướng phát triển hiện tại.",
        ),
        "yolo_detector": ModelCapability(
            status=yolo['status'], model_loaded=yolo['model_loaded'], runtime_ready=yolo['runtime_ready'],
            experimental=True, integrated=True, artifact_files_present=yolo['artifact_files_present'],
            artifact_verified=yolo['artifact_verified'], reason=yolo['reason'],
        ),
        "battery_classifier": ModelCapability(
            status=battery["readiness_status"], model_loaded=battery["ai_model_loaded"],
            runtime_ready=battery["runtime_ready"], experimental=True, integrated=True,
            artifact_files_present=battery["artifact_files_present"], artifact_verified=battery["artifact_verified"],
            reason=battery["readiness_reason"],
        ),
    }
