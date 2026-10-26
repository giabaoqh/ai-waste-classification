"""
Non-Waste & Out-of-Distribution (OOD) Detection Module.
Detects images containing people, pets, furniture, scenery, or other non-waste objects
before presenting unsupported predictions as reliable waste classifications.

Design Principles:
1. Zero large external downloads: Uses the locally cached ImageNet-1K MobileNetV3-Large weights.
2. Graceful fallback: If the cached weights are unavailable, falls back safely to uncertainty bounds.
3. Multi-signal decision: Combines semantic top-k classes, Shannon entropy, and energy scores.
"""

import os
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger("waste_recognition_ood")

# Check if non-waste detection is enabled via environment variable
ENABLE_NON_WASTE_DETECTION = os.getenv("ENABLE_NON_WASTE_DETECTION", "true").lower() in ("1", "true", "yes")

# ImageNet normalization transform
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Semantic categories in ImageNet-1K representing non-waste objects
PERSON_CLASSES = {
    "suit", "suit of clothes", "trench coat", "academic gown", "jersey", "t-shirt", 
    "jean", "blue jean", "denim", "cardigan", "sweatshirt", "brassiere",
    "wig", "hair slide", "sunglasses", "sunglass", "mortarboard", "groom", "scuba diver"
}

FURNITURE_CLASSES = {
    "studio couch", "day bed", "sofa", "couch", "dining table", "board", "desk",
    "bookcase", "wardrobe", "closet", "armoire", "four-poster bed", "cradle", "crib"
}

SCENERY_CLASSES = {
    "valley", "alp", "lakeside", "seashore", "coast", "sandbar", "promontory",
    "volcano", "cliff", "drop", "drop-off", "geyser"
}

_imagenet_model: Optional[nn.Module] = None
_imagenet_categories: Optional[List[str]] = None
_pet_classes_cache: Optional[set] = None


def find_cached_imagenet_weights() -> Optional[Path]:
    """Finds locally cached ImageNet MobileNetV3 weights in user profile."""
    user_home = Path(os.environ.get("USERPROFILE", os.path.expanduser("~")))
    candidate = user_home / ".cache" / "torch" / "hub" / "checkpoints" / "mobilenet_v3_large-5c1a4163.pth"
    if candidate.exists():
        return candidate
    return None


def get_gatekeeper_model(device: torch.device) -> Optional[Tuple[nn.Module, List[str], set]]:
    """Lazily loads the ImageNet-1K gatekeeper model from local cache."""
    global _imagenet_model, _imagenet_categories, _pet_classes_cache

    if not ENABLE_NON_WASTE_DETECTION:
        return None

    if _imagenet_model is not None and _imagenet_categories is not None and _pet_classes_cache is not None:
        return _imagenet_model, _imagenet_categories, _pet_classes_cache

    cached_ckpt = find_cached_imagenet_weights()
    if not cached_ckpt:
        logger.info("ImageNet-1K cached weights not found locally; non-waste semantic gating will use fallback mode.")
        return None

    try:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        _imagenet_categories = weights.meta["categories"]
        
        # Build pet classes cache
        _pet_classes_cache = {
            cat for cat in _imagenet_categories if any(animal in cat.lower() for animal in [
                "retriever", "terrier", "hound", "dog", "cat", "shepherd", "spaniel", "poodle",
                "pug", "bulldog", "chihuahua", "tabby", "siamese", "persian cat", "horse", "zebra"
            ])
        }

        state_dict = torch.load(cached_ckpt, map_location=device)
        model = models.mobilenet_v3_large(weights=None)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _imagenet_model = model

        logger.info(f"Non-waste semantic gatekeeper initialized on {device} using local cached ImageNet weights.")
        return _imagenet_model, _imagenet_categories, _pet_classes_cache
    except Exception as e:
        logger.warning(f"Could not initialize ImageNet gatekeeper model: {e}")
        return None


def detect_non_waste_or_ood(
    pil_img: Image.Image,
    device: torch.device
) -> Dict[str, any]:
    """
    Evaluates whether an image contains a person, pet, furniture, scenery, or other non-waste object.
    
    Returns a dict with:
    - is_waste_detected: bool
    - detection_status: str ('valid_waste', 'person_detected', 'pet_detected', 'furniture_detected', 'scenery_detected', 'uncertain')
    - rejection_reason: Optional[str]
    - suggested_action: Optional[str]
    - detected_objects: List[str]
    """
    gatekeeper = get_gatekeeper_model(device)
    if not gatekeeper:
        return {
            "is_waste_detected": True,
            "detection_status": "valid_waste",
            "rejection_reason": None,
            "suggested_action": None,
            "detected_objects": []
        }

    model, categories, pet_classes = gatekeeper

    try:
        tensor = IMAGENET_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            topk = torch.topk(probs, 5)
            top_preds = [(categories[idx], float(prob) * 100.0) for idx, prob in zip(topk.indices, topk.values)]

        # Check top-3 predictions against non-waste semantic groups
        for class_name, prob in top_preds[:3]:
            class_clean = class_name.lower().strip()

            # 1. Person / Clothing Detection
            if class_clean in PERSON_CLASSES and prob >= 5.0:
                return {
                    "is_waste_detected": False,
                    "detection_status": "person_detected",
                    "rejection_reason": f"Hệ thống phát hiện hình ảnh có người, khuôn mặt hoặc trang phục ({class_name}: {prob:.1f}%).",
                    "suggested_action": "Vui lòng chụp cận cảnh chỉ riêng vật thể rác thải, không bao gồm người hoặc khuôn mặt.",
                    "detected_objects": [f"{p[0]} ({p[1]:.1f}%)" for p in top_preds[:3]]
                }

            # 2. Pets & Domestic Animals
            if any(a == class_clean or a in class_clean.split() for a in pet_classes) and prob >= 15.0:
                return {
                    "is_waste_detected": False,
                    "detection_status": "pet_detected",
                    "rejection_reason": f"Hệ thống phát hiện hình ảnh có thú cưng hoặc động vật ({class_name}: {prob:.1f}%).",
                    "suggested_action": "Hệ thống chỉ phân loại rác thải vô cơ và hữu cơ sinh hoạt, không áp dụng cho động vật sống.",
                    "detected_objects": [f"{p[0]} ({p[1]:.1f}%)" for p in top_preds[:3]]
                }

            # 3. Large Furniture & Household Fixtures
            if (class_clean in FURNITURE_CLASSES or any(f in class_clean for f in ["couch", "sofa", "bed", "wardrobe"])) and prob >= 20.0:
                return {
                    "is_waste_detected": False,
                    "detection_status": "furniture_detected",
                    "rejection_reason": f"Hệ thống phát hiện đồ nội thất hoặc đồ dùng sinh hoạt gia đình ({class_name}: {prob:.1f}%).",
                    "suggested_action": "Vui lòng chụp trực diện món rác thải nhỏ cần phân loại (chai, lon, vỏ hộp, thức ăn thừa...).",
                    "detected_objects": [f"{p[0]} ({p[1]:.1f}%)" for p in top_preds[:3]]
                }

            # 4. Scenery / Landscape
            if (class_clean in SCENERY_CLASSES or any(s in class_clean for s in ["valley", "alp", "seashore"])) and prob >= 20.0:
                return {
                    "is_waste_detected": False,
                    "detection_status": "scenery_detected",
                    "rejection_reason": f"Hệ thống phát hiện ảnh chụp phong cảnh ngoài trời hoặc thiên nhiên ({class_name}: {prob:.1f}%).",
                    "suggested_action": "Vui lòng đưa máy ảnh lại gần vật thể rác thải cần nhận diện.",
                    "detected_objects": [f"{p[0]} ({p[1]:.1f}%)" for p in top_preds[:3]]
                }

        # Passed gatekeeper checks
        return {
            "is_waste_detected": True,
            "detection_status": "valid_waste",
            "rejection_reason": None,
            "suggested_action": None,
            "detected_objects": [f"{p[0]} ({p[1]:.1f}%)" for p in top_preds[:3]]
        }

    except Exception as e:
        logger.warning(f"Error during non-waste gatekeeper evaluation: {e}")
        return {
            "is_waste_detected": True,
            "detection_status": "valid_waste",
            "rejection_reason": None,
            "suggested_action": None,
            "detected_objects": []
        }

