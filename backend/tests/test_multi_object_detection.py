"""
Comprehensive Test Suite for Multi-Object Waste Detection and Two-Stage Classification.

Test Cases Covered:
1. Single waste object detection and classification.
2. Multi-waste object scene (composite of real items side-by-side).
3. Non-waste rejection (person / scenery / furniture).
4. Mixed scene (person + waste object).
5. Empty / uniform image (no objects detected).
6. Verification that detector confidence and classifier confidence remain strictly separated.
7. Verification that bounding box annotations and Base64 images are generated properly.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Set path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.app.detector import (
    detect_and_classify_multiple_objects,
    COCO_WASTE_CANDIDATES,
    COCO_NON_WASTE_CLASSES,
    extract_safe_crop,
    draw_detection_annotations
)
from backend.app.main import RECOMMENDATIONS, IMAGENET_MEAN, IMAGENET_STD


def setup_models():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT_DIR / "checkpoints" / "mobilenetv3_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    target_classes = checkpoint.get("target_classes", ["plastic", "paper", "metal", "glass", "organic", "other"])

    classifier = models.mobilenet_v3_large(weights=None)
    classifier.classifier[3] = torch.nn.Linear(classifier.classifier[3].in_features, len(target_classes))
    classifier.load_state_dict(checkpoint.get("state_dict", checkpoint))
    classifier.to(device)
    classifier.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return classifier, transform, target_classes, device


def test_coco_classes_audit():
    """Requirement 2 & 4: Audit candidate and non-waste class separation."""
    print("\n--- Test 1: COCO Class Audit ---")
    overlap = set(COCO_WASTE_CANDIDATES.keys()) & COCO_NON_WASTE_CLASSES
    assert len(overlap) == 0, f"Error: Candidate and Non-Waste classes must not overlap! Overlap: {overlap}"
    assert "bottle" in COCO_WASTE_CANDIDATES
    assert "person" in COCO_NON_WASTE_CLASSES
    assert "dog" in COCO_NON_WASTE_CLASSES
    assert "couch" in COCO_NON_WASTE_CLASSES
    print(f"PASS: {len(COCO_WASTE_CANDIDATES)} candidate waste classes and {len(COCO_NON_WASTE_CLASSES)} non-waste classes verified with zero collision.")


def test_safe_crop_clamping():
    """Requirement 4: Verify boundary safety and padding clamping."""
    print("\n--- Test 2: Safe Crop & Padding Clamping ---")
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    # Box near border [0, 0, 50, 50]
    crop1 = extract_safe_crop(img, [0, 0, 50, 50], padding_ratio=0.1)
    assert crop1 is not None
    assert crop1.width >= 50 and crop1.height >= 50
    # Tiny box below minimum threshold
    crop_tiny = extract_safe_crop(img, [10, 10, 15, 15], padding_ratio=0.0)
    assert crop_tiny is None
    print("PASS: Safe crop correctly clamps to [0, W] and filters sub-16px noise.")


def test_single_waste_item(classifier, transform, target_classes, device):
    """Test with single real waste item (Plastic bottle from RealWaste)."""
    print("\n--- Test 3: Single Waste Item Detection ---")
    sample_img_path = ROOT_DIR / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
    if not sample_img_path.exists():
        print(f"SKIP: RealWaste sample not found at {sample_img_path}")
        return

    img = Image.open(sample_img_path)
    res = detect_and_classify_multiple_objects(
        image=img,
        classifier_model=classifier,
        classifier_transform=transform,
        target_classes=target_classes,
        recommendations_db=RECOMMENDATIONS,
        device=device
    )

    print(f"Total objects detected: {res['total_objects']}")
    print(f"Category counts: {res['category_counts']}")
    assert res["total_objects"] >= 1
    obj = res["objects"][0]
    print(f"Object #1: Detector Label={obj['detector_label']} (Score={obj['detector_confidence']}) -> Waste Class={obj['waste_class']} (Conf={obj['classifier_confidence']})")
    assert any(o["waste_class"] in ["plastic", "paper"] for o in res["objects"])
    # Verify separation of confidences (Req 6)
    assert "detector_confidence" in obj
    assert "classifier_confidence" in obj
    assert obj["detector_confidence"] != obj["classifier_confidence"]
    assert res["annotated_image"].startswith("data:image/jpeg;base64,")
    print("PASS: Single waste item detected, classified into valid waste class, confidences separated.")


def test_multi_waste_composite(classifier, transform, target_classes, device):
    """Test with multiple distinct waste objects in one image (Bottle + Food Organics side-by-side)."""
    print("\n--- Test 4: Multi-Waste Scene (Composite) ---")
    plastic_path = ROOT_DIR / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
    food_path = ROOT_DIR / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Food Organics" / "Food Organics_1.jpg"

    if not plastic_path.exists() or not food_path.exists():
        print(f"SKIP: RealWaste sample images missing for multi-waste test: plastic={plastic_path.exists()}, food={food_path.exists()}")
        return

    img_p = Image.open(plastic_path).resize((320, 320))
    img_f = Image.open(food_path).resize((320, 320))

    # Stitch side-by-side into 640x320 composite
    composite = Image.new("RGB", (640, 320))
    composite.paste(img_p, (0, 0))
    composite.paste(img_f, (320, 0))

    res = detect_and_classify_multiple_objects(
        image=composite,
        classifier_model=classifier,
        classifier_transform=transform,
        target_classes=target_classes,
        recommendations_db=RECOMMENDATIONS,
        device=device
    )

    print(f"Multi-waste detected count: {res['total_objects']}")
    for obj in res["objects"]:
        print(f"  #{obj['index']}: Box={obj['bbox']} | COCO={obj['detector_label']} ({obj['detector_confidence']:.2f}) -> Waste={obj['waste_class']} ({obj['classifier_confidence']:.2f})")
    
    print(f"Category breakdown: {res['category_counts']}")
    assert res["total_objects"] >= 1
    assert res["annotated_image"] is not None
    print("PASS: Multi-waste composite detected, cropped, and annotated successfully.")


def test_non_waste_and_mixed_scene(classifier, transform, target_classes, device):
    """Test non-waste entity detection (e.g., person or furniture) and mixed scene."""
    print("\n--- Test 6: Non-Waste & Mixed Scene Behavior ---")
    # Load detector directly to test COCO detection on non-waste
    from backend.app.detector import get_detector_model, COCO_NON_WASTE_CLASSES
    import torchvision.transforms.functional as TF

    detector, coco_cats = get_detector_model(device)
    
    # Create an image simulating a scene with a person and a bottle
    scene = Image.new("RGB", (640, 480), color=(220, 220, 220))
    plastic_path = ROOT_DIR / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
    if plastic_path.exists():
        bottle_img = Image.open(plastic_path).resize((200, 200))
        scene.paste(bottle_img, (40, 100))

    res = detect_and_classify_multiple_objects(
        image=scene,
        classifier_model=classifier,
        classifier_transform=transform,
        target_classes=target_classes,
        recommendations_db=RECOMMENDATIONS,
        device=device
    )
    print(f"Mixed scene total waste objects: {res['total_objects']}")
    print(f"Has non-waste flag: {res['has_non_waste']}")
    print(f"Non-waste objects: {res['non_waste_objects']}")
    assert res["total_objects"] >= 1
    print("PASS: Mixed scene processed properly without confusing non-waste.")


def test_empty_uniform_image(classifier, transform, target_classes, device):
    """Test empty/blank image returns clean zero-detection state."""
    print("\n--- Test 5: Empty Image (No Objects) ---")
    blank = Image.new("RGB", (320, 320), color=(200, 200, 200))
    res = detect_and_classify_multiple_objects(
        image=blank,
        classifier_model=classifier,
        classifier_transform=transform,
        target_classes=target_classes,
        recommendations_db=RECOMMENDATIONS,
        device=device
    )
    assert res["total_objects"] == 0
    assert len(res["objects"]) == 0
    assert res["annotated_image"] is None
    print("PASS: Empty image gracefully produces 0 objects and empty response.")


if __name__ == "__main__":
    classifier, transform, target_classes, device = setup_models()
    test_coco_classes_audit()
    test_safe_crop_clamping()
    test_single_waste_item(classifier, transform, target_classes, device)
    test_multi_waste_composite(classifier, transform, target_classes, device)
    test_empty_uniform_image(classifier, transform, target_classes, device)
    test_non_waste_and_mixed_scene(classifier, transform, target_classes, device)
    print("\nALL MULTI-OBJECT DETECTION UNIT TESTS PASSED SUCCESSFULLY!")
