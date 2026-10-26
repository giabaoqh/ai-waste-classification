"""
Multi-Object Waste Detection and Two-Stage Classification Module.

Stage 1: Object Proposal & Localization using SSDLite320-MobileNetV3.
- Detects candidate objects and locations [x1, y1, x2, y2].
- Filters out non-waste entities (people, pets, furniture, vehicles).
- Applies class-agnostic Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.

Stage 2: Fine-Grained Waste Classification using MobileNetV3-Large (mobilenetv3_best.pth).
- Safely crops each detected bounding box with margin padding and image boundary clamping.
- Classifies each crop into the project's 6 canonical waste categories:
  plastic, paper, metal, glass, organic, other.
- Keeps detector confidence and classifier confidence strictly separated.

Stage 3: High-Contrast Visual Annotation.
- Renders red bounding boxes (#ef4444) with scalable stroke width.
- Generates index tags (#1, #2...) and dual-language labels.
- Encodes the annotated image to Base64 JPEG.
"""

import io
import os
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image, ImageDraw, ImageFont
from backend.app.knowledge_base import get_disposal_guidance
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.detection as detection_models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.ops as ops

logger = logging.getLogger("waste_recognition_detector")

# Default Confidence Thresholds
DEFAULT_DETECTOR_SCORE_THRESH = float(os.getenv("DETECTOR_SCORE_THRESH", "0.25"))
DEFAULT_NMS_IOU_THRESH = float(os.getenv("DETECTOR_NMS_IOU_THRESH", "0.45"))
DEFAULT_CLASSIFIER_CONF_THRESH = float(os.getenv("CLASSIFIER_CONF_THRESH", "0.40"))
CROP_PADDING_RATIO = float(os.getenv("CROP_PADDING_RATIO", "0.06"))
MIN_CROP_DIMENSION = int(os.getenv("MIN_CROP_DIMENSION", "16"))

# Candidate COCO categories that represent potential recyclable or disposable waste
COCO_WASTE_CANDIDATES = {
    # Beverage & Food Containers
    "bottle": "Chai / Lọ",
    "wine glass": "Ly / Cốc thủy tinh",
    "cup": "Cốc / Ly",
    "bowl": "Tô / Bát",
    "fork": "Nĩa",
    "knife": "Dao",
    "spoon": "Muỗng / Thìa",
    # Organic / Food Waste
    "banana": "Chuối / Vỏ chuối",
    "apple": "Táo",
    "sandwich": "Bánh mì / Thức ăn thừa",
    "orange": "Cam / Vỏ cam",
    "broccoli": "Rau súp lơ",
    "carrot": "Cà rốt",
    "hot dog": "Xúc xích / Thức ăn",
    "pizza": "Bánh pizza",
    "donut": "Bánh ngọt",
    "cake": "Bánh kem",
    # Paper & Packaging
    "book": "Sách / Giấy / Báo",
    # Other potential waste / recyclables
    "scissors": "Kéo kim loại",
    "vase": "Bình / Lọ hoa",
    "toothbrush": "Bàn chải",
    "cell phone": "Điện thoại cũ (Rác điện tử)",
    "mouse": "Chuột máy tính cũ (Rác điện tử)",
    "remote": "Điều khiển từ xa (Rác điện tử)",
    "keyboard": "Bàn phím cũ (Rác điện tử)",
    "backpack": "Ba lô cũ (Vải)",
    "handbag": "Túi xách cũ",
    "suitcase": "Vali cũ",
    "boat": "Khung / Vỏ kim loại phế liệu"
}

# Explicit Non-Waste COCO categories (suppressed from waste classification)
COCO_NON_WASTE_CLASSES = {
    # People
    "person",
    # Animals / Pets
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # Vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    # Furniture & Fixtures
    "chair", "couch", "bed", "dining table", "toilet", "sink", "refrigerator", "oven", "toaster", "microwave",
    # Outdoor / Infrastructure
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
}

_detector_model: Optional[nn.Module] = None
_coco_categories: Optional[List[str]] = None


def get_detector_model(device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Lazily loads the SSDLite320-MobileNetV3 detector model using official weights."""
    global _detector_model, _coco_categories
    if _detector_model is not None and _coco_categories is not None:
        return _detector_model, _coco_categories

    logger.info("Initializing SSDLite320-MobileNetV3 object detector...")
    weights = detection_models.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    _coco_categories = weights.meta["categories"]
    
    # Load model
    model = detection_models.ssdlite320_mobilenet_v3_large(weights=weights)
    model.to(device)
    model.eval()
    _detector_model = model
    logger.info(f"SSDLite320 detector loaded successfully on device: {device}")
    return _detector_model, _coco_categories


def extract_safe_crop(
    image: Image.Image,
    box: List[float],
    padding_ratio: float = CROP_PADDING_RATIO
) -> Optional[Image.Image]:
    """
    Safely crops an image within bounding box coordinates [x1, y1, x2, y2],
    applying safety padding and clamping to image boundaries.
    """
    width, height = image.size
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    if w < MIN_CROP_DIMENSION or h < MIN_CROP_DIMENSION:
        return None

    # Apply padding
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio

    px1 = max(0, int(round(x1 - pad_w)))
    py1 = max(0, int(round(y1 - pad_h)))
    px2 = min(width, int(round(x2 + pad_w)))
    py2 = min(height, int(round(y2 + pad_h)))

    if (px2 - px1) < MIN_CROP_DIMENSION or (py2 - py1) < MIN_CROP_DIMENSION:
        return None

    return image.crop((px1, py1, px2, py2))


def draw_detection_annotations(
    image: Image.Image,
    detected_objects: List[Dict[str, Any]],
    box_color: str = "#ef4444"
) -> str:
    """
    Draws high-contrast red bounding boxes and labeled badges on the image.
    Returns Base64 encoded JPEG data URI.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Dynamic line width based on image dimensions
    line_width = max(3, int(min(w, h) / 180))
    font_size = max(13, int(min(w, h) / 36))

    # Try loading a system TrueType font, otherwise fallback to default
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for item in detected_objects:
        box = item["bbox"]
        idx = item["index"]
        waste_label_vi = item.get("label_vi", item.get("waste_class", "Rác"))
        class_conf = item["classifier_confidence"]
        
        x1, y1, x2, y2 = box

        # 1. Draw outer high-contrast red bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        # 2. Text badge string: "#1 Nhựa | 98%"
        badge_text = f"#{idx} {waste_label_vi} ({class_conf * 100:.0f}%)"

        # Calculate text bounding box
        try:
            bbox_text = draw.textbbox((x1, y1), badge_text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]
        except Exception:
            text_w = len(badge_text) * (font_size * 0.6)
            text_h = font_size

        pad = 4
        # Determine badge placement (prefer above the box if room, else inside)
        if y1 - (text_h + pad * 2) >= 0:
            bg_y1 = y1 - (text_h + pad * 2)
            bg_y2 = y1
        else:
            bg_y1 = y1
            bg_y2 = y1 + text_h + pad * 2

        bg_x1 = max(0, x1)
        bg_x2 = min(w, bg_x1 + text_w + pad * 2)

        # Draw red badge background
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=box_color)

        # Draw badge text in white
        draw.text((bg_x1 + pad, bg_y1 + pad), badge_text, fill="#ffffff", font=font)

    # Export to Base64 JPEG
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def detect_and_classify_multiple_objects(
    image: Image.Image,
    classifier_model: nn.Module,
    classifier_transform: transforms.Compose,
    target_classes: List[str],
    recommendations_db: Dict[str, Dict[str, Any]],
    device: torch.device,
    detector_score_thresh: float = DEFAULT_DETECTOR_SCORE_THRESH,
    nms_iou_thresh: float = DEFAULT_NMS_IOU_THRESH,
    classifier_conf_thresh: float = DEFAULT_CLASSIFIER_CONF_THRESH
) -> Dict[str, Any]:
    """
    Main pipeline for multi-object detection and two-stage classification:
    1. Detects objects using SSDLite320-MobileNetV3.
    2. Identifies non-waste objects (people, animals, furniture).
    3. Performs class-agnostic NMS to eliminate overlapping proposals.
    4. Crops candidate waste boxes with padding & clamping.
    5. Feeds crops into MobileNetV3-Large classifier.
    6. Annotates original image and aggregates category statistics.
    """
    img_rgb = image.convert("RGB")
    width, height = img_rgb.size

    # Load detector
    detector, coco_cats = get_detector_model(device)

    # Prepare input tensor for detector
    img_tensor = TF.to_tensor(img_rgb).to(device)

    with torch.inference_mode():
        detections = detector([img_tensor])[0]

    raw_boxes = detections["boxes"]
    raw_scores = detections["scores"]
    raw_labels = detections["labels"]

    # Filter by detector score threshold
    score_mask = raw_scores >= detector_score_thresh
    boxes = raw_boxes[score_mask]
    scores = raw_scores[score_mask]
    labels = raw_labels[score_mask]

    detected_non_waste: List[Dict[str, Any]] = []
    candidate_boxes_list: List[torch.Tensor] = []
    candidate_scores_list: List[torch.Tensor] = []
    candidate_labels_list: List[str] = []

    for box, score, label_idx in zip(boxes, scores, labels):
        cat_name = coco_cats[label_idx.item()]
        score_val = float(score.item())

        if cat_name in COCO_NON_WASTE_CLASSES:
            detected_non_waste.append({
                "class_name": cat_name,
                "detector_confidence": round(score_val, 4),
                "bbox": [round(float(c), 1) for c in box.tolist()]
            })
        elif cat_name in COCO_WASTE_CANDIDATES:
            candidate_boxes_list.append(box)
            candidate_scores_list.append(score)
            candidate_labels_list.append(cat_name)

    # Check if real non-waste entities like person or pets are present
    has_person_or_animal = any(
        obj["class_name"] in {"person", "cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
        for obj in detected_non_waste
    )

    # If no candidate waste proposals found
    if not candidate_boxes_list:
        # Check image texture/variance to avoid triggering on blank/uniform images
        img_variance = float(np.std(np.array(img_rgb)))
        if not has_person_or_animal and img_variance > 25.0 and len(boxes) > 0:
            # Fallback: check if the overall scene is bulk/scrapyard waste
            try:
                full_tensor = classifier_transform(img_rgb).unsqueeze(0).to(device)
                with torch.inference_mode():
                    full_logits = classifier_model(full_tensor)
                    full_probs = torch.softmax(full_logits, dim=1)[0].cpu().numpy()
                
                scene_pred_idx = int(full_probs.argmax())
                scene_class = target_classes[scene_pred_idx]
                scene_conf = float(full_probs[scene_pred_idx])
                
                if scene_conf >= 0.40 and scene_class in target_classes:
                    scene_box = [10.0, 10.0, float(width - 10), float(height - 10)]
                    scene_label_vi = f"{scene_class.capitalize()} (Bãi/Đống phế liệu tổng thể)"
                    if scene_class == "metal":
                        scene_label_vi = "Kim loại (Bãi phế liệu sắt thép)"
                    elif scene_class == "other":
                        scene_label_vi = "Rác tổng hợp / Bãi phế liệu hỗn hợp"
                    elif scene_class == "plastic":
                        scene_label_vi = "Nhựa (Cụm rác thải nhựa quy mô lớn)"
                    
                    recom = get_disposal_guidance(scene_class, detected_object="Bãi phế liệu quy mô lớn")
                    
                    scene_object = {
                        "index": 1,
                        "bbox": scene_box,
                        "bbox_normalized": [
                            round(scene_box[0] / width, 4),
                            round(scene_box[1] / height, 4),
                            round(scene_box[2] / width, 4),
                            round(scene_box[3] / height, 4)
                        ],
                        "detector_label": "bulk_waste_scene",
                        "detector_label_vi": "Cảnh bãi rác / Phế liệu tổng thể",
                        "detector_confidence": round(scene_conf, 4),
                        "waste_class": scene_class,
                        "label_vi": scene_label_vi,
                        "classifier_confidence": round(scene_conf, 4),
                        "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(target_classes, full_probs)},
                        "confidence_source": "scene_level_classifier_fallback",
                        "recommendation": recom
                    }
                    
                    annotated_base64 = draw_detection_annotations(img_rgb, [scene_object])
                    cat_counts = {c: 0 for c in target_classes}
                    cat_counts[scene_class] = 1
                    
                    return {
                        "total_objects": 1,
                        "objects": [scene_object],
                        "category_counts": cat_counts,
                        "has_non_waste": False,
                        "non_waste_objects": [],
                        "annotated_image": annotated_base64,
                        "image_dimensions": {"width": width, "height": height},
                        "scene_fallback_applied": True,
                        "scene_note": "Phát hiện cụm phế liệu / bãi rác quy mô lớn (Bulk Scrapyard Scene)."
                    }
            except Exception as e:
                logger.warning(f"Scene-level fallback failed: {e}")

        return {
            "total_objects": 0,
            "objects": [],
            "category_counts": {c: 0 for c in target_classes},
            "has_non_waste": len(detected_non_waste) > 0,
            "non_waste_objects": detected_non_waste,
            "annotated_image": None,
            "image_dimensions": {"width": width, "height": height}
        }

    # Stack candidate tensors
    t_boxes = torch.stack(candidate_boxes_list)
    t_scores = torch.stack(candidate_scores_list)

    # Non-Maximum Suppression (NMS) to eliminate duplicate bounding boxes
    keep_indices = ops.nms(t_boxes, t_scores, nms_iou_thresh)

    final_objects: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {c: 0 for c in target_classes}

    idx = 1
    for k_idx in keep_indices:
        i = k_idx.item()
        box = t_boxes[i].cpu().numpy().tolist()
        det_score = float(t_scores[i].item())
        coco_label = candidate_labels_list[i]
        coco_label_vi = COCO_WASTE_CANDIDATES.get(coco_label, coco_label)

        # 1. Extract crop
        crop = extract_safe_crop(img_rgb, box, padding_ratio=CROP_PADDING_RATIO)
        if crop is None:
            continue

        # 2. Classify crop with fine-tuned MobileNetV3-Large
        crop_tensor = classifier_transform(crop).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = classifier_model(crop_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        waste_class = target_classes[pred_idx]
        class_conf = float(probs[pred_idx])

        # Filter out crops where classifier confidence is unreliably low
        if class_conf < classifier_conf_thresh:
            continue

        # Retrieve verified guidance from Knowledge Base with object detection cues
        recom = get_disposal_guidance(
            waste_class=waste_class,
            detected_object=coco_label,
            is_uncertain=(class_conf < 0.65)
        )
        label_vi = recom.get("label_vi", waste_class.capitalize())

        # Build detected item record
        obj_item = {
            "index": idx,
            "bbox": [round(float(c), 1) for c in box],
            "bbox_normalized": [
                round(box[0] / width, 4),
                round(box[1] / height, 4),
                round(box[2] / width, 4),
                round(box[3] / height, 4)
            ],
            "detector_label": coco_label,
            "detector_label_vi": coco_label_vi,
            "detector_confidence": round(det_score, 4),
            "waste_class": waste_class,
            "label_vi": label_vi,
            "classifier_confidence": round(class_conf, 4),
            "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(target_classes, probs)},
            "recommendation": {
                "bin": recom.get("bin", "Thùng rác phân loại"),
                "bin_color": recom.get("bin_color", "#10b981"),
                "special_destination": recom.get("special_destination"),
                "guidance": recom.get("guidance", ""),
                "preparation_requirements": recom.get("preparation_requirements", []),
                "tips": recom.get("tips", [])[:2],
                "source": recom.get("source"),
                "last_verified": recom.get("last_verified")
            }
        }

        final_objects.append(obj_item)
        category_counts[waste_class] = category_counts.get(waste_class, 0) + 1
        idx += 1

    # Render annotations
    annotated_uri = None
    if final_objects:
        annotated_uri = draw_detection_annotations(img_rgb, final_objects, box_color="#ef4444")

    return {
        "total_objects": len(final_objects),
        "objects": final_objects,
        "category_counts": category_counts,
        "has_non_waste": len(detected_non_waste) > 0,
        "non_waste_objects": detected_non_waste,
        "annotated_image": annotated_uri,
        "image_dimensions": {"width": width, "height": height}
    }

