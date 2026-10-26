"""
Waste Disposal & Recycling Knowledge Base Module.

Provides verified, decoupled disposal guidance, combustibility analysis,
and recyclability classification backed by official standards and legal frameworks
(Vietnam Environmental Protection Law 2020, Decree 08/2022/ND-CP, Circular 02/2022/TT-BTNMT, TCVN).

Key Design Principles:
1. Decoupled from classifier: Combustibility and recyclability are separate, non-conflated dimensions.
2. Safety-first override: Dangerous items (batteries, e-waste, pressurized aerosol cans) are flagged
   and directed to specialized hazardous collection points, NEVER ordinary metal/plastic bins.
3. Configurable & Versioned: Loaded from JSON with jurisdiction and verification metadata.
4. Conditional & Uncertainty Awareness: Reflects real-world conditions when items are contaminated or ambiguous.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("waste_knowledge_base")

KB_FILE_PATH = Path(__file__).resolve().parent / "waste_disposal_kb.json"

_KB_DATA: Optional[Dict[str, Any]] = None


def load_knowledge_base() -> Dict[str, Any]:
    """Loads and caches the disposal knowledge base from JSON."""
    global _KB_DATA
    if _KB_DATA is not None:
        return _KB_DATA

    if not KB_FILE_PATH.exists():
        logger.error(f"Knowledge base file not found at: {KB_FILE_PATH}")
        return {"categories": {}, "default_jurisdiction": "Việt Nam", "version": "1.0.0"}

    try:
        with open(KB_FILE_PATH, "r", encoding="utf-8") as f:
            _KB_DATA = json.load(f)
            logger.info(f"Loaded waste disposal knowledge base v{_KB_DATA.get('version')} with {len(_KB_DATA.get('categories', {}))} categories.")
    except Exception as e:
        logger.error(f"Failed to read knowledge base JSON: {e}", exc_info=True)
        _KB_DATA = {"categories": {}, "default_jurisdiction": "Việt Nam", "version": "1.0.0"}

    return _KB_DATA


def get_all_categories() -> Dict[str, Any]:
    """Returns all category definitions."""
    kb = load_knowledge_base()
    return kb.get("categories", {})


def detect_hazardous_or_special_override(
    waste_class: str,
    detected_object: Optional[str] = None,
    filename: Optional[str] = None
) -> Optional[str]:
    """
    Checks if an item requires safety override (e.g. Battery, E-waste, Composite)
    regardless of whether the visual classifier predicted metal, other, or plastic.
    """
    raw_str = " ".join(filter(None, [detected_object or "", filename or ""])).lower()
    # Normalize separators (underscores, hyphens, periods) into spaces
    check_str = raw_str.replace("_", " ").replace("-", " ").replace(".", " ")

    # 1. Batteries / Accumulators (CRITICAL HAZARD)
    battery_keywords = ["battery", "batteries", "pin", "accu", "ắc quy", "lithium", "alkaline"]
    if any(k in check_str for k in battery_keywords):
        return "battery"

    # 2. Electronic waste
    e_waste_keywords = ["cell phone", "laptop", "remote", "keyboard", "mouse", "điện thoại", "chuột", "bàn phím", "e-waste", "linh kiện"]
    if any(k in check_str for k in e_waste_keywords):
        return "e_waste"

    # 3. Composite cartons (Tetra Pak)
    carton_keywords = ["tetra pak", "vỏ hộp sữa", "hộp sữa", "milk carton"]
    if any(k in check_str for k in carton_keywords):
        return "composite_carton"

    return None


def get_disposal_guidance(
    waste_class: str,
    detected_object: Optional[str] = None,
    filename: Optional[str] = None,
    is_uncertain: bool = False,
    uncertainty_note: Optional[str] = None,
    jurisdiction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resolves comprehensive disposal, combustibility, and recyclability guidance.
    Applies safety overrides for hazardous items (e.g., batteries misclassified as metal).
    """
    categories = get_all_categories()
    kb_meta = load_knowledge_base()

    # Check for safety override
    override_key = detect_hazardous_or_special_override(waste_class, detected_object, filename)
    target_key = override_key if (override_key and override_key in categories) else waste_class

    entry = categories.get(target_key)
    if not entry:
        # Fallback to 'other' if key not found
        entry = categories.get("other", {
            "id": waste_class,
            "label_vi": waste_class.capitalize(),
            "bin": "Thùng rác sinh hoạt",
            "bin_color": "#64748b",
            "guidance": "Vui lòng phân loại theo quy định địa phương.",
            "tips": ["Bỏ vào thùng rác phù hợp."],
            "combustibility": {"is_combustible": False, "status": "Không xác định", "badge_color": "#64748b", "analysis": "Chưa có dữ liệu"},
            "recyclability": {"status": "unknown", "level": "Chưa xác định", "difficulty_score": "Trung bình", "badge_color": "#64748b", "analysis": "Cần kiểm tra thêm"}
        })

    # Deep copy / dictionary construction
    result = {
        "material_id": entry.get("id", target_key),
        "label_vi": entry.get("label_vi", waste_class),
        "bin": entry.get("bin", "Thùng rác phù hợp"),
        "bin_color": entry.get("bin_color", "#64748b"),
        "special_destination": entry.get("special_destination"),
        "guidance": entry.get("guidance", ""),
        "preparation_requirements": entry.get("preparation_requirements", []),
        "tips": list(entry.get("tips", [])),
        "combustibility": entry.get("combustibility"),
        "recyclability": entry.get("recyclability"),
        "processing_steps": entry.get("processing_steps"),
        "environmental_impact": entry.get("environmental_impact"),
        "mixed_material_notes": entry.get("mixed_material_notes"),
        "jurisdiction": jurisdiction or kb_meta.get("default_jurisdiction", "Việt Nam (Luật BVMT 2020)"),
        "source": entry.get("source"),
        "last_verified": entry.get("last_verified"),
        "disclaimer": kb_meta.get("disclaimer"),
        "is_override_applied": override_key is not None,
        "override_reason": f"Phát hiện đặc trưng '{override_key}' yêu cầu xử lý đặc biệt thay vì nhóm '{waste_class}'." if override_key else None
    }

    # If uncertain, add conditional caveat and safety confirmation warnings
    if is_uncertain:
        result["recyclability"] = dict(result["recyclability"]) if result.get("recyclability") else {}
        result["recyclability"]["status"] = "conditional_or_uncertain"
        if uncertainty_note and uncertainty_note not in result["tips"]:
            result["tips"].insert(0, f"Lưu ý độ tin cậy: {uncertainty_note}")

        # Safety Check: Ambiguous metal objects must never be assumed safe
        if target_key == "metal":
            metal_hazard_confirm = (
                "⚠️ CẢNH BÁO XÁC NHẬN AN TOÀN: Độ tin cậy cho thấy vật thể kim loại có thể không phải là lon/hộp tái chế thông thường. "
                "Nếu đây là PIN, ẮC QUY hoặc BÌNH XỊT NÉN KHÍ (AEROSOL), TUYỆT ĐỐI KHÔNG bỏ vào thùng rác tái chế kim loại "
                "vì có nguy cơ phát nổ và rò rỉ hóa chất độc hại! Hãy mang đến điểm thu gom rác nguy hại."
            )
            if metal_hazard_confirm not in result["tips"]:
                result["tips"].insert(0, metal_hazard_confirm)

        # Safety Check: Ambiguous 'other' objects may hide e-waste or batteries
        elif target_key == "other":
            other_hazard_confirm = (
                "⚠️ CẢNH BÁO XÁC NHẬN: Nhóm 'Rác khác' là danh mục hỗn hợp. "
                "Nếu đồ vật có chứa pin, bình ắc quy hoặc bo mạch điện tử bên trong, "
                "vui lòng tháo rời pin và linh kiện điện tử để chuyển giao cho điểm thu gom nguy hại/e-waste."
            )
            if other_hazard_confirm not in result["tips"]:
                result["tips"].insert(0, other_hazard_confirm)

    return result


def build_legacy_recommendations_dict() -> Dict[str, Dict[str, Any]]:
    """
    Constructs a dictionary matching the existing RECOMMENDATIONS structure
    for 100% backward compatibility with existing routes and tests.
    """
    categories = get_all_categories()
    legacy_dict = {}

    for key, data in categories.items():
        legacy_dict[key] = {
            "label_vi": data.get("label_vi", key),
            "bin": data.get("bin", "Thùng rác phù hợp"),
            "bin_color": data.get("bin_color", "#64748b"),
            "guidance": data.get("guidance", ""),
            "tips": data.get("tips", []),
            "combustibility": data.get("combustibility"),
            "recyclability": data.get("recyclability"),
            "processing_steps": data.get("processing_steps"),
            "environmental_impact": data.get("environmental_impact")
        }

    return legacy_dict


# Pre-computed legacy dictionary for drop-in replacement
RECOMMENDATIONS_FROM_KB: Dict[str, Dict[str, Any]] = build_legacy_recommendations_dict()
