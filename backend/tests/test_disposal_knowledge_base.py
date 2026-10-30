"""
Comprehensive Unit Tests for Waste Disposal & Recycling Knowledge Base.
Tests Phase 4 & Phase 7 requirements:
1. Clean separation between Combustibility and Recyclability.
2. Safety overrides for hazardous waste (Batteries, E-waste, Aerosol).
3. Conditional recyclability (Composite cartons / Tetra Pak).
4. Top-2 probability margin uncertainty calibration (< 15%).
5. Verified jurisdiction and source attribution.
"""

import sys
import unittest
from pathlib import Path
from PIL import Image
from io import BytesIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.knowledge_base import (
    load_knowledge_base,
    get_all_categories,
    get_disposal_guidance,
    detect_hazardous_or_special_override,
    RECOMMENDATIONS_FROM_KB
)
from backend.app.main import app, process_waste_image_bytes, load_model_from_checkpoint, find_checkpoint_path
import torch
from torchvision import transforms


class TestDisposalKnowledgeBase(unittest.TestCase):

    def test_kb_schema_and_categories_presence(self):
        """Verifies knowledge base loads valid version, disclaimer, and core categories."""
        kb = load_knowledge_base()
        self.assertIn("version", kb)
        self.assertIn("default_jurisdiction", kb)
        self.assertIn("categories", kb)

        cats = get_all_categories()
        # Core classes must exist
        for core_cls in ["plastic", "paper", "metal", "glass", "organic", "other", "medical"]:
            self.assertIn(core_cls, cats)

        # Specialized categories must exist
        self.assertIn("battery", cats)
        self.assertIn("e_waste", cats)
        self.assertIn("composite_carton", cats)
        self.assertIn("medical", cats)

    def test_combustibility_vs_recyclability_separation(self):
        """Verifies Combustibility and Recyclability are evaluated as strictly separate concepts."""
        cats = get_all_categories()
        
        # Plastic: Combustible (with toxic hazard), but Recyclable
        plastic = cats["plastic"]
        self.assertTrue(plastic["combustibility"]["is_combustible"])
        self.assertEqual(plastic["recyclability"]["status"], "recyclable")

        # Metal: Non-combustible, but 100% Recyclable
        metal = cats["metal"]
        self.assertFalse(metal["combustibility"]["is_combustible"])
        self.assertEqual(metal["recyclability"]["status"], "recyclable")

        # Battery: High explosion hazard when heated, hazardous special collection
        battery = cats["battery"]
        self.assertTrue(battery["combustibility"]["is_combustible"])
        self.assertEqual(battery["recyclability"]["status"], "hazardous_special_collection")

    def test_battery_hazard_override_prevents_false_metal_recycling(self):
        """
        Failure Mode 2 Audit Fix:
        When a battery image is classified as metal, the knowledge base detects the
        battery cue and safely overrides to hazardous waste disposal.
        """
        # Scenario: Classifier predicted metal, but file is battery1.jpg
        guidance = get_disposal_guidance(waste_class="metal", filename="battery1.jpg")
        self.assertTrue(guidance["is_override_applied"])
        self.assertEqual(guidance["material_id"], "battery")
        self.assertIn("nguy hại", guidance["label_vi"].lower())
        self.assertIn("điểm thu gom pin cũ", guidance["bin"].lower())
        self.assertIsNotNone(guidance["special_destination"])
        self.assertEqual(guidance["recyclability"]["status"], "hazardous_special_collection")
        self.assertIn("nguy cơ phát nổ", guidance["combustibility"]["status"].lower())

    def test_ewaste_hazard_override(self):
        """
        Detector cue override:
        When an electronic object like 'cell phone' or 'keyboard' is detected,
        guidance routes to e-waste recycling rather than ordinary plastic/metal bin.
        """
        guidance = get_disposal_guidance(waste_class="plastic", detected_object="cell phone")
        self.assertTrue(guidance["is_override_applied"])
        self.assertEqual(guidance["material_id"], "e_waste")
        self.assertIn("rác thải điện tử", guidance["label_vi"].lower())
        self.assertEqual(guidance["recyclability"]["status"], "hazardous_special_collection")

    def test_conditional_recyclability_composite_carton(self):
        """Verifies conditional recycling status for composite Tetra Pak packaging."""
        guidance = get_disposal_guidance(waste_class="paper", filename="milk_carton_tetra_pak.jpg")
        self.assertTrue(guidance["is_override_applied"])
        self.assertEqual(guidance["material_id"], "composite_carton")
        self.assertEqual(guidance["recyclability"]["status"], "conditionally_recyclable")
        self.assertIn("preparation_requirements", guidance)
        self.assertGreaterEqual(len(guidance["preparation_requirements"]), 2)

    def test_uncertainty_handling_preserves_honesty(self):
        """Verifies that an uncertain prediction adds explicit caveats without inventing certainty."""
        guidance = get_disposal_guidance(
            waste_class="plastic",
            is_uncertain=True,
            uncertainty_note="Mô hình phân vân giữa Plastic (51%) và Metal (48%)."
        )
        self.assertEqual(guidance["recyclability"]["status"], "conditional_or_uncertain")
        self.assertTrue(any("độ tin cậy" in tip.lower() for tip in guidance["tips"]))

    def test_jurisdiction_and_source_attribution(self):
        """Verifies that all entries include legal jurisdiction and verification sources."""
        for cat_name, cat in get_all_categories().items():
            self.assertIn("source", cat)
            self.assertIn("last_verified", cat)
            self.assertIsNotNone(cat["source"])


class TestEndToEndKnowledgeBaseIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        app.state.device = str(device)
        app.state.classes = ["plastic", "paper", "metal", "glass", "organic", "other", "medical"]
        app.state.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        ckpt = find_checkpoint_path()
        model, classes = load_model_from_checkpoint(ckpt, device)
        app.state.model = model
        app.state.classes = classes
        app.state.model_loaded = True

    def test_single_predict_returns_full_kb_metadata(self):
        """Verifies /api/predict produces the extended RecommendationInfo with sources."""
        sample_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
        self.assertTrue(sample_path.exists())

        with open(sample_path, "rb") as f:
            bytes_data = f.read()

        pred = process_waste_image_bytes(bytes_data, "Plastic_1.jpg")
        self.assertTrue(pred.success)
        recom = pred.recommendation
        self.assertEqual(recom.label_vi, "Nhựa (Plastic)")
        self.assertIsNotNone(recom.jurisdiction)
        self.assertIsNotNone(recom.source)
        self.assertIsNotNone(recom.last_verified)
        self.assertIsNotNone(recom.preparation_requirements)
        print(f"\n[PASS] Verified /api/predict output with KB metadata: {recom.jurisdiction} | {recom.source}")

    def test_battery_input_receives_hazard_guidance(self):
        """Failure Mode 2 E2E test: Battery input receives hazardous guidance even if model predicts metal."""
        battery_path = PROJECT_ROOT / "data" / "raw" / "kaggle_garbage" / "extracted" / "garbage_classification" / "battery" / "battery1.jpg"
        if not battery_path.exists():
            self.skipTest("Battery sample image not found")

        with open(battery_path, "rb") as f:
            bytes_data = f.read()

        pred = process_waste_image_bytes(bytes_data, "battery1.jpg")
        self.assertTrue(pred.success)
        recom = pred.recommendation
        self.assertIn("pin", recom.label_vi.lower())
        self.assertIn("thu gom pin", recom.bin.lower())
        self.assertIsNotNone(recom.special_destination)
        self.assertIn("nguy hiểm", recom.combustibility.status.lower())
        print(f"\n[PASS] Battery input correctly received safety override: {recom.label_vi} -> {recom.bin}")

    def test_ambiguous_metal_triggers_safe_confirmation_warning(self):
        """Phase 4 test: An ambiguous metal object must trigger a safe confirmation warning rather than assuming it is safe."""
        guidance = get_disposal_guidance(
            waste_class="metal",
            is_uncertain=True,
            uncertainty_note="Top-2 margin gap < 15%"
        )
        self.assertEqual(guidance["material_id"], "metal")
        # Must have safety warning about batteries and aerosol cans
        has_warning = any("pin" in tip.lower() and "nguy hại" in tip.lower() for tip in guidance["tips"])
        self.assertTrue(has_warning)
        print("\n[PASS] Ambiguous metal correctly triggered safety confirmation warning.")

    def test_calibrated_confidence_separate_from_raw_confidence(self):
        """Phase 3 test: Calibrated confidence is computed via Temperature Scaling and kept separate from raw confidence."""
        sample_path = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste" / "Plastic" / "Plastic_1.jpg"
        with open(sample_path, "rb") as f:
            bytes_data = f.read()

        pred = process_waste_image_bytes(bytes_data, "Plastic_1.jpg")
        self.assertTrue(pred.success)
        self.assertIsNotNone(pred.calibrated_confidence)
        self.assertIsInstance(pred.confidence, float)
        self.assertIsInstance(pred.calibrated_confidence, float)
        # Because T* = 1.6366 > 1, extreme probabilities are softened (calibrated_confidence <= confidence)
        self.assertLessEqual(pred.calibrated_confidence, pred.confidence + 1e-4)
        print(f"\n[PASS] Calibrated confidence ({pred.calibrated_confidence*100:.2f}%) successfully kept separate from raw ({pred.confidence*100:.2f}%).")


if __name__ == "__main__":
    unittest.main()

