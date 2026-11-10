import unittest
from backend.app.safety_guidance import SafetyRequest,assess_safety,collection_contacts

class TestSafetyGuidance(unittest.TestCase):
    def test_missing_information_is_not_safe(self):
        r=assess_safety(SafetyRequest())
        self.assertFalse(r['confirmed_safe']);self.assertEqual(r['radiation_status'],'not_measured')
    def test_radioactive_cue_never_creates_dose_or_dropoff(self):
        r=assess_safety(SafetyRequest(item_type='radiation_suspected'))
        self.assertIsNone(r['radiation_dose']);self.assertEqual(r['collection_contacts'],[])
        self.assertEqual(r['status'],'HAZARD_SUSPECTED')
    def test_medical_cue_is_not_a_disease_diagnosis(self):
        r=assess_safety(SafetyRequest(item_type='medical_sharp'))
        self.assertIsNone(r['diagnosis']);self.assertEqual(r['infection_status'],'not_determinable_from_image')
    def test_smoking_battery_not_routed_to_public_dropoff(self):
        r=assess_safety(SafetyRequest(item_type='battery',battery_condition='hot_or_smoking'))
        self.assertEqual(r['status'],'URGENT_REVIEW');self.assertFalse(r['collection_contacts'])
    def test_damaged_battery_no_intact_packaging_advice(self):
        r=assess_safety(SafetyRequest(item_type='battery',battery_condition='damaged'))
        self.assertFalse(r['collection_contacts']);self.assertFalse(r['can_use_general_recycling_bin'])
    def test_unknown_battery_condition_is_conservative(self):
        self.assertFalse(assess_safety(SafetyRequest(item_type='battery'))['collection_contacts'])
    def test_region_filter_and_source_provenance(self):
        self.assertEqual(collection_contacts('Hà Nội'),[])
        for r in collection_contacts('Da Nang'):
            self.assertFalse(r['acceptance_confirmed_current']);self.assertTrue(r['source_url'].startswith('https://'))
    def test_ordinary_does_not_certify_absence_of_hazard(self):
        self.assertFalse(assess_safety(SafetyRequest(item_type='ordinary'))['confirmed_safe'])

    def test_material_advice_does_not_override_urgent_battery_condition(self):
        result=assess_safety(SafetyRequest(item_type='battery',battery_condition='hot_or_smoking',ordinary_material='plastic'))
        self.assertEqual(result['status'],'URGENT_REVIEW')
        self.assertFalse(any('làm sạch cặn' in step for step in result['steps']))

    def test_material_recycling_remains_conditional(self):
        result=assess_safety(SafetyRequest(item_type='ordinary',ordinary_material='plastic'))
        self.assertFalse(result['confirmed_safe'])
        self.assertTrue(any('ký hiệu nhựa không bảo đảm' in step for step in result['steps']))
