"""Input/provenance checks; these tests do not fabricate model predictions."""
import io
import json
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.safety_routes import router
from backend.app.knowledge_base import get_disposal_guidance, detect_hazardous_or_special_override

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def image_bytes(color=(80, 130, 180), size=(80, 80)):
    data = io.BytesIO()
    Image.new('RGB', size, color).save(data, format='PNG')
    return data.getvalue()


def test_scan_rejects_non_image_and_oversized_upload():
    assert client.post('/api/safety/scan', files={'file': ('fake.jpg', b'not image')}).status_code == 400
    assert client.post('/api/safety/scan', files={'file': ('large.jpg', b'x' * (10 * 1024 * 1024 + 1))}).status_code == 413


def test_scan_rejects_dark_and_small_input():
    for data in [image_bytes(color=(0, 0, 0)), image_bytes(size=(20, 20))]:
        assert client.post('/api/safety/scan', files={'file': ('input.png', data)}).status_code == 400


def test_missing_artifact_returns_unavailable_without_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv('BATTERY_SCREEN_RUN', str(tmp_path))
    assert not client.get('/api/safety/capabilities').json()['artifact_files_present']
    response = client.post('/api/safety/scan', files={'file': ('battery.png', image_bytes())})
    assert response.status_code == 503
    assert 'battery_score' not in response.json()


def test_invalid_item_does_not_silently_become_ordinary():
    assert client.post('/api/safety/assess', json={'item_type': 'anything'}).status_code == 422


def test_contacts_are_candidates_only_and_no_medical_acceptance():
    result = client.get('/api/safety/collection-contacts', params={'city': 'Đà Nẵng'}).json()
    assert len(result['items']) == 2
    assert all(not item['acceptance_confirmed_current'] for item in result['items'])
    assert client.get('/api/safety/collection-contacts', params={'material': 'medical'}).json()['items'] == []


def test_filename_substrings_do_not_create_hazard_claims():
    for name in ['spinning.jpg', 'pillow.jpg', 'masking.jpg']:
        assert detect_hazardous_or_special_override('other', filename=name) is None
    result = get_disposal_guidance('metal', filename='battery1.jpg')
    assert result['material_id'] == 'battery'
    assert 'Tên tệp' in result['override_reason']


def test_uncertain_ewaste_does_not_advise_self_disassembly():
    tips = get_disposal_guidance('other', is_uncertain=True)['tips']
    assert any('không tự tháo pin' in tip for tip in tips)


def test_legacy_directory_exposes_unverified_status():
    from backend.app.main import app as legacy_app
    rows = TestClient(legacy_app).get('/api/recycling-centers').json()
    assert rows
    assert all(row['verification_status'] == 'unverified_legacy_entry' for row in rows)
    assert all(row['acceptance_confirmed_current'] is False for row in rows)


def test_failed_model_is_blocked_before_loading_any_checkpoint(tmp_path):
    from backend.app.battery_screen import _load_model
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'runtime_enabled':False}))
    with pytest.raises(RuntimeError,match='blocked'):
        _load_model(str(tmp_path/'missing.pt'),str(report),0,0)
