"""Mandatory request/readiness regressions. No weights or inference required."""
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import State

from backend.app import main, database, battery_screen, image_input


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.setattr(main.app, 'state', State())
    monkeypatch.setattr(database, 'DB_PATH', tmp_path/'history.db')
    # Explicit missing-model fault injection. Never initialize fake/random weights.
    monkeypatch.setattr(main, 'find_checkpoint_path', lambda: tmp_path/'absent_legacy.pth')
    with TestClient(main.app) as client:
        yield client


def image_bytes(size=(128,128), color=(90,130,170), fmt='JPEG'):
    stream=io.BytesIO()
    Image.new('RGB',size,color).save(stream,format=fmt)
    return stream.getvalue()


def test_health_is_liveness_and_reports_each_model(api_client):
    response=api_client.get('/api/health');body=response.json()
    assert response.status_code==200 and body['service_alive']
    assert body['status']=='degraded'
    assert body['ai_model_loaded'] is False and body['checkpoint_loaded'] is False
    assert body['capabilities']['legacy_classifier']['status']=='legacy_unavailable'
    yolo=body['capabilities']['yolo_detector']
    current=api_client.get('/api/yolo/capabilities').json()
    assert yolo['integrated'] and yolo['runtime_ready']==current['runtime_ready']
    assert yolo['model_loaded']==current['model_loaded'] and yolo['status']==current['status']
    assert all(not row['production_ready'] for row in body['capabilities'].values())


@pytest.mark.parametrize('capability',['legacy_classifier'])
def test_readiness_is_unavailable_for_unserved_models(api_client,capability):
    response=api_client.get('/api/readiness',params={'capability':capability})
    assert response.status_code==503
    assert response.json()['capability']==capability and not response.json()['ready']


def test_readiness_requires_explicit_known_capability(api_client):
    assert api_client.get('/api/readiness').status_code==422
    assert api_client.get('/api/readiness?capability=anything').status_code==422


def test_battery_readiness_matches_safety_capabilities(api_client):
    capabilities=api_client.get('/api/safety/capabilities').json()
    response=api_client.get('/api/readiness?capability=battery_classifier')
    assert response.status_code==(200 if capabilities['runtime_ready'] else 503)
    assert response.json()['ready']==capabilities['runtime_ready']
    assert response.json()['model']['model_loaded']==capabilities['ai_model_loaded']
    assert response.json()['model']['production_ready'] is False


@pytest.mark.parametrize('endpoint',['/api/predict','/api/detect','/api/safety/scan','/api/yolo/detect'])
@pytest.mark.parametrize('content,expected',[(b'',400),(b'not an image',400),(b'x'*(10*1024*1024+1),413)])
def test_bad_upload_precedes_model_readiness(api_client,endpoint,content,expected):
    response=api_client.post(endpoint,files={'file':('input.jpg',content,'image/jpeg')})
    assert response.status_code==expected


@pytest.mark.parametrize('endpoint',['/api/predict','/api/detect','/api/safety/scan','/api/yolo/detect'])
@pytest.mark.parametrize('payload',[{'size':(20,20)},{'color':(0,0,0)},{'color':(255,255,255)},{'fmt':'BMP'}])
def test_image_constraints_precede_readiness(api_client,endpoint,payload):
    assert api_client.post(endpoint,files={'file':('photo.jpg',image_bytes(**payload))}).status_code==400


@pytest.mark.parametrize('endpoint',['/api/predict','/api/detect'])
def test_valid_content_still_reports_missing_legacy(api_client,endpoint):
    response=api_client.post(endpoint,files={'file':('misleading.txt',image_bytes(),'text/plain')})
    assert response.status_code==503 and 'legacy' in response.json()['detail']


def test_pixel_limit_is_a_request_error(api_client,monkeypatch):
    monkeypatch.setattr(image_input,'MAX_IMAGE_PIXELS',100)
    assert api_client.post('/api/predict',files={'file':('large.jpg',image_bytes())}).status_code==413


@pytest.mark.parametrize('endpoint',['/api/predict','/api/detect','/api/safety/scan','/api/yolo/detect'])
def test_missing_upload_is_framework_validation(api_client,endpoint):
    assert api_client.post(endpoint).status_code==422


def test_batch_retains_envelope_and_item_status(api_client):
    response=api_client.post('/api/predict/batch',files=[('files',('bad.jpg',b'broken')),('files',('valid.jpg',image_bytes()))])
    assert response.status_code==200
    body=response.json()
    assert body['total']==2 and body['failed']==2 and body['successful']==0
    assert [row['status_code'] for row in body['items']]==[400,503]


@pytest.mark.parametrize('endpoint',['/api/predict/url','/api/detect/url'])
@pytest.mark.parametrize('content,expected',[(b'broken',400),(image_bytes(),503),(b'x'*(10*1024*1024+1),413)])
def test_url_validation_preserves_request_error_status(api_client,monkeypatch,endpoint,content,expected):
    class Response(io.BytesIO):
        headers={'Content-Length':str(len(content)),'Content-Type':'image/jpeg'}
    monkeypatch.setattr(main.urllib.request,'urlopen',lambda *args,**kwargs:Response(content))
    assert api_client.post(endpoint,json={'image_url':'https://example.invalid/photo.jpg'}).status_code==expected


@pytest.mark.parametrize('endpoint',['/api/predict/url','/api/detect/url'])
def test_invalid_url_is_rejected_without_fetch(api_client,monkeypatch,endpoint):
    def unexpected(*args,**kwargs):pytest.fail('Invalid URL must not be fetched')
    monkeypatch.setattr(main.urllib.request,'urlopen',unexpected)
    assert api_client.post(endpoint,json={'image_url':'ftp://example.invalid/x'}).status_code==400


def test_analytics_disclaimer_communicates_actual_limitation(api_client):
    body=api_client.get('/api/analytics/dashboard').json()
    assert 'không đồng nghĩa' in body['disclaimer'].casefold()
    assert 'tái chế' in body['disclaimer'].casefold()
    assert 'carbon_saved_kg' not in body and 'weight_recycled_kg' not in body


def test_cached_model_for_other_artifact_never_marks_missing_model_loaded(api_client,monkeypatch,tmp_path):
    monkeypatch.setattr(battery_screen,'_loaded_artifact_key',('other.pt','other.json',1,1))
    monkeypatch.setenv('BATTERY_SCREEN_RUN',str(tmp_path/'missing'))
    body=api_client.get('/api/safety/capabilities').json()
    assert not body['ai_model_loaded'] and not body['runtime_ready'] and not body['artifact_verified']
    assert api_client.get('/api/readiness?capability=battery_classifier').status_code==503


@pytest.mark.model
def test_real_battery_artifacts_and_current_cache_identity(monkeypatch,tmp_path):
    """Opt-in inference smoke, no new accuracy claim and no downloads."""
    root=Path(__file__).resolve().parents[2]
    run=root/'runs/battery_screen/yolo26n_cls_multisource_v2'
    manifest=root/'data/processed/battery_multisource_v2/manifest.json'
    if not (run/'weights/best.pt').is_file() or not (run/'battery_report.json').is_file() or not manifest.is_file():
        pytest.skip('Genuine battery artifacts/images unavailable')
    monkeypatch.setenv('BATTERY_SCREEN_RUN',str(run))
    caps=battery_screen.capabilities()
    if battery_screen.importlib.util.find_spec('ultralytics') is None:
        pytest.skip('Optional inference dependency is not installed; no installation in this task')
    assert caps['runtime_ready'], caps['readiness_reason']
    row=next(r for r in json.loads(manifest.read_text()) if r['split']=='test')
    with (manifest.parent/row['image_file']).open('rb') as stream:
        image,quality=battery_screen.prepare_image(stream)
    result=battery_screen.screen_image(image,quality)
    assert 0<=result['battery_score']<=1 and result['is_uncertain'] and not result['confirmed_safe']
    assert battery_screen.capabilities()['ai_model_loaded']
    monkeypatch.setenv('BATTERY_SCREEN_RUN',str(tmp_path))
    assert not battery_screen.capabilities()['ai_model_loaded']


@pytest.fixture
def genuine_battery_copy(tmp_path,monkeypatch):
    import shutil
    root=Path(__file__).resolve().parents[2]/'runs/battery_screen/yolo26n_cls_multisource_v2'
    if not (root/'weights/best.pt').is_file() or not (root/'battery_report.json').is_file():
        pytest.skip('Genuine local battery artifacts unavailable; never create substitute weights')
    (tmp_path/'weights').mkdir()
    shutil.copy2(root/'weights/best.pt',tmp_path/'weights/best.pt')
    shutil.copy2(root/'battery_report.json',tmp_path/'battery_report.json')
    monkeypatch.setenv('BATTERY_SCREEN_RUN',str(tmp_path))
    return tmp_path


@pytest.mark.model
def test_mismatched_or_blocked_report_is_never_ready(genuine_battery_copy):
    report=genuine_battery_copy/'battery_report.json'
    metadata=json.loads(report.read_text())
    metadata['checkpoint_sha256']='0'*64
    report.write_text(json.dumps(metadata))
    caps=battery_screen.capabilities()
    assert caps['readiness_status']=='invalid_artifact' and not caps['runtime_ready']
    metadata['runtime_enabled']=False
    report.write_text(json.dumps(metadata))
    caps=battery_screen.capabilities()
    assert caps['readiness_status']=='blocked' and not caps['ai_model_loaded']
    report.write_text('[]')
    caps=battery_screen.capabilities()
    assert caps['readiness_status']=='invalid_artifact' and not caps['runtime_ready']


@pytest.mark.model
def test_load_failure_changes_readiness(genuine_battery_copy,monkeypatch):
    monkeypatch.setattr(battery_screen,'_failed_artifact_key',None)
    def fail(*args):raise RuntimeError('Injected loader failure; no predictions fabricated')
    monkeypatch.setattr(battery_screen,'_load_model',fail)
    image,quality=battery_screen.prepare_image(io.BytesIO(image_bytes()))
    with pytest.raises(RuntimeError,match='Injected loader failure'):
        battery_screen.screen_image(image,quality)
    caps=battery_screen.capabilities()
    assert caps['readiness_status']=='load_failed' and not caps['runtime_ready'] and not caps['ai_model_loaded']
