"""02A contract/parsing fixtures are synthetic; only opt-in tests run genuine inference."""
import copy
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from backend.app import yolo_detector as yolo
from backend.tests.test_api_contract import api_client, image_bytes


@pytest.fixture(autouse=True)
def isolated_adapter(monkeypatch):
    monkeypatch.setattr(yolo, '_loaded_model', None)
    monkeypatch.setattr(yolo, '_loaded_key', None)
    monkeypatch.setattr(yolo, '_failure', None)
    monkeypatch.delenv('YOLO_CONFIDENCE_THRESHOLD', raising=False)
    monkeypatch.delenv('YOLO_UNCERTAINTY_THRESHOLD', raising=False)


@pytest.fixture
def manifest():
    return json.loads(yolo.MANIFEST_PATH.read_text(encoding='utf-8'))


@pytest.fixture
def stub_artifact(monkeypatch, manifest):
    # Pure test metadata. No checkpoint is created and no constructor reads fake weights.
    artifact = yolo.Artifact(Path('fixture-no-checkpoint.pt'), Path('fixture-summary.json'),
                             manifest, ('fixture-artifact-A',), manifest['training_ultralytics_version'])
    monkeypatch.setattr(yolo, 'inspect_artifact', lambda: artifact)
    return artifact


class FixtureEngine:
    """Contract-only fixture boxes, never reported as measured model accuracy."""
    def __init__(self, rows=(), error=None):
        self.rows, self.error, self.calls = rows, error, []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        if self.error:raise RuntimeError(self.error)
        boxes = SimpleNamespace(xyxy=torch.tensor([r[0] for r in self.rows]).reshape(-1,4),
                                cls=torch.tensor([r[1] for r in self.rows]), conf=torch.tensor([r[2] for r in self.rows]))
        return [SimpleNamespace(boxes=boxes)]


def test_manifest_matches_nine_ordered_object_groups(manifest):
    yolo.validate_manifest(manifest)
    assert [r['class_name'] for r in manifest['class_mapping']]==[
        'bottle_jar','can','cup','carton','cap_lid','bag_film_wrapper','paper','cigarette','other_litter']
    assert manifest['task']=='detect' and manifest['taxonomy']=='object_groups'


@pytest.mark.parametrize('kind',['reordered_ids','duplicate_name','wrong_task','wrong_hash','wrong_size'])
def test_invalid_manifest_rejected(manifest,kind):
    if kind=='reordered_ids':manifest['class_mapping'].reverse()
    elif kind=='duplicate_name':manifest['class_mapping'][1]['class_name']=manifest['class_mapping'][0]['class_name']
    elif kind=='wrong_task':manifest['task']='classify'
    elif kind=='wrong_hash':manifest['checkpoint_sha256']='wrong'
    else:manifest['imgsz']=224
    with pytest.raises(ValueError):yolo.validate_manifest(manifest)


@pytest.mark.parametrize('kind',['mapping','head','task','version'])
def test_loaded_checkpoint_contract_must_match_metadata(manifest,kind):
    model=SimpleNamespace(task='detect', names={r['class_id']:r['class_name'] for r in manifest['class_mapping']},
                          model=SimpleNamespace(model=[SimpleNamespace(nc=9)]), ckpt={'version':manifest['training_ultralytics_version']})
    if kind=='mapping':model.names[0],model.names[1]=model.names[1],model.names[0]
    elif kind=='head':model.model.model[-1].nc=7
    elif kind=='task':model.task='classify'
    else:model.ckpt['version']='different'
    with pytest.raises(yolo.YoloServiceError,match='checkpoint YOLO'):yolo.validate_checkpoint_model(model,manifest)


@pytest.mark.parametrize('value',['bad','nan','inf','0','1.1','0.8'])
def test_invalid_confidence_config_is_unavailable(api_client,monkeypatch,value):
    monkeypatch.setenv('YOLO_CONFIDENCE_THRESHOLD',value)
    caps=api_client.get('/api/yolo/capabilities').json()
    assert caps['status']=='invalid_configuration' and not caps['runtime_ready']
    response=api_client.post('/api/yolo/detect',files={'file':('image.jpg',image_bytes())})
    assert response.status_code==503 and response.json()['detail']['code']=='yolo_invalid_configuration'


def test_missing_artifacts_do_not_load_or_download(api_client,monkeypatch,tmp_path):
    monkeypatch.setenv('YOLO_DETECTOR_RUN',str(tmp_path))
    monkeypatch.setattr(yolo,'_load_model',lambda *args:pytest.fail('Must not construct missing model'))
    caps=api_client.get('/api/yolo/capabilities').json()
    assert caps['integrated'] and caps['status']=='unavailable' and not caps['model_loaded']
    assert not caps['runtime_ready'] and not caps['artifact_files_present']
    assert api_client.get('/api/readiness?capability=yolo_detector').status_code==503
    assert api_client.post('/api/yolo/detect',files={'file':('image.jpg',image_bytes())}).status_code==503
    assert api_client.post('/api/yolo/detect',files={'file':('broken.jpg',b'broken')}).status_code==400


def test_contract_boxes_threshold_uncertainty_and_metadata(api_client,monkeypatch,stub_artifact):
    engine=FixtureEngine([([-2,1,30,40],0,.3),([50,60,200,150],8,.8),([1,1,2,2],1,.1)])
    monkeypatch.setattr(yolo,'_load_model',lambda artifact:engine)
    assert api_client.get('/api/readiness?capability=yolo_detector').json()['model']['status']=='experimental_not_loaded'
    response=api_client.post('/api/yolo/detect',files={'file':('neutral.jpg',image_bytes())})
    assert response.status_code==200
    body=response.json()
    assert body['inference_status']=='completed' and body['total_objects']==2
    assert body['detections'][0]['class_name']=='bottle_jar' and body['detections'][1]['class_name']=='other_litter'
    assert body['detections'][0]['bounding_box']==[0,1,30,40]
    assert body['detections'][1]['bounding_box']==[50,60,128,128]
    assert body['detections'][0]['is_uncertain'] and not body['detections'][1]['is_uncertain']
    assert body['is_uncertain'] and not body['production_ready'] and not body['confirmed_safe']
    assert body['hazard_assessment']=='not_assessed' and body['model']['taxonomy']=='object_groups'
    assert len(body['model']['class_mapping'])==9 and body['model']['checkpoint_sha256']==stub_artifact.manifest['checkpoint_sha256']
    assert body['quality_assessment']['is_acceptable'] and not body['confidence_is_calibrated_probability']
    assert engine.calls[0]['conf']==.25 and engine.calls[0]['device']=='cpu' and not engine.calls[0]['save']
    assert api_client.get('/api/yolo/capabilities').json()['status']=='experimental_loaded'


def test_empty_detection_does_not_certify_absence_or_safety(api_client,monkeypatch,stub_artifact):
    monkeypatch.setattr(yolo,'_load_model',lambda artifact:FixtureEngine())
    body=api_client.post('/api/yolo/detect',files={'file':('image.jpg',image_bytes())}).json()
    assert body['detections']==[] and body['detection_status']=='no_detections_above_threshold'
    assert body['is_uncertain'] and not body['confirmed_safe'] and body['hazard_assessment']=='not_assessed'


def test_boxes_use_exif_oriented_dimensions(api_client,monkeypatch,stub_artifact):
    engine=FixtureEngine([([10,20,70,110],0,.6)])
    monkeypatch.setattr(yolo,'_load_model',lambda artifact:engine)
    image=Image.new('RGB',(120,80),(90,130,170));exif=image.getexif();exif[274]=6
    stream=io.BytesIO();image.save(stream,format='JPEG',exif=exif)
    response=api_client.post('/api/yolo/detect',files={'file':('rotated.jpg',stream.getvalue())})
    assert response.status_code==200
    body=response.json()
    assert (body['image_width'],body['image_height'])==(80,120)
    assert body['coordinate_space']=='exif_oriented_image'
    assert body['detections'][0]['bounding_box']==[10,20,70,110]


@pytest.mark.parametrize('bad_box,bad_class,bad_score',[([1,2,1,3],0,.7),([1,2,3,4],9,.7),([1,2,3,4],.5,.7),
                                                       ([1,2,3,4],0,float('nan')),([1,2,3,4],0,1.1),
                                                       ([1,2,float('inf'),4],0,.7),([200,200,300,300],0,.7)])
def test_bad_engine_outputs_are_rejected(manifest,bad_box,bad_class,bad_score):
    with pytest.raises(ValueError):
        yolo.parse_detections([bad_box],[bad_class],[bad_score],manifest['class_mapping'],128,128,yolo.configuration())


def test_configuration_controls_filter_and_review_band(monkeypatch,manifest):
    monkeypatch.setenv('YOLO_CONFIDENCE_THRESHOLD','.4');monkeypatch.setenv('YOLO_UNCERTAINTY_THRESHOLD','.7')
    config=yolo.configuration()
    result=yolo.parse_detections([[1,1,9,9]]*3,[0,1,2],[.3,.5,.8],manifest['class_mapping'],128,128,config)
    assert len(result)==2 and result[0].is_uncertain and not result[1].is_uncertain


@pytest.mark.parametrize('stage',['load','invalid_artifact','inference'])
def test_service_failures_are_explicit_and_readiness_changes(api_client,monkeypatch,stub_artifact,stage):
    def loader(artifact):
        if stage=='load':raise RuntimeError('fixture loader failure')
        if stage=='invalid_artifact':raise yolo.YoloServiceError('yolo_invalid_artifact','invalid_artifact','Fixture mapping mismatch')
        return FixtureEngine(error='fixture inference failure')
    monkeypatch.setattr(yolo,'_load_model',loader)
    response=api_client.post('/api/yolo/detect',files={'file':('image.jpg',image_bytes())})
    assert response.status_code==(500 if stage=='inference' else 503)
    assert 'detections' not in response.json() and response.json()['detail']['capability']=='yolo_detector'
    assert response.json()['detail']['inference_status']==('failed' if stage=='inference' else 'not_started')
    caps=api_client.get('/api/yolo/capabilities').json()
    assert not caps['runtime_ready'] and caps['status']=={'load':'load_failed','invalid_artifact':'invalid_artifact','inference':'inference_error'}[stage]
    assert api_client.get('/api/readiness?capability=yolo_detector').status_code==503
    # Transient errors may be retried with the same verified artifact.
    monkeypatch.setattr(yolo,'_load_model',lambda artifact:FixtureEngine())
    if stage=='inference':monkeypatch.setattr(yolo,'_loaded_model',FixtureEngine())
    assert api_client.post('/api/yolo/detect',files={'file':('image.jpg',image_bytes())}).status_code==200
    assert api_client.get('/api/yolo/capabilities').json()['runtime_ready']


def test_current_artifact_identity_controls_cache(monkeypatch,stub_artifact):
    loads=[]
    def loader(artifact):loads.append(artifact.key);return FixtureEngine()
    monkeypatch.setattr(yolo,'_load_model',loader)
    image=Image.new('RGB',(128,128),(90,130,170))
    yolo.detect(image,{})
    yolo.detect(image,{})
    other=yolo.Artifact(stub_artifact.weights,stub_artifact.summary,stub_artifact.manifest,('fixture-artifact-B',),stub_artifact.runtime_version)
    monkeypatch.setattr(yolo,'inspect_artifact',lambda:other)
    assert not yolo.capabilities()['model_loaded']
    yolo.detect(image,{})
    assert loads==[stub_artifact.key,other.key]


@pytest.fixture
def real_artifact_copy(monkeypatch,tmp_path):
    for relative in ['weights/best.pt','pilot_summary.json']:
        source=yolo.DEFAULT_RUN/relative
        if not source.is_file():pytest.skip('Genuine YOLO artifact unavailable; no downloads/substitutes')
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    monkeypatch.setenv('YOLO_DETECTOR_RUN',str(tmp_path))
    return tmp_path


@pytest.mark.model
@pytest.mark.parametrize('change',['summary','weights','manifest'])
def test_real_artifact_tampering_is_detected(api_client,monkeypatch,real_artifact_copy,tmp_path,change):
    # Only modify copies in the test temp folder, never the source weights/metadata.
    if change=='summary':
        p=real_artifact_copy/'pilot_summary.json';p.write_bytes(p.read_bytes()+b' ')
    elif change=='weights':
        p=real_artifact_copy/'weights/best.pt';p.write_bytes(p.read_bytes()+b' ')
    else:
        p=tmp_path/'manifest.json';p.write_text('[]');monkeypatch.setattr(yolo,'MANIFEST_PATH',p)
    assert api_client.get('/api/yolo/capabilities').json()['status']=='invalid_artifact'
    response=api_client.post('/api/yolo/detect',files={'file':('valid.jpg',image_bytes())})
    assert response.status_code==503 and response.json()['detail']['code']=='yolo_invalid_artifact'


@pytest.mark.model
def test_real_model_smoke_uses_verified_nine_class_checkpoint(api_client,monkeypatch):
    manifest_path=yolo.ROOT/'data/processed/waste_multisource_v3/manifest.json'
    if not manifest_path.is_file() or not (yolo.DEFAULT_RUN/'weights/best.pt').is_file():
        pytest.skip('Genuine detector/image artifacts unavailable')
    monkeypatch.setenv('YOLO_DETECTOR_RUN',str(yolo.DEFAULT_RUN))
    artifact=yolo.inspect_artifact()
    row=next(r for r in json.loads(manifest_path.read_text()) if r['split']=='val')
    content=(manifest_path.parent/row['image_file']).read_bytes()
    response=api_client.post('/api/yolo/detect',files={'file':('neutral.jpg',content,'image/jpeg')})
    assert response.status_code==200,response.text
    body=response.json()
    assert body['model']['checkpoint_sha256']==artifact.manifest['checkpoint_sha256']
    assert body['model']['class_mapping']==artifact.manifest['class_mapping']
    assert body['is_uncertain'] and body['hazard_assessment']=='not_assessed'
    for box in body['detections']:
        assert artifact.manifest['class_mapping'][box['class_id']]['class_name']==box['class_name']
        x1,y1,x2,y2=box['bounding_box']
        assert 0<=x1<x2<=body['image_width'] and 0<=y1<y2<=body['image_height']
        assert body['confidence_threshold']<=box['confidence']<=1
    caps=api_client.get('/api/yolo/capabilities').json()
    assert caps['status']=='experimental_loaded' and caps['runtime_ready'] and not caps['production_ready']
