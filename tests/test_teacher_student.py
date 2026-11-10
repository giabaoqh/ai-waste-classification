"""Synthetic unit fixtures only: never measured teacher outputs or human approvals."""
from copy import deepcopy
from pathlib import Path
import random

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import yaml

from scripts.teacher_student.audit import audit_dataset
from scripts.teacher_student.common import CLASSES, content_hash, load_dataset, read_json, sha256, snapshot, write_json
from scripts.teacher_student.review import (compare_proposals, export_approved, get_item, import_teacher,
                                           item_path, prepare_pilot, save_review, select_pilot)
from scripts.teacher_student.server import create_app


@pytest.fixture
def pool(tmp_path):
    root = tmp_path / 'dataset'
    rows = []
    rng = random.Random(123)
    for index in range(12):
        split = 'train' if index < 10 else 'val' if index == 10 else 'test'
        image = root / f'images/{split}/fixture_{index}.jpg'
        label = root / f'labels/{split}/fixture_{index}.txt'
        image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
        Image.frombytes('RGB', (80, 64), rng.randbytes(80 * 64 * 3)).save(image)
        label.write_text(f'{index % 9} .5 .5 .25 .25\n')
        rows.append({'image_file': image.relative_to(root).as_posix(), 'label_file': label.relative_to(root).as_posix(),
                     'sha256': sha256(image), 'source_sha256': sha256(image), 'split': split, 'source': 'kaggle_warp',
                     'human_review': 'pending', 'provenance': {'file': f'fixture/{index}.jpg', 'capture_date': split, 'license_declared': 'UNIT_FIXTURE_ONLY'}})
    write_json(root / 'manifest.json', rows)
    (root / 'dataset.yaml').write_text(yaml.safe_dump({'path': str(root), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': dict(enumerate(CLASSES))}))
    audit = audit_dataset(root, tmp_path / 'no-raw-source')
    return root, audit


@pytest.fixture
def workspace(pool, tmp_path):
    root, audit = pool
    target = tmp_path / 'review'
    prepare_pilot(audit, target, count=4)
    return target


def first(workspace):
    key = read_json(workspace / 'workspace.json')['image_ids'][0]
    return get_item(workspace, key)


def approval(record, **updates):
    data = {'expected_revision': record['revision'], 'boxes': deepcopy(record['draft_boxes']), 'issues': [],
            'reviewer': 'UNIT_TEST_REVIEWER_NOT_A_REAL_APPROVAL', 'status': 'approved', 'rights_evidence': 'Synthetic unit fixture; no source image rights claim',
            'checks': {key: True for key in ['all_objects_reviewed', 'heuristics_reviewed', 'provenance_reviewed', 'hazards_reviewed']}}
    data.update(updates)
    return data


def teacher_bundle(record):
    box = deepcopy(record['original_boxes'][0])
    box.update(confidence=.7, confidence_kind='detector_score', raw_label='unit fixture box', proposal_id='fixture-proposal')
    return {'schema_version': 1, 'teacher': {'kind': 'grounding_dino', 'model_id': 'UNIT_FIXTURE_NOT_INFERENCE',
            'version': 'fixture-v1', 'run_id': 'fixture-run', 'generated_at': '2026-11-10T00:00:00Z',
            'producer': 'pytest fixture', 'adapter_version': 'fixture-1', 'execution_mode': 'local', 'prompt_sha256': 'a' * 64,
            'artifact_sha256': 'b' * 64, 'coordinate_space': 'xyxy_pixels_exif_oriented', 'is_simulated': False},
            'images': [{'image_id': record['image_id'], 'image_sha256': record['source']['image_sha256'],
                        'width': 80, 'height': 64, 'split': 'train', 'proposals': [box]}]}


def test_read_only_audit_distribution_and_snapshot(pool):
    root, report = pool
    assert report['split_stats']['train']['images'] == 10
    assert report['split_stats']['train']['boxes'] == 10
    assert not report['structural_errors']
    assert report['snapshot'] == snapshot(root, load_dataset(root)[1])
    assert report['dataset_fingerprint'] == content_hash(report['snapshot'])


@pytest.mark.parametrize('mutation', ['mapping', 'split_path', 'path_escape'])
def test_reject_mapping_split_and_path_confusion(pool, mutation):
    root, _ = pool
    if mutation == 'path_escape':
        rows = read_json(root / 'manifest.json'); rows[0]['image_file'] = '../secret.jpg'; write_json(root / 'manifest.json', rows)
    else:
        data = yaml.safe_load((root / 'dataset.yaml').read_text())
        if mutation == 'mapping': data['names'][0] = 'plastic'
        else: data['train'] = 'images/val'
        (root / 'dataset.yaml').write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError): load_dataset(root)


def test_detect_exact_leakage_without_modifying_splits(pool, tmp_path):
    root, _ = pool; rows = read_json(root / 'manifest.json')
    target = root / rows[-1]['image_file']; target.write_bytes((root / rows[0]['image_file']).read_bytes())
    rows[-1]['sha256'] = rows[-1]['source_sha256'] = sha256(target); write_json(root / 'manifest.json', rows)
    audit = audit_dataset(root, tmp_path / 'empty')
    assert audit['confirmed_identity_cross_split']
    assert rows[0]['sha256'][:20] in audit['blocked_train_ids']
    assert all(r['image_id'] != rows[0]['sha256'][:20] for r in select_pilot(audit, 4))


def test_geometry_anomalies_and_empty_labels_are_candidates(pool, tmp_path):
    root, _ = pool; rows = read_json(root / 'manifest.json')
    (root / rows[0]['label_file']).write_text('0 .5 .5 .001 .001\n1 .5 .5 .001 .001\n')
    (root / rows[1]['label_file']).write_text('')
    report = audit_dataset(root, tmp_path / 'empty')
    assert report['flag_counts']['tiny_boxes'] == 1
    assert report['flag_counts']['overlapping_duplicate_or_conflicting_boxes'] == 1
    assert report['flag_counts']['unverified_negative'] == 1
    assert report['split_stats']['train']['negative_images'] == 1


def test_raw_source_hash_mismatch_is_reported_and_excluded(pool, tmp_path):
    root, _ = pool; rows = read_json(root / 'manifest.json')
    rows[0]['source'] = 'hf_taco'; rows[0]['provenance'] = {'batch': 'fixture', 'original_file': 'fixture/test.jpg'}
    write_json(root / 'manifest.json', rows)
    raw = tmp_path / 'raw-repo/data/raw/hf_multisource/taco_0.jpg'
    raw.parent.mkdir(parents=True); raw.write_bytes(b'UNIT_FIXTURE_RAW_HASH_MISMATCH')
    audit = audit_dataset(root, tmp_path / 'raw-repo')
    assert rows[0]['image_file'] in audit['source_annotation_checks']['raw_image_hash_checks']['mismatches']
    assert rows[0]['sha256'][:20] not in [i['image_id'] for i in select_pilot(audit, 4)]


def test_pilot_is_reproducible_train_only_and_no_teacher(pool, workspace):
    root, audit = pool
    assert [i['image_id'] for i in select_pilot(audit, 4)] == [i['image_id'] for i in select_pilot(audit, 4)]
    for key in read_json(workspace / 'workspace.json')['image_ids']:
        record = get_item(workspace, key)
        assert record['source']['split'] == 'train'
        assert record['status'] == 'pending' and record['teacher'] is None
        assert all(box['confidence'] is None for box in record['original_boxes'])
    with pytest.raises(ValueError): prepare_pilot(audit, workspace, 4)
    with pytest.raises(ValueError): prepare_pilot(audit, root / 'nested-review', 4)


def test_pending_export_does_not_create_training_output(workspace, tmp_path):
    output = tmp_path / 'export'
    with pytest.raises(ValueError, match='Chưa có nhãn'): export_approved(workspace, output)
    assert not output.exists()


def test_teacher_import_preserves_draft_and_never_approves(workspace):
    record = first(workspace); bundle = teacher_bundle(record)
    import_teacher(workspace, bundle); updated = first(workspace)
    assert updated['status'] == 'pending' and updated['approval_signature'] is None
    assert updated['draft_boxes'] == record['draft_boxes']
    assert updated['teacher']['comparison']['proposals'][0]['iou'] == 1.
    with pytest.raises(ValueError, match='đã import'): import_teacher(workspace, bundle)


@pytest.mark.parametrize('mutation', ['version', 'simulation', 'image_hash', 'split', 'dimensions', 'box', 'score', 'mapping'])
def test_teacher_rejects_bad_provenance_and_geometry(workspace, mutation):
    record = first(workspace); bundle = teacher_bundle(record)
    entry = bundle['images'][0]; proposal = entry['proposals'][0]
    if mutation == 'version': del bundle['teacher']['version']
    elif mutation == 'simulation': bundle['teacher']['is_simulated'] = True
    elif mutation == 'image_hash': entry['image_sha256'] = 'f' * 64
    elif mutation == 'split': entry['split'] = 'val'
    elif mutation == 'dimensions': entry['width'] = 640
    elif mutation == 'box': proposal['box'] = [-1, 0, 20, 20]
    elif mutation == 'score': proposal['confidence'] = float('nan')
    elif mutation == 'mapping': proposal['class_name'] = 'battery'
    with pytest.raises(ValueError): import_teacher(workspace, bundle)
    assert first(workspace)['revision'] == 0


def test_unknown_teacher_class_stays_unknown_and_requires_human_mapping(workspace):
    record = first(workspace); bundle = teacher_bundle(record)
    proposal = bundle['images'][0]['proposals'][0]
    proposal.update(class_id=None, class_name=None, raw_label='unmapped object')
    import_teacher(workspace, bundle); record = first(workspace)
    assert record['teacher']['proposals'][0]['class_id'] is None
    saved = save_review(workspace, record['image_id'], approval(record, status='hold', boxes=[{**proposal, 'origin': 'teacher'}]))
    assert saved['draft_boxes'][0]['class_id'] is None and saved['draft_boxes'][0]['raw_label'] == 'unmapped object'
    record = first(workspace)
    bad = approval(record, boxes=[{**proposal, 'origin': 'teacher'}])
    with pytest.raises(ValueError, match='Class/box'): save_review(workspace, record['image_id'], bad)


def test_api_import_needs_approval_and_real_response_receipt_not_fabricated_weights(workspace):
    bundle = teacher_bundle(first(workspace)); teacher = bundle['teacher']
    teacher['execution_mode'] = 'external_api_import'; teacher.pop('artifact_sha256')
    bundle['raw_response'] = {'fixture': 'UNIT_TEST_ONLY_NO_API_CALLED'}
    teacher['response_artifact_sha256'] = content_hash(bundle['raw_response'])
    with pytest.raises(ValueError, match='receipt'): import_teacher(workspace, bundle)
    teacher.update(authorization_reference='UNIT_TEST_APPROVAL_RECEIPT_ONLY', request_id='unit-request')
    import_teacher(workspace, bundle)
    metadata = first(workspace)['teacher']['metadata']
    assert 'artifact_sha256' not in metadata
    assert metadata['response_artifact_sha256'] == content_hash(bundle['raw_response'])


def test_qa_distinguishes_class_disagreement_and_potential_miss():
    source = [{'box': [0, 0, 10, 10], 'class_id': 0}]
    result = compare_proposals(source, [{'proposal_id': 'a', 'class_id': 1, 'box': [0, 0, 10, 10]},
                                        {'proposal_id': 'b', 'class_id': None, 'box': [30, 30, 40, 40]}])
    assert [x['flag'] for x in result['proposals']] == ['suspected_class_disagreement', 'potential_missed_object']


@pytest.mark.parametrize('bad', [{'checks': {}}, {'rights_evidence': None}, {'reviewer': None},
                               {'issues': [{'type': 'missing_object', 'resolved': False}]},
                               {'issues': [{'type': 'bad_box', 'resolved': True, 'note': ''}]},
                               {'checks': {'all_objects_reviewed': 'true'}}])
def test_approval_requires_human_checks_rights_and_issue_resolution(workspace, bad):
    record = first(workspace)
    with pytest.raises(ValueError): save_review(workspace, record['image_id'], approval(record, **bad))
    assert first(workspace)['status'] == 'pending'


def test_revision_conflict_preserves_current_review(workspace):
    record = first(workspace); payload = approval(record, status='draft')
    save_review(workspace, record['image_id'], payload)
    with pytest.raises(ValueError, match='Revision'): save_review(workspace, record['image_id'], payload)
    assert first(workspace)['revision'] == 1


def test_corrected_original_is_traced_as_human_without_mutating_original(workspace):
    record = first(workspace); payload = approval(record, status='draft')
    payload['boxes'][0]['class_id'] = (payload['boxes'][0]['class_id'] + 1) % 9
    result = save_review(workspace, record['image_id'], payload)
    assert result['draft_boxes'][0]['origin'] == 'human'
    assert result['original_boxes'] == record['original_boxes']


def test_export_only_approved_and_readonly_val_test(workspace, tmp_path):
    record = first(workspace); save_review(workspace, record['image_id'], approval(record))
    output = tmp_path / 'approved-export'
    result = export_approved(workspace, output)
    assert result['approved_train_images'] == 1
    assert len(list((output / 'labels/train').glob('*.txt'))) == 1
    assert not (output / 'images/val').exists() and not (output / 'labels/test').exists()
    data = yaml.safe_load((output / 'dataset.yaml').read_text())
    assert data['names'] == dict(enumerate(CLASSES))
    assert data['val'].endswith('/images/val')
    manifest = read_json(output / 'export_manifest.json')
    assert manifest['items'][0]['review']['reviewer'].startswith('UNIT_TEST')
    assert manifest['items'][0]['teacher'] is None


def test_modified_approved_draft_is_blocked(workspace, tmp_path):
    record = first(workspace); approved = save_review(workspace, record['image_id'], approval(record))
    approved['draft_boxes'][0]['class_id'] = (approved['draft_boxes'][0]['class_id'] + 1) % 9
    write_json(item_path(workspace, record['image_id']), approved)
    with pytest.raises(ValueError, match='sau duyệt'): export_approved(workspace, tmp_path / 'export')


def test_mid_export_source_change_quarantines_output_without_active_yaml(pool, workspace, tmp_path, monkeypatch):
    from scripts.teacher_student import review
    root, _ = pool; record = first(workspace); save_review(workspace, record['image_id'], approval(record))
    original_copy = review.shutil.copyfile
    def changed_during_copy(source, target):
        original_copy(source, target)
        (root / 'labels/val/fixture_10.txt').write_text('')
    monkeypatch.setattr(review.shutil, 'copyfile', changed_during_copy)
    output = tmp_path / 'incomplete-export'
    with pytest.raises(ValueError, match='trong export'): export_approved(workspace, output)
    assert not (output / 'dataset.yaml').exists()
    assert read_json(output / 'EXPORT_INCOMPLETE.json')['training_allowed'] is False


def test_import_revokes_previous_approval(workspace, tmp_path):
    record = first(workspace); approved = save_review(workspace, record['image_id'], approval(record))
    import_teacher(workspace, teacher_bundle(approved))
    assert first(workspace)['approval_signature'] is None
    with pytest.raises(ValueError, match='Chưa có nhãn'): export_approved(workspace, tmp_path / 'export')


@pytest.mark.parametrize('change', ['label', 'image', 'new_eval_image'])
def test_source_or_split_changes_block_export(pool, workspace, tmp_path, change):
    root, _ = pool; record = first(workspace); save_review(workspace, record['image_id'], approval(record))
    if change == 'label': (root / 'labels/val/fixture_10.txt').write_text('')
    elif change == 'image': (root / record['source']['image_file']).write_bytes(b'changed')
    else: (root / 'images/val/new.jpg').write_bytes((root / 'images/val/fixture_10.jpg').read_bytes())
    with pytest.raises(ValueError): export_approved(workspace, tmp_path / 'export')


def test_teacher_archive_and_original_boxes_are_immutable(workspace):
    record = first(workspace); result = import_teacher(workspace, teacher_bundle(record))
    path = workspace / 'teacher_runs' / (result['bundle_sha256'] + '.json')
    data = read_json(path); data['teacher']['version'] = 'changed'; write_json(path, data)
    with pytest.raises(ValueError, match='Teacher provenance'): first(workspace)


def test_source_provenance_tamper_is_blocked(workspace):
    record = first(workspace); record['source']['provenance']['file'] = 'invented_source.jpg'
    write_json(item_path(workspace, record['image_id']), record)
    with pytest.raises(ValueError, match='Provenance'): first(workspace)


def test_test_workspace_cannot_approve_or_export(pool, tmp_path):
    _, audit = pool; workspace = tmp_path / 'test-review'; prepare_pilot(audit, workspace, 2, test_only=True)
    record = first(workspace)
    save_review(workspace, record['image_id'], approval(record, status='draft'))
    with pytest.raises(ValueError, match='kiểm thử'): save_review(workspace, record['image_id'], approval(first(workspace)))
    with pytest.raises(ValueError, match='kiểm thử'): export_approved(workspace, tmp_path / 'export')


def test_api_token_origin_and_pilot_allowlist(workspace):
    with TestClient(create_app(workspace), base_url='http://127.0.0.1') as client:
        record = first(workspace); url = f"/api/items/{record['image_id']}/review"
        assert client.get('/').status_code == 200
        assert client.get('/api/image/' + record['image_id']).status_code == 200
        assert client.get('/api/image/not-in-pilot').status_code == 400
        assert client.post(url, json=approval(record)).status_code == 403
        token = client.get('/api/session').json()['csrf_token']
        headers = {'X-Review-Token': token, 'Origin': 'https://outside.example'}
        assert client.post(url, json=approval(record), headers=headers).status_code == 403
        headers['Origin'] = 'http://127.0.0.1'
        assert client.post(url, json=approval(record, status='draft'), headers=headers).status_code == 200
        assert client.post(url, json=approval(record), headers=headers).status_code == 409


def test_teacher_missing_artifact_never_creates_output_or_loads_model(workspace, tmp_path):
    from scripts.teacher_student.local_teacher import propose_local
    output = tmp_path / 'no-inference.json'
    with pytest.raises(ValueError, match='không tự tải'): propose_local(workspace, tmp_path / 'missing', output)
    assert not output.exists()


def test_offline_guard_blocks_network_and_restores_environment(monkeypatch):
    import os, socket
    from scripts.teacher_student.local_teacher import offline_guard
    monkeypatch.setenv('HF_HUB_OFFLINE', 'prior-value')
    original_connect = socket.socket.connect
    with offline_guard():
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        with pytest.raises(RuntimeError, match='network'): socket.create_connection(('no-network.invalid', 443))
        with socket.socket() as connection:
            with pytest.raises(RuntimeError, match='network'): connection.connect(('127.0.0.1', 1))
    assert os.environ['HF_HUB_OFFLINE'] == 'prior-value' and socket.socket.connect is original_connect


def test_teacher_phrase_projection_keeps_unknown_and_raw_clipped_geometry():
    from scripts.teacher_student.local_teacher import normalize_proposals
    proposals, rejected = normalize_proposals([[-2, 1, 8, 8],[20,20,30,30],[200,200,210,210]], [.7,.6,.5], ['bottle','unmapped battery phrase','can'],100,100)
    assert proposals[0]['box'] == [0,1,8,8] and proposals[0]['raw_box'][0] == -2
    assert proposals[0]['box_adjustment'] == 'clipped_to_image'
    assert proposals[1]['class_id'] is None and proposals[1]['raw_label'] == 'unmapped battery phrase'
    assert len(rejected) == 1


def test_teacher_artifact_pin_validation_and_no_pickle(tmp_path, monkeypatch):
    from scripts.teacher_student import local_teacher
    artifact = tmp_path / 'teacher'; artifact.mkdir()
    for name, data in [('config.json', {'model_type':'grounding-dino'}), ('preprocessor_config.json',{}), ('tokenizer_config.json',{}), ('tokenizer.json',{})]:
        write_json(artifact / name, data)
    (artifact / 'model.safetensors').write_bytes(b'UNIT_FIXTURE_NOT_A_MODEL')
    files = {p.name:sha256(p) for p in artifact.iterdir()}
    receipt = {'model_id':local_teacher.MODEL_ID,'revision':local_teacher.REVISION,'authorization_reference':'UNIT_FIXTURE_ONLY','files':files}
    write_json(artifact / 'teacher_artifact.json', receipt)
    with pytest.raises(ValueError, match='Student/random'): local_teacher.validate_artifact(artifact)
    monkeypatch.setattr(local_teacher, 'WEIGHTS_SHA256', files['model.safetensors'])
    assert local_teacher.validate_artifact(artifact)[1]['revision'] == local_teacher.REVISION
    (artifact / 'untrusted.py').write_text('not executed')
    with pytest.raises(ValueError, match='ngoài manifest'): local_teacher.validate_artifact(artifact)


def test_teacher_normalization_rejects_nan_and_length_mismatch():
    from scripts.teacher_student.local_teacher import normalize_proposals
    with pytest.raises(ValueError): normalize_proposals([[0,0,5,5]],[float('nan')],['can'],10,10)
    with pytest.raises(ValueError): normalize_proposals([[0,0,5,5]],[],[],10,10)
