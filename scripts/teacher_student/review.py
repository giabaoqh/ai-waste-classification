"""Versioned human review and approved-only export; never changes the source pool."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import math
import os
from pathlib import Path
import secrets
import shutil

import yaml
from PIL import Image

from .common import CLASSES, contained, content_hash, iou, load_dataset, read_json, read_yolo, sha256, snapshot, valid_box, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def workspace_lock(workspace):
    lock = Path(workspace) / '.write.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError('Workspace đang được ghi; thử lại. Không tự xóa lock của process khác.') from exc
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def config(workspace):
    data = read_json(Path(workspace) / 'workspace.json')
    if data['schema_version'] != 1 or data['taxonomy'] != CLASSES:
        raise ValueError('Workspace schema/taxonomy không phù hợp')
    return data


def item_path(workspace, image_id):
    cfg = config(workspace)
    if image_id not in cfg['image_ids']:
        raise ValueError('Ảnh không thuộc pilot training pool')
    return contained(Path(workspace) / 'items', image_id + '.json')


def verify_item(workspace, record):
    cfg = config(workspace)
    pinned = read_json(Path(workspace) / 'source_snapshot.json')
    if content_hash(pinned) != cfg['dataset_fingerprint']:
        raise ValueError('Source snapshot bị sửa')
    source = record['source']
    if source['split'] != 'train' or Path(source['image_file']).parts[:2] != ('images', 'train') or Path(source['label_file']).parts[:2] != ('labels', 'train'):
        raise ValueError('Chỉ cho phép ảnh/nhãn training pool')
    root = Path(cfg['dataset_root'])
    if sha256(root / 'manifest.json') != pinned.get('manifest.json'):
        raise ValueError('Manifest nguồn đã thay đổi')
    manifest_entry = next((row for row in read_json(root / 'manifest.json') if row['image_file'] == source['image_file']), None)
    if not manifest_entry or any((source.get('original_image_id') if key == 'image_id' else source.get(key)) != value for key, value in manifest_entry.items()):
        raise ValueError('Provenance trong review không khớp manifest nguồn')
    for field, digest in [('image_file', 'image_sha256'), ('label_file', 'label_sha256')]:
        if pinned.get(source[field]) != source[digest] or sha256(contained(root, source[field])) != source[digest]:
            raise ValueError('Ảnh/nhãn nguồn đã đổi hash; không tiếp tục review/export')
    if record['image_id'] != source['image_sha256'][:20]:
        raise ValueError('Image ID/hash không khớp')
    with Image.open(contained(root, source['image_file'])) as image:
        if list(image.size) != [source['width'], source['height']] or image.getexif().get(274, 1) != 1:
            raise ValueError('Kích thước/hướng ảnh nguồn không khớp')
    actual_original = read_yolo(contained(root, source['label_file']), source['width'], source['height'])
    if actual_original != record['original_boxes']:
        raise ValueError('Box gốc trong review đã bị sửa; chỉ được sửa draft')
    if record.get('teacher'):
        teacher = record['teacher']
        bundle = read_json(contained(Path(workspace) / 'teacher_runs', teacher['bundle_sha256'] + '.json'))
        entry = next((i for i in bundle['images'] if i['image_id'] == record['image_id']), None)
        if content_hash(bundle) != teacher['bundle_sha256'] or bundle['teacher'] != teacher['metadata'] or not entry or entry['proposals'] != teacher['proposals']:
            raise ValueError('Teacher provenance/output đã bị sửa')
    return cfg, contained(root, source['image_file'])


def get_item(workspace, image_id):
    record = read_json(item_path(workspace, image_id))
    verify_item(workspace, record)
    return record


def select_pilot(audit, count=150, seed=42):
    blocked = set(audit['blocked_train_ids']) | set(audit['near_cross_split_train_ids'])
    candidates = [i for i in audit['items'] if i['split'] == 'train' and i['image_id'] not in blocked and
                  not set(i['flags']).intersection({'manifest_image_hash_mismatch', 'unexpected_exif_on_processed_image', 'source_image_hash_mismatch'})]
    if len(candidates) < count or not 1 <= count <= 200:
        raise ValueError('Không đủ ảnh train độc lập/hợp lệ cho pilot')
    target = {'hf_taco': round(count * .5333), 'hf_street_negative': round(count * .1)}
    target['kaggle_warp'] = count - sum(target.values())
    quotas = {s: min(n, sum(i['source'] == s for i in candidates)) for s, n in target.items()}
    selected, classes, sources, groups, identities = [], Counter(), Counter(), Counter(), set()
    selected_ids = set()
    tie_scores = {i['image_id']: int(content_hash([seed, i['image_sha256']])[:8], 16) / 2**32 * .01 for i in candidates}

    def eligible(item, quota=True):
        keys = {item['image_sha256'], item['pixel_sha256'], item.get('source_sha256')}
        keys.discard(None)
        return not identities.intersection(keys) and item['image_id'] not in selected_ids and (not quota or sources[item['source']] < quotas.get(item['source'], count))

    def score(item):
        return (sum(1 / (1 + classes[name]) for name in item['class_counts']) +
                min(item['box_count'], 40) / 40 + min(len(item['tiny_box_indices']), 10) * .03 +
                .5 / (1 + groups[item['source_group']]) +
                tie_scores[item['image_id']])

    def add(item):
        selected.append(item)
        selected_ids.add(item['image_id'])
        classes.update(item['class_counts'].keys())
        sources[item['source']] += 1
        groups[item['source_group']] += 1
        identities.update(v for v in (item['image_sha256'], item['pixel_sha256'], item.get('source_sha256')) if v)

    frequency = audit['split_stats']['train']['images_per_class']
    # Cover rare classes first; then source quotas, density/tiny boxes and scene diversity.
    for _ in range(min(8, count)):
        for name in sorted(CLASSES, key=lambda c: frequency[c]):
            if len(selected) >= count:
                break
            pool = [i for i in candidates if name in i['class_counts'] and eligible(i)]
            if classes[name] < min(8, frequency[name]) and pool:
                add(max(pool, key=score))
    for source in sorted(quotas):
        while sources[source] < quotas[source] and len(selected) < count:
            pool = [i for i in candidates if i['source'] == source and eligible(i)]
            if not pool:
                break
            add(max(pool, key=score))
    while len(selected) < count:
        pool = [i for i in candidates if eligible(i, quota=False)]
        if not pool:
            raise ValueError('Không đủ ảnh không trùng sau kiểm identity')
        add(max(pool, key=score))
    return selected


def prepare_pilot(audit, workspace, count=150, seed=42, test_only=False):
    workspace = Path(workspace).resolve()
    root = Path(audit['dataset_root']).resolve()
    if workspace.exists() or workspace.is_relative_to(root) or root.is_relative_to(workspace):
        raise ValueError('Chọn workspace mới, tách khỏi dataset nguồn')
    _, rows = load_dataset(root)
    if snapshot(root, rows) != audit['snapshot']:
        raise ValueError('Audit đã cũ; dữ liệu nguồn đổi sau audit')
    selected = select_pilot(audit, count, seed)
    if len({r['image_id'] for r in selected}) != count:
        raise ValueError('Image ID collision')
    workspace.mkdir(parents=True)
    cfg = {'schema_version': 1, 'created_at': now(), 'dataset_root': str(root),
           'dataset_fingerprint': audit['dataset_fingerprint'], 'taxonomy': CLASSES, 'seed': seed,
           'image_ids': [i['image_id'] for i in selected], 'test_only': test_only,
           'teacher_execution': 'not_run', 'selection': 'train-only; rare classes, source quotas, density/tiny boxes, scene diversity; no inferred errors'}
    write_json(workspace / 'workspace.json', cfg)
    write_json(workspace / 'source_snapshot.json', audit['snapshot'])
    (workspace / 'review.key').write_text(secrets.token_hex(32))
    for source in selected:
        boxes = read_yolo(contained(root, source['label_file']), source['width'], source['height'])
        record = {'schema_version': 1, 'image_id': source['image_id'], 'source': source,
                  'revision': 0, 'status': 'pending', 'original_boxes': boxes,
                  'teacher': None, 'draft_boxes': deepcopy(boxes), 'issues': [], 'review': None,
                  'approval_signature': None, 'history': [{'event': 'queued_train_image', 'at': now()}]}
        write_json(workspace / 'items' / (source['image_id'] + '.json'), record)
    summary = {'count': count, 'sources': dict(Counter(i['source'] for i in selected)),
               'class_image_counts': dict(Counter(name for i in selected for name in i['class_counts'])),
               'boxes': sum(i['box_count'] for i in selected), 'teacher_proposals': 0, 'approved': 0,
               'validation_or_test_selected': 0, 'test_only': test_only, 'as_of': 'queue_creation'}
    write_json(workspace / 'pilot_summary.json', summary)
    return summary


def compare_proposals(original, proposals):
    comparisons, matched = [], set()
    for proposal in proposals:
        scores = [iou(proposal['box'], box['box']) for box in original]
        best = max(range(len(scores)), key=scores.__getitem__) if scores else None
        overlap = scores[best] if best is not None else 0.
        same = best is not None and proposal.get('class_id') == original[best]['class_id']
        if overlap >= .5:
            matched.add(best)
        flag = 'potential_missed_object' if overlap < .1 else 'suspected_class_disagreement' if overlap >= .5 and not same else 'box_difference' if overlap < .5 else 'agreement_candidate'
        comparisons.append({'proposal_id': proposal['proposal_id'], 'original_index': best,
                            'iou': overlap, 'same_class': same, 'flag': flag})
    return {'proposals': comparisons, 'original_unmatched_indices': [i for i in range(len(original)) if i not in matched],
            'note': 'Chỉ là đối chiếu QA; đồng ý hay bất đồng với Teacher không xác định ground truth.'}


def validate_draft(boxes, width, height, allow_unknown=False):
    if not isinstance(boxes, list) or len(boxes) > 1000:
        raise ValueError('Danh sách box không hợp lệ')
    result = []
    for box in boxes:
        cls = box.get('class_id')
        valid_class = (cls is None and allow_unknown) or (type(cls) is int and 0 <= cls < 9)
        if not valid_class or not valid_box(box.get('box'), width, height):
            raise ValueError('Class/box không hợp lệ; không tự sửa hoặc ép unknown sang other_litter')
        origin = box.get('origin', 'human')
        if origin not in ('original', 'teacher', 'human'):
            raise ValueError('Nguồn box không hợp lệ')
        result.append({'class_id': cls, 'class_name': CLASSES[cls] if cls is not None else None, 'box': list(box['box']),
                       'confidence': None, 'origin': origin, 'proposal_id': box.get('proposal_id'),
                       'original_index': box.get('original_index'), 'raw_label': str(box.get('raw_label', ''))[:1000]})
    return result


def signature(workspace, record):
    payload = {k: v for k, v in record.items() if k not in ('approval_signature', 'history')}
    key = (Path(workspace) / 'review.key').read_text().strip().encode()
    return hmac.new(key, content_hash(payload).encode(), hashlib.sha256).hexdigest()


def save_review(workspace, image_id, payload):
    with workspace_lock(workspace):
        record = get_item(workspace, image_id)
        if type(payload.get('expected_revision')) is not int or payload['expected_revision'] != record['revision']:
            raise ValueError('Revision đã thay đổi; tải lại trước khi lưu')
        source = record['source']
        state = payload.get('status')
        if state not in ('draft', 'approved', 'hold', 'rejected'):
            raise ValueError('Trạng thái duyệt không hợp lệ')
        boxes = validate_draft(payload.get('boxes'), source['width'], source['height'], allow_unknown=state != 'approved')
        proposals = {i['proposal_id']: i for i in (record.get('teacher') or {}).get('proposals', [])}
        for box in boxes:
            original_index, proposal_id = box.get('original_index'), box.get('proposal_id')
            if original_index is not None and (type(original_index) is not int or not 0 <= original_index < len(record['original_boxes'])):
                raise ValueError('Original box reference không hợp lệ')
            if proposal_id is not None and proposal_id not in proposals:
                raise ValueError('Teacher proposal reference không có provenance')
            if box['origin'] == 'original':
                basis = record['original_boxes'][original_index] if original_index is not None else None
            elif box['origin'] == 'teacher':
                basis = proposals.get(proposal_id)
                if basis is None:
                    raise ValueError('Không được giả box human thành output Teacher')
            else:
                basis = None
            if basis is None or basis['class_id'] != box['class_id'] or basis['box'] != box['box']:
                box['origin'] = 'human'
        if not isinstance(payload.get('reviewer'), str):
            raise ValueError('Cần tên/mã người kiểm duyệt dạng text')
        reviewer = payload['reviewer'].strip()
        if not reviewer or len(reviewer) > 100:
            raise ValueError('Cần tên/mã người kiểm duyệt')
        issues = payload.get('issues', [])
        if not isinstance(issues, list) or any(not isinstance(i, dict) or not isinstance(i.get('type'), str) or type(i.get('resolved')) is not bool for i in issues):
            raise ValueError('Issue cần type và resolved rõ ràng')
        if any(i['resolved'] and not str(i.get('note', '')).strip() for i in issues):
            raise ValueError('Issue đã giải quyết cần ghi căn cứ')
        checks = payload.get('checks', {})
        if not isinstance(checks, dict) or any(type(v) is not bool for v in checks.values()):
            raise ValueError('Checklist cần giá trị boolean rõ ràng')
        required = ['all_objects_reviewed', 'heuristics_reviewed', 'provenance_reviewed', 'hazards_reviewed']
        if not isinstance(payload.get('rights_evidence', ''), str):
            raise ValueError('Căn cứ quyền sử dụng cần text/link cụ thể')
        rights = payload.get('rights_evidence', '').strip()
        if state == 'approved':
            if config(workspace).get('test_only'):
                raise ValueError('Workspace kiểm thử không được duyệt nhãn training')
            if any(checks.get(key) is not True for key in required) or not rights:
                raise ValueError('Chưa kiểm toàn ảnh/QA/provenance/nguy cơ hoặc thiếu bằng chứng quyền sử dụng')
            if any(not issue['resolved'] for issue in issues):
                raise ValueError('Còn issue chưa giải quyết; chuyển hold thay vì duyệt')
        record.update(draft_boxes=boxes, issues=issues, status=state, revision=record['revision'] + 1,
                      review={'reviewer': reviewer, 'at': now(), 'checks': checks,
                              'rights_evidence': rights, 'notes': str(payload.get('notes', ''))[:10000]})
        record['approval_signature'] = signature(workspace, record) if state == 'approved' else None
        record['history'].append({'event': 'human_review_' + state, 'revision': record['revision'], 'reviewer': reviewer, 'at': now()})
        write_json(item_path(workspace, image_id), record)
        return record


def import_teacher(workspace, bundle):
    """Ingest producer output only. Import does not claim an inference was run here."""
    if bundle.get('schema_version') != 1:
        raise ValueError('Teacher schema version không hợp lệ')
    teacher = bundle.get('teacher', {})
    for field in ('model_id', 'version', 'run_id', 'generated_at', 'producer', 'adapter_version', 'prompt_sha256'):
        if not isinstance(teacher.get(field), str) or not teacher[field].strip():
            raise ValueError('Thiếu provenance Teacher: ' + field)
    mode = teacher.get('execution_mode')
    if mode not in ('local', 'external_api_import'):
        raise ValueError('Cần execution_mode local hoặc external_api_import; tool không gọi API')
    digest_key = 'artifact_sha256' if mode == 'local' else 'response_artifact_sha256'
    for key in ('prompt_sha256', digest_key):
        if not isinstance(teacher.get(key), str) or len(teacher[key]) != 64 or any(c not in '0123456789abcdef' for c in teacher[key]):
            raise ValueError('Teacher hash không hợp lệ')
    if mode == 'external_api_import':
        if not teacher.get('authorization_reference') or not teacher.get('request_id') or not isinstance(bundle.get('raw_response'), dict):
            raise ValueError('Output API cần receipt phê duyệt, request ID và raw_response; không bịa hash weights kín')
        if content_hash(bundle['raw_response']) != teacher['response_artifact_sha256']:
            raise ValueError('Raw response API không khớp hash')
    if teacher.get('coordinate_space') != 'xyxy_pixels_exif_oriented' or teacher.get('kind') not in ('grounding_dino', 'vlm', 'sam2_assisted'):
        raise ValueError('Teacher phải khai báo loại và tọa độ pixel sau EXIF')
    if teacher.get('is_simulated') is not False:
        raise ValueError('Không nhận kết quả giả/mô phỏng vào workspace dữ liệu')
    entries = bundle.get('images')
    if not isinstance(entries, list) or not entries or len({i['image_id'] for i in entries}) != len(entries):
        raise ValueError('Teacher batch rỗng hoặc image ID lặp')
    with workspace_lock(workspace):
        changes = []
        for entry in entries:
            record = get_item(workspace, entry['image_id'])
            source = record['source']
            if entry.get('split') != 'train' or entry.get('image_sha256') != source['image_sha256'] or [entry.get('width'), entry.get('height')] != [source['width'], source['height']]:
                raise ValueError('Teacher hash/split/kích thước không khớp training image')
            proposals = entry.get('proposals')
            if not isinstance(proposals, list) or len(proposals) > 1000:
                raise ValueError('Teacher proposals không hợp lệ')
            seen = set()
            for proposal in proposals:
                cls = proposal.get('class_id')
                score = proposal.get('confidence')
                if cls is not None and (type(cls) is not int or not 0 <= cls < 9 or proposal.get('class_name') != CLASSES[cls]):
                    raise ValueError('Teacher class mapping sai; giữ raw_label và class_id=null nếu chưa ánh xạ')
                if not valid_box(proposal.get('box'), source['width'], source['height']):
                    raise ValueError('Teacher box sai hoặc ngoài ảnh')
                if score is not None and (type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError('Teacher score không hợp lệ')
                if proposal.get('confidence_kind') not in ('detector_score', 'self_reported_uncalibrated', 'not_available') or (score is None) != (proposal.get('confidence_kind') == 'not_available'):
                    raise ValueError('Cần phân biệt detector score/VLM self-report/không có score')
                if not proposal.get('raw_label') or not proposal.get('proposal_id') or proposal['proposal_id'] in seen:
                    raise ValueError('Thiếu/lặp proposal ID hoặc raw label')
                seen.add(proposal['proposal_id'])
            changes.append((record, proposals))
        bundle_hash = content_hash(bundle)
        # Store immutable input alongside item event history for provenance.
        destination = Path(workspace) / 'teacher_runs' / (bundle_hash + '.json')
        if destination.exists():
            raise ValueError('Teacher bundle này đã import; không tạo revision lặp')
        write_json(destination, bundle)
        for record, proposals in changes:
            record['teacher'] = {'metadata': teacher, 'bundle_sha256': bundle_hash,
                                 'proposals': proposals, 'comparison': compare_proposals(record['original_boxes'], proposals)}
            record['revision'] += 1
            record['status'] = 'pending'
            record['review'] = None
            record['approval_signature'] = None
            # Human draft is preserved, never replaced or approved by AI import.
            record['history'].append({'event': 'teacher_output_imported_not_approved', 'at': now(),
                                      'bundle_sha256': bundle_hash, 'revision': record['revision']})
            write_json(item_path(workspace, record['image_id']), record)
        return {'imported': len(changes), 'bundle_sha256': bundle_hash, 'approved': 0}


def export_approved(workspace, output):
    output = Path(output).resolve()
    cfg = config(workspace)
    dataset = Path(cfg['dataset_root']).resolve()
    if output.exists() or output.is_relative_to(dataset) or dataset.is_relative_to(output) or output.is_relative_to(Path(workspace).resolve()):
        raise ValueError('Export phải là thư mục mới, tách khỏi source/review workspace')
    if cfg.get('test_only'):
        raise ValueError('Workspace kiểm thử không được export training')
    with workspace_lock(workspace):
        root, rows = load_dataset(dataset)
        frozen = read_json(Path(workspace) / 'source_snapshot.json')
        if snapshot(root, rows) != frozen or content_hash(frozen) != cfg['dataset_fingerprint']:
            raise ValueError('Dataset/split nguồn đổi; export bị chặn')
        approved = []
        for image_id in cfg['image_ids']:
            record = get_item(workspace, image_id)
            if record['status'] != 'approved':
                continue
            if not record.get('approval_signature') or not hmac.compare_digest(signature(workspace, record), record['approval_signature']):
                raise ValueError('Nhãn sau duyệt bị sửa hoặc thiếu dấu duyệt; cần review lại')
            if not record.get('review') or any(not i['resolved'] for i in record['issues']):
                raise ValueError('Approval chưa đầy đủ')
            validate_draft(record['draft_boxes'], record['source']['width'], record['source']['height'])
            approved.append(record)
        if not approved:
            raise ValueError('Chưa có nhãn đã duyệt; không tạo dataset training')
        output.mkdir(parents=True)
        exported = []
        try:
            for record in approved:
                source = record['source']
                image_id = record['image_id']
                image = output / 'images/train' / (image_id + Path(source['image_file']).suffix.lower())
                image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(contained(root, source['image_file']), image)
                label = output / 'labels/train' / (image_id + '.txt')
                label.parent.mkdir(parents=True, exist_ok=True)
                lines = []
                for box in record['draft_boxes']:
                    x1, y1, x2, y2 = box['box']
                    values = [(x1 + x2) / 2 / source['width'], (y1 + y2) / 2 / source['height'],
                              (x2 - x1) / source['width'], (y2 - y1) / source['height']]
                    lines.append(str(box['class_id']) + ' ' + ' '.join(f'{v:.10f}' for v in values))
                label.write_text('\n'.join(lines) + ('\n' if lines else ''))
                exported.append({'image_id': image_id, 'split': 'train', 'image_sha256': sha256(image),
                                 'label_sha256': sha256(label), 'original_label_sha256': source['label_sha256'],
                                 'source': source, 'review': record['review'], 'revision': record['revision'],
                                 'teacher': record['teacher'], 'approval_signature': record['approval_signature']})
            data = {'path': output.as_posix(), 'train': 'images/train',
                    'val': (root / 'images/val').as_posix(), 'test': (root / 'images/test').as_posix(),
                    'names': dict(enumerate(CLASSES))}
            if snapshot(root, rows) != frozen:
                raise ValueError('Source/split đổi trong export; không phát hành dataset.yaml')
            write_json(output / 'export_manifest.json', {'schema_version': 1, 'created_at': now(),
                       'source_dataset_fingerprint': cfg['dataset_fingerprint'], 'training_allowed': True,
                       'validation_test_policy': 'Read-only references to original frozen splits; not copied or relabeled.',
                       'approved_train_images': len(exported), 'items': exported})
            temporary_yaml = output / 'dataset.yaml.tmp'
            temporary_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
            temporary_yaml.replace(output / 'dataset.yaml')
        except Exception:
            # Keep evidence and explicitly quarantine a partial export, never delete user data.
            write_json(output / 'EXPORT_INCOMPLETE.json', {'training_allowed': False})
            raise
        return {'approved_train_images': len(exported), 'output': str(output)}
