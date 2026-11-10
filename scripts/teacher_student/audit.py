"""Read-only measurements; heuristics never establish semantic label correctness."""
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

from PIL import Image, ImageOps

from scripts.prepare_taco_pilot import category_mapping, normalized_box, ANNOTATIONS_SHA256
from .common import CLASSES, content_hash, contained, iou, load_dataset, read_json, read_yolo, sha256, snapshot


def perceptual_hash(image):
    pixels = list(image.convert('L').resize((9, 8)).get_flattened_data())
    return sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))


def source_group(row):
    info = row.get('provenance', {})
    if row['source'] in ('hf_taco', 'taco_reserved'):
        batch = info.get('batch') or row.get('source_batch')
        return 'taco:' + batch if batch else None
    if row['source'] == 'kaggle_warp' and info.get('capture_date'):
        return 'warp:' + info['capture_date']
    return None


def source_id(row):
    info = row.get('provenance', {})
    if row['source'] in ('hf_taco', 'taco_reserved'):
        return 'taco:' + str(info.get('original_file') or row.get('source_file'))
    return info.get('file') or info.get('url') or row.get('url')


def source_annotation_checks(root, rows, repo):
    """Compare only to pinned existing author files, not to invented ground truth."""
    result = {'checked': 0, 'mismatches': [], 'unavailable': [], 'taco_annotations_hash_matches_pin': None}
    raw_hashes = {'checked': 0, 'mismatches': [], 'unavailable': []}
    result['raw_image_hash_checks'] = raw_hashes
    taco_path = Path(repo) / 'data/raw/taco/annotations.json'
    taco, images, annotations, mapping = None, {}, defaultdict(list), {}
    if taco_path.is_file():
        taco = read_json(taco_path)
        result['taco_annotations_hash_matches_pin'] = sha256(taco_path) == ANNOTATIONS_SHA256
        images = {r['id']: r for r in taco['images']}
        mapping = category_mapping(taco['categories'])
        for ann in taco['annotations']:
            annotations[ann['image_id']].append(ann)
        result['taco_license_definitions'] = taco.get('licenses', [])
    archive = Path(repo) / 'data/raw/warp_kaggle/warp.zip'
    warp = zipfile.ZipFile(archive) if archive.is_file() else None
    if warp:
        names = warp.read('Warp-D/classes.txt').decode().splitlines()
        warp_map = {i: 0 if n.startswith(('bottle-', 'glass-')) else 1 if n == 'cans' else 3 if 'cardboard' in n else 8 for i, n in enumerate(names)}
        recorded_audit = read_json(root / 'audit.json') if (root / 'audit.json').is_file() else {}
        result['warp_archive_sha256'] = sha256(archive)
        result['warp_archive_hash_matches_build_audit'] = result['warp_archive_sha256'] == recorded_audit.get('warp_archive_sha256')
    try:
        for row in rows:
            expected = None
            raw_hash = None
            if row['source'] == 'kaggle_warp' and warp:
                raw_hash = __import__('hashlib').sha256(warp.read(row['provenance']['file'])).hexdigest()
            else:
                if row['source'] == 'taco_reserved':
                    raw_path = Path(repo) / 'data/raw/taco/thumbnails' / f"{row['image_id']:05d}.jpg"
                elif row['source'] == 'hf_taco':
                    number = Path(row['image_file']).stem.split('_')[-1]
                    raw_path = Path(repo) / 'data/raw/hf_multisource' / f'taco_{number}.jpg'
                elif row['source'] == 'hf_street_negative':
                    raw_path = Path(repo) / 'data/raw/hf_multisource' / Path(row['image_file']).name
                else:
                    raw_path = None
                if raw_path and raw_path.is_file():
                    raw_hash = sha256(raw_path)
            if raw_hash is None or not row.get('source_sha256'):
                raw_hashes['unavailable'].append(row['image_file'])
            else:
                raw_hashes['checked'] += 1
                if raw_hash != row['source_sha256']:
                    raw_hashes['mismatches'].append(row['image_file'])
            if row['source'] in ('hf_taco', 'taco_reserved') and taco is not None:
                image_id = row.get('image_id')
                if image_id is None:
                    image_id = int(Path(row['image_file']).stem.split('_')[-1])
                im = images[image_id]
                expected = [[mapping[a['category_id']], *normalized_box(a['bbox'], im['width'], im['height'])] for a in annotations[image_id]]
            elif row['source'] == 'kaggle_warp' and warp:
                path = row['provenance']['file'].replace('/images/', '/labels/').replace('.jpg', '.txt')
                expected = [[warp_map[int(v[0])], *map(float, v[1:])] for line in warp.read(path).decode().splitlines() if (v := line.split())]
            elif row['source'] == 'hf_street_negative':
                # Only checks transcription of an empty label, NOT absence of waste in pixels.
                expected = []
            else:
                result['unavailable'].append(row['image_file'])
                continue
            actual = [[float(x) for x in line.split()] for line in contained(root, row['label_file']).read_text().splitlines() if line.strip()]
            result['checked'] += 1
            if len(actual) != len(expected) or any(len(a) != 5 or any(abs(x - y) > 1e-5 for x, y in zip(a, b)) for a, b in zip(actual, expected)):
                result['mismatches'].append(row['image_file'])
    finally:
        if warp:
            warp.close()
    return result


def audit_dataset(dataset, repo):
    root, rows = load_dataset(dataset)
    before = snapshot(root, rows)
    stats, items = {}, []
    for split in ('train', 'val', 'test'):
        stats[split] = {'images': 0, 'boxes': 0, 'negative_images': 0, 'by_class': {c: 0 for c in CLASSES}, 'images_per_class': {c: 0 for c in CLASSES}}
    sources = defaultdict(lambda: {'images': 0, 'boxes': 0, 'by_class': {c: 0 for c in CLASSES}})
    issues, groups = [], defaultdict(set)
    buckets = {k: defaultdict(list) for k in ('image_sha256', 'pixel_sha256', 'source_sha256', 'source_id')}
    near = []
    for row in rows:
        image_path = contained(root, row['image_file'])
        flags = []
        file_hash = sha256(image_path)
        if file_hash != row.get('sha256'):
            flags.append('manifest_image_hash_mismatch')
        try:
            with Image.open(image_path) as original:
                orientation = original.getexif().get(274, 1)
                image = ImageOps.exif_transpose(original).convert('RGB')
                width, height = image.size
                pixel_hash = content_hash([width, height, __import__('hashlib').sha256(image.tobytes()).hexdigest()])
                dhash = perceptual_hash(image)
            if orientation != 1:
                flags.append('unexpected_exif_on_processed_image')
            boxes = read_yolo(contained(root, row['label_file']), width, height)
        except (OSError, ValueError) as exc:
            issues.append({'image_file': row['image_file'], 'type': 'invalid_image_or_label', 'detail': str(exc)})
            continue
        counts = Counter(box['class_name'] for box in boxes)
        small, extreme, overlaps = [], [], []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box['box']
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < 3 or bw * bh < 16:
                small.append(index)
            if max(bw / bh, bh / bw) > 20 or bw * bh > .9 * width * height:
                extreme.append(index)
            for other_index in range(index):
                score = iou(box['box'], boxes[other_index]['box'])
                if score >= .95:
                    overlaps.append({'a': other_index, 'b': index, 'iou': score,
                                     'same_class': box['class_id'] == boxes[other_index]['class_id']})
        if small:
            flags.append('tiny_boxes')
        if extreme:
            flags.append('extreme_boxes')
        if overlaps:
            flags.append('overlapping_duplicate_or_conflicting_boxes')
        if not boxes:
            flags.append('unverified_negative')
        info = row.get('provenance', {})
        license_value = next((v for v in (info.get('license_declared'), info.get('license'), row.get('source_license')) if v is not None and v != ''), None)
        if license_value is None:
            flags.append('missing_image_license_reference')
        group = source_group(row)
        if group:
            groups[group].add(row['split'])
        else:
            flags.append('scene_group_unknown')
        item = {**row, 'original_image_id': row.get('image_id'), 'image_id': file_hash[:20], 'width': width, 'height': height,
                'image_sha256': file_hash, 'label_sha256': sha256(contained(root, row['label_file'])),
                'pixel_sha256': pixel_hash, 'computed_dhash': f'{dhash:016x}', 'source_id': source_id(row),
                'source_group': group, 'license_reference': license_value, 'flags': flags,
                'class_counts': dict(counts), 'box_count': len(boxes), 'tiny_box_indices': small,
                'extreme_box_indices': extreme, 'overlap_candidates': overlaps}
        items.append(item)
        for field in buckets:
            if item.get(field):
                buckets[field][item[field]].append(item)
        st = stats[row['split']]
        st['images'] += 1
        st['boxes'] += len(boxes)
        st['negative_images'] += not boxes
        for name, count in counts.items():
            st['by_class'][name] += count
            st['images_per_class'][name] += 1
        source = sources[row['source'] + ':' + row['split']]
        source['images'] += 1
        source['boxes'] += len(boxes)
        for name, count in counts.items():
            source['by_class'][name] += count
    duplicates = {field: [[i['image_file'] for i in group] for group in values.values() if len(group) > 1] for field, values in buckets.items()}
    leakage = []
    blocked = set()
    for field, values in buckets.items():
        for group in values.values():
            if len({i['split'] for i in group}) > 1:
                leakage.append({'kind': field, 'images': [i['image_file'] for i in group]})
                blocked.update(i['image_id'] for i in group if i['split'] == 'train')
    group_leaks = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    for item in items:
        if item['source_group'] in group_leaks and item['split'] == 'train':
            blocked.add(item['image_id'])
    cross_near_train = set()
    for index, a in enumerate(items):
        ad = int(a['computed_dhash'], 16)
        for b in items[:index]:
            distance = (ad ^ int(b['computed_dhash'], 16)).bit_count()
            if distance <= 8 and a['image_sha256'] != b['image_sha256']:
                cross = a['split'] != b['split']
                near.append({'a': a['image_file'], 'b': b['image_file'], 'distance': distance, 'cross_split': cross})
                if cross:
                    cross_near_train.update(i['image_id'] for i in (a, b) if i['split'] == 'train')
    source_checks = source_annotation_checks(root, rows, repo)
    for item in items:
        if item['image_file'] in source_checks['mismatches']:
            item['flags'].append('source_annotation_mismatch')
        if item['image_file'] in source_checks['raw_image_hash_checks']['mismatches']:
            item['flags'].append('source_image_hash_mismatch')
    expected_images = {r['image_file'] for r in rows}
    expected_labels = {r['label_file'] for r in rows}
    orphan_images = sorted(p.relative_to(root).as_posix() for p in (root / 'images').rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in expected_images)
    orphan_labels = sorted(p.relative_to(root).as_posix() for p in (root / 'labels').rglob('*.txt') if p.relative_to(root).as_posix() not in expected_labels)
    after = snapshot(root, rows)
    if before != after:
        raise ValueError('Dataset changed while auditing; results are not a stable snapshot')
    return {'schema_version': 1, 'dataset_root': str(root), 'dataset_fingerprint': content_hash(before),
            'snapshot': before, 'split_stats': stats, 'source_stats': dict(sources), 'structural_errors': issues,
            'duplicates': duplicates, 'confirmed_identity_cross_split': leakage, 'cross_split_source_groups': group_leaks,
            'near_duplicate_candidates': near, 'blocked_train_ids': sorted(blocked),
            'near_cross_split_train_ids': sorted(cross_near_train), 'source_annotation_checks': source_checks,
            'orphan_images': orphan_images, 'orphan_labels': orphan_labels,
            'flag_counts': dict(Counter(flag for item in items for flag in item['flags'])), 'items': items,
            'human_review_counts': dict(Counter(row.get('human_review', 'not_recorded') for row in rows)),
            'limitations': ['Geometry/author-annotation agreement is not semantic correctness.',
                            'dHash<=8 is a review candidate, not proof of identity; no scene IDs for Street.',
                            'No teacher inference or exhaustive human missed-object review performed.',
                            'Declared/source license references require rights verification; no blanket license claim.']}
