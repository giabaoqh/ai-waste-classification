import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.prepare_taco_pilot import CLASSES

SCHEMA_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    """Write only new tool metadata, never source annotations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import uuid
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def contained(root, relative):
    root = Path(root).resolve()
    relative = str(relative).replace('\\', '/')
    if Path(relative).is_absolute() or ':' in relative:
        raise ValueError('Absolute source path is not allowed')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source path escapes dataset root')
    return path


def load_dataset(root):
    root = Path(root).resolve()
    config = yaml.safe_load((root / 'dataset.yaml').read_text())
    names = config['names']
    ordered = [names[i] for i in range(9)] if isinstance(names, dict) else names
    if ordered != CLASSES or len(names) != 9:
        raise ValueError('Dataset mapping does not match the existing nine object groups')
    if any(config.get(split) != 'images/' + split for split in ('train', 'val', 'test')):
        raise ValueError('Dataset YAML split paths do not match the frozen source layout')
    rows = read_json(root / 'manifest.json')
    seen = set()
    for row in rows:
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        for key, folder in [('image_file', 'images'), ('label_file', 'labels')]:
            path = contained(root, row[key])
            parts = Path(row[key].replace('\\', '/')).parts
            if len(parts) != 3 or parts[:2] != (folder, row['split']) or not path.is_file():
                raise ValueError('Manifest path/split/file mismatch: ' + row[key])
        if row['image_file'] in seen:
            raise ValueError('Duplicate manifest image entry')
        seen.add(row['image_file'])
    return root, rows


def valid_box(box, width, height):
    return (isinstance(box, list) and len(box) == 4 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box) and
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def read_yolo(path, width, height):
    boxes = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid label row {number}')
        cls, x, y, w, h = values
        if not cls.is_integer() or not 0 <= cls < 9 or w <= 0 or h <= 0:
            raise ValueError(f'Invalid class/size at row {number}')
        norm = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        if min(norm) < -1e-5 or max(norm) > 1.00001:
            raise ValueError(f'Out-of-image box at row {number}')
        box = [max(0., norm[0]) * width, max(0., norm[1]) * height,
               min(1., norm[2]) * width, min(1., norm[3]) * height]
        if not valid_box(box, width, height):
            raise ValueError(f'Degenerate box at row {number}')
        boxes.append({'class_id': int(cls), 'class_name': CLASSES[int(cls)], 'box': box,
                      'confidence': None, 'origin': 'original', 'original_index': number - 1})
    return boxes


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.


def snapshot(root, rows):
    paths = {'manifest.json', 'dataset.yaml', 'audit.json', 'warp_mapping.json'}
    paths.update(row[key] for row in rows for key in ('image_file', 'label_file'))
    # Detect newly added images/labels too; a manifest-only hash misses eval-folder additions.
    paths.update(p.relative_to(root).as_posix() for p in (Path(root) / 'images').rglob('*') if p.is_file())
    paths.update(p.relative_to(root).as_posix() for p in (Path(root) / 'labels').rglob('*.txt') if p.is_file())
    return {name: sha256(contained(root, name)) for name in sorted(paths) if contained(root, name).is_file()}
