"""Grounding DINO adapter: explicitly local artifacts, no network and no approval."""
from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch
import uuid

from PIL import Image

from .common import CLASSES, contained, content_hash, load_dataset, read_json, sha256, snapshot, valid_box, write_json
from .review import config, get_item, now

MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
WEIGHTS_SHA256 = '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
PHRASES = {'bottle': 0, 'jar': 0, 'can': 1, 'cup': 2, 'carton': 3, 'bottle cap': 4,
           'lid': 4, 'bag': 5, 'film': 5, 'wrapper': 5, 'paper': 6, 'cigarette': 7}


def teacher_status():
    modules = {key: importlib.util.find_spec(key) is not None for key in ('torch', 'transformers')}
    return {'model_id': MODEL_ID, 'revision': REVISION, 'modules': modules,
            'dependencies_available': all(modules.values()), 'artifact_verified': False,
            'inference_run': False, 'note': 'Cần artifact local đã duyệt; status không tải/nạp model.'}


def validate_artifact(artifact):
    artifact = Path(artifact).resolve()
    manifest = artifact / 'teacher_artifact.json'
    if not artifact.is_dir() or not manifest.is_file():
        raise ValueError('Teacher artifact local chưa có hoặc chưa có teacher_artifact.json; không tự tải.')
    receipt = read_json(manifest)
    if receipt.get('model_id') != MODEL_ID or receipt.get('revision') != REVISION or not receipt.get('authorization_reference'):
        raise ValueError('Teacher revision/receipt phê duyệt không khớp artifact đã chọn')
    files = receipt.get('files')
    required = {'config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json'}
    if not isinstance(files, dict) or not required.issubset(files) or not {'tokenizer.json', 'vocab.txt'}.intersection(files):
        raise ValueError('Teacher manifest thiếu weights/config/tokenizer local')
    actual = {p.relative_to(artifact).as_posix() for p in artifact.rglob('*') if p.is_file() and p != manifest}
    if actual != set(files) or any(Path(name).suffix not in ('.json', '.txt', '.safetensors') for name in files):
        raise ValueError('Artifact có file ngoài manifest hoặc code/pickle không được nạp')
    for name, expected in files.items():
        if sha256(contained(artifact, name)) != expected:
            raise ValueError('Teacher file checksum mismatch: ' + name)
    if files['model.safetensors'] != WEIGHTS_SHA256 or read_json(artifact / 'config.json').get('model_type') != 'grounding-dino':
        raise ValueError('Không dùng Student/random/Teacher khác thay artifact đã pin')
    return artifact, receipt


@contextmanager
def offline_guard():
    """CLI-only scope. Block even unexpected dependency HTTP/socket connections."""
    keys = ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE']
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = '1'
    def blocked(*_args, **_kwargs):
        raise RuntimeError('Teacher adapter không cho phép network/download')
    try:
        with patch.object(socket.socket, 'connect', blocked), patch.object(socket, 'create_connection', blocked):
            yield
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def normalize_proposals(boxes, scores, phrases, width, height):
    """Pure coordinate conversion of real producer arrays; unit fixtures are not inference."""
    if not len(boxes) == len(scores) == len(phrases):
        raise ValueError('Teacher output array lengths không khớp')
    proposals, rejected = [], []
    for index, (raw, score, phrase) in enumerate(zip(boxes, scores, phrases)):
        if len(raw) != 4 or not all(math.isfinite(v) for v in raw) or not isinstance(phrase, str) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Teacher output có tọa độ/score/phrase không hợp lệ')
        box = [max(0., raw[0]), max(0., raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
        if not valid_box(box, width, height):
            rejected.append({'raw_box': raw, 'raw_label': phrase, 'confidence': score, 'reason': 'degenerate_or_outside_image'})
            continue
        cls = PHRASES.get(phrase.strip().lower().rstrip('.'))
        proposals.append({'proposal_id': f'dino-{index}', 'class_id': cls,
                          'class_name': CLASSES[cls] if cls is not None else None, 'raw_label': phrase,
                          'box': box, 'raw_box': raw, 'box_adjustment': 'clipped_to_image' if box != raw else 'none',
                          'confidence': float(score), 'confidence_kind': 'detector_score'})
    return proposals, rejected


def propose_local(workspace, artifact, output, limit=5, device='cpu', threshold=.25, text_threshold=.25):
    cfg = config(workspace)
    artifact, receipt = validate_artifact(artifact)
    output = Path(output).resolve()
    roots = [Path(cfg['dataset_root']).resolve(), Path(workspace).resolve(), artifact]
    if output.exists() or output.suffix != '.json' or any(output.is_relative_to(root) for root in roots):
        raise ValueError('Teacher output phải là JSON mới, ngoài source/artifact/review workspace')
    if not 1 <= limit <= 200 or device not in ('cpu', 'cuda') or not all(math.isfinite(v) and 0 < v <= 1 for v in (threshold, text_threshold)):
        raise ValueError('Teacher limit/device/threshold không hợp lệ')
    if not teacher_status()['dependencies_available']:
        raise ValueError('Thiếu dependency Teacher; cần duyệt env/Transformers, không tự cài.')
    root, rows = load_dataset(cfg['dataset_root'])
    frozen = read_json(Path(workspace) / 'source_snapshot.json')
    if snapshot(root, rows) != frozen:
        raise ValueError('Source dataset đổi trước Teacher inference')
    entries = []
    settings = {'phrases': PHRASES, 'input_max_edge': 640, 'box_threshold': threshold, 'text_threshold': text_threshold}
    with offline_guard():
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        if device == 'cuda':
            if not torch.cuda.is_available(): raise ValueError('CUDA không sẵn sàng, chọn CPU rõ ràng nếu cần.')
            torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
        processor = AutoProcessor.from_pretrained(str(artifact), local_files_only=True, trust_remote_code=False)
        processor.image_processor.size = {'shortest_edge': 640, 'longest_edge': 640}
        model = AutoModelForZeroShotObjectDetection.from_pretrained(str(artifact), local_files_only=True,
                                                                   trust_remote_code=False, use_safetensors=True).to(device).eval()
        for image_id in cfg['image_ids'][:limit]:
            record = get_item(workspace, image_id)
            source = record['source']
            started = time.perf_counter()
            with Image.open(contained(root, source['image_file'])) as image:
                inputs = processor(images=image.convert('RGB'), text=[list(PHRASES)], return_tensors='pt').to(device)
            with torch.inference_mode():
                predictions = model(**inputs)
            result = processor.post_process_grounded_object_detection(predictions, threshold=threshold,
                         text_threshold=text_threshold, target_sizes=[(source['height'], source['width'])])[0]
            proposals, rejected = normalize_proposals(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(),
                                                       result['text_labels'], source['width'], source['height'])
            entries.append({'image_id': image_id, 'image_sha256': source['image_sha256'], 'split': 'train',
                            'width': source['width'], 'height': source['height'], 'proposals': proposals,
                            'rejected_native_proposals': rejected, 'inference_status': 'completed',
                            'processing_time_ms': (time.perf_counter() - started) * 1000})
        metadata = {'kind': 'grounding_dino', 'execution_mode': 'local', 'model_id': MODEL_ID, 'version': REVISION,
                    'run_id': uuid.uuid4().hex, 'generated_at': now(), 'producer': 'local_teacher_cli',
                    'adapter_version': 'grounding-dino-hf-offline-v1', 'prompt_sha256': content_hash(settings),
                    'artifact_sha256': receipt['files']['model.safetensors'], 'artifact_manifest_sha256': sha256(artifact / 'teacher_artifact.json'),
                    'authorization_reference': receipt['authorization_reference'], 'coordinate_space': 'xyxy_pixels_exif_oriented',
                    'is_simulated': False, 'device': device, 'settings': settings,
                    'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                    'peak_allocated_vram_mib': torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else None}
        del model, processor
        if device == 'cuda': torch.cuda.empty_cache()
    if snapshot(root, rows) != frozen:
        raise ValueError('Source đổi trong inference; không xuất bundle')
    bundle = {'schema_version': 1, 'teacher': metadata, 'images': entries}
    write_json(output, bundle)
    return {'output': str(output), 'images': len(entries), 'approved': 0, 'imported': False}
