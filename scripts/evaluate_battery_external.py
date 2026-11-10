"""Positive-only external-domain diagnostic. Never tunes the locked threshold."""
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'logs/ultralytics_config'))


def main():
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    archive = ROOT/'data/raw/recybat24_v3/recybat24.tar.gz'
    run = ROOT/'runs/battery_screen/yolo26n_cls_v1'
    report = json.loads((run/'battery_report.json').read_text())
    metadata = json.loads((archive.parent/'record.json').read_text())
    expected = next(f['checksum'] for f in metadata['files'] if f['key']=='recybat24.tar.gz')
    assert expected == 'md5:'+hashlib.md5(archive.read_bytes()).hexdigest()
    model = YOLO(str(run/'weights/best.pt'))
    index = next(i for i,name in model.names.items() if name=='battery')
    positive_files=set();source_counts={}
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.name.endswith('/annotations.json'):continue
            annotations=json.load(tar.extractfile(member))
            category_ids={row['id'] for row in annotations['categories'] if row['name'] in {'cylindric','pouch','prismatic'}}
            assert len(category_ids)==3
            positive_ids={row['image_id'] for row in annotations['annotations'] if row['category_id'] in category_ids and row['bbox'][2]>0 and row['bbox'][3]>0}
            for row in annotations['images']:
                if row['id'] in positive_ids:positive_files.add(str(Path(member.name).parent/row['file_name']).replace('\\','/'))
            source_counts[member.name]={'declared_images':len(annotations['images']),'images_with_positive_annotation':len(positive_ids)}
    rows=[]; batch=[]; names=[]
    def predict_batch():
        for name, result in zip(names, model.predict(batch, imgsz=224, device=0, verbose=False, save=False)):
            rows.append({'source_file':name,'battery_score':float(result.probs.data[index])})
        batch.clear();names.clear()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            # This release contains original images. No extraction or path traversal.
            if not member.isfile() or not member.name.lower().endswith(('.jpg','.jpeg','.png')):continue
            if member.name not in positive_files:continue
            if any(word in member.name.lower() for word in ['rotated','brightness','exposure']):
                raise ValueError('Unexpected augmented image in original release')
            stream = tar.extractfile(member)
            with Image.open(io.BytesIO(stream.read())) as image:
                batch.append(ImageOps.exif_transpose(image).convert('RGB'))
            names.append(member.name)
            if len(batch)==16:predict_batch()
        if batch:predict_batch()
    if not rows:raise ValueError('No original external images')
    threshold=report['threshold'];positive=sum(r['battery_score']>=threshold for r in rows)
    result={'task':'positive-only external image-level battery diagnostic',
        'threshold_locked_from_internal_validation':threshold,'images':len(rows),
        'screen_positive':positive,'screen_negative':len(rows)-positive,
        'image_recall':positive/len(rows),
        'source':'https://zenodo.org/records/15226091','source_archive_checksum':expected,
        'source_annotation_counts':source_counts,
        'checkpoint_sha256':report['checkpoint_sha256'],
        'limitations':['Only positives: precision/specificity/FPR cannot be measured',
          'Multiple views/lighting per physical battery; images are not independent objects',
          'Industrial controlled background, not a Da Nang phone-camera test',
          'No box localization or battery damage/chemistry assessment',
          'Archive counts differ from the paper; report observed annotation counts without silently substituting paper counts',
          'Diagnostic only: do not tune threshold on this external sample']}
    (run/'external_recybat_report.json').write_text(json.dumps(result,indent=2))
    (run/'external_recybat_predictions.json').write_text(json.dumps(rows,indent=2))
    print('EXTERNAL_BATTERY_COMPLETE '+json.dumps(result),flush=True)


if __name__=='__main__':main()
