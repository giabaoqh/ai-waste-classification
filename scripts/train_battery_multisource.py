"""Development battery classifier after the documented Kaggle-only domain failure."""
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import random
import shutil
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'logs/ultralytics_config'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))


def metrics(rows,threshold):
    tp=sum(r['truth'] and r['score']>=threshold for r in rows)
    fp=sum(not r['truth'] and r['score']>=threshold for r in rows)
    fn=sum(r['truth'] and r['score']<threshold for r in rows)
    tn=len(rows)-tp-fp-fn
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),accuracy=(tp+tn)/max(1,len(rows)))


def main():
    from PIL import Image,ImageOps
    from ultralytics import YOLO,settings
    settings.update({k:False for k in ['sync','wandb','comet','clearml','mlflow','dvc','tensorboard'] if k in settings})
    output=ROOT/'data/processed/battery_multisource_v2'
    run=ROOT/'runs/battery_screen/yolo26n_cls_multisource_v2'
    if output.exists() or run.exists():raise SystemExit('Use a fresh dataset/run version')
    rng=random.Random(42);manifest=[];hashes=set();dhashes=[]
    def dhash(image):
        px=list(image.convert('L').resize((9,8)).get_flattened_data())
        return sum((px[y*9+x]>px[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
    # Preserve every existing Kaggle split; never move a previous held-out image into train.
    old=ROOT/'data/processed/battery_screen_v1'
    for row in json.loads((old/'manifest.json').read_text()):
        name=Path(row['source_file']).name;target=output/row['split']/row['label']/('kg_'+name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/row['split']/row['label']/name,target)
        with Image.open(target) as image:dhashes.append(dhash(image))
        hashes.add(row['sha256'])
        manifest.append({**row,'source':'kaggle_garbage','image_file':target.relative_to(output).as_posix(),'physical_object_id':None})
    archive=ROOT/'data/raw/recybat24_v3/recybat24.tar.gz'
    metadata=json.loads((archive.parent/'record.json').read_text())
    checksum=next(f['checksum'] for f in metadata['files'] if f['key']=='recybat24.tar.gz')
    assert checksum=='md5:'+hashlib.md5(archive.read_bytes()).hexdigest()
    groups=defaultdict(list)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.name.endswith('/annotations.json'):continue
            data=json.load(tar.extractfile(member))
            ids={a['image_id'] for a in data['annotations'] if a['bbox'][2]>0 and a['bbox'][3]>0}
            for row in data['images']:
                if row['id'] not in ids:continue
                name=Path(row['file_name']).stem;kind=name.rsplit('_',1)[-1]
                assert kind in {'po','pri','cyl'}
                oid=kind+'_'+name.split('_')[0]
                groups[oid].append(str(Path(member.name).parent/row['file_name']).replace('\\','/'))
    split_by_group={}
    for kind in ['po','pri','cyl']:
        keys=sorted(k for k in groups if k.startswith(kind+'_'));rng.shuffle(keys)
        n=len(keys);n_train=int(.7*n);n_val=max(1,int(.15*n))
        for i,key in enumerate(keys):split_by_group[key]='train' if i<n_train else 'val' if i<n_train+n_val else 'test'
    selected={}
    for key,names in groups.items():
        rng.shuffle(names);split=split_by_group[key]
        for name in (names[:8] if split=='train' else names):selected[name]=(split,key)
    # Group identities prevent all views of the same physical battery crossing splits.
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if member.name not in selected:continue
            split,oid=selected[member.name]
            with Image.open(io.BytesIO(tar.extractfile(member).read())) as image:
                image=ImageOps.exif_transpose(image).convert('RGB')
                target=output/split/'battery'/('rb_'+Path(member.name).parent.name+'_'+Path(member.name).name)
                target.parent.mkdir(parents=True,exist_ok=True);image.save(target,'JPEG',quality=95)
                digest=hashlib.sha256(target.read_bytes()).hexdigest()
                if digest in hashes:target.unlink();continue
                hashes.add(digest);dhashes.append(dhash(image))
            manifest.append({'source':'recybat24_v3','source_file':member.name,'split':split,'label':'battery','physical_object_id':oid,'image_file':target.relative_to(output).as_posix(),'sha256':digest})
    # Add unused non-battery Kaggle images to balance training only; old test/val stay locked.
    needed=sum(r['split']=='train' and r['label']=='battery' for r in manifest)-sum(r['split']=='train' and r['label']=='not_battery' for r in manifest)
    used={r['source_file'] for r in manifest if r['source']=='kaggle_garbage'}
    with zipfile.ZipFile(ROOT/'data/raw/kaggle_garbage/garbage.zip') as z:
        choices=[n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png')) and '/battery/' not in n and n not in used]
        rng.shuffle(choices)
        for name in choices:
            if needed<=0:break
            with Image.open(io.BytesIO(z.read(name))) as image:
                image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((640,640));dh=dhash(image)
                if any((dh^old_dh).bit_count()<=4 for old_dh in dhashes):continue
                target=output/'train/not_battery'/('extra_'+Path(name).name)
                if target.exists():continue
                image.save(target,'JPEG',quality=95)
                digest=hashlib.sha256(target.read_bytes()).hexdigest()
                if digest in hashes:target.unlink();continue
                hashes.add(digest);dhashes.append(dh)
            manifest.append({'source':'kaggle_garbage_extra','source_file':name,'split':'train','label':'not_battery','physical_object_id':None,'image_file':target.relative_to(output).as_posix(),'sha256':digest});needed-=1
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    audit={'recybat_groups':split_by_group,'counts':{s:{label:sum(r['split']==s and r['label']==label for r in manifest) for label in ['battery','not_battery']} for s in ['train','val','test']},'limitations':['RecyBat was previously inspected as an external diagnostic; v2 holdout is developmental, not a fresh external benchmark','RecyBat groups use battery type plus physical ID across both original directories','Kaggle physical-object identities unavailable; original splits retained','Exact-hash deduplication across all; added Kaggle negatives also filtered by dHash','RecyBat train capped at 8 views per object; held-out views correlated','No Da Nang validation, no detection boxes or safety certification','No new negatives from an independent dark-conveyor domain']}
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    model=YOLO(str(ROOT/'checkpoints/yolo_pretrained/yolo26n-cls.pt'));start=time.perf_counter()
    model.train(data=str(output),epochs=20,imgsz=224,batch=16,device=0,workers=0,seed=42,deterministic=True,optimizer='AdamW',lr0=.001,patience=7,plots=False,project=str(run.parent),name=run.name,exist_ok=False)
    train_seconds=time.perf_counter()-start
    trained=YOLO(str(run/'weights/best.pt'));index=next(i for i,n in trained.names.items() if n=='battery')
    predictions={}
    for split in ['val','test']:
        rows=[r for r in manifest if r['split']==split];scores=[]
        for start_index in range(0,len(rows),16):
            chunk=rows[start_index:start_index+16]
            results=trained.predict([str(output/r['image_file']) for r in chunk],imgsz=224,device=0,verbose=False,save=False)
            scores.extend({**r,'truth':r['label']=='battery','score':float(pred.probs.data[index])} for r,pred in zip(chunk,results))
        predictions[split]=scores
    options=[(t/100,metrics(predictions['val'],t/100)) for t in range(50,96,5)]
    candidates=[x for x in options if x[1]['recall']>=.95 and x[1]['precision']>=.9]
    threshold=candidates[-1][0] if candidates else .8
    report={'task':'experimental image-level battery screening, multisource development v2','threshold':threshold,'validation_target_met':bool(candidates),'validation_target':{'recall':.95,'precision':.9},'validation':metrics(predictions['val'],threshold),'test':metrics(predictions['test'],threshold),'test_scope':'development holdout, not fresh independent external validation','per_source_test':{source:metrics([r for r in predictions['test'] if r['source']==source],threshold) for source in ['kaggle_garbage','recybat24_v3']},'checkpoint_sha256':hashlib.sha256((run/'weights/best.pt').read_bytes()).hexdigest(),'class_names':trained.names,'image_counts':{s:sum(r['split']==s for r in manifest) for s in ['train','val','test']},'split_class_counts':audit['counts'],'epochs_completed':len(list(csv.DictReader((run/'results.csv').open()))),'train_seconds':train_seconds,'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','ultralytics','numpy','pillow']},'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),'source_archive_checksums':{'recybat24_v3':checksum,'kaggle_garbage':json.loads((ROOT/'runs/battery_screen/yolo26n_cls_v1/battery_report.json').read_text())['source_archive_sha256']},'limitations':audit['limitations']}
    report['runtime_enabled']=bool(candidates)
    report['runtime_status']='experimental_development_only' if candidates else 'blocked_validation_target_not_met'
    report['independent_external_validated']=False
    report['per_source_validation']={source:metrics([r for r in predictions['val'] if r['source']==source],threshold) for source in ['kaggle_garbage','recybat24_v3']}
    report['per_domain_recall_goal_met_on_validation']=all(m['recall']>=.95 for m in report['per_source_validation'].values())
    report['seed']=42
    (run/'battery_report.json').write_text(json.dumps(report,indent=2))
    (run/'heldout_predictions.json').write_text(json.dumps(predictions,indent=2))
    print('BATTERY_MULTISOURCE_COMPLETE '+json.dumps(report),flush=True)


if __name__=='__main__':main()
