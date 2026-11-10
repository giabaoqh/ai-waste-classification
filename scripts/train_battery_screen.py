"""Train an image-level battery screening classifier; never a detector or safety assay."""
import os,json,hashlib,random,zipfile,time,csv,platform,importlib.metadata,subprocess
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'logs/ultralytics_config'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))

def main():
    from PIL import Image,ImageOps
    from ultralytics import YOLO,settings
    import torch
    settings.update({k:False for k in ['sync','wandb','comet','clearml','mlflow','dvc','tensorboard'] if k in settings})
    output=ROOT/'data/processed/battery_screen_v1'
    if output.exists():raise SystemExit('Dataset version already exists')
    manifest=[];seen=[];rng=random.Random(42)
    with zipfile.ZipFile(ROOT/'data/raw/kaggle_garbage/garbage.zip') as z:
        files=[n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))]
        positives=[n for n in files if '/battery/' in n]
        negatives=[n for n in files if '/battery/' not in n]
        rng.shuffle(positives);rng.shuffle(negatives)
        chosen=[('battery',n) for n in positives]+[('not_battery',n) for n in negatives[:len(positives)]]
        for label,name in chosen:
            raw=z.read(name)
            try:
                with Image.open(BytesIO(raw)) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((640,640))
                    px=list(im.convert('L').resize((9,8)).get_flattened_data())
                    dh=sum((px[y*9+x]>px[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
                    if any((dh^old).bit_count()<=4 for old in seen):continue
                    seen.append(dh)
                    split='train' if (u:=rng.random())<.7 else 'val' if u<.85 else 'test'
                    target=output/split/label/Path(name).name;target.parent.mkdir(parents=True,exist_ok=True)
                    im.save(target,'JPEG',quality=95)
                    manifest.append({'source_file':name,'split':split,'label':label,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'dhash':f'{dh:016x}'})
            except (OSError,ValueError):continue
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model=YOLO(str(ROOT/'checkpoints/yolo_pretrained/yolo26n-cls.pt'))
    start=time.perf_counter()
    model.train(data=str(output),epochs=20,imgsz=224,batch=16,device=0,workers=0,amp=True,seed=42,deterministic=True,optimizer='AdamW',lr0=.001,patience=7,plots=False,project=str(ROOT/'runs/battery_screen'),name='yolo26n_cls_v1',exist_ok=False)
    run=ROOT/'runs/battery_screen/yolo26n_cls_v1'
    trained=YOLO(str(run/'weights/best.pt'))
    # Select the abstention threshold using validation only, then evaluate the locked test once.
    bysplit={}
    for split in ['val','test']:
        rows=[]
        for row in manifest:
            if row['split']!=split:continue
            r=trained.predict(str(output/split/row['label']/Path(row['source_file']).name),imgsz=224,device=0,verbose=False,save=False)[0]
            idx=next(i for i,n in trained.names.items() if n=='battery')
            rows.append({'truth':row['label']=='battery','score':float(r.probs.data[idx]),'source_file':row['source_file']})
        bysplit[split]=rows
    def metrics(rows,t):
        tp=sum(r['truth'] and r['score']>=t for r in rows);fp=sum(not r['truth'] and r['score']>=t for r in rows);fn=sum(r['truth'] and r['score']<t for r in rows);tn=len(rows)-tp-fp-fn
        return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'accuracy':(tp+tn)/max(1,len(rows))}
    options=[(t/100,metrics(bysplit['val'],t/100)) for t in range(50,96,5)]
    qualified=[x for x in options if x[1]['recall']>=.9 and x[1]['precision']>=.8]
    threshold=qualified[-1][0] if qualified else .8
    report={'task':'image-level battery screening, no bounding boxes','threshold':threshold,'validation_target_met':bool(qualified),'train_seconds':time.perf_counter()-start,'validation':metrics(bysplit['val'],threshold),'test':metrics(bysplit['test'],threshold),'image_counts':{s:sum(r['split']==s for r in manifest) for s in ['train','val','test']},'source':'https://www.kaggle.com/datasets/mostafaabla/garbage-classification','source_archive_sha256':hashlib.sha256((ROOT/'data/raw/kaggle_garbage/garbage.zip').read_bytes()).hexdigest(),'limitations':['No battery localization, chemistry, damage or fire prediction','Internal dataset split only; scene/physical-object identities unavailable','dHash heuristic deduplication, human QA pending','Low score does not rule out a hidden battery','Not evaluated on phone-camera images from Da Nang']}
    report['checkpoint_sha256']=hashlib.sha256((run/'weights/best.pt').read_bytes()).hexdigest()
    report['training_and_internal_evaluation_seconds']=report.pop('train_seconds')
    report['epoch_loop_seconds']=float(list(csv.DictReader((run/'results.csv').open()))[-1]['time'])
    report['class_names']=trained.names
    report['epochs_completed']=len(list(csv.DictReader((run/'results.csv').open())))
    report['seed']=42
    report['training_parameters']={'epochs_requested':20,'imgsz':224,'batch':16,'workers':0,'optimizer':'AdamW','lr0':.001,'patience':7}
    report['packages']={name:importlib.metadata.version(name) for name in ['torch','torchvision','ultralytics','pillow','numpy']}
    report['python']=platform.python_version()
    report['gpu']=torch.cuda.get_device_name(0)
    report['manifest_sha256']=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    report['initial_weights_sha256']=hashlib.sha256((ROOT/'checkpoints/yolo_pretrained/yolo26n-cls.pt').read_bytes()).hexdigest()
    report['split_class_counts']={split:{label:sum(row['split']==split and row['label']==label for row in manifest) for label in ['battery','not_battery']} for split in ['train','val','test']}
    freeze=subprocess.run([os.sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (run/'environment.txt').write_text(freeze.stdout,encoding='utf-8')
    (run/'battery_report.json').write_text(json.dumps(report,indent=2))
    (run/'heldout_predictions.json').write_text(json.dumps(bysplit,indent=2))
    print('BATTERY_COMPLETE '+json.dumps(report),flush=True)

if __name__=='__main__':main()
