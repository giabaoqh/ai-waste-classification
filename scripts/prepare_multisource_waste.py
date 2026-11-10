"""Build a provenance-tracked 9-class detector dataset from Kaggle and HF."""
import sys
import json, hashlib, random, re, zipfile
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
try:
    from .prepare_taco_pilot import CLASSES, category_mapping, normalized_box
except ImportError:
    from prepare_taco_pilot import CLASSES, category_mapping, normalized_box

ROOT=Path(__file__).resolve().parents[1]
HF_TACO="bfac5b10c53ae66922a252bd00a4eacb95a43591"
HF_STREET="9d80bca6a178b53864737a8967faa350cafba4f3"

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fetch(url,path):
    import requests
    if not path.exists():
        r=requests.get(url,timeout=90);r.raise_for_status()
        temp=path.with_suffix('.part');temp.write_bytes(r.content);temp.replace(path)
    return path.read_bytes()

def canonical(content):
    with Image.open(BytesIO(content)) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        original_size=im.size
        im.thumbnail((640,640))
        pixels=list(im.convert('L').resize((9,8)).get_flattened_data())
        dh=sum((pixels[y*9+x]>pixels[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
        out=BytesIO();im.save(out,'JPEG',quality=95)
        return out.getvalue(),dh,original_size

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'data/processed/waste_multisource_v2');args=parser.parse_args()
    output=args.output.resolve()
    if output.exists(): raise SystemExit('Choose a new dataset version; output already exists')
    output.mkdir(parents=True)
    cache=ROOT/'data/raw/hf_multisource';cache.mkdir(parents=True,exist_ok=True)
    manifest=[];excluded=[];seen=[]
    old=ROOT/'data/processed/taco_pilot_v3'
    old_manifest=json.loads((old/'manifest.json').read_text())
    # Reserve all previous validation/test examples before accepting any new train image.
    for item in old_manifest:
        if item['split']=='train':continue
        content=(old/item['image_file']).read_bytes()
        _,dh,_=canonical(content)
        seen.append((dh,item['split']))
        name=f"reserved_{item['image_id']:05d}"
        for kind in ['images','labels']:(output/kind/item['split']).mkdir(parents=True,exist_ok=True)
        ip=output/'images'/item['split']/(name+'.jpg');ip.write_bytes(content)
        lp=output/'labels'/item['split']/(name+'.txt');lp.write_text((old/item['label_file']).read_text())
        manifest.append({**item,'source':'taco_reserved','image_file':str(ip.relative_to(output)).replace('\\','/'),'label_file':str(lp.relative_to(output)).replace('\\','/')})

    def add(key,split,content,labels,source,provenance):
        data,dh,size=canonical(content)
        if any((dh^other).bit_count()<=4 for other,_ in seen):
            excluded.append({'key':key,'reason':'perceptual_duplicate'});return False
        for line in labels:
            c,x,y,w,h=map(float,line.split())
            if not c.is_integer() or not 0<=c<len(CLASSES) or not (w>0 and h>0 and x-w/2>=-1e-5 and y-h/2>=-1e-5 and x+w/2<=1.00001 and y+h/2<=1.00001):
                excluded.append({'key':key,'reason':'invalid_bbox'});return False
        seen.append((dh,split))
        for kind in ['images','labels']:(output/kind/split).mkdir(parents=True,exist_ok=True)
        ip=output/'images'/split/(key+'.jpg');ip.write_bytes(data)
        lp=output/'labels'/split/(key+'.txt');lp.write_text('\n'.join(labels)+ ('\n' if labels else ''))
        manifest.append({'split':split,'source':source,'sha256':digest(ip),'source_sha256':hashlib.sha256(content).hexdigest(),'dhash':f'{dh:016x}','image_file':ip.relative_to(output).as_posix(),'label_file':lp.relative_to(output).as_posix(),'provenance':provenance,'human_review':'pending'})
        return True

    # Preserve official WaRP test completely; split source training images by date.
    with zipfile.ZipFile(ROOT/'data/raw/warp_kaggle/warp.zip') as z:
        names=z.read('Warp-D/classes.txt').decode().splitlines()
        mapping={i:0 if n.startswith(('bottle-','glass-')) else 1 if n=='cans' else 3 if 'cardboard' in n else 8 for i,n in enumerate(names)}
        files=sorted(n for n in z.namelist() if n.startswith('Warp-D/train/images/') and n.endswith('.jpg'))
        def group(n):
            m=re.search(r'\d{2}-[A-Za-z]{3}',n)
            if not m: raise ValueError('No capture date: '+n)
            return m.group()
        groups=sorted({group(n) for n in files});random.Random(42).shuffle(groups)
        vg=set(groups[:max(1,round(len(groups)*.2))])
        for split,limit in [('val',200),('train',1000)]:
            candidates=[n for n in files if (group(n) in vg)==(split=='val')];random.Random(42).shuffle(candidates)
            accepted=0
            for name in candidates:
                lines=[]
                for line in z.read(name.replace('/images/','/labels/').replace('.jpg','.txt')).decode().splitlines():
                    if not line.strip():continue
                    v=line.split();lines.append(str(mapping[int(v[0])])+' '+' '.join(v[1:]))
                if add('warp_'+Path(name).stem,split,z.read(name),lines,'kaggle_warp',{'ref':'parohod/warp-waste-recycling-plant-dataset','file':name,'capture_date':group(name),'license_declared':'CC0'}):accepted+=1
                if accepted>=limit:break
            print('Kaggle WaRP',split,accepted,flush=True)
        (output/'warp_mapping.json').write_text(json.dumps({names[i]:CLASSES[c] for i,c in mapping.items()},indent=2))
    # HF TACO mirror: original pinned author labels, not potentially inconsistent mirror class names.
    taco=json.loads((ROOT/'data/raw/taco/annotations.json').read_text())
    tm=category_mapping(taco['categories']);anns={i['id']:[] for i in taco['images']}
    for a in taco['annotations']:anns[a['image_id']].append(a)
    batch_split=json.loads((old/'audit.json').read_text())['source_batch_splits']
    for split,limit in [('val',100),('train',600)]:
        candidates=[i for i in taco['images'] if batch_split[Path(i['file_name']).parts[0]]==split];random.Random(42).shuffle(candidates)
        accepted=0
        for offset in range(0,len(candidates),12):
            batch=candidates[offset:offset+12]
            def download(i):
                filename=i['file_name'].replace('/','_')
                url=f'https://huggingface.co/datasets/pictograph/taco/resolve/{HF_TACO}/train/{filename}'
                try:return i,fetch(url,cache/('taco_'+str(i['id'])+'.jpg')),url,None
                except Exception as e:return i,None,url,str(e)
            with ThreadPoolExecutor(8) as pool:items=list(pool.map(download,batch))
            for i,data,url,error in items:
                if error:excluded.append({'key':str(i['id']),'reason':error});continue
                _,_,size=canonical(data)
                if abs((size[0]/size[1])/(i['width']/i['height'])-1)>.02:excluded.append({'key':str(i['id']),'reason':'aspect_mismatch'});continue
                labels=[str(tm[a['category_id']])+' '+' '.join(f'{x:.8f}' for x in normalized_box(a['bbox'],i['width'],i['height'])) for a in anns[i['id']]]
                if add('hftaco_'+str(i['id']),split,data,labels,'hf_taco',{'url':url,'original_file':i['file_name'],'batch':Path(i['file_name']).parts[0],'license':i.get('license')}):accepted+=1
                if accepted>=limit:break
            print('HF TACO',split,accepted,'/',limit,flush=True)
            if accepted>=limit:break
    # Only certified-empty annotations from generic garbage taxonomy are compatible negatives.
    for split,src in [('val','valid'),('train','train')]:
        d=json.loads((ROOT/'data/raw/hf_street'/(src+'.json')).read_text());positive={a['image_id'] for a in d['annotations']}
        candidates=[i for i in d['images'] if i['id'] not in positive]
        def download_empty(i):
            url=f"https://huggingface.co/datasets/Shynuaa/Street-garbage/resolve/{HF_STREET}/coco/{src}/images/{i['file_name']}"
            try:return i,fetch(url,cache/f'street_{src}_{i["id"]}.jpg'),url,None
            except Exception as e:return i,None,url,str(e)
        with ThreadPoolExecutor(8) as pool:items=list(pool.map(download_empty,candidates))
        for i,data,url,error in items:
            if error:excluded.append({'key':'street_'+str(i['id']),'reason':error});continue
            add('street_'+src+'_'+str(i['id']),split,data,[],'hf_street_negative',{'url':url,'license_declared':'MIT','source_annotation':'zero annotations; not manually reverified'})
    config='path: '+json.dumps(output.as_posix())+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {c}\n' for i,c in enumerate(CLASSES))
    (output/'dataset.yaml').write_text(config)
    from collections import Counter
    counts=Counter(m['split'] for m in manifest)
    audit={'image_counts':dict(counts),'by_source_split':dict(Counter(m['source']+':'+m['split'] for m in manifest)),'excluded':excluded,'warp_archive_sha256':digest(ROOT/'data/raw/warp_kaggle/warp.zip'),'limitations':['human taxonomy QA pending','WaRP training split grouped by capture date; official 522-image test untouched','HF TACO is a mirror, not an independent dataset','cross-source dHash excludes near-duplicates but not all possible leakage','different domains and class frequencies; aggregate mAP not directly comparable to 100-image pilot','no medical/polymer/disposal capability validated']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    print('DATASET_COMPLETE',json.dumps(audit)[:1800],flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding="utf-8")
    main()
