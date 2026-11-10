"""Compare a trained detector to the pilot on fixed validation domains only."""
import os,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'logs/ultralytics_config'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    from ultralytics import YOLO
    from prepare_taco_pilot import CLASSES
    dataset=a.dataset.resolve()
    rows=json.loads((dataset/'manifest.json').read_text())
    out=ROOT/'runs/waste_pilot'/a.run/'domain_evaluation';out.mkdir(exist_ok=True)
    domains={
      'old_validation':ROOT/'data/processed/taco_pilot_v3/dataset.yaml',
      'new_validation':dataset/'dataset.yaml'}
    for source in ['kaggle_warp','hf_taco']:
        files=[(dataset/r['image_file']).as_posix() for r in rows if r['split']=='val' and r['source']==source]
        listing=out/(source+'.txt');listing.write_text('\n'.join(files))
        config=out/(source+'.yaml')
        config.write_text('path: '+json.dumps(dataset.as_posix())+'\ntrain: images/train\nval: '+json.dumps(listing.as_posix())+'\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(CLASSES)))
        domains[source]=config
    summary={}
    for label,weights in [('pilot',ROOT/'runs/waste_pilot/yolo26n_pilot10_v1/weights/best.pt'),('expanded',out.parent/'weights/best.pt')]:
        model=YOLO(str(weights));result={}
        for domain,data in domains.items():
            m=model.val(data=str(data),split='val',imgsz=640,batch=4,device=0,workers=0,plots=False,verbose=False,project=str(out),name=label+'_'+domain)
            result[domain]={k:float(v) for k,v in m.results_dict.items()}
        negatives=[r for r in rows if r['split']=='val' and r['source']=='hf_street_negative']
        false_positives=0
        for row in negatives:
            pred=model.predict(str(dataset/row['image_file']),imgsz=640,device=0,conf=.25,verbose=False,save=False)[0]
            false_positives+=int(len(pred.boxes)>0)
        result['negative_diagnostic']={'images':len(negatives),'images_with_detection':false_positives,'conf':.25,'note':'tiny validation sample, not acceptance FPR'}
        # Both checkpoints are already fixed; this reserved test is reporting only.
        m=model.val(data=str(dataset/'dataset.yaml'),split='test',imgsz=640,batch=4,device=0,workers=0,plots=False,verbose=False,project=str(out),name=label+'_reserved_test')
        result['reserved_test']={k:float(v) for k,v in m.results_dict.items()}
        result['reserved_test_images']=sum(r['split']=='test' for r in rows)
        summary[label]=result
    (out.parent/'comparison.json').write_text(json.dumps(summary,indent=2))
    print('EVALUATION_COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
