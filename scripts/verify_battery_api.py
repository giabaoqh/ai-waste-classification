
import hashlib,json,requests,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
run=root/'runs/battery_screen/yolo26n_cls_multisource_v2'
report=json.loads((run/'battery_report.json').read_text())
dataset=root/'data/processed/battery_multisource_v2'
manifest=json.loads((dataset/'manifest.json').read_text())
record={'capabilities_before':requests.get('http://127.0.0.1:8000/api/safety/capabilities',timeout=30).json(),'cases':[]}
for label in ['battery','not_battery']:
 rows=[row for row in manifest if row['split']=='test' and row['label']==label][:2]
 if label=='battery':rows=[next(row for row in manifest if row['split']=='test' and row['source']==source and row['label']==label) for source in ['kaggle_garbage','recybat24_v3']]
 for row in rows:
  path=dataset/row['image_file']
  content=path.read_bytes()
  tick=time.perf_counter()
  response=requests.post('http://127.0.0.1:8000/api/safety/scan',files={'file':('neutral_name.jpg',content,'image/jpeg')},timeout=90)
  assert response.status_code==200,response.text
  body=response.json()
  assert body['is_uncertain'] and not body['confirmed_safe']
  assert body['threshold']==report['threshold']
  assert body['bounding_boxes'] is None
  assert body['quality_assessment']['is_acceptable']
  assert body['battery_suspected']==(body['battery_score']>=report['threshold'])
  record['cases'].append({'source_file':row['source_file'],'truth':label,'image_file':row['image_file'],'sha256':hashlib.sha256(content).hexdigest(),'response':body,'matches_expected_label':body['battery_suspected']==(label=='battery'),'request_seconds':time.perf_counter()-tick})
# Content is unchanged; a misleading filename must not alter visual predictions.
first=record['cases'][0]
path=dataset/first['image_file']
r=requests.post('http://127.0.0.1:8000/api/safety/scan',files={'file':('ordinary_paper.jpg',path.read_bytes(),'image/jpeg')},timeout=90)
assert r.status_code==200,r.text
assert abs(r.json()['battery_score']-first['response']['battery_score'])<1e-6
record['filename_invariance_verified']=True
record['capabilities_after']=requests.get('http://127.0.0.1:8000/api/safety/capabilities',timeout=30).json()
assert record['capabilities_after']['ai_model_loaded']
(root/'logs/battery_api_smoke.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('REAL_BATTERY_API_SMOKE_OK',len(record['cases']))
