import httpx,time,json
from pathlib import Path
with httpx.Client(base_url='http://127.0.0.1:8765',timeout=60) as c:
 a=c.post('/api/demo').json()
 r=c.post('/api/runs',json={'name':'运动记录 · 示例走查','designId':a['design']['id'],'implementationId':a['implementation']['id'],'config':{'design':{'logicalWidth':390},'implementation':{'logicalWidth':390}}}).json()
 for _ in range(120):
  r=c.get('/api/runs/'+r['id']).json()
  if r['status'] not in ('queued','running'):break
  time.sleep(1)
 Path('test-results/demo-result.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
 print(r['status'],r.get('error'),r.get('duration'),r.get('candidateCounts'))
 for i in r.get('issues',[]):print(i['id'],i['category'],i['title'],i['regionLabel'],[(m['metric'],m['delta']) for m in i['measurements']])
 print('RUN',r['id'])
