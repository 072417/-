"""Two independent anonymous visitors must never see or change each other's data."""
import io,os,tempfile,time,zipfile,json
os.environ['PUBLIC_MODE']='true'
os.environ['COOKIE_SECURE']='false'
os.environ['DESIGN_REVIEW_DATA']=tempfile.mkdtemp(prefix='review-public-test-')
from fastapi.testclient import TestClient
from PIL import Image
from server.app import app

def test_public_session_isolation_and_exports():
 with TestClient(app) as a, TestClient(app) as b:
  assert a.get('/api/health').status_code==200
  assert b.get('/api/health').status_code==200
  assert a.cookies.get('review_session')!=b.cookies.get('review_session')
  bio=io.BytesIO();Image.new('RGB',(375,300),'white').save(bio,'PNG')
  response=a.post('/api/assets',files={'file':('image.png',bio.getvalue(),'image/png')});assert response.status_code==200
  asset=response.json();assert 'owner' not in asset
  assert a.get(asset['url']).status_code==200
  assert b.get(asset['url']).status_code==404
  assert b.get('/assets-store/'+asset['id']+'/original.bin').status_code==404
  r=a.post('/api/runs',json={'name':'Isolation test','designId':asset['id'],'implementationId':asset['id']}).json();rid=r['id']
  assert 'owner' not in r
  assert b.get('/api/runs/'+rid).status_code==404
  assert b.patch('/api/runs/'+rid,json={'name':'stolen'}).status_code==404
  assert b.delete('/api/runs/'+rid).status_code==404
  assert b.get('/api/runs/'+rid+'/export/json').status_code==404
  assert b.post('/api/runs',json={'designId':asset['id'],'implementationId':asset['id']}).status_code==404
  assert b.get('/api/runs').json()==[]
  for _ in range(100):
   r=a.get('/api/runs/'+rid).json()
   if r['status'] not in ('running','queued'):break
   time.sleep(.2)
  assert r['status'] in ('partial','completed'),r
  assert r['issues']==[]
  result=a.get('/api/runs/'+rid+'/export/json').json()
  assert 'owner' not in json.dumps(result)
  z=a.get('/api/runs/'+rid+'/export/markdown');assert z.status_code==200
  with zipfile.ZipFile(io.BytesIO(z.content)) as package:
   assert package.testzip() is None
   assert {'report.md','report.json','annotated.png'}<=set(package.namelist())
  assert a.delete('/api/runs/'+rid).status_code==200
  assert a.get(asset['url']).status_code==404

def test_bad_upload_and_csrf():
 with TestClient(app) as a:
  assert a.post('/api/assets',files={'file':('bad.png',b'broken','image/png')}).status_code==400
  assert a.post('/api/demo',headers={'Origin':'https://unrelated.example'}).status_code==403
