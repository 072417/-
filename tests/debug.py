from tests.test_engine import fixture
from server.engine import recognize,regions,match,weight,sampled_color
import numpy as np
for kind in ['font_weight','icon_shape']:
 a=np.array(fixture(390));b=np.array(fixture(390,kind));rs=[]
 for v in [a,b]:
  t,e=recognize(v,lambda:None);r=t+regions(v,t);rs.append(r)
  print(kind,[(x['text'],[round(y,1) for y in x['box']],x['confidence']) for x in t])
 pairs,*_=match(rs[0],rs[1],a,b)
 for i,j,c in pairs:
  r,s=rs[0][i],rs[1][j]
  if 'Typography' in r['text'] or r['kind']=='icon':print(r,s,c,'weight',weight(a,r['box']),weight(b,s['box']),'color',sampled_color(a,r['box'],True),sampled_color(b,s['box'],True))
