"""Evidence-first screenshot analysis. All bboxes originate from local detectors."""
import io, json, math, os, re, subprocess, tempfile, time, shutil, csv
from pathlib import Path
from difflib import SequenceMatcher
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab, deltaE_ciede2000

CATEGORIES = {'text_size':'文字大小','font_weight':'字重','alignment':'对齐','color':'颜色','icon_shape':'Icon 形变','component_position':'组件位置','spacing':'元素间距'}
HELPER = Path(__file__).with_name('ocr-helper')
MOBILE_WIDTHS = (360,375,390,414,430)
TEST_OVERLAY = re.compile(r'(?i)(?:^|[^a-z0-9])(?:c\s*[b8][a4]se|leg[o0]|容器)(?:[^a-z0-9]|$)')
LONG_TEST_ID = re.compile(r'\d{7,}|(?i:pin)[a-z0-9:/_-]{4,}|(?=[a-z0-9:/_-]{10,})(?=[a-z0-9:/_-]*[a-z])(?=[a-z0-9:/_-]*\d)[a-z0-9:/_-]+')

def normalized_text(value):
 return re.sub(r'[^A-Za-z0-9\u3400-\u9fff]','',value or '')

def crop(a,b):
 x,y,w,h = [int(round(v)) for v in b]
 return a[max(0,y):min(a.shape[0],y+h),max(0,x):min(a.shape[1],x+w)]

def iou(a,b):
 x=max(a[0],b[0]); y=max(a[1],b[1]); w=max(0,min(a[0]+a[2],b[0]+b[2])-x); h=max(0,min(a[1]+a[3],b[1]+b[3])-y)
 return w*h/max(1,a[2]*a[3]+b[2]*b[3]-w*h)

def coverage(a,b):
 x=max(a[0],b[0]); y=max(a[1],b[1]); w=max(0,min(a[0]+a[2],b[0]+b[2])-x); h=max(0,min(a[1]+a[3],b[1]+b[3])-y)
 return w*h/max(1,a[2]*a[3])

def recognize(a,check):
 if not HELPER.exists() and not shutil.which('tesseract'): return [], '文字识别引擎不可用，本次文字相关检查未完成。'
 result=[]
 for offset in range(0,a.shape[0],1400):
  check()
  part=a[max(0,offset-100):min(a.shape[0],offset+1500)]
  with tempfile.NamedTemporaryFile(suffix='.png') as f:
   Image.fromarray(part).save(f.name)
   try:
    if HELPER.exists():
     run=subprocess.run([str(HELPER),f.name],capture_output=True,text=True,timeout=35,check=True)
     rows=json.loads(run.stdout)
    else:
     run=subprocess.run(['tesseract',f.name,'stdout','-l','chi_sim+eng','--psm','11','tsv'],capture_output=True,text=True,timeout=35,check=True)
     lines={}
     for row in csv.DictReader(io.StringIO(run.stdout),delimiter='\t'):
      if not row.get('text','').strip() or float(row.get('conf','-1'))<25:continue
      key=(row['block_num'],row['par_num'],row['line_num'])
      lines.setdefault(key,[]).append(row)
     rows=[]
     for words in lines.values():
      x=min(int(w['left']) for w in words);y=min(int(w['top']) for w in words)
      right=max(int(w['left'])+int(w['width']) for w in words);bottom=max(int(w['top'])+int(w['height']) for w in words)
      rows.append({'text':' '.join(w['text'] for w in words),'confidence':sum(float(w['conf']) for w in words)/len(words)/100,'box':[x,y,right-x,bottom-y]})
   except (subprocess.SubprocessError,ValueError): return result, '部分文字识别超时或失败，文字相关检查不完整。'
  for r in rows:
   r['box'][1]+=max(0,offset-100)
   if not re.search(r'[A-Za-z0-9\u3400-\u9fff]',r['text']): continue
   if r['confidence']<.25 or any(iou(r['box'],v['box'])>.55 for v in result): continue
   r['box']=refine_text_box(a,r['box'])
   r['kind']='text'; result.append(r)
 return result,None

def photo_like(a,b):
 p=crop(a,b)
 if not p.size or min(p.shape[:2])<28:return False
 h,w=p.shape[:2]
 if max(h,w)>min(340,a.shape[1]*.72) or h*w>a.shape[0]*a.shape[1]*.2:return False
 sample=cv2.resize(p,(min(96,w),min(96,h)),interpolation=cv2.INTER_AREA)
 quant=(sample//16).reshape(-1,3)
 unique=len(np.unique(quant,axis=0))
 gray=cv2.cvtColor(sample,cv2.COLOR_RGB2GRAY)
 hist=cv2.calcHist([gray],[0],None,[32],[0,256]).ravel();hist=hist/max(1,hist.sum())
 entropy=float(-(hist[hist>0]*np.log2(hist[hist>0])).sum())
 edges=float((cv2.Canny(gray,35,100)>0).mean())
 color_spread=float(np.mean(np.std(sample.astype(float),axis=(0,1))))
 return unique>=18 and entropy>=2.7 and color_spread>=16 and edges>=.025

def regions(a,texts):
 # External contours provide components; text boxes are measured independently.
 gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 edges=cv2.Canny(gray,35,100)
 for t in texts:
  x,y,w,h=map(int,t['box']); cv2.rectangle(edges,(max(0,x-2),max(0,y-2)),(x+w+2,y+h+2),0,-1)
 edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 rs=[]
 for c in sorted(contours,key=cv2.contourArea,reverse=True):
  x,y,w,h=cv2.boundingRect(c)
  if w<9 or h<9 or w*h<110 or w*h>a.shape[0]*a.shape[1]*.85: continue
  # A photo is one visual element. Do not interpret shoes, faces, highlights, or
  # other contours inside it as icons.
  if any(r['kind']=='image' and coverage([x,y,w,h],r['box'])>.82 for r in rs):continue
  if any(iou([x,y,w,h],r['box'])>.8 or (w<85 and h<85 and r['kind']=='icon' and abs(x+w/2-r['box'][0]-r['box'][2]/2)<3 and abs(y+h/2-r['box'][1]-r['box'][3]/2)<3) for r in rs): continue
  if any(coverage([x,y,w,h],t['box'])>.6 for t in texts): continue
  contained_text=[t for t in texts if coverage(t['box'],[x,y,w,h])>.6]
  contains_text=bool(contained_text)
  if photo_like(a,[x,y,w,h]):kind='image'
  elif 10<=w<=85 and 10<=h<=85 and not contains_text and max(w,h)/max(1,min(w,h))<=1.8:kind='icon'
  else:kind='component'
  label={'icon':'图标区域','image':'图片区域','component':'组件区域'}[kind]
  if kind=='component' and contained_text:label+='：'+' / '.join(t['text'] for t in contained_text[:2])
  rs.append({'box':[x,y,w,h],'kind':kind,'text':label,'confidence':.7})
  if len(rs)>=180: break
 return rs

def feature(a,b):
 p=crop(a,b)
 if not p.size: return np.zeros((24,24),np.float32)
 return cv2.resize(cv2.Canny(cv2.cvtColor(p,cv2.COLOR_RGB2GRAY),40,100),(24,24)).astype(float)/255

def match(a,b,aa,bb,cross=False):
 if not a or not b: return [],list(range(len(a))),list(range(len(b)))
 costs=np.ones((len(a),len(b)))*9
 fa=[feature(aa,r['box']) for r in a]; fb=[feature(bb,r['box']) for r in b]
 for i,r in enumerate(a):
  x,y,w,h=r['box']
  for j,s in enumerate(b):
   if r['kind']!=s['kind']: continue
   xx,yy,ww,hh=s['box']
   dist=math.hypot((x-xx)/max(aa.shape[1],bb.shape[1]),(y-yy)/max(600,aa.shape[0],bb.shape[0]))
   size=abs(math.log(max(1,w)/max(1,ww)))+abs(math.log(max(1,h)/max(1,hh)))
   if r['kind']=='text':
    first_text=normalized_text(r['text']);second_text=normalized_text(s['text'])
    similarity=SequenceMatcher(None,first_text,second_text).ratio();length_ratio=min(len(first_text),len(second_text))/max(1,len(first_text),len(second_text))
    # Do not pair a complete line with one OCR fragment: that produces cropped
    # evidence boxes and measurements for different visual targets.
    if similarity<.65 or length_ratio<.82: continue
    costs[i,j]=(1-similarity)*2+dist*.8+min(size,2)*.08
   else:
    shape_delta=float(np.mean(np.abs(fa[i]-fb[j])))
    shape_cosine=float(np.sum(fa[i]*fb[j])/(np.linalg.norm(fa[i])*np.linalg.norm(fb[j])+1e-6))
    aspect_delta=abs(math.log(max(.01,w/max(1,h))/max(.01,ww/max(1,hh))))
    if r['kind']=='icon' and (shape_cosine<.58 or shape_delta>.32 or aspect_delta>.38):continue
    has_label='：' in r['text'];other_has_label='：' in s['text']
    if r['kind']=='component' and (aspect_delta>.5 or has_label!=other_has_label or (not has_label and shape_cosine<.52)):continue
    if r['kind']=='component' and '：' in r['text'] and '：' in s['text']:
     label_similarity=SequenceMatcher(None,normalized_text(r['text'].split('：',1)[1]),normalized_text(s['text'].split('：',1)[1])).ratio()
     if label_similarity<.72:continue
    costs[i,j]=dist*1.4+min(size,3)*.2+shape_delta*.4
 pairs=[]
 for i,j in zip(*linear_sum_assignment(costs)):
  if costs[i,j]<(.52 if a[i]['kind']=='text' else .52): pairs.append((int(i),int(j),float(costs[i,j])))
 return pairs,[i for i in range(len(a)) if i not in {x[0] for x in pairs}],[j for j in range(len(b)) if j not in {x[1] for x in pairs}]

def fragmented_text_counterpart(r,others):
 """Treat same-row substring OCR boxes as one visual text target."""
 target=normalized_text(r.get('text',''))
 if len(target)<2:return False
 x,y,w,h=r['box'];center=y+h/2
 for other in others:
  if other.get('kind')!='text':continue
  xx,yy,ww,hh=other['box'];other_text=normalized_text(other.get('text',''))
  if len(other_text)<2 or abs(center-(yy+hh/2))>max(10,max(h,hh)*1.5):continue
  horizontal_gap=max(0,max(x,xx)-min(x+w,xx+ww))
  if horizontal_gap>max(12,max(h,hh)*6):continue
  if target in other_text or other_text in target:return True
 return False

def background_color(a,b):
 x,y,w,h=[int(round(v)) for v in b];pad=4
 x0=max(0,x-pad);y0=max(0,y-pad);x1=min(a.shape[1],x+w+pad);y1=min(a.shape[0],y+h+pad)
 p=a[y0:y1,x0:x1];mask=np.ones(p.shape[:2],bool)
 mask[max(0,y-y0):min(p.shape[0],y+h-y0),max(0,x-x0):min(p.shape[1],x+w-x0)]=False
 ring=p[mask]
 return np.median(ring,axis=0) if len(ring) else np.array([255,255,255])

def glyph_mask(a,b):
 p=crop(a,b)
 if not p.size:return p,np.zeros((0,0),np.uint8)
 bg=background_color(a,b);distance=np.linalg.norm(p.astype(float)-bg,axis=2)
 values=np.clip(distance[distance>5],0,255).astype(np.uint8)
 if len(values)<5:return p,np.zeros(p.shape[:2],np.uint8)
 threshold=max(18,float(cv2.threshold(values,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]))
 raw=(distance>threshold).astype(np.uint8)
 count,labels,stats,_=cv2.connectedComponentsWithStats(raw,8);clean=np.zeros_like(raw);ph,pw=raw.shape
 for index in range(1,count):
  x,y,w,h,area=stats[index];fill=area/max(1,w*h)
  horizontal_rule=w>=pw*.55 and h<=max(3,ph*.1)
  vertical_rule=h>=ph*.65 and w<=max(3,pw*.035)
  frame_rule=x<=1 and y<=1 and x+w>=pw-1 and y+h>=ph-1 and fill<.35
  if area<2 or horizontal_rule or vertical_rule or frame_rule:continue
  clean[labels==index]=1
 return p,clean

def glyph_height(a,b):
 _,mask=glyph_mask(a,b)
 if not mask.size or not mask.any():return None
 # Measure occupied glyph rows. This avoids treating padding as font size and is
 # more stable than individual connected components, whose strokes may merge or
 # split with antialiasing. For multiline OCR, use the tallest ink band only.
 active=np.flatnonzero(mask.sum(axis=1)>=max(2,round(mask.shape[1]*.01)))
 if not len(active):return None
 runs=[];start=previous=int(active[0])
 for row in active[1:]:
  row=int(row)
  if row>previous+1:runs.append(previous-start+1);start=row
  previous=row
 runs.append(previous-start+1)
 return float(max(runs))

def refine_text_box(a,b):
 # OCR locates text; foreground pixels determine the visible ink bounds.
 x,y,w,h=b;pad=4
 box=[max(0,int(x)-pad),max(0,int(y)-pad),int(w)+pad*2+1,int(h)+pad*2+1]
 p,mask=glyph_mask(a,box)
 if not p.size:return b
 # Keep one visual text line. OCR observations can accidentally include a nearby
 # status icon or the next line even when their recognized string belongs to only
 # one line.
 row_counts=mask.sum(axis=1);active=np.flatnonzero(row_counts>=max(2,round(mask.shape[1]*.03)))
 if len(active):
  bands=[];start=previous=int(active[0])
  for row in active[1:]:
   row=int(row)
   if row>previous+1:bands.append((start,previous));start=row
   previous=row
  bands.append((start,previous))
  top,bottom=max(bands,key=lambda band:int(mask[band[0]:band[1]+1].sum()))
  line=np.zeros_like(mask);line[top:bottom+1]=mask[top:bottom+1];mask=line
 col_counts=mask.sum(axis=0);active=np.flatnonzero(col_counts>0)
 if len(active):
  groups=[];start=previous=int(active[0]);join_gap=max(3,round(mask.shape[0]*.22))
  for column in active[1:]:
   column=int(column)
   if column>previous+join_gap+1:groups.append((start,previous));start=column
   previous=column
  groups.append((start,previous))
  anchor=max(range(len(groups)),key=lambda index:int(mask[:,groups[index][0]:groups[index][1]+1].sum()))
  anchor_ink=int(mask[:,groups[anchor][0]:groups[anchor][1]+1].sum())
  first=last=anchor;neighbor_gap=max(10,round((bottom-top+1)*1.25)) if len(active) else 10
  while first>0 and groups[first][0]-groups[first-1][1]-1<=neighbor_gap and int(mask[:,groups[first-1][0]:groups[first-1][1]+1].sum())>=anchor_ink*.2:first-=1
  while last+1<len(groups) and groups[last+1][0]-groups[last][1]-1<=neighbor_gap and int(mask[:,groups[last+1][0]:groups[last+1][1]+1].sum())>=anchor_ink*.2:last+=1
  left,right=groups[first][0],groups[last][1]
  line=np.zeros_like(mask);line[:,left:right+1]=mask[:,left:right+1];mask=line
 yy,xx=np.nonzero(mask)
 if len(xx)<8:return b
 refined=[box[0]+int(xx.min()),box[1]+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
 if refined[2]<w*.22 or refined[3]<h*.18:return b
 return refined

def sampled_color(a,b,text=False):
 p=crop(a,b)
 if p.size<30:return None
 if not text and p.shape[0]>6 and p.shape[1]>6:p=p[3:-3,3:-3]
 pixels=p.reshape(-1,3)
 if text:
  bg=background_color(a,b);distance=np.linalg.norm(pixels.astype(float)-bg,axis=1)
  # Select solid foreground rather than antialiased edge pixels or the surrounding fill.
  pixels=pixels[distance>=max(30,float(np.percentile(distance,80)))]
  if len(pixels)<5:return None
 quant=(pixels//8)*8
 values,counts=np.unique(quant,axis=0,return_counts=True)
 top=values[np.argmax(counts)];pixels=pixels[np.all(quant==top,axis=1)]
 return np.median(pixels,axis=0)

def uniform_text_color(a,b):
 p=crop(a,b)
 if not p.size:return False
 pixels=p.reshape(-1,3);bg=background_color(a,b);distance=np.linalg.norm(pixels.astype(float)-bg,axis=1)
 pixels=pixels[distance>=max(24,float(np.percentile(distance,85)))]
 if len(pixels)<8:return False
 hsv=cv2.cvtColor(pixels.reshape(-1,1,3).astype(np.uint8),cv2.COLOR_RGB2HSV).reshape(-1,3)
 colored=hsv[:,1]>=48;colored_share=float(colored.mean())
 # A line containing both colored and neutral words is not one color target.
 if .18<colored_share<.82:return False
 if colored_share>=.82:
  hues=hsv[colored,0].astype(float)*2
  center=float(np.angle(np.mean(np.exp(1j*np.deg2rad(hues))),deg=True)%360)
  spread=np.minimum(abs(hues-center),360-abs(hues-center))
  if float(np.percentile(spread,80))>18:return False
 return True

def color_evidence(a,b):
 lab=rgb2lab(np.array([a,b],dtype=float).reshape(1,2,3)/255)[0]
 de=float(deltaE_ciede2000(lab[0][None,None,:],lab[1][None,None,:])[0,0])
 hsv=cv2.cvtColor(np.array([[a,b]],dtype=np.uint8),cv2.COLOR_RGB2HSV)[0]
 hue=abs(float(hsv[0,0])-float(hsv[1,0]))*2;hue=min(hue,360-hue)
 saturation_delta=abs(float(hsv[0,1])-float(hsv[1,1]))/255
 value_delta=abs(float(hsv[0,2])-float(hsv[1,2]))/255
 chroma=min(float(hsv[0,1]),float(hsv[1,1]))/255
 return {'deltaE':de,'hueDelta':hue,'saturationDelta':saturation_delta,'valueDelta':value_delta,'chroma':chroma}

def significant_color_change(a,b,threshold=12):
 evidence=color_evidence(a,b)
 # Device gamut and screenshot rendering often shift brightness/saturation a little.
 # Report hue changes for colored UI, or a very large perceptual change regardless of hue.
 changed=evidence['deltaE']>=threshold and ((evidence['chroma']>=.12 and evidence['hueDelta']>=12) or evidence['deltaE']>=22 or evidence['valueDelta']>=.28)
 return changed,evidence

def weight_features(a,b):
 p,clean=glyph_mask(a,b)
 if not p.size or not clean.any():return None
 yy,xx=np.nonzero(clean)
 if len(xx)<8:return None
 tight=clean[yy.min():yy.max()+1,xx.min():xx.max()+1]
 d=cv2.distanceTransform(tight,cv2.DIST_L2,5);vals=d[d>0]
 if len(vals)<8:return None
 return {'strokeWidth':float(np.percentile(vals,75)*2),'inkDensity':float(tight.mean()),'inkArea':int(tight.sum())}

def weight(a,b):
 f=weight_features(a,b)
 return f['strokeWidth'] if f else None

def significant_weight_change(a,b,tolerance=1):
 stroke=math.log(max(.01,b['strokeWidth'])/max(.01,a['strokeWidth']))
 density=math.log(max(.01,b['inkDensity'])/max(.01,a['inkDensity']))
 area=math.log(max(1,b.get('inkArea',1))/max(1,a.get('inkArea',1))) if 'inkArea' in a and 'inkArea' in b else 0
 # Stroke width alone is unstable under scaling and antialiasing. A clear fill-density
 # change is required so small screenshot rendering differences are not called weight changes.
 # Ink area supplies a fallback when antialiasing leaves the distance-transform
 # stroke estimate unchanged, as can happen between Regular and Medium CJK fonts.
 changed=(abs(density)>.20*tolerance and (stroke*density>=0 or abs(stroke)<.05)) or (abs(area)>.30*tolerance and area*density>0)
 return changed,stroke,density

def overlay_boxes(a,texts):
 boxes=[]
 for t in texts:
  text=t.get('text','').replace(' ','');x,y,w,h=t['box']
  top_identifier=y<max(55,a.shape[0]*.07) and bool(LONG_TEST_ID.search(text))
  if top_identifier:
   strip_height=min(a.shape[0],max(40,y+h+12))
   boxes.append([0,0,a.shape[1],strip_height]);continue
  if TEST_OVERLAY.search(text):
   px=max(22,h*2.4);py=max(14,h*1.6)
   x0=max(0,x-px);y0=max(0,y-py);x1=min(a.shape[1],x+w+px);y1=min(a.shape[0],y+h+py)
   boxes.append([x0,y0,x1-x0,y1-y0])
 # CBase/LEGO inspection builds can also show an unlabeled floating gear control.
 # Detect only dark, compact controls near the right edge to avoid masking App UI.
 gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);dark=(gray<185).astype(np.uint8)*255
 dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
 contours,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 for contour in contours:
  x,y,w,h=cv2.boundingRect(contour)
  if x<a.shape[1]*.76 or not (22<=w<=68 and 22<=h<=68) or max(w,h)/max(1,min(w,h))>1.45:continue
  patch=crop(gray,[x,y,w,h]);dark_share=float((patch<185).mean()) if patch.size else 0
  if dark_share<.28:continue
  pad=6;boxes.append([max(0,x-pad),max(0,y-pad),min(a.shape[1],x+w+pad)-max(0,x-pad),min(a.shape[0],y+h+pad)-max(0,y-pad)])
 merged=[]
 for box in boxes:
  x,y,w,h=box
  for i,other in enumerate(merged):
   xx,yy,ww,hh=other
   if min(x+w,xx+ww)>max(x,xx) and min(y+h,yy+hh)>max(y,yy):
    x0=min(x,xx);y0=min(y,yy);x1=max(x+w,xx+ww);y1=max(y+h,yy+hh)
    merged[i]=[x0,y0,x1-x0,y1-y0];break
  else:merged.append(box)
 return merged

def adaptive_geometry(box,design_width,implementation_width):
 x,_,w,_=box;left=x;right=design_width-x-w;center=x+w/2-design_width/2
 if w/design_width>=.55:return left,max(1,implementation_width-left-right),'左右边距保持'
 if abs(center)<=max(12,design_width*.06):return implementation_width/2+center-w/2,w,'居中保持'
 if x+w/2<design_width/2:return left,w,'左侧锚定'
 return implementation_width-right-w,w,'右侧锚定'

def union_box(a,b):
 x=min(a[0],b[0]);y=min(a[1],b[1]);right=max(a[0]+a[2],b[0]+b[2]);bottom=max(a[1]+a[3],b[1]+b[3])
 return [x,y,right-x,bottom-y]

def immediate_component_parents(items):
 parents={}
 for index,item in enumerate(items):
  if item['kind'] not in ('text','icon'):continue
  area=item['box'][2]*item['box'][3];candidates=[]
  for parent_index,parent in enumerate(items):
   if parent['kind']!='component' or parent_index==index:continue
   parent_area=parent['box'][2]*parent['box'][3]
   if parent_area>area*1.15 and coverage(item['box'],parent['box'])>=.82:
    candidates.append((parent_area,parent_index))
  if candidates:parents[index]=min(candidates)[1]
 return parents

def spacing_differences(regions,pairs,design_width,implementation_width,cross,threshold):
 matched=[(i,j) for i,j,_ in pairs];candidates=[]
 design_parents=immediate_component_parents(regions[0]);implementation_parents=immediate_component_parents(regions[1])
 for i,j in matched:
  first=regions[0][i];qfirst=regions[1][j];x,y,w,h=first['box'];qx,qy,qw,qh=qfirst['box']
  if first['kind'] not in ('text','icon') or i not in design_parents or j not in implementation_parents:continue
  px,pw,_=adaptive_geometry(first['box'],design_width,implementation_width) if cross else (x,w,'同宽坐标')
  nearest={'horizontal':None,'vertical':None}
  for k,l in matched:
   if k==i:continue
   second=regions[0][k];qsecond=regions[1][l]
   if second['kind'] not in ('text','icon') or 'text' not in (first['kind'],second['kind']):continue
   if design_parents.get(k)!=design_parents[i] or implementation_parents.get(l)!=implementation_parents[j]:continue
   if coverage(first['box'],second['box'])>.12 or coverage(second['box'],first['box'])>.12:continue
   xx,yy,ww,hh=second['box'];qxx,qyy,qww,qhh=qsecond['box']
   pxx,pww,_=adaptive_geometry(second['box'],design_width,implementation_width) if cross else (xx,ww,'同宽坐标')
   same_row=abs((y+h/2)-(yy+hh/2))<=max(7,min(h,hh)*.65)
   q_same_row=abs((qy+qh/2)-(qyy+qhh/2))<=max(9,min(qh,qhh)*.8)
   if same_row and q_same_row and xx>=x+w-1 and qxx>=qx+qw-1:
    expected=pxx-(px+pw);actual=qxx-(qx+qw)
    if -1<=expected<=220 and -1<=actual<=260 and (nearest['horizontal'] is None or expected<nearest['horizontal'][0]):nearest['horizontal']=(expected,actual,second,qsecond)
   overlap=max(0,min(x+w,xx+ww)-max(x,xx));qoverlap=max(0,min(qx+qw,qxx+qww)-max(qx,qxx))
   same_column=overlap>=min(w,ww)*.25 or abs((x+w/2)-(xx+ww/2))<=min(w,ww)*.45
   q_same_column=qoverlap>=min(qw,qww)*.2 or abs((qx+qw/2)-(qxx+qww/2))<=min(qw,qww)*.55
   if same_column and q_same_column and yy>=y+h-1 and qyy>=qy+qh-1:
    expected=yy-(y+h);actual=qyy-(qy+qh)
    if -1<=expected<=280 and -1<=actual<=340 and (nearest['vertical'] is None or expected<nearest['vertical'][0]):nearest['vertical']=(expected,actual,second,qsecond)
  for axis,data in nearest.items():
   if not data:continue
   expected,actual,second,qsecond=data;delta=actual-expected
   if abs(delta)<threshold or abs(delta)/max(8,abs(expected))<.12:continue
   r={'box':union_box(first['box'],second['box']),'kind':'component','text':f'{first["text"]} ↔ {second["text"]}'}
   s={'box':union_box(qfirst['box'],qsecond['box']),'kind':'component','text':r['text']}
   candidates.append({'axis':axis,'expected':expected,'actual':actual,'delta':delta,'r':r,'s':s,'indices':(i,k),'priority':2 if 'icon' in (first['kind'],second['kind']) else 1})
 candidates.sort(key=lambda item:(item['priority'],abs(item['delta'])),reverse=True)
 selected=[]
 for candidate in candidates:
  if any(iou(candidate['r']['box'],kept['r']['box'])>.7 and abs(candidate['delta']-kept['delta'])<2 for kept in selected):continue
  selected.append(candidate)
  if len(selected)>=12:break
 return selected

def alignment_differences(regions,pairs,design_width,implementation_width,cross,threshold):
 matched=[(i,j) for i,j,_ in pairs];candidates=[]
 for offset,(i,j) in enumerate(matched):
  first=regions[0][i];qfirst=regions[1][j];x,y,w,h=first['box'];qx,qy,qw,qh=qfirst['box']
  px,pw,_=adaptive_geometry(first['box'],design_width,implementation_width) if cross else (x,w,'同宽坐标')
  for k,l in matched[offset+1:]:
   second=regions[0][k];qsecond=regions[1][l]
   if coverage(first['box'],second['box'])>.12 or coverage(second['box'],first['box'])>.12:continue
   xx,yy,ww,hh=second['box'];qxx,qyy,qww,qhh=qsecond['box']
   pxx,pww,_=adaptive_geometry(second['box'],design_width,implementation_width) if cross else (xx,ww,'同宽坐标')
   x_distance=abs((x+w/2)-(xx+ww/2));y_distance=abs((y+h/2)-(yy+hh/2))
   qx_distance=abs((qx+qw/2)-(qxx+qww/2));qy_distance=abs((qy+qh/2)-(qyy+qhh/2))
   projected_x_distance=abs((px+pw/2)-(pxx+pww/2))
   vertical_stack=abs((y+h/2)-(yy+hh/2))>max(h,hh)*.8 and abs((y+h/2)-(yy+hh/2))<320
   horizontal_row=abs((x+w/2)-(xx+ww/2))>max(w,ww)*.65 and abs((x+w/2)-(xx+ww/2))<420
   if vertical_stack and projected_x_distance<=max(3,threshold*.55) and qx_distance-projected_x_distance>=threshold:
    axis='horizontal_center';expected=projected_x_distance;actual=qx_distance
   elif horizontal_row and y_distance<=max(3,threshold*.55) and qy_distance-y_distance>=threshold:
    axis='vertical_center';expected=y_distance;actual=qy_distance
   else:continue
   r={'box':union_box(first['box'],second['box']),'kind':'component','text':f'{first["text"]} ↔ {second["text"]}'}
   s={'box':union_box(qfirst['box'],qsecond['box']),'kind':'component','text':r['text']}
   candidates.append({'axis':axis,'expected':expected,'actual':actual,'delta':actual-expected,'r':r,'s':s,'indices':(i,k)})
 candidates.sort(key=lambda item:abs(item['delta']),reverse=True)
 selected=[]
 for candidate in candidates:
  if any(iou(candidate['r']['box'],kept['r']['box'])>.7 and candidate['axis']==kept['axis'] for kept in selected):continue
  selected.append(candidate)
  if len(selected)>=12:break
 return selected

def analyze(design,implementation,config,out,progress,check):
 out=Path(out); warnings=[]; selected=config.get('categories',list(CATEGORIES)); start=time.monotonic()
 cov={c:{'status':'unavailable','reason':'用户未选择此类别'} for c in CATEGORIES}
 def stage(i): check(); progress(i)
 stage(0)
 originals=[np.array(Image.open(p).convert('RGB')) for p in (design,implementation)]
 imgs=[]; maps=[]; metadata=[]; logical_widths=[]
 for side,a in zip(('design','implementation'),originals):
  opt=config.get(side,{})
  b=opt.get('crop') or [0,0,a.shape[1],a.shape[0]]
  x,y,w,h=b
  if x<0 or y<0 or w<20 or h<20 or x+w>a.shape[1]+1 or y+h>a.shape[0]+1: raise ValueError('裁剪区域必须位于原图内，且宽高至少 20px。')
  lw=opt.get('logicalWidth');source='user_confirmed' if lw or opt.get('effectiveScale') else 'unknown'
  if not lw and not opt.get('effectiveScale'):
   candidates=[v for v in MOBILE_WIDTHS if 1<=a.shape[1]/v<=4 and abs(a.shape[1]/v-round(a.shape[1]/v))<.01]
   if len(candidates)==1:lw=candidates[0];source='auto_inferred'
  # Width is of the original viewport; crop does not change its density.
  s=opt.get('effectiveScale') or (a.shape[1]/lw if lw else 1)
  if s<=0: raise ValueError('有效倍率必须大于 0。')
  part=crop(a,b); target=(max(1,round(part.shape[1]/s)),max(1,round(part.shape[0]/s)))
  if target[0]>1600 or target[1]>24000 or target[0]*target[1]>16000000: raise ValueError('分析画布过大，请选择逻辑宽度或裁剪范围后重试。')
  imgs.append(cv2.resize(part,target,interpolation=cv2.INTER_AREA if s>=1 else cv2.INTER_CUBIC))
  maps.append({'scale':s,'origin':[x,y],'originalSize':[a.shape[1],a.shape[0]],'logicalKnown':bool(lw or opt.get('effectiveScale'))})
  metadata.append({'logicalWidth':lw,'effectiveScale':s,'source':source,'platform':opt.get('platform','unknown'),'safeTop':opt.get('safeTop',0),'safeBottom':opt.get('safeBottom',0)})
  logical_widths.append(lw)
 aa,bb=imgs
 for side,a in zip(('design','implementation'),imgs): Image.fromarray(a).save(out/f'{side}-normalized.png')
 known=all(m['logicalKnown'] for m in maps)
 widths=logical_widths
 cross=bool(all(widths) and widths[0]!=widths[1])
 if aa.shape[1]!=bb.shape[1] and not all(widths):
  return {'status':'needs_alignment','issues':[],'coverage':cov,'warnings':['两张图片宽度不同，无法确认截图倍率。请选择两侧真实逻辑宽度，或填写有效倍率后重跑。'],'transforms':maps,'metadata':metadata,'mode':'unknown','normalizedSizes':[[a.shape[1],a.shape[0]] for a in imgs]}
 conditions=config.get('conditions',{})
 incompatible=conditions.get('different',False)
 inferred=[f'{side} {m["logicalWidth"]}' for side,m in zip(('设计图','开发截图'),metadata) if m['source']=='auto_inferred']
 if inferred:warnings.append('已按常见移动端宽度自动推断：'+'、'.join(inferred)+'；请在结果中核对。')
 if cross: warnings.append('跨宽适配估算：根据元素的左侧、右侧、居中或左右边距关系预测目标位置；换行和断点规则仍需人工复核。')
 if incompatible: warnings.append('两侧主题、字体缩放或页面状态不同，所有观察保留为待复核参考。')
 if not known: warnings.append('逻辑宽度未确认，所有测量使用设计图基准 px，不等同 App 逻辑单位。')
 # Ignore masks never change coordinate origins.
 masks=[]
 for side,a,m in zip(('design','implementation'),imgs,maps):
  mask=np.ones(a.shape[:2],np.uint8)
  for b in config.get(side,{}).get('ignore',[]):
   x,y,w,h=b; x=round((x-m['origin'][0])/m['scale']);y=round((y-m['origin'][1])/m['scale']);w=round(w/m['scale']);h=round(h/m['scale'])
   cv2.rectangle(mask,(x,y),(x+w,y+h),0,-1)
  masks.append(mask)
 stage(1)
 h=min(aa.shape[0],bb.shape[0]); w=min(aa.shape[1],bb.shape[1])
 delta=np.max(np.abs(aa[:h,:w].astype(float)-bb[:h,:w].astype(float)),axis=2)
 valid=(masks[0][:h,:w]&masks[1][:h,:w]).astype(bool)
 heat=np.ones((max(a.shape[0] for a in imgs),max(a.shape[1] for a in imgs),4),np.uint8)*0
 heat[:h,:w,0]=240; heat[:h,:w,1]=68; heat[:h,:w,2]=56
 heat[:h,:w,3]=np.where(valid & (delta>18),np.clip(delta*1.5,25,210),0).astype(np.uint8)
 Image.fromarray(heat).save(out/'heatmap.png')
 stage(2)
 texts=[]; ocr_errors=[]
 for a in imgs:
  t,e=recognize(a,check);texts.append(t);ocr_errors.append(e)
  if e and e not in warnings: warnings.append(e)
 ignored=[]
 if config.get('autoIgnoreOverlays',True):
  for index,(a,t) in enumerate(zip(imgs,texts)):
   boxes=overlay_boxes(a,t);ignored.append(boxes)
   for x,y,bw,bh in boxes:cv2.rectangle(masks[index],(round(x),round(y)),(round(x+bw),round(y+bh)),0,-1)
   texts[index]=[item for item in t if not any(coverage(item['box'],box)>.15 for box in boxes)]
  count=sum(map(len,ignored))
  if count:warnings.append(f'已自动忽略 {count} 处测试浮层或顶部测试标识。')
  valid=(masks[0][:h,:w]&masks[1][:h,:w]).astype(bool)
  heat[:h,:w,3]=np.where(valid & (delta>18),np.clip(delta*1.5,25,210),0).astype(np.uint8)
  Image.fromarray(heat).save(out/'heatmap.png')
 rs=[]
 for a,t,mask in zip(imgs,texts,masks):
  combined=t+regions(a,t)
  rs.append([r for r in combined if (lambda p:p.size and p.mean()>.8)(crop(mask,r['box']))])
 pairs,missing,extra=match(rs[0],rs[1],aa,bb,cross)
 if max(len(rs[0]),len(rs[1]))>5 and len(pairs)/max(len(rs[0]),len(rs[1]))<.2:
  warnings.append('对应区域过少，图片可能不是同一页面或状态；未匹配项仅供人工复核。')
 stage(3)
 tolerance={'strict':.65,'standard':1,'loose':1.8}[config.get('tolerance','standard')]
 pos=config.get('positionThreshold',2)*tolerance; spacing_threshold=config.get('spacingThreshold',4)*tolerance; size=config.get('sizeThreshold',3)/100*tolerance; color=max(10,max(12,config.get('colorThreshold',12))*tolerance)
 unit='逻辑单位' if known else '设计图基准 px'
 issues=[]
 def original_box(b,index):
  m=maps[index]; x,y,w,h=b
  return {'x':round(x*m['scale']+m['origin'][0],2),'y':round(y*m['scale']+m['origin'][1],2),'width':round(w*m['scale'],2),'height':round(h*m['scale'],2)}
 def add(cat,title,r,s,measurements,method,suggestion,confidence='medium',severity='medium'):
  if cat not in selected and cat in CATEGORIES:return
  idx=len(issues)+1
  issue={'id':f'I{idx:03}','category':cat,'title':title,'regionLabel':r['text'][:50] if r else s['text'][:50],'severity':severity,'confidence':'low' if incompatible else confidence,'rationale':'局部图像测量与候选匹配；截图不能确认源码属性。','status':'pending','designBBox':original_box(r['box'],0) if r else None,'implementationBBox':original_box(s['box'],1) if s else None,'measurements':measurements,'method':method,'suggestion':suggestion,'reviewHistory':[]}
  for index,(key,item) in enumerate((('design',r),('implementation',s))):
   if item:
    b=item['box']; pad=14; b=[max(0,b[0]-pad),max(0,b[1]-pad),b[2]+pad*2,b[3]+pad*2]
    Image.fromarray(crop(imgs[index],b)).save(out/f'{issue["id"]}-{key}.png')
  issues.append(issue)
 def metric(name,a,b,u=unit,estimate=False):
  return {'metric':name,'designValue':round(float(a),2),'implementationValue':round(float(b),2),'delta':round(float(b-a),2),'unit':u,'source':'local_image_measurement','certainty':'estimated' if estimate else 'measured'}
 for c in selected:
  cov[c]={'status':'checked','reason':'已检查可匹配的局部候选；不代表覆盖所有 UI 元素。'}
  if c in ('text_size','font_weight'):
   cov[c]={'status':'partial' if not ocr_errors[0] and not ocr_errors[1] else 'unavailable','reason':'基于 OCR 可见字形与边界估计，需人工复核。' if not any(ocr_errors) else 'OCR 不完整，无法保证文字覆盖。'}
  if c=='icon_shape':cov[c]={'status':'partial','reason':'基于非文本小轮廓候选，并排除照片及其内部轮廓；无法保证识别所有图标。'}
  if cross and c in ('component_position','alignment','spacing'):
   cov[c]={'status':'partial','reason':'按左/右/居中/左右边距锚点估算目标宽度适配，断点与业务布局规则需人工复核。'}
 sameheight=abs(aa.shape[0]-bb.shape[0])<2
 if not sameheight:
  warnings.append('两侧内容高度不同：底部锚定区域可在设置中指定；未知锚定的纵向偏移仅作为候选。')
 spacing_changes=spacing_differences(rs,pairs,aa.shape[1],bb.shape[1],cross,spacing_threshold)
 spacing_members={index for item in spacing_changes for index in item['indices']}
 alignment_changes=alignment_differences(rs,pairs,aa.shape[1],bb.shape[1],cross,pos)
 for i,j,cost in pairs:
  check(); r=rs[0][i];s=rs[1][j]; x,y,rw,rh=r['box'];xx,yy,sw,sh=s['box'];kind=r['kind']
  if min(y+rh,yy+sh)>h and (y>=h or yy>=h):continue
  # OCR localization can drift even when the image content is unchanged.
  ux=min(x,xx);uy=min(y,yy);union=[ux,uy,max(x+rw,xx+sw)-ux,max(y+rh,yy+sh)-uy]
  pa=crop(aa,union);pb=crop(bb,union)
  if not cross and pa.shape==pb.shape and np.array_equal(pa,pb):continue
  conf='high' if cost<.12 and not incompatible else 'medium'
  expected_x=x;expected_w=rw;adaptation='同宽坐标'
  if cross:expected_x,expected_w,adaptation=adaptive_geometry(r['box'],aa.shape[1],bb.shape[1])
  dx=xx-expected_x;dy=yy-y
  anchor=config.get('bottomAnchorFrom')
  bottom=anchor is not None and y>=anchor
  yp=(aa.shape[0]-y-rh-config.get('design',{}).get('safeBottom',0)) if bottom else y
  yq=(bb.shape[0]-yy-sh-config.get('implementation',{}).get('safeBottom',0)) if bottom else yy
  dy=yq-yp
  if abs(dx)>pos or abs(dy)>pos or (cross and kind!='text' and abs(sw-expected_w)>max(2,expected_w*size)):
   dedicated_relation=i in spacing_members
   if not dedicated_relation:
    cat='component_position'
    title='跨宽组件排布不符合预期' if cross else '文字或按钮组件排布位置变化' if kind=='text' else '组件位置偏移'
    if bottom:title='底部锚定间距偏差'
    geometry=[metric('适配预期左边缘 x' if cross else '左边缘 x',expected_x,xx,estimate=cross),metric('距底部安全区' if bottom else '上边缘 y',yp,yq,estimate=cross)]
    if cross and kind!='text':geometry.append(metric('适配预期宽度',expected_w,sw,estimate=True))
    suggestion='检查目标宽度下的约束、边距、居中或拉伸规则。' if cross else '检查对应区域的内边距、约束或相对间距。'
    add(cat,title,r,s,geometry,(adaptation+'估算 · ' if cross else '')+('OCR 边界' if kind=='text' else '轮廓边界'),suggestion,conf if sameheight or bottom else 'low')
  if not cross and known:
   da=config.get('design',{}); ib=config.get('implementation',{})
   dt=da.get('safeTop',0); db=da.get('safeBottom',0);it=ib.get('safeTop',0);bottomInset=ib.get('safeBottom',0)
   design_inside=y>=dt and y+rh<=aa.shape[0]-db
   violation=(it>0 and yy<it and yy+sh>it) or (bottomInset>0 and yy+sh>bb.shape[0]-bottomInset and yy<bb.shape[0]-bottomInset)
   if design_inside and violation:
    add('component_position','疑似侵入安全区',r,s,[metric('上边缘 y',y,yy),metric('下边缘 y',y+rh,yy+sh)],'用户提供的安全区与可见元素边界','核对 App 内容与安全区约束；需确认该区域属于 App 内容。','low')
  if kind in ('component','image') and not cross and (abs(sw/rw-1)>max(size,.05) or abs(sh/rh-1)>max(size,.05)) and abs(sw-rw)+abs(sh-rh)>2:
   image=kind=='image';name='图片' if image else '按钮或组件'
   add('component_position',name+'尺寸不一致',r,s,[metric(name+'宽度',rw,sw),metric(name+'高度',rh,sh)],'图片外边界' if image else '包含文字或横向矩形的组件轮廓','核对图片容器和裁切规则。' if image else '核对按钮或组件的宽高、内边距、最小尺寸与布局约束。',conf)
  if kind=='text':
   design_glyph_height=glyph_height(aa,r['box']);implementation_glyph_height=glyph_height(bb,s['box']);text_size_threshold=max(size,.07)
   box_ink_consistent=bool(design_glyph_height and implementation_glyph_height and abs(design_glyph_height/max(1,rh)-implementation_glyph_height/max(1,sh))<=.22)
   same_text=SequenceMatcher(None,normalized_text(r['text']),normalized_text(s['text'])).ratio()>=.9
   height_scale=implementation_glyph_height/design_glyph_height-1 if design_glyph_height and implementation_glyph_height else 0
   width_scale=sw/rw-1
   width_supports_size=height_scale*width_scale>0 and abs(width_scale)>=.05
   if box_ink_consistent and same_text and width_supports_size and abs(height_scale)>max(text_size_threshold,.12) and abs(implementation_glyph_height-design_glyph_height)>=3:
    add('text_size','疑似字号'+('偏大' if implementation_glyph_height>design_glyph_height else '偏小'),r,s,[metric('字符主体墨迹高度',design_glyph_height,implementation_glyph_height,estimate=True)],'OCR 定位 · 去除按钮边框与长线 · 字符连通域主体高度','核对原生字号和系统字体缩放；测量只使用字符墨迹，不包含行距、上下留白或按钮边框。','medium')
  # Color is only meaningful for the same single-color text or the same icon.
  # Components, photos, and mixed-color OCR lines are not one color target.
  same_color_text=kind=='text' and SequenceMatcher(None,normalized_text(r['text']),normalized_text(s['text'])).ratio()>=.95 and uniform_text_color(aa,r['box']) and uniform_text_color(bb,s['box'])
  color_eligible=kind=='icon' or same_color_text
  ca=sampled_color(aa,r['box'],kind=='text') if color_eligible else None;cb=sampled_color(bb,s['box'],kind=='text') if color_eligible else None
  if ca is not None and cb is not None:
   changed,color_data=significant_color_change(ca,cb,color)
   if changed:
    ha='#'+''.join(f'{int(v):02X}' for v in ca); hb='#'+''.join(f'{int(v):02X}' for v in cb)
    measurements=[{'metric':'截图采样色','designValue':ha,'implementationValue':hb,'delta':round(color_data['deltaE'],2),'unit':'ΔE00','source':'interior_pixel_sampling','certainty':'measured'},{'metric':'色相角差','designValue':0,'implementationValue':round(color_data['hueDelta'],1),'delta':round(color_data['hueDelta'],1),'unit':'°','source':'HSV hue distance','certainty':'estimated'}]
    add('color','文字颜色明显不同' if kind=='text' else '图标颜色明显不同',r,s,measurements,'相同单色文字或形状匹配图标 · 稳定前景采样 · sRGB 归一化 · CIEDE2000 与色相角','核对对应文字或图标色值与透明度；已忽略混合色文字区域及常见设备色域、亮度和饱和度的小幅漂移。',conf)
  weight_text_consistent=kind=='text' and SequenceMatcher(None,normalized_text(r['text']),normalized_text(s['text'])).ratio()>=.9 and box_ink_consistent
  weight_ca=sampled_color(aa,r['box'],True) if weight_text_consistent else None;weight_cb=sampled_color(bb,s['box'],True) if weight_text_consistent else None
  if weight_text_consistent and abs(sh/rh-1)<.35 and abs(sw/rw-1)<.25 and weight_ca is not None and weight_cb is not None and np.linalg.norm(weight_ca-weight_cb)<45:
   wa=weight_features(aa,r['box']);wb=weight_features(bb,s['box'])
   if wa and wb:
    changed_weight,stroke,density=significant_weight_change(wa,wb,tolerance)
    enough_weight_evidence=abs(stroke)>.18*tolerance or min(len(normalized_text(r['text'])),len(normalized_text(s['text'])))>=6
    if changed_weight and enough_weight_evidence:
     direction=density>0
     add('font_weight','疑似字重'+('偏粗' if direction else '偏细'),r,s,[metric('笔画宽度估计',wa['strokeWidth'],wb['strokeWidth'],estimate=True),metric('字形墨色密度',wa['inkDensity'],wb['inkDensity'],'比例',True)],'前景分割 · 笔画距离变换 · 字形墨色密度','核对字体文件、fontWeight、可变字体轴与文字渲染方式。','medium' if max(abs(stroke),abs(density))>.16 else 'low')
  if kind=='icon' and abs((sw/sh)/(rw/rh)-1)>max(size,.06) and abs(sw-rw)+abs(sh-rh)>2:
   add('icon_shape','疑似 Icon 非等比形变',r,s,[metric('宽高比',rw/rh,sw/sh,'比例'),metric('轮廓宽度',rw,sw),metric('轮廓高度',rh,sh)],'保留比例的轮廓测量','核对图标资源与宽高约束；区分图案变化和非等比拉伸。','medium')
 for item in spacing_changes:
  axis='横向' if item['axis']=='horizontal' else '纵向'
  add('spacing',f'组件内{axis}元素间距不一致',item['r'],item['s'],[metric(('适配预期' if cross else '设计稿')+axis+'间距',item['expected'],item['actual'],estimate=cross)],('跨宽锚点估算 · ' if cross else '')+'同一组件内相邻文字或 Icon 的边缘距离','核对组件内部文字与文字、文字与 Icon 之间的 gap、margin 或内边距。','medium')
 for item in alignment_changes:
  relation='上下元素的水平中心线' if item['axis']=='horizontal_center' else '左右元素的垂直中心线'
  add('alignment',f'{relation}未对齐',item['r'],item['s'],[metric(relation+'偏差',item['expected'],item['actual'],estimate=cross)],('跨宽锚点估算 · ' if cross else '')+'元素中心点关系','核对容器的居中、align-items、基线或约束关系。','medium')
 # Do not call offscreen content missing, or flood reports with unmatched contour noise.
 if not cross and not incompatible:
  for index,unmatched in ((0,missing),(1,extra)):
   for k in unmatched:
    r=rs[index][k];x,y,ww,hh=r['box']
    if r['kind']!='text' or r['confidence']<.6 or y+hh>h-2:continue
    # A QA overlay can obscure otherwise identical App content on one side.
    # Suppress the unmatched text when its corresponding location is covered by
    # an automatically ignored overlay in the opposite screenshot.
    if any(coverage(r['box'],box)>.15 for box in ignored[1-index]):continue
    if fragmented_text_counterpart(r,rs[1-index]):continue
    add('missing' if index==0 else 'extra','未匹配的文字，需复核',r if index==0 else None,r if index==1 else None,[],'OCR 未匹配候选','检查是否为内容变化、遮挡或识别失败；不能仅据此确认元素缺失。','low','low')
 # Collapse nested movement duplicates with equal movement, retaining independent style issues.
 movement=[q for q in issues if q['category']=='component_position']
 remove=set()
 for q in issues:
  if q['category'] not in ('alignment','component_position'):continue
  for parent in movement:
   if parent is q:continue
   pb=parent['designBBox'];qb=q['designBBox']
   if pb['width']*pb['height']<=qb['width']*qb['height']:continue
   if coverage([qb[k] for k in ('x','y','width','height')],[pb[k] for k in ('x','y','width','height')])>.9 and all(abs(a['delta']-b['delta'])<1.2 for a,b in zip(q['measurements'],parent['measurements'])):
    remove.add(q['id']);break
 issues=[q for q in issues if q['id'] not in remove]
 stage(4)
 return {'status':'partial' if any(cov[c]['status']!='checked' for c in selected) else 'completed','issues':issues,'coverage':cov,'warnings':warnings,'transforms':maps,'metadata':metadata,'ignoredOverlays':ignored,'mode':'cross_width_reference' if cross else 'same_width','normalizedSizes':[[a.shape[1],a.shape[0]] for a in imgs],'duration':round(time.monotonic()-start,1),'candidateCounts':[len(r) for r in rs],'matchedCandidates':len(pairs),'unit':unit,'analyzerVersion':'1.5.0'}
