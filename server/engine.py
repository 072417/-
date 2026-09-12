"""Evidence-first screenshot analysis. All bboxes originate from local detectors."""
import io, json, math, os, re, subprocess, tempfile, time, shutil, csv
from pathlib import Path
from difflib import SequenceMatcher
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab, deltaE_ciede2000

CATEGORIES = {'text_size':'文字大小','font_weight':'字重','alignment':'对齐','color':'颜色','icon_shape':'Icon 形变','component_position':'组件位置'}
HELPER = Path(__file__).with_name('ocr-helper')
MOBILE_WIDTHS = (360,375,390,414,430)
TEST_OVERLAY = re.compile(r'(?i)(?:^|[^a-z0-9])(?:c\s*base|lego)(?:[^a-z0-9]|$)')

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
  if any(iou([x,y,w,h],r['box'])>.8 or (w<85 and h<85 and r['kind']=='icon' and abs(x+w/2-r['box'][0]-r['box'][2]/2)<3 and abs(y+h/2-r['box'][1]-r['box'][3]/2)<3) for r in rs): continue
  if any(coverage([x,y,w,h],t['box'])>.6 for t in texts): continue
  kind='icon' if 10<=w<=85 and 10<=h<=85 else 'component'
  rs.append({'box':[x,y,w,h],'kind':kind,'text': '图标区域' if kind=='icon' else '组件区域','confidence':.7})
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
    similarity=SequenceMatcher(None,r['text'].replace(' ',''),s['text'].replace(' ','')).ratio()
    if similarity<.65: continue
    costs[i,j]=(1-similarity)*2+dist*.8+min(size,2)*.08
   else:
    costs[i,j]=dist*1.4+min(size,3)*.2+np.mean(np.abs(fa[i]-fb[j]))*.4
 pairs=[]
 for i,j in zip(*linear_sum_assignment(costs)):
  if costs[i,j]<(.52 if a[i]['kind']=='text' else .52): pairs.append((int(i),int(j),float(costs[i,j])))
 return pairs,[i for i in range(len(a)) if i not in {x[0] for x in pairs}],[j for j in range(len(b)) if j not in {x[1] for x in pairs}]

def background_color(a,b):
 x,y,w,h=[int(round(v)) for v in b];pad=4
 x0=max(0,x-pad);y0=max(0,y-pad);x1=min(a.shape[1],x+w+pad);y1=min(a.shape[0],y+h+pad)
 p=a[y0:y1,x0:x1];mask=np.ones(p.shape[:2],bool)
 mask[max(0,y-y0):min(p.shape[0],y+h-y0),max(0,x-x0):min(p.shape[1],x+w-x0)]=False
 ring=p[mask]
 return np.median(ring,axis=0) if len(ring) else np.array([255,255,255])

def refine_text_box(a,b):
 # OCR locates text; foreground pixels determine the visible ink bounds.
 x,y,w,h=b;pad=4
 box=[max(0,int(x)-pad),max(0,int(y)-pad),int(w)+pad*2+1,int(h)+pad*2+1]
 p=crop(a,box)
 if not p.size:return b
 bg=background_color(a,box)
 distance=np.linalg.norm(p.astype(float)-bg,axis=2)
 mask=distance>max(24,float(np.percentile(distance,95))*.5)
 yy,xx=np.nonzero(mask)
 if len(xx)<8:return b
 refined=[box[0]+int(xx.min()),box[1]+int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)]
 if refined[2]<w*.45 or refined[3]<h*.35:return b
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

def weight_features(a,b):
 p=crop(a,b)
 if not p.size:return None
 bg=background_color(a,b);delta=np.linalg.norm(p.astype(float)-bg,axis=2)
 nonzero=delta[delta>6]
 if len(nonzero)<8:return None
 values=np.clip(nonzero,0,255).astype(np.uint8)
 threshold=max(16,float(cv2.threshold(values,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]))
 mask=(delta>threshold).astype(np.uint8)
 count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
 clean=np.zeros_like(mask)
 for i in range(1,count):
  if stats[i,cv2.CC_STAT_AREA]>=2:clean[labels==i]=1
 yy,xx=np.nonzero(clean)
 if len(xx)<8:return None
 tight=clean[yy.min():yy.max()+1,xx.min():xx.max()+1]
 d=cv2.distanceTransform(tight,cv2.DIST_L2,5);vals=d[d>0]
 if len(vals)<8:return None
 return {'strokeWidth':float(np.percentile(vals,75)*2),'inkDensity':float(tight.mean()),'inkArea':int(tight.sum())}

def weight(a,b):
 f=weight_features(a,b)
 return f['strokeWidth'] if f else None

def overlay_boxes(a,texts):
 boxes=[]
 for t in texts:
  if not TEST_OVERLAY.search(t.get('text','')):continue
  x,y,w,h=t['box'];px=max(18,h*2.2);py=max(10,h*1.35)
  x0=max(0,x-px);y0=max(0,y-py);x1=min(a.shape[1],x+w+px);y1=min(a.shape[0],y+h+py)
  boxes.append([x0,y0,x1-x0,y1-y0])
 return boxes

def adaptive_geometry(box,design_width,implementation_width):
 x,_,w,_=box;left=x;right=design_width-x-w;center=x+w/2-design_width/2
 if w/design_width>=.55:return left,max(1,implementation_width-left-right),'左右边距保持'
 if abs(center)<=max(12,design_width*.06):return implementation_width/2+center-w/2,w,'居中保持'
 if x+w/2<design_width/2:return left,w,'左侧锚定'
 return implementation_width-right-w,w,'右侧锚定'

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
  if count:warnings.append(f'已自动忽略 {count} 处 CBase / LEGO 测试浮层。')
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
 pos=config.get('positionThreshold',2)*tolerance; size=config.get('sizeThreshold',3)/100*tolerance; color=config.get('colorThreshold',3)*tolerance
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
  if c in ('text_size','font_weight','alignment'):
   cov[c]={'status':'partial' if not ocr_errors[0] and not ocr_errors[1] else 'unavailable','reason':'基于 OCR 可见字形与边界估计，需人工复核。' if not any(ocr_errors) else 'OCR 不完整，无法保证文字覆盖。'}
  if c=='icon_shape':cov[c]={'status':'partial','reason':'基于非文本小轮廓候选，无法保证识别所有图标。'}
  if cross and c in ('component_position','alignment'):
   cov[c]={'status':'partial','reason':'按左/右/居中/左右边距锚点估算目标宽度适配，断点与业务布局规则需人工复核。'}
 sameheight=abs(aa.shape[0]-bb.shape[0])<2
 if not sameheight:
  warnings.append('两侧内容高度不同：底部锚定区域可在设置中指定；未知锚定的纵向偏移仅作为候选。')
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
   cat='alignment' if kind=='text' else 'component_position'
   title=('跨宽文字对齐不符合预期' if kind=='text' else '跨宽组件适配不符合预期') if cross else ('文字边缘偏移' if kind=='text' else '组件位置偏移')
   if bottom:title='底部锚定间距偏差'
   geometry=[metric('适配预期左边缘 x' if cross else '左边缘 x',expected_x,xx,estimate=cross),metric('距底部安全区' if bottom else '上边缘 y',yp,yq,estimate=cross)]
   if cross and kind!='text':geometry.append(metric('适配预期宽度',expected_w,sw,estimate=True))
   add(cat,title,r,s,geometry,(adaptation+'估算 · ' if cross else '')+('OCR 边界' if kind=='text' else '轮廓边界'),'检查目标宽度下的约束、边距、居中或拉伸规则。' if cross else '检查对应区域的内边距、约束或相对间距。',conf if sameheight or bottom else 'low')
  if not cross and known:
   da=config.get('design',{}); ib=config.get('implementation',{})
   dt=da.get('safeTop',0); db=da.get('safeBottom',0);it=ib.get('safeTop',0);bottomInset=ib.get('safeBottom',0)
   design_inside=y>=dt and y+rh<=aa.shape[0]-db
   violation=(it>0 and yy<it and yy+sh>it) or (bottomInset>0 and yy+sh>bb.shape[0]-bottomInset and yy<bb.shape[0]-bottomInset)
   if design_inside and violation:
    add('component_position','疑似侵入安全区',r,s,[metric('上边缘 y',y,yy),metric('下边缘 y',y+rh,yy+sh)],'用户提供的安全区与可见元素边界','核对 App 内容与安全区约束；需确认该区域属于 App 内容。','low')
  if kind=='text' and abs(sh/rh-1)>size and abs(sh-rh)>1:
   add('text_size','疑似字号'+('偏大' if sh>rh else '偏小'),r,s,[metric('可见字形高度',rh,sh,estimate=True)],'OCR 字形边界','核对原生字号、字体和系统字体缩放；字形高度不是实际字号。','medium')
  ca=sampled_color(aa,r['box'],kind=='text');cb=sampled_color(bb,s['box'],kind=='text')
  if ca is not None and cb is not None:
   de=float(deltaE_ciede2000(rgb2lab(ca[None,None,:]/255),rgb2lab(cb[None,None,:]/255))[0,0])
   if de>color:
    ha='#'+''.join(f'{int(v):02X}' for v in ca); hb='#'+''.join(f'{int(v):02X}' for v in cb)
    add('color','文字颜色偏差' if kind=='text' else '区域颜色偏差',r,s,[{'metric':'截图采样色','designValue':ha,'implementationValue':hb,'delta':round(de,2),'unit':'ΔE00','source':'interior_pixel_sampling','certainty':'measured'}],'稳定区域采样 · CIEDE2000','核对对应颜色与透明度；截图采样色不等同源码颜色。',conf)
  if kind=='text' and abs(sh/rh-1)<.35 and abs(sw/rw-1)<.25 and ca is not None and cb is not None and np.linalg.norm(ca-cb)<45:
   wa=weight_features(aa,r['box']);wb=weight_features(bb,s['box'])
   if wa and wb:
    stroke=math.log(max(.01,wb['strokeWidth'])/max(.01,wa['strokeWidth']));density=math.log(max(.01,wb['inkDensity'])/max(.01,wa['inkDensity']))
    agree=stroke*density>=0 or abs(stroke)<.04 or abs(density)<.04;threshold=.105*tolerance
    if agree and (abs(stroke)>threshold or abs(density)>threshold) and (abs(stroke)+abs(density))/2>.08*tolerance:
     direction=wb['strokeWidth']+wb['inkDensity']*4>wa['strokeWidth']+wa['inkDensity']*4
     add('font_weight','疑似字重'+('偏粗' if direction else '偏细'),r,s,[metric('笔画宽度估计',wa['strokeWidth'],wb['strokeWidth'],estimate=True),metric('字形墨色密度',wa['inkDensity'],wb['inkDensity'],'比例',True)],'前景分割 · 笔画距离变换 · 字形墨色密度','核对字体文件、fontWeight、可变字体轴与文字渲染方式。','medium' if max(abs(stroke),abs(density))>.16 else 'low')
  if kind=='icon' and abs((sw/sh)/(rw/rh)-1)>max(size,.06) and abs(sw-rw)+abs(sh-rh)>2:
   add('icon_shape','疑似 Icon 非等比形变',r,s,[metric('宽高比',rw/rh,sw/sh,'比例'),metric('轮廓宽度',rw,sw),metric('轮廓高度',rh,sh)],'保留比例的轮廓测量','核对图标资源与宽高约束；区分图案变化和非等比拉伸。','medium')
 # Do not call offscreen content missing, or flood reports with unmatched contour noise.
 if not cross and not incompatible:
  for index,unmatched in ((0,missing),(1,extra)):
   for k in unmatched:
    r=rs[index][k];x,y,ww,hh=r['box']
    if r['kind']!='text' or r['confidence']<.6 or y+hh>h-2:continue
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
 return {'status':'partial' if any(cov[c]['status']!='checked' for c in selected) else 'completed','issues':issues,'coverage':cov,'warnings':warnings,'transforms':maps,'metadata':metadata,'ignoredOverlays':ignored,'mode':'cross_width_reference' if cross else 'same_width','normalizedSizes':[[a.shape[1],a.shape[0]] for a in imgs],'duration':round(time.monotonic()-start,1),'candidateCounts':[len(r) for r in rs],'matchedCandidates':len(pairs),'unit':unit,'analyzerVersion':'1.1.0'}
