from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from server.engine import analyze,CATEGORIES

FONT='/System/Library/Fonts/STHeiti Light.ttc'

def fixture(width=390,kind=None,scale=1,dy=0):
 im=Image.new('RGB',(width,700),'white');d=ImageDraw.Draw(im)
 f=lambda n:ImageFont.truetype(FONT,n)
 d.text((24,45),'Design review',font=f(26 if kind=='text_size' else 22),fill='#263141')
 d.text((24,125),'Typography',font=f(22),fill='#263141',stroke_width=1 if kind=='font_weight' else 0)
 d.text((32 if kind=='alignment' else 24,200),'Aligned label',font=f(18),fill='#263141')
 d.rounded_rectangle((24,300+dy,width-24,350+dy),8,fill='#D14A39' if kind=='color' else '#28563F')
 d.text((width/2-30,315+dy),'Continue',font=f(14),fill='white')
 d.ellipse((30,410,66 if kind=='icon_shape' else 54,434),outline='#234561',width=3)
 return im.resize((width*scale,700*scale),Image.Resampling.NEAREST)

def run(tmp,a,b,config=None):
 tmp.mkdir(exist_ok=True);a.save(tmp/'a.png');b.save(tmp/'b.png')
 return analyze(tmp/'a.png',tmp/'b.png',config or {'design':{'logicalWidth':a.width},'implementation':{'logicalWidth':b.width},'categories':list(CATEGORIES)},tmp,lambda _:None,lambda:None)

@pytest.mark.parametrize('width',[360,375,390,414,430])
def test_identity(width,tmp_path):
 im=fixture(width);r=run(tmp_path,im,im);assert r['issues']==[]

@pytest.mark.parametrize('width',[360,375,390,414,430])
@pytest.mark.parametrize('kind',list(CATEGORIES))
def test_categories(width,kind,tmp_path):
 a=fixture(width);b=fixture(width,kind if kind!='component_position' else None,dy=8 if kind=='component_position' else 0)
 r=run(tmp_path,a,b)
 assert any(q['category']==kind for q in r['issues']), (width,kind,[(q['category'],q['title']) for q in r['issues']])
 if kind=='component_position':
  assert any(abs(q['measurements'][1]['delta']-8)<=1 for q in r['issues'] if q['category']==kind)

@pytest.mark.parametrize('scale',[2,3])
def test_density(scale,tmp_path):
 r=run(tmp_path,fixture(),fixture(scale=scale),{'design':{'logicalWidth':390},'implementation':{'logicalWidth':390},'categories':list(CATEGORIES)})
 assert r['issues']==[],[(q['title'],q['measurements']) for q in r['issues']]

def test_crosswidth_not_fake_geometry(tmp_path):
 r=run(tmp_path,fixture(375),fixture(390,dy=8))
 assert r['mode']=='cross_width_reference'
 assert r['coverage']['component_position']['status']=='unavailable'
 assert not any(q['category']=='component_position' for q in r['issues'])

def test_unknown_density_requests_alignment(tmp_path):
 r=run(tmp_path,fixture(),fixture(scale=2),{'design':{},'implementation':{}})
 assert r['status']=='needs_alignment'

def test_masks_do_not_hide_outside(tmp_path):
 c={'design':{'logicalWidth':390,'ignore':[[0,0,390,100]]},'implementation':{'logicalWidth':390,'ignore':[[0,0,390,100]]}}
 r=run(tmp_path,fixture(),fixture(kind='text_size',dy=8),c)
 assert not any(q['category']=='text_size' for q in r['issues'])
 assert any(q['category']=='component_position' for q in r['issues'])

def test_crop_mapping(tmp_path):
 c={'design':{'logicalWidth':390,'crop':[0,100,390,500]},'implementation':{'logicalWidth':390,'crop':[0,100,390,500]}}
 r=run(tmp_path,fixture(),fixture(dy=8),c)
 q=next(q for q in r['issues'] if q['category']=='component_position')
 assert 295<=q['designBBox']['y']<=305
 assert abs(q['implementationBBox']['y']-q['designBBox']['y']-8)<=1

def test_cancel(tmp_path):
 im=fixture();im.save(tmp_path/'a.png')
 def check():raise InterruptedError()
 with pytest.raises(InterruptedError):analyze(tmp_path/'a.png',tmp_path/'a.png',{},tmp_path,lambda _:None,check)
