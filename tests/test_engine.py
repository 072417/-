from pathlib import Path
import numpy as np
import cv2
import pytest
from PIL import Image, ImageDraw, ImageFont
from server.engine import analyze,CATEGORIES,overlay_boxes,significant_color_change,significant_weight_change,regions,glyph_height,photo_like,uniform_text_color,refine_text_box,match,fragmented_text_counterpart,component_order_differences

FONT='/System/Library/Fonts/STHeiti Light.ttc'
MEDIUM_FONT='/System/Library/Fonts/STHeiti Medium.ttc'

def fixture(width=390,kind=None,scale=1,dy=0):
 im=Image.new('RGB',(width,700),'white');d=ImageDraw.Draw(im)
 f=lambda n:ImageFont.truetype(FONT,n)
 d.text((24,45),'Design review',font=f(26 if kind=='text_size' else 22),fill='#263141')
 d.text((24,125),'Typography',font=f(22),fill='#263141',stroke_width=1 if kind=='font_weight' else 0)
 d.text((24,200),'Aligned label',font=f(18),fill='#D14A39' if kind=='color' else '#263141')
 d.rounded_rectangle((24,300+dy,width-24,350+dy),8,fill='#28563F')
 d.text((width/2-30,315+dy),'Continue',font=f(14),fill='white')
 d.ellipse((30,410,66 if kind=='icon_shape' else 54,434),outline='#234561',width=3)
 d.rounded_rectangle((35,245,width-35,285),5,outline='#627189',width=2)
 d.text((52,254),'Left',font=f(16),fill='#263141')
 d.text((135 if kind=='spacing' else 120,254),'Right',font=f(16),fill='#263141')
 d.rounded_rectangle((100,500,180,532),5,outline='#627189',width=2)
 ax=108 if kind=='alignment' else 100
 d.rounded_rectangle((ax,560,ax+80,592),5,outline='#627189',width=2)
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
 a=fixture(width);b=fixture(width,kind,dy=8 if kind=='component_position' else 0)
 r=run(tmp_path,a,b)
 assert any(q['category']==kind for q in r['issues']), (width,kind,[(q['category'],q['title']) for q in r['issues']])
 if kind=='component_position':
  assert any(abs(q['measurements'][1]['delta']-8)<=1 for q in r['issues'] if q['category']==kind)

def test_vertical_text_move_is_component_arrangement_not_alignment(tmp_path):
 a=fixture();b=fixture();d=ImageDraw.Draw(b)
 d.rectangle((18,194,180,240),fill='white')
 d.text((24,216),'Aligned label',font=ImageFont.truetype(FONT,18),fill='#263141')
 r=run(tmp_path,a,b)
 moved=[q for q in r['issues'] if 'Aligned' in q['regionLabel']]
 assert any(q['category']=='component_position' for q in moved)
 assert not any(q['category']=='alignment' for q in moved)

def test_component_row_order_reversal_is_detected_as_one_relationship():
 def text(label,x,y,w=60):return {'box':[x,y,w,16],'kind':'text','text':label,'confidence':1}
 design=[text('当前订单店铺直播中',20,20,150),text('购后小助手',20,70,90),text('查看使用说明书',210,70,120),text('分享商品',40,110),text('联系商家',130,110),text('申请退款',220,110)]
 implementation=[text('当前订单店铺直播中',20,20,150),text('购后小助手',20,115,90),text('查看使用说明书',210,115,120),text('分享商品',40,70),text('联系商家',130,70),text('申请退款',220,70)]
 changes=component_order_differences([design,implementation],[(i,i,0) for i in range(len(design))],2)
 assert len(changes)==1
 assert changes[0]['designRelative']*changes[0]['implementationRelative']<0
 assert changes[0]['r']['text']=='购后小助手组件 ↔ 订单按钮组'

def test_spacing_is_reported_as_its_own_category(tmp_path):
 r=run(tmp_path,fixture(),fixture(kind='spacing'))
 q=next(q for q in r['issues'] if q['category']=='spacing')
 assert q['title'] in ('组件内横向元素间距不一致','组件内纵向元素间距不一致')
 assert '间距' in q['measurements'][0]['metric']

def test_button_border_does_not_change_glyph_height():
 font=ImageFont.truetype(FONT,22);plain=Image.new('RGB',(220,70),'white');bordered=plain.copy()
 ImageDraw.Draw(plain).text((48,20),'Button text',font=font,fill='#263141')
 draw=ImageDraw.Draw(bordered);draw.rounded_rectangle((18,7,202,61),7,outline='#8A929F',width=2);draw.text((48,20),'Button text',font=font,fill='#263141')
 a=np.array(plain);b=np.array(bordered);box=[18,7,184,54]
 assert abs(glyph_height(a,box)-glyph_height(b,box))<=1

def test_text_button_contour_is_component_not_icon():
 im=Image.new('RGB',(220,90),'white');draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,20)
 draw.rounded_rectangle((30,18,190,68),6,outline='#667080',width=2);draw.text((65,31),'Button',font=font,fill='#263141')
 a=np.array(im);texts=[{'box':[65,31,70,20],'text':'Button','kind':'text','confidence':1}]
 candidates=regions(a,texts)
 assert any(r['kind']=='component' and r['box'][2]>130 for r in candidates)
 assert not any(r['kind']=='icon' and r['box'][2]>130 for r in candidates)

def test_button_width_change_is_component_size_issue(tmp_path):
 font=ImageFont.truetype(FONT,18);a=fixture();b=fixture()
 for im,right in ((a,180),(b,205)):
  draw=ImageDraw.Draw(im);draw.rounded_rectangle((40,620,right,670),6,outline='#667080',width=2);draw.text((65,635),'Action',font=font,fill='#263141')
 r=run(tmp_path,a,b)
 assert any(q['category']=='component_position' and q['title']=='按钮或组件尺寸不一致' for q in r['issues'])
 assert not any(q['category']=='icon_shape' and q['designBBox'] and q['designBBox']['y']>600 for q in r['issues'])

def test_photo_and_its_internal_contours_are_not_icons():
 rng=np.random.default_rng(7);im=np.full((700,390,3),255,np.uint8)
 photo=np.zeros((112,112,3),np.uint8);photo[:]=[176,132,92]
 photo=np.clip(photo.astype(int)+rng.normal(0,22,photo.shape),0,255).astype(np.uint8)
 cv2.ellipse(photo,(46,58),(11,37),8,0,360,(238,232,215),-1)
 cv2.rectangle(photo,(0,0),(111,111),(115,82,55),2);im[34:146,44:156]=photo
 assert photo_like(im,[44,34,112,112])
 candidates=regions(im,[])
 assert any(r['kind']=='image' and coverage_for_test(r['box'],[44,34,112,112])>.8 for r in candidates)
 assert not any(r['kind']=='icon' and 44<=r['box'][0]+r['box'][2]/2<=156 and 34<=r['box'][1]+r['box'][3]/2<=146 for r in candidates)

def test_mixed_color_text_line_is_not_one_color_target():
 im=Image.new('RGB',(260,70),'white');draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,20)
 draw.text((20,20),'Pay later',font=font,fill='#35B84A');draw.text((130,20),'amount ¥0',font=font,fill='#62666D')
 assert not uniform_text_color(np.array(im),[16,15,225,35])

def test_refined_text_box_excludes_prefix_icon_and_next_line():
 im=Image.new('RGB',(300,100),'white');draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,19)
 draw.rectangle((10,18,30,42),outline='#36B84A',width=2)
 draw.text((50,16),'Order confirmed',font=font,fill='#36B84A')
 draw.text((50,52),'Yesterday 14:20',font=font,fill='#92979F')
 box=refine_text_box(np.array(im),[8,12,250,70])
 assert box[0]>35 and box[1]<35 and box[2]>120 and box[3]<35

def test_visually_different_icons_do_not_match():
 a=np.full((100,100,3),255,np.uint8);b=a.copy()
 cv2.line(a,(20,50),(42,28),(70,75,80),3);cv2.line(a,(20,50),(42,72),(70,75,80),3)
 cv2.rectangle(b,(16,30),(48,70),(70,180,90),3);cv2.circle(b,(38,60),6,(70,180,90),2)
 ra=[{'box':[14,24,36,52],'kind':'icon','text':'图标区域'}];rb=[{'box':[14,24,36,52],'kind':'icon','text':'图标区域'}]
 pairs,_,_=match(ra,rb,a,b)
 assert pairs==[]

def test_components_with_different_text_are_not_paired():
 a=np.full((90,240,3),255,np.uint8);cv2.rectangle(a,(20,20),(200,65),(90,95,105),2)
 first=[{'box':[20,20,181,46],'kind':'component','text':'组件区域：分享商品'}]
 second=[{'box':[20,20,181,46],'kind':'component','text':'组件区域：申请退款'}]
 pairs,_,_=match(first,second,a,a)
 assert pairs==[]

def test_complete_text_is_not_paired_with_a_cropped_fragment():
 im=np.full((80,360,3),255,np.uint8)
 first=[{'box':[18,20,300,20],'kind':'text','text':'订单号：260802-249340925531089｜复制','confidence':1}]
 second=[{'box':[82,20,158,20],'kind':'text','text':'260802-249340925531089','confidence':1}]
 pairs,_,_=match(first,second,im,im)
 assert pairs==[]

def test_same_row_ocr_fragment_is_not_reported_as_new_text():
 complete={'box':[18,20,238,12],'kind':'text','text':'订单号：260802-249340925531089｜复制','confidence':1}
 fragment={'box':[82,21,158,10],'kind':'text','text':'260802-249340925531089','confidence':1}
 assert fragmented_text_counterpart(fragment,[complete])

def test_unlabelled_right_edge_qa_control_is_ignored():
 im=Image.new('RGB',(390,700),'white');draw=ImageDraw.Draw(im)
 draw.rounded_rectangle((330,620,375,665),10,fill='#8C8F91');draw.ellipse((342,632,363,653),outline='white',width=3)
 boxes=overlay_boxes(np.array(im),[])
 assert any(coverage_for_test([330,620,45,45],box)>.8 for box in boxes)

def coverage_for_test(a,b):
 x=max(a[0],b[0]);y=max(a[1],b[1]);w=max(0,min(a[0]+a[2],b[0]+b[2])-x);h=max(0,min(a[1]+a[3],b[1]+b[3])-y)
 return w*h/max(1,a[2]*a[3])

def test_cross_component_row_distribution_is_not_spacing_or_alignment(tmp_path):
 a=fixture();b=fixture();font=ImageFont.truetype(FONT,18)
 for im,right_x in ((a,310),(b,326)):
  d=ImageDraw.Draw(im);d.text((180,250),'Left',font=font,fill='#263141');d.text((right_x,250),'Right',font=font,fill='#263141')
 r=run(tmp_path,a,b)
 moved=[q for q in r['issues'] if 'Right' in q['regionLabel']]
 assert not any(q['category']=='spacing' for q in moved)
 assert not any(q['category']=='alignment' for q in moved)

@pytest.mark.parametrize('scale',[2,3])
def test_density(scale,tmp_path):
 r=run(tmp_path,fixture(),fixture(scale=scale),{'design':{'logicalWidth':390},'implementation':{'logicalWidth':390},'categories':list(CATEGORIES)})
 assert r['issues']==[],[(q['title'],q['measurements']) for q in r['issues']]

def test_crosswidth_not_fake_geometry(tmp_path):
 r=run(tmp_path,fixture(375),fixture(390))
 assert r['mode']=='cross_width_reference'
 assert r['coverage']['component_position']['status']=='partial'
 assert not any(q['category']=='component_position' for q in r['issues'])

def test_crosswidth_reports_adaptation_error(tmp_path):
 r=run(tmp_path,fixture(375),fixture(390,dy=18))
 q=next(q for q in r['issues'] if q['category']=='component_position' and 295<=q['designBBox']['y']<=305)
 assert q['title']=='跨宽组件排布不符合预期'
 assert any(m['certainty']=='estimated' for m in q['measurements'])

def test_common_widths_are_inferred(tmp_path):
 r=run(tmp_path,fixture(375),fixture(390),{'design':{},'implementation':{},'categories':list(CATEGORIES)})
 assert r['mode']=='cross_width_reference'
 assert [m['logicalWidth'] for m in r['metadata']]==[375,390]
 assert all(m['source']=='auto_inferred' for m in r['metadata'])

def test_real_font_weight_change(tmp_path):
 a=fixture();b=fixture();d=ImageDraw.Draw(b)
 d.rectangle((20,120,180,155),fill='white')
 d.text((24,125),'Typography',font=ImageFont.truetype(MEDIUM_FONT,22),fill='#263141')
 r=run(tmp_path,a,b)
 assert any(q['category']=='font_weight' for q in r['issues'])

def test_antialiasing_stroke_shift_is_not_a_weight_issue():
 changed,_,density=significant_weight_change({'strokeWidth':4.394,'inkDensity':.571},{'strokeWidth':6,'inkDensity':.617})
 assert not changed
 assert abs(density)<.18

def test_regular_to_medium_density_change_is_a_weight_issue():
 changed,_,density=significant_weight_change({'strokeWidth':2,'inkDensity':.191},{'strokeWidth':3,'inkDensity':.251})
 assert changed
 assert density>.18

@pytest.mark.parametrize('label', ['CBase', 'LEGO', '容器'])
def test_common_test_overlay_is_automatically_ignored(tmp_path, label):
 a=fixture();b=fixture();d=ImageDraw.Draw(b)
 d.rounded_rectangle((275,20,375,58),8,fill='#20242A')
 d.text((292,29),label,font=ImageFont.truetype(FONT,16),fill='white')
 r=run(tmp_path,a,b)
 assert any('测试浮层' in warning for warning in r['warnings'])
 assert not any(q['implementationBBox'] and q['implementationBBox']['x']>260 and q['implementationBBox']['y']<70 for q in r['issues'])

@pytest.mark.parametrize('text', ['PIN14:26UO', '1653900554043/分', 'ABC123456789'])
def test_top_test_identifier_ignores_status_strip(text):
 boxes=overlay_boxes(np.zeros((800,390,3),dtype=np.uint8),[{'text':text,'box':[120,9,170,20]}])
 assert boxes==[[0,0,390,41]]

def test_overlapping_top_identifiers_share_one_ignored_strip():
 boxes=overlay_boxes(np.zeros((800,390,3),dtype=np.uint8),[{'text':'PIN14:26UO','box':[20,9,90,20]},{'text':'1653900554043/分','box':[190,8,180,22]}])
 assert boxes==[[0,0,390,42]]

@pytest.mark.parametrize('text', ['14:26', '52', '积分 1234'])
def test_normal_short_numbers_are_not_automatically_ignored(text):
 assert overlay_boxes(np.zeros((800,390,3),dtype=np.uint8),[{'text':text,'box':[120,9,80,20]}])==[]

def test_small_device_gamut_shift_is_ignored():
 changed,evidence=significant_color_change(np.array([40,86,63]),np.array([44,91,68]),12)
 assert not changed
 assert evidence['deltaE']<12

@pytest.mark.parametrize(('design','implementation'),[
 ([220,35,30],[240,120,20]),
 ([240,215,20],[240,125,20]),
])
def test_obvious_hue_changes_are_reported(design,implementation):
 changed,evidence=significant_color_change(np.array(design),np.array(implementation),12)
 assert changed
 assert evidence['hueDelta']>=12

def test_unknown_density_requests_alignment(tmp_path):
 r=run(tmp_path,fixture(400),fixture(400,scale=2),{'design':{},'implementation':{}})
 assert r['status']=='needs_alignment'

def test_masks_do_not_hide_outside(tmp_path):
 c={'design':{'logicalWidth':390,'ignore':[[0,0,390,100]]},'implementation':{'logicalWidth':390,'ignore':[[0,0,390,100]]}}
 r=run(tmp_path,fixture(),fixture(kind='text_size',dy=8),c)
 assert not any(q['category']=='text_size' for q in r['issues'])
 assert any(q['category']=='component_position' for q in r['issues'])

def test_crop_mapping(tmp_path):
 c={'design':{'logicalWidth':390,'crop':[0,100,390,500]},'implementation':{'logicalWidth':390,'crop':[0,100,390,500]}}
 r=run(tmp_path,fixture(),fixture(dy=18),c)
 q=next(q for q in r['issues'] if q['category']=='component_position' and 295<=q['designBBox']['y']<=305)
 assert 295<=q['designBBox']['y']<=305
 assert abs(q['implementationBBox']['y']-q['designBBox']['y']-18)<=1

def test_cancel(tmp_path):
 im=fixture();im.save(tmp_path/'a.png')
 def check():raise InterruptedError()
 with pytest.raises(InterruptedError):analyze(tmp_path/'a.png',tmp_path/'a.png',{},tmp_path,lambda _:None,check)
