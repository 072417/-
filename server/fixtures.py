"""Deterministic mobile UI fixtures. They go through the same analysis path as uploads."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def make_demo(changed=False,width=390,scale=2):
 im=Image.new('RGB',(390,844),'#F8F9FB');d=ImageDraw.Draw(im)
 fp=next((p for p in ['/System/Library/Fonts/STHeiti Light.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()),None)
 def text(x,y,s,size=14,fill='#182127',weight=False):
  f=ImageFont.truetype(fp,size) if fp else ImageFont.load_default(size=size)
  d.text((x,y),s,font=f,fill=fill,stroke_width=1 if weight else 0,stroke_fill=fill)
 text(24,13,'9:41',13);text(322,13,'100%',11)
 text(24,66,'今天，也向前一步。',25 if changed else 22)
 text(24,106,'星期五，9 月 11 日',13,'#808894')
 d.rounded_rectangle((24,151,366,333),18,fill='#E6EFEA')
 text(44,171,'本周运动',14,'#4B6355')
 text(44,203,'128',46,'#244A39');text(142,235,'分钟',14,'#527362')
 for j,h in enumerate([18,36,48,26,60,41,18]):
  x=216+j*17;d.rounded_rectangle((x,282-h,x+9,282),4,fill='#719B83' if j<5 else '#C4D7CB')
 text(44,292,'比上周多运动了 24 分钟',12,'#527362')
 text(24+(7 if changed else 0),367,'为你推荐',19,weight=changed)
 text(313,373,'查看全部',12,'#8A9198')
 d.rounded_rectangle((24,411,366,491),12,fill='white',outline='#E5E8EC')
 d.rounded_rectangle((40,428,88,475),10,fill='#EEF2F5')
 # Functional geometric icon fixture with a deliberately changed aspect ratio.
 d.ellipse((51,437,80 if changed else 73,460),outline='#526677',width=3)
 text(105,426,'轻松晨间拉伸',16);text(105,454,'12 分钟 · 全身舒展',12,'#949AA2')
 d.rounded_rectangle((24,507,366,587),12,fill='white',outline='#E5E8EC')
 d.rounded_rectangle((40,524,88,571),10,fill='#F4EFE8');d.rectangle((55,536,73,559),outline='#9B8365',width=3)
 text(105,522,'找回呼吸的节奏',16);text(105,550,'8 分钟 · 正念呼吸',12,'#949AA2')
 y=637+(8 if changed else 0)
 d.rounded_rectangle((24,y,366,y+52),12,fill='#4981AD' if changed else '#24563F')
 text(156,y+14,'开始运动',16,'white')
 text(95,714,'按自己的节奏，保持一点进步。',12,'#969DA5')
 d.rectangle((0,773,390,844),fill='white')
 for x,s in [(51,'今日'),(178,'发现'),(302,'我的')]:text(x,790,s,13,'#24563F' if x==51 else '#929AA1')
 d.rounded_rectangle((133,830,257,834),2,fill='#182127')
 # Responsive fixture: horizontal canvas extends; fixed margins maintained via full fixture for demo only.
 if width!=390:im=im.resize((width,round(844*width/390)))
 if scale!=1:im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
 return im
