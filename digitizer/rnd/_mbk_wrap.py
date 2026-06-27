import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, cv2
from PIL import Image, ImageDraw; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract
import detect_masks as dm, extract_instances as ei

tpl=r"F:\nds\projects\Yatskivska_001\wlg\Yatskivska_1_MBK_3180_3580_200_D1.nlgx"
img=r"F:\nds\projects\Yatskivska_001\img\Yatskivska_1_MBK_3180_3580_200_D1.jpg"
m=extract(tpl); rgb=np.asarray(Image.open(img).convert("RGB")); H,W=rgb.shape[:2]
masks=dm.compute_masks(rgb,m); g=masks["geom"]
xl=g["x_left"]; green=g["x_right"] or 702
body=np.zeros((H,W),np.uint8); body[g["top_y"]:g["bottom_y"],max(0,xl-5):min(W,green+30)]=1
chans=ei.classify_ink(rgb,body); black=(chans["black"]&(masks["exclude"]==0)).astype(np.uint8)
da=m["depth_axis"]; dpx=(da["bottom_y"]-da["top_y"])/(da["bottom_depth"]-da["top_depth"])
def y_of(d): return int(da["top_y"]+(d-da["top_depth"])*dpx)

def runs(y, gap=5):
    xs=np.nonzero(black[y])[0]
    if not len(xs): return []
    return [int(c.mean()) for c in np.split(xs,np.nonzero(np.diff(xs)>gap)[0]+1)]

# wrap-aware трекер: уровень растёт на правом крае (входящий back-up слева), падает на левом.
def trace_wrap(ty, by, slmax=30, edge=25, span=None):
    span = span or (green-xl)
    tr={}; lv={}; x=v=None; level=0
    for y in range(ty,by):
        e=runs(y)
        if not e:
            if x is not None: x=x+float(np.clip(v,-slmax,slmax))
            continue
        if x is None:
            x=min(e); v=0.0; level=0; tr[y]=x; lv[y]=0; continue
        pred=x+float(np.clip(v,-slmax,slmax))
        # WRAP UP: упёрлись в правый край, идём вправо, и есть линия в левой трети -> ×N+1
        if x>=green-edge and v>=0:
            left=[r for r in e if r < xl+0.4*span]
            if left:
                level+=1; nx=min(left); v=2.0; x=nx; tr[y]=x; lv[y]=level; continue
        # WRAP DOWN: упёрлись в левый край на уровне>0, идём влево, есть линия справа -> ×N-1
        if x<=xl+edge and v<=0 and level>0:
            right=[r for r in e if r > green-0.4*span]
            if right:
                level-=1; nx=max(right); v=-2.0; x=nx; tr[y]=x; lv[y]=level; continue
        nx=min(e,key=lambda c:abs(c-pred))
        v=0.6*v+0.4*(nx-x); x=nx; tr[y]=x; lv[y]=level
    return tr, lv

# тест на верхнем регионе с оборотами
d0,d1=3180,3260
ty,by=y_of(d0),y_of(d1)
tr,lv=trace_wrap(ty,by)
levels=sorted(set(lv.values()))
print("регион %d-%dм: уровни=%s; точек=%d"%(d0,d1,levels,len(tr)))
for L in levels:
    print("  level %d: %d строк"%(L, sum(1 for v in lv.values() if v==L)))
# визуализация: трасса цветом по уровню
COL={0:(255,0,0),1:(0,120,255),2:(0,200,0),3:(255,0,255)}
crop=rgb[ty:by,:].copy(); im=Image.fromarray(crop); dr=ImageDraw.Draw(im)
dr.line([(green,0),(green,by-ty)],fill=(0,180,0),width=1)
prev=None
for y in range(ty,by):
    if y in tr:
        p=(int(tr[y]),y-ty)
        if prev: dr.line([prev,p],fill=COL.get(lv[y],(0,0,0)),width=2)
        prev=p
    else: prev=None
im=im.resize((W*2,(by-ty)*2),Image.NEAREST)
im.save(r"F:\nds\output\mbk_wrap_levels.png")
print("viz -> F:/nds/output/mbk_wrap_levels.png  (level0=красн,1=син,2=зел,3=маджента; верт зелёная=край value2)")
