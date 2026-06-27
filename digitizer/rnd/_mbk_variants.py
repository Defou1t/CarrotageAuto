import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, cv2
from collections import defaultdict
from PIL import Image, ImageDraw; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
import dataset as ds, extract_instances as ei, detect_masks as dm
import track_identity as ti
from digitize_auto import trace_color

tpl=r"F:\nds\projects\Yatskivska_001\wlg\Yatskivska_1_MBK_3180_3580_200_D1.nlgx"
img=r"F:\nds\projects\Yatskivska_001\img\Yatskivska_1_MBK_3180_3580_200_D1.jpg"
m=extract(tpl); rgb=np.asarray(Image.open(img).convert("RGB")); H,W=rgb.shape[:2]
lines,masks=ti.track_identity(rgb,m); g=masks["geom"]; ty,by=g["top_y"],min(H,g["bottom_y"])
xl=max(0,g["x_left"]-5); xr=min(W,(g["x_right"] or W)+30)
body=np.zeros((H,W),np.uint8); body[ty:by,xl:xr]=1
chans=ei.classify_ink(rgb,body); black=(chans["black"]&(masks["exclude"]==0)).astype(np.uint8)
dblack=cv2.dilate(black,cv2.getStructuringElement(cv2.MORPH_RECT,(11,3)))

def runs_ext(row, gap=6):
    xs=np.nonzero(row)[0]
    if not len(xs): return []
    return [(int(c.mean()),int(c[0]),int(c[-1])) for c in np.split(xs,np.nonzero(np.diff(xs)>gap)[0]+1)]

def bridged(row,a,b,bg=12):
    lo,hi=min(a,b),max(a,b)
    if hi-lo<8: return True
    xs=np.nonzero(row[lo:hi+1])[0]
    if not len(xs): return False
    gaps=np.diff(xs)
    return (xs[0]<=bg) and (xs[-1]>=(hi-lo-bg)) and ((gaps.max() if len(gaps) else 0)<=bg)

def trace_bridge(slmax=25):
    tr={}; x=v=None
    for y in range(ty,by):
        e=runs_ext(black[y])
        if not e:
            if x is not None: x=x+float(np.clip(v,-slmax,slmax))
            continue
        if x is None:
            cx=min(e,key=lambda r:r[0])[0]; x=cx; v=0.0; tr[y]=x; continue
        pred=x+float(np.clip(v,-slmax,slmax))
        reach=[r for r in e if bridged(black[y],int(x),r[0])]
        pool=reach if reach else [min(e,key=lambda r:abs(r[0]-pred))]
        nx=min(pool,key=lambda r:abs(r[0]-pred))[0]
        v=0.6*v+0.4*(nx-x); x=nx; tr[y]=x
    return tr

def trace_b3(slmax=25):
    blk=[L for L in lines if L["color"]=="black"]
    cand=defaultdict(list)
    for L in blk:
        for y,xx in L["tr"].items(): cand[int(y)].append(xx)
    tr={}; x=v=None
    for y in range(ty,by):
        e=cand.get(y)
        if not e:
            if x is not None: x=x+float(np.clip(v,-slmax,slmax))
            continue
        if x is None: x=float(np.median(e)); v=0.0; tr[y]=x; continue
        pred=x+float(np.clip(v,-slmax,slmax)); nx=min(e,key=lambda c:abs(c-pred))
        v=0.6*v+0.4*(nx-x); x=nx; tr[y]=x
    return tr

def despike(tr,thr=45,iters=2):
    ys=sorted(tr); xs=[tr[y] for y in ys]; n=len(ys)
    for _ in range(iters):
        for i in range(1,n-1):
            if ys[i]-ys[i-1]>3 or ys[i+1]-ys[i]>3: continue
            d1=xs[i]-xs[i-1]; d2=xs[i]-xs[i+1]
            if d1*d2>0 and min(abs(d1),abs(d2))>thr: xs[i]=0.5*(xs[i-1]+xs[i+1])
    return {ys[i]:xs[i] for i in range(n)}

V1=trace_color(black,ty,by); V2=trace_bridge(); V3=trace_b3(); V4=despike(V1)

def metr(tr):
    ys=np.array(sorted(tr)); xs=np.array([int(round(tr[y])) for y in ys]); n=len(ys)
    off=(dblack[np.clip(ys,0,H-1),np.clip(xs,0,W-1)]==0).mean()*100
    dx=np.abs(np.diff(xs)); sp=0
    for i in range(1,n-1):
        if ys[i]-ys[i-1]>3 or ys[i+1]-ys[i]>3: continue
        if (xs[i]-xs[i-1])*(xs[i]-xs[i+1])>0 and min(abs(xs[i]-xs[i-1]),abs(xs[i]-xs[i+1]))>45: sp+=1
    return n,off,sp,float(np.median(dx)),int((dx>100).sum()),int((dx>200).sum())

print("%-15s%6s%7s%7s%7s%8s%8s"%("вариант","n","off%","спайк","медΔx","бр>100","бр>200"))
for nm,tr in [("V1 per-row",V1),("V2 bridge",V2),("V3 b3-stitch",V3),("V4 despike",V4)]:
    n,off,sp,md,b1,b2=metr(tr)
    print("%-15s%6d%7.1f%7d%7.1f%8d%8d"%(nm,n,off,sp,md,b1,b2))

da=m["depth_axis"]; dpx=(da["bottom_y"]-da["top_y"])/(da["bottom_depth"]-da["top_depth"])
d0,d1=3437,3447; y0=int(da["top_y"]+(d0-da["top_depth"])*dpx); y1=int(da["top_y"]+(d1-da["top_depth"])*dpx)
allx=[tr[y] for tr in [V1,V2,V3,V4] for y in range(y0,y1) if y in tr]
cx0=max(0,int(min(allx))-40); cx1=min(W,int(max(allx))+40); base=rgb[y0:y1,cx0:cx1]
panels=[]
for nm,tr,col in [("V1",V1,(255,0,0)),("V2",V2,(0,150,255)),("V3",V3,(0,200,0)),("V4",V4,(255,0,255))]:
    im=Image.fromarray(base.copy()); dr=ImageDraw.Draw(im)
    pts=[(int(tr[y])-cx0,y-y0) for y in range(y0,y1) if y in tr]
    dr.line(pts,fill=col,width=1); dr.text((3,3),nm,fill=col)
    panels.append(np.asarray(im))
sep=np.full((panels[0].shape[0],4,3),180,np.uint8)
comp=panels[0]
for p in panels[1:]: comp=np.concatenate([comp,sep,p],1)
out=Image.fromarray(comp); out=out.resize((out.width*2,out.height*2),Image.NEAREST)
out.save(r"F:\nds\output\mbk_variants_compare.png")
print("\nкомпозит -> F:\\nds\\output\\mbk_variants_compare.png  (%d-%dм, панели V1|V2|V3|V4)"%(d0,d1))
