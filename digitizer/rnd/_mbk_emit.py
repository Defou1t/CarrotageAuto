import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, cv2
from pathlib import Path
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
import dataset as ds, extract_instances as ei, detect_masks as dm
import track_identity as ti
from digitize_auto import trace_color
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd
from digitize_b3 import curve_pred

tpl=r"F:\nds\projects\Yatskivska_001\wlg\Yatskivska_1_MBK_3180_3580_200_D1.nlgx"
img=r"F:\nds\projects\Yatskivska_001\img\Yatskivska_1_MBK_3180_3580_200_D1.jpg"
outdir=Path(r"F:\nds\output\yat_auto"); outdir.mkdir(parents=True, exist_ok=True)
m=extract(tpl); rgb=np.asarray(Image.open(img).convert("RGB")); H,W=rgb.shape[:2]
lines,masks=ti.track_identity(rgb,m); g=masks["geom"]; ty,by=g["top_y"],min(H,g["bottom_y"])
xl=max(0,g["x_left"]-5); xr=min(W,(g["x_right"] or W)+30)
body=np.zeros((H,W),np.uint8); body[ty:by,xl:xr]=1
chans=ei.classify_ink(rgb,body); chans["black"]=(chans["black"]&(masks["exclude"]==0)).astype(np.uint8)

def despike(tr,thr=45,iters=2):
    ys=sorted(tr); xs=[tr[y] for y in ys]; n=len(ys)
    for _ in range(iters):
        for i in range(1,n-1):
            if ys[i]-ys[i-1]>3 or ys[i+1]-ys[i]>3: continue
            d1=xs[i]-xs[i-1]; d2=xs[i]-xs[i+1]
            if d1*d2>0 and min(abs(d1),abs(d2))>thr: xs[i]=0.5*(xs[i-1]+xs[i+1])
    return {ys[i]:xs[i] for i in range(n)}

trV1=trace_color(chans["black"],ty,by)
trV4=despike(trV1)

curves=[c for c in m["curves"] if not c["name"].split()[0].startswith("DA")]

def emit(tr,suffix):
    ifds=read_full(open(tpl,"rb").read()); wrote=0
    for c in curves:
        short=c["name"].split()[0]; n=c["n_rows"]; top_y=c["top_y"]
        new_xs=[int(round(tr[top_y+i])) if (top_y+i) in tr else NULL for i in range(n)]
        idxs=find_ifd(ifds, curve_pred(short))
        if not idxs: continue
        set_tag(ifds, idxs[0], 35490, 4, new_xs); wrote+=sum(1 for x in new_xs if x!=NULL)
        # ОДИН сегмент на весь диапазон, level 0 (без перевыносов): без 35492/94/96/98
        # NeuraLOG не рисует кривую (рамка MBK1 имела 35492=0, массивы отсутствовали)
        set_tag(ifds, idxs[0], 35492, 4, [1])
        set_tag(ifds, idxs[0], 35494, 4, [int(c["top_y"])])
        set_tag(ifds, idxs[0], 35496, 4, [int(c["bottom_y"])])
        set_tag(ifds, idxs[0], 35498, 4, [0])
    for i in find_ifd(ifds, lambda tags: 34878 in tags):
        set_tag(ifds, i, 34878, 2, str(img))
    data=write_full(ifds)
    dst=outdir/("Yatskivska_1_MBK_3180_3580_200_D1_auto_%s.nlgx"%suffix)
    open(dst,"wb").write(data); open(dst.with_suffix(".bck"),"wb").write(write_bck(data))
    # верификация
    m2=extract(str(dst)); ok=sum(1 for c2 in ds.real_curves(m2) if any(x!=NULL for x in c2["xs"]))
    print("  %-9s точек=%-6d кривых_с_трассой=%d -> %s"%(suffix,wrote,ok,dst.name))

print("эмит вариантов MBK:")
emit(trV1,"faithful")
emit(trV4,"smooth")
print("\nготово. Оба в F:\\nds\\output\\yat_auto\\ (+ .bck), скан в 34878 -> NeuraLOG откроет со сканом.")
