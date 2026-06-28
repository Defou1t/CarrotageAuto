import sys
import numpy as np
sys.path.insert(0, r"F:\nds\Auto")
import auto.imaging as im
from auto.config import DEFAULT
rgb = im.load_rgb(r"F:\nds\Auto\Yatskivska_1_MK_3130_3760_200_D1.jpg"); p=DEFAULT.cv
V = im.value_channel(rgb)
fg = np.load(r"C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds-Auto\5bbdb336-3b32-4e7a-b006-ae7aeb0160e5\scratchpad\mk_fg_clean.npy")
TY,BY,XL,XR=1665,21296,80,320
# на ЧЁТКО раздельных строках: тёмность ТОНКОГО vs ТОЛСТОГО рана (min V и mean V)
thinV=[]; thickV=[]; thinW=[]; thickW=[]
for y in range(TY,BY,2):
    rr=sorted(im.row_runs(fg[y,XL:XR],gap=2),key=lambda r:r[0])
    if len(rr)!=2: continue
    L,R=rr; gap=R[0]-L[1]
    if gap<15: continue
    wL=L[1]-L[0]+1; wR=R[1]-R[0]+1
    vL=V[y,XL+L[0]:XL+L[1]+1].min(); vR=V[y,XL+R[0]:XL+R[1]+1].min()  # самый тёмный пиксель рана
    if wL==wR: continue
    if wL>wR: thick=(wL,vL); thin=(wR,vR)
    else: thick=(wR,vR); thin=(wL,vL)
    thickW.append(thick[0]); thickV.append(thick[1]); thinW.append(thin[0]); thinV.append(thin[1])
thinV=np.array(thinV);thickV=np.array(thickV)
print(f"раздельных строк: {len(thinV)}")
print(f"ТОЛСТЫЙ(MGZ): ширина мед={np.median(thickW):.0f}  min-V мед={np.median(thickV):.0f} (меньше=темнее)")
print(f"ТОНКИЙ(MPZ):  ширина мед={np.median(thinW):.0f}  min-V мед={np.median(thinV):.0f}")
print(f"MPZ СВЕТЛЕЕ MGZ (min-V больше): {100*np.mean(thinV>thickV):.0f}% строк | разница V мед={np.median(thinV-thickV):.0f}")
# абсолютные пороги: какой V делит толстый/тонкий?
print(f"толстый min-V: p25={np.percentile(thickV,25):.0f} p50={np.percentile(thickV,50):.0f} p75={np.percentile(thickV,75):.0f}")
print(f"тонкий  min-V: p25={np.percentile(thinV,25):.0f} p50={np.percentile(thinV,50):.0f} p75={np.percentile(thinV,75):.0f}")
