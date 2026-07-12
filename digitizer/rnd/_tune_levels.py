# -*- coding: utf-8 -*-
"""Тюнинг decode_rail на ЧИСТОЙ экспертной трассе vs GT-уровни (тег 35498) — изолирует логику
уровней от шума нашей трассы. Грид по rail_frac / min_jump_frac / min_run → level-acc."""
import sys, itertools
from pathlib import Path
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
import dataset as ds, decode_levels as DL
from auto import refine as R

ARCH = Path(r"F:\nds\projects\Archive")
LIMIT = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else 120

def expert_trace(c):
    ty=c["top_y"]; return {ty+i:x for i,x in enumerate(c["xs"]) if x!=NULL}
def gt_levels(c):
    out={}
    for s,e,l in (c.get("segments") or []):
        for y in range(s,e+1): out[y]=l
    return out

# собрать резист. кривые с 5х-цепочкой И реальными переходами (GTmax≥1)
curves=[]
for wlg in sorted(ARCH.glob("*/wlg")):
    for nlgx in sorted(wlg.glob("*.nlgx")):
        if "_auto" in nlgx.stem: continue
        try: m=extract(str(nlgx))
        except: continue
        for c in ds.real_curves(m):
            fam=DL.build_family(m,c)
            gl=gt_levels(c)
            if DL.is_resistive(c["name"]) and len(fam)>=2 and gl and max(gl.values())>=1:
                et=expert_trace(c)
                if len(et)>=200:
                    ratio=max(1.5, abs(fam[1]["v_right"]-fam[1]["v_left"])/(abs(fam[0]["v_right"]-fam[0]["v_left"]) or 1))
                    curves.append((et, gl, fam[0]["x_left"], fam[0]["x_right"], ratio))
        if len(curves)>=LIMIT: break
    if len(curves)>=LIMIT: break
print(f"кривых с реальными 5х-переходами: {len(curves)}")

def evalp(rail_frac, mjf, min_run):
    accs=[]; fp5=[]  # доля ложных 5х (наши=5х, GT=1х)
    for et,gl,xl,xr,ratio in curves:
        lv=R.decode_rail(et,xl,xr,ratio=ratio,rail_frac=rail_frac,min_jump_frac=mjf,min_run=min_run)
        common=[y for y in et if y in gl]
        if not common: continue
        acc=np.mean([lv.get(y,0)==gl[y] for y in common])
        accs.append(acc)
        # ложные 5х: наши>0 где GT=0
        n0=[y for y in common if gl[y]==0]
        if n0: fp5.append(np.mean([lv.get(y,0)>0 for y in n0]))
    return np.median(accs), np.mean(accs), (np.median(fp5) if fp5 else 0)

# СРАВНЕНИЕ: старый DP decode_levels (+ min_run) на тех же чистых трассах
def eval_dp(lam, min_run):
    accs=[]; fp5=[]
    for et,gl,xl,xr,ratio in curves:
        # восстановить family с v — нужен полный fam; пере-decode через DL напрямую
        lv=R.enforce_min_run(_dp_cache[id(et)](lam), min_run)
        common=[y for y in et if y in gl]
        if not common: continue
        accs.append(np.mean([lv.get(y,0)==gl[y] for y in common]))
        n0=[y for y in common if gl[y]==0]
        if n0: fp5.append(np.mean([lv.get(y,0)>0 for y in n0]))
    return np.median(accs),np.mean(accs),(np.median(fp5) if fp5 else 0)

# кэш DP-декода: нужен полный fam (со шкалами) — пересоберём curves с fam
print("собираю fam для DP...")
_dp_cache={}
curves2=[]
for wlg in sorted(ARCH.glob("*/wlg")):
    for nlgx in sorted(wlg.glob("*.nlgx")):
        if "_auto" in nlgx.stem: continue
        try: m=extract(str(nlgx))
        except: continue
        for c in ds.real_curves(m):
            fam=DL.build_family(m,c); gl=gt_levels(c)
            if DL.is_resistive(c["name"]) and len(fam)>=2 and gl and max(gl.values())>=1:
                et=expert_trace(c)
                if len(et)>=200:
                    curves2.append((et,gl,fam))
        if len(curves2)>=LIMIT: break
    if len(curves2)>=LIMIT: break
def dp_acc(lam,min_run):
    accs=[]
    for et,gl,fam in curves2:
        lv=R.enforce_min_run(DL.decode({y:float(x) for y,x in et.items()},fam,lam),min_run)
        common=[y for y in et if y in gl]
        if common: accs.append(np.mean([lv.get(y,0)==gl[y] for y in common]))
    return np.median(accs),np.mean(accs)
print("\nСТАРЫЙ DP decode_levels:")
for lam in (0.4,0.7,1.5):
    for mr in (1,25):
        md,mn=dp_acc(lam,mr); print(f"  lam={lam} min_run={mr}: level-acc мед={md:.2f} сред={mn:.2f}")

# бейзлайн rail
print("\nRAIL текущие дефолты (rail_frac=0.85 mjf=0.45 min_run=25):")
print("  level-acc мед=%.2f сред=%.2f ложн5х-мед=%.2f" % evalp(0.85,0.45,25))

print("\nгрид:")
best=None
for rf, mjf, mr in itertools.product((0.75,0.82,0.88,0.93),(0.30,0.40,0.50),(15,25,40)):
    med,mean,fp=evalp(rf,mjf,mr)
    sc=mean-0.5*fp   # штраф за ложные 5х
    if best is None or sc>best[0]:
        best=(sc,rf,mjf,mr,med,mean,fp)
print("ЛУЧШЕЕ: rail_frac=%.2f mjf=%.2f min_run=%d → level-acc мед=%.2f сред=%.2f ложн5х=%.2f" % best[1:])
# топ-5
res=[]
for rf,mjf,mr in itertools.product((0.75,0.82,0.88,0.93),(0.30,0.40,0.50),(15,25,40)):
    med,mean,fp=evalp(rf,mjf,mr); res.append((mean-0.5*fp,rf,mjf,mr,med,mean,fp))
for r in sorted(res,reverse=True)[:6]:
    print("  rf=%.2f mjf=%.2f mr=%d: acc_мед=%.2f acc_сред=%.2f ложн5х=%.2f" % (r[1],r[2],r[3],r[4],r[5],r[6]))
