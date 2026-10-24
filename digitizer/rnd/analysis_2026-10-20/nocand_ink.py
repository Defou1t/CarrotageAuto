"""Слоты без честного кандидата (поле + вне, сканы line_choice_i / line_choice_out_i): лучший кандидат (покрытие, медиана) и есть ли
тушь под эталоном — доля строк эталона, где темнота (бумага − серый) в ±2 px от x эталона ≥ 40. Высокая доля — линия на скане
видна, а декодер её не дал; низкая — эксперт вёл там, где туши почти нет. Пишет примеры в `--out` для просмотра.
nocand_ink.py [--limit N]"""
import sys, pickle, glob, hashlib, json, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=r"F:/nds/output/taskS/nocand_ink.pkl")
a = ap.parse_args()
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from auto import level_ink as LI
TS = Path(r"F:/nds/output/taskS")
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
HLD = lst("holdoutA_sheets.txt")
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
B = Counter(); INK = defaultdict(list); CTL = defaultdict(list); FAM = Counter(); EX = []; tot = 0
files = [(f, "tcache") for f in sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl")))] + \
        [(f, "tcache_out") for f in sorted(glob.glob(str(TS / "line_choice_out_i" / "*.pkl")))]
if a.limit:
    files = files[::max(1, len(files) // a.limit)]
for n_, (f, cache) in enumerate(files):
    r = pickle.load(open(f, "rb")); sh = r["sheet"]
    if sh in HLD or sh not in SRC:
        continue
    miss = []
    for s in r["slots"]:
        if not s["has_truth"]:
            continue
        tot += 1
        labs = [c["lab"] for c in s["cands"] if c["lab"] is not None]
        if any(l[0] for l in labs):
            continue
        cov, med = max(((l[1], -l[2]) for l in labs), default=(0.0, -99.0)); med = -med
        b = "нет кандидатов" if not labs else ("покр ≥ 0.9, мед > 15" if cov >= 0.9 and med > 15 else "покр ≥ 0.9, мед 3–15" if cov >= 0.9
                                              else "покр < 0.9")
        miss.append((s, b, cov, med))
    if not miss:
        continue
    q = SRC[sh]
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    tc = TS / cache / (key + ".pkl")
    if not tc.exists():
        continue
    try:
        gray = np.asarray(Image.open(pickle.load(open(tc, "rb"))["image"]).convert("L"))
    except Exception:
        continue
    paper = LI.paper_of(gray).astype(np.int16); H, W = gray.shape
    mt = {c["name"]: c for c in extract(str(q))["curves"]}
    for s, b, cov, med in miss:
        c = mt.get(s["name"])
        if c is None:
            continue
        pts = [(s["top_y"] + i, x) for i, x in enumerate(c["xs"]) if x != NULL]
        if not pts:
            continue
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts], float)
        ok = (ys >= 0) & (ys < H)
        ys, xs = ys[ok], xs[ok]
        if len(ys) < 50:
            continue
        sel = np.linspace(0, len(ys) - 1, min(400, len(ys))).astype(int)
        dk = []; dc = []
        for i in sel:
            x1 = int(round(xs[i])) + 25
            xa, xb = max(0, x1 - 2), min(W, x1 + 3)
            dc.append(int((paper[ys[i]] - gray[ys[i], xa:xb].astype(np.int16)).max()) if xa < xb else 0)
            x0 = int(round(xs[i]))
            xa, xb = max(0, x0 - 2), min(W, x0 + 3)
            if xa >= xb:
                dk.append(0); continue
            dk.append(int((paper[ys[i]] - gray[ys[i], xa:xb].astype(np.int16)).max()))
        fr = float(np.mean(np.array(dk) >= 40)); frc = float(np.mean(np.array(dc) >= 40))
        B[b] += 1; INK[b].append(fr); CTL[b].append(frc); FAM[s["root"]] += 1
        EX.append(dict(sheet=sh, cache=cache, slot=s["name"], root=s["root"], cat=b, cov=cov, med=med, ink=fr, ink_ctl=frc, top_y=s["top_y"], band=s["band"]))
    if (n_ + 1) % 200 == 0:
        print(f"  {n_ + 1}/{len(files)}", flush=True)
print(f"★ слотов с эталоном (без сорта A) {tot}, без честного кандидата {sum(B.values())}")
for b, v in B.most_common():
    x = np.array(INK[b])
    print(f"   {v:5d}  {b}: тушь под эталоном — медиана доли {np.median(x):.2f}; ≥ 0.8 у {int((x >= 0.8).sum())}, < 0.5 у {int((x < 0.5).sum())}; контроль +25 px — медиана {np.median(CTL[b]):.2f}")
print("   по семействам:", dict(FAM.most_common(14)))
pickle.dump(EX, open(a.out, "wb"))
