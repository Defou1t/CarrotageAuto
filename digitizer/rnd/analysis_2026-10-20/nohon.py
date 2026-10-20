"""Слоты поля без честного кандидата: почему — покрытие или расстояние; есть ли у эталона развороты (скачок x > половины полосы)."""
import sys, pickle, glob, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL

TS = Path(r"F:/nds/output/taskS")
FLD = {l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
C = Counter(); W = Counter(); FAM = Counter(); FAMH = Counter(); EX = []
for f in sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl"))):
    r = pickle.load(open(f, "rb"))
    if r["sheet"] not in FLD or r["sheet"] not in SRC:
        continue
    m = None
    for s in r["slots"]:
        if not s["has_truth"]:
            continue
        labs = [c["lab"] for c in s["cands"] if c["lab"] is not None]
        hon = any(l[0] for l in labs)
        if m is None:
            m = {c["name"]: c for c in extract(str(SRC[r["sheet"]]))["curves"]}
        gt = m.get(s["name"])
        wraps = 0
        if gt is not None:
            pts = [(i, x) for i, x in enumerate(gt["xs"]) if x != NULL]
            bw = abs(s["band"][1] - s["band"][0])
            wraps = sum(1 for (y0, x0), (y1, x1) in zip(pts, pts[1:]) if y1 - y0 <= 5 and abs(x1 - x0) > 0.5 * bw)
        key = "есть честный" if hon else "нет честного"
        W[(key, "разворотов 0" if wraps == 0 else ("1–2" if wraps <= 2 else "≥3"))] += 1
        FAMH[(s["root"], hon)] += 1
        if hon:
            continue
        if not labs:
            C["нет кандидатов в полосе"] += 1; continue
        near = [l for l in labs if l[2] <= 3.0]
        full = [l for l in labs if l[1] >= 0.9]
        if near:
            cat = f"медиана ≤ 3, покрытие < 0.9 (лучшее {max(l[1] for l in near):.1f})"
            cat = "медиана ≤ 3, покрытие " + ("0.7–0.9" if max(l[1] for l in near) >= 0.7 else ("0.4–0.7" if max(l[1] for l in near) >= 0.4 else "< 0.4"))
        elif full:
            md = min(l[2] for l in full)
            cat = "покрытие ≥ 0.9, медиана " + ("3–6" if md <= 6 else ("6–15" if md <= 15 else "> 15"))
        else:
            cat = "покрытие < 0.9 и медиана > 3"
        C[cat] += 1
        EX.append((r["sheet"], s["name"], cat, wraps))
print("★ слоты поля без честного кандидата:", sum(C.values()))
for k, v in C.most_common():
    print(f"   {v:5d}  {k}")
print("★ развороты эталона (скачок x > ½ полосы за ≤ 5 строк):")
for k, v in sorted(W.items()):
    print(f"   {v:5d}  {k}")
print("★ по семействам (нет честного / всего):", "; ".join(f"{f_} {FAMH[(f_, False)]}/{FAMH[(f_, False)] + FAMH[(f_, True)]}"
      for f_, _ in Counter({k[0]: 0 for k in FAMH}).items() if FAMH[(f_, False)] + FAMH[(f_, True)] >= 40))
pickle.dump(EX, open(TS / "_nohon_ex.pkl", "wb"))
