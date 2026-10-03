r"""_rail_scan.py — УПОР ПЕРА 1× НА СКАНЕ КАК ПРИЗНАК УРОВНЯ (03.10, §6.265).

Две записи одной кривой (перья 1× и 5×): в строках, где эксперт берёт 5×, перо 1× стоит на упоре и рисует вертикальную линию в
постоянной колонке; в строках 1× этой линии нет. По эталону (каждый N-й лист поля и сорта A, кривые с переходами и цепочкой): в
полосе шкалы 1× для каждой колонки — доля тёмных строк среди строк уровня ≥ 1 (f1) и уровня 0 (f0), каждая 3-я строка; колонка
упора — максимум f1 − f0 (сетка тёмная в обоих и контраста не даёт). Кривая «с упором», если f1 − f0 ≥ 0.5 и f1 ≥ 0.7. Для неё —
точность правила «уровень ≥ 1 там, где в колонке упора есть тушь (±2 px)» против уровней эксперта по строкам.

  _rail_scan.py --every 3
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
from auto import meta as M
import decode_levels as DL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--thr", type=int, default=40)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/rail_scan.pkl")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
        IMGS.setdefault(q.stem, q)
R = []
for sh in (lst("wellmap_sheets.txt") + lst("holdoutA_sheets.txt"))[::a.every]:
    q = SRC.get(sh)
    if not q or q.stem not in IMGS:
        continue
    mt = extract(str(q))
    cur = [c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and any(l > 0 for _, _, l in (c.get("segments") or []))
           and len(DL.build_family(mt, c)) >= 2]
    if not cur:
        continue
    gray = np.asarray(Image.open(IMGS[q.stem]).convert("L"), np.int16)
    paper = np.percentile(gray[:, ::4], 90, axis=1)
    H, W = gray.shape
    for c in cur:
        fam = DL.build_family(mt, c)
        xl, xr = int(min(fam[0]["x_left"], fam[0]["x_right"])), int(max(fam[0]["x_left"], fam[0]["x_right"]))
        xl, xr = max(0, xl - 10), min(W, xr + 10)
        rows = np.arange(int(c["top_y"]), int(c["top_y"]) + len(c["xs"]), 3)
        rows = rows[(rows >= 0) & (rows < H)]
        lv = np.zeros(len(rows), np.int16)
        for y0, y1, l in c["segments"]:
            if l:
                lv[(rows >= y0) & (rows <= y1)] = l
        if (lv > 0).sum() < 30 or (lv == 0).sum() < 30:
            continue
        D = gray[rows][:, xl:xr] < (paper[rows][:, None] - a.thr)          # тёмные пиксели [строки, колонки]
        # колонка «с запасом ±2 px»: тушь в любой из 5 соседних колонок
        Dw = np.zeros_like(D)
        for s_ in range(-2, 3):
            Dw |= np.roll(D, s_, axis=1)
        f1 = Dw[lv > 0].mean(0); f0 = Dw[lv == 0].mean(0)
        k = int(np.argmax(f1 - f0))
        con, a1 = float(f1[k] - f0[k]), float(f1[k])
        pred = Dw[:, k]
        acc = float(np.mean((pred > 0) == (lv > 0)))
        R.append(dict(sheet=sh, name=c["name"], root=M.mnem_root(c["name"]), con=con, f1=a1, f0=float(f0[k]), col=k + xl,
                      rel=(k + xl - fam[0]["x_left"]) / max(1, fam[0]["x_right"] - fam[0]["x_left"]), acc=acc,
                      up=float(np.mean(lv > 0))))
    del gray
pickle.dump(R, open(a.dump, "wb"))
con = np.array([r["con"] for r in R]); f1 = np.array([r["f1"] for r in R])
rail = (con >= 0.5) & (f1 >= 0.7)
print(f"★ кривых с переходами (каждый {a.every}-й лист): {len(R)}; «с упором» (f1 − f0 ≥ 0.5, f1 ≥ 0.7): {int(rail.sum())} ({rail.mean():.0%})")
acc = np.array([r["acc"] for r in R])
print(f"   правило «уровень ≥ 1 ⇔ тушь в колонке упора»: верных строк у кривых с упором — медиана {np.median(acc[rail]):.2f}, "
      f"≥ 0.9 у {np.mean(acc[rail] >= 0.9):.0%}; у прочих — медиана {np.median(acc[~rail]):.2f}")
rel = np.array([r["rel"] for r in R])[rail]
print(f"   колонка упора в долях шкалы 1×: медиана {np.median(rel):.2f}, квартили {np.percentile(rel, 25):.2f}–{np.percentile(rel, 75):.2f}")
fam = Counter(r["root"] for r in R)
for f, n in fam.most_common(12):
    m = np.array([r["root"] == f for r in R])
    print(f"   {f:8s} {n:4d}: с упором {np.mean(rail[m]):.0%}, точность правила у них {np.median(acc[m & rail]) if (m & rail).any() else float('nan'):.2f}")
