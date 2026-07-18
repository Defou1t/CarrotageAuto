"""МУЛЬТИ-ВЕТКА, шаг 1: ловят ли НИТИ (link_strands по связности ранов) кривые мульти-листа?
Полосы по x опровергнуты (§5.2), разбор по счёту тоже (§5.3). Нить — третий вариант: вести
штрих по связности, как в BKZ M1 (там детект нитей покрывал GT на 100%, med 1.0px).

Меряем ПОТОЛОК подхода: для каждой GT-кривой берём ЛУЧШУЮ нить (оракул по med|dx|) и печатаем
med/≤3px/покрытие. Это НЕ метрика продукта — это ответ на вопрос «есть ли что выбирать».
Плюс N (сколько длинных нитей) против K (сколько кривых в рамке) — цена задачи выбора.

  python _multi_strand_probe.py [N листов] [--tok ...] [--prob DIR] [--min-cov 0.2]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from bkz_track import link_strands
from _multi_replica_probe import dense
from auto import imaging as im, meta as M, frame as F, emit as E
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=12)
ap.add_argument("--tok", default="BK, IK|GK, NGK|BKZ, DS|MBK, MDS|STK+DS|BK+IK|MK, MDS")
ap.add_argument("--per-well", type=int, default=1)
ap.add_argument("--prob", default="")
ap.add_argument("--min-cov", type=float, default=0.2, help="доля высоты рамки для «длинной» нити")
a = ap.parse_args()
TOKS = [t.strip().upper() for t in a.tok.split("|")]
p = Config().cv
PROBDIR = Path(a.prob) if a.prob else None

cands, per_well = [], {}
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    well = wlg.parent.name
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem or per_well.get(well, 0) >= a.per_well:
            continue
        m = M.parse_filename(n.name, MN)
        if m.curves_token.strip().upper() not in TOKS or not find_image(n):
            continue
        cands.append((well, n, find_image(n), m))
        per_well[well] = per_well.get(well, 0) + 1
cands = cands[:a.n]
print(f"листов: {len(cands)}")

allm = []
for well, n, img, m in cands:
    mo = extract(str(n))
    rgb = im.load_rgb(str(img))
    fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
    prob = None
    if PROBDIR is not None:
        q = PROBDIR / f"{Path(img).stem}_prob.npy"
        if q.is_file():
            from auto.prob import prob_from_npy
            prob = prob_from_npy(q)(rgb)
    fg = im.ink_foreground(rgb, p, prob=prob)
    gts = [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
           and M.mnem_root(c["name"]) != "DA"]
    tid = Counter(E._slot_track(mo, c, fr) for c in gts)
    best_t = tid.most_common(1)[0][0]
    gts = [c for c in gts if E._slot_track(mo, c, fr) == best_t]
    if not gts:
        continue
    t = fr.tracks[best_t if best_t is not None else 0]
    y0, y1 = int(fr.top_y), int(fr.bottom_y)
    lo, hi = int(t.x_left) + 3, int(t.x_right) - 3
    H = max(1, y1 - y0)
    strands = link_strands(fg.astype(bool), lo, hi, y0, y1)
    long_s = [s for s in strands if len(s) >= a.min_cov * H]
    K = len(gts)
    line = f"{well:<14} {m.curves_token:<12} K={K} нитей={len(strands):>4} длинных={len(long_s):>3}  "
    for c in gts:
        d = dense(c)
        best, bmed, bcov = None, 1e9, 0.0
        for s in long_s:
            common = [y for y in s if y in d]
            if len(common) < 50:
                continue
            e = np.array([abs(s[y] - d[y]) for y in common], float)
            if np.median(e) < bmed:
                bmed, best, bcov = float(np.median(e)), s, len(common) / max(1, len(d))
        nm = c["name"].split()[0]
        if best is None:
            line += f"{nm}:НЕТ "; continue
        e = np.array([abs(best[y] - d[y]) for y in best if y in d], float)
        line += f"{nm}:med{bmed:.0f}/≤3px{(e<=3).mean()*100:.0f}%/cov{bcov:.2f} "
        allm.append((bmed, float((e <= 3).mean()), bcov, len(long_s), K))
    print(line)

if allm:
    med = np.array([x[0] for x in allm]); p3 = np.array([x[1] for x in allm])
    cov = np.array([x[2] for x in allm])
    print(f"\nПОТОЛОК НИТЕЙ (оракул выбирает лучшую) по {len(allm)} кривым:")
    print(f"  med(med)={np.median(med):.1f}px  кривых med<=3px: {(med<=3).sum()}/{len(med)}  "
          f"≤3px сред {p3.mean()*100:.0f}%  покрытие сред {cov.mean():.2f}")
    print(f"  цена выбора: длинных нитей на лист {np.median([x[3] for x in allm]):.0f} против K={np.median([x[4] for x in allm]):.0f}")
