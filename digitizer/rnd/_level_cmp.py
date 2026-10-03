r"""_level_cmp.py — ПАРНОЕ СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ УРОВНЯ НА СТЕНДЕ (03.10, к §6.269).

Каждая сторона — дамп `_level_ink.py` (уровни вне фолда на поле, модель на всём поле для сорта A); поверх обеих — участки §6.267
для сдвиговых цепочек (`_level_tm.py` → `level_tm_levels.pkl`), как в проде. Кривые без уровней модели — уровни прода.
Мера — именные честные в значениях; парный знаковый тест по листам; разрез по семействам.

  _level_cmp.py --a F:/nds/output/taskS/level_ink_pred_v2r.pkl --b F:/nds/output/taskS/level_ink_pred_v3.pkl
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from collections import Counter, defaultdict
import numpy as np
from _level_ink import honest

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--tm", default=r"F:/nds/output/taskS/level_tm_levels.pkl", help="\"\" — без участков")
a = ap.parse_args()
CUR = pickle.load(open(a.cache, "rb"))
LA = pickle.load(open(a.a, "rb"))["levels"]; LB = pickle.load(open(a.b, "rb"))["levels"]
TM = pickle.load(open(a.tm, "rb")) if a.tm else {}
C = Counter(); PER = defaultdict(lambda: [0, 0]); FH = Counter()
for ci, cv in enumerate(CUR):
    ha = honest(cv, TM[ci] if ci in TM else LA.get(ci))
    hb = honest(cv, TM[ci] if ci in TM else LB.get(ci))
    C[(cv["set"], 0)] += ha; C[(cv["set"], 1)] += hb
    PER[(cv["set"], cv["sheet"])][0] += ha; PER[(cv["set"], cv["sheet"])][1] += hb
    if len(cv["chain"]) >= 2:
        f = re.sub(r"^BKZ_", "", cv["root"])
        FH[(cv["set"], f, 0)] += ha; FH[(cv["set"], f, 1)] += hb
rng = np.random.default_rng(0)
for sn in ("поле", "сорт A"):
    d = np.array([v1 - v0 for (s, _), (v0, v1) in PER.items() if s == sn]); nz = d[d != 0]
    p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
    print(f"★ {sn}: именных честных в значениях — A {C[(sn, 0)]}, B {C[(sn, 1)]} (Δ {C[(sn, 1)] - C[(sn, 0)]:+d}, "
          f"листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
    fams = sorted({k[1] for k in FH if k[0] == sn}, key=lambda f_: -abs(FH[(sn, f_, 1)] - FH[(sn, f_, 0)]))
    print("   по семействам (A → B): " + "; ".join(f"{f_} {FH[(sn, f_, 0)]} → {FH[(sn, f_, 1)]}" for f_ in fams[:10]
                                                   if FH[(sn, f_, 1)] != FH[(sn, f_, 0)]))
