r"""_knob_cmp.py — ПАРНЫЕ СРАВНЕНИЯ РЕЖИМОВ ПОВТОРА С БАЗОЙ ПО ДАМПАМ СЧЁТА (04.10, §6.275).

Каждый режим посчитан `_name_cost_prod.py --mode X --dump <префикс>_X.pkl` (мера приёмки). Здесь — по полю и сорту A: именных
и безымянных, Δ к базе, листов ↑/↓, перестановочный p; приговор по критерию §6.275 (отбор: поле именных Δ > 0 при p < α / k
и безымянных Δ ≥ −5; принятие: сорт A именных и безымянных Δ ≥ 0).

  _knob_cmp.py --prefix F:/nds/output/taskS/rp_knob --base K0 --modes K1 K2 K3 K4 K5 K6
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--prefix", required=True)
ap.add_argument("--base", default="K0")
ap.add_argument("--modes", nargs="+", required=True)
ap.add_argument("--alpha", type=float, default=0.05)
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
load = lambda m: pickle.load(open(f"{a.prefix}_{m}.pkl", "rb"))[m]
B = load(a.base)
rng = np.random.default_rng(0)
k = len(a.modes); thr = a.alpha / k


def test(d):
    nz = d[d != 0]
    if not len(nz):
        return 1.0
    return float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum())))


for sn, S in SETS.items():
    S0 = [s for s in S if s in B]
    print(f"★ {sn}: база {a.base} — именных {sum(B[s][0] for s in S0)}, безымянных {sum(B[s][1] for s in S0)} ({len(S0)} листов)")
best = None
for m in a.modes:
    try:
        V = load(m)
    except Exception as e:
        print(f"   {m}: нет дампа ({e})"); continue
    res = {}
    for sn, S in SETS.items():
        S0 = [s for s in S if s in B and s in V]
        r = {}
        for j, what in ((0, "именных"), (1, "безымянных")):
            d = np.array([V[s][j] - B[s][j] for s in S0])
            r[what] = (int(d.sum()), int((d > 0).sum()), int((d < 0).sum()), test(d))
        res[sn] = r
        print(f"   {m} {sn}: " + "; ".join(f"{w} Δ {v[0]:+d} (↑{v[1]}/↓{v[2]}, p = {v[3]:.4f})" for w, v in r.items()))
    f, h = res["поле"], res["сорт A"]
    sel = f["именных"][0] > 0 and f["именных"][3] < thr and f["безымянных"][0] >= -5
    acc = sel and h["именных"][0] >= 0 and h["безымянных"][0] >= 0
    print(f"   → {m}: {'ОТОБРАН' if sel else 'не отобран'}{', ПРИНЯТ' if acc else ''} (порог p < {thr:.4f})")
    if acc and (best is None or f["именных"][0] > best[1]):
        best = (m, f["именных"][0])
print(f"★ ИТОГ: {'принят ' + best[0] + f' (поле именных {best[1]:+d})' if best else 'ни один вариант не принят'}")
