r"""_level_viterbi.py — ВИТЕРБИ ПО ВЕРОЯТНОСТЯМ МОДЕЛИ УРОВНЯ ВМЕСТО ПОЗИЦИОННОГО ARGMAX (04.10, §6.270; критерии — в ROADMAP).

Путь уровней по всей кривой: стоимость позиции −ln(p_k + 1e-6) по среднему softmax ансамбля (дамп `_level_ink.py`: вне фолда на
поле, модель на всём поле для сорта A), λ за смену уровня; путь → на все строки по ближайшей позиции → enforce_min_run 25.
Сдвиговые цепочки — участки §6.267 (`level_tm_levels.pkl`), как в базе. База — v2 + участки (argmax по позициям).

  _level_viterbi.py --lams 2 5 10
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from collections import Counter, defaultdict
import numpy as np
from _level_ink import honest
from auto.level_ink import min_run_runs

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--pred", default=r"F:/nds/output/taskS/level_ink_pred_v2r.pkl")
ap.add_argument("--tm", default=r"F:/nds/output/taskS/level_tm_levels.pkl")
ap.add_argument("--lams", nargs="+", type=float, default=[2.0, 5.0, 10.0])
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--min-run", type=int, default=25)
a = ap.parse_args()
CUR = pickle.load(open(a.cache, "rb"))
D = pickle.load(open(a.pred, "rb"))
LV, PRED, PROB = D["levels"], D["pred"], D["prob"]
TM = pickle.load(open(a.tm, "rb"))
half = a.stride // 2


def to_changes(Pp, p):
    Pp = Pp.astype(np.int64)
    ch_ = np.flatnonzero(np.diff(np.concatenate([[-1], p])) != 0)
    starts = Pp[ch_] - half; starts[0] = Pp[0] - half
    ends = np.concatenate([Pp[ch_[1:]] - half - 1, [Pp[-1] - half + a.stride - 1]])
    R = min_run_runs(starts, p[ch_], ends, a.min_run)
    return {r[0]: r[2] for r in R}


def viterbi(cost, lam):
    """cost [K, n] → путь минимальной стоимости с λ за смену уровня"""
    K, n = cost.shape
    dp = cost[:, 0].copy(); back = np.zeros((n, K), np.int16)
    for t in range(1, n):
        stay = dp; best = int(np.argmin(dp)); sw = dp[best] + lam
        take_sw = sw < stay
        back[t] = np.where(take_sw, best, np.arange(K))
        dp = np.where(take_sw, sw, stay) + cost[:, t]
    k = int(np.argmin(dp)); path = np.empty(n, np.int16); path[-1] = k
    for t in range(n - 1, 0, -1):
        k = int(back[t, k]); path[t - 1] = k
    return path


def base_lo(ci):
    return TM[ci] if ci in TM else LV.get(ci)


VAR = {"база": {}}
for lam in a.lams:
    over = {}
    for ci, (Pp, arg) in PRED.items():
        if ci in TM:
            continue
        cost = -np.log(PROB[ci].astype(np.float64) + 1e-6)
        over[ci] = to_changes(Pp, viterbi(cost, lam))
    VAR[f"λ={lam:g}"] = over
RES = {}
for nm, over in VAR.items():
    C = Counter(); PER = defaultdict(int); FH = Counter()
    for ci, cv in enumerate(CUR):
        h = honest(cv, over.get(ci, base_lo(ci)))
        C[cv["set"]] += h; PER[(cv["set"], cv["sheet"])] += h
        if len(cv["chain"]) >= 2:
            FH[(cv["set"], re.sub(r"^BKZ_", "", cv["root"]))] += h
    RES[nm] = (C, PER, FH)
    print(f"★ {nm:7s}: именных честных в значениях — поле {C['поле']}, сорт A {C['сорт A']}")
rng = np.random.default_rng(0)
C0, P0, F0 = RES["база"]
for nm in RES:
    if nm == "база":
        continue
    C1, P1, F1 = RES[nm]
    for sn in ("поле", "сорт A"):
        keys = {k for k in list(P0) + list(P1) if k[0] == sn}
        d = np.array([P1[k] - P0[k] for k in keys]); nz = d[d != 0]
        p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
        fam = "; ".join(f"{f} {F0[(sn, f)]}→{F1[(sn, f)]}" for f in sorted({k[1] for k in F0 if k[0] == sn},
                                                                         key=lambda f: -abs(F1[(sn, f)] - F0[(sn, f)]))[:6]
                        if F1[(sn, f)] != F0[(sn, f)])
        print(f"   {nm} против базы, {sn}: Δ {int(d.sum()):+d} (листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f}); {fam}")
