r"""_level_gate.py — ШЛЮЗ УВЕРЕННОСТИ МОДЕЛИ УРОВНЯ (03.10, §6.268; критерии — в ROADMAP до прогона).

База — v2 (§6.266) + участки для сдвиговых цепочек (§6.267). Для кривых НЕ со сдвиговой цепочкой: в позиции, где максимум среднего
softmax ансамбля < τ, берётся уровень прежнего декодера (уровень выдачи прода в этой позиции); дальше как в v2 — на все строки по
ближайшей позиции, enforce_min_run 25. Вероятности — дамп `_level_ink.py` (вне фолда на поле, модель на всём поле для сорта A).

  _level_gate.py --taus 0.6 0.7 0.8 0.9
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from collections import Counter, defaultdict
import numpy as np
from _level_ink import honest, seg_levels
from auto.level_ink import is_shift, min_run_runs

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--pred", default=r"F:/nds/output/taskS/level_ink_pred_v2r.pkl")
ap.add_argument("--tm", default=r"F:/nds/output/taskS/level_tm_levels.pkl")
ap.add_argument("--taus", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
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


def base_lo(ci):
    if ci in TM:
        return TM[ci]
    return LV.get(ci)


VAR = {"база": {}}
for tau in a.taus:
    VAR[f"τ={tau:g}"] = {}
    for ci, (Pp, arg) in PRED.items():
        cv = CUR[ci]
        if ci in TM or is_shift(cv["chain"]):
            continue
        pr = PROB[ci].astype(np.float32)
        conf = pr.max(0)
        lwP = seg_levels(cv["lw"], Pp.astype(np.int64)).astype(np.int16)
        lev = np.where(conf >= tau, arg, lwP).astype(np.int16)
        VAR[f"τ={tau:g}"][ci] = to_changes(Pp, lev)
RES = {}
for nm, over in VAR.items():
    C = Counter(); PER = defaultdict(int); FH = Counter()
    for ci, cv in enumerate(CUR):
        lo = over.get(ci, base_lo(ci))
        h = honest(cv, lo)
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
        fam = "; ".join(f"{f} {F0[(sn, f)]}→{F1[(sn, f)]}" for f in sorted({k[1] for k in F0 if k[0] == sn}, key=lambda f: -abs(F1[(sn, f)] - F0[(sn, f)]))[:6]
                        if F1[(sn, f)] != F0[(sn, f)])
        print(f"   {nm} против базы, {sn}: Δ {int(d.sum()):+d} (листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f}); {fam}")
