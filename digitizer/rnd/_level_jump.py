r"""_level_jump.py — УРОВЕНЬ МАСШТАБА МЕНЯЕТСЯ ТОЛЬКО НА ПРЫЖКЕ ПЕРА (05.10, §6.293; критерии — в ROADMAP).

Глазами (`visual_2026-10-03/H_level_loss/ask`): у старых ГЗ перо ОДНО, и прибор переключает масштаб. Перо прыгает справа налево
(переход на 5×, «пунктир от завершения масштаба до начала» — ответ заказчика 03.10) или слева направо (назад на 1×), а пера-
близнеца нет (поэтому тушь близнецов не разделяла уровни, §6.265 / §6.289). ⇒ уровень может смениться только там, где трасса
прыгает, и значение до прыжка на уровне k совпадает со значением после прыжка на уровне k ± 1.

Стенд: кривые `_level_bench.py build` (трассы выдачи) + вероятности модели уровня по позициям (дамп `_level_ink.py train`,
вне фолда). Витерби по позициям: состояния — уровни; цена позиции — −log p модели; смена уровня на позиции с прыжком, где
значение сходится (|пиксель на новом уровне − x после прыжка| ≤ tol), стоит `--c-evt`, без такого прыжка — `--c-free`.
Дальше как в проде: пробеги → enforce_min_run → именные честные в значениях (`honest` из `_level_ink.py`).
Сравнение: уровни выдачи (прежний декодер), argmax модели, Витерби с прыжками.

  _level_jump.py --cache level_bench_nslx.pkl --pred level_ink_pred_v3r.pkl --grid 0.5:30 1:20 2:10
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench_nslx.pkl")
ap.add_argument("--pred", default=r"F:/nds/output/taskS/level_ink_pred_v3r.pkl")
ap.add_argument("--grid", nargs="+", default=["0.5:30", "1:20", "2:10"], help="c_evt:c_free")
ap.add_argument("--jump", type=float, default=0.25, help="прыжок — |Δx| за ≤ 8 строк больше этой доли ширины полосы")
ap.add_argument("--tol", type=float, default=0.04, help="сходимость значения — доля ширины полосы")
ap.add_argument("--min-run", type=int, default=25)
ap.add_argument("--stride", type=int, default=8)
a = ap.parse_args()
import _level_ink as LI_STAND   # noqa: E402  (функции honest / min_run_runs / val / pix; стенд не исполняется при импорте — main за __name__)

CUR = pickle.load(open(a.cache, "rb"))
PR = pickle.load(open(a.pred, "rb"))
prob = PR["prob"]
KMAX = 4
half = a.stride // 2


def events(cv, Pp, K):
    """→ {позиция: множество разрешённых переходов (k, k2)} по прыжкам трассы"""
    ty, tx, ch = cv["ty"].astype(np.int64), cv["tx"].astype(np.float64), cv["chain"]
    xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
    W = max(1.0, xr - xl)
    out = defaultdict(set)
    if len(ty) < 3:
        return out
    dy = np.diff(ty); dx = np.diff(tx)
    J = np.flatnonzero((dy <= 8) & (np.abs(dx) > a.jump * W))
    for j in J:
        xb, xa = tx[j], tx[j + 1]
        t = int(np.clip(np.searchsorted(Pp, ty[j + 1]), 0, len(Pp) - 1))
        for k in range(K):
            for k2 in (k - 1, k + 1):
                if not 0 <= k2 < K:
                    continue
                v = LI_STAND.val(ch[k], xb)
                if abs(LI_STAND.pix(ch[k2], v) - xa) <= a.tol * W:
                    out[t].add((k, k2))
    return out


from numba import njit


@njit(cache=True)
def _vit(lp, evm, c_evt, c_free):
    K, n = lp.shape
    cost = -lp[:, 0].copy(); bp = np.zeros((K, n), np.int16)
    new = np.empty(K)
    for t in range(1, n):
        for k2 in range(K):
            best = cost[k2]; bk = k2
            for k in range(K):
                if k == k2:
                    continue
                c = cost[k] + (c_evt if evm[t, k, k2] else c_free)
                if c < best:
                    best = c; bk = k
            new[k2] = best - lp[k2, t]; bp[k2, t] = bk
        for k2 in range(K):
            cost[k2] = new[k2]
    s = 0
    for k in range(K):
        if cost[k] < cost[s]:
            s = k
    path = np.empty(n, np.int16)
    for t in range(n - 1, -1, -1):
        path[t] = s; s = bp[s, t]
    return path


def viterbi(lp, ev, c_evt, c_free):
    """lp — (K, n) логарифмы вероятностей; ev — {t: {(k, k2)}}"""
    K, n = lp.shape
    evm = np.zeros((n, K, K), np.bool_)
    for t, st in ev.items():
        for k, k2 in st:
            evm[t, k, k2] = True
    return _vit(lp, evm, float(c_evt), float(c_free))


def to_lv(Pp, p):
    ch_ = np.flatnonzero(np.diff(np.concatenate([[-1], p])) != 0)
    starts = Pp[ch_] - half; starts[0] = Pp[0] - half
    ends = np.concatenate([Pp[ch_[1:]] - half - 1, [Pp[-1] - half + a.stride - 1]])
    R = LI_STAND.min_run_runs(starts, p[ch_], ends, a.min_run)
    return {r[0]: r[2] for r in R}


VAR = ["выдача", "argmax"] + [f"прыжки {g}" for g in a.grid]
C = Counter(); PER = defaultdict(lambda: Counter()); NEV = Counter()
for i, cv in enumerate(CUR):
    ci = i
    h_out = LI_STAND.honest(cv, None)
    res = {"выдача": h_out}
    if len(cv["chain"]) >= 2 and ci in PR["pred"] and ci in prob:
        Pp, pa = PR["pred"][ci]; Pp = np.asarray(Pp, np.int64)
        pr = prob[ci].astype(np.float64)
        K = min(len(cv["chain"]), KMAX, pr.shape[0])
        lp = np.log(np.clip(pr[:K], 1e-6, 1.0))
        res["argmax"] = LI_STAND.honest(cv, to_lv(Pp, np.asarray(pa, np.int16)))
        ev = events(cv, Pp, K)
        NEV[cv["set"]] += len(ev)
        for g in a.grid:
            ce, cf = (float(v) for v in g.split(":"))
            res[f"прыжки {g}"] = LI_STAND.honest(cv, to_lv(Pp, viterbi(lp, ev, ce, cf)))
    else:
        for v in VAR[1:]:
            res[v] = h_out
    for v in VAR:
        C[(cv["set"], v)] += res[v]; PER[(cv["set"], cv["sheet"])][v] += res[v]
rng = np.random.default_rng(0)
for sn in ("поле", "сорт A"):
    print(f"\n★ {sn}: именных честных в значениях (позиций с прыжками {NEV[sn]})")
    for v in VAR:
        d = np.array([c[v] - c["argmax"] for (s, _), c in PER.items() if s == sn]); nz = d[d != 0]
        p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
        print(f"   {v:16s} {C[(sn, v)]:5d}   против argmax {C[(sn, v)] - C[(sn, 'argmax')]:+4d} (листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
