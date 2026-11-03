r"""_level_decode.py — ДЕКОДЕР УРОВНЕЙ МАСШТАБА ПО ВЕРОЯТНОСТЯМ МОДЕЛИ (03.11, §6.354; критерии — в ROADMAP до прогона).

Разбор `_switch_audit.py` / `_switch_cases.py`: уровни эталона при той же трассе дали бы поле+вне +430 именных (сорт A +64); серии
чужого уровня в основном начинаются там, где трасса НЕ прыгает (наша смена без скачка — 30% строк, смена эталона, которую наша
трасса не повторила, — 36%), смены эталона почти всегда со скачком своей трассы (медиана 0.31 ширины шкалы). Прод:
argmax вероятностей модели уровня по позициям (каждая 8-я строка) → `enforce_min_run` 25 строк.
Здесь — Витерби по тем же вероятностям (вне фолда, `level_ink_pred_v7e.pkl` на стенде `_level_ink.py`): цена смены уровня
ε + λ·(1 − min(1, скачок / J0)), скачок — размах x трассы выдачи между соседними позициями в долях ширины шкалы 1×; по желанию
μ·|Δ lg значения| на смене (непрерывность значения при перевыносе). Мера — именные честные в значениях стенда (`_level_ink.honest`).

  _level_decode.py [--grid] [--workers 3]
"""
import sys, argparse, pickle, time, itertools
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import defaultdict, Counter
from math import comb
import numpy as np

TS = Path(r"F:/nds/output/taskS")
STRIDE, HALF, KMAX = 8, 4, 4


def load():
    import _level_ink as L
    CUR = pickle.load(open(TS / "level_bench_nslx.pkl", "rb"))
    CUR = CUR + [dict(cv, set="вне") for cv in pickle.load(open(TS / "level_bench_lxb_out.pkl", "rb"))]
    PR = pickle.load(open(TS / "level_ink_pred_v7e.pkl", "rb"))
    return L, CUR, PR


def runs_to_lo(L, Pp, p, min_run=25):
    ch_ = np.flatnonzero(np.diff(np.concatenate([[-1], p])) != 0)
    starts = Pp[ch_] - HALF; starts[0] = Pp[0] - HALF
    ends = np.concatenate([Pp[ch_[1:]] - HALF - 1, [Pp[-1] - HALF + STRIDE - 1]])
    R = L.min_run_runs(starts, p[ch_], ends, min_run) if min_run > 1 else \
        [[int(s), int(e), int(l)] for s, e, l in zip(starts, ends, p[ch_])]
    return {r[0]: r[2] for r in R}


def jumps(cv, Pp):
    """размах x трассы выдачи на отрезке [P_i, P_{i+1}] в долях ширины шкалы 1× (нет трассы — nan)"""
    ty, tx = cv["ty"].astype(np.int64), cv["tx"].astype(np.float64)
    s0 = cv["chain"][0]; W0 = abs(s0["x_right"] - s0["x_left"]) or 1.0
    a = np.searchsorted(ty, Pp[:-1]); b = np.searchsorted(ty, Pp[1:], side="right")
    J = np.full(len(Pp) - 1, np.nan)
    for i in range(len(Pp) - 1):
        if b[i] - a[i] >= 2:
            seg = tx[a[i]:b[i]]; J[i] = (seg.max() - seg.min()) / W0
    return J


def vals(cv, Pp, K):
    """lg значения трассы на позициях при каждом уровне k < K (сдвиговые цепочки — значение / ширину диапазона)"""
    import _level_ink as L
    ch = cv["chain"]; ty, tx = cv["ty"].astype(np.int64), cv["tx"].astype(np.float64)
    x = np.interp(Pp, ty, tx)
    V = np.zeros((len(Pp), K))
    w0 = abs(ch[0]["v_right"] - ch[0]["v_left"]) or 1.0
    logv = all(min(s["v_left"], s["v_right"]) >= 0 for s in ch[:K]) and not (ch[1]["v_left"] == ch[0]["v_right"])
    for k in range(K):
        v = L.val(ch[k], x)
        V[:, k] = np.log10(np.clip(v, 1e-3, None)) if logv else v / w0
    return V


def viterbi(pr, J, V, lam, J0, eps, mu):
    """pr [K, n] вероятности; J [n − 1] скачки; V [n, K] значения или None → уровни [n]"""
    K, n = pr.shape
    E = -np.log(np.clip(pr, 1e-6, 1.0)).T                # [n, K]
    D = E[0].copy(); BP = np.zeros((n, K), np.int8)
    off = ~np.eye(K, dtype=bool)
    for i in range(1, n):
        j = J[i - 1]
        sc = eps + lam * (1.0 - min(1.0, (j if j == j else 0.0) / J0))
        T = np.where(off, sc, 0.0)
        if mu and V is not None:
            T = T + np.where(off, mu * np.abs(V[i][None, :] - V[i - 1][:, None]), 0.0)
        M = D[:, None] + T                                # [от, к]
        BP[i] = M.argmin(0); D = M.min(0) + E[i]
    p = np.zeros(n, np.int16); p[-1] = int(D.argmin())
    for i in range(n - 1, 0, -1):
        p[i - 1] = BP[i, p[i]]
    return p


def work(args):
    ci, cv, pr, Pp, cfgs = args
    import _level_ink as L
    pr = pr.astype(np.float32); Pp = Pp.astype(np.int64)
    K = min(len(cv["chain"]), KMAX); pr = pr[:K]
    out = {"base": L.honest(cv, runs_to_lo(L, Pp, pr.argmax(0).astype(np.int16))), "prod": L.honest(cv, None)}
    lt = {int(s): int(l) for s, e, l in (cv["lt"] or [])}
    out["oracle"] = L.honest(cv, lt or {}) if lt else L.honest(cv, {})
    if len(Pp) < 3:
        for c in cfgs:
            out[c] = out["base"]
        return ci, out
    J = jumps(cv, Pp)
    V = vals(cv, Pp, K) if any(c[3] for c in cfgs) else None
    for c in cfgs:
        lam, J0, eps, mu, mr = c
        p = viterbi(pr, J, V, lam, J0, eps, mu)
        out[c] = L.honest(cv, runs_to_lo(L, Pp, p, mr))
    return ci, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--grid", default="small")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.grid == "small":
        cfgs = [(lam, J0, eps, 0.0, 1) for lam in (2.0, 5.0, 10.0, 20.0) for J0 in (0.15, 0.3) for eps in (0.5, 2.0)]
        cfgs += [(0.0, 0.3, eps, 0.0, 1) for eps in (1.0, 3.0, 6.0)]                  # без учёта скачка — контроль
        cfgs += [(lam, 0.3, 1.0, mu, 1) for lam in (5.0, 10.0) for mu in (1.0, 3.0)]
    else:
        cfgs = [tuple(float(x) if i < 4 else int(x) for i, x in enumerate(c.split(":"))) for c in a.grid.split(",")]
    L, CUR, PR = load()
    ids = sorted(ci for ci in PR["prob"] if ci < len(CUR) and len(CUR[ci]["chain"]) >= 2)
    if a.limit:
        ids = ids[::max(1, len(ids) // a.limit)][:a.limit]
    print(f"★ кривых с цепочкой и вероятностями {len(ids)}: " + str(Counter(CUR[ci]["set"] for ci in ids)) + f"; вариантов {len(cfgs)}")
    par = sum(1 for ci in ids if runs_to_lo(L, PR["pred"][ci][0].astype(np.int64), PR["prob"][ci].astype(np.float32)[
        :min(len(CUR[ci]["chain"]), KMAX)].argmax(0).astype(np.int16)) == PR["levels"][ci])
    print(f"  сверка argmax+min_run с уровнями дампа v7e: {par} из {len(ids)}")
    t0 = time.time(); RES = {}
    from multiprocessing import Pool
    KEEP = ("set", "sheet", "chain", "gy", "gx", "lt", "ty", "tx", "lw")
    jobs = [(ci, {k: CUR[ci][k] for k in KEEP}, PR["prob"][ci], PR["pred"][ci][0], cfgs) for ci in ids]
    _ids = set(ids)
    CUR = [None if i not in _ids else {"set": c["set"], "sheet": c["sheet"]} for i, c in enumerate(CUR)]; PR = None
    with Pool(a.workers) as pool:
        for k, (ci, o) in enumerate(pool.imap_unordered(work, jobs, chunksize=8), 1):
            RES[ci] = o
            if k % 500 == 0:
                print(f"  {k}/{len(ids)} кривых, {time.time() - t0:.0f} с")
    pickle.dump({"cfgs": cfgs, "res": RES}, open(TS / "_level_decode.pkl", "wb"))
    sets = {"поле": [], "вне": [], "сорт A": []}
    for ci in RES:
        sets.setdefault(CUR[ci]["set"], []).append(ci)
    sets["поле+вне"] = sets["поле"] + sets["вне"]
    for nm in ("поле+вне", "поле", "вне", "сорт A"):
        S = sets[nm]
        b = sum(RES[ci]["base"] for ci in S)
        print(f"\n## {nm}: кривых {len(S)}; прод-уровни стенда {sum(RES[ci]['prod'] for ci in S)}, argmax+min_run (v7e) {b}, "
              f"оракул (уровни эталона) {sum(RES[ci]['oracle'] for ci in S)}")
        rows = []
        for c in cfgs:
            per = defaultdict(int)
            for ci in S:
                per[CUR[ci]["sheet"]] += int(RES[ci][c]) - int(RES[ci]["base"])
            up = sum(1 for v in per.values() if v > 0); dn = sum(1 for v in per.values() if v < 0); n = up + dn
            p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(up, dn) + 1)) / 2 ** n) if n else 1.0
            rows.append((sum(RES[ci][c] for ci in S) - b, c, up, dn, p))
        for d, c, up, dn, p in sorted(rows, key=lambda t: -t[0])[:12]:
            print(f"  λ {c[0]:5.1f} J0 {c[1]:.2f} ε {c[2]:.1f} μ {c[3]:.1f} min_run {c[4]}: Δ {d:+d} (листов ↑{up}/↓{dn}, p = {p:.4f})")


if __name__ == "__main__":
    main()
