r"""_level_reset.py — УРОВЕНЬ НЕ ПЕРЕНОСИТСЯ ЧЕРЕЗ ПРЫЖОК НА ДРУГУЮ ТУШЬ (10.11, §6.357; критерий — в ROADMAP до прогона).

Разбор 40 серий неверного уровня (§6.354): трасса на туши эксперта, но уровень перенесён с чужой туши через прыжок или сменён на
сплошной линии без события. Здесь — пост-правила к argmax вероятностей v7e вне фолда (те же данные и мера, что `_level_decode.py`):
отрезки трассы между прыжками (размах x на шаге позиций > J·ширины шкалы 1×); прыжок «без события» — на шаге ни одна точка трассы не
ближе e·ширины к краю полосы уровня 0 (не обруб и не перенос).
  V2 τ — отрезок, начатый прыжком без события: уровень 0, если средняя P(0) на отрезке > τ;
  V3   — внутри каждого отрезка один уровень (argmax средних вероятностей);
  V4 τ — V3, затем V2;   V5 — V3 только у отрезков, начатых прыжком без события.

  _level_reset.py [--workers 2] [--limit N]
"""
import sys, argparse, pickle, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import defaultdict, Counter
from math import comb
import numpy as np

TS = Path(r"F:/nds/output/taskS")
KMAX = 4
E = 0.03


def segs(cv, Pp, Jth):
    """границы отрезков по позициям: [(i0, i1, начат_прыжком_без_события)]"""
    ty, tx = cv["ty"].astype(np.int64), cv["tx"].astype(np.float64)
    s0 = cv["chain"][0]; W0 = abs(s0["x_right"] - s0["x_left"]) or 1.0
    xl, xr = min(s0["x_left"], s0["x_right"]), max(s0["x_left"], s0["x_right"])
    a = np.searchsorted(ty, Pp[:-1]); b = np.searchsorted(ty, Pp[1:], side="right")
    out, start, quiet = [], 0, False
    for i in range(len(Pp) - 1):
        if b[i] - a[i] < 2:
            continue
        seg = tx[a[i]:b[i]]
        if (seg.max() - seg.min()) / W0 > Jth:
            edge = (np.min(np.minimum(np.abs(seg - xl), np.abs(seg - xr))) / W0) <= E
            out.append((start, i, quiet)); start = i + 1; quiet = not edge
    out.append((start, len(Pp) - 1, quiet))
    return [(i0, i1, q) for i0, i1, q in out if i1 >= i0]


def work(args):
    ci, cv, pr, Pp, cfgs = args
    import _level_ink as L, _level_decode as D
    pr = pr.astype(np.float32); Pp = Pp.astype(np.int64)
    K = min(len(cv["chain"]), KMAX); pr = pr[:K]
    base = pr.argmax(0).astype(np.int16)
    out = {"base": L.honest(cv, D.runs_to_lo(L, Pp, base))}
    cache = {}
    for c in cfgs:
        v, Jth, tau = c
        if Jth not in cache:
            cache[Jth] = segs(cv, Pp, Jth)
        S = cache[Jth]; p = base.copy()
        if v == "V3":                                 # тот же код, что прод (`auto.level_ink.seg_vote`)
            from auto import level_ink as LI
            p = LI.seg_vote(pr, Pp, cv["ty"], cv["tx"], cv["chain"], Jth)
        if v in ("V4", "V5"):
            for i0, i1, q in S:
                if v == "V5" and not q:
                    continue
                p[i0:i1 + 1] = int(pr[:, i0:i1 + 1].mean(1).argmax())
        if v in ("V2", "V4"):
            for i0, i1, q in S:
                if q and pr[0, i0:i1 + 1].mean() > tau:
                    p[i0:i1 + 1] = 0
        out[c] = L.honest(cv, D.runs_to_lo(L, Pp, p))
    return ci, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfgs = [("V2", J, t) for J in (0.1, 0.2, 0.3) for t in (0.05, 0.15, 0.3)] + [("V3", J, 0.0) for J in (0.1, 0.2, 0.3)] + \
           [("V4", J, t) for J in (0.1, 0.2, 0.3) for t in (0.05, 0.15, 0.3)] + [("V5", J, 0.0) for J in (0.1, 0.2, 0.3)]
    import _level_decode as D
    L, CUR, PR = D.load()
    ids = sorted(ci for ci in PR["prob"] if ci < len(CUR) and len(CUR[ci]["chain"]) >= 2)
    if a.limit:
        ids = ids[::max(1, len(ids) // a.limit)][:a.limit]
    print(f"★ кривых с цепочкой {len(ids)}: " + str(Counter(CUR[ci]["set"] for ci in ids)) + f"; вариантов {len(cfgs)}")
    KEEP = ("set", "sheet", "chain", "gy", "gx", "lt", "ty", "tx", "lw")
    jobs = [(ci, {k: CUR[ci][k] for k in KEEP}, PR["prob"][ci], PR["pred"][ci][0], cfgs) for ci in ids]
    _ids = set(ids)
    CUR = [None if i not in _ids else {"set": c["set"], "sheet": c["sheet"]} for i, c in enumerate(CUR)]; PR = None
    RES = {}; t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for k, (ci, o) in enumerate(pool.imap_unordered(work, jobs, chunksize=8), 1):
            RES[ci] = o
            if k % 500 == 0:
                print(f"  {k}/{len(ids)} кривых, {time.time() - t0:.0f} с")
    out = TS / "_level_reset.pkl"; tmp = out.with_name(out.name + ".tmp")
    pickle.dump({"cfgs": cfgs, "res": RES}, open(tmp, "wb")); tmp.replace(out)
    sets = {"поле": [], "вне": [], "сорт A": []}
    for ci in RES:
        sets.setdefault(CUR[ci]["set"], []).append(ci)
    sets["поле+вне"] = sets["поле"] + sets["вне"]
    for nm in ("поле+вне", "сорт A"):
        S = sets[nm]; b = sum(RES[ci]["base"] for ci in S)
        print(f"\n## {nm}: кривых {len(S)}; argmax+min_run (v7e) {b}")
        rows = []
        for c in cfgs:
            per = defaultdict(int)
            for ci in S:
                per[CUR[ci]["sheet"]] += int(RES[ci][c]) - int(RES[ci]["base"])
            up = sum(1 for v in per.values() if v > 0); dn = sum(1 for v in per.values() if v < 0); n = up + dn
            p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(up, dn) + 1)) / 2 ** n) if n else 1.0
            rows.append((sum(RES[ci][c] for ci in S) - b, c, up, dn, p))
        for d, c, up, dn, p in sorted(rows, key=lambda t: -t[0]):
            print(f"  {c[0]} J {c[1]:.1f} τ {c[2]:.2f}: Δ {d:+d} (листов ↑{up}/↓{dn}, p = {p:.4f})")


if __name__ == "__main__":
    main()
