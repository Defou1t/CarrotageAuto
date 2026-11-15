r"""_level_oracle2.py — §6.379 (б): оракул «наш уровень верен на строках, где у эталона уровень ≥ 2» (15.11).

Стенд §6.354 (`_level_decode.py`): вероятности v7e вне фолда, argmax + min_run 25 = база; мера — именные честные в значениях стенда
(`_level_ink.honest`). Оракулы по позициям (каждая 8-я строка): O2 — уровень эталона там, где он ≥ 2, иначе база; O1 — уровень эталона
там, где он ≤ 1, иначе база; полный — уровни эталона везде (как §6.354).

  _level_oracle2.py [--workers 3]
"""
import sys, argparse, pickle, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from collections import Counter
import numpy as np
from _level_decode import load, runs_to_lo, HALF, KMAX


def expand(lo, Pp):
    st = np.array(sorted(lo)); lv = np.array([lo[s] for s in st])
    i = np.clip(np.searchsorted(st, Pp - HALF, side="right") - 1, 0, len(st) - 1)
    return lv[i]


def work(args):
    ci, cv, pr, Pp = args
    import _level_ink as L
    pr = pr.astype(np.float32); Pp = Pp.astype(np.int64)
    K = min(len(cv["chain"]), KMAX); pr = pr[:K]
    lo = runs_to_lo(L, Pp, pr.argmax(0).astype(np.int16))
    out = {"base": L.honest(cv, lo)}
    lt = cv["lt"] or []
    out["oracle"] = L.honest(cv, {int(s): int(l) for s, e, l in lt}) if lt else L.honest(cv, {})
    tl = np.full(len(Pp), -1, np.int64)
    for s, e, l in lt:
        tl[(Pp >= s) & (Pp <= e)] = int(l)
    b = expand(lo, Pp)
    o2 = np.where(tl >= 2, tl, b); o1 = np.where((tl >= 0) & (tl <= 1), tl, b)
    out["O2"] = L.honest(cv, runs_to_lo(L, Pp, o2.astype(np.int16), 1))
    out["O1"] = L.honest(cv, runs_to_lo(L, Pp, o1.astype(np.int16), 1))
    out["tmax"] = int(tl.max()) if len(tl) else -1
    return ci, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    L, CUR, PR = load()
    ids = sorted(ci for ci in PR["prob"] if ci < len(CUR) and len(CUR[ci]["chain"]) >= 2)
    KEEP = ("set", "sheet", "chain", "gy", "gx", "lt", "ty", "tx", "lw")
    jobs = [(ci, {k: CUR[ci][k] for k in KEEP}, PR["prob"][ci], PR["pred"][ci][0]) for ci in ids]
    sets = {ci: CUR[ci]["set"] for ci in ids}
    CUR = PR = None
    from multiprocessing import Pool
    t0 = time.time(); RES = {}
    with Pool(a.workers) as pool:
        for k, (ci, o) in enumerate(pool.imap_unordered(work, jobs, chunksize=8), 1):
            RES[ci] = o
    print(f"★ кривых с цепочкой {len(RES)}, {time.time() - t0:.0f} с; " + str(Counter(sets.values())))
    for nm, S in (("поле+вне", [c for c in RES if sets[c] in ("поле", "вне")]), ("сорт A", [c for c in RES if sets[c] == "сорт A"])):
        b = sum(RES[c]["base"] for c in S)
        print(f"## {nm}: кривых {len(S)}; база {b}; O2 (эталон на уровнях ≥ 2) {sum(RES[c]['O2'] for c in S)} (Δ {sum(RES[c]['O2'] for c in S) - b:+d});"
              f" O1 (эталон на 0–1) {sum(RES[c]['O1'] for c in S)} (Δ {sum(RES[c]['O1'] for c in S) - b:+d}); полный оракул {sum(RES[c]['oracle'] for c in S)}"
              f" (Δ {sum(RES[c]['oracle'] for c in S) - b:+d})")


if __name__ == "__main__":
    main()
