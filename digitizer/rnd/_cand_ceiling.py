r"""_cand_ceiling.py — ПОТОЛОК ВЫБОРА ЛИНИИ ДЛЯ СЛОТА В МЕРЕ ПРИЁМКИ (04.10, к §6.279).

§6.278: треть кривых теряется потому, что в слот попадает другая настоящая линия трека. Здесь — сколько кривых эталона можно было бы
сделать честными, выбрав для слота ДРУГУЮ трассу из уже посчитанных кандидатов кэша (прод-путь `traces` и декодер `alt`).
Честность — на плоскости в обе стороны по пикселям (медианы ≤ 3 px, покрытие ≥ 0.9), без уровней масштаба (у кандидатов их нет):
  «выдача честна»            — трасса под этим именем в повторе честна;
  «есть честный кандидат»     — выдача нечестна, но хотя бы одна трасса кэша честна для этой кривой;
     из них «свободный» — этот кандидат не стоит ни в одном слоте выдачи; «занят» — стоит в другом слоте;
  «нет честного кандидата»    — ни одна трасса кэша не честна.

  _cand_ceiling.py --every 3 --workers 4
"""
import sys, argparse, pickle, hashlib, time
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def _tree(tr):
    from scipy.spatial import cKDTree
    ys = np.array(sorted(tr), float); xs = np.array([tr[int(y)] for y in ys], float)
    P = [np.stack([ys, xs], 1)]
    if len(ys) > 1:
        dy = np.diff(ys); dx = np.diff(xs)
        for i in np.flatnonzero((dy <= 30) & (np.abs(dx) > 1) & (np.abs(dx) <= 3000)):
            n = int(np.ceil(abs(dx[i]))); t = np.arange(1, n) / n
            P.append(np.stack([ys[i] + dy[i] * t, xs[i] + dx[i] * t], 1))
    return cKDTree(np.concatenate(P))


def honest(tr, gt, tt=None, tg=None):
    com = [y for y in gt if y in tr]
    if len(com) < 30 or len(com) / max(1, len(gt)) < 0.9:
        return False
    tt = tt or _tree(tr); tg = tg or _tree(gt)
    d1, _ = tt.query(np.array([[y, gt[y]] for y in com], float))
    if np.median(d1) > 3:
        return False
    d2, _ = tg.query(np.array([[y, tr[y]] for y in com], float))
    return bool(np.median(d2) <= 3)


def dense_tr(tr, max_gap=200):
    ys = sorted(tr); out = {}
    for y0, y1 in zip(ys, ys[1:]):
        out[y0] = float(tr[y0])
        if 0 < y1 - y0 <= max_gap:
            for yy in range(y0 + 1, y1):
                out[yy] = tr[y0] + (tr[y1] - tr[y0]) * (yy - y0) / (y1 - y0)
    if ys:
        out[ys[-1]] = float(tr[ys[-1]])
    return out


def one(job):
    sh, tc, got, src = job
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from auto import meta as M
    c = pickle.load(open(tc, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    cands = [("prod", i, dense_tr(un(t))) for i, (L, t) in enumerate(c["traces"] or [])]
    cands += [("dec", i, dense_tr(un(t))) for i, (L, t) in enumerate(c["alt"] or [])]
    cands = [x for x in cands if len(x[2]) >= 50]
    G = {cc["name"]: dense(cc) for cc in extract(src)["curves"] if M.mnem_root(cc["name"]) != "DA"
         and sum(1 for x in cc["xs"] if x != NULL) >= 50}
    W = {cc["name"]: dense(cc) for cc in extract(got)["curves"] if M.mnem_root(cc["name"]) != "DA"}
    TC = [_tree(x[2]) for x in cands]
    # какая кандидатная трасса стоит в каком слоте выдачи: совпадение x на ≥ 90% общих строк в 1 px
    used = {}
    for wn, w in W.items():
        if len(w) < 50:
            continue
        for k, (_, _, ct) in enumerate(cands):
            com = [y for y in w if y in ct]
            if len(com) >= 0.5 * len(w) and np.mean([abs(w[y] - ct[y]) <= 1.0 for y in com[::5]]) >= 0.9:
                used.setdefault(k, wn)
    out = []
    for g, gt in G.items():
        tg = _tree(gt)
        w = W.get(g)
        h_out = bool(w) and len(w) >= 30 and honest(w, gt, None, tg)
        hc = [k for k, (_, _, ct) in enumerate(cands) if honest(ct, gt, TC[k], tg)]
        out.append(dict(sheet=sh, name=g, out=h_out, cands=len(hc), free=sum(1 for k in hc if k not in used),
                        busy=sum(1 for k in hc if k in used and used[k] != g)))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
    ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_lvl/NS")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/cand_ceiling.pkl")
    a = ap.parse_args()
    import multiprocessing as mp
    TS = Path(a.ts)
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
    jobs, setof = [], {}
    for sn, S in SETS.items():
        for sh in S[::a.every]:
            q = SRC.get(sh)
            if not q:
                continue
            key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
            tc = Path(a.cache) / (key + ".pkl"); pd = Path(a.dir) / key
            got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
            if tc.exists() and got:
                jobs.append((sh, str(tc), str(got), str(q))); setof[sh] = sn
    print(f"листов {len(jobs)}")
    R = []; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs)):
            R.extend(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    for sn in SETS:
        rs = [r for r in R if setof.get(r["sheet"]) == sn]
        C = Counter()
        for r in rs:
            if r["out"]:
                C["выдача честна"] += 1
            elif r["cands"]:
                C["есть честный кандидат"] += 1
                C["  свободный"] += r["free"] > 0
                C["  только занятый другим слотом"] += r["free"] == 0
            else:
                C["нет честного кандидата"] += 1
        n = len(rs)
        print(f"★ {sn} (каждый {a.every}-й лист, кривых {n}): " + "; ".join(f"{k} {v} ({v / max(1, n):.0%})" for k, v in C.items()))


if __name__ == "__main__":
    main()
