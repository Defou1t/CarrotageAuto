r"""_despike_bench.py — ВЫБРОСЫ ТРАССЫ НА ЛИНИИ СЕТКИ: СКОЛЬКО КРИВЫХ СТАНУТ ЧЕСТНЫМИ, ЕСЛИ ИХ ВЫЧИСТИТЬ (04.10, к §6.284).

Разбор «почти честных» (`visual_2026-10-03/C_line_choice/near`): трасса идёт по своей туши, но на отдельных строках уходит далеко
по горизонтали и возвращается — ран кривой на этой строке слился с тёмной линией сетки, и точка взята в его середине. Медиана
расстояний на плоскости поднимается выше 3 px.

Здесь — для каждого слота с эталоном (поле + сорт A) кандидат, стоящий в слоте выдачи NS (`_line_choice_scan.py`, флаг here),
до и после вычистки выбросов: отрезок ≤ L строк, где трасса отходит от хорды между соседями дальше D px, а соседи по краям отрезка
близки друг к другу (≤ E px). Строки выброса выбрасываются (мостик закрывает). Мера — на плоскости в обе стороны по пикселям.
Режим `--ink` дополнительно требует, чтобы на строке выброса ран туши в точке трассы был длинным (≥ W px), т. е. это линия
сетки, а не острый пик кривой.

  _despike_bench.py --grid 3:25:8 5:25:8 3:15:6 --workers 4
  _despike_bench.py --grid med:3 med:5 mean:3 mean:5     # сглаживание (медиана / среднее по окну строк)
"""
import sys, argparse, pickle, hashlib, time, os
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def despike(tr, L, D, E):
    """tr — {y: x} (сырые точки трассы) → копия без выбросов"""
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys], float)
    n = len(ys)
    drop = np.zeros(n, bool)
    i = 1
    while i < n - 1:
        # отрезок [i, j]: подряд идущие строки (без разрыва), длиной ≤ L
        best = None
        for j in range(i, min(n - 1, i + L)):
            if ys[j + 1] - ys[i - 1] != j - i + 2:
                break
            a, b = xs[i - 1], xs[j + 1]
            if abs(a - b) > E:
                continue
            t = (ys[i:j + 1] - ys[i - 1]) / (ys[j + 1] - ys[i - 1])
            chord = a + (b - a) * t
            if np.all(np.abs(xs[i:j + 1] - chord) > D):
                best = j
                break
        if best is not None:
            drop[i:best + 1] = True
            i = best + 2
        else:
            i += 1
    return {int(y): float(x) for y, x, d_ in zip(ys, xs, drop) if not d_}, int(drop.sum())


def smooth(tr, kind, w):
    """сглаживание по подряд идущим строкам: kind = med / mean, окно w строк (нечётное)"""
    from scipy.ndimage import median_filter, uniform_filter1d
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys], float)
    out = xs.copy()
    cut = np.flatnonzero(np.diff(ys) != 1) + 1
    for a_, b_ in zip(np.r_[0, cut], np.r_[cut, len(ys)]):
        seg = xs[a_:b_]
        if len(seg) >= w:
            out[a_:b_] = median_filter(seg, size=w, mode="nearest") if kind == "med" else uniform_filter1d(seg, size=w, mode="nearest")
    return {int(y): float(x) for y, x in zip(ys, out)}, int(np.sum(np.abs(out - xs) > 0.5))


def one(job):
    rec_path, tc_path, src, grid = job
    sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    from auto import meta as M
    rec = pickle.load(open(rec_path, "rb"))
    c = pickle.load(open(tc_path, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    raw = [un(t) for L, t in (c["traces"] or [])] + [un(t) for L, t in (c["alt"] or [])]
    raw = [r for r in raw if len(dense_tr(r)) >= 50]
    mt = extract(src)
    fc = {cc["name"]: cc for cc in mt["curves"] if M.mnem_root(cc["name"]) != "DA"}
    out = []
    for s in rec["slots"]:
        if not s["has_truth"]:
            continue
        here = [r["k"] for r in s["cands"] if r["here"]]
        if not here:
            continue
        k = here[0]
        gt = dense(fc[s["name"]]); tg = _tree(gt)
        base = dense_tr(raw[k]); h0 = honest_d(base, _tree(base), gt, tg)
        res = dict(sheet=rec["sheet"], slot=s["name"], root=s["root"], h0=h0[0], m0=h0[2], modes={})
        for g in grid:
            if isinstance(g[0], str):
                ds, nd = smooth(raw[k], g[0], int(g[1]))
            else:
                L, D, E = g
                ds, nd = despike(raw[k], L, D, E)
            if nd == 0:
                res["modes"][g] = (h0[0], h0[2], 0)
                continue
            dd = dense_tr(ds); h1 = honest_d(dd, _tree(dd), gt, tg)
            res["modes"][g] = (h1[0], h1[2], nd)
        out.append(res)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
    ap.add_argument("--scan", default=r"F:/nds/output/taskS/line_choice")
    ap.add_argument("--grid", nargs="+", default=["3:25:8", "5:25:8", "3:15:6"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/despike_bench.pkl")
    a = ap.parse_args()
    import multiprocessing as mp
    grid = []
    for g in a.grid:
        v = g.split(":")
        grid.append((v[0], int(v[1])) if v[0] in ("med", "mean") else (int(v[0]), float(v[1]), float(v[2])))
    TS = Path(a.ts)
    lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for f in sorted(Path(a.scan).glob("*.pkl")):
        sh = pickle.load(open(f, "rb"))["sheet"]
        q = SRC[sh]
        key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        jobs.append((str(f), str(Path(a.cache) / (key + ".pkl")), str(q), grid))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"листов {len(jobs)}")
    R = []; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            R += r
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    rng = np.random.default_rng(0)
    for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
        rs = [r for r in R if r["sheet"] in S]
        print(f"\n★ {sn}: слотов с эталоном и кандидатом выдачи {len(rs)}, честных до {sum(r['h0'] for r in rs)}")
        for g in grid:
            up = sum(1 for r in rs if r["modes"][g][0] and not r["h0"]); dn = sum(1 for r in rs if r["h0"] and not r["modes"][g][0])
            ch = sum(1 for r in rs if r["modes"][g][2] > 0)
            D = {}
            for r in rs:
                D[r["sheet"]] = D.get(r["sheet"], 0) + int(r["modes"][g][0]) - int(r["h0"])
            d = np.array(list(D.values())); nz = d[d != 0]
            p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
            nm = f"{g[0]} {g[1]}" if isinstance(g[0], str) else f"L {g[0]}, D {g[1]:.0f}, E {g[2]:.0f}"
            print(f"   {nm}: трасс с вычисткой {ch}; стала честной {up}, перестала {dn}; Δ {up - dn:+d} (p = {p:.4f})")


if __name__ == "__main__":
    main()
