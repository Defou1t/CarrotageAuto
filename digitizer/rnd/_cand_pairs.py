r"""_cand_pairs.py — ПОПАРНАЯ БЛИЗОСТЬ КАНДИДАТОВ ЛИСТА: H[i, j] = ДОЛЯ СТРОК i В ≤ 3 PX ОТ j (21.10, §6.333).

Разбор промахов v11 (§6.333): 196 слотов с честной свободной трассой держат трассу, честную для СОСЕДНЕГО слота, который и сам честен —
два слота держат две копии одной кривой (кластеры дублей назначения ловят только «те же концы ±5 строк и квантили ±2 px»). Здесь —
матрица H по каждому листу скана (`line_choice_i` / `line_choice_out_i`, порядок пула скана), строки с шагом 4. Пишет
`--out/<имя файла скана>` = float16 [n, n].

  _cand_pairs.py --workers 4
"""
import sys, argparse, pickle, hashlib, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np

TS = Path("F:/nds/output/taskS")
_SRC = None
OUT = None


def pool_of(key):
    """как `_cand_overlap.pool_of`: prod traces, alt, затем доп. пути (поле: pk3 alt + inert extra; вне: out extra + out_inert extra)"""
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    fld = TS / "tcache" / (key + ".pkl")
    if fld.exists() and (TS / "tcache_pk3" / (key + ".pkl")).exists():
        c = pickle.load(open(fld, "rb")); ex = []
        for d, f in ((TS / "tcache_pk3", "alt"), (TS / "tcache_inert", "extra")):
            if (d / (key + ".pkl")).exists():
                ex += pickle.load(open(d / (key + ".pkl"), "rb")).get(f) or []
    else:
        c = pickle.load(open(TS / "tcache_out" / (key + ".pkl"), "rb"))
        ex = list(c.get("extra") or [])
        if (TS / "tcache_out_inert" / (key + ".pkl")).exists():
            ex += pickle.load(open(TS / "tcache_out_inert" / (key + ".pkl"), "rb")).get("extra") or []
    return c, ex, un


def pairs_H(dense_list, step=4, tol=3.0):
    """H[i, j] — доля строк i (шаг step) в ≤ tol px от j (строки, где j нет, — промах)"""
    n = len(dense_list)
    if not n:
        return np.zeros((0, 0), np.float16)
    ymax = max((max(d) for d in dense_list if d), default=0)
    T = ymax // step + 1
    MX = np.full((n, T), np.nan, np.float32)
    for k, d in enumerate(dense_list):
        if d:
            yy = np.array([y for y in sorted(d) if y % step == 0])
            if len(yy):
                MX[k, yy // step] = [d[y] for y in yy]
    has = np.isfinite(MX)
    H = np.zeros((n, n), np.float32)
    for i in range(n):
        hi = has[i]
        if not hi.any():
            continue
        dx = np.abs(MX[:, hi] - MX[i, hi][None, :])
        H[i] = (dx <= tol).sum(1) / hi.sum()
    np.fill_diagonal(H, 1.0)
    return H.astype(np.float16)


def one(f):
    global _SRC
    from _cand_ceiling import dense_tr
    out = Path(OUT) / Path(f).name
    if out.exists():
        return 0
    if _SRC is None:
        _SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
    rec = pickle.load(open(f, "rb")); q = _SRC[rec["sheet"]]
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    try:
        c, ex, un = pool_of(key)
    except Exception:
        return -1
    D = []
    for cs in rec["cands"]:
        t = c["traces"][cs["idx"]][1] if cs["src"] == "prod" else (c["alt"][cs["idx"]][1] if cs["idx"] < 1000 else ex[cs["idx"] - 1000][1])
        D.append(dense_tr(un(t)))
    pickle.dump(pairs_H(D), open(out, "wb"))
    return 1


def _init(o):
    global OUT
    OUT = o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="F:/nds/output/taskS/cand_pairs")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    fs = sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl"))) + sorted(glob.glob(str(TS / "line_choice_out_i" / "*.pkl")))
    done = 0
    with mp.Pool(a.workers, initializer=_init, initargs=(a.out,)) as pool:
        for k, r in enumerate(pool.imap_unordered(one, fs, chunksize=4)):
            done += r == 1
            if (k + 1) % 300 == 0:
                print(f"  {k + 1}/{len(fs)}")
    print(f"★ записано {done} → {a.out}")


if __name__ == "__main__":
    main()
