r"""_cand_overlap.py — ПЕРЕКРЫТИЕ КАНДИДАТОВ ДРУГ С ДРУГОМ: ПРИЗНАК «ТРАССЫ-СМЕСИ» (18.10, §6.316).

Разбор промахов v8 (`visual_2026-10-18/junk`): чаще всего вместо честной ставится трасса-смесь — на пересечении она переходит с
одной кривой на другую и честна ни для кого. Её строки частично покрывают ДВА разных других кандидата. По каждому кандидату
скана (`line_choice_i` / `line_choice_out_i`, тот же порядок пула, что в скане): доля строк в ≤ 3 px от каждого другого кандидата
(без почти-дублей ≥ 0.9) → лучший, второй, число частичных (0.2…0.9), покрытие объединением частичных.
Пишет `--out/<имя файла скана>` = массив [кандидатов, 4].

  _cand_overlap.py --workers 4
"""
import sys, argparse, pickle, hashlib, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
_SRC = None


def pool_of(sh, key):
    """кандидаты в порядке скана: prod traces, alt, затем доп. пути (поле: pk3 alt + inert extra; вне: out extra + out_inert extra)"""
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    fld = (TS / "tcache" / (key + ".pkl"))
    if fld.exists() and (TS / "tcache_pk3" / (key + ".pkl")).exists():
        c = pickle.load(open(fld, "rb"))
        ex = []
        for d, f in ((TS / "tcache_pk3", "alt"), (TS / "tcache_inert", "extra")):
            if (d / (key + ".pkl")).exists():
                ex += pickle.load(open(d / (key + ".pkl"), "rb")).get(f) or []
    else:
        c = pickle.load(open(TS / "tcache_out" / (key + ".pkl"), "rb"))
        ex = list(c.get("extra") or [])
        if (TS / "tcache_out_inert" / (key + ".pkl")).exists():
            ex += pickle.load(open(TS / "tcache_out_inert" / (key + ".pkl"), "rb")).get("extra") or []
    return c, ex, un


def one(f):
    global _SRC
    from _cand_ceiling import dense_tr
    out = Path(A.out) / Path(f).name
    if out.exists():
        return 0
    if _SRC is None:
        _SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    rec = pickle.load(open(f, "rb")); sh = rec["sheet"]; q = _SRC[sh]
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    try:
        c, ex, un = pool_of(sh, key)
    except Exception:
        return -1
    T = []
    for cs in rec["cands"]:
        if cs["src"] == "prod":
            t = c["traces"][cs["idx"]][1]
        elif cs["idx"] < 1000:
            t = c["alt"][cs["idx"]][1]
        else:
            t = ex[cs["idx"] - 1000][1]
        d = dense_tr(un(t))
        ys = np.array(sorted(d))[::4]; T.append((ys, np.array([d[y] for y in ys], float), d))
    n = len(T); F = np.zeros((n, 4), np.float32)
    for i in range(n):
        ys, xs, _ = T[i]
        fr = []; cov = np.zeros(len(ys), bool)
        for j in range(n):
            if j == i:
                continue
            dj = T[j][2]
            hit = np.array([abs(dj.get(int(y), 1e9) - x) <= 3 for y, x in zip(ys, xs)])
            fj = float(hit.mean()) if len(hit) else 0.0
            if fj >= 0.9:
                continue                                  # почти-дубль — не партнёр
            fr.append(fj)
            if fj >= 0.2:
                cov |= hit
        fr.sort(reverse=True)
        F[i] = [fr[0] if fr else 0.0, fr[1] if len(fr) > 1 else 0.0, sum(1 for v in fr if v >= 0.2), float(cov.mean()) if len(cov) else 0.0]
    pickle.dump(F, open(out, "wb"))
    return 1


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=r"F:/nds/output/taskS/cand_ov")
    A = ap.parse_args()
    Path(A.out).mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    fs = sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl"))) + sorted(glob.glob(str(TS / "line_choice_out_i" / "*.pkl")))
    done = 0
    with mp.Pool(A.workers, initializer=_init, initargs=(A.out,)) as pool:
        for k, r in enumerate(pool.imap_unordered(one, fs, chunksize=4)):
            done += r == 1
            if (k + 1) % 300 == 0:
                print(f"  {k + 1}/{len(fs)}")
    print(f"★ готово: {done} листов")


def _init(out):
    global A
    import types
    A = types.SimpleNamespace(out=out)


A = None
if __name__ == "__main__":
    main()
