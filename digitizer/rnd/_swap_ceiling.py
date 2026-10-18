r"""_swap_ceiling.py — ПОТОЛОК «РАСПЛЕТАНИЯ» СМЕСЕЙ: ОБМЕН ХВОСТАМИ КАНДИДАТОВ В ТОЧКАХ ПЕРЕСЕЧЕНИЯ (18.10, §6.317).

Трасса-смесь (§6.316) на пересечении переходит с одной кривой на другую. Если кандидаты A и B пересекаются в строке y*
(|xA − xB| ≤ 2), то A[:y*] + B[y*:] и B[:y*] + A[y*:] — кандидаты, в которых один из двух — чистая линия. По слотам с эталоном
без честного кандидата в пуле скана (`line_choice_i`): есть ли честный среди обменов (мера приёмки по пикселям, `honest_d`).
Обмены — только для пар с частичным перекрытием (0.1…0.9 строк ≤ 3 px), по всем точкам пересечения пары (≤ `--max-x` на пару).

  _swap_ceiling.py --limit 300
"""
import sys, argparse, pickle, hashlib, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
_SRC = None


def one(f):
    global _SRC
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    from _cand_overlap import pool_of
    if _SRC is None:
        _SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    rec = pickle.load(open(f, "rb")); sh = rec["sheet"]; q = _SRC[sh]
    todo = [s for s in rec["slots"] if s["has_truth"] and not any(c["lab"] and c["lab"][0] for c in s["cands"])]
    if not todo:
        return []
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    c, ex, un = pool_of(sh, key)
    T = []
    for cs in rec["cands"]:
        t = c["traces"][cs["idx"]][1] if cs["src"] == "prod" else (c["alt"][cs["idx"]][1] if cs["idx"] < 1000 else ex[cs["idx"] - 1000][1])
        T.append(dense_tr(un(t)))
    # пересечения пар
    SW = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a_, b_ = T[i], T[j]
            com = sorted(set(a_) & set(b_))
            if len(com) < 100:
                continue
            ca = np.array(com); dx = np.array([a_[y] - b_[y] for y in com])
            near = np.abs(dx) <= 3
            fr = near.mean()
            if not (0.1 <= fr <= 0.9):
                continue
            sg = np.sign(dx); xs = np.flatnonzero((np.abs(dx) <= 2) & (np.r_[0, np.diff(sg)] != 0))
            xs = xs[:A.max_x]
            for k in xs:
                y = int(ca[k])
                SW.append({**{yy: v for yy, v in a_.items() if yy < y}, **{yy: v for yy, v in b_.items() if yy >= y}})
                SW.append({**{yy: v for yy, v in b_.items() if yy < y}, **{yy: v for yy, v in a_.items() if yy >= y}})
    mt = {cc["name"]: cc for cc in extract(str(q))["curves"]}
    out = []
    for s in todo:
        gt = dense(mt[s["name"]]); tg = _tree(gt)
        xl, xr = s["band"]; wd = max(1.0, xr - xl)
        got = False
        for t in SW:
            if len(t) < 50:
                continue
            u = (np.median(list(t.values())) - xl) / wd
            if -0.1 <= u <= 1.1 and honest_d(t, _tree(t), gt, tg)[0]:
                got = True; break
        out.append(dict(sheet=sh, slot=s["name"], nsw=len(SW), got=got))
    return out


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-x", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    A = ap.parse_args()
    import multiprocessing as mp, random
    FIELD = {l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    fs = sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl")))
    random.seed(17); random.shuffle(fs); fs = fs[:A.limit]
    R = []
    with mp.Pool(A.workers, initializer=_init, initargs=(A.max_x,)) as pool:
        for r in pool.imap_unordered(one, fs, chunksize=2):
            R += r
    R = [r for r in R if r["sheet"] in FIELD]
    print(f"слотов поля без честного в пуле (из {A.limit} листов): {len(R)}; честный среди обменов: {sum(r['got'] for r in R)}; "
          f"обменов на лист — медиана {np.median([r['nsw'] for r in R]) if R else 0:.0f}")


def _init(mx):
    global A
    import types
    A = types.SimpleNamespace(max_x=mx)


A = None
if __name__ == "__main__":
    main()
