r"""_track2_eval.py — СРАВНЕНИЕ ВАРИАНТОВ ПУТЕЙ `_track2.py` ПО СЛОТАМ С ЭТАЛОНОМ (13.10, §6.305).

Для каждого слота с эталоном (скан `line_choice_u`) и каждого варианта (каталоги `_track2.py --out`): лучший путь варианта —
доля строк эталона, где путь в ≤ 3 px по x (`frac3`), и честность в мере приёмки по пикселям (`honest_d`). Парное сравнение
вариантов: Δ числа честных, Δ средней frac3, слоты лучше / хуже (frac3 ± 0.05); отдельно — слоты без честного кандидата в пуле.

  _track2_eval.py F:/nds/output/taskS/tk_base F:/nds/output/taskS/tk_inert
"""
import sys, pickle, hashlib, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
DIRS = [Path(d) for d in sys.argv[1:]]
_SRC = None


def one(f):
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    rec = pickle.load(open(f, "rb")); sh = rec["sheet"]
    global _SRC
    if _SRC is None:
        _SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    SRC = _SRC
    if sh not in SRC:
        return []
    q = SRC[sh]; key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    paths = []
    for d in DIRS:
        pf = d / (key + ".pkl")
        if not pf.exists():
            return []
        un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
        paths.append([dense_tr(un(t)) for L, t in pickle.load(open(pf, "rb"))["alt"]])
    mt = {c["name"]: c for c in extract(str(q))["curves"]}
    out = []
    for s in rec["slots"]:
        if not s["has_truth"]:
            continue
        gt = dense(mt[s["name"]]); tg = _tree(gt)
        had = any(c["lab"] and c["lab"][0] for c in s["cands"])
        xl, xr = s["band"]; wd = max(1.0, xr - xl)
        row = dict(sheet=sh, slot=s["name"], root=s["root"], had=had)
        for vi, P in enumerate(paths):
            bf, bh = 0.0, False
            for p in P:
                com = [y for y in gt if y in p]
                if len(com) < 30:
                    continue
                fr = float(np.mean([abs(p[y] - gt[y]) <= 3 for y in com])) * len(com) / len(gt)
                if fr > bf:
                    bf = fr
                if not bh and fr > 0.3 and honest_d(p, _tree(p), gt, tg)[0]:
                    bh = True
            row[f"f{vi}"] = bf; row[f"h{vi}"] = bh
        out.append(row)
    return out


def main():
    import multiprocessing as mp
    FIELD = {l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    fs = sorted(glob.glob(str(TS / "line_choice_u" / "*.pkl")))
    fs = [f for f in fs]
    R = []
    with mp.Pool(4) as pool:
        for r in pool.imap_unordered(one, fs, chunksize=4):
            R += r
    R = [r for r in R if r["sheet"] in FIELD]
    print(f"слотов поля с эталоном на листах вариантов: {len(R)} (листов {len({r['sheet'] for r in R})})")
    for vi, d in enumerate(DIRS):
        print(f"  {d.name:12s}: честных {sum(r[f'h{vi}'] for r in R):4d}, средняя доля строк ≤ 3 px {np.mean([r[f'f{vi}'] for r in R]):.3f}; "
              f"без честного в пуле: честных {sum(r[f'h{vi}'] for r in R if not r['had'])} из {sum(1 for r in R if not r['had'])}")
    for vi in range(1, len(DIRS)):
        d = np.array([r[f"f{vi}"] - r["f0"] for r in R])
        print(f"  {DIRS[vi].name} против {DIRS[0].name}: доля строк лучше у {int((d > 0.05).sum())}, хуже у {int((d < -0.05).sum())}; "
              f"честных +{sum(1 for r in R if r[f'h{vi}'] and not r['h0'])} / −{sum(1 for r in R if r['h0'] and not r[f'h{vi}'])}")
    pickle.dump(R, open(TS / "track2_eval.pkl", "wb"))


if __name__ == "__main__":
    main()
