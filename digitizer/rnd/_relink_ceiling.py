r"""_relink_ceiling.py — ПОТОЛОК СКЛЕЙКИ ПУТЕЙ ДЕКОДЕРА ПОСЛЕ ЧИСТКИ ВЫБРОСОВ (13.10, §6.305).

Картинки (`visual_2026-10-13/cross_far`): линию декодер находит целиком, но кусками в РАЗНЫХ путях (идентичность пути меняется
на разворотах и пересечениях), а пути усыпаны одно-строчными выбросами далеко в сторону — поэтому склейка «сырых» кусков
(§6.284) требовала сотен переключений. Здесь: выбросы прочь (точка дальше `--spike` px от медианы ±`--win` строк своего пути),
затем по строкам эталона — Витерби по путям трека (+ «нет пути»): цена строки 0, если путь ≤ 3 px от эталона, иначе 1; смена
пути — `--sw`. Сколько слотов без честного кандидата в пуле стали бы честными (покрытие ≥ 0.9 строками ≤ 3 px) и сколько нужно
переключений. Пути — каталоги `_track2.py --out` (alt) и/или кэши (traces + alt).

  _relink_ceiling.py --paths F:/nds/output/taskS/tk_base
"""
import sys, argparse, pickle, hashlib, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--paths", nargs="+", default=[r"F:/nds/output/taskS/tk_base"])
ap.add_argument("--spike", type=float, default=15.0)
ap.add_argument("--win", type=int, default=10)
ap.add_argument("--sw", type=float, default=3.0)
ap.add_argument("--all", action="store_true", help="все слоты с эталоном (не только без честного в пуле)")
a = ap.parse_args()
TS = Path(r"F:/nds/output/taskS")
_SRC = None


def despike(tr):
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys], float)
    if len(ys) < 2 * a.win + 1:
        return tr
    from scipy.ndimage import median_filter
    med = median_filter(xs, size=2 * a.win + 1, mode="nearest")
    keep = np.abs(xs - med) <= a.spike
    return {int(y): float(x) for y, x in zip(ys[keep], xs[keep])}


def one(f):
    global _SRC
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    if _SRC is None:
        _SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    rec = pickle.load(open(f, "rb")); sh = rec["sheet"]
    q = _SRC.get(sh)
    if q is None:
        return []
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    P = []
    for d in a.paths:
        pf = Path(d) / (key + ".pkl")
        if not pf.exists():
            return []
        c = pickle.load(open(pf, "rb"))
        for fld in ("traces", "alt"):
            for L, t in (c.get(fld) or []):
                if len(t[0]) >= 30:
                    P.append(despike(dict(zip(t[0].tolist(), t[1].tolist()))))
    if not P:
        return []
    mt = {cc["name"]: cc for cc in extract(str(q))["curves"]}
    out = []
    for s in rec["slots"]:
        if not s["has_truth"]:
            continue
        had = any(c["lab"] and c["lab"][0] for c in s["cands"])
        if had and not a.all:
            continue
        gt = dense(mt[s["name"]]); ys = sorted(gt)
        # ok[t, j]: путь j на строке t ≤ 3 px (с интерполяцией внутри пути не возимся — путь плотный после dense? нет: точки)
        from scipy.spatial import cKDTree
        G = np.array([[y, gt[y]] for y in ys], float)
        M = np.zeros((len(ys), len(P)), bool)
        for j, p in enumerate(P):
            if not p:
                continue
            py = np.fromiter(p.keys(), float); px = np.fromiter(p.values(), float)
            m = (py >= ys[0] - 5) & (py <= ys[-1] + 5)
            if m.sum() < 5:
                continue
            d, _ = cKDTree(np.c_[py[m], px[m]]).query(G)
            M[:, j] = d <= 3.0                                  # расстояние на плоскости (как мера приёмки)
        cover_union = float(M.any(1).mean())
        # Витерби: состояния — пути + «нет» (последний), цена строки 0/1, смена — sw
        n = len(P) + 1
        cost = np.where(M[0], 0.0, 1.0); cost = np.append(cost, 1.0)
        nsw = np.zeros(n, int)
        for t in range(1, len(ys)):
            row = np.append(np.where(M[t], 0.0, 1.0), 1.0)
            b = int(np.argmin(cost)); stay = cost; sw_ = cost[b] + a.sw
            take_sw = sw_ < stay
            new_nsw = np.where(take_sw, nsw[b] + 1, nsw)
            cost = np.minimum(stay, sw_) + row; nsw = new_nsw
        e = int(np.argmin(cost))
        cov = 1.0 - cost[e] / len(ys)                       # грубо: доля строк ≤ 3 px за вычетом штрафов — нижняя оценка
        out.append(dict(sheet=sh, slot=s["name"], root=s["root"], had=had, union=cover_union, cov=float(cov), nsw=int(nsw[e])))
    return out


def main():
    import multiprocessing as mp
    FIELD = {l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    fs = sorted(glob.glob(str(TS / "line_choice_u" / "*.pkl")))
    R = []
    with mp.Pool(4) as pool:
        for r in pool.imap_unordered(one, fs, chunksize=4):
            R += r
    R = [r for r in R if r["sheet"] in FIELD]
    print(f"слотов поля {'всех' if a.all else 'без честного в пуле'}: {len(R)} (листов {len({r['sheet'] for r in R})})")
    u = np.array([r["union"] for r in R]); c = np.array([r["cov"] for r in R]); ns = np.array([r["nsw"] for r in R])
    print(f"  объединение путей покрывает ≥ 0.9 строк ≤ 3 px: {int((u >= 0.9).sum())}; ≥ 0.7: {int((u >= 0.7).sum())}")
    for th in (5, 10, 20):
        print(f"  склейка ≤ {th:2d} переключений и покрытие ≥ 0.9: {int(((c >= 0.9) & (ns <= th)).sum())}")
    print(f"  медиана переключений у склеек с покрытием ≥ 0.9: {np.median(ns[c >= 0.9]) if (c >= 0.9).any() else '—'}")
    pickle.dump(R, open(TS / "relink_ceiling.pkl", "wb"))


if __name__ == "__main__":
    main()
