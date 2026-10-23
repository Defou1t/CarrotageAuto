r"""_cut_trace.py — ОБРУБ ПО ТРАССАМ ДЕКОДЕРА: ВЕРТИКАЛЬНЫЙ ОТРЕЗОК ПУТИ У КРАЯ ШКАЛЫ 1× (23.10, §6.338).

Пиксельные детекторы обруба тонут в штампах и печатных линиях у края. Здесь — по кандидатам ведения (кэш `tcache`/`tcache_out`:
`traces` + `alt` + доп. пути): участок пути, где x почти не меняется (|x − медиана окна| ≤ 2 px) 15…400 строк подряд и лежит в
[x_e − 60, x_e + 60] от края шкалы уровня 0 со стороны v_right (печатная линия трека тянется через весь лист — длиннее 400).
Сигнал строки кривой = 1, если такой отрезок есть в ±8 строках. Мера по кривым эталона с умножающей цепочкой, где эксперт
переключает: доля строк уровня ≥ 1 с сигналом (полнота) и доля строк с сигналом, где уровень ≥ 1 (точность); AUC — по сигналу
как 0/1. Каждый `--every`-й лист.

  _cut_trace.py --every 3
"""
import sys, argparse, pickle, hashlib, time
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np

TS = Path("F:/nds/output/taskS")


def vert_runs(ys, xs, tol=2.0, lmin=15, lmax=400):
    """отрезки (y0, y1, x) пути, где x в пределах tol от медианы, длина lmin…lmax строк"""
    out = []
    i, n = 0, len(ys)
    while i < n:
        j = i
        while j + 1 < n and ys[j + 1] - ys[j] <= 2 and abs(xs[j + 1] - xs[i]) <= tol:
            j += 1
        L = ys[j] - ys[i]
        if lmin <= L <= lmax:
            out.append((int(ys[i]), int(ys[j]), float(np.median(xs[i:j + 1]))))
        i = j + 1
    return out


def one(job):
    from extract_nlgx import extract, NULL
    from auto import level_ink as LI, meta as M
    import decode_levels as DL
    sh, q, tc, extras = job
    try:
        mt = extract(str(q)); c0 = pickle.load(open(tc, "rb"))
    except Exception:
        return sh, []
    paths = [t for _, t in (c0.get("traces") or [])] + [t for _, t in (c0.get("alt") or [])]
    for dd, fld in extras:
        xp = Path(dd) / tc.name
        if xp.exists():
            paths += [t for _, t in (pickle.load(open(xp, "rb")).get(fld) or [])]
    runs = []
    for t in paths:
        ys, xs = np.asarray(t[0]), np.asarray(t[1], float)
        if len(ys) > 15:
            runs += vert_runs(ys, xs)
    out = []
    for c in mt["curves"]:
        if M.mnem_root(c["name"]) == "DA":
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2 or LI.is_shift(ch):
            continue
        lv = np.full(c["n_rows"], -1, np.int8)
        for y0, y1, l in c.get("segments") or []:
            a, b = int(y0) - c["top_y"], int(y1) - c["top_y"] + 1
            lv[max(0, a):max(0, b)] = l
        if (lv >= 1).sum() < 50 or (lv == 0).sum() < 50:
            continue
        xe = ch[0]["x_right"]
        sig = np.zeros(c["n_rows"], bool)
        for y0, y1, x in runs:
            if abs(x - xe) <= 60:
                a, b = y0 - c["top_y"] - 8, y1 - c["top_y"] + 9
                sig[max(0, a):max(0, min(c["n_rows"], b))] = True
        m = lv >= 0
        out.append(dict(curve=c["name"], root=M.mnem_root(c["name"]), sig=sig[m][::4], lv=lv[m][::4]))
    return sh, out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="F:/nds/output/taskS/cut_trace.pkl")
    a = ap.parse_args()
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for f, cache, ex in (("tcache_sheets.txt", "tcache", [(TS / "tcache_pk3", "alt"), (TS / "tcache_inert", "extra")]),
                         ("outside_sheets.txt", "tcache_out", [(TS / "tcache_out", "extra"), (TS / "tcache_out_inert", "extra")])):
        for sh in lst(f)[::a.every]:
            q = SRC.get(sh)
            if q is None:
                continue
            key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
            tc = TS / cache / (key + ".pkl")
            if tc.exists():
                jobs.append((sh, q, tc, ex))
    import multiprocessing as mp
    R = {}; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, (sh, o) in enumerate(pool.imap_unordered(one, jobs)):
            if o:
                R[sh] = o
    pickle.dump(R, open(a.out, "wb"))
    C = [c for o in R.values() for c in o]
    sig = np.concatenate([c["sig"] for c in C]); y = np.concatenate([c["lv"] >= 1 for c in C])
    print(f"★ кривых {len(C)}, строк {len(y)} (уровень ≥ 1 — {y.mean():.2f}), {time.time() - t0:.0f} с")
    print(f"★ сигнал есть у {sig.mean():.3f} строк; полнота (строк ≥ 1 с сигналом) {sig[y].mean():.3f}; точность (с сигналом — ≥ 1) {y[sig].mean():.3f}")
    print(f"   у строк уровня 0 сигнал {sig[~y].mean():.3f}")
    from sklearn.metrics import roc_auc_score
    print(f"★ AUC (0/1) {roc_auc_score(y, sig):.4f}")


if __name__ == "__main__":
    main()
