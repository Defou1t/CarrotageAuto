r"""_line_choice_trace.py — ПРИЗНАКИ ФОРМЫ ТРАССЫ ДЛЯ ВЫБОРА ТРАССЫ: ПРЯМИЗНА, РАЗМАХ, ДРОЖАНИЕ, СКАЧКИ (04.10, к §6.283).

Разбор промахов v2 глазами (`visual_2026-10-03/C_line_choice`): в слот ставится ПРЯМАЯ ПЕЧАТНАЯ ЛИНИЯ (граница трека, красная
вертикаль сетки) или трасса по штриховке — при честной волнистой линии рядом. Признаки §6.280 такую трассу не отличают (извилистость
нормирована на размах). Здесь — по тем же кандидатам кэша, что `_line_choice_scan.py` (прод-путь `traces`, декодер `alt`; плотная
трасса, ≥ 50 строк), без скана:
- straight — доля окон по 200 точек (шаг 100), где трасса в пределах 1 px от прямой (печатная линия);
- xr_px — размах x (p95 − p5), px;
- mad_dx — средний |Δx| между соседними строками;
- jumps — скачков |Δx| > 20 px на 1000 строк;
- flat — самый длинный отрезок с неизменным x / длина;
- agree_src — наибольшая доля строк кандидата, где кандидат ДРУГОГО источника (прод-путь / декодер) в ≤ 2 px;
- agree_any — то же среди всех прочих кандидатов.
Пишет `--out/<тот же md5>.pkl` = {sheet, feats: [по кандидату]}; `--workers` процессов.

  _line_choice_trace.py --workers 4
"""
import sys, argparse, pickle, hashlib, time, os
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


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


def shape_feats(d):
    """форма плотной трассы {y: x} → dict признаков (как в проде: auto/line_choice.shape_feats)"""
    ys = np.array(sorted(d)); xs = np.array([d[y] for y in ys], float)
    n = len(xs)
    st, tot = 0, 0
    for i in range(0, max(1, n - 200 + 1), 100):
        w = xs[i:i + 200]; yy = ys[i:i + 200].astype(float)
        if len(w) < 50:
            continue
        tot += 1
        A = np.vstack([yy - yy.mean(), np.ones(len(yy))]).T
        coef, *_ = np.linalg.lstsq(A, w, rcond=None)
        if np.max(np.abs(w - A @ coef)) <= 1.0:
            st += 1
    dx = np.abs(np.diff(xs)); dy = np.diff(ys)
    cont = dy == 1
    z = np.r_[False, np.diff(np.round(xs)) == 0, False].astype(np.int8)
    dz = np.diff(z)
    st_, en_ = np.flatnonzero(dz == 1), np.flatnonzero(dz == -1)
    flat = int((en_ - st_).max()) if len(st_) else 0
    return dict(straight=st / tot if tot else np.nan, xr_px=float(np.percentile(xs, 95) - np.percentile(xs, 5)),
                mad_dx=float(dx[cont].mean()) if cont.any() else np.nan,
                jumps=float((dx[cont] > 20).sum() * 1000.0 / max(1, n)), flat=flat / max(1, n))


def agreement(cands):
    """доля строк кандидата, где другой кандидат (другого источника / любой) в ≤ 2 px"""
    if not cands:
        return []
    y0 = min(min(d) for _, d in cands); y1 = max(max(d) for _, d in cands)
    A = np.full((len(cands), y1 - y0 + 1), np.nan, np.float32)
    for i, (_, d) in enumerate(cands):
        ys = np.fromiter(d.keys(), int); A[i, ys - y0] = np.fromiter(d.values(), float)
    src = np.array([s_ == "dec" for s_, _ in cands])
    out = []
    for i in range(len(cands)):
        has = ~np.isnan(A[i]); n = has.sum()
        close = np.abs(A[:, has] - A[i, has]) <= 2.0              # NaN → False
        close[i] = False
        fr = close.sum(1) / max(1, n)
        oth = src != src[i]
        out.append(dict(agree_src=float(fr[oth].max()) if oth.any() else 0.0, agree_any=float(fr.max())))
    return out


def one(job):
    sh, tc_path, out = job
    outp = Path(out) / (hashlib.md5(sh.encode("utf-8")).hexdigest()[:16] + ".pkl")
    if outp.exists():
        return sh
    c = pickle.load(open(tc_path, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    cands = [("prod", dense_tr(un(t))) for L, t in (c["traces"] or [])] + [("dec", dense_tr(un(t))) for L, t in (c["alt"] or [])]
    cands = [x for x in cands if len(x[1]) >= 50]
    feats = [shape_feats(d) for _, d in cands]
    for f_, a_ in zip(feats, agreement(cands)):
        f_.update(a_)
    tmp = outp.with_suffix(".tmp"); pickle.dump(dict(sheet=sh, feats=feats), open(tmp, "wb"), protocol=4); os.replace(tmp, outp)
    return sh


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
    ap.add_argument("--scan", default=r"F:/nds/output/taskS/line_choice")
    ap.add_argument("--out", default=r"F:/nds/output/taskS/line_choice_tr")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import multiprocessing as mp
    Path(a.out).mkdir(parents=True, exist_ok=True)
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for f in sorted(Path(a.scan).glob("*.pkl")):
        sh = pickle.load(open(f, "rb"))["sheet"]
        q = SRC[sh]
        key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        jobs.append((sh, str(Path(a.cache) / (key + ".pkl")), a.out))
    print(f"листов {len(jobs)}")
    t0 = time.time(); done = 0
    with mp.Pool(a.workers) as pool:
        for _ in pool.imap_unordered(one, jobs):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    print("готово")


if __name__ == "__main__":
    main()
