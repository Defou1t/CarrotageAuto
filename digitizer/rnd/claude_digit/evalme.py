"""Оценка моей ручной оцифровки против эталона и выдачи прода на том же участке.

  python evalme.py <лист.nlgx> "<имя кривой эталона>" points.json [--prod <выдача _auto.nlgx>] [--prod-name "<имя>"]
points.json: [[y, x], ...] — вершины ломаной (как у эксперта), линейная интерполяция по строкам между ними.
"""
import sys, json, argparse, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL

ap = argparse.ArgumentParser()
ap.add_argument("nlgx")
ap.add_argument("name")
ap.add_argument("points")
ap.add_argument("--prod-dir", default=r"F:/nds/output/taskS/rp_vc/N")
a = ap.parse_args()


def dense_poly(pts, gap=200):
    pts = sorted((int(round(y)), float(x)) for y, x in pts)
    out = {}
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        out[y0] = x0
        if 0 < y1 - y0 <= gap:
            for yy in range(y0 + 1, y1):
                out[yy] = x0 + (x1 - x0) * (yy - y0) / (y1 - y0)
    if pts:
        out[pts[-1][0]] = pts[-1][1]
    return out


def dense_curve(c, gap=200):
    return dense_poly([(c["top_y"] + i, x) for i, x in enumerate(c["xs"]) if x != NULL], gap)


def d2(tr, gt, rows):
    """расстояние от точки эталона до ломаной трассы (уплотнённой по x) по строкам rows"""
    ys = np.array(sorted(tr), float); xs = np.array([tr[int(y)] for y in ys])
    P = [np.stack([ys, xs], 1)]
    for i in range(len(ys) - 1):
        dx = xs[i + 1] - xs[i]
        if ys[i + 1] - ys[i] <= 30 and abs(dx) > 1:
            n = int(np.ceil(abs(dx))); t = np.arange(1, n) / n
            P.append(np.stack([ys[i] + (ys[i + 1] - ys[i]) * t, xs[i] + dx * t], 1))
    P = np.concatenate(P)
    out = []
    for y in rows:
        q = np.array([y, gt[y]])
        out.append(float(np.min(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]))))
    return np.array(out)


def report(tag, tr, gt, y0, y1):
    rows = [y for y in range(y0, y1 + 1) if y in gt and y in tr]
    if not rows:
        print(f"{tag}: нет общих строк"); return
    e = np.array([abs(tr[y] - gt[y]) for y in rows])
    dd = d2(tr, gt, rows)
    cov = len(rows) / max(1, len([y for y in range(y0, y1 + 1) if y in gt]))
    print(f"{tag:10s} строк {len(rows):4d}, покрытие {cov:.2f} | |Δx|: медиана {np.median(e):.1f}, 90% {np.percentile(e, 90):.1f}, "
          f"≤ 3 px {np.mean(e <= 3):.2f} | на плоскости: медиана {np.median(dd):.1f}, ≤ 3 px {np.mean(dd <= 3):.2f}")


m = extract(a.nlgx)
gt = dense_curve(next(c for c in m["curves"] if c["name"] == a.name))
mine = dense_poly(json.load(open(a.points, encoding="utf-8")))
y0, y1 = min(mine), max(mine)
report("я", mine, gt, y0, y1)
stem = Path(a.nlgx).stem
pd = Path(a.prod_dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
got = next(pd.glob("*_auto.nlgx"), None)
if got:
    P = {c["name"]: dense_curve(c) for c in extract(str(got))["curves"]}
    if a.name in P:
        report("прод", P[a.name], gt, y0, y1)
    # лучшая из выдачи на участке
    best = min(((np.median([abs(t[y] - gt[y]) for y in range(y0, y1 + 1) if y in t and y in gt] or [1e9]), k) for k, t in P.items()))
    print(f"ближайшая кривая выдачи на участке: «{best[1]}» (медиана |Δx| {best[0]:.1f})")
