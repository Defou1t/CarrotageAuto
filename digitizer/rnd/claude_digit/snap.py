"""Притяжка моей ломаной к туши: в каждой строке — центр ближайшего рана «темнее бумаги» в ±R px от моей x.
Своя минимальная реализация (без кода проекта): бумага строки = 90-й перцентиль яркости, тушь = темнее на thr.

  python snap.py <скан> mine.json out.json [--r 8] [--thr 30]
"""
import sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("img"); ap.add_argument("mine"); ap.add_argument("out")
ap.add_argument("--r", type=int, default=8)
ap.add_argument("--thr", type=int, default=30)
a = ap.parse_args()
pts = sorted((int(y), float(x)) for y, x in json.load(open(a.mine, encoding="utf-8")))
dense = {}
for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
    for yy in range(y0, y1):
        dense[yy] = x0 + (x1 - x0) * (yy - y0) / max(1, y1 - y0)
dense[pts[-1][0]] = pts[-1][1]
im = Image.open(a.img).convert("L")
ys = sorted(dense)
y0, y1 = ys[0], ys[-1] + 1
G = np.asarray(im.crop((0, y0, im.width, y1)), np.int16)
paper = np.percentile(G[:, ::3], 90, axis=1)
out, moved = [], 0
for y in ys:
    row = G[y - y0]; xm = dense[y]; c = int(round(xm))
    lo, hi = max(0, c - a.r), min(len(row), c + a.r + 1)
    ink = row[lo:hi] < paper[y - y0] - a.thr
    if not ink.any():
        out.append([y, xm]); continue
    d = np.diff(np.concatenate([[0], ink.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    # ран, содержащий мою x или ближайший; центр — по ПОЛНОМУ рану (может выйти за окно)
    dist = [0 if s + lo <= xm < e + lo else min(abs(s + lo - xm), abs(e - 1 + lo - xm)) for s, e in zip(st, en)]
    i = int(np.argmin(dist))
    l, r = st[i] + lo, en[i] - 1 + lo
    full = row < paper[y - y0] - a.thr
    while l > 0 and full[l - 1] and xm - l < 3 * a.r:
        l -= 1
    while r < len(row) - 1 and full[r + 1] and r - xm < 3 * a.r:
        r += 1
    out.append([y, (l + r) / 2.0]); moved += 1
json.dump(out, open(a.out, "w"))
print(f"строк {len(out)}, притянуто {moved}")
