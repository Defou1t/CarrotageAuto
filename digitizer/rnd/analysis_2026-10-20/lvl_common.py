import numpy as np
from scipy.spatial import cKDTree
def tree(tr):
    ys = np.array(sorted(tr), float); xs = np.array([tr[int(y)] for y in ys], float)
    P = [np.stack([ys, xs], 1)]
    dy = np.diff(ys); dx = np.diff(xs)
    for i in np.flatnonzero((dy <= 30) & (np.abs(dx) > 1) & (np.abs(dx) <= 3000)):
        n = int(np.ceil(abs(dx[i]))); t = np.arange(1, n) / n
        P.append(np.stack([ys[i] + dy[i] * t, xs[i] + dx[i] * t], 1))
    return cKDTree(np.concatenate(P))


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return 99.0, 0.0
    d1, _ = tree(tr).query(np.array([[y, gt[y]] for y in com], float))
    d2, _ = tree(gt).query(np.array([[y, tr[y]] for y in com], float))
    return max(float(np.median(d1)), float(np.median(d2))), len(com) / max(1, len(gt))


def lv_rows(c):
    out = {}
    for y0, y1, lv in c.get("segments") or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


def relevel(tr, lw, lg, chain):
    if len(chain) < 2:
        return tr
    out = {}
    for y, x in tr.items():
        lo, lt = lw.get(y, 0), lg.get(y, 0)
        if lo == lt:
            out[y] = x; continue
        if lo >= len(chain) or lt >= len(chain):
            out[y] = x + 1e4; continue
        so, sg = chain[lo], chain[lt]
        v = so["v_left"] + (x - so["x_left"]) * (so["v_right"] - so["v_left"]) / ((so["x_right"] - so["x_left"]) or 1)
        out[y] = sg["x_left"] + (v - sg["v_left"]) * (sg["x_right"] - sg["x_left"]) / ((sg["v_right"] - sg["v_left"]) or 1)
    return out


HON = lambda e: e[0] <= 3.0 and e[1] >= 0.9
