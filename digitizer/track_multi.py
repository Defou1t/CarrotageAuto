r"""
track_multi.py — v2: разделение НАЛОЖЕННЫХ кривых геометрическим трекером поверх
ЧИСТОЙ U-Net маски (foreground). Ведёт N линий сверху-вниз по непрерывности:
предсказание x по наклону + назначение прогонов чернил линиям (Венгерский алгоритм),
цвет — вспомогательная стоимость на пересечениях. Покрывает и монохром (цвет λ=0 эффект).

Проверка: сопоставить N трасс с GT-кривыми nlgx → пиксельная ошибка на кривую,
сравнить с U-Net-бинарным центроидом (который наложенные усредняет).

ЗАПУСК (venv ComfyUI):
  python track_multi.py <ckpt> --nlgx <f> [--d0 D --d1 D] [--lam 0.15]
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image
from unet import UNet
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from infer import load_model, predict_prob

Image.MAX_IMAGE_PIXELS = None


def row_runs(rowprob, thr=0.4, min_gap=3):
    """Прогоны foreground в строке -> [(x_center, x0, x1)]."""
    m = rowprob > thr
    idx = np.nonzero(m)[0]
    if idx.size == 0:
        return []
    runs = []; s = p = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - p > min_gap:
            runs.append(((s+p)/2.0, s, p)); s = v
        p = v
    runs.append(((s+p)/2.0, s, p))
    return runs


def color_at(rgb, y, x, win=2):
    H, W, _ = rgb.shape
    if not (0 <= y < H and win <= x < W-win):
        return np.array([128., 128., 128.])
    patch = rgb[y, int(x)-win:int(x)+win+1].astype(np.int32)
    return patch[np.argmin(patch.sum(1))].astype(np.float64)


def track(prob, rgb, y0, y1, thr=0.4, lam=0.15, max_jump=40, coast_max=25):
    rows = list(range(y0, y1))
    rr = [row_runs(prob[y], thr) for y in rows]
    counts = np.array([len(r) for r in rr])
    nz = counts[counts > 0]
    if nz.size == 0:
        return [], 0
    N = max(1, int(np.percentile(nz, 85)))
    # seed: первая строка сверху с >=N прогонами (там кривые разделены)
    seed = next((i for i, r in enumerate(rr) if len(r) >= N), int(np.argmax(counts)))
    seeds = sorted([r[0] for r in rr[seed]])[:N]
    lines = [{"x": x, "sl": 0.0, "col": color_at(rgb, rows[seed], x),
              "tr": {rows[seed]: x}, "gap": 0} for x in seeds]

    def march(order):
        for i in order:
            y = rows[i]; runs = rr[i]
            preds = np.array([L["x"] + L["sl"] for L in lines])
            if not runs:
                for L in lines:
                    L["x"] += L["sl"]; L["gap"] += 1
                    if L["gap"] > coast_max:
                        L["sl"] = 0.0
                continue
            rx = np.array([r[0] for r in runs])
            rcol = np.array([color_at(rgb, y, r[0]) for r in runs])
            lcol = np.array([L["col"] for L in lines])
            # стоимость: |pred-x| + lam*цвет-расстояние (continuity по предсказанной позиции)
            cost = np.abs(preds[:, None] - rx[None, :])
            cdist = np.linalg.norm(lcol[:, None, :] - rcol[None, :, :], axis=2)
            cost = cost + lam * cdist
            cost[cost > max_jump + lam*442] = 1e6  # гейт по прыжку
            n, m = len(lines), len(runs)
            sz = max(n, m)
            pad = np.full((sz, sz), 1e6)
            pad[:n, :m] = cost
            ri, ci = linear_sum_assignment(pad)
            assigned = set()
            for li, rj in zip(ri, ci):
                if li < n and rj < m and pad[li, rj] < 1e6:
                    nx = rx[rj]
                    L = lines[li]
                    L["sl"] = 0.7*L["sl"] + 0.3*(nx - L["x"])
                    L["x"] = nx; L["tr"][y] = nx; L["gap"] = 0
                    L["col"] = 0.8*L["col"] + 0.2*rcol[rj]
                    assigned.add(li)
            for li, L in enumerate(lines):
                if li not in assigned:
                    L["x"] += L["sl"]; L["gap"] += 1
                    if L["gap"] > coast_max:
                        L["sl"] = 0.0
    march(range(seed, len(rows)))   # вниз
    march(range(seed, -1, -1))      # вверх
    return lines, N


def evaluate(ckpt, nlgx, d0=None, d1=None, lam=0.15, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load_model(ckpt, device)
    m = extract(nlgx)
    rgb = np.asarray(Image.open(find_image(Path(nlgx))).convert("RGB"))
    H, W, _ = rgb.shape
    da = m["depth_axis"]
    def yofd(d): return int(da["top_y"] + (d-da["top_depth"])*da["span_px"]/da["span_depth"])
    y0 = yofd(d0) if d0 else da["top_y"]
    y1 = yofd(d1) if d1 else min(H, da["bottom_y"])
    prob = predict_prob(net, rgb, device, y0=y0, y1=y1)
    lines, N = track(prob, rgb, y0, y1, lam=lam)
    print(f"{Path(nlgx).stem[:46]}  est N={N} lines={len(lines)}  GT curves={len(ds.real_curves(m))}")

    # GT-кривые в окне
    gts = []
    for c in ds.real_curves(m):
        ty = c["top_y"]
        d = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL and y0 <= ty+i < y1}
        if len(d) >= 20:
            gts.append((c["name"].split()[0], d))
    # жадно сопоставить каждую GT с лучшей линией (min медиана |dx|)
    print(f"   {'curve':<9}{'tracker_px':>11}{'binCentroid_px':>15}")
    used = set()
    for name, gd in gts:
        best = None
        for li, L in enumerate(lines):
            common = set(L["tr"]) & set(gd)
            if len(common) < 15:
                continue
            err = np.median([abs(L["tr"][y]-gd[y]) for y in common])
            if best is None or err < best[0]:
                best = (err, li)
        # бинарный центроид в окне колонки кривой (как U-Net-бинарный baseline)
        gx = np.array(list(gd.values()))
        wx0 = max(0, int(gx.min()-0.4*(gx.max()-gx.min())-20))
        wx1 = min(W, int(gx.max()+0.4*(gx.max()-gx.min())+20))
        cerr = []
        for y, gx_ in gd.items():
            seg = prob[y, wx0:wx1]; mm = seg > 0.4
            if mm.any():
                cx = (np.arange(wx0, wx1)[mm]*seg[mm]).sum()/seg[mm].sum()
                cerr.append(abs(cx-gx_))
        cmed = np.median(cerr) if cerr else float('nan')
        tmed = best[0] if best else float('nan')
        print(f"   {name:<9}{tmed:>11.1f}{cmed:>15.1f}")


def main():
    ckpt = sys.argv[1]; args = sys.argv[2:]
    nlgx = None; d0 = d1 = None; lam = 0.15
    i = 0
    while i < len(args):
        if args[i] == "--nlgx": nlgx = args[i+1]; i += 2
        elif args[i] == "--d0": d0 = float(args[i+1]); i += 2
        elif args[i] == "--d1": d1 = float(args[i+1]); i += 2
        elif args[i] == "--lam": lam = float(args[i+1]); i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate(ckpt, nlgx, d0, d1, lam)


if __name__ == "__main__":
    main()
