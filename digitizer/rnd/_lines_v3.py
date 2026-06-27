r"""
_lines_v3.py — ИНСТАНСЫ из НОВОГО (recall) prob: счёт линий + per-инстанс ЦВЕТ + позиция, без nlgx.
prob высокого recall видит бледное и игнорит сетку/пятна → пики плотности по столбцам = #линий и x-дома;
для каждой линии — трасса prob-центроидом в её полосе + ЦВЕТ по доминантному hue вдоль трассы (кайма=меньшинство,
игнорится). Структуру-остаток (рамки) режем по сверх-плотным столбцам. nlgx — только ВАЛИДАЦИЯ.

python _lines_v3.py <prob.npy> <nlgx>
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from dataset_build import find_image


def peaks_prom(d, min_prom, min_dist):
    n = len(d); cand = [i for i in range(1, n - 1) if d[i] >= d[i - 1] and d[i] > d[i + 1]]
    pr = []
    for i in cand:
        j = i - 1; lb = d[i]
        while j >= 0 and d[j] <= d[i]:
            lb = min(lb, d[j]); j -= 1
        j = i + 1; rb = d[i]
        while j < n and d[j] <= d[i]:
            rb = min(rb, d[j]); j += 1
        if d[i] - max(lb, rb) >= min_prom:
            pr.append((i, d[i]))
    pr.sort(key=lambda t: -t[1]); kept = []
    for i, h in pr:
        if all(abs(i - k) >= min_dist for k in kept):
            kept.append(i)
    return sorted(kept)


def color_of(arr, gray, trace, bg):
    """доминантный цвет вдоль трассы (ink-пиксели): red/green/blue/black."""
    px = []
    for y, x in trace.items():
        xi = int(round(x))
        if 0 <= y < arr.shape[0] and 0 <= xi < arr.shape[1] and gray[y, xi] < bg - 20:
            px.append(arr[y, xi])
    if len(px) < 10:
        return "?", 0
    px = np.array(px); R, G, B = np.median(px[:, 0]), np.median(px[:, 1]), np.median(px[:, 2])
    if R - G > 22 and R - B > 12:
        c = "red"
    elif G - R > 8 and G - B > -3:
        c = "green"
    elif B - R > 12 and B - G > 4:
        c = "blue"
    else:
        c = "black"
    return c, int(np.median(gray[[y for y in list(trace)[:1]]] if False else [0]))


def trace_band(prob, xc, ty, by, W, half=70, thr=0.35):
    xa, xb = max(0, xc - half), min(W, xc + half); cols = np.arange(xa, xb); out = {}
    for y in range(ty, by):
        seg = prob[y, xa:xb]; mk = seg > thr
        if mk.any():
            out[y] = float((cols[mk] * seg[mk]).sum() / seg[mk].sum())
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    probpath, F = sys.argv[1], sys.argv[2]
    prob = np.load(probpath)
    m = extract(F); img = find_image(Path(F)) or m["img_path"]
    arr = np.asarray(Image.open(img).convert("RGB")); H, W = arr.shape[:2]
    gray = np.asarray(Image.open(img).convert("L")); bg = int(np.median(gray[::7, ::7]))
    prob = prob[:H, :W]
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, prob.shape[0], da["bottom_y"])
    gtc = {}
    for c in m["curves"]:
        nm = c["name"].split()[0]
        if nm.startswith("DA"):
            continue
        v = [x for x in c["xs"] if x != NULL]
        if v:
            gtc[nm] = int(np.median(v))
    print(f"{Path(F).name[:44]} prob={Path(probpath).name} | y[{ty}..{by}]")
    print("nlgx (цель): " + ", ".join(f"{k}@{v}" for k, v in sorted(gtc.items(), key=lambda kv: kv[1])))

    Hh = by - ty
    dens = (prob[ty:by] > 0.35).sum(0).astype(float)
    dens[dens > 0.7 * Hh] = 0                              # вырезать сверх-плотные столбцы (рамки/оси)
    d = np.convolve(dens, np.ones(11) / 11, mode="same")
    peaks = peaks_prom(d, 0.10 * d.max(), 45) if d.max() > 0 else []
    print(f"\nИНСТАНСЫ (prob density peaks): {len(peaks)} линий")
    for p in peaks:
        tr = trace_band(prob, p, ty, by, W)
        col, _ = color_of(arr, gray, tr, bg)
        nm = min(gtc, key=lambda k: abs(gtc[k] - p)) if gtc else "?"
        dd = abs(gtc.get(nm, -999) - p)
        print(f"  x-дом={p:4} цвет={col:6} строк={len(tr)} → {nm}@{gtc.get(nm,'?')} (Δ={dd}){'  ✓' if dd < 60 else ''}")
    print(f"\nИТОГО линий image-only={len(peaks)} vs nlgx={len(gtc)}")


if __name__ == "__main__":
    main()
