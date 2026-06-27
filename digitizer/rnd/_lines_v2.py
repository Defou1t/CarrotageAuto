r"""
_lines_v2.py — IMAGE-FIRST детект линий БЕЗ nlgx (фундамент, §6.6.13).
Разделители (эмпирика STK+DS): сетка СВЕТЛАЯ (V~166), кривые: black PZ/DS тёмные (V<110),
green GZ по hue (G−R>6), red SP (R−G>25). → грид уходит сам. Рамки — из профиля плотности кривых.
Векторизация CC → счёт линий per кадр×цвет. nlgx — только ВАЛИДАЦИЯ (счёт/цвет).

python _lines_v2.py <nlgx>
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from dataset_build import find_image

OUT = Path(r"F:\nds\output")
VIS = {"black": (20, 20, 20), "green": (0, 170, 0), "red": (220, 0, 0)}


def structure_mask(gray, vlen=120, hlen=120):
    """ПРЯМЫЕ длинные линии = тяжёлая сетка/рамки/оси (постоянный x или y). Кривая вьётся → не ловится
    длинным осевым open. Источник — тёмный ink (gray<150), где тяжёлые деления тёмные."""
    ink = (gray < 150).astype(np.uint8)
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen)))
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1)))
    return cv2.dilate(vl | hl, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0


def curve_masks(arr, gray):
    R, G, Bl = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    red = (R - G > 25) & (R - Bl > 15) & (gray < 150)     # V<150: отсечь светлые бурые ПЯТНА
    green = (G - R > 8) & (G - Bl > -2) & (gray < 155) & (~red)  # V<155: отсечь светлую цветную сетку
    black = (gray < 110) & (~red) & (~green)
    st = structure_mask(gray)
    return {"red": red & ~st, "green": green & ~st, "black": black & ~st}


def detect_frames(curve_any, ty, by, W, gapmin=40):
    """кадры из пикселей: столбцовая плотность кривых → регионы, разделённые пустыми пролётами."""
    d = curve_any[ty:by].sum(0).astype(float)
    d = np.convolve(d, np.ones(15) / 15, mode="same")
    on = d > max(d.max() * 0.02, 2)
    frames = []; s = None; gap = 0
    for i, v in enumerate(on):
        if v:
            if s is None:
                s = i
            gap = 0
        else:
            if s is not None:
                gap += 1
                if gap >= gapmin:
                    frames.append((s, i - gap + 1)); s = None
    if s is not None:
        frames.append((s, len(on)))
    return [(a, b) for a, b in frames if b - a >= 60]


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


def count_lines(mask, ty, by, W, prom_frac=0.10, min_dist=45):
    """#линий цвета = пики ПЛОТНОСТИ по столбцам (сумма по строкам убирает фрагментацию штриха).
    Каждый пик = x-дом линии. Возвращает [x-дом], покрытие-строк."""
    Hh = by - ty
    dens = mask[ty:by].sum(0).astype(float)
    dens[dens > 0.6 * Hh] = 0                             # вырезать ВЕРТИКАЛЬНЫЕ структуры (рамки/оси/грид)
    d = np.convolve(dens, np.ones(11) / 11, mode="same")
    if d.max() <= 0:
        return [], 0
    peaks = peaks_prom(d, prom_frac * d.max(), min_dist)
    cov = int(mask[ty:by].any(1).sum())
    return peaks, cov


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    F = sys.argv[1]
    m = extract(F); img = find_image(Path(F)) or m["img_path"]
    arr = np.asarray(Image.open(img).convert("RGB")); H, W, _ = arr.shape
    gray = np.asarray(Image.open(img).convert("L"))
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    mk = curve_masks(arr, gray)
    curve_any = mk["red"] | mk["green"] | mk["black"]
    curve_any[:ty] = False; curve_any[by:] = False

    # валидация-цель из nlgx
    gtc = {}
    for c in m["curves"]:
        nm = c["name"].split()[0]
        if nm.startswith("DA"):
            continue
        v = [x for x in c["xs"] if x != NULL]
        if v:
            gtc[nm] = (int(np.median(v)),)
    print(f"{Path(F).name[:46]} | {W}x{H} y[{ty}..{by}]")
    print(f"nlgx кривые (вал-цель): " + ", ".join(f"{k}@{v[0]}" for k, v in sorted(gtc.items(), key=lambda kv: kv[1][0])))

    print("\nСЧЁТ ЛИНИЙ per цвет (пики плотности по столбцам, image-only):")
    total = 0; found = []
    for color in ("black", "green", "red"):
        peaks, cov = count_lines(mk[color], ty, by, W)
        total += len(peaks)
        for p in peaks:
            found.append((p, color))
        if peaks:
            print(f"  {color:6}: линий={len(peaks)} x-дома={peaks} строк-покрытие={100*cov/(by-ty):.0f}%")
    print(f"\nИТОГО линий image-only={total} vs nlgx={len(gtc)}")
    print("сверка позиций (image x-дом → ближайшая nlgx):")
    for p, color in sorted(found):
        nm = min(gtc, key=lambda k: abs(gtc[k][0] - p))
        d = abs(gtc[nm][0] - p)
        print(f"  {color:6} x={p:4} → {nm}@{gtc[nm][0]} (Δ={d}){'  ✓' if d < 60 else '  ?'}")
    # вис: маски цветом
    vis = (np.ones_like(arr) * 255)
    for color in ("black", "green", "red"):
        vis[mk[color]] = VIS[color]
    Image.fromarray(vis).crop((30, ty, min(W, 1180), by)).resize((1150, (by - ty) // 4), Image.LANCZOS).save(OUT / "lines_v2_vis.png")
    print("вис (чистые маски: black/green/red) -> F:\\nds\\output\\lines_v2_vis.png")


if __name__ == "__main__":
    main()
