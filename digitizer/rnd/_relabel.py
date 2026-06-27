r"""
_relabel.py — НОВАЯ разметка для доучивания U-Net (v3): тест на одном nlgx.
  • маска по РЕАЛЬНОЙ ширине чернила каждой кривой (а не фикс-3px) → толщина сохраняется;
  • вес-карта АКЦЕНТА на обороты: рост веса в зонах level>=1 (×5/×25) и у переходов уровней (35498).
Сохраняет overlay (маска зелёным + вес красным) + печатает статистику.

python _relabel.py <nlgx> [--image f.jpg]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image

OUT = Path(r"F:\nds\output\unet_v3")


def ink_width(gray, x, y, bg, d=40, maxr=30):
    H, W = gray.shape
    xi = int(round(x))
    if not (0 <= y < H and 0 <= xi < W) or gray[y, xi] >= bg - d:
        return 0
    l = xi
    while l > 0 and gray[y, l - 1] < bg - d and xi - l < maxr:
        l -= 1
    r = xi
    while r < W - 1 and gray[y, r + 1] < bg - d and r - xi < maxr:
        r += 1
    return r - l + 1


def level_at_fn(segs):
    def f(y):
        for s, e, l in segs:
            if s <= y <= e:
                return l
        return 0
    return f


def build_targets(m, gray):
    """mask (union, ПОСТОЯННАЯ медианная ширина на кривую — чисто/обучаемо)
       + weight (акцент на обороты, ТОЛЬКО на чернило ±кайма, не на фон)."""
    H, W = gray.shape
    bg = int(np.median(gray))
    mask = np.zeros((H, W), np.uint8)
    weight = np.ones((H, W), np.float32)
    stats = []
    for c in ds.real_curves(m):
        ty = c["top_y"]; xs = c["xs"]; segs = c.get("segments", []) or []
        lvl = level_at_fn(segs)
        trans_rows = {s for s, e, l in segs} | {e for s, e, l in segs}
        # ПРОХОД 1: измерить ширины (для постоянной медианы)
        pts = []
        for i, x in enumerate(xs):
            if x == NULL:
                continue
            y = ty + i
            if not (0 <= y < H):
                continue
            w = ink_width(gray, x, y, bg) or 3
            pts.append((y, int(round(x)), w))
        if len(pts) < 30:
            continue
        medw = int(np.median([w for _, _, w in pts]))
        hw = max(1, medw // 2)                           # постоянная половина-ширина кривой
        # ПРОХОД 2: рендер постоянной ширины + вес на чернило
        for y, xi, _ in pts:
            x0, x1 = max(0, xi - hw), min(W, xi + hw + 1)
            mask[y, x0:x1] = 1
            wt = 2.0 if lvl(y) >= 1 else 1.0
            if any(abs(y - tr) <= 8 for tr in trans_rows):
                wt = 3.0
            if wt > 1.0:                                 # вес на штрих+узкая кайма (±3px), НЕ на фон
                weight[max(0, y - 1):y + 2, max(0, x0 - 3):min(W, x1 + 3)] = np.maximum(
                    weight[max(0, y - 1):y + 2, max(0, x0 - 3):min(W, x1 + 3)], wt)
        stats.append((c["name"].split()[0], medw, len(segs), sum(1 for s, e, l in segs if l >= 1)))
    return mask, weight, stats


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    nlgx = a[0]
    OUT.mkdir(parents=True, exist_ok=True)
    m = extract(nlgx)
    img = a[a.index("--image") + 1] if "--image" in a else (m.get("img_path") if m.get("img_path") and Path(m["img_path"]).is_file() else find_image(Path(nlgx)))
    gray = np.asarray(Image.open(img).convert("L")); H, W = gray.shape
    mask, weight, stats = build_targets(m, gray)
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    print(f"{Path(nlgx).stem}: {W}x{H}, кривых={len(stats)}")
    print(f"{'кривая':<8}{'ширина_px':>10}{'сегментов':>10}{'оборотных':>10}")
    for nm, w, ns, nw in stats:
        print(f"{nm:<8}{w:>10}{ns:>10}{nw:>10}")
    band = weight[ty:by]
    print(f"акцент-вес: доля пикс с весом>1 = {(band>1).mean()*100:.1f}% (макс {band.max():.0f}); маска покрытие={mask[ty:by].mean()*100:.2f}%")
    # overlay: середина трека
    yc = (ty + by) // 2; y0, y1 = max(0, yc - 250), min(H, yc + 250)
    base = np.stack([gray[y0:y1]] * 3, -1).astype(np.uint8)
    mk = mask[y0:y1] > 0; wt = weight[y0:y1]
    base[..., 1] = np.where(mk, 255, base[..., 1])                      # маска зелёным
    base[..., 0] = np.maximum(base[..., 0], ((wt - 1) / 3 * 255).clip(0, 255).astype(np.uint8))  # вес красным
    p = OUT / f"relabel_{Path(nlgx).stem}.png"; Image.fromarray(base).save(p)
    print(f"overlay -> {p}")


if __name__ == "__main__":
    main()
