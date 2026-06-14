"""
demo_masks.py — визуальная проверка per-curve масок (особенно наложенные кривые).
Каждая реальная кривая рисуется своим цветом поверх кропа скана.

python demo_masks.py <nlgx> <image> <out.png> [d_from d_to]
"""
import sys
import numpy as np
from PIL import Image
from extract_nlgx import extract
import dataset as ds

PALETTE = [(230,30,30),(30,90,230),(20,160,60),(210,140,0),(160,40,200),
           (0,170,180),(200,80,140),(120,100,40),(80,80,255),(255,120,0)]


def y_of_depth(m, d):
    da = m["depth_axis"]
    return int(da["top_y"] + (d - da["top_depth"]) * da["span_px"] / da["span_depth"])


def main():
    nlgx, image, out = sys.argv[1], sys.argv[2], sys.argv[3]
    m = extract(nlgx)
    img = Image.open(image).convert("RGB")
    W, H = img.size
    da = m["depth_axis"]
    if len(sys.argv) > 5:
        y0, y1 = y_of_depth(m, float(sys.argv[4])), y_of_depth(m, float(sys.argv[5]))
    else:
        y0, y1 = da["top_y"], min(da["top_y"] + 1600, da["bottom_y"])
    base = np.asarray(img).copy()
    curves = ds.real_curves(m)
    legend = []
    for k, c in enumerate(curves):
        col = PALETTE[k % len(PALETTE)]
        mask = ds.curve_mask(c, H, W, stroke=3)
        # подсветить маску цветом (полупрозрачно)
        ys, xs = np.where(mask[y0:y1])
        base[y0:y1][ys, xs] = (0.35 * base[y0:y1][ys, xs] + 0.65 * np.array(col)).astype(np.uint8)
        legend.append((c["name"].split()[0], col, len(xs)))
    crop = Image.fromarray(base[y0:y1])
    crop.save(out)
    print(f"saved {out} size={crop.size} depth window rows {y0}..{y1}")
    for nm, col, npx in legend:
        print(f"  {nm:<8} color={col} px_in_window={npx}")


if __name__ == "__main__":
    main()
