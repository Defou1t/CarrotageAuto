r"""
_bkz_graphify.py — визуализация базовой BKZ-трассы БЕЗ NeuraLOG: рисует трассу из
_auto.nlgx поверх скана (красным), отдаёт оверлей-обзор + зум-кропы. Чтобы эксперт
увидел результат, пока NeuraLOG крашит на загрузке.

python _bkz_graphify.py <auto.nlgx>
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL

OUT = Path(r"F:\nds\output\yats_bkz")


def trace_xs(m):
    c = [x for x in m["curves"] if not x["name"].split()[0].startswith("DA")][0]
    ty = c["top_y"]
    return {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL}, c, m["depth_axis"]


def y_of_depth(da, d):
    return int(da["top_y"] + (d - da["top_depth"]) / (da["bottom_depth"] - da["top_depth"]) * (da["bottom_y"] - da["top_y"]))


def draw_trace(img, tr, w=2, col=(220, 0, 0)):
    d = ImageDraw.Draw(img)
    pts = sorted(tr.items())
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        if y1 - y0 <= 3:
            d.line([(x0, y0), (x1, y1)], fill=col, width=w)
        else:
            d.point((x0, y0), fill=col)
    return img


def main(nlgx):
    sys.stdout.reconfigure(encoding="utf-8")
    m = extract(nlgx)
    img = m["img_path"]
    rgb = Image.open(img).convert("RGB")
    W, H = rgb.size
    tr, c, da = trace_xs(m)
    stem = Path(nlgx).stem.replace(".frame_auto", "")
    print(f"{stem}: {W}x{H}, трасса {len(tr)} точек, depth {da['top_depth']}..{da['bottom_depth']}")

    # 1) ОБЗОР: даунскейл скан, трассу рисуем в координатах даунскейла (иначе 1px исчезнет)
    sc = 1400 / H
    ov = rgb.resize((max(1, int(W * sc)), 1400), Image.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(ov)
    pts = [(int(x * sc), int(y * sc)) for y, x in sorted(tr.items())]
    for p0, p1 in zip(pts, pts[1:]):
        if abs(p1[1] - p0[1]) <= 2:
            d.line([p0, p1], fill=(220, 0, 0), width=1)
    p_over = OUT / f"{stem}_overview.png"
    ov.save(p_over)

    # 2) ЗУМ-КРОПЫ full-res вокруг интересных глубин (спайки 3347-3380 + контрольная чистая)
    crops = []
    for tag, dc in [("spikes_3360", 3360.0), ("mid_3250", 3250.0), ("low_3460", 3460.0)]:
        yc = y_of_depth(da, dc)
        y0, y1 = max(0, yc - 350), min(H, yc + 350)
        crop = rgb.crop((0, y0, W, y1)).convert("RGB")
        sub = {y - y0: x for y, x in tr.items() if y0 <= y < y1}
        draw_trace(crop, sub, w=2)
        p = OUT / f"{stem}_{tag}.png"
        crop.save(p)
        crops.append(str(p))
        print(f"  кроп {tag} @depth {dc} y[{y0}..{y1}]")
    print("OVERVIEW", p_over)
    for c2 in crops:
        print("CROP", c2)


if __name__ == "__main__":
    main(sys.argv[1])
