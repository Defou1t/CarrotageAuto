r"""
overlay_diff.py — диагностический оверлей: трасса ЭКСПЕРТА (orig nlgx) vs трасса
НАШЕГО трекера (auto nlgx) на скане, бок-о-бок, одинаковые цвета на кривую.
Своп виден как «наша линия цвета X повторяет форму линии цвета Y».

python overlay_diff.py --orig <o.nlgx> --auto <a.nlgx> [--image f.jpg]
                       --d0 D --d1 D [--out png] [--scale 0.5]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from extract_nlgx import extract, NULL
from dataset_build import find_image

Image.MAX_IMAGE_PIXELS = None

COLORS = {  # short name -> RGB
    "GZ21": (220, 30, 30), "GZ2": (220, 30, 30),
    "GZ11": (245, 170, 0), "GZ1": (245, 170, 0),
    "GZ31": (30, 90, 235), "GZ3": (30, 90, 235),
    "OGZ1": (210, 30, 210), "OGZ": (210, 30, 210),
}
DEF = (0, 170, 0)


def yofd(da, d):
    ty, by, td, bd = da["top_y"], da["bottom_y"], da["top_depth"], da["bottom_depth"]
    return int(ty + (d - td) * (by - ty) / (bd - td))


def draw_panel(rgb, m, y0, y1, title, x0=0, x1=None):
    x1 = rgb.shape[1] if x1 is None else x1
    crop = rgb[y0:y1, x0:x1].copy()
    img = Image.fromarray(crop)
    dr = ImageDraw.Draw(img)
    for c in m["curves"]:
        nm = c["name"].split()[0]
        if nm.startswith("DA"):
            continue
        col = COLORS.get(nm, DEF)
        ty = c["top_y"]
        pts = [(x - x0, ty + i - y0) for i, x in enumerate(c["xs"])
               if x != NULL and y0 <= ty + i < y1 and x0 <= x < x1]
        if len(pts) >= 2:
            dr.line(pts, fill=col, width=3)
    # легенда
    yy = 6
    for nm in ("GZ2", "GZ1", "GZ3", "OGZ"):
        dr.rectangle([6, yy, 26, yy + 12], fill=COLORS[nm])
        dr.text((30, yy), nm, fill=(0, 0, 0))
        yy += 16
    dr.text((6, yy + 4), title, fill=(0, 0, 0))
    return img


def main():
    a = sys.argv[1:]
    orig = auto = image = out = None
    d0 = d1 = None; scale = 0.5; x0 = 0; x1 = None
    i = 0
    while i < len(a):
        if a[i] == "--orig": orig = a[i+1]; i += 2
        elif a[i] == "--auto": auto = a[i+1]; i += 2
        elif a[i] == "--image": image = a[i+1]; i += 2
        elif a[i] == "--d0": d0 = float(a[i+1]); i += 2
        elif a[i] == "--d1": d1 = float(a[i+1]); i += 2
        elif a[i] == "--x0": x0 = int(a[i+1]); i += 2
        elif a[i] == "--x1": x1 = int(a[i+1]); i += 2
        elif a[i] == "--out": out = a[i+1]; i += 2
        elif a[i] == "--scale": scale = float(a[i+1]); i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8")
    mo = extract(orig); ma = extract(auto)
    img_path = image or find_image(Path(orig))
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    H = rgb.shape[0]
    da = mo["depth_axis"]
    y0 = max(0, yofd(da, d0)); y1 = min(H, yofd(da, d1))
    left = draw_panel(rgb, mo, y0, y1, f"ЭКСПЕРТ {d0:.0f}-{d1:.0f}", x0, x1)
    right = draw_panel(rgb, ma, y0, y1, f"НАШ ТРЕКЕР {d0:.0f}-{d1:.0f}", x0, x1)
    gap = 12
    W = left.width + right.width + gap
    canvas = Image.new("RGB", (W, left.height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    if scale != 1.0:
        canvas = canvas.resize((int(W*scale), int(left.height*scale)), Image.LANCZOS)
    out = out or rf"F:\nds\output\overlay_diff_{d0:.0f}_{d1:.0f}.png"
    canvas.save(out)
    print(f"-> {out}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
