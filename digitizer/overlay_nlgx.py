"""
Наложение пиксель-трассы эксперта (из nlgx) на скан — визуальная проверка,
что калибровка/трасса корректны (штрихи ложатся на чернила).
Рисует: трассы кривых (по цвету), линии Depth Grid, вертикали x_left/x_right шкалы.

python overlay_nlgx.py <file.nlgx> <image.jpg> <out_dir> [d_from d_to ...]
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from extract_nlgx import extract, depth_of, NULL

CURVE_COLORS = {"BK": (220, 30, 30), "MBK": (30, 90, 220)}
DEFAULT = (220, 30, 30)


def y_of_depth(m, d):
    da = m["depth_axis"]
    return da["top_y"] + (d - da["top_depth"]) * da["span_px"] / da["span_depth"]


def render_window(m, img, d_from, d_to, out):
    y0 = int(y_of_depth(m, d_from)); y1 = int(y_of_depth(m, d_to))
    crop = img.crop((0, y0, img.width, y1))
    dr = ImageDraw.Draw(crop)

    # depth grid (faint)
    dg = m.get("depth_grid", {})
    for gy in dg.get("ys", []):
        if y0 <= gy <= y1:
            dr.line([(0, gy-y0), (img.width, gy-y0)], fill=(150, 200, 150), width=1)
            dd = depth_of(m, gy)
            dr.text((2, gy-y0-10), f"{dd:.0f}", fill=(40, 120, 40))

    # scale axis verticals (track edges)
    if m["scale_axes"]:
        sa = m["scale_axes"][0]
        for xx, lab in [(sa["x_left"], "L"), (sa["x_right"], "R")]:
            dr.line([(xx, 0), (xx, y1-y0)], fill=(180, 180, 0), width=1)

    # traces
    for c in m["curves"]:
        base = c["name"].strip().split()[0].rstrip("0123456789").upper()
        if c.get("xs") and len([x for x in c["xs"] if x != NULL]) < 10:
            continue  # skip the depth-axis artifact curve
        col = CURVE_COLORS.get(base, DEFAULT)
        top_y = c["top_y"]
        for i, x in enumerate(c["xs"]):
            if x == NULL:
                continue
            y = top_y + i
            if y0 <= y <= y1:
                yy = y - y0
                dr.point([(x-1, yy), (x, yy), (x+1, yy)], fill=col)
    crop.save(out)
    print(f"saved {out} size={crop.size} depth {d_from}..{d_to}")


def main():
    nlgx, image, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    windows = sys.argv[4:]
    m = extract(nlgx)
    img = Image.open(image).convert("RGB")
    print(f"image {img.size}, depth axis y {m['depth_axis']['top_y']}..{m['depth_axis']['bottom_y']}")
    Path(outdir).mkdir(parents=True, exist_ok=True)

    pairs = []
    if windows:
        for i in range(0, len(windows), 2):
            pairs.append((float(windows[i]), float(windows[i+1])))
    else:
        td, bd = m["top_depth"], m["bottom_depth"]
        pairs = [(4360, 4380), (4470, 4490), (4540, 4560)]

    for a, b in pairs:
        out = Path(outdir) / f"overlay_{int(a)}_{int(b)}.png"
        render_window(m, img, a, b, str(out))

    # full overview downscaled
    da = m["depth_axis"]
    full = img.crop((0, da["top_y"], img.width, da["bottom_y"]))
    dr = ImageDraw.Draw(full)
    for c in m["curves"]:
        base = c["name"].strip().split()[0].rstrip("0123456789").upper()
        if len([x for x in c["xs"] if x != NULL]) < 10:
            continue
        col = CURVE_COLORS.get(base, DEFAULT)
        for i, x in enumerate(c["xs"]):
            if x != NULL:
                y = c["top_y"] + i - da["top_y"]
                dr.point([(x, y)], fill=col)
    scale = 1000.0 / full.height
    full = full.resize((max(1,int(full.width*scale)), 1000))
    ov = Path(outdir) / "overlay_full.png"
    full.save(ov)
    print(f"saved {ov} size={full.size}")


if __name__ == "__main__":
    main()
