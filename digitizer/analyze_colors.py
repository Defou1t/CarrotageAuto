r"""
analyze_colors.py — v2-разведка: разделимы ли НАЛОЖЕННЫЕ кривые по ЦВЕТУ?
Для каждой кривой берём реальный цвет чернил вдоль трассы эксперта (из nlgx):
сэмплируем RGB в пиксельных точках трассы → медианный цвет + разброс. Затем в
пределах файла считаем попарные расстояния цветов кривых. Большое мин. расстояние
⇒ кривые цвето-разделимы (v2 = цвето/инстанс-сегментация сработает); малое ⇒
одноцветные наложения (трудный случай, нужна геометрия/continuity).

python analyze_colors.py <nlgx> [<nlgx> ...]
"""
import sys
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image

Image.MAX_IMAGE_PIXELS = None


def curve_color(rgb, curve, step=8, win=2):
    """Медианный RGB чернил вдоль трассы (берём самый ТЁМНЫЙ пиксель в окне ±win
       по x на каждой сэмпл-строке — это перо, не фон)."""
    H, W, _ = rgb.shape
    ty = curve["top_y"]
    samp = []
    pts = [(i, x) for i, x in enumerate(curve["xs"]) if x != NULL]
    for k in range(0, len(pts), step):
        i, x = pts[k]
        y = ty + i
        if not (0 <= y < H and win <= x < W-win):
            continue
        patch = rgb[y, x-win:x+win+1].astype(np.int32)   # окно по x
        dark = patch[np.argmin(patch.sum(axis=1))]        # самый тёмный = чернила
        samp.append(dark)
    if not samp:
        return None, None
    samp = np.array(samp)
    return np.median(samp, axis=0), samp.std(axis=0).mean()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for nlgx in sys.argv[1:]:
        m = extract(nlgx)
        img = find_image(__import__("pathlib").Path(nlgx))
        if not img:
            print(f"{nlgx}: NO IMAGE"); continue
        rgb = np.asarray(Image.open(img).convert("RGB"))
        curves = ds.real_curves(m)
        print(f"\n=== {__import__('pathlib').Path(nlgx).stem[:50]} ({len(curves)} кривых) ===")
        cols = []
        for c in curves:
            col, spread = curve_color(rgb, c)
            if col is None:
                continue
            cols.append((c["name"].split()[0], col))
            print(f"  {c['name'].split()[0]:<9} median RGB=({col[0]:3.0f},{col[1]:3.0f},{col[2]:3.0f})  разброс±{spread:.0f}")
        # попарные расстояния
        if len(cols) >= 2:
            dmin = 1e9; pair = None
            for a in range(len(cols)):
                for b in range(a+1, len(cols)):
                    d = np.linalg.norm(cols[a][1] - cols[b][1])
                    if d < dmin:
                        dmin = d; pair = (cols[a][0], cols[b][0])
            verdict = ("ЦВЕТО-РАЗДЕЛИМЫ" if dmin > 60 else
                       "частично" if dmin > 30 else "ОДНОЦВЕТНЫЕ (трудно)")
            print(f"  -> min попарное расстояние цвета = {dmin:.0f} ({pair[0]}↔{pair[1]}) ⇒ {verdict}")


if __name__ == "__main__":
    main()
