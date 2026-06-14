"""
render_labels.py — отрисовка эталонного LAS как штрихов поверх скана.

Цель (фундамент обеих веток — эвристики и нейросети):
  • подтвердить калибровку (штрихи должны лечь на реальные чёрные линии);
  • определить ИСТИННЫЕ масштабы перевыносов (какой s_max кладёт штрих на линию);
  • получить АВТО-РАЗМЕТКУ для обучения сети (пиксельные маски кривой по scale).

Логика перевыносов (по объяснению эксперта):
  значение V кривой рисуется в позиции x = bx0 + (V / s_max) * band_w для КАЖДОГО
  масштаба s_max, в который V влезает (V<=s_max). 1× (мелкий s_max) — правее,
  5×/25× — левее. «Основной» штрих = наименьший s_max, в который V влезает
  (максимальное разрешение), его и читает эксперт.

Запуск:
  python render_labels.py <stem> <curve> <depth_from> <depth_to> [s1 s2 s3 ...]
  напр.:
  python render_labels.py Semeguniv_20_BK+MBK_4350_4610_200_D1 BK 4400 4445 19 95 475
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

PROJ_IMG = r"F:\nds\projects\Semeguniv_020\img"
PROJ_LAS = r"F:\nds\projects\Semeguniv_020\las"
OUT = r"F:\nds\output"

# цвета по уровню масштаба: 1× / 5× / 25× / 125×
LEVEL_COLORS = [(29, 158, 117), (216, 90, 48), (83, 74, 183), (200, 120, 0)]


def load_las(path):
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    # имена кривых из ~Curve
    names = []
    sec = None
    for l in lines:
        s = l.strip()
        if s.startswith("~"):
            sec = s[1].upper(); continue
        if sec == "C" and s and not s.startswith("#"):
            names.append(s.split(".")[0].strip())
    di = next(i for i, l in enumerate(lines) if l.strip().startswith("~A"))
    rows = []
    for l in lines[di + 1:]:
        if not l.strip():
            continue
        try:
            rows.append([float(x) for x in l.split()])
        except ValueError:
            pass
    arr = np.array(rows)
    return names, arr


def main():
    stem = sys.argv[1]
    curve = sys.argv[2]
    d_from = float(sys.argv[3]); d_to = float(sys.argv[4])
    scales = [float(x) for x in sys.argv[5:]] or [19, 95, 475]

    ana = json.load(open(Path(PROJ_IMG) / f"{stem}_analysis.json", encoding="utf-8"))
    cal = ana["pixel_calibration"]; mpp = cal["meters_per_pixel"]; b = cal["depth_at_y0"]
    cinfo = next(c for c in ana["curves"] if c["name"] == curve)
    bx0, bx1 = cinfo["track_band_px"]; W = bx1 - bx0

    names, arr = load_las(Path(PROJ_LAS) / f"{stem}.las")
    ci = names.index(curve)
    depth = arr[:, 0]; val = arr[:, ci]

    def d2y(d): return (d - b) / mpp

    y0 = int(d2y(d_from)); y1 = int(d2y(d_to))
    x0 = max(0, bx0 - 20); x1 = bx1 + 20
    img = Image.open(Path(PROJ_IMG) / f"{stem}.jpg").convert("RGB").crop((x0, y0, x1, y1))
    d = ImageDraw.Draw(img)

    scales = sorted(scales)  # по возрастанию: [19,95,475]
    drawn = {i: 0 for i in range(len(scales))}
    for D, V in zip(depth, val):
        if V <= -900 or D < d_from or D > d_to:
            continue
        y = d2y(D) - y0
        # основной штрих = наименьший масштаб, в который V влезает
        for lvl, s in enumerate(scales):
            if abs(V) <= s:
                x = bx0 + (V / s) * W - x0
                col = LEVEL_COLORS[lvl % len(LEVEL_COLORS)]
                d.ellipse([x-2.5,y-2.5,x+2.5,y+2.5],fill=col)
                drawn[lvl] += 1
                break
    # рамка полосы
    d.line([(bx0 - x0, 0), (bx0 - x0, y1 - y0)], fill=(0, 0, 0), width=1)
    d.line([(bx1 - x0, 0), (bx1 - x0, y1 - y0)], fill=(0, 0, 0), width=1)

    out = Path(OUT) / f"labels_{curve}_{int(d_from)}_{int(d_to)}.png"
    img.save(out)
    print("saved", out, "size", img.size)
    print("scales(asc):", scales, "| points per level:", drawn)
    print(f"value range in window: {val[(depth>=d_from)&(depth<=d_to)&(val>-900)].min():.1f}"
          f"..{val[(depth>=d_from)&(depth<=d_to)&(val>-900)].max():.1f}")


if __name__ == "__main__":
    main()
