r"""
_bkz_probe.py — ЭМПИРИЧЕСКАЯ характеристика BKZ-скана БЕЗ vision-токенов:
сколько цветов-кривых, в одном треке или раздельных колонках, где края треков,
вертикальный охват данных. Чисто numpy/cv2, ничего не пишет.

python _bkz_probe.py <image.jpg>
"""
import sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import extract_instances as ei

def main(path):
    sys.stdout.reconfigure(encoding="utf-8")
    rgb = np.asarray(Image.open(path).convert("RGB"))
    H, W = rgb.shape[:2]
    print(f"image {W}x{H}")
    body = np.ones((H, W), np.uint8)
    ch = ei.classify_ink(rgb, body)
    nb = 24  # x-бины
    edges = np.linspace(0, W, nb + 1).astype(int)
    print(f"\nx-бины (ширина {W/nb:.0f}px):", " ".join(f"{edges[i]:>4}" for i in range(nb)))
    for col in ("black", "red", "blue", "green"):
        m = ch[col]
        tot = int(m.sum())
        if tot < 500:
            print(f"{col:6} total={tot:>9}  (пусто)")
            continue
        colprof = m.sum(0)  # на каждый x
        hist = [int(colprof[edges[i]:edges[i+1]].sum()) for i in range(nb)]
        mx = max(hist) or 1
        bar = "".join("#" if h > 0.15*mx else ("." if h > 0.02*mx else " ") for h in hist)
        xs = np.nonzero(colprof > 0.02*colprof.max())[0]
        print(f"{col:6} total={tot:>9}  x[{xs.min()}..{xs.max()}]  |{bar}|")
    # вертикальные длинные линии (края треков): чёрные столбцы с высоким покрытием по Y
    blk = ch["black"]
    colcov = blk.sum(0) / H
    vlines = np.nonzero(colcov > 0.45)[0]
    # сгруппировать соседние
    groups = []
    if len(vlines):
        s = vlines[0]; p = vlines[0]
        for x in vlines[1:]:
            if x - p > 8:
                groups.append((s, p)); s = x
            p = x
        groups.append((s, p))
    print(f"\nвертикальные линии (cov>45%): {[(int(a),int(b)) for a,b in groups]}")
    # вертикальный охват данных: строки с любым цветным чернилом
    anycol = (ch["red"] | ch["blue"] | ch["green"]).sum(1)
    anyink = (ch["black"] | ch["red"] | ch["blue"] | ch["green"]).sum(1)
    rows_data = np.nonzero(anyink > 0.02*anyink.max())[0]
    rows_color = np.nonzero(anycol > 0)[0]
    print(f"строки с чернилом: y[{rows_data.min()}..{rows_data.max()}] (всего H={H})")
    if len(rows_color):
        print(f"строки с ЦВЕТНЫМ: y[{rows_color.min()}..{rows_color.max()}]")

    # СКОЛЬКО КРИВЫХ: число прогонов цветного чернила в строке (в теле, не в шапке)
    def runs_in_row(rowmask, gap=6):
        xs = np.nonzero(rowmask)[0]
        if not len(xs):
            return 0
        return 1 + int((np.diff(xs) > gap).sum())
    y0, y1 = int(0.10 * H), int(0.98 * H)
    for col in ("blue", "red", "black"):
        m = ch[col]
        rc = np.array([runs_in_row(m[y] > 0) for y in range(y0, y1, 3)])
        nz = rc[rc > 0]
        if not len(nz):
            print(f"runs/строка {col:6}: нет"); continue
        import numpy as _np
        print(f"runs/строка {col:6}: медиана={int(_np.median(nz))} p90={int(_np.percentile(nz,90))} "
              f"max={nz.max()} | доля строк с чернилом={len(nz)/len(rc)*100:.0f}% "
              f"| гистограмма#прогонов={ {int(k):int((nz==k).sum()) for k in _np.unique(nz)[:8]} }")

if __name__ == "__main__":
    main(sys.argv[1])
