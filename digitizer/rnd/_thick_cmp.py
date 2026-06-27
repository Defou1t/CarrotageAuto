r"""
_thick_cmp.py — измерить, РАЗЛИЧАЕТ ли маска модели толщину (тонкая MPZ vs толстая MGZ).
Берёт prob.npy (карта модели на AK), считает ширины прогонов маски в треке. Бимодальность
(заметная доля тонких 1-4px И толстых ≥8px) = модель сохраняет толщину; сплошь широкие = блобит.

python _thick_cmp.py <prob.npy> <метка>
"""
import sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract

FRAME = r"F:\nds\projects\Pn_Zavoda_001\wlg\Pn_Zavoda_1_AK_5508_5716_200_D1.nlgx"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    prob = np.load(sys.argv[1]); label = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    m = extract(FRAME); H, W = prob.shape
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    sa = m["scale_axes"][0]; xlo, xhi = max(0, sa["x_left"] - 30), min(W, sa["x_right"] + 120)
    mask = prob[ty:by, xlo:xhi] > 0.4
    wid = []
    for row in mask:
        xs = np.nonzero(row)[0]
        if len(xs):
            for s in np.split(xs, np.nonzero(np.diff(xs) > 2)[0] + 1):
                wid.append(len(s))
    wid = np.array(wid)
    if not len(wid):
        print(f"[{label}] маска пуста"); return
    h, e = np.histogram(wid, bins=[1, 3, 5, 8, 12, 18, 30, 60, 300])
    thin = np.mean(wid <= 4) * 100; thick = np.mean(wid >= 8) * 100
    print(f"[{label}] прогонов={len(wid)} медиана_ширины={int(np.median(wid))}px | "
          f"тонких(≤4px)={thin:.0f}%  толстых(≥8px)={thick:.0f}%")
    print("   гистограмма ширин:", {f"{e[i]}-{e[i+1]-1}": int(h[i]) for i in range(len(h))})
    verdict = "БИМОДАЛЬНО (различает толщину)" if (thin >= 20 and thick >= 20) else \
              "блобит (толщину НЕ различает)" if thick >= 60 else "в основном тонко"
    print(f"   ВЕРДИКТ: {verdict}")


if __name__ == "__main__":
    main()
