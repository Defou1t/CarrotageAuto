r"""
_ak_mask_track.py — трекер ПО МАСКЕ U-Net (ak_prob.npy), не по сырому шуму.
Сеть уже убрала грид → разделяем по ТОЛЩИНЕ маски: MGZ толстая (центроид) / MPZ тонкая,
ведём каждую по непрерывности. Инъекция в РЕАЛЬНУЮ рамку Pn_Zavoda + кропы.

python _ak_mask_track.py [thr] [Wmin] [Wmax]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import struct
from extract_nlgx import extract, NULL
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd

FRAME = r"F:\nds\projects\Pn_Zavoda_001\wlg\Pn_Zavoda_1_AK_5508_5716_200_D1.nlgx"
OUT = Path(r"F:\nds\output\test_single")


def runs(rowmask, gap=2):
    xs = np.nonzero(rowmask)[0]
    if not len(xs):
        return []
    return [(float(s.mean()), len(s)) for s in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1)]


def track(cents_by_row, ty, by, x0, slmax=16):
    tr = {}; x = v = None
    for y in range(ty, by):
        cs = cents_by_row.get(y, [])
        if not cs:
            if x is not None:
                x = x + float(np.clip(v, -slmax, slmax))
            continue
        if x is None:
            x = float(min(cs, key=lambda c: abs(c - x0))); v = 0.0; tr[y] = x; continue
        pred = x + float(np.clip(v, -slmax, slmax))
        nx = float(min(cs, key=lambda c: abs(c - pred)))
        v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = x
    return tr


def inject(ifds, short, tr, ty):
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]
            xs = [int(round(tr[ty + j])) if (ty + j) in tr else NULL for j in range(n)]
            set_tag(ifds, i, 35490, 4, xs)
            for t, v in [(35492, 1), (35494, ty), (35496, ty + n - 1), (35498, 0)]:
                set_tag(ifds, i, t, 4, [v])
            vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
            if vx:
                rws = [ty + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, v in [(35478, min(xv)), (35480, min(rws)), (35482, max(xv)), (35484, max(rws))]:
                    set_tag(ifds, i, t, 4, [v])
            return len(vx)
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    Wmin = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    Wmax = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    prob = np.load(OUT / "ak_prob.npy")
    m = extract(FRAME); H, W = prob.shape
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    sa = m["scale_axes"][0]; xl, xr = sa["x_left"], sa["x_right"]
    xlo, xhi = max(0, xl - 25), min(W, xr + 110)
    # ТОЛЩИНА по СЫРОМУ штриху, но ГЕЙТ маской U-Net (она убирает грид). Маска шире штриха →
    # берём тёмное чернило ВНУТРИ зоны, где сеть уверена (prob>thr_lo).
    g = np.asarray(Image.open(m["img_path"]).convert("L")); bg = int(np.median(g))
    mask = (g < bg - 45) & (prob > min(thr, 0.25))
    thick = {}; thin = {}; wh = []
    for y in range(ty, by):
        for xc, w in runs(mask[y, xlo:xhi]):
            wh.append(w)
            if Wmin <= w <= Wmax:
                thick.setdefault(y, []).append(xlo + xc)
            elif w < Wmin:
                thin.setdefault(y, []).append(xlo + xc)
    wh = np.array(wh); h, e = np.histogram(wh, bins=[1, 2, 3, 4, 5, 6, 8, 11, 15, 25, 45, 200])
    print(f"thr={thr} Wmin={Wmin} | ширины маски:", {f'{e[i]}-{e[i+1]-1}': int(h[i]) for i in range(len(h))})
    nthk = np.array([len(thick.get(y, [])) for y in range(ty, by)])
    nthn = np.array([len(thin.get(y, [])) for y in range(ty, by)])
    print(f"MGZ(толст)/строку медиана={int(np.median(nthk[nthk>0])) if (nthk>0).any() else 0} покрытие={np.mean(nthk>0)*100:.0f}% | "
          f"MPZ(тонк)/строку медиана={int(np.median(nthn[nthn>0])) if (nthn>0).any() else 0} покрытие={np.mean(nthn>0)*100:.0f}%")
    cx = (xl + xr) / 2
    tr_mgz = track(thick, ty, by, cx); tr_mpz = track(thin, ty, by, cx)

    ifds = read_full(open(FRAME, "rb").read())
    nm = inject(ifds, "MGZ1", tr_mgz, ty); npz = inject(ifds, "MPZ1", tr_mpz, ty)
    data = write_full(ifds)
    open(OUT / "Pn_Zavoda_1_AK_5508_5716_200_D1_maskauto.nlgx", "wb").write(data)
    open(OUT / "Pn_Zavoda_1_AK_5508_5716_200_D1_maskauto.bck", "wb").write(write_bck(data))
    print(f"MGZ {nm} | MPZ {npz} точек -> F:\\nds\\output\\test_single\\..._maskauto.nlgx")

    img = Image.open(m["img_path"]).convert("RGB")
    for tag, dc in [("a", 5540.0), ("b", 5600.0), ("c", 5660.0)]:
        yc = int(ty + (dc - da["top_depth"]) / (da["bottom_depth"] - da["top_depth"]) * (by - ty))
        y0, y1 = max(0, yc - 220), min(H, yc + 220)
        crop = img.crop((0, y0, W, y1)).convert("RGB"); d = ImageDraw.Draw(crop)
        for tr, col in [(tr_mgz, (0, 165, 0)), (tr_mpz, (255, 0, 200))]:
            pts = [(int(x), y - y0) for y, x in sorted(tr.items()) if y0 <= y < y1]
            for p0, p1 in zip(pts, pts[1:]):
                if abs(p1[1] - p0[1]) <= 3:
                    d.line([p0, p1], fill=col, width=2)
        crop.save(OUT / f"maskauto_crop_{tag}.png")
    print("кропы MGZ зелёный/MPZ малиновый -> F:\\nds\\output\\test_single\\maskauto_crop_[a,b,c].png")


if __name__ == "__main__":
    main()
