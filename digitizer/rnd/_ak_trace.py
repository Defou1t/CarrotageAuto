r"""
_ak_trace.py — MGZ(толстая, центроид) + MPZ(тонкая) в РЕАЛЬНУЮ рамку Pn_Zavoda.
Кривые ЧЁРНЫЕ (как сетка) → разделяем структурно:
  • грид-сетку давим: постоянные ВЕРТ. столбцы (клетки) + ГОР. линии (полнострочные) вычитаем;
  • MGZ = чёрные прогоны ШИРЕ грид-линии (Wmin..Wmax) → центроид; MPZ = тонкие, не на грид-столбцах.
Пер-роу центроид-трекер по непрерывности. БАЗА (5× обороты → спайки, для оценки).

python _ak_trace.py [Wmin] [Wmax] [darkΔ]  -> F:\nds\output\test_single\<stem>_auto.nlgx(+bck)+кропы
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd
from digitize_b3 import curve_pred

FRAME = r"F:\nds\projects\Pn_Zavoda_001\wlg\Pn_Zavoda_1_AK_5508_5716_200_D1.nlgx"
OUT = Path(r"F:\nds\output\test_single")


def runs(rowmask, gap=2):
    xs = np.nonzero(rowmask)[0]
    if not len(xs):
        return []
    return [(int(s.mean()), len(s), int(s[0]), int(s[-1])) for s in
            np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1)]


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
    import struct
    idxs = find_ifd(ifds, curve_pred(short))
    if not idxs:
        print(f"  слот {short} не найден"); return 0
    i0 = idxs[0]
    tags = {t: raw for t, typ, c, raw in ifds[i0]["entries"]}
    n = struct.unpack("<I", tags[35488][:4])[0]          # длина = n_rows слота (иначе краш)
    xs = [int(round(tr[ty + i])) if (ty + i) in tr else NULL for i in range(n)]
    set_tag(ifds, i0, 35490, 4, xs)
    set_tag(ifds, i0, 35492, 4, [1]); set_tag(ifds, i0, 35494, 4, [ty])
    set_tag(ifds, i0, 35496, 4, [ty + n - 1]); set_tag(ifds, i0, 35498, 4, [0])
    vx = [(i, x) for i, x in enumerate(xs) if x != NULL]
    if vx:
        rws = [ty + i for i, _ in vx]; xv = [x for _, x in vx]
        for t, val in [(35478, min(xv)), (35480, min(rws)), (35482, max(xv)), (35484, max(rws))]:
            set_tag(ifds, i0, t, 4, [val])
    return len(vx)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    Wmin = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    Wmax = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    dD = int(sys.argv[3]) if len(sys.argv) > 3 else 55
    OUT.mkdir(parents=True, exist_ok=True)
    m = extract(FRAME)
    g = np.asarray(Image.open(m["img_path"]).convert("L")); H, W = g.shape
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    sa = m["scale_axes"][0]; xl, xr = sa["x_left"], sa["x_right"]
    xlo, xhi = max(0, xl - 25), min(W, xr + 110)
    bg = int(np.median(g)); dark = (g < bg - dD)

    band = dark[ty:by, xlo:xhi]
    col_cov = band.mean(0)                       # доля тёмного по столбцу (грид-вертикали высокие)
    grid_col = col_cov > 0.45
    row_cov = band.mean(1)                       # доля тёмного по строке (грид-горизонтали ~1.0)
    print(f"грид-столбцов(клетки) подавлено: {int(grid_col.sum())} из {xhi-xlo}; "
          f"строк-горизонталей(>0.7): {int((row_cov>0.7).sum())}")

    thick = {}; thin = {}; wid_hist = []
    for j, y in enumerate(range(ty, by)):
        if row_cov[j] > 0.7:                     # горизонтальная грид-линия → пропуск
            continue
        rr = runs(band[j])
        for xc, w, a, b in rr:
            wid_hist.append(w)
            if Wmin <= w <= Wmax:
                thick.setdefault(y, []).append(xlo + xc)          # MGZ
            elif w < Wmin and not grid_col[xc]:                   # MPZ тонкая, не на клетке
                thin.setdefault(y, []).append(xlo + xc)
    wh = np.array(wid_hist)
    h, e = np.histogram(wh, bins=[1, 2, 3, 4, 5, 6, 8, 11, 15, 25, 45, 200])
    print("чёрные ширины:", {f'{e[i]}-{e[i+1]-1}': int(h[i]) for i in range(len(h))})
    nthk = np.array([len(thick.get(y, [])) for y in range(ty, by)])
    nthn = np.array([len(thin.get(y, [])) for y in range(ty, by)])
    print(f"MGZ(толст {Wmin}-{Wmax}px)/строку медиана={int(np.median(nthk[nthk>0])) if (nthk>0).any() else 0} "
          f"покрытие={np.mean(nthk>0)*100:.0f}% | MPZ(тонк)/строку медиана={int(np.median(nthn[nthn>0])) if (nthn>0).any() else 0} "
          f"покрытие={np.mean(nthn>0)*100:.0f}%")

    cx = (xl + xr) / 2
    tr_mgz = track(thick, ty, by, cx); tr_mpz = track(thin, ty, by, cx)
    ifds = read_full(open(FRAME, "rb").read())
    nm = inject(ifds, "MGZ1", tr_mgz, ty); np_ = inject(ifds, "MPZ1", tr_mpz, ty)
    data = write_full(ifds); stem = Path(FRAME).stem
    open(OUT / f"{stem}_auto.nlgx", "wb").write(data)
    open(OUT / f"{stem}_auto.bck", "wb").write(write_bck(data))
    print(f"MGZ {nm} точек | MPZ {np_} точек -> F:\\nds\\output\\test_single\\{stem}_auto.nlgx")

    img = Image.open(m["img_path"]).convert("RGB")
    for tag, dc in [("a", 5540.0), ("b", 5600.0), ("c", 5660.0)]:
        yc = int(ty + (dc - da["top_depth"]) / (da["bottom_depth"] - da["top_depth"]) * (by - ty))
        y0, y1 = max(0, yc - 220), min(H, yc + 220)
        crop = img.crop((0, y0, W, y1)).convert("RGB"); d = ImageDraw.Draw(crop)
        for tr, col in [(tr_mgz, (0, 165, 0)), (tr_mpz, (255, 0, 200))]:
            pts = [(int(x), y - y0) for y, x in sorted(tr.items()) if y0 <= y < y1]
            for p0, p1 in zip(pts, pts[1:]):
                if abs(p1[1] - p0[1]) <= 2:
                    d.line([p0, p1], fill=col, width=2)
        crop.save(OUT / f"{stem}_crop_{tag}.png")
    print("кропы: MGZ зелёный, MPZ малиновый -> F:\\nds\\output\\test_single\\")


if __name__ == "__main__":
    main()
