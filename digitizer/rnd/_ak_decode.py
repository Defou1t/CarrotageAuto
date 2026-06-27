r"""
_ak_decode.py — ПУТЬ B: декод оборотов на маске СТАРОЙ модели (ak_prob.npy).
  • маска U-Net гейтит грид → берём СЫРОЕ чернило внутри (толщина сохранена);
  • MGZ толстая / MPZ тонкая по ширине;
  • WRAP-AWARE трекер: уход за правый край трека + вход слева = тот же логический штрих, уровень+1 (×5).
Пишет nlgx с сегментами-уровнями + оверлей (цвет по уровню: база зелёный / ×5 синий / ×25 красный).

python _ak_decode.py [Wsplit] [thr]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import struct
from extract_nlgx import extract, NULL
from write_nlgx import read_full, write_full, write_bck, set_tag

FRAME = r"F:\nds\projects\Pn_Zavoda_001\wlg\Pn_Zavoda_1_AK_5508_5716_200_D1.nlgx"
PROB = r"F:\nds\output\test_single\ak_prob.npy"
OUT = Path(r"F:\nds\output\test_single")
LVLCOL = {0: (0, 170, 0), 1: (0, 90, 255), 2: (230, 0, 0), 3: (200, 0, 200)}


def runs(rowmask, gap=2):
    xs = np.nonzero(rowmask)[0]
    if not len(xs):
        return []
    return [(float(s.mean()), len(s)) for s in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1)]


def wrap_track(cents, ty, by, x0, xL, xR, gate=14, slmax=18, Lmax=2):
    """КОНСЕРВАТИВНО: тугой гейт (ведём ОДНУ непрерывную линию, не прыгаем на дальние), коаст в
    разрывах; оборот ТОЛЬКО у реального края (x достиг края + кандидат у противоположного), ≤Lmax,
    с дебаунсом. cents[y]=list(xc). Возвращает tr, lv."""
    tr = {}; lv = {}; x = v = None; L = 0; lost = 0; last_wrap = -9999
    for y in range(ty, by):
        cs = cents.get(y, [])
        if not cs:
            if x is not None:
                x = x + float(np.clip(v, -slmax, slmax)); lost += 1
            continue
        if x is None:
            x = float(min(cs, key=lambda c: abs(c - x0))); v = 0.0; tr[y] = x; lv[y] = L; continue
        pred = x + float(np.clip(v, -slmax, slmax))
        nx = float(min(cs, key=lambda c: abs(c - pred)))
        if abs(nx - pred) <= gate:
            v = 0.6 * v + 0.4 * (nx - x); x = nx; lost = 0             # ведём непрерывную линию
        else:
            leftc = [c for c in cs if c < xL + 55]; rightc = [c for c in cs if c > xR - 55]
            if x > xR - 35 and leftc and L < Lmax and y - last_wrap > 40:
                x = float(min(leftc, key=lambda c: abs(c - xL))); v = 0.0; L += 1; last_wrap = y
            elif x < xL + 35 and rightc and L > 0 and y - last_wrap > 40:
                x = float(min(rightc, key=lambda c: abs(c - xR))); v = 0.0; L -= 1; last_wrap = y
            else:                                                      # не прыгаем на дальнюю — коаст
                x = x + float(np.clip(v, -slmax, slmax)); lost += 1
                if lost > 120:
                    x = nx; v = 0.0; lost = 0                          # реально потеряли → пере-захват
        tr[y] = x; lv[y] = L
    return tr, lv


def build_segs(tr, lv, ty, n, W):
    xs = [NULL] * n; segs = []; cur = None; s = None
    for i in range(n):
        y = ty + i
        if y in tr:
            xv = int(round(tr[y]))
            xs[i] = xv if 0 <= xv < W else NULL          # за краем картинки → нет данных
        L = lv.get(y)
        if L is None:
            if cur is not None:
                segs.append((s, ty + i - 1, cur)); cur = None
            continue
        if L != cur:
            if cur is not None:
                segs.append((s, y - 1, cur))
            cur = L; s = y
    if cur is not None:
        segs.append((s, ty + n - 1, cur))
    return xs, segs


def inject(ifds, short, xs, segs):
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]
            xs = (xs + [NULL] * n)[:n]
            set_tag(ifds, i, 35490, 4, xs)
            if segs:
                set_tag(ifds, i, 35492, 4, [len(segs)])
                set_tag(ifds, i, 35494, 4, [s for s, e, l in segs])
                set_tag(ifds, i, 35496, 4, [e for s, e, l in segs])
                set_tag(ifds, i, 35498, 4, [l for s, e, l in segs])
            vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
            if vx:
                ty0 = struct.unpack("<I", tags[35474][:4])[0]
                rws = [ty0 + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, val in [(35478, min(xv)), (35480, min(rws)), (35482, max(xv)), (35484, max(rws))]:
                    set_tag(ifds, i, t, 4, [val])
            return sum(1 for x in xs if x != NULL), len(segs)
    return 0, 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    Wsplit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    prob = np.load(PROB)
    m = extract(FRAME); H, W = prob.shape
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    sa = m["scale_axes"][0]; xL, xR = sa["x_left"], sa["x_right"]
    g = np.asarray(Image.open(m["img_path"]).convert("L")); bg = int(np.median(g))
    xlo, xhi = max(0, xL - 30), min(W, xR + 120)
    ink = (g < bg - 45) & (prob > thr)                         # сырое чернило, гейт старой моделью
    thick = {}; thin = {}
    for y in range(ty, by):
        for xc, w in runs(ink[y, xlo:xhi]):
            (thick if w >= Wsplit else thin).setdefault(y, []).append(xlo + xc)
    cx = (xL + xR) / 2
    tg, lg = wrap_track(thick, ty, by, cx, xL, xR)
    tn, ln = wrap_track(thin, ty, by, cx, xL, xR)
    nseg_g = len(set(lg.values())); nseg_n = len(set(ln.values()))
    print(f"Wsplit={Wsplit} | MGZ: {len(tg)} точек, уровни до {max(lg.values()) if lg else 0} | "
          f"MPZ: {len(tn)} точек, уровни до {max(ln.values()) if ln else 0}")

    ifds = read_full(open(FRAME, "rb").read())
    n = [c["n_rows"] for c in m["curves"] if c["name"].split()[0] == "MGZ1"][0]
    xg, sg = build_segs(tg, lg, ty, n, W); xn, sn = build_segs(tn, ln, ty, n, W)
    a, b = inject(ifds, "MGZ1", xg, sg); c, d = inject(ifds, "MPZ1", xn, sn)
    data = write_full(ifds); stem = Path(FRAME).stem
    open(OUT / f"{stem}_decode.nlgx", "wb").write(data)
    open(OUT / f"{stem}_decode.bck", "wb").write(write_bck(data))
    print(f"MGZ {a} точек/{b} сегм | MPZ {c} точек/{d} сегм -> F:\\nds\\output\\test_single\\{stem}_decode.nlgx")

    img = Image.open(m["img_path"]).convert("RGB")
    for tag, dc in [("a", 5540.0), ("b", 5600.0), ("c", 5660.0)]:
        yc = int(ty + (dc - da["top_depth"]) / (da["bottom_depth"] - da["top_depth"]) * (by - ty))
        y0, y1 = max(0, yc - 220), min(H, yc + 220)
        crop = img.crop((0, y0, W, y1)).convert("RGB"); dr = ImageDraw.Draw(crop)
        for tr, lv in [(tg, lg)]:                          # только MGZ (MPZ тонкая пока шумна)
            pts = sorted((y, x) for y, x in tr.items() if y0 <= y < y1)
            for (y0a, x0a), (y1a, x1a) in zip(pts, pts[1:]):
                if abs(y1a - y0a) <= 3:
                    dr.line([(x0a, y0a - y0), (x1a, y1a - y0)], fill=LVLCOL.get(lv.get(y0a, 0), (150, 150, 150)), width=2)
        crop.save(OUT / f"decode_crop_{tag}.png")
    print("оверлей (цвет=уровень: база зелёный/×5 синий/×25 красный) -> F:\\nds\\output\\test_single\\decode_crop_[a,b,c].png")


if __name__ == "__main__":
    main()
