r"""
_bkz_frame.py — LIGHT-рамка для автономного BKZ («база-линия»): берёт скелет
MBK-done (1 Depth Axis + 1 Scale Axis + 1 слот кривой), патчит ГЕОМЕТРИЮ под
конкретный BKZ-скан:
  • top_y/bottom_y/x_left/x_right — из СИНЕГО штриха (кривая BKZ = синяя);
  • top_depth/bottom_depth — из имени файла (_3160_3570_);
  • depth-горизонтали ПРОПУЩЕНЫ (эксперт правит при QC, приоритет §6.6.9);
  • слот кривой переименован MBK1 -> BKZ1.
Печатает детект для контроля. Ничего не трассирует — это вход для digitize_auto.

python _bkz_frame.py <bkz_image.jpg> [--out DIR] [--skel <nlgx>] [--name BKZ1]
-> <out>/<stem>.frame.nlgx
"""
import sys, re
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import extract_instances as ei
import write_nlgx as wn

SKEL = r"F:\nds\projects\Yatskivska_001\wlg\Yatskivska_1_MBK_3180_3580_200_D1.nlgx"


def depth_from_name(stem):
    m = re.search(r"_(\d{3,5})_(\d{3,5})_\d+_D", stem)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def detect_geom(rgb, color="blue"):
    """top_y/bottom_y/x_left/x_right по полосе, держащей 99% синего штриха (робастно к краевому шуму)."""
    H, W = rgb.shape[:2]
    ch = ei.classify_ink(rgb, np.ones((H, W), np.uint8))
    m = ch[color]
    rows = m.sum(1).astype(np.float64)
    cols = m.sum(0).astype(np.float64)
    def band(prof, lo=0.005, hi=0.995):
        c = np.cumsum(prof); tot = c[-1]
        if tot <= 0:
            return 0, len(prof) - 1
        return int(np.searchsorted(c, lo * tot)), int(np.searchsorted(c, hi * tot))
    top_y, bottom_y = band(rows)
    x_left, x_right = band(cols)
    tot = {k: int(v.sum()) for k, v in ch.items()}
    return top_y, bottom_y, x_left, x_right, tot


def patch(tags_pred, ifds, sets):
    idxs = wn.find_ifd(ifds, tags_pred)
    if not idxs:
        raise SystemExit(f"IFD не найден: {tags_pred}")
    for tag, typ, v in sets:
        wn.set_tag(ifds, idxs[0], tag, typ, v)
    return idxs[0]


def build(img, out=None, skel=SKEL, name="BKZ1"):
    img = str(img); stem = Path(img).stem
    out = Path(out) if out else Path(r"F:\nds\output")
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(img).convert("RGB")); H, W = rgb.shape[:2]
    td, bd = depth_from_name(stem)
    ty, by, xl, xr, tot = detect_geom(rgb)
    nrows = by - ty + 1
    print(f"{stem}\n  image {W}x{H}; ink totals {tot}")
    print(f"  depth(имя) {td}..{bd}; track y[{ty}..{by}] (n={nrows}) x[{xl}..{xr}]")
    if td is None:
        raise SystemExit("не распарсил глубины из имени")

    ifds = wn.read_full(open(skel, "rb").read())
    L, D = 4, 12  # LONG, DOUBLE
    # doc: только img_path (35006/35008 у рабочего MBK-done нет — не добавляем, минимум отклонений)
    patch(lambda t: 34878 in t, ifds, [(34878, 2, img)])
    # Depth Axis (35180=1 — флаг «ось активна», как у рабочих; у MBK-скелета 0)
    di = patch(lambda t: 35184 in t and 35190 in t, ifds,
          [(35184, L, ty), (35188, L, by), (35190, D, td), (35192, D, bd),
           (35194, D, bd - td), (35196, L, by - ty), (35180, L, 1)])
    # Scale Axis (x трека; y-линейки 35294/35298/35307 — у ВЕРХА трека, согласованно)
    patch(lambda t: 35292 in t and 35300 in t, ifds,
          [(35292, L, xl), (35296, L, xr), (35308, L, xr - xl),
           (35294, L, ty), (35298, L, ty), (35307, L, ty)])
    # Curve-слоты: ПЕРЕСОБРАТЬ массив 35490 под новую длину (иначе n_rows!=len(35490) → краш NeuraLOG).
    # DA1 (само-ось) = NULL + эндпойнты x_top/x_bot; BKZ1 = всё NULL (digitize_auto заполнит). Сегмент согласован.
    NULL = 0xFFFFFFFF
    import struct as _st
    dt = {t: raw for t, typ, c, raw in ifds[di]["entries"]}
    x_top = _st.unpack("<I", dt[35182][:4])[0] if 35182 in dt else xl
    x_bot = _st.unpack("<I", dt[35186][:4])[0] if 35186 in dt else xl
    for ifd_i, ifd in enumerate(ifds):
        tg = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tg and 35488 in tg:
            nm = tg[35470].split(b"\x00")[0].decode("latin1")
            short = nm.split()[0]
            # bbox трассы 35478/80/82/84: y=[ty..by]; x=[xl..xr] для данных, [x_top..x_bot] для DA1.
            # (для слота данных digitize_auto перезапишет x/y по ФАКТ. трассе — здесь провизорно-согласованно)
            bxl, bxr = (min(x_top, x_bot), max(x_top, x_bot)) if short.startswith("DA") else (xl, xr)
            for tag, v in [(35474, ty), (35476, by), (35488, nrows),
                           (35478, bxl), (35480, ty), (35482, bxr), (35484, by),
                           (35492, 1), (35494, ty), (35496, by), (35498, 0)]:
                wn.set_tag(ifds, ifd_i, tag, L, v)
            if short.startswith("DA"):
                arr = [NULL] * nrows
                arr[0] = x_top; arr[-1] = x_bot
                wn.set_tag(ifds, ifd_i, 35490, L, arr)
            else:
                wn.set_tag(ifds, ifd_i, 35490, L, [NULL] * nrows)
                newnm = name + nm[len(short):]      # 'MBK1 DA1 SA1' -> 'BKZ1 DA1 SA1'
                wn.set_tag(ifds, ifd_i, 35470, 2, newnm)
                print(f"  слот '{nm}' -> '{newnm}'")
    data = wn.write_full(ifds)
    dst = out / f"{stem}.frame.nlgx"
    open(dst, "wb").write(data)
    print(f"-> {dst}")
    return str(dst)


def main():
    a = sys.argv[1:]
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if not a:
        print(__doc__); return
    out = a[a.index("--out") + 1] if "--out" in a else None
    skel = a[a.index("--skel") + 1] if "--skel" in a else SKEL
    name = a[a.index("--name") + 1] if "--name" in a else "BKZ1"
    build(a[0], out, skel, name)


if __name__ == "__main__":
    main()
