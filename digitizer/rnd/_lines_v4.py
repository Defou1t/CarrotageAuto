r"""
_lines_v4.py — ПОЛНЫЙ ТРАССЕР инстансов (продукт): счёт+цвет (v3) → трасса per-линия по ВСЕЙ рамке
(scale-axis range, без узкого SA-клипа), скользящим окном (лево/право-выход), цвет/prob-гейт,
ДЕТЕКТ 5х-оборотов на краях рамки (level-сегменты), структура (рамки/оси) вырезана → не прыгает.
Инъекция в КОПИЮ → stkds_out. Цвет per-инстанс (доминантный hue). Чёрные ведём по prob (сквозь сетку).

python _lines_v4.py <prob.npy> <nlgx>   (nlgx — рамки+слоты инъекции; image-first по сути)
"""
import sys, struct
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from dataset_build import find_image
from write_nlgx import read_full, write_full, write_bck, set_tag

OUT = Path(r"F:\nds\output\stkds_out")
VIS = {"black": (20, 20, 20), "green": (0, 160, 0), "red": (220, 0, 0), "blue": (40, 60, 210)}
SL, GATE = 26.0, 26.0


def sa_for(curve, m):
    suf = " ".join(curve["name"].split()[1:])
    for s in m["scale_axes"]:
        if s["name"].endswith(suf):
            return int(s["x_left"]), int(s["x_right"])
    s = m["scale_axes"][0]; return int(s["x_left"]), int(s["x_right"])


def structure_cols(gray, m, ty, by, W):
    """столбцы рамок/осей (известные края ±3 + сверх-устойчивые вертикали) — вырезать из источника."""
    ink = gray < 150; excl = np.zeros(W, bool)
    xb = set()
    for s in m["scale_axes"]:
        xb.add(int(s["x_left"])); xb.add(int(s["x_right"]))
    da = m["depth_axis"]
    for k in ("x_top", "x_bot"):
        if k in da:
            xb.add(int(da[k]))
    for x in xb:
        excl[max(0, x - 3):min(W, x + 4)] = True
    colf = ink[ty:by].mean(0)
    excl |= (colf > 0.80)
    return excl


def cmasks(arr, gray):
    R, G, Bl = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    red = (R - G > 22) & (R - Bl > 12) & (gray < 150)
    green = (G - R > 8) & (G - Bl > -3) & (gray < 158) & (~red)
    black = (gray < 110) & (~red) & (~green)
    return {"red": red, "green": green, "black": black}


def runs_centroids(rowmask, weight, xa, xb, gap=3):
    seg = rowmask[xa:xb]; xs = np.nonzero(seg)[0]
    if not len(xs):
        return []
    cols = np.arange(xa, xb); out = []
    for r in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1):
        w = weight[xa:xb][r]
        out.append(float((cols[r] @ w / w.sum()) if w.sum() > 0 else cols[r].mean()))
    return out


def trace_instance(src, weight, fxL, fxR, ty, by, x0):
    """скользящее окно + непрерывность (min-jerk) ПО ВСЕЙ рамке; ОБОРОТ на краю рамки → level±1.
    Возвращает {y:x}, segments[(start_y,end_y,level)]."""
    out = {}; x = float(x0); v = 0.0; level = 0
    segs = []; seg_start = ty; W_ = fxR - fxL; last_wrap = -10**9
    for y in range(ty, by):
        rc = runs_centroids(src[y], weight[y], fxL, fxR)
        pred = x + float(np.clip(v, -SL, SL))
        near = [c for c in rc if abs(c - pred) <= GATE]
        deb = (y - last_wrap) > 120                          # дебаунс: оборот не чаще ~раза в 120 строк
        if near:
            nx = min(near, key=lambda c: abs(c - pred))
            v = 0.6 * v + 0.4 * (nx - x); x = nx; out[y] = x
        elif rc and deb and pred >= fxR - 3 and x >= fxR - 0.18 * W_:   # реально у ПРАВОГО края → оборот
            left = [c for c in rc if c <= fxL + 0.30 * W_]
            if left:
                segs.append((seg_start, y - 1, level)); level += 1; seg_start = y
                x = min(left); v = 0.0; out[y] = x; last_wrap = y
        elif rc and deb and pred <= fxL + 3 and x <= fxL + 0.18 * W_:   # реально у ЛЕВОГО края
            right = [c for c in rc if c >= fxR - 0.30 * W_]
            if right:
                segs.append((seg_start, y - 1, level)); level -= 1; seg_start = y
                x = max(right); v = 0.0; out[y] = x; last_wrap = y
        else:
            x = float(np.clip(pred, fxL, fxR)); v *= 0.6    # коаст внутри рамки, БЕЗ эмита
    segs.append((seg_start, by - 1, level))
    return out, [s for s in segs if s[1] - s[0] > 20]


def color_of(arr, gray, tr, bg):
    px = [arr[y, int(round(x))] for y, x in tr.items()
          if 0 <= y < arr.shape[0] and 0 <= int(round(x)) < arr.shape[1] and gray[y, int(round(x))] < bg - 20]
    if len(px) < 10:
        return "black"
    px = np.array(px); R, G, B = np.median(px[:, 0]), np.median(px[:, 1]), np.median(px[:, 2])
    if R - G > 22 and R - B > 12: return "red"
    if G - R > 8 and G - B > -3: return "green"
    if B - R > 12 and B - G > 4: return "blue"
    return "black"


def inject(ifds, short, tr, segs, W):
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]; ty0 = struct.unpack("<I", tags[35474][:4])[0]
            xs = [int(round(tr[ty0 + j])) if (ty0 + j) in tr and 0 <= round(tr[ty0 + j]) < W else NULL for j in range(n)]
            set_tag(ifds, i, 35490, 4, xs)
            if segs:                                       # уровни → ≥0 (тег беззнаковый)
                lv0 = min(s[2] for s in segs); segs = [(s, e, l - lv0) for s, e, l in segs]
            ns = len(segs) if segs else 1
            set_tag(ifds, i, 35492, 4, [ns])
            set_tag(ifds, i, 35494, 4, [s[0] for s in segs] if segs else [ty0])
            set_tag(ifds, i, 35496, 4, [s[1] for s in segs] if segs else [ty0 + n - 1])
            set_tag(ifds, i, 35498, 4, [s[2] for s in segs] if segs else [0])
            vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
            if vx:
                rw = [ty0 + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, v in [(35478, min(xv)), (35480, min(rw)), (35482, max(xv)), (35484, max(rw))]:
                    set_tag(ifds, i, t, 4, [v])
            return sum(1 for x in xs if x != NULL)
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    probpath, F = sys.argv[1], sys.argv[2]
    prob = np.load(probpath)
    m = extract(F); img = find_image(Path(F)) or m["img_path"]
    arr = np.asarray(Image.open(img).convert("RGB")); H, W = arr.shape[:2]
    gray = np.asarray(Image.open(img).convert("L")); bg = int(np.median(gray[::7, ::7]))
    prob = prob[:H, :W]
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, prob.shape[0], da["bottom_y"])
    mk = cmasks(arr, gray); excl = structure_cols(gray, m, ty, by, W)
    darkw = (bg - gray.astype(np.int16)).clip(0).astype(np.float32)
    for c in ("red", "green", "black"):
        mk[c][:, excl] = False
    probg = prob.copy(); probg[:, excl] = 0
    def name_color(s):
        if s.startswith("GZ") or s.startswith("OGZ"): return "green"
        if s.startswith("SP"): return "red"
        return "black"                                     # PZ/DS/GK/NGK/резистив — чёрные (V<110 отделит)
    curves = [c for c in m["curves"] if not c["name"].split()[0].startswith("DA")]
    ifds = read_full(open(F, "rb").read()); traces = {}
    print(f"{Path(F).name[:44]} | y[{ty}..{by}] (цвет по имени, трасса по цвето-маске)")
    for c in curves:
        short = c["name"].split()[0]; fxL, fxR = sa_for(c, m); fxR = min(W, fxR)
        col = name_color(short); src = mk[col]; wt = darkw * mk[col]
        dens = np.convolve(src[ty:by, fxL:fxR].sum(0).astype(float), np.ones(15) / 15, mode="same")
        if dens.max() <= 0:
            print(f"  {short:5} рамка[{fxL}..{fxR}] цвет={col}: ПУСТО"); continue
        x0 = fxL + int(np.argmax(dens))                    # сид = пик плотности цвето-маски в рамке
        tr, segs = trace_instance(src, wt, fxL, fxR, ty, by, x0)
        traces[short] = (tr, col)
        nn = inject(ifds, short, tr, segs, W)
        print(f"  {short:5} рамка[{fxL}..{fxR}] цвет={col:6} сид_x={x0} трасса={nn}стр сегм={len(segs)} уровни={sorted(set(s[2] for s in segs))}")
    data = write_full(ifds)
    op = OUT / f"{Path(F).stem.replace('_auto','')}_v4.nlgx"
    op.write_bytes(data); op.with_suffix(".bck").write_bytes(write_bck(data))
    print(f"-> {op}")
    # оверлей
    drw = Image.fromarray(arr.copy()); dr = ImageDraw.Draw(drw)
    for short, (tr, col) in traces.items():
        for (ya, xa), (yb, xb) in zip(sorted(tr.items()), sorted(tr.items())[1:]):
            if abs(yb - ya) <= 3:
                dr.line([(xa, ya), (xb, yb)], fill=VIS.get(col, (255, 0, 255)), width=2)
    for tag, fr in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        yc = int(ty + fr * (by - ty)); drw.crop((30, max(0, yc - 320), 1180, min(H, yc + 320))).save(OUT / f"v4_qc_{tag}.png")
    print(f"оверлей -> {OUT}\\v4_qc_[a,b,c].png")


if __name__ == "__main__":
    main()
