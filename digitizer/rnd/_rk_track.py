r"""
_rk_track.py — RK по НЕПРЕРЫВНОСТИ, посеяно частичными трассами эксперта (не фикс-сплит).
Каждую кривую ведём от её сид-точек (GK/NGK) через ВЕСЬ трек: тугой гейт + коаст (не прыгаем
на чужую линию, где близко). Работает на сохранённой rk_prob.npy (без GPU).
Scale-change NGK (обороты масштаба) — следующий шаг; пока ведём линию.

python _rk_track.py [gate]
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

FRAME = r"F:\nds\output\test_single\Yatskivska_1_RK_3180_4080_200_D1.nlgx"
PROB = r"F:\nds\output\test_single\rk_prob.npy"
OUT = Path(r"F:\nds\output\test_single")


def centroids(row_mask, xlo, prob_row, gap=4):
    xs = np.nonzero(row_mask)[0]
    if not len(xs):
        return []
    out = []
    for s in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1):
        w = prob_row[s]; out.append(float((xlo + s) @ w / w.sum()))
    return out


def seeded_track(prob, seed, ty, by, xlo, xhi, thr=0.4, gate=18, slmax=22):
    tr = {}; x = v = None
    sy = sorted(seed)
    for y in range(ty, by):
        if y in seed:                                   # якорь на трассу эксперта
            nx = float(seed[y]); v = (nx - x) if x is not None else 0.0; x = nx; tr[y] = x; continue
        seg = prob[y, xlo:xhi]; mk = seg > thr
        cents = centroids(mk, xlo, seg)
        if not cents:
            if x is not None:
                x = x + float(np.clip(v, -slmax, slmax))
            continue
        if x is None:
            x = min(cents, key=lambda c: abs(c - seed[sy[0]])); v = 0.0; tr[y] = x; continue
        pred = x + float(np.clip(v, -slmax, slmax))
        nx = min(cents, key=lambda c: abs(c - pred))
        if abs(nx - pred) <= gate:
            v = 0.6 * v + 0.4 * (nx - x); x = nx          # ведём свою линию
        else:
            x = pred; v *= 0.5                            # коаст с затуханием (без разгона)
        x = float(np.clip(x, xlo - 30, xhi + 30))         # держим в треке (без взрыва)
        tr[y] = x
    return tr


def inject(ifds, short, tr, W):
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]; ty0 = struct.unpack("<I", tags[35474][:4])[0]
            xs = [int(round(tr[ty0 + j])) if (ty0 + j) in tr and 0 <= round(tr[ty0 + j]) < W else NULL for j in range(n)]
            set_tag(ifds, i, 35490, 4, xs)
            for t, v in [(35492, 1), (35494, ty0), (35496, ty0 + n - 1), (35498, 0)]:
                set_tag(ifds, i, t, 4, [v])
            vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
            if vx:
                rw = [ty0 + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, v in [(35478, min(xv)), (35480, min(rw)), (35482, max(xv)), (35484, max(rw))]:
                    set_tag(ifds, i, t, 4, [v])
            return sum(1 for x in xs if x != NULL)
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    gate = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    prob = np.load(PROB); m = extract(FRAME); H, W = prob.shape
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    xL, xR = m["scale_axes"][0]["x_left"], min(W, m["scale_axes"][0]["x_right"])
    seeds = {}
    for c in m["curves"]:
        sh = c["name"].split()[0]
        if sh.startswith("DA"): continue
        seeds[sh] = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
    print(f"сиды: " + ", ".join(f"{k}={len(v)}тчк y[{min(v)}..{max(v)}]" for k, v in seeds.items() if v))
    auto = {sh: seeded_track(prob, seeds[sh], ty, by, xL, xR, gate=gate) for sh in ("GK1", "NGK1")}
    for sh in ("GK1", "NGK1"):
        g = seeds[sh]; common = [y for y in g if y in auto[sh]]
        d = np.array([abs(auto[sh][y] - g[y]) for y in common]) if common else np.array([0])
        print(f"  {sh}: {len(auto[sh])}тчк | x[{int(min(auto[sh].values()))}..{int(max(auto[sh].values()))}] "
              f"| vs сид: |Δx| мед={np.median(d):.1f}px")

    ifds = read_full(open(FRAME, "rb").read())
    for sh in ("GK1", "NGK1"):
        print(f"  инъекция {sh}: {inject(ifds, sh, auto[sh], W)} точек")
    data = write_full(ifds)
    open(OUT / "Yatskivska_1_RK_3180_4080_200_D1_rktrack.nlgx", "wb").write(data)
    open(OUT / "Yatskivska_1_RK_3180_4080_200_D1_rktrack.bck", "wb").write(write_bck(data))
    print("-> ..._rktrack.nlgx")

    img = Image.open(m["img_path"]).convert("RGB")
    for tag, frac in [("a", 0.02), ("b", 0.35), ("c", 0.7)]:
        yc = int(ty + frac * (by - ty)); y0, y1 = max(0, yc - 240), min(H, yc + 240)
        crop = img.crop((0, y0, W, y1)).convert("RGB"); dr = ImageDraw.Draw(crop)
        for sh, col in [("GK1", (0, 170, 0)), ("NGK1", (0, 90, 255))]:
            pts = sorted((y, x) for y, x in auto[sh].items() if y0 <= y < y1)
            for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
                if abs(yb - ya) <= 3:
                    dr.line([(xa, ya - y0), (xb, yb - y0)], fill=col, width=2)
        crop.save(OUT / f"rktrack_{tag}.png")
    print("оверлей (GK зелёный/NGK синий) -> rktrack_[a,b,c].png")


if __name__ == "__main__":
    main()
