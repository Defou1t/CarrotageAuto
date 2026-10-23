"""Где эксперт меняет уровень масштаба: окно ±350 строк вокруг границы сегмента эталона, вся полоса шкалы с полями.
Эталон — цветом уровня (зелёный 0 = 1-я шкала, синий 1, пурпурный 2); горизонталь — строка перехода; крестики — где была бы та же
точка эталона на шкале соседнего уровня (по цепочке шкал). viz_level_switch.py <out_dir> [n] [seed]"""
import sys, pickle, random, hashlib
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract
from _multi_replica_probe import dense
import decode_levels as DL

TS = Path("F:/nds/output/taskS")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
E = [e for e in pickle.load(open(TS / "_lvl_loss_8t.pkl", "rb")) if not e["shift"]]
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
COL = [(0, 160, 0), (0, 70, 255), (220, 0, 220), (255, 140, 0)]


def val(s, x):
    return s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / ((s["x_right"] - s["x_left"]) or 1)


def pix(s, v):
    return s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / ((s["v_right"] - s["v_left"]) or 1)


done = 0
for e in random.sample(E, min(len(E), 6 * n)):
    tc = TS / "tcache" / (e["dir"] + ".pkl")
    if not tc.exists():
        tc = TS / "tcache_out" / (e["dir"] + ".pkl")
    if not tc.exists():
        continue
    mt = extract(str(SRC[e["sheet"]])); g = next(c for c in mt["curves"] if c["name"] == e["curve"])
    segs = g.get("segments") or []
    bnd = [(segs[i][0], segs[i - 1][2], segs[i][2]) for i in range(1, len(segs)) if segs[i][2] != segs[i - 1][2]]
    if not bnd:
        continue
    yb, lv_from, lv_to = random.choice(bnd)
    ch = DL.build_family(mt, g)
    gt = dense(g)
    lv = {}
    for y0, y1, l in segs:
        for y in range(int(y0), int(y1) + 1):
            lv[y] = int(l)
    xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
    im = Image.open(pickle.load(open(tc, "rb"))["image"]).convert("RGB")
    m = int(0.12 * (xr - xl)); x0, x1 = max(0, int(xl) - m), min(im.width, int(xr) + m)
    y0, y1 = max(0, int(yb) - 350), min(im.height, int(yb) + 350)
    cr = im.crop((x0, y0, x1, y1)).copy(); d = ImageDraw.Draw(cr)
    for xb in (xl, xr):
        d.line([(xb - x0, 0), (xb - x0, cr.height)], fill=(255, 160, 0), width=1)
    pts = [(y, gt[y]) for y in range(y0, y1) if y in gt]
    for (ya, xa), (yb2, xb2) in zip(pts, pts[1:]):
        if yb2 - ya <= 3:
            d.line([(xa - x0 - 6, ya - y0), (xb2 - x0 - 6, yb2 - y0)], fill=COL[min(3, lv.get(ya, 0))], width=3)
    for y, x in pts[::25]:                       # та же точка на шкале соседнего уровня
        l = lv.get(y, 0)
        for l2 in (l - 1, l + 1):
            if 0 <= l2 < len(ch):
                xo = pix(ch[l2], val(ch[l], x))
                if x0 <= xo <= x1:
                    d.line([(xo - x0 - 3, y - y0 - 3), (xo - x0 + 3, y - y0 + 3)], fill=COL[min(3, l2)], width=1)
                    d.line([(xo - x0 - 3, y - y0 + 3), (xo - x0 + 3, y - y0 - 3)], fill=COL[min(3, l2)], width=1)
    d.line([(0, yb - y0), (cr.width, yb - y0)], fill=(255, 0, 0), width=1)
    d.text((4, yb - y0 + 3), f"switch {lv_from}->{lv_to} at row {yb}", fill=(255, 0, 0))
    d.text((4, 4), f"{e['curve'].split()[0]} scales " + " | ".join(f"{s['v_left']:g}..{s['v_right']:g}" for s in ch), fill=(255, 0, 0))
    fn = out / f"{done:02d}_{e['sheet'][:36]}_{e['curve'].split()[0]}_r{yb}.png"
    cr.save(fn)
    print(fn.name, f"переход {lv_from}->{lv_to}, шкалы", [(s["v_left"], s["v_right"]) for s in ch])
    done += 1
    if done >= n:
        break
