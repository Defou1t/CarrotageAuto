"""Примеры потерь на уровнях (§6.321): кривая честна в пикселях, уровень масштаба не тот. Скан полосы трека; эталон слева (−5 px),
выдача справа (+5 px), цвет — уровень (зелёный 1×, синий 2-й, пурпурный 3-й). viz_level_err.py <out_dir> [n] [seed]"""
import sys, pickle, random
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
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
E = [e for e in pickle.load(open(TS / "_lvl_loss_8t.pkl", "rb")) if not e["shift"] and 0.3 <= e["frac"] <= 0.7]
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
COL = [(0, 170, 0), (0, 80, 255), (220, 0, 220), (255, 140, 0)]


def lv_rows(c):
    o = {}
    for y0, y1, lv in c.get("segments") or []:
        for y in range(int(y0), int(y1) + 1):
            o[y] = int(lv)
    return o


done = 0
for e in random.sample(E, min(len(E), 4 * n)):
    tc = TS / "tcache" / (e["dir"] + ".pkl")
    if not tc.exists():
        tc = TS / "tcache_out" / (e["dir"] + ".pkl")
    if not tc.exists():
        continue
    mt = extract(str(SRC[e["sheet"]])); g = next(c for c in mt["curves"] if c["name"] == e["curve"])
    got = next((TS / "rp_lvl" / "NSE8T" / e["dir"]).glob("*_auto.nlgx"))
    w = next(c for c in extract(got.as_posix())["curves"] if c["name"] == e["curve"])
    gt, wr = dense(g), dense(w); lg, lw = lv_rows(g), lv_rows(w)
    com = sorted(y for y in wr if y in gt)
    bad = [y for y in com if lg.get(y, 0) != lw.get(y, 0)]
    if not bad:
        continue
    yc = bad[len(bad) // 2]; y0, y1 = max(com[0], yc - 900), min(com[-1], yc + 900)
    ch = DL.build_family(mt, g)
    xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
    im = Image.open(pickle.load(open(tc, "rb"))["image"]).convert("RGB")
    m = int(0.08 * (xr - xl)); x0, x1 = max(0, int(xl) - m), min(im.width, int(xr) + m)
    cr = im.crop((x0, y0, x1, y1)).copy(); d = ImageDraw.Draw(cr)
    for tr, lv, dx in ((gt, lg, -5), (wr, lw, 5)):
        pts = [(y, tr[y]) for y in range(y0, y1) if y in tr]
        for (ya, xa), (yb, xb) in zip(pts, pts[1:]):
            if yb - ya <= 3 and abs(xb - xa) < 0.5 * (xr - xl):
                d.line([(xa - x0 + dx, ya - y0), (xb - x0 + dx, yb - y0)], fill=COL[min(3, lv.get(ya, 0))], width=2)
    d.text((4, 4), f"{e['curve']}  шкалы {[(round(a, 1), round(b, 1)) for a, b in e['chain']]}", fill=(255, 0, 0))
    d.text((4, 18), "эталон: слева, выдача: справа; зелёный 1-й уровень, синий 2-й, пурпурный 3-й", fill=(255, 0, 0))
    if cr.height > 1400:
        cr = cr.resize((int(cr.width * 1400 / cr.height), 1400))
    cr.save(out / f"{done:02d}_{e['sheet'][:40]}_{e['curve'].split()[0]}.png")
    print(e["sheet"], e["curve"], f"неверно {e['frac']:.2f} строк")
    done += 1
    if done >= n:
        break
