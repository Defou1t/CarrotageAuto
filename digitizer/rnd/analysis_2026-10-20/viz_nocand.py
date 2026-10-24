"""Слоты без честного кандидата (`nocand_ink.py` → nocand_ink.pkl): кусок скана полосы, где выдача NSE15 дальше всего от эталона.
Эталон — зелёный (точки), кривая выдачи под тем же именем — красный, прочие кривые выдачи в полосе — синий.
viz_nocand.py <out_dir> [n] [seed] [категория-подстрока]"""
import sys, pickle, random, hashlib
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract
from _multi_replica_probe import dense

TS = Path("F:/nds/output/taskS")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
cat = sys.argv[4] if len(sys.argv) > 4 else "мед > 15"
E = [e for e in pickle.load(open(TS / "nocand_ink.pkl", "rb")) if cat in e["cat"] and e["ink"] >= 0.8]
random.shuffle(E)
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
done = 0
for e in E:
    if done >= n:
        break
    q = SRC[e["sheet"]]
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted((TS / "rp_lvl" / "NSE15" / key).glob("*_auto.nlgx"))), None)
    tc = TS / e["cache"] / (key + ".pkl")
    if got is None or not tc.exists():
        continue
    gt = {c["name"]: c for c in extract(str(q))["curves"]}.get(e["slot"])
    mw = extract(got.as_posix())["curves"]
    if gt is None:
        continue
    g = dense(gt)
    w = {c["name"]: dense(c) for c in mw}
    me = w.get(e["slot"], {})
    ys = np.array(sorted(g))
    dev = np.array([abs(me[y] - g[y]) if y in me else 300 for y in ys])
    win = 700
    sm = np.convolve(dev, np.ones(win) / win, mode="same")
    yc = int(ys[int(np.argmax(sm))]); y0, y1 = yc - win // 2, yc + win // 2
    img = Image.open(pickle.load(open(tc, "rb"))["image"]).convert("RGB")
    b0, b1 = sorted(e["band"]); x0, x1 = max(0, b0 - 30), min(img.width, b1 + 30)
    y0 = max(0, y0); y1 = min(img.height, y1)
    cr = img.crop((x0, y0, x1, y1)).copy(); dr = ImageDraw.Draw(cr)
    for nm, d in w.items():
        if nm == e["slot"]:
            continue
        pts = [(d[y] - x0, y - y0) for y in range(y0, y1) if y in d and b0 - 30 <= d[y] <= b1 + 30]
        for p_ in pts[::3]:
            dr.point(p_, fill=(0, 90, 255))
    pts = [(me[y] - x0, y - y0) for y in range(y0, y1) if y in me]
    if len(pts) > 1:
        dr.line(pts, fill=(255, 0, 0), width=1)
    for y in range(y0, y1, 4):
        if y in g:
            dr.point((g[y] - x0 - 4, y - y0), fill=(0, 170, 0)); dr.point((g[y] - x0 - 5, y - y0), fill=(0, 170, 0))
    sc = min(1.0, 900 / cr.width)
    if sc < 1:
        cr = cr.resize((int(cr.width * sc), int(cr.height * sc)))
    fn = out / f"{done:02d}_{e['root']}_{q.stem[:30]}.png"
    cr.save(fn)
    print(f"{fn.name}: {e['slot']} | {e['cat']} | ink {e['ink']:.2f} | строки {y0}–{y1} | полоса {e['band']}")
    done += 1
