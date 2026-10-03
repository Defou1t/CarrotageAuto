"""Проверка глазами: окно листа с выдачей прода (цвет + подпись на кривую) и эталоном (тонкий зелёный).

  python audit_viz.py --n 12 --seed 3 --rows 2000 --scale 0.45 --out audit
Печатает для каждой картинки: лист, окно, кривые выдачи с цветами и метрики (|Δx| медиана, на плоскости ≤ 3 px) против
эталона ТОГО ЖЕ имени в окне — для сверки с моим вердиктом после просмотра.
"""
import sys, json, argparse, hashlib, random
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from extract_nlgx import extract, NULL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--sheets", default=r"F:/nds/output/taskS/wellmap_sheets.txt")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--rows", type=int, default=2000)
ap.add_argument("--scale", type=float, default=0.45)
ap.add_argument("--out", default="audit")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(exist_ok=True)
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    IMGS.setdefault(q.stem, q)
COLS = [(230, 0, 0), (0, 0, 230), (200, 0, 200), (0, 150, 150), (230, 120, 0), (120, 60, 0), (0, 0, 0), (150, 150, 0)]
try:
    font = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()


def dense_curve(c, gap=200):
    pts = [(c["top_y"] + i, float(x)) for i, x in enumerate(c["xs"]) if x != NULL]
    out = {}
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        out[y0] = x0
        if 0 < y1 - y0 <= gap:
            for yy in range(y0 + 1, y1):
                out[yy] = x0 + (x1 - x0) * (yy - y0) / (y1 - y0)
    if pts:
        out[pts[-1][0]] = pts[-1][1]
    return out


names = [l.strip() for l in open(a.sheets, encoding="utf-8") if l.strip()]
rng = random.Random(a.seed)
rng.shuffle(names)
made = 0
for sh in names:
    if made >= a.n:
        break
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem
    pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    img = IMGS.get(stem)
    if not got or not img:
        continue
    G = {c["name"]: dense_curve(c) for c in extract(str(q))["curves"]
         if not c["name"].startswith("DA") and sum(1 for x in c["xs"] if x != NULL) >= 50}
    W = {c["name"]: dense_curve(c) for c in extract(str(got))["curves"] if not c["name"].startswith("DA")}
    W = {k: v for k, v in W.items() if v}
    if not G or not W:
        continue
    ys = sorted(set().union(*[set(g) for g in G.values()]))
    lo, hi = ys[0], ys[-1]
    if hi - lo > a.rows:
        y0 = rng.randint(lo, hi - a.rows)
    else:
        y0 = lo
    y1 = y0 + a.rows
    xs = [x for g in list(G.values()) + list(W.values()) for y, x in g.items() if y0 <= y < y1]
    if not xs:
        continue
    x0, x1 = int(max(0, min(xs) - 60)), int(max(xs) + 60)
    im = Image.open(img).convert("RGB")
    x1 = min(x1, im.width)
    cr = im.crop((x0, y0, x1, min(y1, im.height)))
    s = a.scale
    cr = cr.resize((int(cr.width * s), int(cr.height * s)), Image.LANCZOS)
    dr = ImageDraw.Draw(cr)
    for g in G.values():
        pts = [((x - x0) * s, (y - y0) * s) for y, x in sorted(g.items()) if y0 <= y < y1]
        for i in range(0, len(pts) - 1, 2):
            dr.line([pts[i], pts[i + 1]], fill=(0, 200, 0), width=1)
    leg = []
    for j, (k, w) in enumerate(sorted(W.items())):
        col = COLS[j % len(COLS)]
        pts = [((x - x0) * s, (y - y0) * s) for y, x in sorted(w.items()) if y0 <= y < y1]
        for p in pts[::3]:
            dr.point([(p[0] + 2, p[1])], fill=col)
            dr.point([(p[0] + 3, p[1])], fill=col)
        if pts:
            dr.text((pts[0][0] + 6, max(0, pts[0][1])), k.split()[0], fill=col, font=font)
        # метрики против эталона того же имени в окне
        gt = G.get(k)
        if gt:
            com = [y for y in range(y0, y1) if y in w and y in gt]
            e = np.array([abs(w[y] - gt[y]) for y in com]) if com else np.array([np.nan])
            leg.append(f"{k} [{col}]: строк {len(com)}, |Δx| медиана {np.nanmedian(e):.1f}")
        else:
            leg.append(f"{k} [{col}]: эталона с этим именем нет")
    nm = f"{made:02d}_{stem[:30]}.png".replace(" ", "_")
    cr.save(OUT / nm)
    print(f"{nm}: окно {y0}–{y1}, x {x0}–{x1}; эталон: {', '.join(sorted(G))}")
    for l in leg:
        print("    " + l)
    made += 1
