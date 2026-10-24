"""К `viz_nocand.py`: тот же кусок, но все кандидаты слота (цвета по порядку) и эталон (зелёный пунктир слева на 5 px); подпись —
покрытие / медиана каждого кандидата. viz_nocand_cands.py <out_dir> <лист-подстрока> <слот> <y0> <y1>"""
import sys, pickle, hashlib, glob
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract
from _multi_replica_probe import dense
from _cand_ceiling import dense_tr

TS = Path("F:/nds/output/taskS")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sub, slot, y0, y1 = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
rec = None
for d, cache, ex in (("line_choice_i", "tcache", "tcache_pk3|alt;tcache_inert|extra"), ("line_choice_out_i", "tcache_out", "tcache_out|extra;tcache_out_inert|extra")):
    for f in glob.glob(str(TS / d / "*.pkl")):
        r = pickle.load(open(f, "rb"))
        if sub in r["sheet"]:
            rec, CACHE, EXTRA = r, cache, ex
            break
    if rec:
        break
q = SRC[rec["sheet"]]
key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
c = pickle.load(open(TS / CACHE / (key + ".pkl"), "rb"))
un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
xsrc = []
for spec in EXTRA.split(";"):
    dd, _, fld = spec.partition("|")
    xp = TS / dd / (key + ".pkl")
    if xp.exists():
        xsrc += pickle.load(open(xp, "rb")).get(fld or "alt") or []
s = next(s for s in rec["slots"] if s["name"] == slot)
g = dense({c_["name"]: c_ for c_ in extract(str(q))["curves"]}[slot])
img = Image.open(c["image"]).convert("RGB")
b0, b1 = sorted(s["band"]); x0, x1 = max(0, b0 - 30), min(img.width, b1 + 30)
cr = img.crop((x0, y0, x1, y1)).copy(); dr = ImageDraw.Draw(cr)
PAL = [(255, 0, 0), (0, 90, 255), (255, 140, 0), (200, 0, 200), (0, 160, 160), (120, 60, 0), (255, 0, 120), (90, 90, 255)]
TOP = int(sys.argv[6]) if len(sys.argv) > 6 else 99
for j, sc in enumerate(sorted(s["cands"], key=lambda z: (z["lab"] or (0, 0, 99))[2])[:TOP]):
    cs = rec["cands"][sc["k"]]
    t = c["traces"][cs["idx"]][1] if cs["src"] == "prod" else (c["alt"][cs["idx"]][1] if cs["idx"] < 1000 else xsrc[cs["idx"] - 1000][1])
    d = dense_tr(un(t))
    pts = [(d[y] - x0, y - y0) for y in range(y0, y1) if y in d]
    col = PAL[j % len(PAL)]
    for a_, b_ in zip(pts, pts[1:]):
        if b_[1] - a_[1] <= 3:
            dr.line([a_, b_], fill=col, width=1)
    lab = sc["lab"]
    print(f"  кандидат {j} цвет {col}: k {sc['k']} src {cs['src']} строк {cs['n']} {cs['y0']}–{cs['y1']}; "
          f"метка {None if lab is None else (bool(lab[0]), round(lab[1], 2), round(lab[2], 1))}; here {sc['here']}")
for y in range(y0, y1, 4):
    if y in g:
        dr.point((g[y] - x0 - 5, y - y0), fill=(0, 170, 0)); dr.point((g[y] - x0 - 6, y - y0), fill=(0, 170, 0))
k = min(1.0, 900 / cr.width)
if k < 1:
    cr = cr.resize((int(cr.width * k), int(cr.height * k)))
cr.save(out / f"cands_{slot.split()[0]}_{y0}.png")
