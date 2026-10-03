"""Именование глазами: шапка листа + окно с кривыми ВЫДАЧИ (цвет + номер, без имён и без эталона).

  python name_viz.py --dump F:/nds/output/taskS/name_swaps_field.pkl --n 10 --seed 1 --out names
Печатает ключ (номер → имя выдачи; и какой эталон честно сопоставлен) в names/key.json — смотреть ПОСЛЕ своего решения.
"""
import sys, json, argparse, hashlib, pickle, random
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from extract_nlgx import extract, NULL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="names")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(exist_ok=True)
ROWS = pickle.load(open(a.dump, "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    IMGS.setdefault(q.stem, q)
COLS = [(230, 0, 0), (0, 0, 230), (200, 0, 200), (0, 150, 150), (230, 120, 0), (0, 160, 0), (120, 60, 0), (90, 90, 90)]
try:
    font = ImageFont.truetype("arialbd.ttf", 18)
except Exception:
    font = ImageFont.load_default()
sheets = sorted({r["sheet"] for r in ROWS})
rng = random.Random(a.seed); rng.shuffle(sheets)
KEY = {}
for j, sh in enumerate(sheets[:a.n]):
    q = SRC[sh]; stem = q.stem
    pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"))
    W = [c for c in extract(str(got))["curves"] if not c["name"].startswith("DA") and any(x != NULL for x in c["xs"])]
    img = Image.open(IMGS[stem]).convert("RGB")
    ys = [c["top_y"] + i for c in W for i, x in enumerate(c["xs"]) if x != NULL]
    top = min(ys)
    # шапка: всё выше начала кривых (не больше 6000 строк), ширина листа → 1100 px
    h0 = max(0, top - 6000)
    hd = img.crop((0, h0, img.width, top + 300))
    s = 1100 / hd.width
    hd = hd.resize((1100, int(hd.height * s)), Image.LANCZOS)
    hd.save(OUT / f"{j:02d}_header.png")
    # окно кривых: 1500 строк из середины, кривые выдачи цветом с номером
    mid = (top + max(ys)) // 2
    y0, y1 = mid - 750, mid + 750
    xs = [x for c in W for i, x in enumerate(c["xs"]) if x != NULL and y0 <= c["top_y"] + i < y1]
    x0, x1 = max(0, int(min(xs)) - 80), min(img.width, int(max(xs)) + 80)
    cr = img.crop((x0, y0, x1, y1))
    sc = min(1.0, 1100 / cr.width, 1400 / cr.height)
    cr = cr.resize((int(cr.width * sc), int(cr.height * sc)), Image.LANCZOS)
    dr = ImageDraw.Draw(cr)
    key = {}
    for i, c in enumerate(W):
        col = COLS[i % len(COLS)]
        pts = [((x - x0) * sc, (c["top_y"] + t - y0) * sc) for t, x in enumerate(c["xs"]) if x != NULL and y0 <= c["top_y"] + t < y1]
        for p in pts[::2]:
            dr.ellipse([(p[0] - 1, p[1] - 1), (p[0] + 1, p[1] + 1)], fill=col)
        if pts:
            p = pts[len(pts) // 8]
            dr.text((p[0] + 8, p[1]), f"#{i + 1}", fill=col, font=font)
        key[f"#{i + 1}"] = c["name"]
    cr.save(OUT / f"{j:02d}_curves.png")
    truth = [c["name"] for c in extract(str(q))["curves"] if not c["name"].startswith("DA") and sum(1 for x in c["xs"] if x != NULL) >= 50]
    KEY[f"{j:02d}"] = dict(sheet=sh, out=key, truth_names=truth,
                           swaps=[(r["truth"], r["out"]) for r in ROWS if r["sheet"] == sh])
    print(f"{j:02d}: {stem} — кривых выдачи {len(W)}; имена эталона: {', '.join(truth)}")
json.dump(KEY, open(OUT / "key.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
