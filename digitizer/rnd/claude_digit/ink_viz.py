"""Окна решений модели уровня по скану (§6.266): где уровень модели сделал кривую честной в значениях и где сломал.

  python ink_viz.py --n 6 --out <папка> [--pred F:/nds/output/taskS/level_ink_pred.pkl]
Слева полосы уровней: эталон / прод / модель (зелёный 1×, фиолетовый 5×, оранжевый 25×, серый выше). На скане: эталон — точки
по уровню эталона; трасса выдачи (+10 px) — по уровню модели. Красная риска — строка, где модель и прод расходятся сильнее всего.
"""
import sys, argparse, pickle, hashlib, random
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from _level_ink import honest, seg_levels

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--pred", default=r"F:/nds/output/taskS/level_ink_pred.pkl")
ap.add_argument("--half", type=int, default=450)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
        IMGS.setdefault(q.stem, q)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except Exception:
    font = ImageFont.load_default()
COL = [(0, 160, 0), (150, 0, 200), (240, 140, 0), (120, 120, 120)]
CUR = pickle.load(open(a.cache, "rb"))
PR = pickle.load(open(a.pred, "rb"))
LV = PR["levels"]


def rows_levels(lo, rows):
    ks = np.array(sorted(lo), np.int64); vs = np.array([lo[k] for k in ks], np.int16)
    pos = np.searchsorted(ks, rows, side="right") - 1
    return np.where(pos >= 0, vs[np.clip(pos, 0, len(vs) - 1)], 0)


gain, lost = [], []
for ci, lo in LV.items():
    cv = CUR[ci]
    h0, h1 = honest(cv, None), honest(cv, lo)
    if h1 and not h0:
        gain.append(ci)
    elif h0 and not h1:
        lost.append(ci)
print(f"стало честных {len(gain)}, перестало {len(lost)}")
rnd = random.Random(a.seed)
rnd.shuffle(gain); rnd.shuffle(lost)
made = 0
for kind, lst in (("стала", gain), ("сломана", lost)):
    for ci in lst[:a.n]:
        cv = CUR[ci]; sh = cv["sheet"]; q = SRC.get(sh)
        if not q or q.stem not in IMGS:
            continue
        ty = cv["ty"].astype(np.int64)
        lm = rows_levels(LV[ci], ty); lp = seg_levels(cv["lw"], ty); lt = seg_levels(cv["lt"], ty)
        dif = (lm != lp).astype(np.float64)
        if not dif.any():
            continue
        k = int(np.argmax(np.convolve(dif, np.ones(201), mode="same")))
        yc = int(ty[k]); y0, y1 = yc - a.half, yc + a.half
        im = Image.open(IMGS[q.stem]).convert("RGB")
        ch = cv["chain"]
        xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
        x0 = max(0, int(xl) - 40); x1 = min(im.width, int(xr) + 40)
        y0c, y1c = max(0, y0), min(im.height, y1)
        cr = im.crop((x0, y0c, x1, y1c)).copy(); dr = ImageDraw.Draw(cr)
        gy, gx = cv["gy"], cv["gx"]
        sel = (gy >= y0c) & (gy < y1c)
        lg = seg_levels(cv["lt"], gy[sel].astype(np.int64))
        for y, x, l in zip(gy[sel][::3], gx[sel][::3], lg[::3]):
            dr.ellipse([(x - x0 - 2, y - y0c - 2), (x - x0 + 2, y - y0c + 2)], fill=COL[min(int(l), 3)])
        selw = (ty >= y0c) & (ty < y1c)
        for y, x, l in zip(ty[selw][::3], cv["tx"][selw][::3], lm[selw][::3]):
            c_ = COL[min(int(l), 3)]
            dr.rectangle([(x - x0 + 9, y - y0c - 1), (x - x0 + 12, y - y0c + 1)], fill=(c_[0] // 2 + 100, c_[1] // 2, c_[2] // 2 + 60))
        dr.line([(0, yc - y0c), (30, yc - y0c)], fill=(255, 0, 0), width=4)
        BAR = 3 * 14 + 10
        can = Image.new("RGB", (cr.width + BAR, cr.height + 70), "white")
        can.paste(cr, (BAR, 70)); d2 = ImageDraw.Draw(can)
        for j, lv_ in enumerate((lt, lp, lm)):
            for y, l in zip(ty[selw], lv_[selw]):
                d2.line([(4 + j * 14, 70 + y - y0c), (14 + j * 14, 70 + y - y0c)], fill=COL[min(int(l), 3)])
        d2.text((6, 4), f"{q.stem[:70]} — {cv['name'].split()[0]} ({cv['set']}): кривая {kind} честной в значениях", fill=(0, 0, 0), font=font)
        d2.text((6, 24), "полосы слева: эталон / прод / модель; цвет уровня: зелёный 1×, фиолетовый 5×, оранжевый 25×", fill=(0, 0, 0), font=font)
        d2.text((6, 44), "точки на скане — эталон по его уровню; штрихи (+10 px) — трасса выдачи по уровню модели", fill=(0, 0, 0), font=font)
        if can.width > 1600:
            can = can.resize((1600, int(can.height * 1600 / can.width)), Image.LANCZOS)
        nm = f"I{made:02d}_{kind}_{q.stem[:34]}_{cv['name'].split()[0]}.png".replace(" ", "_").replace(",", "")
        can.save(OUT / nm)
        print(nm)
        made += 1
