"""Где трасса прода сбивает декодер переходов: окна у переходов эталона, где уровень по трассе прода неверен.

  python level_diag.py --n 6 --out <папка>
Эталон — точки по уровню (зелёный 1×, фиолетовый 5×); трасса прода — точки по уровню декодера на ней (красный 1×,
синий 5×), сдвиг +10 px; красная риска — строка перехода эталона.
"""
import sys, argparse, hashlib, random
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, refine
from auto.config import DEFAULT
import decode_levels as DL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--out", required=True)
ap.add_argument("--fams", default="GZ,OGZ,PZ,IK")
ap.add_argument("--nojump", action="store_true")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
TS = Path(r"F:/nds/output/taskS")
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
        IMGS.setdefault(q.stem, q)
FAMS = set(a.fams.split(","))
try:
    font = ImageFont.truetype("arial.ttf", 15)
except Exception:
    font = ImageFont.load_default()


def lv_rows(c):
    out = {}
    for y0, y1, lv in c.get("segments") or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


def dec(xs, fam):
    return refine.enforce_min_run(DL.decode(xs, fam, lam=DEFAULT.cv.level_lam), DEFAULT.cv.level_min_run)


sheets = [l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
random.Random(a.seed).shuffle(sheets)
made = 0
for sh in sheets:
    if made >= a.n:
        break
    q = SRC.get(sh)
    if not q or q.stem not in IMGS:
        continue
    stem = q.stem
    pd = TS / "rp_vc" / "N" / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    if not got:
        continue
    mt = extract(str(q)); mw = {c["name"]: c for c in extract(str(got))["curves"]}
    for c in mt["curves"]:
        if M.mnem_root(c["name"]) not in FAMS or c["name"] not in mw or made >= a.n:
            continue
        lt = lv_rows(c)
        if not any(v > 0 for v in lt.values()):
            continue
        fam = DL.build_family(mt, c)
        if len(fam) < 2:
            continue
        gt = dense(c); w = mw[c["name"]]; tr = dense(w)
        com = [y for y in tr if y in gt]
        if len(com) < 200 or np.median([abs(tr[y] - gt[y]) for y in com]) > 3:
            continue                                             # только честные по пикселям
        xt = {c["top_y"] + i: float(x) for i, x in enumerate(c["xs"]) if x != NULL}
        xw = {w["top_y"] + i: float(x) for i, x in enumerate(w["xs"]) if x != NULL}
        lt_d, lw_d = dec(xt, fam), dec(xw, fam)
        rows = [y for y in com if y in lt]
        acc_t = np.mean([lt_d.get(y, 0) == lt[y] for y in rows if y in lt_d] or [0])
        acc_w = np.mean([lw_d.get(y, 0) == lt[y] for y in rows if y in lw_d] or [0])
        if a.nojump:
            pass
        elif not (acc_t >= 0.9 and acc_w < 0.7):
            continue
        # первая строка перехода эталона, где уровень по трассе прода неверен
        ys = sorted(lt)
        trans = [y for y0, y in zip(ys, ys[1:]) if lt[y] != lt[y0]]
        if a.nojump:
            wyv = np.array(sorted(xw)); wxv = np.array([xw[y] for y in wyv])
            bad = []
            for y in trans:
                mb = (wyv >= y - 25) & (wyv < y); ma = (wyv >= y) & (wyv <= y + 25)
                if mb.any() and ma.any() and abs(wxv[ma][0] - wxv[mb][-1]) < 30:
                    bad.append(y)
        else:
            bad = [y for y in trans if lw_d.get(y + 30, -1) != lt.get(y + 30, -2)]
        if not bad:
            continue
        yc = bad[0]
        y0, y1 = yc - 350, yc + 350
        xs = [v for d in (gt, tr) for y, v in d.items() if y0 <= y < y1]
        if not xs:
            continue
        im = Image.open(IMGS[stem]).convert("RGB")
        x0 = max(0, int(min(xs)) - 60); x1 = min(im.width, int(max(xs)) + 80)
        z = 1.0
        cr = im.crop((x0, y0, x1, y1)).copy(); dr = ImageDraw.Draw(cr)
        for y, x in gt.items():
            if y0 <= y < y1:
                col = (0, 170, 0) if lt.get(y, 0) == 0 else (150, 0, 200)
                dr.ellipse([(x - x0 - 2, y - y0 - 2), (x - x0 + 2, y - y0 + 2)], fill=col)
        for y, x in tr.items():
            if y0 <= y < y1:
                col = (225, 0, 0) if lw_d.get(y, 0) == 0 else (20, 60, 230)
                dr.ellipse([(x - x0 + 9, y - y0 - 1), (x - x0 + 11, y - y0 + 1)], fill=col)
        dr.line([(0, yc - y0), (25, yc - y0)], fill=(255, 0, 0), width=4)
        can = Image.new("RGB", (max(cr.width, 900), cr.height + 70), "white")
        can.paste(cr, (0, 70)); d2 = ImageDraw.Draw(can)
        d2.text((6, 4), f"{stem[:60]} — {c['name'].split()[0]}, переход эталона в строке {yc}", fill=(0, 0, 0), font=font)
        d2.text((6, 24), f"декодер на трассе эксперта верен {acc_t:.0%} строк, на трассе прода {acc_w:.0%}; "
                         f"прод выдал сегментов {len(w.get('segments') or [])}", fill=(0, 0, 0), font=font)
        d2.text((6, 44), "эталон: зелёный 1×, фиолетовый 5×; трасса прода (+10 px): красный 1×, синий 5× — по декодеру на ней",
                fill=(0, 0, 0), font=font)
        nm = f"L{made:02d}_{stem[:30]}_{c['name'].split()[0]}.png".replace(" ", "_").replace(",", "")
        can.save(OUT / nm)
        print(f"{nm}: acc эксперт {acc_t:.2f} / прод {acc_w:.2f}; переход {yc}")
        made += 1
