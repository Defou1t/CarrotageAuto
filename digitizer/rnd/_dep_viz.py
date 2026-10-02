r"""_dep_viz.py — КАРТИНКИ УХОДОВ ЛУЧШЕЙ ТРАССЫ С КРИВОЙ (02.10, к `_cand_fail.py` / §6.256).

Для кривых без честного кандидата — те же лучшая трасса и уходы, что в `_cand_fail.py`. Рисует вырезку листа вокруг
ухода: эталон своей кривой — зелёный, лучшая трасса — красная, прочие кривые эталона — синие. Подпись — вид ухода, путь
трассы (прод / декодер), расстояние, на которое трасса ушла через 30 строк.

  _dep_viz.py --every 40 --out <папка> --per-sheet 1
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--every", type=int, default=40)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--per-sheet", type=int, default=1)
ap.add_argument("--kinds", default="фон,чужая")
a = ap.parse_args()
TS = Path(a.ts); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
KINDS = a.kinds.split(",")
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def arr(d, n):
    out = np.full(n, np.nan, np.float32)
    if d:
        ys = np.fromiter(d.keys(), int, len(d)); xs = np.fromiter(d.values(), float, len(d))
        ok = ys < n
        out[ys[ok]] = xs[ok]
    return out


def bridged(t, n, gap=30):
    ys = np.asarray(t[0], int); xs = np.asarray(t[1], float)
    o = np.argsort(ys); ys, xs = ys[o], xs[o]
    ok = ys < n; ys, xs = ys[ok], xs[ok]
    raw = np.full(n, np.nan, np.float32); raw[ys] = xs
    br = raw.copy()
    if len(ys) > 1:
        d = np.diff(ys)
        for i in np.flatnonzero((d > 1) & (d <= gap)):
            yy = np.arange(ys[i] + 1, ys[i + 1])
            br[yy] = xs[i] + (xs[i + 1] - xs[i]) * (yy - ys[i]) / d[i]
    return raw, br


made = 0
for sh in lst("wellmap_sheets.txt")[a.offset::a.every]:
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem; key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    f = Path(a.cache) / f"{key}.pkl"
    if not f.exists():
        continue
    v = pickle.load(open(f, "rb"))
    Gd = {c["name"]: dense(c) for c in extract(str(q))["curves"]
          if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not Gd:
        continue
    T = [("прод", t) for _, t in v["traces"]] + ([("декодер", t) for _, t in v["alt"]] if v.get("alt") else [])
    n = 1 + max([max(g) for g in Gd.values()] + [int(np.max(t[0])) for _, t in T if len(t[0])])
    G = {k: arr(g, n) for k, g in Gd.items()}
    TB = [(src, *bridged(t, n)) for src, t in T]
    shots = 0
    img = None
    for g, gt in G.items():
        if shots >= a.per_sheet:
            break
        r = ~np.isnan(gt)
        if r.sum() < 50:
            continue
        best, bgood, bsrc, honest = None, -1, None, False
        for src, raw, br in TB:
            e = np.abs(raw[r] - gt[r]); com = ~np.isnan(e)
            if com.sum() >= 30 and np.median(e[com]) <= 3.0 and (~np.isnan(br[r])).mean() >= 0.9:
                honest = True; break
            good = int(np.sum(np.abs(br[r] - gt[r]) <= 3.0))
            if good > bgood:
                best, bgood, bsrc = br, good, src
        if honest or best is None:
            continue
        rows = np.flatnonzero(r)
        x = best[r]; gr = gt[r]
        others = {k2: o for k2, o in G.items() if k2 != g}
        st = np.full(len(rows), 3, np.int8)
        pres = ~np.isnan(x)
        st[pres] = 2
        st[pres & (np.abs(x - gr) <= 10.0)] = 4
        if others:
            O = np.vstack([o[r] for o in others.values()])
            with np.errstate(all="ignore"):
                st[pres & (np.nanmin(np.where(np.isnan(O), np.inf, np.abs(O - x[None, :])), axis=0) <= 3.0)] = 1
                contact = np.nanmin(np.where(np.isnan(O), np.inf, np.abs(O - gr[None, :])), axis=0) <= 10.0
        else:
            contact = np.zeros(len(rows), bool)
        st[pres & (np.abs(x - gr) <= 3.0)] = 0
        okr = st == 0
        i = 0
        while i < len(rows) and shots < a.per_sheet:
            if not okr[i]:
                i += 1; continue
            j = i
            while j < len(rows) and okr[j]:
                j += 1
            if j - i >= 20 and j < len(rows):
                k = j
                while k < len(rows) and not okr[k]:
                    k += 1
                if k - j >= 20:
                    tail = st[j:min(k, j + 50)]
                    where = ("чужая", "фон", "нет", "рядом")[int(np.bincount(tail, minlength=5)[1:].argmax())]
                    at = bool(contact[max(0, j - 5):j + 5].any())
                    if where in KINDS and not at:
                        y = int(rows[j])
                        if img is None:
                            img = Image.open(v["image"]).convert("RGB")
                        jj = min(j + 30, len(rows) - 1)
                        dist = float(abs(x[jj] - gr[jj])) if not np.isnan(x[jj]) else float("nan")
                        xs_ = [gr[max(0, j - 200):j + 200], x[max(0, j - 200):j + 200]]
                        xc = np.concatenate([q_[~np.isnan(q_)] for q_ in xs_])
                        x0 = int(max(0, np.min(xc) - 120)); x1 = int(min(img.width, np.max(xc) + 120))
                        if x1 - x0 < 500:
                            pad = (500 - (x1 - x0)) // 2; x0 = max(0, x0 - pad); x1 = min(img.width, x1 + pad)
                        y0, y1 = max(0, y - 300), min(img.height, y + 300)
                        cr = img.crop((x0, y0, x1, y1)).copy()
                        dr = ImageDraw.Draw(cr)
                        for k2, o in others.items():
                            pts = [(float(o[yy]) - x0, yy - y0) for yy in range(y0, y1) if yy < n and not np.isnan(o[yy])]
                            if len(pts) > 1:
                                dr.line(pts, fill=(0, 90, 255), width=1)
                        pts = [(float(gt[yy]) - x0, yy - y0) for yy in range(y0, y1) if yy < n and not np.isnan(gt[yy])]
                        if len(pts) > 1:
                            dr.line(pts, fill=(0, 200, 0), width=1)
                        for yy in range(y0, y1):
                            if yy < n and not np.isnan(best[yy]):
                                xx = float(best[yy]) - x0
                                dr.point([(xx + 3, yy - y0)], fill=(255, 0, 0))
                        dr.line([(0, y - y0), (12, y - y0)], fill=(255, 0, 255), width=3)
                        nm = f"{made:02d}_{where}_{bsrc}_{stem[:20]}_{g}_y{y}.png".replace(" ", "_")
                        cr.save(OUT / nm)
                        print(f"{nm}: уход «{where}» ({bsrc}), через 30 строк трасса в {dist:.0f} px от своей кривой; "
                              f"кривая {g} ({M.mnem_root(g)}), строк эталона {int(r.sum())}, верно {okr.mean():.2f}")
                        made += 1; shots += 1
            i = j
print(f"картинок {made}")
