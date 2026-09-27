r"""_tolerance_whatif.py — ЧТО ДАСТ ДОПУСК, ЗАВИСЯЩИЙ ОТ ТОЛЩИНЫ ШТРИХА (решение заказчика, §6.216 / §6.234).

§6.216: 124 кривые «рядом» (3 < медиана ≤ 10 px) идут по ТОЙ ЖЕ туши на толстых штрихах, где эксперт сам не держит центр —
допуск 3 px по медиане ниже точности эталона. Здесь — точное число на выдаче нынешнего прода (повтор `--dir`, поле): для
каждой эталонной кривой ширина штриха по скану (горизонтальный ран туши в точке эталона, притянутой к туши ±3 px, поправка
на наклон, медиана по строкам), честность со строгим (3 px) и мягким допуском max(3, `--k`·ширина), сопоставление 1:1
пересчитывается заново (мягкий допуск может переставить пары).

  _tolerance_whatif.py --dir F:/nds/output/taskS/rp_fill --mode NF --k 0.35
"""
import sys, argparse, pickle, hashlib, math
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im, trace2d as T2
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_fill")
ap.add_argument("--mode", default="NF")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--k", type=float, default=0.35)
ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()
TS = Path(a.ts)
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
field = [l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()][::a.every]


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def match(rows, cols, ok):
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    for r in rows:
        try_(r, set())
    return {r: c for c, r in pair.items()}


C = Counter(); gained = Counter(); widths = []
for si, sh in enumerate(field, 1):
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem
    pd = Path(a.dir) / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    img = find_image(q)
    if not got or not img:
        continue
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    rgb = im.load_rgb(str(img))
    ink = T2._color_fg(rgb, "black", DEFAULT.cv).copy()
    sm = im.structure_mask(rgb, DEFAULT.cv)
    for cm in im.color_channels(rgb, DEFAULT.cv).values():
        ink |= (cm & ~sm)
    Hh, Ww = ink.shape
    wid = {}
    for g, gt in G.items():
        ys = sorted(gt)[::7]; vals = []
        for y in ys:
            if not (0 <= y < Hh):
                continue
            xi = int(round(gt[y])); lo, hi = max(0, xi - 3), min(Ww, xi + 4)
            idx = np.flatnonzero(ink[y, lo:hi])
            if not len(idx):
                continue
            x0 = lo + int(idx[np.argmin(np.abs(lo + idx - xi))])
            l = r = x0
            while l > 0 and ink[y, l - 1]:
                l -= 1
            while r < Ww - 1 and ink[y, r + 1]:
                r += 1
            s = (gt.get(y + 5, gt[y]) - gt.get(y - 5, gt[y])) / 10.0
            if abs(s) <= 2:
                vals.append((r - l + 1) / math.sqrt(1 + s * s))
        wid[g] = float(np.median(vals)) if len(vals) >= 10 else 0.0
        widths.append(wid[g])
    del rgb, ink, sm
    tm = smap.get(sh, {})
    for t in {tm.get(g) for g in G if tm.get(g) is not None}:
        gs = [g for g in G if tm.get(g) == t]; ws = [k for k in W if tm.get(k) == t and W[k]]
        ok3, okw = {}, {}
        for g in gs:
            tol = max(3.0, a.k * wid[g])
            for k in ws:
                m, c = st(W[k], G[g])
                ok3[(g, k)] = m is not None and m <= 3.0 and c >= 0.9
                okw[(g, k)] = m is not None and m <= tol and c >= 0.9
        m3 = match(gs, ws, ok3); mw = match(gs, ws, okw)
        C["кривых"] += len(gs); C["честных при 3 px"] += len(m3); C["честных при мягком допуске"] += len(mw)
        for g in gs:
            if g in mw and g not in m3:
                gained[M.mnem_root(g)] += 1
    if si % 100 == 0:
        print(f"  … {si}/{len(field)}", file=sys.stderr)
w = np.array([x for x in widths if x > 0])
print(f"★ поле: кривых {C['кривых']}; честных при 3 px {C['честных при 3 px']}, при max(3, {a.k}·ширина) "
      f"{C['честных при мягком допуске']} (+{C['честных при мягком допуске'] - C['честных при 3 px']})")
print(f"  ширина штриха эталона (px): p25 {np.percentile(w,25):.1f}, медиана {np.median(w):.1f}, p75 {np.percentile(w,75):.1f}, "
      f"p90 {np.percentile(w,90):.1f}; допуск > 3 px у кривых шире {3/a.k:.1f} px — {100*np.mean(w > 3/a.k):.0f}%")
print("  прибавка по семействам: " + ", ".join(f"{k} {v}" for k, v in gained.most_common(12)))
