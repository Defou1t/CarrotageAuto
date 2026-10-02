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
ap.add_argument("--sheets", default="wellmap_sheets.txt", help="список листов в --ts (поле; сорт A — holdoutA_sheets.txt)")
ap.add_argument("--d2", action="store_true", help="★ 02.10 (§6.257): мера «до линии трассы на плоскости» — ошибка строки эталона = "
                "расстояние от точки эталона до ломаной трассы (уплотнённой по x), а не |Δx| по строке; честно — медиана ≤ 3 px")
ap.add_argument("--dump-gained", default="", help="★ 02.10: pickle прибавки по нормали — (лист, кривая эталона, кривая выдачи, "
                "медиана |Δ| px, медиана |Δ|/допуск) для просмотра")
ap.add_argument("--normal", type=float, default=0.0, help="★ 02.10 (§6.257): ещё и допуск ПО НОРМАЛИ — N px от линии эталона, "
                "то есть N·√(1+s²) по строке при наклоне s (0 — не считать)")
a = ap.parse_args()
TS = Path(a.ts)
smap = pickle.load(open(TS / a.map, "rb"))
# ★ карта слотов `slotmap.pkl` — только поле; для прочих листов трек берём из пуловых дампов, как `_name_cost_prod.py`
POOLS = ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_heldout", "pools_all"]
for root in POOLS:
    for f in sorted((TS / root).glob("*.pkl")):
        if f.stem in smap:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        smap.setdefault(d["name"], {s["name"]: s["track"] for s in d["slots"]})
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
field = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()][::a.every]


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def st_norm(tr, gt, tol):
    """★ 02.10: медиана |Δ| / допуск строки (≤ 1 — честно по нормали) и покрытие"""
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) / tol[y] for y in com])), len(com) / max(1, len(gt))


def dense_pts(tr, gap=30):
    """★ 02.10: точки ломаной трассы с шагом ≤ 1 px по обеим осям (крутой отрезок строки — много точек по x)"""
    ys = np.array(sorted(tr), float)
    if len(ys) < 2:
        return np.array([[ys[0], tr[ys[0]]]]) if len(ys) else np.zeros((0, 2))
    xs = np.array([tr[int(y)] for y in ys], float)
    out = [np.stack([ys, xs], 1)]
    dy = np.diff(ys); dx = np.diff(xs)
    for i in np.flatnonzero((dy <= gap) & (np.abs(dx) > 1)):
        n = int(np.ceil(abs(dx[i])))
        t = np.arange(1, n) / n
        out.append(np.stack([ys[i] + dy[i] * t, xs[i] + dx[i] * t], 1))
    return np.concatenate(out)


def held(tr, gt, tree=None):
    """★ 02.10: доли строк эталона (из общих с трассой), где |Δx| ≤ 3 px и где расстояние до ломаной трассы ≤ 3 px"""
    from scipy.spatial import cKDTree
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return 0.0, 0.0
    tree = tree or cKDTree(dense_pts(tr))
    d, _ = tree.query(np.array([[y, gt[y]] for y in com], float))
    h1 = float(np.mean([abs(tr[y] - gt[y]) <= 3.0 for y in com]))
    return h1, float(np.mean(d <= 3.0))


def st_d2(tr, gt, tree=None):
    """★ 02.10: медиана расстояния от точки эталона (строка, где есть трасса) до ломаной трассы на плоскости; покрытие — как `st`"""
    from scipy.spatial import cKDTree
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    tree = tree or cKDTree(dense_pts(tr))
    d, _ = tree.query(np.array([[y, gt[y]] for y in com], float))
    return float(np.median(d)), len(com) / max(1, len(gt))


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


C = Counter(); gained = Counter(); gainedN = Counter(); gainedD = Counter(); widths = []; GAINED = []; GAINED_D = []; HELD = []
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
        ok3, okw, okn, okd = {}, {}, {}, {}
        if a.d2:
            from scipy.spatial import cKDTree
            trees = {k: cKDTree(dense_pts(W[k])) for k in ws if len(W[k]) >= 2}
        for g in gs:
            tol = max(3.0, a.k * wid[g])
            if a.normal:
                gg = G[g]
                tn = {y: a.normal * math.sqrt(1.0 + ((gg.get(y + 1, x) - gg.get(y - 1, x)) / 2.0) ** 2) for y, x in gg.items()}
            for k in ws:
                m, c = st(W[k], G[g])
                ok3[(g, k)] = m is not None and m <= 3.0 and c >= 0.9
                okw[(g, k)] = m is not None and m <= tol and c >= 0.9
                if a.normal:
                    mn, cn = st_norm(W[k], G[g], tn)
                    okn[(g, k)] = mn is not None and mn <= 1.0 and cn >= 0.9
                if a.d2 and k in trees:
                    md, cd = st_d2(W[k], G[g], trees[k])
                    okd[(g, k)] = md is not None and md <= 3.0 and cd >= 0.9
        m3 = match(gs, ws, ok3); mw = match(gs, ws, okw)
        C["кривых"] += len(gs); C["честных при 3 px"] += len(m3); C["честных при мягком допуске"] += len(mw)
        if a.d2:
            md_ = match(gs, ws, okd)
            C["честных на плоскости"] += len(md_)
            for g, k_ in m3.items():
                if k_ in trees:
                    HELD.append(("честна при 3 px",) + held(W[k_], G[g], trees[k_]))
            for g, k_ in md_.items():
                if g not in m3 and k_ in trees:
                    HELD.append(("прибавка на плоскости",) + held(W[k_], G[g], trees[k_]))
            for g in gs:
                if g in md_ and g not in m3:
                    gainedD[M.mnem_root(g)] += 1
                    k_ = md_[g]
                    GAINED_D.append((sh, g, k_, st(W[k_], G[g])[0], st_d2(W[k_], G[g], trees[k_])[0]))
                if g in m3 and g not in md_:
                    C["потеряно на плоскости"] += 1
        if a.normal:
            mn_ = match(gs, ws, okn)
            C["честных по нормали"] += len(mn_)
            for g in gs:
                if g in mn_ and g not in m3:
                    gainedN[M.mnem_root(g)] += 1
                    k_ = mn_[g]
                    gt_ = G[g]
                    tn_ = {y: a.normal * math.sqrt(1.0 + ((gt_.get(y + 1, x) - gt_.get(y - 1, x)) / 2.0) ** 2)
                           for y, x in gt_.items()}
                    GAINED.append((sh, g, k_, st(W[k_], gt_)[0], st_norm(W[k_], gt_, tn_)[0]))
        for g in gs:
            if g in mw and g not in m3:
                gained[M.mnem_root(g)] += 1
    if si % 100 == 0:
        print(f"  … {si}/{len(field)}", file=sys.stderr)
w = np.array([x for x in widths if x > 0])
print(f"★ {a.sheets}: кривых {C['кривых']}; честных при 3 px {C['честных при 3 px']}, при max(3, {a.k}·ширина) "
      f"{C['честных при мягком допуске']} (+{C['честных при мягком допуске'] - C['честных при 3 px']})")
print(f"  ширина штриха эталона (px): p25 {np.percentile(w,25):.1f}, медиана {np.median(w):.1f}, p75 {np.percentile(w,75):.1f}, "
      f"p90 {np.percentile(w,90):.1f}; допуск > 3 px у кривых шире {3/a.k:.1f} px — {100*np.mean(w > 3/a.k):.0f}%")
print("  прибавка по семействам: " + ", ".join(f"{k} {v}" for k, v in gained.most_common(12)))
if a.dump_gained:
    pickle.dump(GAINED_D if a.d2 else GAINED, open(a.dump_gained, "wb"))
if a.d2 and HELD:
    for grp in ("честна при 3 px", "прибавка на плоскости"):
        H = np.array([h[1:] for h in HELD if h[0] == grp])
        if len(H):
            print(f"  {grp}: {len(H)}; доля строк |Δx| ≤ 3 px — медиана {np.median(H[:, 0]):.2f}; на плоскости ≤ 3 px — "
                  f"медиана {np.median(H[:, 1]):.2f}, квартили {np.percentile(H[:, 1], 25):.2f}–{np.percentile(H[:, 1], 75):.2f}, "
                  f"< 0.7 у {np.mean(H[:, 1] < 0.7):.0%}")
if a.d2:
    print(f"★ на плоскости (расстояние от точки эталона до ломаной трассы, медиана ≤ 3 px): честных {C['честных на плоскости']} "
          f"({C['честных на плоскости'] - C['честных при 3 px']:+d} к 3 px; потеряно из честных при 3 px {C['потеряно на плоскости']}); "
          "прибавка: " + ", ".join(f"{k} {v}" for k, v in gainedD.most_common(12)))
if a.normal:
    print(f"★ по нормали {a.normal:g} px (N·√(1+s²) по строке): честных {C['честных по нормали']} "
          f"({C['честных по нормали'] - C['честных при 3 px']:+d} к 3 px); прибавка: " +
          ", ".join(f"{k} {v}" for k, v in gainedN.most_common(12)))
