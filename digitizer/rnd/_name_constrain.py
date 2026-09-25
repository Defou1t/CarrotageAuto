r"""_name_constrain.py — СКОЛЬКО ИМЁН ВЕРНУТ ЖЁСТКИЕ ОГРАНИЧЕНИЯ ШАБЛОНА (полоса шкалы + ноль шкалы) ПРИ НЫНЕШНЕЙ ГЕОМЕТРИИ (21.09, B4).

Оценка без модели раскладки: в каждом треке замороженной выдачи G набор выданных кривых фиксирован, имена
слотов фиксированы; ищется перестановка имён, которая (1) удовлетворяет ограничениям шаблона для каждой пары
(кривая в полосе своей шкалы ±pad; для неотрицательных величин со сдвинутым нулём — не левее нуля более чем на
`--viol` долю строк), (2) среди допустимых — минимально отличается от нынешней раскладки (модель «остальное
оставляем как есть»). Меряется именной честный счёт до/после; безымянный не меняется по построению.
Перебор: треки до 8 слотов — все перестановки; больше — жадно (таких единицы).

  <ComfyUI>\python_embeded\python.exe _name_constrain.py
  <ComfyUI>\python_embeded\python.exe _name_constrain.py --no-band     # только ноль
  <ComfyUI>\python_embeded\python.exe _name_constrain.py --no-zero     # только полоса
"""
import sys, argparse, pickle, hashlib, re, itertools
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--dir", default="ab_pregate/G")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--pad", type=float, default=15.0)
ap.add_argument("--viol", type=float, default=0.5)
ap.add_argument("--no-band", action="store_true")
ap.add_argument("--no-zero", action="store_true")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
SIGNED = {"MV", "M", "US", "US/M", "DB/M", "UE", "G/CM3", "MKS", "MKS/M", "%"}
WELL = {}


def sa_suffix(nm):
    m = re.search(r"(DA\d+\s+SA\d+)\s*$", nm)
    return m.group(1) if m else None


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return None
    return {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}


def slot_geom(s):
    """(x_left, x_right, zero_x или None)"""
    xl, xr = s.get("x_left"), s.get("x_right")
    if xl is None or xr is None:
        return None
    u = str(s.get("units", "")).upper()
    vl, vr = s.get("v_left"), s.get("v_right")
    zx = None
    if u not in SIGNED and vl is not None and vr is not None and vr != vl and vl < 0:
        zx = float(xl) + (0.0 - vl) / (vr - vl) * (xr - xl)
    return (float(xl), float(xr), zx)


def allowed(d, geom):
    if geom is None or not d:
        return True
    xs = np.fromiter(d.values(), float)
    xl, xr, zx = geom
    med = float(np.median(xs))
    if not a.no_band and not (xl - a.pad <= med <= xr + a.pad):
        return False
    if not a.no_zero and zx is not None and float(np.mean(xs < zx - a.pad)) > a.viol:
        return False
    return True


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q; WELL[q.name] = wlg.parent.name
sheets = sorted({r[0] for r in trk})
per_sheet = {}; C = Counter(); fam = Counter()
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    G = extract(str(q))
    ax = {str(s.get("name", "")).strip(): s for s in G.get("scale_axes", [])}
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    geom = {}
    for nm in set(gts) | set(W):
        suf = sa_suffix(nm)
        if suf and suf in ax:
            geom[nm] = slot_geom(ax[suf])
    tm = smap.get(sh, {})
    W2 = dict(W)
    bytr = defaultdict(list)
    for nm in W:
        t = tm.get(nm)
        if t is not None and W[nm]:
            bytr[t].append(nm)
    for t, names in bytr.items():
        if (sh, t) not in KOF or len(names) < 2:
            continue
        cur_ok = all(allowed(W[nm], geom.get(nm)) for nm in names)
        if cur_ok:
            continue
        C["треков с нарушением"] += 1
        curves = [W[nm] for nm in names]
        best = None
        if len(names) <= 8:
            for perm in itertools.permutations(range(len(names))):
                if all(allowed(curves[j], geom.get(names[i])) for i, j in enumerate(perm)):
                    changes = sum(1 for i, j in enumerate(perm) if i != j)
                    if best is None or changes < best[0]:
                        best = (changes, perm)
        if best is None:
            C["треков без допустимой перестановки"] += 1
            continue
        C["треков переставлено"] += 1
        for i, j in enumerate(best[1]):
            W2[names[i]] = curves[j]
    b = sum(1 for g in gts if g in W and W[g] and HON(*st(W[g], gts[g])))
    af = sum(1 for g in gts if g in W2 and W2[g] and HON(*st(W2[g], gts[g])))
    per_sheet[sh] = (b, af)
    for g in gts:
        r = M.mnem_root(g)
        fam[(r, "было")] += g in W and bool(W[g]) and HON(*st(W[g], gts[g]))
        fam[(r, "стало")] += g in W2 and bool(W2[g]) and HON(*st(W2[g], gts[g]))
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

B = sum(v[0] for v in per_sheet.values()); A = sum(v[1] for v in per_sheet.values())
d = np.array([v[1] - v[0] for v in per_sheet.values()])
up, dn = int((d > 0).sum()), int((d < 0).sum())
rng = np.random.default_rng(0); nz = d[d != 0]
p = float((np.abs(np.array([(nz * rng.choice([-1, 1], size=len(nz))).sum() for _ in range(100000)])) >= abs(d.sum())).mean()) if len(nz) else 1.0
wells = defaultdict(int)
for sh, (b_, a_) in per_sheet.items():
    wells[WELL.get(sh, "?")] += a_ - b_
wl = sorted(wells.values())
print(f"\n★★ ОГРАНИЧЕНИЯ ШАБЛОНА ({'полоса ' if not a.no_band else ''}{'ноль' if not a.no_zero else ''}): {dict(C)}")
print(f"   именных честных {B} → {A} ({A-B:+d}), листов ↑{up}/↓{dn}, p = {p:.4f}; "
      f"скважин в плюсе {sum(1 for v in wl if v > 0)}, в минусе {sum(1 for v in wl if v < 0)}, худшая {wl[0]:+d}, лучшая {wl[-1]:+d}")
print("| семейство | было | стало | Δ |")
print("|---|---|---|---|")
roots = sorted({r for r, _ in fam}, key=lambda r: -abs(fam[(r, 'стало')] - fam[(r, 'было')]))
for r in roots:
    if fam[(r, "стало")] != fam[(r, "было")]:
        print(f"| {r} | {fam[(r, 'было')]} | {fam[(r, 'стало')]} | {fam[(r, 'стало')] - fam[(r, 'было')]:+d} |")
