r"""_name_zero.py — НУЛЬ ШКАЛЫ СЛОТА КАК ОГРАНИЧЕНИЕ РАСКЛАДКИ (21.09, B4).

ОТКУДА. Bilch_Volyts BKZ2: три слота на одной полосе (GZ4 SA1, OGZ SA2, GZ5 SA3), цвета и классы в
словаре одинаковы, полосы шкал одинаковы — но у SA3 шкала СДВИНУТА: v_left = −11, v_right = 8 Ohmm,
то есть ноль шкалы стоит на x ≈ 1002, а не у левого края трека. Сопротивление отрицательным не бывает
⇒ кривая слота GZ5 обязана лежать ПРАВЕЕ x ≈ 1002; на листе так и есть (x med 1183), а раскладка отдала
это имя кривой с x ≈ 350. Раскладка про ноль шкалы не знает (`slot_model.rows`: 14 признаков, нуля нет).

ЧТО СЧИТАЕТ по замороженной выдаче G и шаблонам:
  1. сколько слотов имеют сдвинутый ноль (v_left < 0) при неотрицательной величине (единицы не mV/m/us);
  2. санитарная проверка по ЭТАЛОНУ: доля строк эталонной кривой левее нуля своей шкалы (ожидание ≈ 0);
  3. по ВЫДАЧЕ: для слотов с честной кривой под своим именем и с неверным именем — доля строк левее нуля;
  4. сколько неверных имён нарушают ограничение (кривая под именем в основном левее нуля слота) —
     столько имён запрет исправил бы как минимум частично (перестановка внутри трека).
"""
import sys, argparse, pickle, hashlib, re
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
ap.add_argument("--pad", type=float, default=15.0, help="допуск левее нуля, px")
ap.add_argument("--viol", type=float, default=0.5, help="нарушение = доля строк левее нуля выше этой")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
from auto.slot_geom import SIGNED_UNITS as SIGNED   # ★ 26.09 (аудит): одна константа с продом (там есть «MM» — так измерен A/B §6.215)


def sa_suffix(nm):
    m = re.search(r"(DA\d+\s+SA\d+)\s*$", nm)
    return m.group(1) if m else None


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


def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return None
    return {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}


def zero_x(s):
    """x нуля шкалы; None, если величина знаковая или ноль не внутри/левее полосы"""
    u = str(s.get("units", "")).upper()
    if u in SIGNED:
        return None
    vl, vr = s.get("v_left"), s.get("v_right")
    xl, xr = s.get("x_left"), s.get("x_right")
    if vl is None or vr is None or xl is None or xr is None or vr == vl:
        return None
    if vl >= 0:
        return float(xl)                         # ноль у левого края (или левее) — не различает
    return float(xl) + (0.0 - vl) / (vr - vl) * (xr - xl)


def frac_left(d, zx):
    xs = np.fromiter(d.values(), float)
    return float(np.mean(xs < zx - a.pad)) if len(xs) else 0.0


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})
C = Counter(); gt_fl = []; ok_fl = []; bad_fl = []; fixable = Counter(); shifted_tracks = 0
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
    zx = {}
    for nm in set(gts) | set(W):
        suf = sa_suffix(nm)
        if suf and suf in ax:
            z = zero_x(ax[suf])
            if z is not None:
                zx[nm] = (z, ax[suf].get("v_left", 0) < 0)
    C["слотов с полосой"] += sum(1 for nm in gts if nm in zx)
    C["слотов со сдвинутым нулём (v_left < 0, неотриц. величина)"] += sum(1 for nm in gts if nm in zx and zx[nm][1])
    for nm, d in gts.items():
        if nm in zx and zx[nm][1] and d:
            gt_fl.append(frac_left(d, zx[nm][0]))
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        if any(nm in zx and zx[nm][1] for nm in ns):
            shifted_tracks += 1
        Wt = [k for k in W if tm.get(k) == t]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        for g, k in mt.items():
            own = g in W and ok.get((g, g), False)
            if g in W and W[g] and g in zx and zx[g][1]:
                fl = frac_left(W[g], zx[g][0])
                (ok_fl if own else bad_fl).append(fl)
                if not own and fl > a.viol:
                    fixable["неверное имя, кривая под ним левее нуля слота"] += 1
            if not own:
                C["неверных имён всего"] += 1
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

print(f"\n★ слотов с полосой {C['слотов с полосой']}, из них со сдвинутым нулём {C['слотов со сдвинутым нулём (v_left < 0, неотриц. величина)']}; "
      f"треков с хотя бы одним таким слотом {shifted_tracks}")
if gt_fl:
    g = np.array(gt_fl)
    print(f"★ ЭТАЛОН: доля строк левее нуля своей шкалы (−{a.pad:.0f} px): медиана {np.median(g):.3f}, "
          f"кривых с нарушением > {a.viol}: {int((g > a.viol).sum())} из {len(g)}")
if ok_fl:
    o = np.array(ok_fl); print(f"★ ВЫДАЧА, имя верно: кривых {len(o)}, с долей левее нуля > {a.viol}: {int((o > a.viol).sum())}")
if bad_fl:
    b = np.array(bad_fl); print(f"★ ВЫДАЧА, имя неверно: кривых {len(b)}, с долей левее нуля > {a.viol}: {int((b > a.viol).sum())}")
print(f"⇒ неверных имён всего {C['неверных имён всего']}; из них нарушают ноль своей шкалы (запрет их бы не допустил): {dict(fixable)}")
