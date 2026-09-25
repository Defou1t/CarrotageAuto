r"""_name_value.py — ПРАВДОПОДОБИЕ ЗНАЧЕНИЙ ПО СЕМЕЙСТВУ КАК ПРИЗНАК РАСКЛАДКИ (21.09, B4).

ОТКУДА. §6.214: 256 честных кривых лежат под именем ЧУЖОГО корня на том же треке (OGZ↔GZ 55, SP↔GZ 26,
NGK→GK 12, DS→GZ 11, TAU→ZAT 11, DTP↔T 15 …). У большинства таких пар ШКАЛЫ РАЗНЫЕ (NGK/GK 98%, SP/GZ 97%,
DS/GZ, T/DTP, TAU/ZAT — 100%): одна и та же кривая, прочитанная через шкалу чужого слота, даёт значения
другой величины. Если у семейств характерные диапазоны значений различны, кривая «сама скажет», чей она слот.

ЧТО СЧИТАЕТ:
  1. по ЭТАЛОНУ — распределение медианы значения кривой по семействам (значение = v_left + (x − x_left) /
     px_width · (v_right − v_left) по базовой шкале слота), log10 для положительных величин;
  2. разделимость: для каждой пары семейств, встречающихся на одном треке, — AUC медианы log-значения;
  3. по ВЫДАЧЕ G: для честной кривой под чужим именем k (эталон g) — правдоподобие её значений через
     шкалу k в семействе k против правдоподобия через шкалу g в семействе g (гауссова плотность по log-
     медиане, параметры по эталону БЕЗ этой скважины); сколько ошибок признак развернул бы.
"""
import sys, argparse, pickle, hashlib, re, math
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
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
WELL = {}


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


def value_of(d, s):
    """медиана значения кривой через базовую шкалу s (None, если шкалы нет)"""
    if not d or s is None:
        return None
    xl, xr, vl, vr = s.get("x_left"), s.get("x_right"), s.get("v_left"), s.get("v_right")
    if None in (xl, xr, vl, vr) or xr == xl:
        return None
    med = float(np.median(list(d.values())))
    return vl + (med - xl) / (xr - xl) * (vr - vl)


def logv(v):
    return math.log10(max(v, 0.05)) if v is not None else None


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q; WELL[q.name] = wlg.parent.name
sheets = sorted({r[0] for r in trk})

# ── 1. распределения по эталону ────────────────────────────────────────────────────────────────
fam_vals = defaultdict(list)          # root → [(well, logv)]
per_sheet = {}
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    G = extract(str(q))
    ax = {str(s.get("name", "")).strip(): s for s in G.get("scale_axes", [])}
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    per_sheet[sh] = (G, ax, gts)
    for nm, d in gts.items():
        s = ax.get(sa_suffix(nm) or "")
        v = logv(value_of(d, s))
        if v is not None:
            fam_vals[M.mnem_root(nm)].append((WELL.get(sh, "?"), v))
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

print("★ ЭТАЛОН: медиана log10-значения по семействам (n ≥ 15):")
print("| семейство | кривых | p10 | p50 | p90 |")
print("|---|---|---|---|---|")
for root, lst in sorted(fam_vals.items(), key=lambda kv: -len(kv[1])):
    if len(lst) < 15:
        continue
    v = np.array([x for _, x in lst])
    print(f"| {root} | {len(v)} | {np.percentile(v,10):.2f} | {np.median(v):.2f} | {np.percentile(v,90):.2f} |")


def stats(root, exclude_well):
    v = np.array([x for w, x in fam_vals.get(root, []) if w != exclude_well])
    if len(v) < 8:
        return None
    return float(np.mean(v)), float(np.std(v) + 0.15)


def loglik(v, ms):
    if v is None or ms is None:
        return None
    m, s = ms
    return -0.5 * ((v - m) / s) ** 2 - math.log(s)


# ── 3. по выдаче: развернул бы признак ошибку? ─────────────────────────────────────────────────
C = Counter(); conf = Counter()
for sh, (G, ax, gts) in per_sheet.items():
    W = read_out(a.dir, sh)
    if W is None:
        continue
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    well = WELL.get(sh, "?")
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        Wt = [k for k in W if tm.get(k) == t]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        for g, k in mt.items():
            own = g in W and ok.get((g, g), False)
            rg, rk = M.mnem_root(g), M.mnem_root(k)
            if own or rg == rk:
                continue
            C["чужой корень"] += 1
            d = W[k]                                  # честная кривая эталона g, лежит под именем k
            v_k = logv(value_of(d, ax.get(sa_suffix(k) or "")))
            v_g = logv(value_of(d, ax.get(sa_suffix(g) or "")))
            ll_k = loglik(v_k, stats(rk, well)); ll_g = loglik(v_g, stats(rg, well))
            if ll_k is None or ll_g is None:
                C["  нет распределения/шкалы"] += 1
                continue
            if ll_g > ll_k:
                C["  признак предпочёл бы ВЕРНОЕ имя"] += 1; conf[(rg, rk, "верно")] += 1
            else:
                C["  признак оставил бы неверное"] += 1; conf[(rg, rk, "неверно")] += 1
            # и наоборот: не разворачивает ли он ВЕРНЫЕ имена? — проверяем на честных под своим именем
        for g, k in mt.items():
            own = g in W and ok.get((g, g), False)
            if not own:
                continue
            rg = M.mnem_root(g)
            d = W[g]
            v_g = logv(value_of(d, ax.get(sa_suffix(g) or "")))
            ll_g = loglik(v_g, stats(rg, well))
            # худший конкурент среди других корней того же трека
            worst = None
            for k2 in ns:
                rk2 = M.mnem_root(k2)
                if rk2 == rg:
                    continue
                v2 = logv(value_of(d, ax.get(sa_suffix(k2) or "")))
                ll2 = loglik(v2, stats(rk2, well))
                if ll2 is not None and (worst is None or ll2 > worst):
                    worst = ll2
            if ll_g is None or worst is None:
                continue
            C["верных имён с конкурентом другого корня"] += 1
            C["  признак увёл бы верное имя к чужому корню"] += (worst > ll_g)

print(f"\n★★ ВЫДАЧА G: {dict(C)}")
print("\n★ по парам (эталон, выдано): признак верно / неверно")
for (rg, rk, res), n in sorted(conf.items(), key=lambda kv: -kv[1])[:20]:
    print(f"   {rg:>5} под именем {rk:<5} {res:<8} {n}")
