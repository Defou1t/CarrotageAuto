r"""_near_miss.py — «РЯДОМ» (3 < med ≤ 10 px): СМЕЩЕНИЕ ИЛИ ДРОЖАНИЕ? (21.09, анализ точности).

`_loss_ledger.py`: 393 эталонных кривых (14.4% поля) имеют выданную кривую с медианной ошибкой 3…10 px — почти
честные. Если ошибка — систематическое смещение (знак постоянен вдоль кривой: трасса идёт по краю штриха, а
эксперт — по центру; или сдвиг калибровки), её лечит доводка; если это дрожание/перескоки — нет.
Для каждой такой кривой: знаковая медиана (трасса − эталон), доля строк с |ошибка| ≤ 3, IQR знаковой ошибки,
наклон ошибки по глубине (px на 1000 строк), «односторонность» = |медиана знаковой| / медиана |ошибки|.
Разрезы по семействам и по K; гистограмма медиан.
"""
import sys, argparse, pickle, hashlib
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
ap.add_argument("--dir", default="ab_slot/RA")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--lo", type=float, default=3.0)
ap.add_argument("--hi", type=float, default=10.0)
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


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


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})
rec = []          # (fam, K, med_abs, med_signed, frac3, iqr, slope, cov, same_name)
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        Wt = [k for k in W if tm.get(k) == t and W[k]]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        for g in ns:
            if g in mt or not Wt:
                continue
            best = None
            for k in Wt:
                m, c = st(W[k], gts[g])
                if m is None:
                    continue
                key = (0 if m <= 3 else 1, m)
                if best is None or key < best[0]:
                    best = (key, m, c, k)
            if best is None:
                continue
            _, m, c, k = best
            if not (a.lo < m <= a.hi):
                continue
            d = gts[g]; w = W[k]
            ys = np.array(sorted(y for y in w if y in d)); e = np.array([w[y] - d[y] for y in ys])
            frac3 = float(np.mean(np.abs(e) <= 3))
            iqr = float(np.percentile(e, 75) - np.percentile(e, 25))
            slope = float(np.polyfit(ys, e, 1)[0] * 1000) if len(ys) > 100 else 0.0
            rec.append((M.mnem_root(g), min(len(ns), 5), m, float(np.median(e)), frac3, iqr, slope, c, k == g))
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

R = np.array([(r[2], r[3], r[4], r[5], r[6], r[7]) for r in rec], float)
print(f"\nкривых «рядом» ({a.lo} < med ≤ {a.hi}): {len(rec)}; под своим именем {sum(1 for r in rec if r[8])}")
med_abs, med_sgn, frac3, iqr, slope, cov = R.T
one = np.abs(med_sgn) / np.maximum(med_abs, 1e-6)
print(f"  медиана |ошибки|: p25 {np.percentile(med_abs,25):.1f}, p50 {np.median(med_abs):.1f}, p75 {np.percentile(med_abs,75):.1f} px")
print(f"  односторонность |знаковая медиана| / медиана |ошибки|: p50 {np.median(one):.2f}; кривых с ≥ 0.8 (чистое смещение): {int((one >= 0.8).sum())} ({100*(one>=0.8).mean():.0f}%)")
print(f"  знак смещения: трасса ПРАВЕЕ эталона у {int((med_sgn > 0).sum())}, ЛЕВЕЕ у {int((med_sgn < 0).sum())}")
print(f"  доля строк с |ошибка| ≤ 3: p50 {np.median(frac3):.2f}; IQR знаковой ошибки p50 {np.median(iqr):.1f} px")
print(f"  наклон по глубине |px/1000 строк|: p50 {np.median(np.abs(slope)):.2f}; кривых с |наклон| > 2: {int((np.abs(slope) > 2).sum())}")
print(f"  покрытие: p50 {np.median(cov):.2f}; кривых с покрытием < 0.9: {int((cov < 0.9).sum())}")
print("\n  гистограмма медианы |ошибки| (px):", " ".join(f"{lo}-{lo+1}:{int(((med_abs > lo) & (med_abs <= lo+1)).sum())}" for lo in range(3, 10)))
print("\n★ ПО СЕМЕЙСТВАМ (n ≥ 10)")
print("| семейство | кривых | med |ошибки| p50 | знаковая p50 | односторонних ≥0.8 | правее / левее | IQR p50 | покрытие < 0.9 |")
print("|---|---|---|---|---|---|---|---|---|")
byf = defaultdict(list)
for i, r in enumerate(rec):
    byf[r[0]].append(i)
for fam, idx in sorted(byf.items(), key=lambda kv: -len(kv[1])):
    if len(idx) < 10:
        continue
    ii = np.array(idx)
    print(f"| {fam} | {len(idx)} | {np.median(med_abs[ii]):.1f} | {np.median(med_sgn[ii]):+.1f} | {int((one[ii] >= 0.8).sum())} | "
          f"{int((med_sgn[ii] > 0).sum())} / {int((med_sgn[ii] < 0).sum())} | {np.median(iqr[ii]):.1f} | {int((cov[ii] < 0.9).sum())} |")
print("\n★ ПО K")
byk = defaultdict(list)
for i, r in enumerate(rec):
    byk[r[1]].append(i)
for K in sorted(byk):
    ii = np.array(byk[K])
    print(f"  K={K}{'+' if K == 5 else ''}: {len(ii)} кривых, med |ошибки| p50 {np.median(med_abs[ii]):.1f}, односторонних {int((one[ii] >= 0.8).sum())}, IQR p50 {np.median(iqr[ii]):.1f}")
