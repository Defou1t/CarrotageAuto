r"""_loss_ledger.py — КУДА УХОДЯТ КРИВЫЕ: ПОСТАТЕЙНЫЙ УЧЁТ ПОТЕРЬ ТОЧНОСТИ НА ЗАМОРОЖЕННОМ ПОЛЕ (21.09).

Заказчик: «делай параллельно анализ, почему у нас не хватает точности». Одна таблица на всё поле 1123 листов /
2737 эталонных кривых: каждая кривая относится РОВНО к одной статье, по выдаче прода (после §6.213 — `ab_slot/RA`):
  ★ взята 1:1 и имя верно            — успех;
  ★ взята 1:1, имя неверно           — успех геометрии, потеря имени (§6.214);
  ✘ короткая: med ≤ 3 px, покрытие < 0.9 — трасса верна, но не дотянута (обрыв, перевынос);
  ✘ рядом: 3 < med ≤ 10 px             — почти: смещение/дрожание больше допуска;
  ✘ промах 10–100 px                    — соседняя линия / частично чужая;
  ✘ далеко > 100 px                     — не та линия;
  ✘ на треке нет выданных кривых        — трек пуст (U1 не нашёл линий / раскладка ничего не назначила);
  ✘ есть кривые, но ни одна не пересекается ≥ 30 строк — выдача не там по глубине.
Для каждой статьи — сколько из этих кривых честно берёт ДРУГОЙ путь (декодер всегда, `ab_rdhonest/B`) —
это «резерв выбора пути». Разрезы: по K трека и по семейству.

  <ComfyUI>\python_embeded\python.exe _loss_ledger.py --dir ab_slot/RA
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
ap.add_argument("--alt", default="ab_rdhonest/B", help="другой путь для колонки «берёт декодер»")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
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


CATS = ["★ взята, имя верно", "★ взята, имя неверно", "✘ короткая (med ≤ 3, покрытие < 0.9)",
        "✘ рядом (3 < med ≤ 10)", "✘ промах 10–100", "✘ далеко > 100", "✘ трек пуст", "✘ нет пересечения по глубине"]
trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})
tot = Counter(); alt_ok = Counter(); byK = defaultdict(Counter); byF = defaultdict(Counter); byKalt = defaultdict(Counter)
n_gt = 0; done = 0
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    A = read_out(a.alt, sh) or {}
    done += 1
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
        At = [k for k in A if tm.get(k) == t and A[k]]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        oka = {(g, k): HON(*st(A[k], gts[g])) for g in ns for k in At}
        mt = match(ns, Wt, ok); mta = match(ns, At, oka)
        K = min(len(ns), 5)
        for g in ns:
            n_gt += 1
            if g in mt:
                cat = CATS[0] if (g in W and ok.get((g, g), False)) else CATS[1]
            elif not Wt:
                cat = CATS[6]
            else:
                best = None
                for k in Wt:
                    m, c = st(W[k], gts[g])
                    if m is None:
                        continue
                    key = (0 if m <= 3 else 1, m)
                    if best is None or key < best[0]:
                        best = (key, m, c)
                if best is None:
                    cat = CATS[7]
                else:
                    _, m, c = best
                    cat = CATS[2] if m <= 3 else CATS[3] if m <= 10 else CATS[4] if m <= 100 else CATS[5]
            tot[cat] += 1; byK[K][cat] += 1; byF[M.mnem_root(g)][cat] += 1
            if g in mta:
                alt_ok[cat] += 1; byKalt[K][cat] += 1
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

print(f"\nлистов {done}, эталонных кривых {n_gt}; выдача {a.dir}, другой путь {a.alt}")
print("\n★★ УЧЁТ ПОТЕРЬ (каждая кривая — в одной статье)")
print("| статья | кривых | % поля | из них честно берёт другой путь |")
print("|---|---|---|---|")
for c in CATS:
    print(f"| {c} | {tot[c]} | {100*tot[c]/max(1,n_gt):.1f}% | {alt_ok[c]} |")
print(f"| **всего** | {n_gt} | 100% | {sum(alt_ok.values())} |")
print("\n★ ПО K ТРЕКА (кривых; в скобках — берёт другой путь)")
print("| K | всего | " + " | ".join(c for c in CATS) + " |")
print("|---|---|" + "---|" * len(CATS))
for K in sorted(byK):
    n = sum(byK[K].values())
    print(f"| {K}{'+' if K == 5 else ''} | {n} | " + " | ".join(f"{byK[K][c]} ({byKalt[K][c]})" for c in CATS) + " |")
print("\n★ ПО СЕМЕЙСТВАМ (n ≥ 40), доля успеха геометрии = взята 1:1")
print("| семейство | кривых | взята 1:1 | имя верно | короткая | рядом | 10–100 | > 100 | трек пуст / нет пересечения |")
print("|---|---|---|---|---|---|---|---|---|")
for fam, c in sorted(byF.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(c.values())
    if n < 40:
        continue
    print(f"| {fam} | {n} | {c[CATS[0]]+c[CATS[1]]} ({100*(c[CATS[0]]+c[CATS[1]])/n:.0f}%) | {c[CATS[0]]} | {c[CATS[2]]} | {c[CATS[3]]} | {c[CATS[4]]} | {c[CATS[5]]} | {c[CATS[6]]+c[CATS[7]]} |")
