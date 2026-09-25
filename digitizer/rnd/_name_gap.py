r"""_name_gap.py — ГДЕ ТЕРЯЮТСЯ ИМЕНА: разрыв между безымянным 1:1 и именным счётом по составу (21.09).

ОТКУДА ВОПРОС. Тестовый прогон BEZLUD_051 (BK, IK): геометрия 3 кривых из 4 верна (≤ 2 px), а имя верно
у одной — три IK-кривые (IK1/IKA1/IKR1, две шкалы DA1/DA2) переставлены раскладкой. На поле 1123
листов безымянных 1144, именных 750: разрыв 394 кривые — треть безымянного счёта. Что это за разрыв?

ЧТО СЧИТАЕТ (по замороженной выдаче, без растров): для каждой эталонной кривой, взятой безымянным 1:1
(`match`, как в `_name_cost_prod.py`), — честна ли выданная кривая ПОД ЕЁ ИМЕНЕМ; если нет — под каким
именем лежит честная: тот же корень мнемоники (перестановка в семействе: IK1 ↔ IKR1), тот же трек, но
другой корень, или имя вовсе не выдано. Колонки по K трека и по семейству.

  <ComfyUI>\python_embeded\python.exe _name_gap.py --dir ab_pregate/G
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
ap.add_argument("--dir", default="ab_pregate/G")
ap.add_argument("--sheets", default="")
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
    """→ {эталон: колонка} максимального 1:1"""
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
    out = {}
    for r in rows:
        if try_(r, set()):
            pass
    for c, r in pair.items():
        out[r] = c
    return out


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
if a.sheets:
    want = {l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()}
    sheets = [s for s in sheets if s in want]

CATS = ["имя верно", "перестановка в семействе (тот же корень)", "имя другого корня на том же треке", "честная есть, а под своим именем ничего/не честно (иное)"]
byK = defaultdict(Counter); byFam = defaultdict(Counter); tot = Counter(); conf2 = Counter(); conf1 = Counter(); col_mis = Counter()
n_un = n_named = 0; done = 0
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
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
        Wt = [k for k in W if tm.get(k) == t]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        K = min(len(ns), 5)
        full = len(mt) == len(ns)          # весь трек взят 1:1 (набор верен) или лишь часть
        for g, k in mt.items():
            n_un += 1
            own = g in W and ok.get((g, g), False)
            if own:
                cat = CATS[0]; n_named += 1
            elif M.mnem_root(k) == M.mnem_root(g):
                cat = CATS[1]
            elif k in Wt:
                cat = CATS[2]
            else:
                cat = CATS[3]
            byK[K][cat] += 1; byFam[M.mnem_root(g)][cat] += 1; tot[cat] += 1
            tot[('полный трек' if full else 'частичный трек', cat)] += 1
            if cat == CATS[2]:
                conf2[(M.mnem_root(g), M.mnem_root(k))] += 1
            elif cat == CATS[1]:
                conf1[(g.split()[0], k.split()[0])] += 1
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

print(f"\nлистов {done}; безымянных 1:1 {n_un}, из них именно верно {n_named} (разрыв {n_un - n_named})")
print("\n★★ КУДА УХОДИТ ИМЯ У КРИВОЙ, ВЗЯТОЙ БЕЗЫМЯННЫМ 1:1 (по K трека)")
print("| K | взято 1:1 | " + " | ".join(CATS) + " |")
print("|---|---|" + "---|" * len(CATS))
for K in sorted(byK):
    n = sum(byK[K].values())
    print(f"| {K}{'+' if K == 5 else ''} | {n} | " + " | ".join(str(byK[K][c]) for c in CATS) + " |")
print(f"| **всего** | {n_un} | " + " | ".join(f"**{tot[c]}**" for c in CATS) + " |")
print("\n★ ПО СЕМЕЙСТВАМ (корень мнемоники), где разрыв больше 10 кривых:")
rows = sorted(byFam.items(), key=lambda kv: -(sum(kv[1].values()) - kv[1][CATS[0]]))
print("| семейство | взято 1:1 | имя верно | перестановка в семействе | другой корень | иное |")
print("|---|---|---|---|---|---|")
for fam, c in rows:
    n = sum(c.values()); gap = n - c[CATS[0]]
    if gap < 10:
        continue
    print(f"| {fam} | {n} | {c[CATS[0]]} | {c[CATS[1]]} | {c[CATS[2]]} | {c[CATS[3]]} |")

print("\n★ ЧУЖОЙ КОРЕНЬ: под каким именем лежит честная кривая эталона (эталон → выдача), топ-25:")
for (g, k), n in conf2.most_common(25):
    print(f"   {g:>6} → {k:<6} {n}")
print("\n★ ПЕРЕСТАНОВКИ В СЕМЕЙСТВЕ (эталон → выдача), топ-20:")
for (g, k), n in conf1.most_common(20):
    print(f"   {g:>6} → {k:<6} {n}")

print("\n★★ ПОЛНЫЙ ТРЕК (все K кривых взяты 1:1) ПРОТИВ ЧАСТИЧНОГО — где теряются имена:")
for full in ("полный трек", "частичный трек"):
    print(f"   {full}: " + ", ".join(f"{c}: {tot[(full, c)]}" for c in CATS if tot[(full, c)]))
