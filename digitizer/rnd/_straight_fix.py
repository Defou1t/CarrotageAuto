r"""_straight_fix.py — ПРЫЖКИ ТРАССЫ НА ПРЯМЫЕ ВЕРТИКАЛИ (сетка, рамка): поиск без эталона и штопка интерполяцией (§6.218, 25.09).

ОТКУДА. `_swap_anatomy.py`: из 1376 не взятых кривых многокривых треков 47% идут ≥ половину строк по туши вне эталона, 32% —
смесь. На конкретных листах (DRUZH_053 GK/NGK, PDGRAK_002 RK, BOGAT_011 BKZ/DS) это ТЕЛЕПОРТ: трасса на десятки–сотни строк
перескакивает на толстую вертикаль сетки/рамки (x постоянен до пикселя) и возвращается. Настоящая кривая так не рисует:
прямой отрезок у неё бывает (каверномер на номинале), но без скачка в сотни px на входе и выходе.

ПРАВИЛО (эталон не используется): в выданной трассе (строки подряд, после `dense`) ищутся отрезки, где x держится в
±`--tol` px не меньше `--min-len` строк, И на входе и на выходе отрезка x скачет не меньше чем на `--jump` px относительно
соседних строк вне отрезка. Такой отрезок заменяется линейной интерполяцией между последней строкой до и первой после.
Счёт — безымянный 1:1 по треку и именной, до/после, на замороженной выдаче; разбор по листам ↑/↓.

  <ComfyUI>\python_embeded\python.exe _straight_fix.py --dir ab_slot/RA
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
ap.add_argument("--sheets", default="", help="ограничить списком (например holdoutA_sheets.txt)")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--tol", type=float, default=1.0)
ap.add_argument("--min-len", type=int, default=20)
ap.add_argument("--jump", type=float, default=40.0)
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
    return sum(1 for r in rows if try_(r, set()))


def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return None
    return {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}


def fix(tr):
    """→ (исправленная трасса, число заштопанных строк)"""
    if len(tr) < a.min_len + 2:
        return tr, 0
    ys = np.array(sorted(tr)); x = np.array([tr[y] for y in ys], float)
    n = len(x); out = x.copy(); fixed = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(x[j + 1] - x[i]) <= a.tol and ys[j + 1] - ys[j] <= 3:
            j += 1
        L = j - i + 1
        if L >= a.min_len and i > 0 and j < n - 1:
            jin = abs(x[i] - x[i - 1]); jout = abs(x[j + 1] - x[j])
            if jin >= a.jump and jout >= a.jump:
                y0, y1 = ys[i - 1], ys[j + 1]; x0, x1 = x[i - 1], x[j + 1]
                out[i:j + 1] = x0 + (x1 - x0) * (ys[i:j + 1] - y0) / max(1, (y1 - y0))
                fixed += L
        i = j + 1
    return dict(zip(ys.tolist(), out.tolist())), fixed


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
C = Counter(); per = []; byK = defaultdict(lambda: [0, 0])
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    W2 = {}
    for k, tr in W.items():
        W2[k], f = fix(tr) if tr else (tr, 0)
        if f:
            C["кривых заштопано"] += 1; C["строк заштопано"] += f
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    u0 = u1 = 0
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        Wt = [k for k in W if tm.get(k) == t and W[k]]
        ok0 = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        ok1 = {(g, k): HON(*st(W2[k], gts[g])) for g in ns for k in Wt}
        m0, m1 = match(ns, Wt, ok0), match(ns, Wt, ok1)
        u0 += m0; u1 += m1
        K = min(len(ns), 5); byK[K][0] += m0; byK[K][1] += m1
    n0 = sum(1 for g in gts if g in W and W[g] and HON(*st(W[g], gts[g])))
    n1 = sum(1 for g in gts if g in W2 and W2[g] and HON(*st(W2[g], gts[g])))
    per.append((u0, u1, n0, n1))
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

P = np.array(per)
du, dn = P[:, 1] - P[:, 0], P[:, 3] - P[:, 2]
rng = np.random.default_rng(0)


def pp(d):
    nz = d[d != 0]
    if not len(nz):
        return 1.0
    sims = (rng.choice([-1, 1], size=(100000, len(nz))) * np.abs(nz)).sum(1)
    return float((np.abs(sims) >= abs(d.sum())).mean())


print(f"\n★ ПАРАМЕТРЫ: tol {a.tol} px, min-len {a.min_len} строк, jump {a.jump} px; листов {len(per)}; {dict(C)}")
print(f"★★ БЕЗЫМЯННЫХ {P[:,0].sum()} → {P[:,1].sum()} ({du.sum():+d}), листов ↑{(du>0).sum()}/↓{(du<0).sum()}, p = {pp(du):.4f}")
print(f"★★ ИМЕННЫХ    {P[:,2].sum()} → {P[:,3].sum()} ({dn.sum():+d}), листов ↑{(dn>0).sum()}/↓{(dn<0).sum()}, p = {pp(dn):.4f}")
print("   по K (безымянных): " + ", ".join(f"K={K}{'+' if K == 5 else ''}: {v[0]}→{v[1]}" for K, v in sorted(byK.items())))
