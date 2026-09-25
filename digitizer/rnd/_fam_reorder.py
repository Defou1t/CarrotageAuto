r"""_fam_reorder.py — ПЕРЕИМЕНОВАНИЕ ВНУТРИ СЕМЕЙСТВА ПО ФОРМЕ: сколько имён вернёт правило «короткий зонд извилистее» (21.09, B4).

ОТКУДА. `_gz_order.py` по эталону: у GZ признак wiggle (медиана |dx| / размах) упорядочивает зонды по индексу
с парной точностью 80% (порядок трека целиком 66%), а нынешняя гипотеза раскладки «индекс ↔ x» — 21%.

ЧТО ДЕЛАЕТ. Берёт замороженную выдачу (G), в каждом треке для каждого корня с ≥ 2 выданными кривыми
разных индексов ПЕРЕИМЕНОВЫВАЕТ их: кривые сортируются по признаку (по убыванию), имена семейства —
по возрастанию индекса. Геометрия и безымянный счёт не меняются по построению; меняется только
ИМЕННОЙ счёт (честная кривая под своим именем). Печатает до/после, по семействам, листов ↑/↓ и
перестановочный p. ⚠ Правило без обучения — держать нечего; контроль — знак по скважинам.

  <ComfyUI>\python_embeded\python.exe _fam_reorder.py --feat wiggle
  <ComfyUI>\python_embeded\python.exe _fam_reorder.py --feat rev --fam GZ
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
ap.add_argument("--feat", default="wiggle", choices=["wiggle", "rev", "hf", "rough"])
ap.add_argument("--fam", default="", help="только эти корни через запятую (пусто = все)")
ap.add_argument("--min-ratio", type=float, default=1.0,
                help="переставлять пару только если признаки различаются не меньше чем в N раз (1.0 = всегда)")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
WELL = {}


def idx_of(nm):
    short = nm.split()[0]
    root = M.mnem_root(short)
    m = re.match(re.escape(root) + r"(\d+)$", short)
    if not m:
        return root, None
    digits = m.group(1)
    return root, int(digits[:-1]) if len(digits) >= 2 else None


def feat(d):
    ys = sorted(d)
    x = np.array([d[y] for y in ys], float)
    if len(x) < 50:
        return None
    dx = np.diff(x)
    span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
    if a.feat == "wiggle":
        return float(np.median(np.abs(dx))) / span
    if a.feat == "rev":
        return float(np.mean((dx[:-1] * dx[1:]) < 0))
    if a.feat == "hf":
        return float(np.mean(np.abs(np.diff(x, 2)))) / span
    sm = np.convolve(x, np.ones(101) / 101, mode="same")
    return float(np.std(x - sm)) / span


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


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q; WELL[q.name] = wlg.parent.name
sheets = sorted({r[0] for r in trk})
fams = set(a.fam.split(",")) if a.fam else None

before = Counter(); after = Counter(); per_sheet = {}; groups = 0; changed = 0
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    G = extract(str(q))
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    # переименование: группы (трек, корень) с ≥ 2 выданными кривыми разных индексов
    W2 = dict(W)
    grp = defaultdict(list)
    for nm, d in W.items():
        if not d:
            continue
        root, idx = idx_of(nm)
        t = tm.get(nm)
        if idx is None or t is None or (fams and root not in fams):
            continue
        f = feat(d)
        if f is not None:
            grp[(t, root)].append((idx, nm, f))
    for key, lst in grp.items():
        if len(lst) < 2 or len({i for i, _, _ in lst}) < len(lst):
            continue
        groups += 1
        names_by_idx = [nm for _, nm, _ in sorted(lst, key=lambda z: z[0])]
        curves_by_feat = sorted(lst, key=lambda z: -z[2])
        if a.min_ratio > 1.0:
            fs = [z[2] for z in curves_by_feat]
            if min(fs) <= 0 or max(fs) / min(fs) < a.min_ratio:
                continue
        new = {nm_new: W[nm_old] for nm_new, (_, nm_old, _) in zip(names_by_idx, curves_by_feat)}
        if any(W2[k] is not v for k, v in new.items()):
            changed += 1
        W2.update(new)
    b = sum(1 for g in gts if g in W and W[g] and HON(*st(W[g], gts[g])))
    af = sum(1 for g in gts if g in W2 and W2[g] and HON(*st(W2[g], gts[g])))
    per_sheet[sh] = (b, af)
    for g in gts:
        root = M.mnem_root(g)
        before[root] += g in W and bool(W[g]) and HON(*st(W[g], gts[g]))
        after[root] += g in W2 and bool(W2[g]) and HON(*st(W2[g], gts[g]))
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

B = sum(v[0] for v in per_sheet.values()); A = sum(v[1] for v in per_sheet.values())
d = np.array([v[1] - v[0] for v in per_sheet.values()])
up, dn = int((d > 0).sum()), int((d < 0).sum())
rng = np.random.default_rng(0); nz = d[d != 0]
p = float((np.abs(np.array([(nz * rng.choice([-1, 1], size=len(nz))).sum() for _ in range(100000)])) >= abs(d.sum())).mean()) if len(nz) else 1.0
print(f"\n★★ ПЕРЕИМЕНОВАНИЕ ПО {a.feat} (семейства: {a.fam or 'все'}, групп {groups}, из них переставлено {changed}):")
print(f"   именных честных {B} → {A} ({A-B:+d}), листов ↑{up}/↓{dn}, p = {p:.4f}")
wells = defaultdict(int)
for sh, (b_, a_) in per_sheet.items():
    wells[WELL.get(sh, '?')] += a_ - b_
wl = sorted(wells.values())
print(f"   по скважинам: в плюсе {sum(1 for v in wl if v > 0)}, в минусе {sum(1 for v in wl if v < 0)}, худшая {wl[0]:+d}, лучшая {wl[-1]:+d}")
print("| семейство | было | стало | Δ |")
print("|---|---|---|---|")
for root in sorted(before, key=lambda r: -(abs(after[r] - before[r]))):
    if after[root] != before[root]:
        print(f"| {root} | {before[root]} | {after[root]} | {after[root]-before[root]:+d} |")
