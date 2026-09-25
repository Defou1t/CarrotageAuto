r"""_fam_diag.py — ГРУППЫ ОДНОГО СЕМЕЙСТВА НА ТРЕКЕ: верен ли НАБОР кривых и верен ли ПОРЯДОК, и какой упорядочиватель прав (21.09, B4).

`_fam_reorder.py` показал: переименование по wiggle ВРЕДИТ выдаче (GZ −28), хотя по эталону wiggle упорядочивает
зонды с парной точностью 80%. Здесь разбирается почему. Для каждой группы (трек, корень) с ≥ 2 выданными
кривыми разных индексов, по замороженной выдаче G:
  набор верен — каждая выданная кривая группы честна против КАКОЙ-ТО эталонной кривой того же корня (1:1);
  среди групп с верным набором: порядок нынешней раскладки верен? порядок по wiggle (убыв.) верен? по x (возр.)?
  по wiggle И x согласны и верны? — и парная точность каждого упорядочивателя на ВЫДАННЫХ кривых.
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
ap.add_argument("--fam", default="GZ")
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def idx_of(nm):
    short = nm.split()[0]
    root = M.mnem_root(short)
    m = re.match(re.escape(root) + r"(\d+)$", short)
    if not m:
        return root, None
    digits = m.group(1)
    return root, int(digits[:-1]) if len(digits) >= 2 else None


def feats(d):
    ys = sorted(d)
    x = np.array([d[y] for y in ys], float)
    if len(x) < 50:
        return None
    dx = np.diff(x)
    span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
    return dict(wiggle=float(np.median(np.abs(dx))) / span, rev=float(np.mean((dx[:-1] * dx[1:]) < 0)),
                medx=float(np.median(x)), n=len(x))


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
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})
C = Counter(); pairs = Counter(); byK = defaultdict(Counter)
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
    grp = defaultdict(list)
    for nm, d in W.items():
        root, idx = idx_of(nm)
        t = tm.get(nm)
        if d and idx is not None and t is not None and root == a.fam and (sh, t) in KOF:
            grp[t].append((idx, nm, d))
    for t, lst in grp.items():
        if len(lst) < 2 or len({i for i, _, _ in lst}) < len(lst):
            continue
        C["групп"] += 1
        # истинный индекс каждой выданной кривой: эталон того же корня и трека, честный против неё
        gt_fam = [g for g in gts if tm.get(g) == t and idx_of(g)[0] == a.fam and idx_of(g)[1] is not None]
        truth = {}
        for idx, nm, d in lst:
            hon = [g for g in gt_fam if HON(*st(d, gts[g]))]
            if len(hon) == 1:
                truth[nm] = idx_of(hon[0])[1]
        if len(truth) < len(lst) or len(set(truth.values())) < len(lst):
            C["набор неверен (есть нечестная/двойная)"] += 1
            continue
        C["набор верен"] += 1
        K = min(len(lst), 5); byK[K]["набор верен"] += 1
        cur_ok = all(truth[nm] == idx for idx, nm, _ in lst)
        C["  порядок нынешний верен"] += cur_ok; byK[K]["нынешний верен"] += cur_ok
        fs = {nm: feats(d) for _, nm, d in lst}
        if any(v is None for v in fs.values()):
            continue
        names_by_idx = [nm for idx, nm, _ in sorted(lst, key=lambda z: z[0])]
        idx_sorted = sorted(idx for idx, _, _ in lst)
        def order_ok(key, reverse):
            ordered = sorted(lst, key=lambda z: fs[z[1]][key], reverse=reverse)
            return all(truth[nm] == i for i, (_, nm, _) in zip(idx_sorted, ordered))
        w_ok = order_ok("wiggle", True); x_ok = order_ok("medx", False); r_ok = order_ok("rev", True)
        C["  порядок по wiggle верен"] += w_ok; C["  порядок по x верен"] += x_ok; C["  порядок по rev верен"] += r_ok
        C["  wiggle и x согласны"] += (w_ok == x_ok and w_ok) or (sorted(lst, key=lambda z: fs[z[1]]["wiggle"], reverse=True) == sorted(lst, key=lambda z: fs[z[1]]["medx"]))
        C["  нынешний неверен, wiggle верен"] += (not cur_ok) and w_ok
        C["  нынешний верен, wiggle неверен"] += cur_ok and (not w_ok)
        byK[K]["wiggle верен"] += w_ok; byK[K]["x верен"] += x_ok
        # парная точность на выданных кривых
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                na, nb = lst[i][1], lst[j][1]
                lo, hi = (na, nb) if truth[na] < truth[nb] else (nb, na)
                pairs["n"] += 1
                pairs["wiggle: меньший индекс извилистее"] += fs[lo]["wiggle"] > fs[hi]["wiggle"]
                pairs["x: меньший индекс левее"] += fs[lo]["medx"] < fs[hi]["medx"]
                pairs["rev: меньший индекс больше смен знака"] += fs[lo]["rev"] > fs[hi]["rev"]
                cur_lo = [idx for idx, nm, _ in lst if nm == lo][0]; cur_hi = [idx for idx, nm, _ in lst if nm == hi][0]
                pairs["нынешняя раскладка: пара в верном порядке"] += cur_lo < cur_hi
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

print(f"\n★★ СЕМЕЙСТВО {a.fam}, выдача {a.dir}")
for k, v in C.items():
    print(f"   {k}: {v}")
print("\n★ парная точность на ВЫДАННЫХ кривых с верным набором:")
for k, v in pairs.items():
    if k != "n":
        print(f"   {k}: {v} из {pairs['n']} = {100*v/max(1,pairs['n']):.0f}%")
print("\n| K | набор верен | нынешний верен | wiggle верен | x верен |")
print("|---|---|---|---|---|")
for K in sorted(byK):
    print(f"| {K}{'+' if K == 5 else ''} | {byK[K]['набор верен']} | {byK[K]['нынешний верен']} | {byK[K]['wiggle верен']} | {byK[K]['x верен']} |")
