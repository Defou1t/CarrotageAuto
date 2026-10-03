r"""_snap_holdout.py — ПРОВЕРКА ПОРОГА 45 ПРИТЯЖКИ НА НЕПРОСМОТРЕННЫХ ЛИСТАХ ПОЛЯ (03.10, §6.259).

Дамп `_snap_whatif.py --dump` хранит счёт по ОБРАБОТАННЫМ листам по порядку. Здесь тот же отбор листов (без загрузки картинок)
восстанавливает номер листа в списке, и счёт делится: листы 1–350 (их суммы видены до задания проверки) и 351–1123 (проверка).

  _snap_holdout.py --dump F:/nds/output/taskS/snap_field.pkl --split 350
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc")
ap.add_argument("--mode", default="N")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--sheets", default="wellmap_sheets.txt")
ap.add_argument("--dump", required=True)
ap.add_argument("--split", type=int, default=350)
a = ap.parse_args()
TS = Path(a.ts)
D = pickle.load(open(a.dump, "rb"))
smap = pickle.load(open(TS / a.map, "rb"))
for root in ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_heldout", "pools_all"]:
    for f in sorted((TS / root).glob("*.pkl")):
        if f.stem in smap:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        smap.setdefault(d["name"], {s["name"]: s["track"] for s in d["slots"]})
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
sheets = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
idx = []
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem
    pd = Path(a.dir) / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    if not got or not find_image(q):
        continue
    G = [c["name"] for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50]
    tm = smap.get(sh, {})
    if not {tm.get(g) for g in G if tm.get(g) is not None}:
        continue
    idx.append(si)
PER, PERN = D["PER"], D["PERN"]
keys = list(PER)
if "SI" in D:                         # ★ номера листов записаны в дамп
    idx = D["SI"]
n = len(PER[keys[0]])
assert n == len(idx), f"листов в дампе {n}, восстановлено {len(idx)}"
idx = np.array(idx)


def ptest(d):
    d = np.asarray(d); nz = d[d != 0]
    if not len(nz):
        return 1.0
    rng = np.random.default_rng(0)
    sims = (rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)
    return float(np.mean(np.abs(sims) >= abs(d.sum())))


for part, m in (("листы 1–%d (видены)" % a.split, idx <= a.split), ("листы %d–%d (ПРОВЕРКА)" % (a.split + 1, len(sheets)), idx > a.split),
                ("всё поле", idx > 0)):
    b = np.array(PER[keys[0]])[m]; bn = np.array(PERN[keys[0]])[m]
    print(f"★ {part}: листов {int(m.sum())}, база {int(b.sum())} (именных {int(bn.sum())})")
    for k in keys[1:]:
        d = np.array(PER[k])[m] - b; dn = np.array(PERN[k])[m] - bn
        print(f"   {k:18s}: Δ {int(d.sum()):+d} (↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {ptest(d):.4f}); "
              f"именных Δ {int(dn.sum()):+d} (p = {ptest(dn):.4f})")
