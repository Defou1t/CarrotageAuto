"""§6.345: потолок сегментации туши — слоты без честного кандидата (`line_choice_i`) на листах `orc_sheets.txt`: есть ли честный
(plane2 ≤ 3 px, покрытие ≥ 0.9, в пикселях) среди путей декодера (alt + extra) кэша с оракульной маской (`tcache_orc`) против тех же
путей без маски (`tcache_inert`). Слоты с честным кандидатом — сохраняется ли он среди путей оракула.
orc_ceiling.py [кэш оракула[+кэш…]]"""
import sys, pickle, glob, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from extract_nlgx import extract
from _multi_replica_probe import dense
from _cand_ceiling import dense_tr
src = open(Path(__file__).with_name("lvl_loss.py"), encoding="utf-8").read()
exec(src[src.index("def tree"):src.index("def lv_rows")])           # tree, err
HON = lambda e: e[0] <= 3.0 and e[1] >= 0.9
TS = Path("F:/nds/output/taskS")
ORCS = [Path(x) for x in sys.argv[1].split("+")] if len(sys.argv) > 1 else [TS / "tcache_orc"]   # «a+b» — пути вместе
SH = {l.strip() for l in (TS / "orc_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
C = Counter(); FAM = Counter()
for f in sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl"))):
    rec = pickle.load(open(f, "rb")); sh = rec["sheet"]
    if sh not in SH:
        continue
    q = SRC[sh]
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    pos, pb = [o / (key + ".pkl") for o in ORCS], TS / "tcache_inert" / (key + ".pkl")
    if not all(po.exists() for po in pos):
        continue
    C["листов"] += 1
    paths = {}
    for nm, pps in (("orc", pos), ("base", [pb])):
        paths[nm] = []
        for pp in pps:
            c = pickle.load(open(pp, "rb"))
            paths[nm] += [dense_tr(un(t)) for L, t in (c.get("alt") or []) + (c.get("extra") or [])]
    gts = {cc["name"]: cc for cc in extract(str(q))["curves"]}
    for s in rec["slots"]:
        if not s["has_truth"] or s["name"] not in gts:
            continue
        g = dense(gts[s["name"]])
        had = any(sc["lab"] is not None and sc["lab"][0] for sc in s["cands"])
        ho = any(HON(err(d, g)) for d in paths["orc"] if d)
        hb = any(HON(err(d, g)) for d in paths["base"] if d)
        C[("без честного" if not had else "с честным", "оракул", ho)] += 1
        C[("без честного" if not had else "с честным", "декодер без маски", hb)] += 1
        if not had and ho:
            FAM[s["root"]] += 1
print(f"★ листов {C['листов']}")
for grp in ("без честного", "с честным"):
    for nm in ("декодер без маски", "оракул"):
        t, f_ = C[(grp, nm, True)], C[(grp, nm, False)]
        print(f"   слоты {grp} кандидата в пуле: {nm} — честный путь декодера есть у {t} из {t + f_} ({t / max(1, t + f_):.1%})")
print("   стали честными по семействам:", dict(FAM.most_common(12)))
