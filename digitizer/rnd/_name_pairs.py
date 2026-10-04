r"""_name_pairs.py — ВЕРНО ПРОВЕДЕНЫ, НО НЕВЕРНО НАЗВАНЫ — МЕРОЙ ПРИЁМКИ (04.10, к §6.258 / §6.260).

Читает пары 1:1 без имени из `_name_cost_prod.py --pairs-dump` (на плоскости в обе стороны, в значениях) и раскладывает
неверно названные: перестановка A↔B в треке, имя выдачи есть в эталоне трека, имени выдачи нет в эталоне; частые пары семейств;
разрез поле / сорт A.

  _name_pairs.py --pairs F:/nds/output/taskS/rp_lvl_S_pairs.pkl --mode S
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
from collections import Counter, defaultdict
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--pairs", required=True)
ap.add_argument("--mode", default="S")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
P = pickle.load(open(a.pairs, "rb"))[a.mode]
by = defaultdict(dict)                           # (лист, трек) → {эталон: выдача}
for sh, t, g, w in P:
    by[(sh, t)][g] = w
for sn, S in SETS.items():
    C = Counter(); fam = Counter()
    for (sh, t), mt in by.items():
        if sh not in S:
            continue
        outs = set(mt.values())
        for g, w in mt.items():
            C["пар"] += 1
            if g == w:
                C["имя верно"] += 1; continue
            swap = mt.get(w) == g
            kind = "перестановка A↔B" if swap else ("имя выдачи есть в эталоне трека" if w in mt or w in outs else "имени выдачи нет в эталоне трека")
            C[kind] += 1
            fam[(M.mnem_root(g), M.mnem_root(w))] += 1
    bad = C["пар"] - C["имя верно"]
    print(f"★ {sn}: честных без имени {C['пар']}, имя верно {C['имя верно']}, НЕВЕРНО {bad}")
    for k in ("перестановка A↔B", "имя выдачи есть в эталоне трека", "имени выдачи нет в эталоне трека"):
        print(f"   {k}: {C[k]}")
    same = sum(n for (x, y), n in fam.items() if x == y)
    print(f"   то же семейство (номер зонда / индекс): {same} из {bad}")
    print("   частые пары (эталон → выдача): " + ", ".join(f"{x}→{y} {n}" for (x, y), n in fam.most_common(14)))
