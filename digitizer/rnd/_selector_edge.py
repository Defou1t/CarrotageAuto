r"""_selector_edge.py — ЧТО СЕЛЕКТОР УМЕЕТ ТАКОГО, ЧЕГО НЕ ДАЁТ ВЫКЛЮЧЕНИЕ ВЕРШИНЫ (§6.102).

ЗАЧЕМ. Лестница §6.102 показала: корзина «>30px, личность потеряна» у правки `wide_run` не двигается
ВООБЩЕ (252 → 252), а у селектора уходит на 7 кривых (252 → 245). Значит остаток преимущества
селектора — это ровно то, что ищет вся ветка: УДЕРЖАНИЕ ЛИЧНОСТИ. Семь кривых мало для статистики,
но достаточно, чтобы посмотреть, ЧЕМ они отличаются, — и, может быть, вынуть признак в цену перехода.

КАК. Читаются УЖЕ ВЫДАННЫЕ файлы трёх режимов A/B (`_trace_prod_ab.py`), считается медиана |Δx| по
каждой кривой, отбираются те, где A и B потеряли личность (>30px), а C её удержал. К ним
подтягиваются свойства линии из пула: цвет, класс, число линий на листе и на треке.

  <ComfyUI>\python_embeded\python.exe _selector_edge.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\prod_ab_trace")
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
a = ap.parse_args()

WLG, WELL = {}, {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.stem[:40]] = q; WELL[q.stem[:40]] = wlg.parent.name
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.stem[:40], q); WELL.setdefault(q.stem[:40], "Semeguniv")
POOL = {}
for root in a.pools:
    for f in Path(root).glob("*.pkl"):
        POOL.setdefault(f.stem[:40], f)


def err(o, g):
    c = [y for y in o if y in g]
    if len(c) < 30:
        return None, 0.0
    d = np.array([abs(o[y] - g[y]) for y in c], float)
    return float(np.median(d)), len(c) / max(1, len(g))


D = Path(a.dir)
modes = ["A", "B", "C"]
common = sorted(set.intersection(*[{p.name for p in (D / m).iterdir() if p.is_dir()}
                                   for m in modes]))
rows = []
for stem in common:
    n = WLG.get(stem)
    if n is None:
        continue
    G = extract(str(n))
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not gts:
        continue
    got = {}
    for m in modes:
        f = next(iter(sorted((D / m / stem).glob("*_auto.nlgx"))), None)
        got[m] = ({c["name"]: dense(c) for c in extract(str(f))["curves"]
                   if M.mnem_root(c["name"]) != "DA"} if f else {})
    for nm, gd in gts.items():
        md = {}
        for m in modes:
            w = got[m].get(nm)
            md[m] = err(w, gd)[0] if w else None
        rows.append(dict(sheet=stem, well=WELL.get(stem, "?"), curve=nm, **{f"m{m}": md[m]
                                                                           for m in modes}))

BIG = lambda v: v is None or v > 30.0
won = [r for r in rows if BIG(r["mA"]) and BIG(r["mB"]) and r["mC"] is not None and r["mC"] <= 3.0]
lost = [r for r in rows if r["mA"] is not None and r["mA"] <= 3.0 and BIG(r["mC"])]
print(f"кривых {len(rows)} на {len(common)} листах")
print(f"★ СЕЛЕКТОР ВЕРНУЛ ЛИЧНОСТЬ (A и B >30px, C ≤3px): {len(won)}")
print(f"  обратных случаев (A ≤3px, C потерял): {len(lost)}\n")
print(f"{'скважина':<18}{'кривая':<10}{'медиана A':>11}{'B':>9}{'C':>7}   линий/лист  линий/трек цвет")
for r in sorted(won, key=lambda q: q["well"]):
    pf = POOL.get(r["sheet"])
    nl = tr_ = col = "?"
    if pf:
        d = pickle.load(open(pf, "rb"))
        nl = len(d["lines"])
        cnt = Counter(l["track"] for l in d["lines"])
        tr_ = f"{np.mean(list(cnt.values())):.1f}"
        col = ",".join(sorted({(l["color"] or "black")[:5] for l in d["lines"]}))
    f2 = lambda v: f"{v:.1f}" if v is not None else "нет"
    print(f"{r['well'][:17]:<18}{r['curve'][:9]:<10}{f2(r['mA']):>11}{f2(r['mB']):>9}"
          f"{f2(r['mC']):>7}   {str(nl):>10}  {tr_:>10} {col}")
print(f"\nпо скважинам: {dict(Counter(r['well'] for r in won))}")
