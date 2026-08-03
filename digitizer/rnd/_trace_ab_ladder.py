r"""_trace_ab_ladder.py — ЛЕСТНИЦА КОРЗИН по УЖЕ ВЫДАННЫМ файлам A/B (§6.96), без прогона пайплайна.

ЗАЧЕМ. Счётчик «честных» — это ворота 3px, и §6.96 требует рядом ВСЕГДА публиковать лестницу по той
же медиане: иначе улучшение медианы с 90px до 20px даст РОВНО НОЛЬ честных кривых и настоящий
прогресс будет выглядеть провалом. Здесь лестница считается по файлам, которые `_trace_prod_ab.py`
уже положил на диск, поэтому замер стоит секунды и его можно снимать на любом этапе A/B.

  <ComfyUI>\python_embeded\python.exe _trace_ab_ladder.py [--dir …\prod_ab_trace]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\prod_ab_trace")
a = ap.parse_args()

WLG = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WLG[q.stem[:40]] = q
for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
    WLG.setdefault(q.stem[:40], q)


def err(ours, gt):
    com = [y for y in ours if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(ours[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def leaked(tr, raw):
    er = np.array([raw["top_y"] + i for i, x in enumerate(raw["xs"]) if x != NULL])
    ex = np.array([x for x in raw["xs"] if x != NULL], float)
    oy = np.array(sorted(tr)); ox = np.array([tr[y] for y in oy], float)
    if len(oy) == len(er) and np.array_equal(oy, er) and np.allclose(ox, ex):
        return True
    com = np.intersect1d(oy, er)
    if len(com) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(er.tolist(), ex.tolist()))
        if np.mean([abs(om[y] - em[y]) < 1e-9 for y in com]) > 0.5:
            return True
    return False


modes = sorted(p.name for p in Path(a.dir).iterdir() if p.is_dir() and len(p.name) == 1)
# ⚠ Считаем ТОЛЬКО по листам, что есть у ВСЕХ режимов: иначе режим, успевший меньше, выглядел бы
# хуже просто по объёму (§6.71 — «ноль различий значит проверь, что сравниваешь»).
common = None
for m in modes:
    s = {p.name for p in (Path(a.dir) / m).iterdir() if p.is_dir()}
    common = s if common is None else (common & s)
common = sorted(common or ())
print(f"режимов {len(modes)} ({', '.join(modes)}), листов общих {len(common)}\n")

GT = {}
res = {m: [] for m in modes}
for stem in common:
    n = WLG.get(stem)
    if n is None:
        continue
    if stem not in GT:
        G = extract(str(n))
        raws = {c["name"]: c for c in G["curves"]
                if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        GT[stem] = (raws, {nm: dense(c) for nm, c in raws.items()})
    raws, gts = GT[stem]
    if not gts:
        continue
    for m in modes:
        got = next(iter(sorted((Path(a.dir) / m / stem).glob("*_auto.nlgx"))), None)
        if got is None:
            continue
        W = {c["name"]: dense(c) for c in extract(str(got))["curves"]
             if M.mnem_root(c["name"]) != "DA"}
        for nm, gd in gts.items():
            if nm not in W or not W[nm] or leaked(W[nm], raws[nm]):
                continue
            md, cv = err(W[nm], gd)
            res[m].append((md if md is not None else float("nan"), cv))

print(f"{'режим':<8}{'кривых':>8}{'≤3px':>7}{'3-10':>7}{'10-30':>7}{'>30':>7}"
      f"{'с пометкой':>12}{'медиана':>9}")
for m in modes:
    v = res[m]
    md = np.array([q[0] for q in v]); cv = np.array([q[1] for q in v])
    ok = ~np.isnan(md)
    lad = [int(((md <= 3) & ok).sum()), int(((md > 3) & (md <= 10) & ok).sum()),
           int(((md > 10) & (md <= 30) & ok).sum()), int((((md > 30) & ok) | ~ok).sum())]
    mark = int(((md <= 10) & (cv >= 0.9) & ok).sum())
    print(f"{m:<8}{len(v):>8}{lad[0]:>7}{lad[1]:>7}{lad[2]:>7}{lad[3]:>7}{mark:>12}"
          f"{np.nanmedian(md[ok]) if ok.any() else float('nan'):>9.1f}")
print("\n⚠ «честных» = ≤3px И покрытие ≥90%; колонка ≤3px здесь БЕЗ условия покрытия, "
      "поэтому она больше счётчика честных — это разные корзины.")
