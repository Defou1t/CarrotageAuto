r"""_holdout_verdict.py — ПРОВЕРКА ПРИНЯТЫХ ПРАВОК НА ДЕРЖАННОМ СОРТЕ A (§6.218, 25.09).

ОТКУДА. Аудит §6.217: поле 1123 листа на 44% из сорта B, на сорте A поля (139 листов, 362 кривые) три правки дали +2
безымянных и +14 именных. Вне поля лежат 312 листов сорта A (910 кривых, 12 скважин, 10 из них в поле не
встречались) — ни одно решение ветки на них не мерилось. Здесь — знак каждой правки на этом наборе.

КРИТЕРИЙ (задан ДО прогона, §6.218). Для каждой пары «до → после» по ведущему для неё счёту:
  ★ ПОДТВЕРЖДЕНО — прирост > 0;
  ⛔ ОПРОВЕРГНУТО — прирост < 0 при p < 0.05 (парная перестановка знаков по листам);
  ⚠ НЕ РАЗЛИЧИМО — иначе.
Пары: P → K (§6.209 kslots, ведущий безымянный), K → RA (§6.213 выбор по слоту, безымянный),
RA → N (§6.215 имена, ведущий ИМЕННОЙ; для включения в прод дополнительно безымянный ≥ −5 —
пересчёт допуска поля −15 из 2737 на 910 кривых).
Печатает также разрез по скважинам (сколько в плюсе/минусе) — держанность по скважинам и есть смысл набора.
"""
import sys, argparse, pickle, csv, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--prefix", default="percurve_holdA_")
ap.add_argument("--perm", type=int, default=200000)
a = ap.parse_args()
TS = Path(a.ts)
wm = json.load(open(TS / "rowdec_wellmap.json", encoding="utf-8"))


def load(m):
    p = TS / f"{a.prefix}{m}.pkl"
    if not p.exists():
        sys.exit(f"⛔ нет {p.name} — прогон не закончен (код 3)")
    return pickle.load(open(p, "rb"))[m]


def perm_p(d):
    nz = d[d != 0]
    if not len(nz):
        return 1.0
    rng = np.random.default_rng(0)
    sims = (rng.choice([-1, 1], size=(a.perm, len(nz))) * np.abs(nz)).sum(1)
    return float((np.abs(sims) >= abs(d.sum()) - 1e-9).mean())


R = {m: load(m) for m in ("P", "K", "RA", "N")}
common = sorted(set.intersection(*(set(v) for v in R.values())))
n_curves = sum(R["P"][s][2] for s in common)
print(f"★ ДЕРЖАННЫЙ СОРТ A: листов во всех режимах {len(common)}, эталонных кривых {n_curves}, "
      f"скважин {len({wm.get(Path(s).stem, '?') for s in common})}")
print("| режим | безымянных | именных |")
print("|---|---|---|")
for m in ("P", "K", "RA", "N"):
    print(f"| {m} | {sum(R[m][s][1] for s in common)} | {sum(R[m][s][0] for s in common)} |")

PAIRS = [("§6.209 kslots", "P", "K", 1), ("§6.213 выбор по слоту", "K", "RA", 1), ("§6.215 имена", "RA", "N", 0)]
code = 0
print("\n| правка | ведущий счёт | Δ | листов ↑/↓ | p | второй счёт Δ | скважин +/− | вердикт |")
print("|---|---|---|---|---|---|---|---|")
for name, x, y, lead in PAIRS:
    d = np.array([R[y][s][lead] - R[x][s][lead] for s in common], float)
    d2 = np.array([R[y][s][1 - lead] - R[x][s][1 - lead] for s in common], float)
    p = perm_p(d)
    wells = defaultdict(float)
    for s, v in zip(common, d):
        wells[wm.get(Path(s).stem, "?")] += v
    wp = sum(1 for v in wells.values() if v > 0); wn = sum(1 for v in wells.values() if v < 0)
    if d.sum() > 0:
        verdict = "★ ПОДТВЕРЖДЕНО"
    elif d.sum() < 0 and p < 0.05:
        verdict = "⛔ ОПРОВЕРГНУТО"; code = max(code, 2)
    else:
        verdict = "⚠ НЕ РАЗЛИЧИМО"
    if name.startswith("§6.215") and verdict.startswith("★") and d2.sum() < -5:
        verdict += " (но безымянный < −5 — в прод НЕ включать)"
    print(f"| {name} | {'безымянный' if lead else 'ИМЕННОЙ'} | {int(d.sum()):+d} | {int((d > 0).sum())}/{int((d < 0).sum())} | "
          f"{p:.4f} | {int(d2.sum()):+d} | {wp}/{wn} | {verdict} |")
sys.exit(code)
