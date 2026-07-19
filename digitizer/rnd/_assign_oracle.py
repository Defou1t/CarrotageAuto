"""СКОЛЬКО ТЕРЯЕТСЯ НА РАСКЛАДКЕ ПО СЛОТАМ, а сколько на самом ведении.

Контекст (QC Эдуарда 19.07): наши трассы лежат на чернилах на 99.9-100%, но часто идут ПО ЧУЖОЙ
кривой ⇒ промах 400-900px это не «кривая трасса», а «не та кривая». Вопрос: если бы мы
разложили ТЕ ЖЕ трассы по слотам ОПТИМАЛЬНО, насколько стало бы лучше?

Считаем на готовой выдаче (`*_auto.nlgx`) против экспертных кривых:
  ФАКТ     — как сейчас: трасса слота X против экспертной кривой X;
  ОПТИМУМ  — жадное назначение по минимальной медианной ошибке (кривых 4-8, жадность годится);
  ПОТОЛОК  — для каждой экспертной кривой ЛУЧШАЯ из наших трасс (оракул, назначение может
             быть не-взаимно-однозначным) — верхняя граница того, что вообще есть в выдаче.
Разрыв ФАКТ→ОПТИМУМ = цена раскладки. Разрыв ОПТИМУМ→ПОТОЛОК = цена требования 1:1.

  python _assign_oracle.py <dir с *_auto.nlgx> [--tol 3]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ARCHIVE = Path(r"F:\nds\projects\Archive")
ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--tol", type=float, default=3.0)
a = ap.parse_args()

# индекс исходных рамок по stem
src_by_stem = {}
for wlg in ARCHIVE.glob("*/wlg"):
    for f in wlg.glob("*.nlgx"):
        if "_auto" not in f.stem:
            src_by_stem.setdefault(f.stem, f)


def series(c):
    return {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}


def err(ours, gt):
    common = [y for y in ours if y in gt]
    if len(common) < 30:
        return None, 0.0
    d = np.array([abs(ours[y] - gt[y]) for y in common], float)
    return float(np.median(d)), len(common) / max(1, len(gt))


tot = {"факт": [], "оптимум": [], "потолок": []}
for auto in sorted(Path(a.dir).glob("*_auto.nlgx")):
    stem = auto.stem[:-5]
    src = src_by_stem.get(stem)
    if not src:
        continue
    A = extract(str(auto)); G = extract(str(src))
    ours = {c["name"]: series(c) for c in A["curves"]
            if M.mnem_root(c["name"]) != "DA" and any(x != NULL for x in c["xs"])}
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not ours or not gts:
        continue
    # матрица ошибок
    cost = {}
    for on, od in ours.items():
        for gn, gd in gts.items():
            m, cov = err(od, gd)
            if m is not None:
                cost[(on, gn)] = (m, cov)
    if not cost:
        continue
    print(f"\n{stem[:52]}  наших трасс {len(ours)}, экспертных кривых {len(gts)}")
    # ФАКТ
    fact = []
    for gn in gts:
        if gn in ours and (gn, gn) in cost:
            fact.append(cost[(gn, gn)])
    # ОПТИМУМ (жадно по возрастанию ошибки, 1:1)
    pairs = sorted(cost.items(), key=lambda kv: kv[1][0])
    uo, ug, opt = set(), set(), []
    for (on, gn), v in pairs:
        if on in uo or gn in ug:
            continue
        uo.add(on); ug.add(gn); opt.append((gn, v))
    # ПОТОЛОК
    ceil = []
    for gn in gts:
        best = min((v for (on, g2), v in cost.items() if g2 == gn), key=lambda v: v[0], default=None)
        if best:
            ceil.append(best)
    for tag, lst in (("факт", fact), ("оптимум", [v for _, v in opt]), ("потолок", ceil)):
        if not lst:
            continue
        med = np.median([v[0] for v in lst])
        good = sum(1 for v in lst if v[0] <= a.tol and v[1] >= 0.9)
        print(f"   {tag:<8} кривых {len(lst):>2}  med(med) {med:7.1f}px  честных {good}/{len(lst)}")
        tot[tag] += lst
    # какие слоты в оптимуме получили ДРУГУЮ трассу
    swap = [(gn, on) for (on, gn), _ in pairs if (on, gn) in
            {(o, g) for (o, g), _ in pairs} and False]
    diff = [f"{gn.split()[0]}←{on.split()[0]}" for (on, gn), v in pairs
            if gn in ug and on in uo and on != gn and (gn, v) in opt]
    if diff:
        print(f"   оптимум переставил: {', '.join(diff[:8])}")

print("\n=== ИТОГО ===")
for tag in ("факт", "оптимум", "потолок"):
    lst = tot[tag]
    if not lst:
        continue
    med = np.median([v[0] for v in lst])
    good = sum(1 for v in lst if v[0] <= a.tol and v[1] >= 0.9)
    print(f"  {tag:<8} кривых {len(lst):>3}  med(med) {med:7.1f}px  "
          f"ЧЕСТНЫХ (med<={a.tol:.0f}px И cov>=0.9) {good}/{len(lst)}")
print("\nразрыв ФАКТ→ОПТИМУМ = цена неверной раскладки по слотам;")
print("разрыв ОПТИМУМ→ПОТОЛОК = цена требования взаимно-однозначного назначения.")
