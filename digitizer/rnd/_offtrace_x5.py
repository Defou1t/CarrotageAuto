r"""_offtrace_x5.py — ЧТО ЭТО ЗА НЕОЦИФРОВАННЫЕ ЛИНИИ: НЕ ПЕРЕВЫНОСЫ ЛИ ×5/×25 (§6.189-§6.190 → B1).

ОТКУДА ВОПРОС. Разбор по маске показал: 85% трасс декодера и 92% трасс прода, названных «вне
линий», лежат на НАСТОЯЩЕЙ туши — линия на листе есть, эксперт её не оцифровал. Осталось назвать,
что это за линии, потому что от этого зависит правка:

  ★ ПЕРЕВЫНОС ×5/×25 (та же кривая, перерисованная в сжатом масштабе, когда перо уходит за поле) —
    тогда трасса ВЕРНА как геометрия и неверна лишь как объект: её надо узнать и не подавать под
    именем основной кривой. Это разбор уже названный проектом (P0-1, P2-9), а не новая работа.
  ★ Посторонняя кривая другого прибора / служебная разметка — тогда нужен отбор по признакам.

★ ПРОВЕРЯЕТСЯ БЕЗ КАРТИНКИ И БЕЗ ГЕОМЕТРИИ ТРЕКА. Перевынос ×k связан с оригиналом жёстко:
все шкалы делят один x-диапазон и линейны, поэтому `t(y) = a + (g(y) − a)/k` при неизвестном
общем `a` (левый край базы). Значит `a_y = (k·t(y) − g(y))/(k−1)` обязано быть ПОСТОЯННЫМ по всем
строкам — это и проверяется: берём медиану `a`, считаем остаток и требуем med|r| ≤ `--tol`.
⚠ Проверяются ОБА направления: трасса может быть перевыносом эталона, а может быть его оригиналом
(эксперт оцифровал как раз перевынос).

  <ComfyUI>\python_embeded\python.exe _offtrace_x5.py --dump offtrace_ДЕКОДЕР_3_4.pkl
"""
import sys, argparse, pickle
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
ap.add_argument("--dump", default="offtrace_ДЕКОДЕР_3_4.pkl")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--ks", nargs="+", type=float, default=[5.0, 25.0, 125.0])
ap.add_argument("--tol", type=float, default=3.0, help="med|остаток| для признания перевыносом")
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
TS = Path(a.ts)

SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

OFF = pickle.load(open(TS / a.dump, "rb"))
by_sheet = defaultdict(list)
for sh, gname, wname, tr in OFF:
    by_sheet[sh].append((gname, wname, tr))
sheets = sorted(by_sheet)
if a.cap:
    sheets = sheets[:a.cap]
print(f"★ ТРАСС «ВНЕ ЛИНИЙ»: {sum(len(by_sheet[s]) for s in sheets)} на {len(sheets)} листах")


def replica_of(t, g, k):
    """→ med|остаток| гипотезы «t — это ×k-перевынос g» (общий левый край подбирается медианой)."""
    ys = [y for y in t if y in g]
    if len(ys) < 30:
        return None
    tv = np.array([t[y] for y in ys], float)
    gv = np.array([g[y] for y in ys], float)
    av = (k * tv - gv) / (k - 1.0)          # из t = a + (g−a)/k
    aa = float(np.median(av))
    pred = aa + (gv - aa) / k
    return float(np.median(np.abs(tv - pred)))


cnt, best_k = Counter(), Counter()
for sh in sheets:
    q = SRC.get(sh)
    if not q:
        cnt["⚠ нет разметки"] += len(by_sheet[sh]); continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    for gname, wname, tr in by_sheet[sh]:
        hit = None
        for k in a.ks:
            for g2, gv in gts.items():
                for kk, tt, gg in ((k, tr, gv), (k, gv, tr)):   # оба направления
                    r = replica_of(tt, gg, kk)
                    if r is not None and r <= a.tol:
                        hit = (k, g2)
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            cnt[f"★ ПЕРЕВЫНОС ×{hit[0]:g}"] += 1
            best_k[hit[0]] += 1
        else:
            cnt["не перевынос — посторонняя линия"] += 1

n = sum(cnt.values())
print(f"\n★★ ЧТО ЭТО ЗА ЛИНИИ (трасс {n}, допуск med|остаток| ≤ {a.tol:g}px)")
for k, v in cnt.most_common():
    print(f"| {k} | {v} | {100*v/max(1,n):.0f}% |")
if best_k:
    print("\n★ по кратности: " + ", ".join(f"×{k:g} → {v}" for k, v in sorted(best_k.items())))
print("\n⚠ Проверка ГЕОМЕТРИЧЕСКАЯ: совпадение по формуле не доказывает, что линия — перевынос "
      "именно этой кривой; оно доказывает, что она связана с ней масштабом. Для правки этого "
      "достаточно (обе трактовки требуют одного: узнать и не подавать под чужим именем).")
