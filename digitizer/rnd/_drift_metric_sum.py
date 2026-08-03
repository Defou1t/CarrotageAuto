r"""_drift_metric_sum.py — СВОДКА ДРЕЙФА: дрейф это или дребезг, и сколько стоит держать личность.

Читает шарды `_drift_metric.py`. Три режима ведения по ОДНИМ И ТЕМ ЖЕ ранам:
  oracle (ран ближе к эталону этой строки) → hist (к ПРАВИЛЬНОМУ предыдущему x) → self (к СВОЕМУ).
Разница hist и self — цена накопления ошибки, то есть дрейф.

ЧИТАЕТСЯ ТАК:
  много срывов + высокая доля самовозвратов + короткие серии ⇒ ДРЕБЕЗГ (лечится сглаживанием,
    медианой по окну, деспайком — то есть локально);
  мало срывов + низкие самовозвраты + одна длинная серия и потом мимо ⇒ ДРЕЙФ (лечится gap-closing
    и глобальной линковкой, §6.92: laptrack/Stone Soup; локальные правки бесполезны).

  <ComfyUI>\python_embeded\python.exe _drift_metric_sum.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\drift_metric")
ap.add_argument("--ladder", default=r"F:\nds\output\taskS\oracle_ladder_v3")
a = ap.parse_args()

R = []
for f in sorted(Path(a.dir).glob("rows_*.pkl")):
    R.extend(pickle.load(open(f, "rb")))
if not R:
    print("нет данных"); sys.exit(1)

# корзина из лестницы (§6.94): считаем дрейф ТОЛЬКО там, где тушь вообще есть — иначе мерим шум
INK = {}
for f in sorted(Path(a.ladder).glob("rows_*.pkl")):
    for r in pickle.load(open(f, "rb")):
        INK[(r["sheet"], r["curve"])] = r["ink3"]
for r in R:
    r["ink3"] = INK.get((r["sheet"], r["curve"]), np.nan)

W = 104
print(f"кривых {len(R)} из {len({r['sheet'] for r in R})} листов; с тушью (ink3≥0.9) "
      f"{sum(1 for r in R if r['ink3'] >= 0.9)}")

GRP = [("★ тушь есть, прод НЕ берёт", lambda r: r["ink3"] >= 0.9 and not r["prod"]),
       ("тушь есть, прод берёт", lambda r: r["ink3"] >= 0.9 and r["prod"]),
       ("туши нет (контроль)", lambda r: r["ink3"] < 0.9)]

print(f"\n{'='*W}\nДОЛЯ СТРОК «НА ЦЕЛИ» (±3px) по режиму ведения\n{'='*W}")
print(f"{'группа':<28}{'кривых':>7}{'оракул':>9}{'история':>9}{'сам':>7}"
      f"{'цена накопления':>17}{'медиана |Δ| сам':>17}")
for nm, sel in GRP:
    v = [r for r in R if sel(r)]
    if not v:
        continue
    o = np.mean([r["on_oracle"] for r in v]); h = np.mean([r["on_hist"] for r in v])
    s = np.mean([r["on_self"] for r in v])
    md = np.nanmedian([r["med_self"] for r in v])
    print(f"{nm:<28}{len(v):>7}{o:>8.0%}{h:>9.0%}{s:>7.0%}{h - s:>16.0%}{md:>15.1f}px")

# ── ПЕРЕВОД В ЕДИНИЦЫ ОТГРУЗКИ (требование §6.96) ─────────────────────────────────────────────
# ⚠ Доли строк нельзя сравнивать с точностью прода: приёмка считает КРИВЫЕ по критерию
# «медиана |Δx| ≤ 3px И покрытие ≥ 90%». Здесь тот же критерий применяется к каждому режиму, и это
# единственные числа этого стенда, сравнимые с 147 честными кривыми отгрузки.
MODES = [m for m in ("oracle", "hist", "self", "band", "dp") if f"med_{m}" in R[0]]
honest = lambda r, m: (r[f"med_{m}"] == r[f"med_{m}"]) and r[f"med_{m}"] <= 3.0 \
    and (1.0 - r["norun"]) >= 0.9
print(f"\n{'='*W}\n★ В ЕДИНИЦАХ ОТГРУЗКИ: честных кривых по критерию приёмки (медиана ≤3px и "
      f"покрытие ≥90%)\n{'='*W}")
print(f"{'режим':<34}{'честных':>9}{'доля':>8}   для сравнения: отгрузка 147 (7.8%)")
for m in MODES:
    n = sum(1 for r in R if honest(r, m))
    print(f"{m:<34}{n:>9}{100*n/len(R):>7.1f}%")
if "band" in MODES:
    nb = sum(1 for r in R if honest(r, "band")); ns = sum(1 for r in R if honest(r, "self"))
    sh = 100.0 * nb / max(1, len(R))
    print(f"\n★★ РАЗВИЛКА §6.96 (полоса прода забирает часть «цены накопления»): полоса даёт "
          f"{sh:.1f}% против {100*ns/max(1,len(R)):.1f}% у жадного без полосы")
    print("   ⇒ " + ("полоса УЖЕ держит анти-дрейф: приз линковки резко меньше заявленного, "
                     "линковку опустить ниже порога темноты" if sh >= 40 else
                     "приз реален, но примерно вдвое меньше заявленного: делать ДП-Виттерби (часы), "
                     "а не день на laptrack" if sh >= 20 else
                     "заявка ветки ПОДТВЕРЖДЕНА: полоса дрейф не спасает, день на глобальную "
                     "линковку оправдан"))
print(f"\n{'='*W}\nДРЕЙФ ИЛИ ДРЕБЕЗГ (режим `сам`, только там где тушь есть)\n{'='*W}")
v = [r for r in R if r["ink3"] >= 0.9]
if v:
    for tag, key in (("срывов на 1000 строк", "br_self"), ("доля самовозвратов", "back_self"),
                     ("самая длинная серия, строк", "maxlen_self"),
                     ("медианная серия, строк", "medlen_self"),
                     ("число серий на кривую", "nstreak_self")):
        arr = np.array([r[key] for r in v], float)
        print(f"  {tag:<30} медиана {np.nanmedian(arr):>8.1f}   "
              f"p10 {np.nanpercentile(arr, 10):>7.1f}   p90 {np.nanpercentile(arr, 90):>8.1f}")
    nrows = np.array([r["nrows"] for r in v], float)
    mx = np.array([r["maxlen_self"] for r in v], float)
    print(f"  {'самая длинная серия / длина кривой':<30} медиана "
          f"{np.nanmedian(mx / np.maximum(1, nrows)):>8.1%}")
    back = np.nanmedian([r["back_self"] for r in v])
    frac = np.nanmedian(mx / np.maximum(1, nrows))
    print(f"\n⇒ ВЕРДИКТ: " + ("ДРЕБЕЗГ — ведение срывается и САМО возвращается, лечится локально "
                              "(сглаживание/медиана окна/деспайк)" if back > 0.7 and frac < 0.3 else
                              "ДРЕЙФ — ведение уходит и не возвращается, локальные правки не помогут; "
                              "адрес — gap-closing и глобальная линковка (§6.92)" if back < 0.4 else
                              "СМЕШАННЫЙ режим: есть и уходы, и дребезг — мерить по семействам ниже"))
    print(f"   (самовозвраты {back:.0%}, самая длинная серия {frac:.0%} длины кривой)")

print(f"\n{'='*W}\nЦЕНА ВОССТАНОВЛЕНИЯ: через сколько строк надо возвращать к эталону для 90% на цели\n{'='*W}")
if v:
    got = [r for r in v if r["relatch_every"]]
    print(f"  кривых, где 90% достижимо возвратами: {len(got)} из {len(v)} "
          f"({100*len(got)/max(1,len(v)):.0f}%)")
    if got:
        e = np.array([r["relatch_every"] for r in got], float)
        nn = np.array([r["relatch_n"] for r in got], float)
        print(f"  интервал возврата: медиана {np.median(e):.0f} строк, p10 {np.percentile(e,10):.0f}, "
              f"p90 {np.percentile(e,90):.0f}")
        print(f"  возвратов на кривую: медиана {np.median(nn):.0f}")
    print(f"  ⚠ у {len(v)-len(got)} кривых 90% НЕ достигается даже возвратом каждые 25 строк — "
          f"там ведение по ранам не работает в принципе")

print(f"\n{'='*W}\nПО СЕМЕЙСТВАМ (тушь есть, прод не берёт)\n{'='*W}")
print(f"{'семейство':<13}{'кривых':>7}{'оракул':>8}{'история':>9}{'сам':>6}{'самовозвр.':>12}"
      f"{'длин.серия%':>13}{'возврат':>9}")
fam = {}
for r in R:
    if r["ink3"] >= 0.9 and not r["prod"]:
        fam.setdefault(r["fam"], []).append(r)
for f_, vv in sorted(fam.items(), key=lambda q: -len(q[1])):
    mx = np.array([x["maxlen_self"] for x in vv], float)
    nr = np.array([x["nrows"] for x in vv], float)
    ev = [x["relatch_every"] for x in vv if x["relatch_every"]]
    print(f"{f_[:12]:<13}{len(vv):>7}{np.mean([x['on_oracle'] for x in vv]):>7.0%}"
          f"{np.mean([x['on_hist'] for x in vv]):>9.0%}{np.mean([x['on_self'] for x in vv]):>6.0%}"
          f"{np.nanmedian([x['back_self'] for x in vv]):>11.0%}"
          f"{np.nanmedian(mx/np.maximum(1,nr)):>12.0%}"
          f"{(f'{int(np.median(ev))}' if ev else '—'):>9}")
