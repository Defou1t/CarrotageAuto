r"""_cov_ink_sum.py — сборка шардов `_cov_ink.py` в один ответ.

⚠⚠ ПЕРВЫМ ДЕЛОМ ПЕЧАТАЕТ, СКОЛЬКО ШАРДОВ НАЙДЕНО ИЗ СКОЛЬКИ, И СВЕРЯЕТ ЧИСЛО КРИВЫХ СО СПИСКОМ.
13.08 сводка `_trace_ab_sum.py` дважды переписывалась в таблицу с неполного прогона (10 дампов из
12 подписаны как 11), и оба раза числа занижались. Здесь неполнота обязана быть видна в первой
строке и в подписи к каждому числу.

  python _cov_ink_sum.py [--dir F:\nds\output\taskS] [--list ...]
"""
import sys, argparse, re, collections
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS")
ap.add_argument("--list", default=r"F:\nds\output\taskS\cov_written_short.tsv")
a = ap.parse_args()

D = Path(a.dir)
files = sorted(D.glob("cov_ink_*of*.tsv"))
if not files:
    sys.exit("нет шардов cov_ink_*of*.tsv")
# ⚠⚠ ЗНАМЕНАТЕЛЬ — САМЫЙ СВЕЖИЙ ПО ВРЕМЕНИ, А НЕ ПО СТРОКЕ. Ровно этот дефект описан в
# `_trace_ab_sum.py`: маска `cov_ink_*of*.tsv` загребает дамп ДЫМОВОГО прогона (`_0of60.tsv`),
# а сортировка строк ставит "60" после "6" — и сводка молча собралась бы с одного дымового шарда.
den = sorted({f.stem.split("of")[1] for f in files},
             key=lambda d: max(f.stat().st_mtime for f in files if f.stem.endswith("of" + d)))[-1]
drop = [f.name for f in files if not f.stem.endswith("of" + den)]
files = [f for f in files if f.stem.endswith("of" + den)]
if drop:
    print(f"⚠ пропущены дампы ДРУГОГО разбиения: {', '.join(drop)}")
N = int(den)
full = len(files) == N
print(f"шардов {len(files)} из {N}" + ("" if full else "  ⚠⚠ ПРОГОН НЕПОЛНЫЙ — числа занижены"))

want = sum(1 for _ in open(a.list, encoding="utf-8")) - 1
rows = []
for f in files:
    for l in list(open(f, encoding="utf-8"))[1:]:
        rows.append(l.rstrip("\n").split("\t"))
print(f"кривых собрано {len(rows)} из {want} по списку   "
      + ("★ СОШЛОСЬ" if len(rows) == want else
         f"{'⚠ неполный прогон' if not full else '⛔ РАСХОЖДЕНИЕ ПРИ ПОЛНОМ ПРОГОНЕ'}"))

# ---- построчные доли из логов шардов (там три допуска, в TSV только рабочий) -----------------
tol_tot = collections.defaultdict(collections.Counter)
mism = 0
for lg in sorted(D.glob("cov_ink_sh*.log")):
    txt = lg.read_text(encoding="utf-8")
    cur = None
    for ln in txt.splitlines():
        m = re.match(r"ПРОПУЩЕННЫЕ СТРОКИ, допуск ±(\d+)px", ln)
        if m:
            cur = int(m.group(1)); continue
        m = re.match(r"\s+(вне_окна|есть_тушь|нет_туши)\s+(\d+)", ln)
        if m and cur is not None:
            tol_tot[cur][m.group(1)] += int(m.group(2))
    m = re.search(r"расходится со списком у (\d+) кривых", txt)
    if m:
        mism += int(m.group(1))

W = 96
print(f"  КОНТРОЛЬ cov по всем шардам: расходится у {mism} кривых   "
      + ("★ СОШЛОСЬ" if mism == 0 else "⛔ ЧИТАЕТСЯ НЕ ТОТ ОБЪЕКТ"))
print("=" * W)

for t in sorted(tol_tot):
    s = sum(tol_tot[t].values()) or 1
    mark = " ★(порог §6.23)" if t == 4 else ""
    print(f"\nПРОПУЩЕННЫЕ СТРОКИ ЭКСПЕРТА, допуск ±{t}px{mark}: всего {s}")
    for k in ("вне_окна", "есть_тушь", "нет_туши"):
        v = tol_tot[t][k]
        print(f"  {k:<12}{v:>10}{100*v/s:>7.1f}%  " + "█" * int(46 * v / s))

miss = np.array([int(r[3]) for r in rows], float)
ink = np.array([int(r[5]) for r in rows], float)
hole = np.array([int(r[6]) for r in rows], float)
cov0 = np.array([float(r[4]) for r in rows], float)
cova = np.array([float(r[7]) for r in rows], float)
ok = cova >= 0.9

print(f"\n{'='*W}\n★★ ГЛАВНОЕ: СКОЛЬКО КРИВЫХ ПЕРЕХОДЯТ ПОРОГ 0.9, ЕСЛИ ДОБИТЬ ТОЛЬКО СТРОКИ С ТУШЬЮ\n{'='*W}")
print(f"  дотянули до 0.9:      {int(ok.sum()):>5} из {len(rows)} ({100*ok.mean():.1f}%)")
print(f"  медиана cov: было {np.median(cov0):.2f} → стало {np.median(cova):.2f}")
print(f"  строк с тушью {int(ink.sum())} из {int(miss.sum())} пропущенных "
      f"({100*ink.sum()/max(1,miss.sum()):.1f}%), из них внутри пролёта {int(hole.sum())} "
      f"({100*hole.sum()/max(1,ink.sum()):.0f}%)")

print(f"\nРАСПРЕДЕЛЕНИЕ ПРИРОСТА ПОКРЫТИЯ (cov_после − cov_до)")
dl = cova - cov0
for lo, hi in ((-.001, .01), (.01, .05), (.05, .10), (.10, .20), (.20, .50), (.50, 1.01)):
    n = int(((dl > lo) & (dl <= hi)).sum())
    print(f"  {lo:+.2f}..{hi:+.2f}{n:>7}{100*n/len(dl):>7.1f}%  " + "█" * int(40 * n / len(dl)))

print(f"\nКТО ИМЕННО ДОТЯГИВАЕТ (по исходному покрытию)")
print(f"  {'cov до':<12}{'кривых':>8}{'дотянули':>10}{'доля':>8}")
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9)):
    sel = (cov0 > lo) & (cov0 <= hi)
    n = int(sel.sum())
    if n:
        print(f"  {lo:.1f}-{hi:.1f}{'':<6}{n:>8}{int(ok[sel].sum()):>10}"
              f"{100*ok[sel].mean():>7.0f}%")

print(f"\n⇒ ВЕРХНЯЯ ГРАНИЦА РЫЧАГА: {int(ok.sum())} кривых сверх нынешних 832 честных на этой "
      f"выборке = {100*(832+ok.sum())/6863:.1f}% против 12.1%")
print("⚠ Это ПОТОЛОК: он допускает идеальную добивку по всей туши и сохранение med≤3px. Механизм,"
      " который станет его брать, обязан пройти нуль-контроль §6.20.3.")
