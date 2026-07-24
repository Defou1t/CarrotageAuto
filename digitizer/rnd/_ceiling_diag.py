r"""_ceiling_diag.py — КУДА ДЕВАЮТСЯ 342 КРИВЫЕ ИЗ 417 (разбор ПОТОЛКА, §6.50).

§6.49 закрыл отбор: из потолка 75 честных прод берёт 55. Но сам потолок — 75 из 417, то есть
82% кривых теряются ДО отбора. Пока не известно, где именно, любая работа над качеством —
угадывание. Этот разбор отвечает на вопрос дешёво: кэш `_pick_gate.py` уже содержит и трассы, и
плотный GT по 106 листам, новых прогонов пайплайна не нужно.

КЛАССИФИКАЦИЯ каждой GT-кривой (по ЛУЧШЕЙ из ВСЕХ трасс листа, назначение 1-к-1 как в §6.36):
  ★ЧЕСТНАЯ      med<=3 И cov>=0.9 — то, что считается результатом;
  ОБРЫВОК       med<=3, но cov<0.9 — трасса ТОЧНА, но покрывает не всю кривую;
  СМЕЩЕНА       3<med<=20 — рядом, но не по кривой (полоса/дрожание);
  ЧУЖАЯ         med>20 — трасса ушла на соседнюю кривую;
  НЕТ ТРАССЫ    ни одной трассы с >=30 общих строк — кривую не вели вообще.

★ ГЛАВНЫЙ РАЗРЕЗ — ХВАТИЛО ЛИ ТРАСС: если пайплайн нашёл линий МЕНЬШЕ, чем в рамке кривых
(N < K), часть кривых обречена независимо от качества трассировки. Это не потеря точности, а
потеря ДЕТЕКЦИИ, и лечится в другом месте (U1/confidence, а не refine/trace2d).

  python _ceiling_diag.py [--caches dir1,dir2,dir3]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

BASE = Path(r"F:\nds\output\taskS\pick_gate")
ap = argparse.ArgumentParser()
ap.add_argument("--caches", default=str(BASE / "cache") + "," + str(BASE / "holdout") + ","
                            + str(BASE / "wide"))
a = ap.parse_args()
MINPTS, MINCOM = 30, 30


def classify(med, cov):
    if med is None:
        return "НЕТ ТРАССЫ"
    if med <= 3 and cov >= 0.9:
        return "★ЧЕСТНАЯ"
    if med <= 3:
        return "ОБРЫВОК"
    if med <= 20:
        return "СМЕЩЕНА"
    return "ЧУЖАЯ"


ORDER = ["★ЧЕСТНАЯ", "ОБРЫВОК", "СМЕЩЕНА", "ЧУЖАЯ", "НЕТ ТРАССЫ"]
tot = {k: 0 for k in ORDER}
by_enough = {True: {k: 0 for k in ORDER}, False: {k: 0 for k in ORDER}}
sheets_enough = sheets_short = 0
curves_enough = curves_short = 0
zero_trace_sheets = 0
covs = []

for d in a.caches.split(","):
    for f in sorted(Path(d).glob("*.pkl")):
        p = pickle.load(open(f, "rb"))
        cand = [t for t in p["traces"] if len(t) >= MINPTS]
        GM = p["GM"]
        if not GM:
            continue
        enough = len(cand) >= len(GM)          # хватило ли вообще трасс на все кривые
        if not cand:
            zero_trace_sheets += 1
        sheets_enough += enough
        sheets_short += (not enough)
        curves_enough += len(GM) * enough
        curves_short += len(GM) * (not enough)
        # оракульное назначение 1-к-1 среди ВСЕХ трасс (это и есть потолок)
        pairs = []
        for nm, gm in GM.items():
            for ti, tr in enumerate(cand):
                com = [y for y in tr if y in gm]
                if len(com) < MINCOM:
                    continue
                dd = np.array([abs(tr[y] - gm[y]) for y in com])
                pairs.append((float(np.median(dd)), len(com) / len(gm), nm, ti))
        pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
        got, used = {}, set()
        for med, cov, nm, ti in pairs:
            if nm in got or ti in used:
                continue
            got[nm] = (med, cov); used.add(ti)
        for nm in GM:
            med, cov = got.get(nm, (None, 0.0))
            k = classify(med, cov)
            tot[k] += 1
            by_enough[enough][k] += 1
            if k == "ОБРЫВОК":
                covs.append(cov)

N = sum(tot.values())
print(f"листов: трасс хватило {sheets_enough}, НЕ ХВАТИЛО {sheets_short} "
      f"(в т.ч. {zero_trace_sheets} без единой трассы)")
print(f"кривых: {N}   (на листах с достатком трасс {curves_enough}, с нехваткой {curves_short})\n")
print(f"{'исход кривой':<14}{'всего':>8}{'доля':>8}   {'трасс хватило':>16}{'трасс НЕ хватило':>18}")
for k in ORDER:
    e, s = by_enough[True][k], by_enough[False][k]
    print(f"{k:<14}{tot[k]:>8}{100*tot[k]/N:>7.0f}%   {e:>10} {100*e/max(1,curves_enough):>4.0f}%"
          f"{s:>12} {100*s/max(1,curves_short):>4.0f}%")

lost = N - tot["★ЧЕСТНАЯ"]
print(f"\nПОТЕРЯНО {lost} кривых из {N}. Из них:")
print(f"  нехватка ТРАСС (детекция линий) ....... {curves_short - by_enough[False]['★ЧЕСТНАЯ']:>4}"
      f"  {100*(curves_short - by_enough[False]['★ЧЕСТНАЯ'])/lost:>4.0f}%")
print(f"  трассы были, но не легли на кривую .... {curves_enough - by_enough[True]['★ЧЕСТНАЯ']:>4}"
      f"  {100*(curves_enough - by_enough[True]['★ЧЕСТНАЯ'])/lost:>4.0f}%")
if covs:
    print(f"\nОБРЫВКИ: медианное покрытие {np.median(covs):.2f}, "
          f"четверть хуже {np.percentile(covs, 25):.2f} — "
          f"вопрос ДОТЯГИВАНИЯ концов, не точности")

# ─── ГДЕ ИМЕННО ТЕРЯЕТСЯ ПОКРЫТИЕ У ОБРЫВКОВ. Трасса ТОЧНА (med<=3), но короткая. Две разные
# болезни: не дотянули КОНЦЫ (лечится продлением, `trace2d._extend_ends`) или ДЫРЫ В СЕРЕДИНЕ
# (трасса рвётся на пересечениях — это уже идентичность). Разделять обязательно: адреса разные.
sh_top = sh_bot = sh_hole = 0.0
nsh = 0
for d in a.caches.split(","):
    for f in sorted(Path(d).glob("*.pkl")):
        p = pickle.load(open(f, "rb"))
        cand = [t for t in p["traces"] if len(t) >= MINPTS]
        GM = p["GM"]
        if not GM or not cand:
            continue
        pairs = []
        for nm, gm in GM.items():
            for ti, tr in enumerate(cand):
                com = [y for y in tr if y in gm]
                if len(com) < MINCOM:
                    continue
                dd = np.array([abs(tr[y] - gm[y]) for y in com])
                pairs.append((float(np.median(dd)), len(com) / len(gm), nm, ti))
        pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
        got, used = {}, set()
        for med, cov, nm, ti in pairs:
            if nm in got or ti in used:
                continue
            got[nm] = (med, cov, ti); used.add(ti)
        for nm, (med, cov, ti) in got.items():
            if not (med <= 3 and cov < 0.9):
                continue
            gy = sorted(GM[nm]); tr = cand[ti]
            lo, hi = min(tr), max(tr)
            miss = [y for y in gy if y not in tr]
            if not miss:
                continue
            nsh += 1
            n = len(gy)
            sh_top += sum(1 for y in miss if y < lo) / n
            sh_bot += sum(1 for y in miss if y > hi) / n
            sh_hole += sum(1 for y in miss if lo <= y <= hi) / n
if nsh:
    print(f"\nГДЕ ТЕРЯЮТ ПОКРЫТИЕ {nsh} ОБРЫВКОВ (доля кривой, в среднем):")
    print(f"  не дотянули СВЕРХУ ....... {100*sh_top/nsh:>5.1f}%")
    print(f"  не дотянули СНИЗУ ........ {100*sh_bot/nsh:>5.1f}%")
    print(f"  ДЫРЫ ВНУТРИ трассы ....... {100*sh_hole/nsh:>5.1f}%")
    ends = sh_top + sh_bot
    print(f"  ⇒ концы {100*ends/(ends+sh_hole):.0f}% недобора против дыр "
          f"{100*sh_hole/(ends+sh_hole):.0f}%")

# ─── РАЗРЕЗ ПО ЧИСЛУ КРИВЫХ НА ЛИСТЕ. Прямо отвечает на вопрос §6.7/§6.8: годится ли транш
# K=2 (21% архива) на существующем пайплайне — БЕЗ его запуска.
print(f"\n{'кривых на листе':<16}{'листов':>7}{'кривых':>8}{'★честных':>10}{'доля':>7}"
      f"{'чужая':>8}{'нет трассы':>12}")
groups = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-7", 5, 7), ("8+", 8, 10 ** 6)]
for label, lo, hi in groups:
    ns = nc = nh = nf = nz = 0
    for d in a.caches.split(","):
        for f in sorted(Path(d).glob("*.pkl")):
            p = pickle.load(open(f, "rb"))
            GM = p["GM"]
            if not GM or not (lo <= len(GM) <= hi):
                continue
            cand = [t for t in p["traces"] if len(t) >= MINPTS]
            pairs = []
            for nm, gm in GM.items():
                for ti, tr in enumerate(cand):
                    com = [y for y in tr if y in gm]
                    if len(com) < MINCOM:
                        continue
                    dd = np.array([abs(tr[y] - gm[y]) for y in com])
                    pairs.append((float(np.median(dd)), len(com) / len(gm), nm, ti))
            pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
            got, used = {}, set()
            for med, cov, nm, ti in pairs:
                if nm in got or ti in used:
                    continue
                got[nm] = (med, cov); used.add(ti)
            ns += 1; nc += len(GM)
            for nm in GM:
                k = classify(*got.get(nm, (None, 0.0)))
                nh += k == "★ЧЕСТНАЯ"; nf += k == "ЧУЖАЯ"; nz += k == "НЕТ ТРАССЫ"
    if nc:
        print(f"{label:<16}{ns:>7}{nc:>8}{nh:>10}{100*nh/nc:>6.0f}%{nf:>8}{nz:>12}")
