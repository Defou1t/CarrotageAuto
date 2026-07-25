r"""_cross_diag.py — ГДЕ ИМЕННО ТРАССА ТЕРЯЕТ СВОЮ КРИВУЮ (§6.56).

§6.50 отвёл ~50% всех потерь архива на срыв идентичности («чужая» + дыры внутри обрывков), а
§6.55 показал, что этот срыв ОДИНАКОВ у обоих путей эмиссии — значит он и есть главная линия.
Но до сих пор «срыв идентичности» был словом. Здесь он меряется: в КАКОЙ СТРОКЕ трасса уходит
со своей кривой и ЧТО в этой строке происходит.

МЕТОД (GT только для разметки места срыва, в отбор не входит):
 1. трасса привязывается к своей GT-кривой обычным назначением 1-к-1 (§6.36);
 2. идём по строкам сверху вниз, находим ПЕРВУЮ строку, где |trace − своя GT| > `--depart` px
    и остаётся большим (не одиночный выброс, а уход);
 3. в этой строке смотрим, БЫЛА ЛИ РЯДОМ ЧУЖАЯ КРИВАЯ: минимальное расстояние до любой ДРУГОЙ
    GT-кривой листа в той же строке.

ЧТО ЭТО РАЗЛИЧАЕТ:
  * срыв ПРИ СБЛИЖЕНИИ (чужая ближе `--near` px) — классическое пересечение, адрес P2 §6.8:
    идентичность надо вести, а не восстанавливать потом;
  * срыв В ОДИНОЧЕСТВЕ (рядом никого) — это НЕ пересечение, а обрыв/выцветание/полоса, и
    лечится совсем другим.
Пропорция между ними и решает, куда вкладываться.

  python _cross_diag.py [--depart 20] [--near 25]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

BASE = Path(r"F:\nds\output\taskS\pick_gate")
ap = argparse.ArgumentParser()
ap.add_argument("--caches", default=",".join(str(BASE / d) for d in
                                             ("cache", "holdout", "wide", "k2")))
ap.add_argument("--depart", type=float, default=20.0, help="px: уход со своей кривой")
ap.add_argument("--near", type=float, default=25.0, help="px: чужая кривая считается рядом")
ap.add_argument("--hold", type=int, default=50, help="строк подряд, чтобы уход был не выбросом")
a = ap.parse_args()
MINPTS = MINCOM = 30

near_cnt = alone_cnt = 0
near_gap, alone_gap = [], []
depart_frac = []
no_depart = 0
never_on = 0
total = 0
jumps, slides = [], []

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
            got[nm] = ti; used.add(ti)
        for nm, ti in got.items():
            tr, gm = cand[ti], GM[nm]
            ys = [y for y in sorted(tr) if y in gm]
            if len(ys) < a.hold * 2:
                continue
            total += 1
            err = np.array([abs(tr[y] - gm[y]) for y in ys])
            bad = err > a.depart
            # ⚠⚠ ОБЯЗАТЕЛЬНЫЙ ОТСЕВ (без него вывод ложный): назначение 1-к-1 ЖЁСТКОЕ, и если
            # хороших трасс меньше, чем кривых, кривой достаётся трасса, которая на ней НИКОГДА
            # не была. Такой случай выглядит как «срыв на 1% глубины», хотя срыва не было —
            # трасса просто чужая с самого начала. Считаем «ушла» только ту, что СНАЧАЛА шла
            # по своей кривой.
            if bool(bad[:a.hold].mean() > 0.5):
                never_on += 1
                continue
            # ПЕРВЫЙ уход, который держится `hold` строк подряд — иначе это одиночный выброс
            k = None
            run = 0
            for i, b in enumerate(bad):
                run = run + 1 if b else 0
                if run >= a.hold:
                    k = i - run + 1
                    break
            if k is None:
                no_depart += 1
                continue
            y = ys[k]
            depart_frac.append(k / len(ys))
            # ПРЫЖОК ИЛИ СПОЛЗАНИЕ: скачок самой трассы в момент срыва против роста ошибки за
            # следующие 200 строк. Прыжок = зацепилась за чужую тушь; сползание = увели полоса
            # и доводка. Лечения разные, поэтому меряем отдельно.
            jump = abs(tr[ys[k]] - tr[ys[k - 1]]) if k > 0 else 0.0
            kk = min(k + 200, len(ys) - 1)
            growth = float(err[kk] - err[max(0, k - 1)])
            (jumps if jump >= 10 else slides).append((jump, growth))
            # ЧТО РЯДОМ В ЭТОЙ СТРОКЕ: ближайшая ЧУЖАЯ кривая
            mine = gm[y]
            best = None
            for onm, ogm in GM.items():
                if onm == nm or y not in ogm:
                    continue
                dd = abs(ogm[y] - mine)
                best = dd if best is None else min(best, dd)
            if best is not None and best <= a.near:
                near_cnt += 1; near_gap.append(best)
            else:
                alone_cnt += 1
                if best is not None:
                    alone_gap.append(best)

n = near_cnt + alone_cnt
print(f"кривых с назначенной трассой: {total}")
print(f"  ни разу не ушли дальше {a.depart:.0f}px .......... {no_depart} ({100*no_depart/max(1,total):.0f}%)")
print(f"  ушли и уход разобран ...................... {n}\n")
if n:
    print(f"{'место срыва':<34}{'кривых':>8}{'доля':>7}   медианный зазор до чужой")
    print(f"{'★ ПРИ СБЛИЖЕНИИ (чужая <= ' + str(int(a.near)) + 'px)':<34}{near_cnt:>8}"
          f"{100*near_cnt/n:>6.0f}%   {np.median(near_gap) if near_gap else 0:.0f}px")
    print(f"{'  В ОДИНОЧЕСТВЕ (рядом никого)':<34}{alone_cnt:>8}{100*alone_cnt/n:>6.0f}%   "
          f"{np.median(alone_gap) if alone_gap else float('nan'):.0f}px")
    print(f"\nгде по глубине происходит срыв (доля пройденной кривой): "
          f"медиана {np.median(depart_frac):.2f}, четверть до {np.percentile(depart_frac, 25):.2f}")
