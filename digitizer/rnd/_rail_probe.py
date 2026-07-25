r"""_rail_probe.py — ГДЕ НА САМОМ ДЕЛЕ ЛЕЖИТ «УПОР» ПЕРА (подсказка Эдуарда, 23.07).

Эдуард: при переходе на новый масштаб на ТОЙ ЖЕ ГЛУБИНЕ появляется линия с ТЕМ ЖЕ ЗНАЧЕНИЕМ.
И отдельно: рамка часто СДВОЕННАЯ — вместо 0-20 объявляют 0-42.5 (или 0-8 → 0-19), чтобы одной
шкалой закрыть весь лист и не рисовать вторую рамку.

ЧТО ЭТО МЕНЯЕТ. §6.40 закрыл гейт по упору как «не разделяющий»: у 96 кривых доля строк у
`x_right` оси ≈0, а переходы есть. Разбор контрольного листа объясняет почему:
  Semeguniv GZ11: ось объявлена x[448..2685] (ВСЯ ширина листа), но тушь живёт в 443..1468 —
  дальше начинается ВТОРАЯ КОЛОНКА (1708..2681). Перо упирается в край СВОЕЙ КОЛОНКИ (~1468),
  что по шкале 0 даёт значение 1.73, и продолжает на 5×-шкале при том же значении:
  x = 448 + 1.73/19*2237 = 651, фактически 650.
⇒ Упор — это НЕ `x_right` объявленной оси, а правый край фактической колонки кривой.

ЧТО МЕРИТСЯ (по экспертным трассам, без картинки):
 1. RAIL_HIT — на скольких переходах значение ПЕРЕД переходом близко к «значению упора»
    (значение, соответствующее p99.5 x самой трассы). Контроль: та же доля в случайных строках.
 2. WRAP_PREDICT — насколько точно позиция ПОСЛЕ перехода предсказывается формулой
    «то же значение, другая шкала»: |x_факт − x_предсказ| в px и в долях ширины.
 3. Сколько кривых вообще имеют «сдвоенную» рамку (тушь занимает малую долю объявленной оси).

  python _rail_probe.py [--sheets 60]
"""
import sys, io, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
import decode_levels as DL
from _multi_replica_probe import dense
from _decoder_data import train_sheets

ap = argparse.ArgumentParser()
ap.add_argument("--sheets", type=int, default=60)
a = ap.parse_args()


def inv(s, v):
    """Значение → x на шкале s (обратное scale_map)."""
    dv = (s["v_right"] - s["v_left"]) or 1e-9
    return s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / dv


RAIL_HIT, RAIL_RND, PRED_ERR, FRAC = [], [], [], []
NEAR_EXPL = 0
TOT_TR = 0
rows_out = []
for n in train_sheets(a.sheets):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = extract(str(n))
    except Exception:
        continue
    for c in ds.real_curves(m):
        fam = DL.build_family(m, c)
        gl = DL.gt_levels(c)
        if len(fam) < 2 or not gl:
            continue
        d = dense(c)
        if len(d) < 200:
            continue
        rows = sorted(d)
        xs = np.array([d[y] for y in rows], float)
        maps = [DL.scale_map(s) for s in fam]
        wdecl = abs(fam[0]["x_right"] - fam[0]["x_left"]) or 1
        FRAC.append((xs.max() - xs.min()) / wdecl)
        # ФАКТИЧЕСКИЙ УПОР кривой: край её собственного размаха (p99.5, устойчиво к выбросам)
        rail_x = float(np.percentile(xs, 99.5))
        tr = [i for i in range(1, len(rows)) if gl.get(rows[i], 0) != gl.get(rows[i - 1], 0)]
        if not tr:
            continue
        # 1) близость к упору ПЕРЕД переходом ВВЕРХ (только вверх: вниз возврат плавный)
        ups = [i for i in tr if gl.get(rows[i], 0) > gl.get(rows[i - 1], 0)]
        for i in ups:
            RAIL_HIT.append(abs(xs[i - 1] - rail_x) / wdecl)
        rnd = np.random.default_rng(0).choice(len(rows), size=min(len(ups) * 3 + 5, len(rows)),
                                              replace=False)
        for i in rnd:
            RAIL_RND.append(abs(xs[i] - rail_x) / wdecl)
        # 2) предсказание позиции ПОСЛЕ перехода: то же значение, другая шкала
        for i in tr:
            kp, k = gl.get(rows[i - 1], 0), gl.get(rows[i], 0)
            if kp >= len(fam) or k >= len(fam):
                continue
            v_before = maps[kp](xs[i - 1])
            x_pred = inv(fam[k], v_before)
            PRED_ERR.append(abs(xs[i] - x_pred) / wdecl)
            TOT_TR += 1
            if abs(xs[i] - x_pred) / wdecl <= 0.05:
                NEAR_EXPL += 1
        rows_out.append((n.parent.parent.name, c["name"].split()[0], len(fam), rail_x,
                         maps[0](rail_x), len(tr)))

R = np.array(RAIL_HIT); Rr = np.array(RAIL_RND); P = np.array(PRED_ERR); F = np.array(FRAC)
print(f"кривых с цепочкой ≥2: {len(rows_out)}, переходов: {TOT_TR}\n")
print("1) СДВОЕННАЯ РАМКА (тушь против объявленной оси):")
print(f"   доля объявленной ширины, занятая тушью: med {np.median(F):.2f}  "
      f"p10 {np.percentile(F,10):.2f}  p90 {np.percentile(F,90):.2f}")
print(f"   кривых, где тушь занимает <60% оси: {100*(F<0.6).mean():.0f}%  "
      f"<40%: {100*(F<0.4).mean():.0f}%")
print("\n2) БЛИЗОСТЬ К ФАКТИЧЕСКОМУ УПОРУ перед переходом ВВЕРХ (в долях объявленной ширины):")
print(f"   в точках перехода : med {np.median(R):.3f}  доля <0.05: {100*(R<0.05).mean():.0f}%  n={len(R)}")
print(f"   в случайных строках: med {np.median(Rr):.3f}  доля <0.05: {100*(Rr<0.05).mean():.0f}%  n={len(Rr)}")
print("\n3) ПРЕДСКАЗАНИЕ ПОЗИЦИИ ПОСЛЕ ПЕРЕХОДА «то же значение, другая шкала»:")
print(f"   |x_факт − x_предсказ| / ширина: med {np.median(P):.3f}  p25 {np.percentile(P,25):.3f}  "
      f"p75 {np.percentile(P,75):.3f}")
print(f"   переходов, объяснённых формулой в пределах 5% ширины: {100*NEAR_EXPL/max(1,TOT_TR):.0f}%")
print("\nЧитать: если (2) в точках перехода СИЛЬНО ближе к упору, чем в случайных строках —")
print("признак рабочий, и упор надо брать из САМОЙ ТРАССЫ, а не из x_right объявленной оси.")
