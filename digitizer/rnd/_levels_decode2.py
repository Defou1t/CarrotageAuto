r"""_levels_decode2.py — ВАРИАНТ ДЕКОДЕРА УРОВНЕЙ под уточнённый домен (Эдуард, 22.07).

Штатный `decode_levels.decode` писался под РЕЗИСТИВНЫЕ перевыносы ×5/×25 и несёт два допущения,
которые для backup right-left НЕВЕРНЫ:

  1. ЦЕНА ПЕРЕХОДА = lam·|Δlevel| — штраф РАСТЁТ с длиной прыжка. Но по домену возврат с любого
     backup сразу на 1× — нормальный ход пера, а цепочки бывают длиной 26-29 звеньев
     (YULIIV_007 MNSS1/MNDS21/MNDS31), где возврат стоит 29·lam и не выбирается НИКОГДА.
     → `flat=True`: цена перехода не зависит от |Δlevel|.
  2. НЕПРЕРЫВНОСТЬ ЗНАЧЕНИЯ меряется в ЛОГАРИФМЕ (`logv`). Для ×5 это уместно, для АДДИТИВНОГО
     backup (0-20 → 20-40 → 40-60) — нет: один и тот же ход пера получает разный вес у нуля и в
     середине диапазона.
     → `linear=True`: непрерывность в линейных значениях, нормированная шириной шкалы (иначе
     кривые с разными единицами несравнимы).

Остальное (направленный wrap-гейт, rail-гейт, level_bias, down_mult) сохранено БЕЗ изменений —
проверяются ровно две правки, а не «новый декодер».
"""
import sys
sys.path.insert(0, r"F:\nds\Auto\digitizer")
import numpy as np
from decode_levels import scale_map, logv


def is_additive(family):
    """АДДИТИВНАЯ цепочка (backup right: 0-20 → 20-40 → 40-60, размах ПОСТОЯНЕН) против
    МУЛЬТИПЛИКАТИВНОЙ (перевынос ×5: размах растёт кратно). Считается ИЗ САМОЙ РАМКИ — разметки
    и участия эксперта не требует.
    Замер 22.07 (342 кривые): на аддитивных линейная непрерывность даёт полноту переходов
    33%→73% на держанных скважинах, на мультипликативных ВРЕДИТ (81%→41%) ⇒ метрику надо
    выбирать по типу, а не назначать одну на всех."""
    sp = [abs(s["v_right"] - s["v_left"]) or 1e-9 for s in family]
    if len(sp) < 2:
        return False
    return float(np.median([sp[i + 1] / sp[i] for i in range(len(sp) - 1)])) < 1.5


def decode2(xs_by_row, family, lam=0.7, dxfrac=0.12, gate_w=4.0, level_bias=0.0, rail_gate=0.0,
            down_mult=1.0, flat=False, linear=False, auto=False, min_jump=0.0,
            rail_from_trace=False, rail_span=0.05, rail_hard=False):
    """min_jump — ЖЁСТКИЙ минимум прыжка пера для перехода, в долях ширины шкалы.

    Зачем отдельно от dxfrac. `dxfrac` задаёт МЯГКУЮ цену: consistency = min(1, |Δx|/dxthr), т.е.
    маленький порог просто делает переход дешёвым для всех. Замер 22.07 (150 кривых, |Δx| в долях
    ширины шкалы): переходы эксперта med **0.190**, наши совпавшие 0.211, наши ЛОЖНЫЕ **0.011** —
    в 20 раз меньше. Порог 0.05 сохраняет 87% экспертных и отсекает 83% ложных.
    Физика: при обороте перо идёт ОТ КРАЯ ДО КРАЯ, а не дрожит на месте."""
    if auto:
        linear = is_additive(family)
    maps = [scale_map(s) for s in family]
    K = len(family)
    rows = sorted(xs_by_row)
    if K <= 1 or len(rows) < 2:
        return {y: 0 for y in rows}
    width = abs(family[0]["x_right"] - family[0]["x_left"])
    dxthr = max(1.0, dxfrac * width)
    xl = min(family[0]["x_left"], family[0]["x_right"])
    # ★ УПОР БЕРЁТСЯ ИЗ САМОЙ ТРАССЫ, а не из x_right объявленной оси (замер 23.07, 5178 переходов
    # по СЫРЫМ вершинам): перед переходом ВВЕРХ перо стоит в 5% от собственного упора у 47%
    # переходов против 4% в случайных строках (12× обогащение); по объявленной оси сигнала нет.
    # Причина — СДВОЕННАЯ РАМКА (Эдуард): ось объявляют на весь лист (0-42.5 вместо 0-20), поэтому
    # тушь занимает медианно 50% её ширины, а перо упирается в край СВОЕЙ КОЛОНКИ.
    _xv = np.array([xs_by_row[y] for y in rows], float)
    rail_x = (float(np.percentile(_xv, 99.5)) - rail_span * width) if rail_from_trace \
        else (xl + rail_gate * width)
    # масштаб значений для ЛИНЕЙНОЙ непрерывности: полный размах шкалы уровня 0
    vspan = abs(family[0]["v_right"] - family[0]["v_left"]) or 1.0

    def val(k, x):
        if linear:
            return maps[min(k, K - 1)](x) / vspan
        return logv(maps, k, x)

    dp = [level_bias * k for k in range(K)]
    back = []
    for i in range(1, len(rows)):
        x = xs_by_row[rows[i]]; xp = xs_by_row[rows[i - 1]]; dx = x - xp
        ndp = [1e18] * K; bk = [0] * K
        lvp = [val(kp, xp) for kp in range(K)]
        for k in range(K):
            vk = val(k, x)
            best = 1e18; bki = 0
            for kp in range(K):
                if k == kp:
                    tc = 0.0
                elif min_jump and abs(dx) < min_jump * width:
                    tc = 1e17                          # ★ ЖЁСТКИЙ порог: перо не прыгнуло — оборота нет
                else:
                    nlev = k - kp
                    want = -1 if nlev > 0 else 1
                    aligned = (dx * want) > 0
                    mag = min(1.0, abs(dx) / dxthr)
                    consistency = mag if aligned else 0.0
                    step = 1.0 if flat else abs(nlev)      # ★ правка 1
                    tc = lam * step + gate_w * lam * (1.0 - consistency)
                    if nlev > 0 and (rail_gate > 0 or rail_from_trace) and xp < rail_x:
                        # rail_hard: переход ВВЕРХ вдали от упора не удорожается, а ЗАПРЕЩАЕТСЯ.
                        # Мягкий штраф при lam=0.05/gate_w=8 стоит всего 0.4 и не спасает кривые,
                        # у которых упор совпадает с краем оси (Semeguniv OGZ1: тушь до 2681 при
                        # крае 2685, а ложные переходы идут в середине шкалы).
                        tc = 1e17 if rail_hard else tc + gate_w * lam * (1.0 if flat else nlev)
                    if nlev < 0:
                        tc *= down_mult
                c = dp[kp] + (vk - lvp[kp]) ** 2 + tc + level_bias * k
                if c < best:
                    best = c; bki = kp
            ndp[k] = best; bk[k] = bki
        dp = ndp; back.append(bk)
    k = int(np.argmin(dp))
    lv = {rows[-1]: k}
    for i in range(len(rows) - 1, 0, -1):
        k = back[i - 1][k]; lv[rows[i - 1]] = k
    return lv
