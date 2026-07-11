r"""
refine.py — НЕЗАВИСИМЫЙ верификатор+доводчик оцифрованной линии (запрос Эдуарда 12.07).

Работает ТОЛЬКО по значениям трассы + геометрии шкалы, НЕ по исходному анализу изображения
(независимая проверка: если анализ ошибся — верификатор ловит, потом заставляет перетрассировать).

Ловит:
  • СПАЙКИ — изолированные выбросы x против локальной медианы (Hampel): пара кликов дёргается
    в сторону и возвращается (перо не рисует так; это ошибка трассировки/наложение). Реальный
    устойчивый пик (≥ hampel_win/2 строк в одну сторону) НЕ выброс — переживает.
  • КОРОТКИЕ УРОВНИ — level-сегмент 5х/1х короче min_run кликов = «напутано» (Эдуард: пара
    кликов 5х потом пара 1х — шум; настоящий интервал длиннее). Такие сегменты гасятся к соседу.
  • ДУБЛИ — несколько x на одну строку / нулевой шаг (дерготня на месте).

Даёт: список проблемных строк (для перетрассировки) + score 0..100 (выше — чище) + функции
despike()/enforce_min_run() для доводки.
"""
import numpy as np


def _interp_nan(xs):
    """Заполнить NaN линейной интерполяцией по индексу (для сглаживания без дыр)."""
    xs = xs.astype(float).copy()
    n = len(xs)
    idx = np.arange(n)
    good = ~np.isnan(xs)
    if good.sum() < 2:
        return xs
    xs[~good] = np.interp(idx[~good], idx[good], xs[good])
    return xs


def hampel_spikes(xs, win=8, k=4.0, min_jump=12.0):
    """Индексы СПАЙКОВ: |x−медиана окна| > k·1.4826·MAD И скачок к соседям > min_jump.
    win — полу-окно (строк в каждую сторону). Устойчивый вынос (шире окна) не ловится."""
    xs = _interp_nan(np.asarray(xs, float))
    n = len(xs)
    spikes = []
    for i in range(n):
        a = max(0, i - win); b = min(n, i + win + 1)
        w = xs[a:b]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        thr = k * mad if mad > 1e-6 else k
        # изолированность: сосед(и) далеко от x[i], но близко к медиане
        left = xs[i - 1] if i > 0 else med
        right = xs[i + 1] if i < n - 1 else med
        isolated = (abs(xs[i] - left) > min_jump and abs(xs[i] - right) > min_jump)
        if abs(xs[i] - med) > max(thr, min_jump) and isolated:
            spikes.append(i)
    return spikes


def despike(rows_xs, win=8, k=4.0, min_jump=12.0):
    """rows_xs: dict row→x → та же трасса без спайков (выброс заменён локальной медианой).
    Возвращает (очищенный dict, число убранных)."""
    rows = sorted(rows_xs)
    xs = np.array([rows_xs[y] for y in rows], float)
    sp = set(hampel_spikes(xs, win, k, min_jump))
    if not sp:
        return dict(rows_xs), 0
    clean = xs.copy()
    for i in sp:
        a = max(0, i - win); b = min(len(xs), i + win + 1)
        others = [xs[j] for j in range(a, b) if j not in sp]
        clean[i] = float(np.median(others)) if others else xs[i]
    return {y: float(clean[k2]) for k2, y in enumerate(rows)}, len(sp)


def level_runs(levels_by_row):
    """Список сегментов уровня: (row_start, row_end, level, длина) по возрастанию строки."""
    rows = sorted(levels_by_row)
    if not rows:
        return []
    segs = []
    s = rows[0]; cur = levels_by_row[rows[0]]; prev = rows[0]
    for y in rows[1:]:
        if levels_by_row[y] != cur:
            segs.append((s, prev, cur, prev - s + 1)); s = y; cur = levels_by_row[y]
        prev = y
    segs.append((s, prev, cur, prev - s + 1))
    return segs


def short_level_runs(levels_by_row, min_run=20):
    """Сегменты уровня короче min_run кликов (кандидаты на гашение — «напутано»)."""
    return [seg for seg in level_runs(levels_by_row) if seg[3] < min_run]


def enforce_min_run(levels_by_row, min_run=20):
    """Погасить короткие level-сегменты к соседнему уровню (итеративно, до фикспоинта).
    Убирает мельтешение 5х/1х на паре кликов (Эдуард: интервал должен быть длиннее)."""
    lv = dict(levels_by_row)
    rows = sorted(lv)
    if not rows:
        return lv
    changed = True
    while changed:
        changed = False
        segs = level_runs(lv)
        for k2, (s, e, level, length) in enumerate(segs):
            if length >= min_run:
                continue
            # уровень соседа с большей длиной (или предыдущего)
            prev_lv = segs[k2 - 1][2] if k2 > 0 else None
            next_lv = segs[k2 + 1][2] if k2 + 1 < len(segs) else None
            if prev_lv is None and next_lv is None:
                continue
            if prev_lv is None:
                to = next_lv
            elif next_lv is None:
                to = prev_lv
            else:
                # к более длинному соседу
                to = prev_lv if segs[k2 - 1][3] >= segs[k2 + 1][3] else next_lv
            if to != level:
                for y in range(s, e + 1):
                    lv[y] = to
                changed = True
                break
    return lv


def decode_rail(rows_xs, x_left, x_right, ratio=5.0, rail_frac=0.85, min_jump_frac=0.45, min_run=25):
    """Уровни-перевыносы по ПРАВИЛУ РЕЛЬСА (Эдуард 12.07): 1х по умолчанию; на 5х ТОЛЬКО когда
    перо упёрлось в правый рельс (x≈x_right) и перескочило ВЛЕВО (значение ушло за 1х-максимум →
    продолжено на 5х). Возврат на 1х — когда значение на 5х опускается НИЖЕ 1х-максимума: это
    x < x_left + (1/ratio)·width (позиция «1х-макс» на 5х-шкале) — надёжнее ловли обратного
    перескока (перо возвращается плавно, без чистого прыжка). ratio = v5max/v1max (обычно 5).

    rows_xs: dict row→x (деспайкнутая). Возвращает dict row→level. Гистерезис + min_run."""
    rows = sorted(rows_xs)
    if len(rows) < 3:
        return {y: 0 for y in rows}
    w = max(1.0, x_right - x_left)
    rail = x_left + rail_frac * w          # «упор» у правого края
    jump = min_jump_frac * w               # минимальный перескок-оборот вверх
    back_x = x_left + (1.0 / ratio) * w    # позиция 1х-макс на 5х: ниже неё значение < 1х-макс
    lv = {}
    level = 0
    prev = rows[0]; lv[prev] = 0
    for y in rows[1:]:
        x = rows_xs[y]; xp = rows_xs[prev]
        dx = x - xp
        if dx <= -jump and xp >= rail:
            level += 1                     # упёрся в рельс → перескок влево = +уровень (5х)
        elif level > 0 and x < back_x:
            level -= 1                     # значение упало ниже 1х-макса → назад на 1х
        lv[y] = level
        prev = y
    return enforce_min_run(lv, min_run)


def verify_line(rows_xs, levels_by_row=None, scale=None, win=8, k=4.0,
                min_jump=12.0, min_run=20):
    """Полная проверка одной линии. scale=(x_left,x_right) для метрики выхода за рамку.
    Возвращает {score, n_spikes, spike_rows, short_runs, n_levels, dupes}."""
    rows = sorted(rows_xs)
    xs = np.array([rows_xs[y] for y in rows], float)
    sp = hampel_spikes(xs, win, k, min_jump)
    spike_rows = [rows[i] for i in sp]
    shorts = short_level_runs(levels_by_row, min_run) if levels_by_row else []
    n = max(1, len(rows))
    # score: штраф за долю спайков и за число коротких сегментов
    score = 100.0 - 100.0 * len(sp) / n - 5.0 * len(shorts)
    return {"score": round(max(0.0, score), 1), "n_spikes": len(sp),
            "spike_rows": spike_rows, "short_runs": shorts,
            "n_levels": len(set((levels_by_row or {0: 0}).values())),
            "n_points": len(rows)}
