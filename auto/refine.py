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


def _rolling_median(xs, win):
    n = len(xs); h = win // 2; out = np.empty(n)
    for i in range(n):
        out[i] = np.median(xs[max(0, i - h):min(n, i + h + 1)])
    return out


def robust_baseline(xs, base_win=301, frac=0.10, track_w=736):
    """Робастная оценка положения МАЖОРИТАРНОЙ линии (rolling-median, 2-й проход по нефлагнутым).
    Возвращает (base, flag): flag — строки, устойчиво отошедшие от базлайна (кандидаты свапа)."""
    xs = np.asarray(xs, float)
    thr = max(50.0, frac * track_w)
    base = _rolling_median(xs, base_win)
    flag = np.abs(xs - base) > thr
    if flag.any() and (~flag).sum() > 2:
        idx = np.arange(len(xs))
        xc = xs.copy(); xc[flag] = np.interp(idx[flag], idx[~flag], xs[~flag])
        base = _rolling_median(xc, base_win); flag = np.abs(xs - base) > thr
    return base, flag


def _swap_blocks(flag, min_len, merge_gap=8):
    """Блоки подряд-флагнутых строк длиной ≥ min_len (короче = спайк, чинит despike)."""
    blocks = []; i = 0; n = len(flag)
    while i < n:
        if not flag[i]:
            i += 1; continue
        j = i
        while j + 1 < n and (flag[j + 1] or (j + 1 - i < merge_gap and flag[min(n - 1, j + merge_gap)])):
            j += 1
        if j - i + 1 >= min_len:
            blocks.append((i, j))
        i = j + 1
    return blocks


def repair_swaps(rows_xs, fg, p, n_same_color=1):
    """РЕМОНТ УСТОЙЧИВЫХ СВАПОВ (Эдуард 12.07): трасса латчится на ПАРАЛЛЕЛЬНУЮ одноцветную кривую
    на десятки-сотни строк (не спайк — деспайк это не ловит; PZ чёрный ↔ DS/DN чёрные). Признак
    свапа (только по трассе + черниле, БЕЗ эксперта):
      (1) блок устойчиво далеко от робастного базлайна (положения мажоритарной линии);
      (2) у БАЗЛАЙНА есть чернило того же цвета (реальная линия там присутствует, доля ≥ ink_frac);
      (3) наша трасса на ДРУГОМ черниле, чем базлайн (|трасса−базлайн| ≥ gap_min) — иначе это
          РЕАЛЬНЫЙ вынос пера к рельсу (BK), не латч → НЕ ТРОГАЕМ.
    КУРС-ГЕЙТ: одинокая широко-качающаяся резистивная (BK: базлайн≠истина) → repair НЕ применяем.
    n_same_color — сколько кривых листа этого цвета (латч возможен только при ≥2). Возвращает
    (dict row→x, n_fixed)."""
    rows = sorted(rows_xs)
    if len(rows) < 3 * p.swap_min_len:
        return dict(rows_xs), 0
    xs = np.array([rows_xs[y] for y in rows], float)
    W = fg.shape[1]
    base, flag = robust_baseline(xs, p.swap_base_win, p.swap_frac, W)
    spread = (np.percentile(base, 85) - np.percentile(base, 15)) / W
    if spread >= p.swap_max_spread and n_same_color < 2:   # одинокая широкая (BK) → базлайн≠истина
        return dict(rows_xs), 0
    blocks = _swap_blocks(flag, p.swap_min_len)
    out = xs.copy(); fixed = 0
    for (i0, i1) in blocks:
        if float(np.median(np.abs(xs[i0:i1 + 1] - base[i0:i1 + 1]))) < p.swap_gap_min:
            continue                                   # трасса ~на базлайне → реальный вынос, не свап
        snaps = []; ink = 0
        for k in range(i0, i1 + 1):
            tgt = base[k]; y = rows[k]
            a = max(0, int(tgt) - p.swap_band); b = min(W, int(tgt) + p.swap_band + 1)
            cols = np.where(fg[y, a:b])[0]
            if len(cols):
                snaps.append((k, float(a + cols[np.argmin(np.abs(cols + a - tgt))]))); ink += 1
            else:
                snaps.append((k, float(tgt)))
        if ink / (i1 - i0 + 1) < p.swap_ink_frac:      # у базлайна чернила нет → реальный вынос
            continue
        for k, sx in snaps:
            out[k] = sx
        fixed += 1
    return {y: float(out[i]) for i, y in enumerate(rows)}, fixed


def refine_trace(fg, line, frame, p, trace_line, track, n_same_color=1):
    """REFINE-ПЕТЛЯ трассы одной AUTO-линии (Эдуард 12.07): трасса → верификатор → перетрасс.
    (1) тесный band (полоса линии). (2) детект НЕДОТЯГА: у правого/левого края band есть чернило
    того же цвета за границей (перо ушло к упору, а трасса не догнала — калибровка: BK терял выносы
    к рельсу). (3) при недотяге — перетрасс с band ДО ТРЕКА (можно за Scale Axis, Эдуард: значения
    экстраполируются). (4) деспайк. Берём вариант с бОльшим охватом упора без роста спайков."""
    import numpy as np
    # ЖЁСТКИЙ предел полосы (understand._single_curve_rescue): свипующее перо (MBK) — «дотяг до
    # упора» ниже расширял band ДО ТРЕКА и затаскивал трассу на вертикаль рамки (QC 18.07).
    hlo = getattr(line, "x_hard_lo", None); hhi = getattr(line, "x_hard_hi", None)
    hard = hlo is not None and hhi is not None
    tr = (trace_line(fg, line, frame, p, x_range=(int(hlo), int(hhi))) if hard
          else trace_line(fg, line, frame, p))
    if len(tr) < 30:
        return tr
    band_pad = 8
    lo = max(0, int(line.x_lo) - band_pad); hi = min(fg.shape[1], int(line.x_hi) + band_pad + 1)
    tl, tr_ = max(0, track.x_left), min(fg.shape[1], track.x_right)
    if hard:
        lo = max(lo, int(hlo)); hi = min(hi, int(hhi))
        tl = max(tl, int(hlo)); tr_ = min(tr_, int(hhi))
    # НЕДОТЯГ = band отрезал чернило: доля строк трассы, где ТОГО ЖЕ ЦВЕТА чернило есть ЗА band
    # (перо ушло к упору, band это срезал). Не «трасса у края» (недотяг = трасса НЕ дошла).
    rows = sorted(tr)
    beyond = 0
    for y in rows:
        if (hi < tr_ and fg[y, hi:tr_].any()) or (tl < lo and fg[y, tl:lo].any()):
            beyond += 1
    if beyond / max(1, len(rows)) >= 0.03:       # ≥3% строк с чернилом за band → band до трека
        tr_wide = trace_line(fg, line, frame, p, x_range=(tl, tr_))
        if len(tr_wide) >= 0.8 * len(tr):
            xw = np.array([tr_wide[y] for y in sorted(tr_wide)])
            xt = np.array([tr[y] for y in rows])
            # расширенная лучше, если реально достаёт дальше к упору (охват вырос)
            if xw.max() > xt.max() + 5 or xw.min() < xt.min() - 5:
                tr = tr_wide
    tr, _ = despike(tr, win=p.despike_win, k=p.despike_k, min_jump=p.despike_min_jump)
    # ПРИМ (session2): repair_swaps НЕ вызывается в продакшн-пути. На узкой выборке (40 планшетов
    # BEZLUD/BOGAT_002) выглядел net-плюсом (PZ 0.82→0.96), но широкая проверка (130 планшетов,
    # плашки BOGAT BKZ с 7 слипшимися чёрными GZ) дала net −1.71пп и КАТАСТРОФЫ (GZ31 0.90→0.28
    # одной правкой): «снап к ближайшему черницу базлайна» хватает НЕ ту из пучка одноцветных.
    # Безопасен только для одиночной линии с ОДНИМ хорошо разделённым одноцветным соседом (PZ↔DS).
    # Оставлен как инструмент; гейт для пучков не решён. [[track4]]
    return tr


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
