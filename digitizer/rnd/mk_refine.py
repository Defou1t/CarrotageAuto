r"""Итеративная доводка трасс MK по ЧЕРНИЛАМ (QC эксперта 03.07: «не дотягивает линии;
после дотяжки до пиковых значений сгладить и пройтись повторно — несколько раз анализировать»).

Один пасс:
  1. ДОТЯЖКА: строка, где трасса сидит в связном тёмном ране, который тянется дальше неё
     существенно дальше толщины штриха (горизонтальный выброс/пик), → точка к дальнему краю
     рана минус полутолщина (вершина). Ран общий с другой трассой → выброс отдаётся той,
     что БЛИЖЕ к дальнему краю (вторая остаётся на теле).
  2. СГЛАЖИВАНИЕ дрожи: лёгкая тяга к median-3 соседей, только где |Δ| < 3px — спайки
     не трогаем (пиковая кривая — домен).
Пассы повторяются (default 3): после дотяжки меняется локальная форма — следующий анализ
дотягивает дальше по хвосту пика. NB: prob здесь не нужна — работаем по native-скану.
"""
import numpy as np


def _run_at(row, x, W, gap=2, max_half=140, labrow=None, bridge=28, hops=3):
    """Связный тёмный ран вокруг x (допуск gap светлых px). labrow (строка карты 2D-компонент):
    ран МОСТИТСЯ через зазор до bridge px, если тёмное за зазором — ТА ЖЕ компонента
    (лепестки горба соединены с телом диагонально, построчно рвутся — QC №5, 3180).
    Возврат (a, b) или None."""
    xi = int(round(x))
    if not (0 <= xi < W) or not row[xi]:
        for d in (1, -1, 2, -2):                       # трасса могла встать на светлый зазор
            if 0 <= xi + d < W and row[xi + d]:
                xi += d; break
        else:
            return None
    lab = labrow[xi] if labrow is not None else None

    def grow(p, step, lim):
        g = 0
        while p + step != lim:
            q = p + step
            if row[q]:
                p = q; g = 0
            elif g < gap:
                p = q; g += 1
            else:
                break
        return p - step * g

    def hop(p, step, lim):
        for h in range(1, bridge + 1):                 # мост по 2D-компоненте
            q = p + step * h
            if q == lim:
                return None
            if row[q]:
                return q if (lab is not None and labrow[q] == lab) else None
        return None

    a = grow(xi, -1, max(0, xi - max_half) - 1)
    for _ in range(hops):
        j = hop(a, -1, max(0, xi - max_half) - 1) if lab is not None else None
        if j is None:
            break
        a = grow(j, -1, max(0, xi - max_half) - 1)
    b = grow(xi, +1, min(W - 1, xi + max_half) + 1)
    for _ in range(hops):
        j = hop(b, +1, min(W - 1, xi + max_half) + 1) if lab is not None else None
        if j is None:
            break
        b = grow(j, +1, min(W - 1, xi + max_half) + 1)
    return a, b


def _stroke_width(dark, tr, sample=7):
    """Медианная толщина штриха трассы (по каждой sample-й строке)."""
    ws = []
    ys = sorted(tr)
    W = dark.shape[1]
    for y in ys[::sample]:
        r = _run_at(dark[y], tr[y], W)
        if r:
            ws.append(r[1] - r[0] + 1)
    return float(np.median(ws)) if ws else 4.0


def refine_pass(dark, tr, other, w0, max_ext=90, probc=None, prob_min=0.10, labels=None):
    """Один пасс дотяжки. Возврат (новая трасса, число дотянутых строк).
    probc (канал модели, native): тянуть только куда модель видит СВОЮ кривую —
    кляксы/пятна она подавляет (LEVEN-регресс без гейта: MPZ 17.6→21px, гейт 03.07),
    реальные горбы — нет."""
    W = dark.shape[1]
    out = dict(tr)
    spike_thr = max(3.0 * w0, w0 + 6.0)
    fixed = 0
    for y, x in tr.items():
        r = _run_at(dark[y], x, W, labrow=(labels[y] if labels is not None else None))
        if not r:
            continue
        a, b = r
        if (b - a + 1) <= spike_thr:
            continue                                   # обычная толщина — не выброс
        xo = other.get(y)
        if xo is None:
            continue                                   # без референса направление не определить
        far = a if abs(a - xo) > abs(b - xo) else b    # вершина = ДАЛЬНИЙ ОТ ПАРТНЁРА край
        if abs(far - xo) <= abs(x - xo) + 2:
            continue                                   # движение НЕ наружу — пропуск (2D-мост
                                                       # соединяет горб с телом: агностичная
                                                       # дотяжка утаскивала горб К ТЕЛУ, QC №5)
        ext = abs(far - x)
        if ext < max(2.5 * w0, 12.0):
            continue                                   # мелкие дотяжки вредят — цель = БОЛЬШИЕ горбы
        if abs(xo - far) < abs(x - far):               # выброс принадлежит партнёру
            continue
        vertex = far - w0 / 2 if far == b else far + w0 / 2
        if ext <= max_ext:
            target = vertex                            # вершина в пределах шага — сразу к ней
        else:                                          # ПОШАГОВО к вершине (горб 3181: вершина в
            step_dir = 1.0 if far == b else -1.0       # ~200px, разовый прыжок был отрезан max_ext;
            target = x + step_dir * max_ext            # prob-гейт проверяет каждый шаг пути)
        if probc is not None:
            ti = int(round(target))
            lo, hi = max(0, ti - 2), min(W, ti + 3)
            if not hi > lo or float(probc[y, lo:hi].max()) < prob_min:
                continue                               # НИ ОДИН канал кривую не видит — клякса/чужое
                                                       # (гейт по combined: на зигзагах идентичность
                                                       # каналов путается, но тушь кривой модель видит)
        out[y] = float(target)
        fixed += 1
    return out, fixed


def smooth_pass(tr, tol=3.0, alpha=0.5, max_gap=6):
    """Мягкое сглаживание дрожи: тяга к медиане 3 соседних строк, только где |Δ| < tol."""
    ys = sorted(tr)
    out = dict(tr)
    for i in range(1, len(ys) - 1):
        y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
        if y1 - y0 > max_gap or y2 - y1 > max_gap:
            continue
        med = float(np.median([tr[y0], tr[y1], tr[y2]]))
        if abs(tr[y1] - med) < tol:
            out[y1] = (1 - alpha) * tr[y1] + alpha * med
    return out


def fill_from_partner(tr, partner, probc, win=12, thr_lo=0.12, edge=8):
    """Заполнение дыр трассы от ПАРТНЁРА в зонах слипания (QC №4: после 3500м кривые
    сплетаются в один жгут — канал MGZ не пробивает порог 0.4 и строки выпадают, MGZ
    «игнорирует»). Домен: в зоне наложения обе кривые идут по одной туши. Строка есть
    у партнёра, нет у нас → пик СВОЕГО канала в ±win от партнёра (≥thr_lo), иначе —
    позиция партнёра как есть."""
    W = probc.shape[1]
    out = dict(tr)
    filled = 0
    for y, xp in partner.items():
        if y in out:
            continue
        xi = int(round(xp))
        lo, hi = max(0, xi - win), min(W, xi + win + 1)
        if hi <= lo:
            continue
        seg = probc[y, lo:hi]
        if seg.max() >= thr_lo:
            a = int(np.argmax(seg))
            l2, h2 = max(0, a - edge), min(len(seg), a + edge + 1)
            w = seg[l2:h2]
            out[y] = float(((np.arange(l2, h2) + lo) * w).sum() / max(w.sum(), 1e-6))
        else:
            out[y] = float(xp)                          # канал молчит — слипание, идём по партнёру
        filled += 1
    return out, filled


def bridge_gaps(tr, prob_comb, max_gap=40, snap_win=10, thr=0.08):
    """Мост дыр, где трассы НЕТ (узел 3193: 30 строк без ОБЕИХ трасс — fill от партнёра
    бессилен): линейная интерполяция между концами + подтяжка к пику combined prob ±snap_win.
    Большие разрывы (>max_gap) не мостим — могут быть законными (домен: отрывы/перевыносы)."""
    ys = sorted(tr)
    out = dict(tr)
    W = prob_comb.shape[1]
    n = 0
    for a, b in zip(ys, ys[1:]):
        if not (1 < b - a <= max_gap):
            continue
        for y in range(a + 1, b):
            xi = tr[a] + (tr[b] - tr[a]) * (y - a) / (b - a)
            lo, hi = max(0, int(xi) - snap_win), min(W, int(xi) + snap_win + 1)
            seg = prob_comb[y, lo:hi]
            if seg.size and seg.max() >= thr:
                j = int(np.argmax(seg))
                l2, h2 = max(0, j - 6), min(len(seg), j + 7)
                w = seg[l2:h2]
                xi = float(((np.arange(l2, h2) + lo) * w).sum() / max(w.sum(), 1e-6))
            out[y] = float(xi); n += 1
    return out, n


def refine_traces(rgb, mgz, mpz, iters=3, dark_thr=110, prob=None):
    """Главный вход: native rgb + трассы (+prob 2×H×W native) → дотянутые/сглаженные + stats."""
    dark = rgb.max(2) < dark_thr
    import cv2
    _, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    m, p = dict(mgz), dict(mpz)
    w0m, w0p = _stroke_width(dark, m), _stroke_width(dark, p)
    p0 = prob[0] if prob is not None else None
    p1 = prob[1] if prob is not None else None
    pc = np.maximum(p0, p1) if prob is not None else None
    fill_m = fill_p = br_m = br_p = 0
    if p0 is not None:                                  # дыры в зонах слипания — ДО дотяжки
        m, fill_m = fill_from_partner(m, p, p0)
        p, fill_p = fill_from_partner(p, m, p1)
        m, br_m = bridge_gaps(m, pc)                    # дыры БЕЗ обеих трасс (зигзаг-узлы)
        p, br_p = bridge_gaps(p, pc)
    total_m = total_p = 0
    for _ in range(iters):
        # гейт дотяжки по COMBINED prob: на зигзагах канальная идентичность путается,
        # но тушь кривой модель видит; кляксы давит в обоих каналах
        m, fm = refine_pass(dark, m, p, w0m, probc=pc, prob_min=0.08, labels=labels)
        p, fp = refine_pass(dark, p, m, w0p, probc=pc, prob_min=0.08, labels=labels)
        m, p = smooth_pass(m), smooth_pass(p)
        total_m += fm; total_p += fp
        if fm + fp == 0:
            break
    for _ in range(2):                                  # финальная гладкость (QC №6: «немного
        m = smooth_pass(m, tol=2.5)                     # сгладить после оцифровки») — только
        p = smooth_pass(p, tol=2.5)                     # дрожь <2.5px, пики не трогаем
    return m, p, {"w0_mgz": round(w0m, 1), "w0_mpz": round(w0p, 1),
                  "ext_mgz": total_m, "ext_mpz": total_p,
                  "fill_mgz": fill_m + br_m, "fill_mpz": fill_p + br_p}
