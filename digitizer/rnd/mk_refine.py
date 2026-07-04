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


def _vert_thin(dark, y, x, w0, k=3.0):
    """Вертикальная толщина туши в колонке x (макс. по x±1): лепесток-пик тонкий
    (1-3 строки), клякса/пятно — толстые. Порог k*w0."""
    H = dark.shape[0]
    lim = int(k * w0) + 1
    best = lim + 1
    for xx in (int(round(x)) - 1, int(round(x)), int(round(x)) + 1):
        if not (0 <= xx < dark.shape[1]) or not dark[y, xx]:
            continue
        u = d = 0
        while u < lim and y - u - 1 >= 0 and dark[y - u - 1, xx]:
            u += 1
        while d < lim and y + d + 1 < H and dark[y + d + 1, xx]:
            d += 1
        best = min(best, u + d + 1)
    return best <= lim


def reseat_pass(dark, tr, w0, labels=None, search=None):
    """Пере-посадка строк, где трасса стоит на БЕЛОМ (сглаживание/fill в жгуте
    уводит её в зазор между прядями — диагностика 04.07: 41/70 плохих кликов низа
    Yatskivska = no_run_at_our_x, вся построчная дотяжка на них отключалась).
    Строка мимо чернил ±2px → центр ближайшего тёмного рана в ±search px; если
    рана нет — строку не трогаем (законный разрыв)."""
    W = dark.shape[1]
    if search is None:
        search = max(int(1.5 * w0), 6)
    out = dict(tr)
    n = 0
    for y, x in tr.items():
        xi = int(round(x))
        if not (0 <= xi < W):
            continue
        if dark[y, max(0, xi - 2):min(W, xi + 3)].any():
            continue                                   # уже на чернилах
        best = None
        for d in range(3, search + 1):
            for xx in (xi - d, xi + d):
                if 0 <= xx < W and dark[y, xx]:
                    best = xx; break
            if best is not None:
                break
        if best is None:
            continue
        r = _run_at(dark[y], best, W, labrow=(labels[y] if labels is not None else None))
        if r:
            out[y] = (r[0] + r[1]) / 2.0
            n += 1
    return out, n


def refine_pass(dark, tr, other, w0, max_ext=90, probc=None, prob_min=0.10, labels=None,
                petal=False):
    """Один пасс дотяжки. Возврат (новая трасса, число дотянутых строк).
    probc (канал модели, native): тянуть только куда модель видит СВОЮ кривую —
    кляксы/пятна она подавляет (LEVEN-регресс без гейта: MPZ 17.6→21px, гейт 03.07),
    реальные горбы — нет. Модель слепа на ВЕРШИНАХ тонких лепестков (клики эксперта
    сидят именно там, низ Yatskivska) → альтернативный допуск 04.07: цель в ТОЙ ЖЕ
    2D-компоненте (ран уже смощён hop'ом по labels) и вертикально ТОНКАЯ (_vert_thin)
    — лепесток, не клякса."""
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
            prob_ok = hi > lo and float(probc[y, lo:hi].max()) >= prob_min
            if not prob_ok and not (petal and _vert_thin(dark, y, far, w0)):
                continue                               # модель кривую не видит И это не тонкий
                                                       # лепесток (same-CC гарантирован hop'ом по
                                                       # labels) — клякса/чужое
        out[y] = float(target)
        fixed += 1
    return out, fixed


def recenter_pass(dark, tr, w0, labels=None, max_w_mult=2.2, max_shift=None):
    """ЧЕРНИЛЬНОЕ ПОСТ-ЦЕНТРИРОВАНИЕ (план 95-99%, этап C; реализовано 04.07):
    эксперт сидит на ЦЕНТРЕ штриха (ink_jitter: median 1.0px) — а наш peak центрирован
    по prob. Для каждой строки: ран под трассой обычной толщины (≤max_w_mult*w0 —
    не слипание и не пик-выброс) → x к центру рана. Широкие раны не трогаем (их
    ведёт дотяжка/партнёр-логика). max_shift страхует от прыжка на чужой ран."""
    W = dark.shape[1]
    if max_shift is None:
        max_shift = max(1.5 * w0, 6.0)
    out = dict(tr)
    n = 0
    for y, x in tr.items():
        r = _run_at(dark[y], x, W, labrow=(labels[y] if labels is not None else None))
        if not r:
            continue
        a, b = r
        if (b - a + 1) > max_w_mult * w0:
            continue
        c = (a + b) / 2.0
        if abs(c - x) < 0.6 or abs(c - x) > max_shift:
            continue
        out[y] = c
        n += 1
    return out, n


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


def refine_traces(rgb, mgz, mpz, iters=3, dark_thr=110, prob=None,
                  do_fill=True, do_bridge=True, do_reseat=True, do_ext=True,
                  petal=False, do_smooth=True, do_recenter=False):
    # Гейт-матрица vs эталон Yatskivska (04.07, mk_knots_iter): сырые peak 64.0% eff
    # (≤3px на кликах, дыры=fail) → fill 66.1 → +bridge 67.5 → +дотяжка 68.3 (ЛУЧШИЙ,
    # без petal/recenter). petal (дотяжка в тонкие лепестки без prob) — эффект 0;
    # recenter (центр рана) — РЕГРЕСС 68→57 (в жгуте ран = обе кривые, центр чужой);
    # smooth −0.9пп на кликах, но оставлен для сдачи (QC эксперта «сгладить»).
    """Главный вход: native rgb + трассы (+prob 2×H×W native) → дотянутые/сглаженные + stats.
    Переключатели стадий — для гейт-матрицы vs эталон (04.07)."""
    dark = rgb.max(2) < dark_thr
    import cv2
    _, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    m, p = dict(mgz), dict(mpz)
    w0m, w0p = _stroke_width(dark, m), _stroke_width(dark, p)
    p0 = prob[0] if prob is not None else None
    p1 = prob[1] if prob is not None else None
    pc = np.maximum(p0, p1) if prob is not None else None
    fill_m = fill_p = br_m = br_p = 0
    if p0 is not None and do_fill:                      # дыры в зонах слипания — ДО дотяжки
        m, fill_m = fill_from_partner(m, p, p0)
        p, fill_p = fill_from_partner(p, m, p1)
    if pc is not None and do_bridge:
        m, br_m = bridge_gaps(m, pc)                    # дыры БЕЗ обеих трасс (зигзаг-узлы)
        p, br_p = bridge_gaps(p, pc)
    rs_m = rs_p = 0
    if do_reseat:
        m, rs_m = reseat_pass(dark, m, w0m, labels=labels)
        p, rs_p = reseat_pass(dark, p, w0p, labels=labels)
    rc_m = rc_p = 0
    if do_recenter:                                     # чернильное пост-центрирование (этап C)
        m, rc_m = recenter_pass(dark, m, w0m, labels=labels)
        p, rc_p = recenter_pass(dark, p, w0p, labels=labels)
    total_m = total_p = 0
    for _ in range(iters if do_ext else 0):
        # гейт дотяжки по COMBINED prob: на зигзагах канальная идентичность путается,
        # но тушь кривой модель видит; кляксы давит в обоих каналах
        m, fm = refine_pass(dark, m, p, w0m, probc=pc, prob_min=0.08, labels=labels, petal=petal)
        p, fp = refine_pass(dark, p, m, w0p, probc=pc, prob_min=0.08, labels=labels, petal=petal)
        if do_smooth:
            m, p = smooth_pass(m), smooth_pass(p)
        if do_reseat:
            m, rm = reseat_pass(dark, m, w0m, labels=labels)  # сглаживание могло увести в зазор
            p, rp = reseat_pass(dark, p, w0p, labels=labels)
            rs_m += rm; rs_p += rp
        total_m += fm; total_p += fp
        if fm + fp == 0:
            break
    for _ in range(2 if do_smooth else 0):              # финальная гладкость (QC №6: «немного
        m = smooth_pass(m, tol=2.5)                     # сгладить после оцифровки») — только
        p = smooth_pass(p, tol=2.5)                     # дрожь <2.5px, пики не трогаем
    return m, p, {"w0_mgz": round(w0m, 1), "w0_mpz": round(w0p, 1),
                  "ext_mgz": total_m, "ext_mpz": total_p,
                  "fill_mgz": fill_m + br_m, "fill_mpz": fill_p + br_p,
                  "reseat_mgz": rs_m, "reseat_mpz": rs_p,
                  "recenter_mgz": rc_m, "recenter_mpz": rc_p}
