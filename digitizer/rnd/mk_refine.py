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


def _run_at(row, x, W, gap=2, max_half=140):
    """Связный тёмный ран вокруг x (допуск gap светлых px). Возврат (a, b) или None."""
    xi = int(round(x))
    if not (0 <= xi < W) or not row[xi]:
        for d in (1, -1, 2, -2):                       # трасса могла встать на светлый зазор
            if 0 <= xi + d < W and row[xi + d]:
                xi += d; break
        else:
            return None
    a = xi
    g = 0
    while a - 1 >= max(0, xi - max_half):
        if row[a - 1]:
            a -= 1; g = 0
        elif g < gap:
            a -= 1; g += 1
        else:
            break
    a += g
    b = xi
    g = 0
    while b + 1 <= min(W - 1, xi + max_half):
        if row[b + 1]:
            b += 1; g = 0
        elif g < gap:
            b += 1; g += 1
        else:
            break
    b -= g
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


def refine_pass(dark, tr, other, w0, max_ext=90, probc=None, prob_min=0.10):
    """Один пасс дотяжки. Возврат (новая трасса, число дотянутых строк).
    probc (канал модели, native): тянуть только куда модель видит СВОЮ кривую —
    кляксы/пятна она подавляет (LEVEN-регресс без гейта: MPZ 17.6→21px, гейт 03.07),
    реальные горбы — нет."""
    W = dark.shape[1]
    out = dict(tr)
    spike_thr = max(3.0 * w0, w0 + 6.0)
    fixed = 0
    for y, x in tr.items():
        r = _run_at(dark[y], x, W)
        if not r:
            continue
        a, b = r
        if (b - a + 1) <= spike_thr:
            continue                                   # обычная толщина — не выброс
        ext_r, ext_l = b - x, x - a
        far, ext = (b, ext_r) if ext_r >= ext_l else (a, ext_l)
        if ext < max(2.5 * w0, 12.0) or ext > max_ext:
            continue                                   # мелкие дотяжки вредят — цель = БОЛЬШИЕ горбы
        xo = other.get(y)
        if xo is not None and a - 2 <= xo <= b + 2:    # общий ран: выброс — тому, кто ближе
            if abs(xo - far) < abs(x - far):
                continue
        target = far - w0 / 2 if far == b else far + w0 / 2
        if probc is not None:
            ti = int(round(target))
            lo, hi = max(0, ti - 2), min(W, ti + 3)
            if not hi > lo or float(probc[y, lo:hi].max()) < prob_min:
                continue                               # модель там кривую не видит — клякса/чужое
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


def refine_traces(rgb, mgz, mpz, iters=3, dark_thr=110, prob=None):
    """Главный вход: native rgb + трассы (+prob 2×H×W native) → дотянутые/сглаженные + stats."""
    dark = rgb.max(2) < dark_thr
    m, p = dict(mgz), dict(mpz)
    w0m, w0p = _stroke_width(dark, m), _stroke_width(dark, p)
    p0 = prob[0] if prob is not None else None
    p1 = prob[1] if prob is not None else None
    fill_m = fill_p = 0
    if p0 is not None:                                  # дыры в зонах слипания — ДО дотяжки
        m, fill_m = fill_from_partner(m, p, p0)
        p, fill_p = fill_from_partner(p, m, p1)
    total_m = total_p = 0
    for _ in range(iters):
        m, fm = refine_pass(dark, m, p, w0m, probc=p0)
        p, fp = refine_pass(dark, p, m, w0p, probc=p1)
        m, p = smooth_pass(m), smooth_pass(p)
        total_m += fm; total_p += fp
        if fm + fp == 0:
            break
    return m, p, {"w0_mgz": round(w0m, 1), "w0_mpz": round(w0p, 1),
                  "ext_mgz": total_m, "ext_mpz": total_p,
                  "fill_mgz": fill_m, "fill_mpz": fill_p}
