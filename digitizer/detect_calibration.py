"""
detect_calibration.py — детект геометрии планшета из ИЗОБРАЖЕНИЯ (без nlgx):
  • горизонтальные линии depth-сетки (жирные 4 м) -> depth<->y;
  • вертикальные края трека / деления шкалы -> value<->x.
Валидируется против nlgx ground-truth (extract_nlgx): точные Y сетки (35596),
x_left/x_right Scale Axis, top_y/bottom_y Depth Axis.

python detect_calibration.py <nlgx> <image>      # детект + сверка с nlgx
"""
import sys
import numpy as np
from PIL import Image
from extract_nlgx import extract
Image.MAX_IMAGE_PIXELS = None


def _profile_peaks(score, min_dist, prominence):
    """Простой поиск локальных максимумов с мин. расстоянием и prominence."""
    peaks = []
    n = len(score)
    i = 1
    while i < n-1:
        if score[i] >= score[i-1] and score[i] > score[i+1] and score[i] >= prominence:
            # уточнить как локальный максимум в окне min_dist
            lo = max(0, i-min_dist); hi = min(n, i+min_dist+1)
            if score[i] >= score[lo:hi].max():
                peaks.append(i)
                i += min_dist
                continue
        i += 1
    return np.array(peaks)


def detect_depth_grid(gray, thr=140):
    """
    Детект горизонтальных линий сетки. Возвращает (ys, spacing_fine, ys_bold).
    Идея: профиль числа тёмных пикселей по строкам -> пики = горизонтали.
    Жирные (4 м) — самые выраженные; шаг сетки — по автокорреляции профиля.
    """
    H, W = gray.shape
    # центральная x-полоса, чтобы горизонтали (через весь лист) набирали счёт,
    # а краевые подписи/шум не мешали
    x0, x1 = int(0.12*W), int(0.95*W)
    band = x1 - x0
    ink = (gray[:, x0:x1] < thr)
    row = ink.sum(axis=1).astype(np.float64)
    # вычесть медленный тренд (затемнение/пятна)
    k = 51
    base = np.convolve(row, np.ones(k)/k, mode="same")
    sig = np.clip(row - base, 0, None)

    # все горизонтали (тонкая сетка): пики с малым min_dist
    all_peaks = _profile_peaks(sig, min_dist=3, prominence=np.percentile(sig, 75))
    # шаг тонкой сетки = мода малых межпиковых интервалов
    fine = 12
    if len(all_peaks) > 5:
        diffs = np.diff(all_peaks)
        small = diffs[(diffs >= 4) & (diffs <= 40)]
        if len(small):
            fine = int(np.median(small))

    # ЖИРНЫЕ 4-м линии = длинные горизонтали (покрывают почти всю ширину) и
    # выраженные; ищем с min_dist ~ 5*fine, prominence высокий
    longline = (row >= 0.5 * band)           # линия через >=50% ширины
    sig_bold = sig * longline
    bold = _profile_peaks(sig_bold, min_dist=max(20, int(4.5*fine)),
                          prominence=np.percentile(sig_bold[sig_bold > 0], 50) if (sig_bold > 0).any() else 1)
    # шаг жирных
    bold_sp = float(np.median(np.diff(bold))) if len(bold) > 2 else None
    return all_peaks, fine, bold, bold_sp


def detect_bold_hgrid(gray, frac_lo=0.15, frac_hi=0.90):
    """ЖИРНЫЕ горизонтали (каждые 10 клеток): цепочка с шагом ~10×fine по МАГНИТУДЕ отклика.
    Снап «к ближайшей из всех» цеплял тонкие соседки (~12px), когда изгиб уводит жирную
    дальше полушага тонкой (QC эксперта 03.07: сетка мимо жирных). Здесь жирные выбираются
    явно: seed = сильнейший пик; цепочка вверх/вниз с окном ±0.25 шага, в окне — линия с max
    магнитудой; нет линии — шаг экстраполяцией (стёртая/бледная). Возвращает
    (ys_bold, step_px, miss_frac)."""
    H, W = gray.shape
    x0, x1 = int(frac_lo * W), int(frac_hi * W)
    band = max(1, x1 - x0)
    bg = float(np.median(gray))
    prof = (gray[:, x0:x1] < bg - 22).sum(1).astype(np.float64) / band
    prof -= np.convolve(prof, np.ones(31) / 31, mode="same")
    prof = np.clip(prof, 0, None)
    ys, mag = [], []
    i = 1
    while i < H - 1:
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.02:
            a, b, c = prof[i - 1], prof[i], prof[i + 1]
            off = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
            ys.append(i + max(-1.0, min(1.0, off))); mag.append(float(b))
            i += 4
        else:
            i += 1
    ys = np.array(ys); mag = np.array(mag)
    if len(ys) < 5:
        return ys, 0.0, 1.0
    fine = float(np.median(np.diff(np.sort(ys))))
    step = 10.0 * fine                                   # жирная каждые 10 клеток
    seed = int(np.argmax(mag))
    win = 0.25 * step
    bold = [float(ys[seed])]
    miss = 0; total = 0
    for direction in (+1, -1):
        cur = float(ys[seed])
        while True:
            pred = cur + direction * step
            if pred < ys.min() - win or pred > ys.max() + win:
                break
            total += 1
            sel = np.nonzero(np.abs(ys - pred) <= win)[0]
            if len(sel):
                j = sel[int(np.argmax(mag[sel]))]        # в окне — самая жирная
                cur = float(ys[j])
            else:
                cur = pred; miss += 1                    # экстраполяция
            bold.append(cur)
    bold = np.array(sorted(bold))
    return bold, step, (miss / max(1, total))


def detect_bold_hgrid_2band(gray, left=(0.04, 0.35), right=(0.60, 0.92)):
    """Жирные горизонтали С НАКЛОНОМ: detect_bold_hgrid отдельно в ЛЕВОЙ и ПРАВОЙ x-полосах,
    пары по ближайшему y (в пределах 0.3 шага). QC эксперта 03.07: один профиль на всю ширину
    даёт «средний» y — при наклоне линии оба конца врут на полнаклона, правый край уплывает.
    Возвращает (pairs [(yL,yR)...], xLc, xRc, step_px): y на ЦЕНТРАХ полос — наклон per-line
    экстраполируется на любой x."""
    H, W = gray.shape
    bl, sl, ml = detect_bold_hgrid(gray, frac_lo=left[0], frac_hi=left[1])
    br, sr, mr = detect_bold_hgrid(gray, frac_lo=right[0], frac_hi=right[1])
    step = sl or sr
    if not len(bl) or not len(br) or not step:
        return [], 0, 0, step or 0.0
    xLc = 0.5 * (left[0] + left[1]) * W
    xRc = 0.5 * (right[0] + right[1]) * W
    pairs = []
    for yl in bl:
        j = int(np.abs(br - yl).argmin())
        if abs(br[j] - yl) <= 0.3 * step:
            pairs.append((float(yl), float(br[j])))
    return pairs, xLc, xRc, float(step)


def detect_bold_hgrid_multiband(gray, bands=((0.08, 0.20), (0.20, 0.34), (0.36, 0.58),
                                             (0.60, 0.74), (0.76, 0.90))):
    """Жирные горизонтали, робастно к РУКОПИСНЫМ ПОДПИСЯМ глубин (QC №4: подпись «3180» над
    линией в левой полосе дала ложный пик — линия уехала вверх и испортила наклон пары).
    Опора = цепочка в ЦЕНТРЕ трека (подписей нет), затем y каждой линии меряется в 5 полосах
    (локальный пик профиля полосы в ±0.15 шага от опоры) и фитится прямой y(x) по медианному
    наклону с выбросом аутлаеров >5px (подпись портит максимум одну полосу).
    Возвращает (lines [(y_c, slope, x_c)...], step_px): y_c на центре трека x_c."""
    H, W = gray.shape
    bg = float(np.median(gray))
    profs = []
    for lo, hi in bands:
        x0, x1 = int(lo * W), int(hi * W)
        p = (gray[:, x0:x1] < bg - 22).sum(1).astype(np.float64) / max(1, x1 - x0)
        p -= np.convolve(p, np.ones(31) / 31, mode="same")
        profs.append((np.clip(p, 0, None), 0.5 * (x0 + x1)))
    ci = len(bands) // 2                                 # центральная полоса — опора
    lo, hi = bands[ci]
    anchor, step, _ = detect_bold_hgrid(gray, frac_lo=lo, frac_hi=hi)
    if not len(anchor) or not step:
        return [], 0.0
    x_c = profs[ci][1]
    win = 0.15 * step
    lines = []
    for ya in anchor:
        pts = []
        for prof, xc in profs:
            a, b = max(1, int(ya - win)), min(H - 1, int(ya + win))
            if b <= a:
                continue
            j = a + int(np.argmax(prof[a:b]))
            if prof[j] > 0.02:
                pts.append((xc, float(j)))
        if len(pts) < 3:
            lines.append((float(ya), None, x_c)); continue
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        sl = np.median([(ys[j] - ys[i]) / (xs[j] - xs[i])
                        for i in range(len(pts)) for j in range(i + 1, len(pts))])
        y0 = float(np.median(ys - sl * (xs - x_c)))      # y на центре по всем полосам
        good = np.abs(ys - (y0 + sl * (xs - x_c))) <= 5.0
        if good.sum() >= 3 and good.sum() < len(pts):    # рефит без аутлаеров (подписи)
            xs, ys = xs[good], ys[good]
            sl = np.median([(ys[j] - ys[i]) / (xs[j] - xs[i])
                            for i in range(len(xs)) for j in range(i + 1, len(xs))])
            y0 = float(np.median(ys - sl * (xs - x_c)))
        lines.append((y0, float(sl), x_c))
    return lines, float(step)


def _band_profile(gray, x0, x1, exclude_x=None):
    """Профиль покрытия строк тёмным в полосе [x0,x1) с ИСКЛЮЧЕНИЕМ зоны кривых
    (QC №5: почти-горизонтальные сегменты пиков кривой дают ложные пики профиля —
    наклоны 3172/3188 уехали до −17/+26 px/1000)."""
    cols = np.ones(gray.shape[1], bool)
    cols[:x0] = False; cols[x1:] = False
    if exclude_x:
        cols[max(0, exclude_x[0]):exclude_x[1]] = False
    n = int(cols.sum())
    if n < 30:
        return None
    bg = float(np.median(gray))
    p = (gray[:, cols] < bg - 22).sum(1).astype(np.float64) / n
    p -= np.convolve(p, np.ones(31) / 31, mode="same")
    return np.clip(p, 0, None)


def fit_bold_grid(gray, ys_pred, xs_axis, step_px, exclude_x=None,
                  bands=((0.08, 0.20), (0.20, 0.34), (0.36, 0.58), (0.60, 0.74), (0.76, 0.90)),
                  win_frac=0.35, clus_tol=6.0, min_bands=3):
    """ГОЛОСОВАНИЕ ПОЛОС ПО СЛОТАМ (QC №5, замена цепочке): для каждой предсказанной 4м-линии
    кандидаты = локальные пики профилей 5 полос в окне ±win_frac·шаг; кластеры по y (±clus_tol);
    побеждает кластер ≥min_bands полос с max Σмагнитуд (жирная бьёт тонкую массой; подпись/
    кривая живут в 1-2 полосах и проигрывают всегда). y и наклон каждой линии — медианный фит
    по кластеру, экстраполяция на ось. Возвращает (ys_axis, slopes, matched_flags)."""
    H, W = gray.shape
    profs = []
    for lo, hi in bands:
        p = _band_profile(gray, int(lo * W), int(hi * W), exclude_x)
        if p is not None:
            profs.append((p, 0.5 * (int(lo * W) + int(hi * W))))
    win = win_frac * step_px
    ys_out, sl_out, ok_out = [], [], []
    prev = -1e18
    raw = []
    for k, yp in enumerate(ys_pred):
        cands = []                                    # (xc, y, mag)
        for p, xc in profs:
            a, b = max(1, int(yp - win)), min(H - 2, int(yp + win))
            if b - a < 3:
                continue
            seg = p[a:b]
            for j in range(1, len(seg) - 1):          # все локальные пики окна (не только max)
                if seg[j] >= seg[j - 1] and seg[j] > seg[j + 1] and seg[j] > 0.02:
                    cands.append((xc, float(a + j), float(seg[j])))
        best = None                                   # кластеризация по y
        for _, yc, _ in cands:
            cl = [c for c in cands if abs(c[1] - yc) <= clus_tol]
            bset = {c[0] for c in cl}
            if len(bset) < min_bands:
                continue
            mass = sum(c[2] for c in cl)
            if best is None or mass > best[0]:
                best = (mass, cl)
        if best is None:
            raw.append(None); continue
        cl = best[1]
        xs = np.array([c[0] for c in cl]); ys = np.array([c[1] for c in cl])
        sl = float(np.median([(ys[j] - ys[i]) / (xs[j] - xs[i])
                              for i in range(len(cl)) for j in range(i + 1, len(cl))
                              if xs[j] != xs[i]])) if len(cl) > 1 else 0.0
        y_ax = float(np.median(ys - sl * (xs - xs_axis[k])))
        raw.append((y_ax, sl))
    med_sl = float(np.median([r[1] for r in raw if r])) if any(raw) else 0.0
    for k, yp in enumerate(ys_pred):
        r = raw[k]
        if r and r[0] > prev + 0.5 * step_px:         # монотонность
            ys_out.append(r[0]); sl_out.append(r[1]); ok_out.append(True)
            prev = r[0]
        else:
            ys_out.append(float(yp)); sl_out.append(med_sl); ok_out.append(False)
            prev = max(prev, float(yp))
    return ys_out, sl_out, ok_out


def fit_bold_ladder(gray, step_px, y_top, y_bot, exclude_x=None, bands_n=5):
    """ЖИРНЫЕ = ЛЕСТНИЦА ПО СЫРЫМ ЛИНИЯМ (гейт vs эталон 03.07): позиции из detect_hgrid
    (эксперт покрыт med 1.4px — точнее любого профильного пере-детекта), ЖИРНОСТЬ мерится
    напрямую (интегральная темнота строки±1 вне зоны кривых), лестница выбирается DP:
    подпоследовательность линий с шагом ~step_px и max Σ жирности (лестница тонких имеет
    тот же шаг — побеждается жирностью; подписи/кривые в жирность почти не вносят).
    Наклон per-line: локальные пики узких полос ±6px вокруг выбранной линии.
    Возвращает [(y_на_x0, slope, matched)]."""
    H, W = gray.shape
    ys_all, fine = detect_hgrid(gray)
    if fine and 6 <= fine <= 40:                       # жирная = каждые 10 клеток; шаг ОТ ПЛАНШЕТА
        step10 = 10.0 * fine                           # (масштабонезависимо: 1:500 жирная = 10м,
        if not step_px or abs(step10 - step_px) / step10 > 0.25:   # шаг «4м из рамки» там врал 2.5×)
            step_px = step10
    ys_all = np.asarray([y for y in ys_all if y_top - step_px < y < y_bot + step_px], float)
    if len(ys_all) < 5:
        return []
    cols = np.ones(W, bool)
    cols[:int(0.04 * W)] = False; cols[int(0.92 * W):] = False
    if exclude_x:
        lo, hi = max(0, exclude_x[0]), min(W, exclude_x[1])
        if (hi - lo) < 0.5 * W:                        # BKZ-фикс: кривые на всю ширину — не исключаем
            cols[lo:hi] = False
    bg = float(np.median(gray))
    dark = np.clip(bg - gray[:, cols].astype(np.float64), 0, 60)
    B = np.array([dark[max(0, int(y) - 1):int(y) + 2].mean() for y in ys_all])
    # DP: лестница с шагом [0.82..1.18]*step и max Σ жирности (допускаем пропуск 1 ступени)
    n = len(ys_all)
    dp = B.copy(); prev = np.full(n, -1)
    for i in range(n):
        for span, pen in ((1.0, 0.0), (2.0, 0.3)):     # шаг или пропуск ступени
            lo, hi = ys_all[i] - span * step_px * 1.18, ys_all[i] - span * step_px * 0.82
            js = np.nonzero((ys_all >= lo) & (ys_all <= hi))[0]
            if len(js):
                j = js[np.argmax(dp[js])]
                cand = dp[j] + B[i] - pen * B.mean()
                if cand > dp[i]:
                    dp[i] = cand; prev[i] = j
    i = int(np.argmax(dp))
    chain_idx = []
    while i >= 0:
        chain_idx.append(i); i = prev[i]
    chain_idx = chain_idx[::-1]
    # наклон per-line: локальные пики узких полос вокруг выбранной линии
    bands = [(k / bands_n * 0.84 + 0.06, (k + 1) / bands_n * 0.84 + 0.06) for k in range(bands_n)]
    profs = []
    for lo, hi in bands:
        p = _band_profile(gray, int(lo * W), int(hi * W), exclude_x if exclude_x and
                          (exclude_x[1] - exclude_x[0]) < 0.5 * W else None)
        if p is not None:
            profs.append((p, 0.5 * (int(lo * W) + int(hi * W))))
    out = []
    sls = []
    for ci in chain_idx:
        ya = float(ys_all[ci])
        pts = []
        for p, xc in profs:
            a, b = max(1, int(ya - 6)), min(H - 2, int(ya + 7))
            if b - a < 3:
                continue
            seg = p[a:b]
            j = int(np.argmax(seg))
            if seg[j] > 0.02:
                pts.append((xc, a + j))
        if len(pts) >= 3:
            xs = np.array([q[0] for q in pts], float); yy = np.array([q[1] for q in pts], float)
            sl = float(np.median([(yy[j] - yy[i2]) / (xs[j] - xs[i2])
                                  for i2 in range(len(pts)) for j in range(i2 + 1, len(pts))
                                  if xs[j] != xs[i2]]))
            good = np.abs(yy - (ya + sl * (xs - np.median(xs)))) <= 5
            if good.sum() >= 3:
                xs2, yy2 = xs[good], yy[good]
                sl = float(np.median([(yy2[j] - yy2[i2]) / (xs2[j] - xs2[i2])
                                      for i2 in range(len(xs2)) for j in range(i2 + 1, len(xs2))
                                      if xs2[j] != xs2[i2]]))
        else:
            sl = None
        sls.append(sl)
    med_sl = float(np.median([s for s in sls if s is not None])) if any(s is not None for s in sls) else 0.0
    for ci, sl in zip(chain_idx, sls):
        ya = float(ys_all[ci])
        s = sl if sl is not None else med_sl
        # ya измерен «в среднем по ширине» (детект по полосе 0.15-0.9W) → к x=0 через центр
        out.append((ya - s * 0.5 * W, s, True))
    # мостим пропуски ступеней интерполяцией
    filled = []
    for (a_, b_) in zip(out, out[1:]):
        filled.append(a_)
        gapn = round((b_[0] - a_[0]) / step_px)
        for k in range(1, int(gapn)):
            t = k / gapn
            filled.append((a_[0] + (b_[0] - a_[0]) * t, med_sl, False))
    if out:
        filled.append(out[-1])
    return filled


def fit_bold_chain(gray, step_px, y_top, y_bot, exclude_x=None,
                   bands=((0.08, 0.20), (0.20, 0.34), (0.36, 0.58), (0.60, 0.74), (0.76, 0.90)),
                   clus_tol=6.0, min_bands=3):
    """Жирные горизонтали со СВОБОДНОЙ ФАЗОЙ (гейт vs эталон 03.07: фаза жирных задаётся
    ПЛАНШЕТОМ, а не «глубинами кратными шагу» — привязка к слотам давала промах в полшага
    (61px) там, где фаза рамки не совпадает с жирными; у эксперта линии на реальных жирных).
    Цепочка: старт = сильнейший кластер голосования полос во всём [y_top..y_bot], затем шаги
    ±step_px с окном ±0.3 шага и тем же голосованием (≥min_bands полос, max Σмагнитуд);
    пропуск (бледная) — экстраполяция и ход дальше. Возвращает [(y_на_x0, slope, matched)].
    y отдаётся на x=0 (пересчёт на любой якорь: y + slope*x)."""
    H, W = gray.shape
    profs = []
    for lo, hi in bands:
        p = _band_profile(gray, int(lo * W), int(hi * W), exclude_x)
        if p is not None:
            profs.append((p, 0.5 * (int(lo * W) + int(hi * W))))
    if not profs:
        return []

    def vote(yc, win):
        cands = []
        for p, xc in profs:
            a, b = max(1, int(yc - win)), min(H - 2, int(yc + win))
            if b - a < 3:
                continue
            seg = p[a:b]
            for j in range(1, len(seg) - 1):
                if seg[j] >= seg[j - 1] and seg[j] > seg[j + 1] and seg[j] > 0.02:
                    cands.append((xc, float(a + j), float(seg[j])))
        best = None
        for _, yy, _ in cands:
            cl = [c for c in cands if abs(c[1] - yy) <= clus_tol]
            if len({c[0] for c in cl}) < min_bands:
                continue
            mass = sum(c[2] for c in cl)
            if best is None or mass > best[0]:
                best = (mass, cl)
        if best is None:
            return None
        cl = best[1]
        xs = np.array([c[0] for c in cl]); ys = np.array([c[1] for c in cl])
        sl = float(np.median([(ys[j] - ys[i]) / (xs[j] - xs[i])
                              for i in range(len(cl)) for j in range(i + 1, len(cl))
                              if xs[j] != xs[i]])) if len(cl) > 1 else 0.0
        y0 = float(np.median(ys - sl * xs))            # y на x=0
        xc0 = float(np.median(xs))
        return y0, sl, best[0], xc0

    # старт: сильнейший кластер по всему диапазону (скан крупными окнами)
    seed = None
    for yc in np.arange(y_top + step_px * 0.5, y_bot - step_px * 0.5, step_px * 0.5):
        v = vote(yc, step_px * 0.45)
        if v and (seed is None or v[2] > seed[2]):
            seed = v
    if seed is None:
        return []
    y0s, sls, _, xc0 = seed
    lines = {}
    for direction in (+1, -1):                         # цепочка вверх и вниз от старта
        y_prev = y0s + sls * xc0                       # в координате центра детекта
        sl_prev = sls
        k = 0
        while True:
            k += 1
            yc = y_prev + direction * step_px
            if not (y_top - step_px * 0.4 <= yc <= y_bot + step_px * 0.4):
                break
            v = vote(yc, step_px * 0.3)
            if v:
                y0, sl, _, xcv = v
                yk = y0 + sl * xcv
                lines[round(yk, 1)] = (y0, sl, True)
                y_prev, sl_prev = yk, sl
            else:                                      # бледная — экстраполяция, идём дальше
                y0 = yc - sl_prev * xc0
                lines[round(yc, 1)] = (y0, sl_prev, False)
                y_prev = yc
    lines[round(y0s + sls * xc0, 1)] = (y0s, sls, True)
    return [lines[k] for k in sorted(lines)]


def detect_hgrid(gray, frac_lo=0.15, frac_hi=0.90, min_cover=0.02):
    """Субпиксельный детект ВСЕХ горизонтальных линий сетки (тонких + жирных).
    Профиль покрытия строки чернилами в центральной x-полосе при АДАПТИВНОМ пороге
    (медиана-фона − 22: ловит и светлую сетку ≈170-190, и тёмную). Детренд + локальные
    максимумы + параболическое уточнение субпикселя. Возвращает (ys_subpx, fine_spacing).
    Применение — A2b: снап 4 м-линий Depth Grid к РЕАЛЬНЫМ линиям (поглощает неравномерность
    бумаги, угол уже учтён A2). Жирные 4 м отдельно НЕ выделяем — снап к ближайшей в узком
    допуске не требует различать жирную/тонкую."""
    H, W = gray.shape
    x0, x1 = int(frac_lo * W), int(frac_hi * W)
    band = max(1, x1 - x0)
    bg = float(np.median(gray))
    prof = (gray[:, x0:x1] < bg - 22).sum(1).astype(np.float64) / band
    prof -= np.convolve(prof, np.ones(31) / 31, mode="same")
    prof = np.clip(prof, 0, None)
    ys = []
    i = 1
    while i < H - 1:
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > min_cover:
            a, b, c = prof[i - 1], prof[i], prof[i + 1]
            off = 0.5 * (a - c) / (a - 2 * b + c + 1e-9)   # вершина параболы
            ys.append(i + max(-1.0, min(1.0, off)))
            i += 4
        else:
            i += 1
    ys = np.array(ys)
    fine = float(np.median(np.diff(np.sort(ys)))) if len(ys) > 5 else 12.0
    return ys, fine


def detect_track_edges(gray, thr=140):
    """Вертикальные края трека: профиль тёмных пикселей по столбцам -> пики."""
    H, W = gray.shape
    y0, y1 = int(0.2*H), int(0.8*H)
    ink = (gray[y0:y1, :] < thr)
    col = ink.sum(axis=0).astype(np.float64)
    k = 31; base = np.convolve(col, np.ones(k)/k, mode="same")
    sig = np.clip(col - base, 0, None)
    peaks = _profile_peaks(sig, min_dist=15, prominence=np.percentile(sig, 90))
    return peaks, sig


def validate(nlgx, image):
    m = extract(nlgx)
    g = np.asarray(Image.open(image).convert("L"))
    H, W = g.shape
    da = m["depth_axis"]; dg = m.get("depth_grid", {})
    true_ys = np.array(dg.get("ys", []))
    print(f"image {W}x{H}; nlgx depth axis y {da['top_y']}..{da['bottom_y']}, "
          f"grid lines={len(true_ys)} spacing~{(true_ys[-1]-true_ys[0])/(len(true_ys)-1):.1f}" if len(true_ys)>1 else "")

    peaks, fine, bold, bold_sp = detect_depth_grid(g)
    print(f"\nDEPTH GRID detect: {len(peaks)} h-lines, fine-spacing={fine}px, "
          f"bold(4m)={len(bold)} bold_spacing={bold_sp}")
    if len(true_ys) > 1:
        true_sp = (true_ys[-1]-true_ys[0])/(len(true_ys)-1)
        # истинные линии -> ближайшая ЖИРНАЯ детектированная (это и есть 4-м сетка)
        for label, det in [("all-peaks", peaks), ("bold", bold)]:
            if len(det):
                d = np.abs(true_ys[:, None] - det[None, :]).min(axis=1)
                print(f"  true grid -> nearest {label:9}: median={np.median(d):.1f}px "
                      f"within2={np.mean(d<=2)*100:.0f}% within5={np.mean(d<=5)*100:.0f}% (n_det={len(det)})")
        print(f"  true 4m spacing={true_sp:.2f}px ; detected bold_spacing={bold_sp}")

    edges, esig = detect_track_edges(g)
    print(f"\nTRACK EDGES detect: {len(edges)} v-lines at x={list(edges[:12])}")
    xs_true = sorted({s["x_left"] for s in m["scale_axes"]} | {s["x_right"] for s in m["scale_axes"]})
    print(f"  nlgx scale x_left/x_right set: {xs_true}")
    if len(edges):
        for xt in xs_true:
            dd = np.abs(edges - xt).min()
            print(f"    x={xt}: nearest detected edge {edges[np.abs(edges-xt).argmin()]} (|d|={dd}px)")


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
