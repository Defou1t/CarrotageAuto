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
