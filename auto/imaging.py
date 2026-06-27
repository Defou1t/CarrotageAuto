r"""
imaging.py — общий CV-слой «карты разделимости линий» (durable, PLAN §6.6.13 / ROADMAP §5).

ГЛАВНЫЙ урок: сетка СВЕТЛАЯ (V med ~166, период ~12px), чёрная кривая V<110 → передний план
по АБСОЛЮТНОЙ темноте, НЕ по «темнее локального фона» (adaptiveThreshold ловит и сетку).
Цветная кривая — по разнице каналов (красная SP: R−G>25; зелёная GZ слабее). Прямые структуры
(рамка/деления масштаба) — морфологией (длинное открытие vert/horiz: прямые выживают, волна нет).

Рецепт переднего плана кривых:
    ink = (тёмное | цветное) − морф.структура  [× prob-гейт recall-модели, если есть]

Чистый OpenCV/NumPy (без scipy/skimage). Функции не знают про nlgx — работают по голому RGB.
"""
import numpy as np
import cv2


def load_rgb(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path).convert("RGB"))


def value_channel(rgb):
    """V = max(R,G,B) — яркость; сетка светлая (высокий V), чёрная кривая низкий V."""
    return rgb.max(2)


def dark_mask(rgb, p):
    """Чёрные кривые: АБСОЛЮТНАЯ темнота V<dark_v (≈94-95% чёрной кривой, ~7% сетки)."""
    return (value_channel(rgb) < p.dark_v)


def color_channels(rgb, p):
    """Цветные штрихи по разнице каналов (исключая светлую сетку). dict bool-масок red/blue/green."""
    R, G, B = [rgb[..., i].astype(np.int16) for i in range(3)]
    mx = rgb.max(2).astype(np.int16); mn = rgb.min(2).astype(np.int16)
    sat = mx - mn
    colored = (sat >= p.sat_thr) & (mx < 245)            # не пере-светлое
    red = colored & (R - G > p.rg_thr) & (R >= B)
    green = colored & (G - R > p.rg_thr) & (G > B)
    blue = colored & (B - R > 8) & (B >= G - 4) & ~green
    return {"red": red, "green": green, "blue": blue}


def structure_mask(rgb, p, min_len=None):
    """Длинные ПРЯМЫЕ структуры (рамка трека, вертикальные оси, деления масштаба, жирная сетка).
    Морфология: бинарь тёмного → раздельное открытие вертикалью и горизонталью длинным ядром.
    Прямая линия (рамка/ось) выживает; волнистая кривая — нет. Возвращает bool-маску структуры."""
    H = rgb.shape[0]
    Lh = int(min_len or p.struct_open_len)                       # горизонталь: верх/низ-правила
    Lv = max(Lh, int(p.struct_vert_frac * H))                    # вертикаль: ТОЛЬКО полно-высотная
    dark = (value_channel(rgb) < p.grid_v_hi).astype(np.uint8)   #   грань — не зубцы пиковой кривой
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, Lv)))
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (Lh, 1)))
    s = (vert | horiz)
    return cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0


def ink_foreground(rgb, p, prob=None, prob_thr=0.5, drop_structure=True):
    """Передний план КРИВЫХ: (тёмное | цветное) − структура [| prob-recall].
    prob — карта recall-модели HxW float[0,1] (опц.): ДОБАВляет бледные линии, что правила теряют
    (recall-ретрейн, ROADMAP §6). Это UNION, НЕ гейт: на дармной пиковой кривой модель недокрывает
    пики (обучена на бледное), и пересечение fg&prob срезало бы 53% ясного чернила (замер MK, монтаж
    stages). Поэтому модель только ПОДНИМАЕТ recall; ясное чернило правил не режется. Структуру
    (сетку/рамку) модель не до-бавляет — обучена с негативами, плюс маска структуры вычитается и из неё."""
    fg = dark_mask(rgb, p)
    for m in color_channels(rgb, p).values():
        fg = fg | m
    struct = structure_mask(rgb, p) if drop_structure else None
    if drop_structure:
        fg = fg & ~struct
    if prob is not None:
        add = (prob >= prob_thr)
        fg = fg | (add & ~struct if drop_structure else add)
    return fg.astype(np.uint8)


def col_density(mask, x0=0, x1=None, smooth=1):
    """Профиль плотности по СТОЛБЦАМ (для счёта линий: пики = кривые). mask — bool/uint8."""
    x1 = mask.shape[1] if x1 is None else x1
    col = mask[:, x0:x1].astype(np.float64).sum(0)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        col = np.convolve(col, k, mode="same")
    return col


def row_density(mask, y0=0, y1=None, x0=0, x1=None):
    """Профиль по СТРОКАМ (для детекта сетки/верх-низ рамки)."""
    y1 = mask.shape[0] if y1 is None else y1
    x1 = mask.shape[1] if x1 is None else x1
    return mask[y0:y1, x0:x1].astype(np.float64).sum(1)


def find_peaks(score, min_dist, prominence):
    """Локальные максимумы score с мин. расстоянием и порогом prominence (без scipy)."""
    n = len(score)
    peaks = []
    i = 1
    while i < n - 1:
        if score[i] >= score[i - 1] and score[i] > score[i + 1] and score[i] >= prominence:
            lo = max(0, i - min_dist); hi = min(n, i + min_dist + 1)
            if score[i] >= score[lo:hi].max():
                peaks.append(i); i += min_dist; continue
        i += 1
    return np.array(peaks, dtype=int)


def find_valleys(score, peaks, ratio):
    """Между соседними пиками: есть ли долина глубиной < ratio×меньшего пика (= разнесённые линии).
    Возвращает индексы пиков, между которыми долина достаточно глубока для СПЛИТА."""
    splits = []
    for a, b in zip(peaks, peaks[1:]):
        if b - a < 2:
            continue
        valley = score[a:b].min()
        if valley < ratio * min(score[a], score[b]):
            splits.append((a, b, int(a + np.argmin(score[a:b]))))
    return splits


def row_runs(rowmask, gap=4):
    """Прогоны (раны) ненулевых пикселей в одной СТРОКЕ → list[(x0, x1, x_center)].
    Основа run-adjacency графа для 2D-обхода штриха (trace2d). gap мостит JPEG-разрывы."""
    xs = np.nonzero(rowmask)[0]
    if not len(xs):
        return []
    cuts = np.nonzero(np.diff(xs) > gap)[0] + 1
    out = []
    for seg in np.split(xs, cuts):
        out.append((int(seg[0]), int(seg[-1]), float(seg.mean())))
    return out
