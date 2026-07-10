r"""
frame.py — U0: АВТОНОМНЫЙ детект рамки из ИЗОБРАЖЕНИЯ (без шаблона nlgx).
Отвечает на «где рамка начинается/заканчивается, чтобы НЕ путалась с линиями» (запрос заказчика).

Рамка = ПРЯМЫЕ структуры (вертикальные грани треков/осей + горизонтальные верх/низ + сетка),
отделимые от ВОЛНИСТЫХ кривых морфологией (imaging.structure_mask). Шаги:
  1. вертикальные грани → разбиение листа на КОЛОНКИ ТРЕКОВ;
  2. верх/низ рамки → вертикальный диапазон планшета;
  3. глубинная калибровка: глубины из ИМЕНИ файла (meta) + детект верх/низ-y → depth↔y;
  4. период горизонтальной сетки (для опц. Depth Grid).

Промоут идеи digitizer/detect_calibration.py (там был лишь ВАЛИДАТОР vs nlgx; здесь — СТРОИТЕЛЬ).
DURABLE: detect_calibration не генерализовал с Archive на Yatskivska (PLAN §6.6.9) — поэтому
опираемся на сильный приор ГЛУБИН ИЗ ИМЕНИ и возвращаем диагностику/уверенность, а не «слепой» детект.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
from . import imaging as im


@dataclass
class Track:
    x_left: int
    x_right: int
    index: int

    @property
    def width(self):
        return self.x_right - self.x_left


@dataclass
class Frame:
    img_w: int
    img_h: int
    top_y: int
    bottom_y: int
    tracks: list = field(default_factory=list)
    top_depth: Optional[float] = None
    bottom_depth: Optional[float] = None
    grid_period_px: Optional[float] = None
    grid_ys: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)        # диагностика/уверенность детекта

    @property
    def px_per_m(self):
        if (self.top_depth is None or self.bottom_depth is None
                or self.bottom_depth == self.top_depth):
            return None
        return (self.bottom_y - self.top_y) / (self.bottom_depth - self.top_depth)

    def depth_of(self, y):
        ppm = self.px_per_m
        return None if not ppm else self.top_depth + (y - self.top_y) / ppm


def _vertical_lines(rgb, p, min_h_frac=0.5):
    """X-позиции длинных вертикальных структур (грани рамки/оси). Открытие вертикалью + столбцовый
    профиль → пики. min_h_frac — вертикаль должна крыть ≥ долю высоты листа."""
    H, W = rgb.shape[:2]
    dark = (im.value_channel(rgb) < p.grid_v_hi).astype(np.uint8)
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(min_h_frac * H)))))
    col = vert.sum(0).astype(np.float64)
    thr = max(min_h_frac * H * 0.6, 1.0)
    peaks = im.find_peaks(col, min_dist=8, prominence=thr)
    return sorted(int(x) for x in peaks), col


def _tracks_from_verticals(rgb, p, xs, fg):
    """Колонки треков = промежутки между сильными вертикалями, содержащие достаточно ЧЕРНИЛА кривых.
    Отсекает узкие межосевые зазоры и поля. fg — передний план кривых (imaging.ink_foreground)."""
    H, W = rgb.shape[:2]
    bounds = [0] + xs + [W]
    cand = []
    min_w = max(40, int(0.04 * W))
    colden = fg.sum(0)
    for a, b in zip(bounds, bounds[1:]):
        if b - a < min_w:
            continue
        ink = float(colden[a:b].sum())
        cand.append((a, b, ink))
    if not cand:
        return [Track(int(0.06 * W), int(0.96 * W), 0)]    # фолбэк: весь лист как 1 трек
    # порог по чернилу: трек содержит кривые (не пустое поле между осями)
    inks = sorted(c[2] for c in cand)
    floor = inks[len(inks) // 2] * 0.25 + 1
    tracks = [Track(a + 1, b - 1, 0) for a, b, ink in cand if ink >= floor]
    if not tracks:
        a, b, _ = max(cand, key=lambda c: c[2]); tracks = [Track(a + 1, b - 1, 0)]
    for i, t in enumerate(sorted(tracks, key=lambda t: t.x_left)):
        t.index = i
    return sorted(tracks, key=lambda t: t.x_left)


def _frame_top_bottom(rgb, p, tracks):
    """Верх/низ рамки = крайние ДЛИННЫЕ горизонтальные структуры внутри ширины треков.
    Фолбэк — вертикальный размах ЧЕРНИЛА кривых (если рамка не сплошная)."""
    H, W = rgb.shape[:2]
    if not tracks:
        x0, x1 = int(0.06 * W), int(0.96 * W)
    else:
        x0, x1 = tracks[0].x_left, tracks[-1].x_right
    dark = (im.value_channel(rgb) < p.grid_v_hi).astype(np.uint8)
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(0.5 * (x1 - x0))), 1)))
    rows = horiz[:, x0:x1].sum(1).astype(np.float64)
    strong = np.nonzero(rows >= 0.5 * (x1 - x0))[0]
    if len(strong) >= 2:
        return int(strong[0]), int(strong[-1])
    # фолбэк по чернилу
    fg = im.ink_foreground(rgb, p)
    rr = fg[:, x0:x1].sum(1)
    nz = np.nonzero(rr > 0.02 * (x1 - x0))[0]
    if len(nz):
        return int(nz[0]), int(nz[-1])
    return int(0.05 * H), int(0.95 * H)


def _tape_edge_zone(colfrac, W, side, thr=0.08, limit_frac=1 / 3):
    """Внутренняя граница тёмной краевой зоны (фон сканера + перфорация). None если зоны нет.
    Допуск на светлые прогалины между дырками/кромкой — max(8, W//25) колонок
    (BKZ_3080: перфозона отстоит от кромки скана на ~27 светлых колонок при W=743).
    limit_frac — насколько глубоко зона может заходить в лист (дрейфующая лента: до ~0.55)."""
    tol = max(8, W // 25)
    rng = range(W) if side == "L" else range(W - 1, -1, -1)
    last, misses = None, 0
    for x in rng:
        if colfrac[x] > thr:
            last, misses = x, 0
        else:
            misses += 1
            if misses > tol:
                break
    if last is None:
        return None
    # зона обязана быть краевой (не вертикаль в поле данных)
    if side == "L" and last > W * limit_frac:
        return None
    if side == "R" and last < W * (1 - limit_frac):
        return None
    return last


def _tape_periodic_holes(very_dark, zone_x0, zone_x1):
    """Перфорация = РЕГУЛЯРНЫЕ дырки: в зоне ищем колонку с частичной темнотой (дырки, не сплошной
    фон) и проверяем периодичность блобов по вертикали."""
    H = very_dark.shape[0]
    for x in range(zone_x0, zone_x1, max(1, (zone_x1 - zone_x0) // 12 or 1)):
        frac = very_dark[:, x].mean()
        if not (0.05 < frac < 0.6):
            continue
        rows = np.nonzero(very_dark[:, x])[0]
        if len(rows) < 12:
            continue
        cuts = np.nonzero(np.diff(rows) > 5)[0] + 1
        centers = [float(seg.mean()) for seg in np.split(rows, cuts)]
        if len(centers) < 6:
            continue
        d = np.diff(centers)
        med = float(np.median(d))
        if med > 4 and float(np.median(np.abs(d - med))) < 0.35 * med:
            return True
    return False


def _tape_data_span(rgb, p, x0, x1, block=64, k=3):
    """Верх/низ ДАННЫХ на ленте: рукописная шапка/хвост vs зона кривой.
    Блок 64 строк «кривая»: ≥75% строк имеют 1..rmax ранов чернила (кривая непрерывна, рукопись —
    россыпь штрихов либо пусто). Данные = от первых k подряд «кривых» блоков до последних.
    rmax=6 (замер Semeguniv, 1-3 кривые). МУЛЬТИ-кривые ломают потолок 6 (STK_3280: 4-5 кривых,
    медиана ранов 5-8 → проходили лишь обрывки, span 256 строк) → если span < 40% H, ретрай с
    rmax=12; берём широкий вариант, если он ≥1.5× длиннее (шапку k-подряд всё равно отсекает:
    рукопись даёт и строки >12 ранов, её блоки не выстраиваются в k подряд)."""
    fg = im.ink_foreground(rgb, p)[:, x0:x1]
    H = fg.shape[0]
    nb = H // block
    if nb < k:
        return 0, H, 0.0
    rr_blocks = [np.array([len(im.row_runs(fg[i * block + j])) for j in range(block)])
                 for i in range(nb)]

    def span_for(rmax):
        ok = np.array([float(((rr >= 1) & (rr <= rmax)).mean()) > 0.75 for rr in rr_blocks])
        runs = [i for i in range(nb - k + 1) if ok[i:i + k].all()]
        if not runs:
            return None
        return runs[0] * block, min(H, (runs[-1] + k) * block), float(ok.mean())

    s6 = span_for(6)
    if s6 is not None and (s6[1] - s6[0]) >= 0.4 * H:
        return s6
    s12 = span_for(12)
    if s12 is not None and (s6 is None or (s12[1] - s12[0]) >= 1.5 * (s6[1] - s6[0])):
        return s12
    return s6 if s6 is not None else (0, H, 0.0)


def _tape_drift(very_dark, W, H, n=16):
    """ДРЕЙФУЮЩАЯ лента (косой/кривой скан, BK_190: край плывёт ±170px за 30000 строк —
    глобальные краевые зоны размазаны и _tape_edge_zone молчит). По горизонтальным слэбам ищем
    (zl, zr): лента = согласованная ШИРИНА (zr−zl) + светлая середина в ≥60% слэбов.
    Возвращает (row_shift, x0, x1, диаг) в ВЫПРЯМЛЕННЫХ координатах, либо None."""
    zs, cfs = [], []
    for i in range(n):
        s = very_dark[i * H // n:(i + 1) * H // n]
        cf = s.mean(0)
        cfs.append(cf)
        zs.append((_tape_edge_zone(cf, W, "L", limit_frac=0.55),
                   _tape_edge_zone(cf, W, "R", limit_frac=0.55)))
    # 1-й проход: слэбы с ОБОИМИ краями и светлой серединой → медианная ширина ленты
    both = []
    for (zl, zr), cf in zip(zs, cfs):
        if zl is not None and zr is not None and zr - 3 > zl + 3 \
                and float(np.median(cf[zl + 3:zr - 3])) < 0.25:
            both.append((zl, zr))
    if len(both) < 0.35 * n:
        return None
    widths = np.array([zr - zl for zl, zr in both], float)
    wmed = float(np.median(widths))
    if not (0.25 * W <= wmed <= 0.9 * W) or float(np.median(np.abs(widths - wmed))) > 0.1 * wmed:
        return None                                   # ширина гуляет — это не жёсткая лента
    w = int(round(wmed))
    # 2-й проход: центры; ОДНОСТОРОННИЕ слэбы достраиваем известной шириной (дрейф уводит второй
    # край за позиционный лимит — BK_190 слэбы 0,6-8,13-15)
    centers = np.full(n, np.nan)
    for i, ((zl, zr), cf) in enumerate(zip(zs, cfs)):
        if zl is not None and zr is not None:
            if abs((zr - zl) - wmed) < 0.15 * wmed:
                centers[i] = (zl + zr) / 2.0
        elif zl is not None and zl + w <= W + 0.1 * w:
            hi = min(W, zl + w) - 3
            if hi > zl + 3 and float(np.median(cf[zl + 3:hi])) < 0.25:
                centers[i] = zl + wmed / 2.0
        elif zr is not None and zr - w >= -0.1 * w:
            lo = max(0, zr - w) + 3
            if zr - 3 > lo and float(np.median(cf[lo:zr - 3])) < 0.25:
                centers[i] = zr - wmed / 2.0
    valid = ~np.isnan(centers)
    if valid.sum() < 0.6 * n:
        return None
    ys = (np.arange(n) * H + H // 2) // n             # центр каждого слэба по y
    row_c = np.interp(np.arange(H), ys[valid], centers[valid])
    cmed = float(np.median(row_c))
    row_shift = np.round(cmed - row_c).astype(np.int32)   # сдвиг строки → лента центрируется
    x0, x1 = int(cmed - wmed / 2), int(cmed + wmed / 2)
    diag = {"drift_px": int(np.ptp(row_c)), "tape_w_px": int(wmed), "slabs_ok": int(valid.sum())}
    return row_shift, x0, x1, diag


def apply_row_shift(rgb, row_shift):
    """Выпрямление дрейфующей ленты: каждая строка сдвигается на row_shift[y] (см. _tape_drift).
    Группируем строки по величине сдвига — один np.roll на группу."""
    out = np.empty_like(rgb)
    for s in np.unique(row_shift):
        rows = np.nonzero(row_shift == s)[0]
        out[rows] = np.roll(rgb[rows], int(s), axis=1) if s else rgb[rows]
    return out


# px/m-приоры корпуса (Semeguniv, скан ~150dpi): 1:200 → ~29-31, 1:500 → ~11-12.
_TAPE_PPM_PRIOR = {200: (22.0, 40.0), 500: (8.0, 16.0)}


def detect_tape(rgb, meta=None, p=None):
    """Перфолента (Semeguniv и т.п.): узкий однотрековый бланк с перфорацией по краям и рукописной
    шапкой. Возвращает Frame или None (не лента). Триггер: краевая тёмная зона с ПЕРИОДИЧНЫМИ
    дырками хотя бы с одной стороны."""
    from .config import DEFAULT
    p = p or DEFAULT.cv
    H, W = rgb.shape[:2]
    v = im.value_channel(rgb)
    very_dark = (v < 80)
    colfrac = very_dark.mean(0)
    zl = _tape_edge_zone(colfrac, W, "L")
    zr = _tape_edge_zone(colfrac, W, "R")
    periodic = ((zl is not None and _tape_periodic_holes(very_dark, 0, zl + 1)) or
                (zr is not None and _tape_periodic_holes(very_dark, zr, W)))
    # Запасной триггер (perf НЕ подтверждена, но геометрия ленты однозначна): тёмные краевые зоны
    # С ОБЕИХ сторон + СВЕТЛАЯ середина (данные, не сплошь-тёмный скан) + нет вертикалей рамки (гейт
    # в detect_frame). Замер Semeguniv 10.07: DS/MK/RK/BKZ имеют zl&zr, interiorMed 0.002-0.014 — это
    # ленты, но дырки бледные/нерегулярные и периодичность не ловится → 7/22 падали в «весь лист».
    # Защита от сплошь-тёмного скана (BK_190: colfrac~0.97) — середина обязана быть светлой (<0.25).
    two_sided = False
    if not periodic and zl is not None and zr is not None and zr - zl > 0.3 * W:
        interior = float(np.median(colfrac[zl + 3:zr - 3])) if zr - 3 > zl + 3 else 1.0
        two_sided = interior < 0.25
    # Третий триггер: ДРЕЙФУЮЩАЯ лента (косой скан) — глобальные зоны размазаны, ищем по слэбам.
    drift = None
    row_shift = None
    if not periodic and not two_sided:
        drift = _tape_drift(very_dark, W, H)
        if drift is None:
            return None
    if drift is not None:
        row_shift, dx0, dx1, ddiag = drift
        rgb = apply_row_shift(rgb, row_shift)          # дальше работаем в выпрямленных координатах
        x0, x1 = dx0 + 9, dx1 - 8
    else:
        x0 = (zl + 9) if zl is not None else 10
        x1 = (zr - 8) if zr is not None else W - 10
    if x1 - x0 < 0.3 * W:
        return None
    top_y, bottom_y, ok_frac = _tape_data_span(rgb, p, x0, x1)
    period, gys = _grid_period(rgb, p, top_y, bottom_y, x0, x1)
    td = getattr(meta, "top_depth", None)
    bd = getattr(meta, "bottom_depth", None)
    fr = Frame(img_w=W, img_h=H, top_y=top_y, bottom_y=bottom_y,
               tracks=[Track(x0, x1, 0)], top_depth=td, bottom_depth=bd,
               grid_period_px=period, grid_ys=gys,
               diag={"source": "tape", "n_tracks": 1, "perf_left": zl, "perf_right": zr,
                     "perf_confirmed": bool(periodic),
                     "data_blocks_frac": round(ok_frac, 2),
                     "depth_from_filename": td is not None})
    if drift is not None:
        fr.diag["tape_drift"] = ddiag                  # косой скан: строки выпрямлены
        fr.row_shift = row_shift                       # пайплайн обязан применить apply_row_shift
    # честность: длина данных должна сходиться с интервалом из имени (px/m в приоре масштаба)
    ppm = fr.px_per_m
    lohi = _TAPE_PPM_PRIOR.get(getattr(meta, "scale", None) or 0)
    if ppm is None or (lohi and not (lohi[0] <= ppm <= lohi[1])):
        fr.diag["low_confidence"] = True
        fr.diag["advise"] = (f"px/м={ppm and round(ppm,1)} вне приора {lohi} для 1:{getattr(meta,'scale',None)} — "
                             "длина ленты не сходится с интервалом из имени; глубинная калибровка "
                             "ненадёжна (нужны метки глубин/сетка)")
    else:
        fr.diag["low_confidence"] = False
    return fr


def _grid_period(rgb, p, top_y, bottom_y, x0, x1):
    """Период горизонтальной сетки в светло-серой полосе (для опц. Depth Grid). (period, ys)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    band = ((g >= p.grid_v_lo) & (g < p.grid_v_hi)).astype(np.uint8)
    prof = band[top_y:bottom_y, x0:x1].sum(1).astype(np.float64)
    if len(prof) > 31:
        prof = prof - np.convolve(prof, np.ones(31) / 31, mode="same")
    prof = np.clip(prof, 0, None)
    peaks = im.find_peaks(prof, min_dist=4, prominence=float(np.percentile(prof, 80) or 1))
    if len(peaks) < 3:
        return None, []
    period = float(np.median(np.diff(peaks)))
    return period, [int(top_y + y) for y in peaks]


def frame_from_nlgx(nlgx_path, meta=None, p=None, rgb=None) -> Frame:
    """U0 ИЗ ШАБЛОНА: построить Frame по лёгкой рамке NeuraLOG (Depth/Scale Axis), когда она дана.

    ROADMAP: шаблон ОПЦИОНАЛЕН, но если эксперт дал лёгкую рамку — её калибровка ТОЧНЕЕ авто-детекта.
    DURABLE-обоснование (замер на Yatskivska): автономный детект рамки тут проваливается — НЕТ тёмных
    рамочных линий (грани/верх-низ нарисованы тоном СВЕТЛОЙ сетки V~166), а сетка идёт по всей высоте
    листа за пределами данных; авто-детект откатывался на «весь лист» (top_y=0, трек = вся ширина).
    Беря рамку из nlgx, весь конвейер (understand/trace/emit) работает в ОДНОЙ системе координат."""
    from .config import DEFAULT
    p = p or DEFAULT.cv
    from extract_nlgx import extract                      # digitizer на sys.path (см. __init__)
    model = extract(str(nlgx_path))
    da = model.get("depth_axis") or {}
    H, W = (rgb.shape[:2] if rgb is not None else (0, 0))
    top_y = int(da.get("top_y") or 0)
    bottom_y = int(da.get("bottom_y") or (H - 1 if H else 0))
    # глубины: приоритет калибровке рамки, фолбэк — имя файла (meta)
    td, bd = da.get("top_depth"), da.get("bottom_depth")
    if td is None or bd is None or td == bd:
        td, bd = getattr(meta, "top_depth", None), getattr(meta, "bottom_depth", None)
    # треки = слитые по x пересекающиеся scale-оси (SA1/SA2 одного трека → один трек)
    spans = sorted((int(s["x_left"]), int(s["x_right"])) for s in model.get("scale_axes", [])
                   if s.get("x_left") is not None and s.get("x_right") is not None)
    merged = []
    for x0, x1 in spans:
        if merged and x0 <= merged[-1][1] + 5:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    if not merged:
        merged = [[int(0.06 * W), int(0.96 * W)]] if W else [[0, 1]]
    tracks = [Track(a, b, i) for i, (a, b) in enumerate(merged)]
    period, gys = (None, [])
    if rgb is not None and bottom_y > top_y:
        period, gys = _grid_period(rgb, p, top_y, bottom_y, tracks[0].x_left, tracks[-1].x_right)
    return Frame(img_w=W, img_h=H, top_y=top_y, bottom_y=bottom_y, tracks=tracks,
                 top_depth=td, bottom_depth=bd, grid_period_px=period, grid_ys=gys,
                 diag={"source": "nlgx", "n_tracks": len(tracks),
                       "n_scale_axes": len(model.get("scale_axes", [])),
                       "depth_from": "nlgx" if da.get("top_depth") is not None else "filename"})


def detect_frame(rgb, meta=None, p=None) -> Frame:
    """Главная точка U0: RGB (+ meta из имени файла) → Frame (треки, верх/низ, depth-калибровка, сетка)."""
    from .config import DEFAULT
    p = p or DEFAULT.cv
    H, W = rgb.shape[:2]
    xs, colprof = _vertical_lines(rgb, p)
    if len(xs) < 2:                                 # нет граней рамки → возможно перфолента
        tape = detect_tape(rgb, meta, p)
        if tape is not None:
            return tape
    fg = im.ink_foreground(rgb, p)
    tracks = _tracks_from_verticals(rgb, p, xs, fg)
    top_y, bottom_y = _frame_top_bottom(rgb, p, tracks)
    x0 = tracks[0].x_left if tracks else int(0.06 * W)
    x1 = tracks[-1].x_right if tracks else int(0.96 * W)
    period, gys = _grid_period(rgb, p, top_y, bottom_y, x0, x1)
    fr = Frame(img_w=W, img_h=H, top_y=top_y, bottom_y=bottom_y, tracks=tracks,
               top_depth=getattr(meta, "top_depth", None),
               bottom_depth=getattr(meta, "bottom_depth", None),
               grid_period_px=period, grid_ys=gys,
               diag={"n_vertical_lines": len(xs), "n_tracks": len(tracks),
                     "depth_from_filename": meta is not None and meta.top_depth is not None})
    # ЧЕСТНАЯ уверенность: если грань-сигнала нет (сетка-бумага, как Yatskivska — грани/верх-низ тоном
    # светлой сетки, нет тёмных линий), детект вырождается в «весь лист». Не молчим — флагаем, чтобы
    # pipeline/UI посоветовал дать --frame (durable §6.6.9: автономный U0 тут не строится прямыми).
    full_w = bool(tracks) and (tracks[-1].x_right - tracks[0].x_left) > 0.85 * W
    full_h = (bottom_y - top_y) > 0.90 * H
    fr.diag["low_confidence"] = bool(len(tracks) <= 1 and full_w and full_h)
    if fr.diag["low_confidence"]:
        fr.diag["advise"] = "нет грань-сигнала (сетка-бумага) — дайте --frame для точной калибровки"
    return fr
