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


def _tape_edge_zone(colfrac, W, side, thr=0.08):
    """Внутренняя граница тёмной краевой зоны (фон сканера + перфорация). None если зоны нет.
    Допуск на светлые прогалины между дырками/кромкой — max(8, W//40) колонок."""
    tol = max(8, W // 40)
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
    if side == "L" and last > W // 3:
        return None
    if side == "R" and last < 2 * W // 3:
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
    Блок 64 строк «кривая»: ≥75% строк имеют 1..6 ранов чернила (кривая непрерывна, рукопись — россыпь
    штрихов либо пусто). Данные = от первых k подряд «кривых» блоков до последних (замер Semeguniv)."""
    fg = im.ink_foreground(rgb, p)[:, x0:x1]
    H = fg.shape[0]
    nb = H // block
    if nb < k:
        return 0, H, 0.0
    ok = np.zeros(nb, bool)
    for i in range(nb):
        rr = np.array([len(im.row_runs(fg[i * block + j])) for j in range(block)])
        ok[i] = float(((rr >= 1) & (rr <= 6)).mean()) > 0.75
    runs = [i for i in range(nb - k + 1) if ok[i:i + k].all()]
    if not runs:
        return 0, H, 0.0
    return runs[0] * block, min(H, (runs[-1] + k) * block), float(ok.mean())


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
    if not periodic:
        return None
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
                     "data_blocks_frac": round(ok_frac, 2),
                     "depth_from_filename": td is not None})
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
