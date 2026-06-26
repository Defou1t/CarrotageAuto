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


def detect_frame(rgb, meta=None, p=None) -> Frame:
    """Главная точка U0: RGB (+ meta из имени файла) → Frame (треки, верх/низ, depth-калибровка, сетка)."""
    from .config import DEFAULT
    p = p or DEFAULT.cv
    H, W = rgb.shape[:2]
    fg = im.ink_foreground(rgb, p)
    xs, colprof = _vertical_lines(rgb, p)
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
    return fr
