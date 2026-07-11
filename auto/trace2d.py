r"""
trace2d.py — 2D-обход штриха для AUTO-бакета (замена per-row x(y), durable §6.6.8/§6.6.9).

ПОЧЕМУ не per-row: x(y) однозначен на ВЕРТИКАЛЬНЫХ участках, но на горизонтальном спайке
(резкий пик резистива / уход микрозонда) один ряд = длинный прогон → центроид «срезает угол,
садится на базлайн». Решение — вести по СВЯЗНОСТИ РАНОВ (run-adjacency граф, PLAN §6.6.11 ст.0):
ряд→ряд переходим в перекрывающийся по x ран (стрих физически непрерывен), а X на строке берём
как ВЕРШИНУ выноса (дальний край рана от базлайна), а не его центр — так пик доводится до конца.

Только для линий, помеченных confidence='AUTO' (одиночные/разнесённые/цвето-уникальные). Сбитый
пучок (FLAG) сюда НЕ идёт. Чистый numpy/cv2, без scipy/skimage.
"""
import numpy as np
from . import imaging as im


def _color_fg(rgb, color, p):
    """Бинарь переднего плана одного цвета (для трассировки конкретной линии)."""
    if color == "black":
        m = im.dark_mask(rgb, p)
        for cm in im.color_channels(rgb, p).values():
            m = m & ~cm
        return m & ~im.structure_mask(rgb, p)
    cm = im.color_channels(rgb, p).get(color)
    if cm is None:
        return np.zeros(rgb.shape[:2], bool)
    return cm & ~im.structure_mask(rgb, p)


def trace_line(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14):
    """Трасса одной AUTO-линии как x(row). fg — bool-маска цвета линии.
    band_pad — допуск вокруг x-полосы линии (анти-перескок на соседа). slmax — кламп скорости.
    wide_run — ран шире этого = горизонтальный спайк → берём ВЕРШИНУ (дальний край), не центр."""
    H, W = fg.shape
    lo = max(0, int(line.x_lo) - band_pad); hi = min(W, int(line.x_hi) + band_pad + 1)
    base = line.x_center            # ориентир базлайна (вершину пика тянем ОТ него)
    x = None; v = 0.0
    tr = {}
    for y in range(max(0, line.y0), min(H, line.y1 + 1)):
        runs = im.row_runs(fg[y, lo:hi])
        runs = [(a + lo, b + lo, c + lo) for a, b, c in runs]
        if not runs:
            if x is not None:                          # коаст по инерции через короткий разрыв
                x = x + float(np.clip(v, -slmax, slmax))
            continue
        if x is None:
            a, b, c = min(runs, key=lambda r: abs(r[2] - base))
            x = c; v = 0.0; tr[y] = x; continue
        pred = x + float(np.clip(v, -slmax, slmax))
        # выбрать РАН, перекрывающий предсказание (связность штриха), иначе ближайший по центру
        cont = [r for r in runs if r[0] - 2 <= pred <= r[1] + 2]
        a, b, c = (min(cont, key=lambda r: abs(r[2] - pred)) if cont
                   else min(runs, key=lambda r: abs(r[2] - pred)))
        if (b - a) >= wide_run:                        # горизонтальный спайк → вершина выноса
            nx = b if abs(b - base) >= abs(a - base) else a
        else:
            nx = c
        v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
    _extend_ends(tr, fg, lo, hi, slmax)
    return tr


def _extend_ends(tr, fg, lo, hi, slmax, max_gap=25):
    """Доводка трассы за [y0,y1] по СВЯЗНОМУ чернилу цвета (G4/BK_4020: верхний заход кривой
    4017-4020 терялся при группировке — линия стартовала на 4020). Идём от края вверх/вниз,
    садимся на ближайший к предсказанию ран, стоп после max_gap пустых строк подряд."""
    H = fg.shape[0]
    if not tr:
        return
    for direction in (-1, +1):
        y0 = min(tr) if direction < 0 else max(tr)
        x = tr[y0]; gap = 0; y = y0 + direction
        while 0 <= y < H and gap <= max_gap:
            runs = im.row_runs(fg[y, lo:hi])
            if not runs:
                gap += 1; y += direction; continue
            runs = [(a + lo, b + lo, c + lo) for a, b, c in runs]
            a, b, c = min(runs, key=lambda r: abs(r[2] - x))
            if abs(c - x) > slmax + (b - a):        # разрыв идентичности — не тянем на соседа
                break
            x = c; tr[y] = float(c); gap = 0; y += direction


def trace_auto(rgb, sheet, p=None):
    """Трассировать все AUTO-линии листа. Возвращает list[(Line, {row:x})].
    Каждая трасса ДЕСПАЙКается (refine.despike): изолированные выбросы-спайки (перескок на рамку/
    сосед на 1-2 строки) заменяются локальной медианой; устойчивый пик кривой сохраняется."""
    from .config import DEFAULT
    from . import refine
    p = p or DEFAULT.cv
    out = []
    fg_cache = {}
    for L in sheet.lines:
        if L.confidence != "AUTO":
            continue
        if L.color not in fg_cache:
            fg_cache[L.color] = _color_fg(rgb, L.color, p)
        tr = trace_line(fg_cache[L.color], L, sheet.frame, p)
        if len(tr) >= 30:
            tr, _ = refine.despike(tr, win=p.despike_win, k=p.despike_k,
                                   min_jump=p.despike_min_jump)
            out.append((L, tr))
    return out
