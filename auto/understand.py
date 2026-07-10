r"""
understand.py — U1: ГЛУБОКИЙ РАЗБОР снимка → структурное ПОНИМАНИЕ листа (первичный артефакт v2).
Отвечает на запрос заказчика: «изначально понимать, СКОЛЬКО ЛИНИЙ и СКОЛЬКО ПЕРЕХОДОВ МАСШТАБА
у каждой». Работает ОТ ИЗОБРАЖЕНИЯ (frame из U0), БЕЗ экспертной трассы.

На каждый трек:
  • передний план кривых (imaging: тёмное|цветное − структура [× prob-гейт recall-модели]);
  • по каждому цвету — инстансы-штрихи (CC, линия-подобные) с центр-трассой/толщиной/поведением;
  • счёт РАЗНЕСЁННЫХ линий по долинам столбцовой плотности (band-детект, PLAN §6.6.11);
  • оценка числа уровней-перевыносов (передаётся в scales.py для уточнения по линейке A1).

Поведение (rough_n): SP гладкая / резистив-CALI пиковая (B2). Без nlgx — px/m из frame (U0).
Никакого torch: recall-модель опциональна (config.prob_provider). Чистый OpenCV/NumPy.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
from . import imaging as im


@dataclass
class Line:
    track_index: int
    color: str
    x_center: float
    x_lo: float
    x_hi: float
    y0: int
    y1: int
    thickness: float
    rough_n: Optional[float]
    behavior: str                 # 'smooth' | 'peaky' | '?'
    n_strokes: int                # фрагментация (перевынос/окклюзия → много штрихов)
    density: float                # средняя столбцовая плотность (для AUTO/FLAG)
    depth_start: Optional[float] = None
    depth_end: Optional[float] = None
    n_levels_est: Optional[int] = None     # оценка переходов масштаба (уточняет scales.py)
    n_runs_med: float = 0.0                # медиана инк-ранов на строку в полосе (U2): ~1 одиночная / ≥2 пучок
    row_cov: float = 0.0                   # доля строк y0..y1, покрытых чернилами (кривая ~непрерывна,
    #                                        метки глубин/мусор — россыпь: G2-дискриминатор)
    strokes_xstd_med: float = 9.9          # медиана x-std членов-штрихов: ось по линейке ~0.4px,
    #                                        даже выцветшая кривая ≥1.8px (G2-замер Semeguniv)
    confidence: Optional[str] = None       # 'AUTO' | 'FLAG' (заполняет confidence.classify)
    flag_reason: Optional[str] = None

    @property
    def x_band(self):
        return self.x_hi - self.x_lo


@dataclass
class Sheet:
    meta: object
    frame: object
    lines: list = field(default_factory=list)
    per_track: dict = field(default_factory=dict)     # track_index → число линий
    diag: dict = field(default_factory=dict)

    def to_dict(self):
        m, f = self.meta, self.frame
        return {
            "well": getattr(m, "well", None), "curves_token": getattr(m, "curves_token", None),
            "depth": [getattr(m, "top_depth", None), getattr(m, "bottom_depth", None)],
            "scale": getattr(m, "scale", None),
            "expected_curves": getattr(m, "expected_curves", []),
            "frame": {"tracks": [[t.x_left, t.x_right] for t in f.tracks],
                      "top_y": f.top_y, "bottom_y": f.bottom_y,
                      "px_per_m": f.px_per_m, "grid_period_px": f.grid_period_px,
                      "diag": f.diag},
            "n_lines_total": len(self.lines),
            "per_track_line_count": self.per_track,
            "lines": [{
                "track": L.track_index, "color": L.color,
                "x_center": round(L.x_center, 1), "x_band": round(L.x_band, 1),
                "depth": [None if L.depth_start is None else round(L.depth_start, 1),
                          None if L.depth_end is None else round(L.depth_end, 1)],
                "thickness": round(L.thickness, 1), "behavior": L.behavior,
                "rough_n": L.rough_n, "n_strokes": L.n_strokes,
                "n_runs_med": round(L.n_runs_med, 2),
                "row_cov": round(L.row_cov, 2),
                "n_levels_est": L.n_levels_est,
                "confidence": L.confidence, "flag_reason": L.flag_reason,
            } for L in self.lines],
            "diag": self.diag,
        }


def _instances(mask, color, track_index, min_h, min_px=200):
    """CC канала → линия-подобные инстансы (центр-трасса x(row), толщина). Без nlgx."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h or area < min_px:
            continue
        sub = lab[y:y + h, x:x + w] == i
        xs_row = np.full(h, np.nan)
        th = []
        ys, xsx = np.where(sub)
        for r in range(h):
            cols = xsx[ys == r]
            if len(cols):
                xs_row[r] = x + np.median(cols); th.append(len(cols))
        if (~np.isnan(xs_row)).sum() < min_h * 0.3:
            continue
        out.append({"color": color, "track": track_index, "y0": int(y), "y1": int(y + h),
                    "xs_row": xs_row, "row0": int(y), "area": int(area),
                    "thickness": float(np.median(th)) if th else 0.0})
    return out


def _behavior(xs_row, row0, px_per_m):
    """rough_n из центр-трассы инстанса (B2: SP гладкая ~0.01 / пиковая >0.03)."""
    valid = ~np.isnan(xs_row)
    if valid.sum() < 30 or not px_per_m:
        return None, "?"
    ys = np.where(valid)[0] + row0
    xs = xs_row[valid]
    grid = np.arange(ys.min(), ys.max() + 1)
    xi = np.interp(grid, ys, xs)
    W = max(5, int(round(1.5 * px_per_m)) | 1)
    if len(xi) <= W:
        return None, "?"
    sm = np.convolve(xi, np.ones(W) / W, mode="same")
    hf = (xi - sm)[W:-W] if len(xi) > 2 * W else (xi - sm)
    span = float(np.percentile(xs, 97) - np.percentile(xs, 3)) or 1.0
    rn = round(float(np.std(hf)) / span, 4)
    return rn, ("smooth" if rn < 0.025 else "peaky")


def _row_multiplicity(cmask, y0, y1, x_lo, x_hi, gap=4, max_rows=800):
    """Медиана числа ИНК-РАНОВ на строку в x-полосе линии = мультипликативность (durable §6.6.11).
    ОДИНОЧНАЯ кривая даёт ~1 ран/строку ДАЖЕ на широком размахе (band велик, но в каждой строке —
    один штрих); сбитый ПУЧОК пересекающихся даёт ≥2 (на пересечении нет ни цвета, ни толщины, но
    ранов на строке считается ≥2). Различитель «одиночная широкая vs пучок» БЕЗ порога по band."""
    H, W = cmask.shape
    lo = max(0, int(x_lo)); hi = min(W, int(x_hi) + 1)
    y0 = max(0, int(y0)); y1 = min(H, int(y1) + 1)
    if hi <= lo or y1 <= y0:
        return 0.0
    step = max(1, (y1 - y0) // max_rows)                  # равномерный сэмпл ≤max_rows строк (скорость)
    counts = [len(im.row_runs(cmask[y, lo:hi], gap=gap)) for y in range(y0, y1, step)]
    counts = [c for c in counts if c]                     # только строки с чернилами
    return float(np.median(counts)) if counts else 0.0


def _row_profile(members):
    """Абсолютный профиль x(row) группы штрихов: (y0, xs[]) — nan там, где чернил нет."""
    y0 = min(s["row0"] for s in members)
    y1 = max(s["row0"] + len(s["xs_row"]) for s in members)
    prof = np.full(y1 - y0, np.nan)
    for s in members:
        v = ~np.isnan(s["xs_row"])
        prof[np.nonzero(v)[0] + s["row0"] - y0] = s["xs_row"][v]
    return y0, prof


def _try_merge_drift(groups, track_h, p):
    """ДРЕЙФУЮЩАЯ кривая режется x-сплитом на куски (G2-замер BKZ_3690: красная → 3 «линии» со
    стыком y 5087/5088). Физика: сплит по x-долине легален, только если полосы КОНКУРИРУЮТ за одни
    строки (две линии сосуществуют по глубине); куски одного цвета, НЕ пересекающиеся по строкам и
    непрерывные по x на стыке — одна кривая. Слияние до фикспоинта."""
    def profiles(g):
        if "_prof" not in g:
            g["_y0"], g["_prof"] = _row_profile(g["members"])
        return g["_y0"], g["_prof"]

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ga, gb = groups[i], groups[j]
                ya, pa = profiles(ga); yb, pb = profiles(gb)
                # конкуренция за строки: доля строк, покрытых ОБЕИМИ, от меньшего покрытия
                lo = max(ya, yb); hi = min(ya + len(pa), yb + len(pb))
                both = 0
                if hi > lo:
                    va = ~np.isnan(pa[lo - ya:hi - ya]); vb = ~np.isnan(pb[lo - yb:hi - yb])
                    both = int((va & vb).sum())
                na = int((~np.isnan(pa)).sum()); nb = int((~np.isnan(pb)).sum())
                if both / max(1, min(na, nb)) >= p.drift_max_concur:
                    continue
                # порядок по глубине: A выше B (по медиане покрытых строк)
                ra = ya + np.nonzero(~np.isnan(pa))[0]; rb = yb + np.nonzero(~np.isnan(pb))[0]
                if np.median(ra) > np.median(rb):
                    ga, gb = gb, ga; ra, rb = rb, ra
                    ya, pa = profiles(ga); yb, pb = profiles(gb)
                gap = int(rb.min() - ra.max())
                if not (-0.02 * track_h <= gap <= p.drift_max_gap_frac * track_h):
                    continue
                # непрерывность x на стыке: хвост A ≈ голова B
                k = 100
                tail = pa[~np.isnan(pa)][-k:]; head = pb[~np.isnan(pb)][:k]
                if abs(float(np.median(tail)) - float(np.median(head))) > p.drift_max_dx:
                    continue
                merged = {"members": ga["members"] + gb["members"]}
                groups[i] = merged
                del groups[j]
                changed = True
                break
            if changed:
                break
    return groups


def _group_into_lines(insts, cmask, color, track_index, frame, p):
    """Сгруппировать штрихи одного цвета в ЛИНИИ по x-полосам (band-детект, PLAN §6.6.11).
    Разнесённые (долина плотности) → отдельные линии ~1px; сбитый одноцветный пучок → 1 'multi'
    линия (FLAG разрешает U2). Куски дрейфа склеиваются обратно (_try_merge_drift).
    Каждая линия — агрегат своих штрихов. Возвращает (lines, excluded_marks)."""
    if not insts:
        return [], []
    # столбцовая плотность чернил всех штрихов цвета (гистограмма x) → пики/долины = линии
    xs_all = np.concatenate([s["xs_row"][~np.isnan(s["xs_row"])] for s in insts])
    if not len(xs_all):
        return [], []
    lo, hi = int(xs_all.min()), int(xs_all.max()) + 1
    hist = np.zeros(hi - lo + 1, np.float64)
    for xv in xs_all.astype(int):
        hist[xv - lo] += 1
    hist = np.convolve(hist, np.ones(p.density_smooth) / p.density_smooth, mode="same")
    peaks = im.find_peaks(hist, min_dist=10, prominence=float(np.percentile(hist, 70) or 1))
    splits = im.find_valleys(hist, peaks, p.valley_ratio) if len(peaks) > 1 else []
    # границы x-полос = долины-сплиты; иначе одна полоса (весь диапазон)
    cuts = sorted(lo + s[2] for s in splits)
    edges = [lo] + cuts + [hi]
    px_per_m = frame.px_per_m
    track_h = frame.bottom_y - frame.top_y
    groups = []
    for a, b in zip(edges, edges[1:]):
        members = [s for s in insts if a <= float(np.nanmedian(s["xs_row"])) < b]
        if not members:
            continue
        xs_m = np.concatenate([s["xs_row"][~np.isnan(s["xs_row"])] for s in members])
        # отсев одиночных специй: вертикальное покрытие полосы должно быть выше шумового пола
        # (xs_m — по медиане x на строку, т.е. ~число покрытых строк). Реальная линия покрывает
        # заметную долю интервала; синий 1-штриховой артефакт ~10-40px — отсекаем.
        cov_min = max(2 * p.min_line_h_px, int(p.min_line_cov_frac * track_h))
        if len(xs_m) < cov_min:
            continue
        # ПРЯМАЯ референс/грань-вертикаль (x почти константа над cov-порогом) ≠ кривая (та варьирует x —
        # это измерение). Светлую вертикаль structure_mask не ловит (тон ~бумаги, рвётся на полной высоте).
        # Различитель — сам x-размах линии: ≤straight_max_band при достаточном покрытии = прямая (Yatskivska
        # синяя печатная вертикаль x394 band4). Реальная даже тонкая кривая виляет шире.
        if float(xs_m.max() - xs_m.min()) <= p.straight_max_band:
            continue
        groups.append({"members": members})
    # ЛЕНТЫ: ось-вертикаль с прицепленными рукописными МЕТКАМИ ГЛУБИН у ЛЕВОГО края трека
    # (G2: BK_2410 x≈114, 57 штрихов, покрытие 24%). Исключаем НА УРОВНЕ ГРУПП, ДО слияний —
    # иначе merge-одной-кривой втягивает ось+метки в кривую (регресс BK_2410 got=0).
    # Ключ = ПРЯМИЗНА членов (ось по линейке x-std med ~0.4px; даже выцветшая кривая ≥1.8 —
    # покрытие и периодичность НЕ разделяют, замер MK_190/STK_200). Только чёрный: метки
    # пишутся карандашом/тушью, цветное у края = кривая.
    excluded = []
    if frame.diag.get("source") == "tape" and color == "black" and groups:
        tr = frame.tracks[track_index]
        keep = []
        for g in groups:
            ms = g["members"]
            xs_g = np.concatenate([s["xs_row"][~np.isnan(s["xs_row"])] for s in ms])
            _, prof = _row_profile(ms)
            xc = float(np.median(xs_g))
            cov = float((~np.isnan(prof)).mean())
            xstd = float(np.median([np.std(s["xs_row"][~np.isnan(s["xs_row"])]) for s in ms]))
            if ((xc - tr.x_left) < p.marks_left_frac * max(1, tr.width)
                    and cov < p.marks_max_cov and len(ms) >= p.marks_min_strokes
                    and xstd <= p.marks_max_xstd):
                excluded.append({"track": track_index, "color": color,
                                 "x_center": round(xc, 1), "n_strokes": len(ms),
                                 "row_cov": round(cov, 2), "xstd_med": round(xstd, 2)})
            else:
                keep.append(g)
        groups = keep
    # ОДНА дико-пиковая кривая режется x-сплитом на куски (G2-замер BK_4020: 1→5 «линий»
    # th 13-19). Канон §6.6.11 (мультипликативность): если штрихи цвета почти нигде НЕ
    # сосуществуют в одной строке (медиана числа штрихов на покрытую строку ~1) — на треке
    # ОДНА кривая этого цвета, любой сплит нелегален. Считаем по ЧЛЕНАМ выживших групп
    # (прямые/метки уже отсеяны — ось в cmask завысила бы счёт до 2).
    if len(groups) > 1:
        alls = [s for g in groups for s in g["members"]]
        ya = min(s["row0"] for s in alls)
        yb = max(s["row0"] + len(s["xs_row"]) for s in alls)
        cnt = np.zeros(yb - ya, np.int16)
        for s in alls:
            v = ~np.isnan(s["xs_row"])
            cnt[np.nonzero(v)[0] + s["row0"] - ya] += 1
        covered = cnt > 0
        if covered.any() and float(np.median(cnt[covered])) <= p.single_curve_max_runs:
            groups = [{"members": alls}]
    groups = _try_merge_drift(groups, track_h, p)
    lines = []
    for g in groups:
        members = g["members"]
        xs_m = np.concatenate([s["xs_row"][~np.isnan(s["xs_row"])] for s in members])
        y0 = min(s["y0"] for s in members); y1 = max(s["y1"] for s in members)
        # покрытие строк полосы (уникальные строки с чернилами / высота): кривая непрерывна (высоко),
        # метки глубин/мусор — россыпь (низко). G2-дискриминатор, честная метрика для U2/фильтров.
        covered = np.zeros(max(1, y1 - y0 + 1), bool)
        for s in members:
            vr = np.nonzero(~np.isnan(s["xs_row"]))[0] + s["row0"] - y0
            covered[vr[(vr >= 0) & (vr < len(covered))]] = True
        # центр-трасса полосы = объединение штрихов (для поведения берём самый длинный)
        longest = max(members, key=lambda s: (~np.isnan(s["xs_row"])).sum())
        rn, beh = _behavior(longest["xs_row"], longest["row0"], px_per_m)
        L = Line(track_index=track_index, color=color,
                 x_center=float(np.median(xs_m)), x_lo=float(xs_m.min()), x_hi=float(xs_m.max()),
                 y0=int(y0), y1=int(y1),
                 thickness=float(np.median([s["thickness"] for s in members])),
                 rough_n=rn, behavior=beh, n_strokes=len(members),
                 density=float(hist[max(0, int(np.median(xs_m)) - lo)]))
        L.row_cov = float(covered.mean())
        L.strokes_xstd_med = float(np.median(
            [np.std(s["xs_row"][~np.isnan(s["xs_row"])]) for s in members]))
        L.n_runs_med = _row_multiplicity(cmask, y0, y1, L.x_lo, L.x_hi)
        if px_per_m:
            L.depth_start = frame.depth_of(y0); L.depth_end = frame.depth_of(y1)
        lines.append(L)
    lines.sort(key=lambda L: L.x_center)
    return lines, excluded


def understand(rgb, frame, meta=None, p=None, prob=None) -> Sheet:
    """Главная точка U1: RGB + Frame (U0) → Sheet (сколько линий, их свойства, оценка уровней)."""
    from .config import DEFAULT
    p = p or DEFAULT.cv
    if prob is None and DEFAULT.prob_provider is not None:
        prob = DEFAULT.prob_provider(rgb)
    sheet = Sheet(meta=meta, frame=frame)
    fg_full = im.ink_foreground(rgb, p, prob=prob)
    for t in frame.tracks:
        track_h = frame.bottom_y - frame.top_y
        # ПОЛ шума, не доля огромной высоты: clip(доля, пол, потолок) — иначе на полосе ~20000px
        # min_h≈800 отсекает ВСЕ фрагменты пиковой кривой (баг калибровки, найден на Yatskivska).
        min_h = int(np.clip(p.min_line_h_frac * track_h, p.min_line_h_px, p.min_line_h_cap))
        sub = np.zeros(rgb.shape[:2], bool)
        sub[frame.top_y:frame.bottom_y, t.x_left:t.x_right] = True
        # цветовые каналы + чёрный, ограниченные телом трека и передним планом
        chans = {**{c: m & sub & (fg_full > 0) for c, m in im.color_channels(rgb, p).items()}}
        dark = im.dark_mask(rgb, p) & sub & (fg_full > 0)
        for cm in chans.values():
            dark = dark & ~cm                                   # чёрный = тёмное минус цветное
        chans["black"] = dark
        track_lines = []
        for color, cmask in chans.items():
            insts = _instances(cmask, color, t.index, min_h)
            lines_c, excl_c = _group_into_lines(insts, cmask, color, t.index, frame, p)
            track_lines += lines_c
            if excl_c:      # ось+метки глубин, исключённые на уровне групп (см. _group_into_lines)
                sheet.diag.setdefault("excluded_marks", []).extend(excl_c)
        track_lines.sort(key=lambda L: L.x_center)
        sheet.per_track[t.index] = len(track_lines)
        sheet.lines += track_lines
    sheet.diag.update({"n_tracks": len(frame.tracks), "prob_used": prob is not None,
                       "expected_n_curves": len(getattr(meta, "expected_curves", []) or [])})
    return sheet
