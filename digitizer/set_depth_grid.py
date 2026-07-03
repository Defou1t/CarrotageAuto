r"""
set_depth_grid.py — перегенерировать Depth Grid (тип-8 IFD) на заданный шаг (по умолчанию
4 м для 1:200, канон §4) поверх калибровки Depth Axis. Линии ставятся на КРАТНЫЕ шагу
глубины (…3632, 3636…), как грани рамки в NeuraLOG (НЕ на рукописные подписи). Опционально
патчит скан-путь (тег 34878) на реальный — для QC. Пишет nlgx (+bck).

python set_depth_grid.py --nlgx <in.nlgx> [--step 4] [--scan <jpg>] [--out <out.nlgx>]
"""
import sys, struct
from pathlib import Path
import numpy as np
from write_nlgx import read_full, write_full, set_tag, find_ifd
from extract_nlgx import extract


def _tagval(ifd, tag, default=None):
    for t, typ, count, raw in ifd["entries"]:
        if t == tag:
            fmt = {3: "H", 4: "I", 12: "d"}.get(typ, "I")
            return struct.unpack("<" + fmt, raw[:struct.calcsize(fmt)])[0]
    return default


def snap_to_lines(ys_pred, det_ys, tol):
    """Снап (A2b): каждую предсказанную осью Y 4 м-линию двигаем к БЛИЖАЙШЕЙ детектированной
    линии сетки, если та в пределах `tol` px (иначе оставляем предсказание — никогда не хуже).
    Поглощает неравномерность бумаги (реальные диффы 123/120/126), угол уже задан A2.
    ⚠ НЕ давать сюда ВСЕ линии (тонкие ~12px): при изгибе > полушага тонкой снап цепляет
    тонкую соседку мимо жирной (QC 03.07) — для жирных использовать snap_bold_monotone."""
    if det_ys is None or not len(det_ys):
        return list(ys_pred), 0
    det = np.asarray(det_ys, float)
    out, moved = [], 0
    for p in ys_pred:
        j = int(np.abs(det - p).argmin())
        if abs(det[j] - p) <= tol:
            out.append(float(det[j])); moved += 1
        else:
            out.append(float(p))
    return out, moved


def snap_bold_monotone(ys_pred, bold_ys, step_px):
    """Монотонный матчинг предсказанных 4 м-линий на ЦЕПОЧКУ ЖИРНЫХ (detect_bold_hgrid):
    идём по предсказаниям по порядку, каждой берём ближайшую жирную в окне ±0.45 шага,
    СТРОГО ПОСЛЕ предыдущей выбранной (+0.5 шага) — две линии не сядут на одну жирную,
    большой изгиб (больше полушага тонкой) поглощается. Возврат (ys, snapped, max_shift)."""
    if bold_ys is None or not len(bold_ys):
        return list(ys_pred), 0, 0.0
    bold = np.asarray(sorted(bold_ys), float)
    tol = 0.45 * step_px
    out, moved, mx = [], 0, 0.0
    prev = -1e9
    for p in ys_pred:
        cand = bold[(np.abs(bold - p) <= tol) & (bold > prev + 0.5 * step_px)]
        if len(cand):
            y = float(cand[int(np.abs(cand - p).argmin())])
            out.append(y); moved += 1; mx = max(mx, abs(y - p)); prev = y
        else:
            out.append(float(p)); prev = float(p)
    return out, moved, mx


def predict_lines(m, step=4.0):
    """Линейные предсказания 4м-линий по оси: (depths, ys_axis, xs_axis, step_px) или None."""
    import math
    da = m["depth_axis"]
    ty, by = da["top_y"], da["bottom_y"]
    td, bd = da["top_depth"], da["bottom_depth"]
    xt, xb = da.get("x_top") or 0, da.get("x_bot") or 0
    if bd == td or by == ty:
        return None
    pxm = (by - ty) / (bd - td)
    xpm = (xb - xt) / (bd - td)
    lo, hi = min(td, bd), max(td, bd)
    d = math.ceil(lo / step) * step
    depths, ys, xs = [], [], []
    while d <= hi + 1e-6:
        depths.append(round(d, 3))
        ys.append(ty + (d - td) * pxm)
        xs.append(xt + (d - td) * xpm)
        d += step
    return depths, ys, xs, abs(step * pxm)


def regrid(ifds, m, step=4.0, det_ys=None, snap_tol=8.0, bold_ys=None, bold_step=None,
           bold_geo=None, lines_geo=None):
    """Перегенерировать Depth Grid на шаг `step` (м), КОРРЕКТНО с геометрией наклонённых
    сегментов. Каждая линия = сегмент (x_start,y_start)->(x_end,y_end), наклон сохраняется из
    тега 35570. Все per-line массивы (35594/35596/35598/35600/35601) пишутся согласованной длины,
    иначе NeuraLOG рисует неверный угол/веер (баг наивного регрида).
    A2b: если передан `det_ys` (детектированные линии сетки из картинки) — Y каждой линии
    снапится к ближайшей реальной линии в пределах snap_tol px (поглощает неравномерность)."""
    import math
    da = m["depth_axis"]
    ty, by = da["top_y"], da["bottom_y"]
    td, bd = da["top_depth"], da["bottom_depth"]
    xt, xb = da.get("x_top") or 0, da.get("x_bot") or 0
    if bd == td or by == ty:
        return None
    pxm = (by - ty) / (bd - td)        # px на метр (Y)
    xpm = (xb - xt) / (bd - td)        # наклон оси по X (px/м)
    idxs = find_ifd(ifds, lambda tags: 34768 in tags and
                    struct.unpack("<I", tags[34768][2][:4])[0] == 8)
    if not idxs:
        return None  # тип-8 IFD отсутствует — добавление нового IFD пока не делаем
    i = idxs[0]
    ifd = ifds[i]
    width = _tagval(ifd, 35568, 2269) or 2269          # ширина трека (px)
    slope = _tagval(ifd, 35570, 0.0) or 0.0            # наклон горизонтали (dy/dx)
    lo, hi = min(td, bd), max(td, bd)
    d0 = math.ceil(lo / step) * step
    depths, ys_pred = [], []
    d = d0
    while d <= hi + 1e-6:
        depths.append(round(d, 3))
        ys_pred.append(ty + (d - td) * pxm)            # левый Y (глубина), float
        d += step
    line_slopes = None
    if lines_geo and lines_geo[0] and isinstance(lines_geo[0], tuple):   # СВОБОДНАЯ ФАЗА (fit_bold_chain):
        # [(y_на_x0, slope, matched)] — фаза от ПЛАНШЕТА (гейт vs эталон: слоты по глубинам
        # промахивались на полшага там, где фаза жирных не совпадает с рамкой)
        chain = lines_geo
        ax_k = (xb - xt) / max(by - ty, 1)             # x оси как функция y
        step_px_c = float(np.median(np.diff([c[0] for c in chain]))) if len(chain) > 2 else 100.0
        y_lo, y_hi = min(ty, by), max(ty, by)
        # якорные линии (крестики) NeuraLOG создаёт САМ — наши у якорей выкидываем,
        # но пишем ДВЕ якорные с точным y оси и ИЗМЕРЕННЫМ углом (уточнение эксперта 03.07)
        body = [c for c in chain
                if abs(c[0] - y_lo) > 0.55 * step_px_c and abs(c[0] - y_hi) > 0.55 * step_px_c
                and y_lo - 2 <= c[0] <= y_hi + 2]
        sl_med = float(np.median([c[1] for c in chain])) if chain else 0.0
        sl_top = min(chain, key=lambda c: abs(c[0] - y_lo))[1] if chain else sl_med
        sl_bot = min(chain, key=lambda c: abs(c[0] - y_hi))[1] if chain else sl_med
        x_top_ax = xt + (y_lo - ty) * ax_k             # якоря заданы НА ОСИ → в координату x=0
        x_bot_ax = xt + (y_hi - ty) * ax_k
        full = ([(float(y_lo) - sl_top * x_top_ax, sl_top, True)] + body
                + [(float(y_hi) - sl_bot * x_bot_ax, sl_bot, True)])
        xs0, ys0, xs1, ys1 = [], [], [], []
        for y0l, sl, ok_ in full:
            xa = xt + (y0l - ty) * ax_k                # x оси на высоте линии (итерация 1 достаточно:
            ya = y0l + sl * xa                         # поправка sl·Δx < 1px)
            xs0.append(int(round(xa))); ys0.append(int(round(ya)))
            xs1.append(int(round(xa + width)))
            ys1.append(int(round(ya + sl * width)))
        set_tag(ifds, i, 35570, 12, sl_med)            # глобальный угол (авто-линии NeuraLOG)
        chain = full
        n = len(chain)
        set_tag(ifds, i, 35594, 4, xs0)
        set_tag(ifds, i, 35596, 4, ys0)
        set_tag(ifds, i, 35598, 4, xs1)
        set_tag(ifds, i, 35600, 4, ys1)
        set_tag(ifds, i, 35601, 4, [3] * n)
        set_tag(ifds, i, 35590, 4, n)
        set_tag(ifds, i, 35586, 4, int(round(step * pxm)))
        for t in (35578, 35582, 35584):
            set_tag(ifds, i, t, 12, float(step))
        print(f"  сетка со свободной фазой: {n} линий ({sum(1 for _,_,o in chain if o)} с детектом)")
        return [round(td + (y - ty) / pxm, 2) for y in ys0], ys0
    elif lines_geo:                                    # ГОТОВАЯ геометрия (fit_bold_grid, QC №5):
        ys_fit, sl_fit, ok = lines_geo                 # y на оси + наклон per-line
        ys_snap = list(ys_fit); line_slopes = list(sl_fit)
        print(f"  сетка по голосованию полос: {sum(ok)}/{len(ok)} линий с детектом")
    elif bold_geo:                                     # мультиполосный детект: (y_c, slope, x_c)
        lines, bstep = bold_geo
        x_c = lines[0][2] if lines else 0.5 * width
        med_sl = float(np.median([l[1] for l in lines if l[1] is not None])) if lines else slope
        # предсказания и цепочка сравниваются В ЦЕНТРЕ трека (там же, где детект — без подписей)
        ys_pred_c = [y + med_sl * (x_c - (xt + (d - td) * xpm)) for d, y in zip(depths, ys_pred)]
        yCs = [l[0] for l in lines]
        ys_snap_c, moved, mx = snap_bold_monotone(ys_pred_c, yCs, bstep or step * pxm)
        by_yc = {round(l[0], 1): l for l in lines}
        line_slopes, det_flag, ys_snap = [], [], []
        for d, yc in zip(depths, ys_snap_c):
            l = by_yc.get(round(yc, 1))
            det_flag.append(l is not None)
            sl = (l[1] if (l and l[1] is not None) else med_sl)
            line_slopes.append(sl)
            xa = xt + (d - td) * xpm
            ys_snap.append(yc + sl * (xa - x_c))       # y НА ОСИ (якорь в левую границу)
        got = sum(det_flag)
        print(f"  снап к ЖИРНЫМ (мультиполосный): {moved}/{len(ys_pred)} линий, max сдвиг {mx:.0f}px, "
              f"наклон med {med_sl*1000:.2f}px/1000px ({got} с детектом)")
    elif bold_ys is not None and len(bold_ys):
        ys_snap, moved, mx = snap_bold_monotone(ys_pred, bold_ys, bold_step or step * pxm)
        print(f"  снап к ЖИРНЫМ: {moved}/{len(ys_pred)} линий, max сдвиг {mx:.0f}px")
    else:
        ys_snap, moved = snap_to_lines(ys_pred, det_ys, snap_tol)
    xs0, ys0, xs1, ys1 = [], [], [], []
    for k, (d, ys) in enumerate(zip(depths, ys_snap)):
        xs = int(round(xt + (d - td) * xpm))           # левый X = ось глубин (левая граница)
        sl = line_slopes[k] if line_slopes else slope  # (мультиполосный путь: ys уже на оси)
        ys = int(round(ys))
        xs0.append(xs); ys0.append(ys)
        xs1.append(xs + int(width))                    # правый X
        ys1.append(int(round(ys + sl * width)))        # правый Y (наклон per-line)
    n = len(depths)
    if det_ys is not None:
        print(f"  снап A2b: сдвинуто {moved}/{n} линий к детектированным (tol {snap_tol}px)")
    set_tag(ifds, i, 35594, 4, xs0)        # X начала (левый)
    set_tag(ifds, i, 35596, 4, ys0)        # Y начала (левый, = глубина)
    set_tag(ifds, i, 35598, 4, xs1)        # X конца (правый)
    set_tag(ifds, i, 35600, 4, ys1)        # Y конца (правый, наклон)
    set_tag(ifds, i, 35601, 4, [3] * n)    # стиль/вес линии
    set_tag(ifds, i, 35590, 4, n)          # N линий
    set_tag(ifds, i, 35586, 4, int(round(step * pxm)))  # px на шаг
    for t in (35578, 35582, 35584):
        set_tag(ifds, i, t, 12, float(step))            # шаг (м)
    return depths, ys0


def _resid(ys, ref):
    """median/within-N резидуал ys к ближайшей линии ref (для валидации vs экспертная сетка)."""
    if ref is None or not len(ref):
        return None
    ys = np.asarray(ys, float); ref = np.asarray(ref, float)
    d = np.abs(ys[:, None] - ref[None, :]).min(1)
    return (float(np.median(d)), float(np.mean(d <= 2) * 100),
            float(np.mean(d <= 3) * 100), float(np.mean(d <= 5) * 100))


def main():
    a = sys.argv[1:]
    nlgx = a[a.index("--nlgx")+1]
    step = float(a[a.index("--step")+1]) if "--step" in a else 4.0
    scan = a[a.index("--scan")+1] if "--scan" in a else None
    snap = "--snap" in a
    out = a[a.index("--out")+1] if "--out" in a else nlgx.replace(".nlgx", "_grid.nlgx")
    sys.stdout.reconfigure(encoding="utf-8")
    m = extract(nlgx)
    gt_ys = m.get("depth_grid", {}).get("ys", [])     # существующая (экспертная) сетка = GT
    ifds = read_full(open(nlgx, "rb").read())

    det_ys = None
    if snap or "--scan" in a:
        from detect_calibration import detect_hgrid
        from dataset_build import find_image
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        img = scan or find_image(Path(nlgx))
        if img and snap:
            import cv2
            g = cv2.cvtColor(np.asarray(Image.open(img).convert("RGB")), cv2.COLOR_RGB2GRAY)
            det_ys, fine = detect_hgrid(g)
            print(f"детект сетки: {len(det_ys)} линий, шаг тонкой ~{fine:.1f}px")

    res = regrid(ifds, m, step, det_ys=det_ys if snap else None)
    if res is None:
        print("нет тип-8 Depth Grid IFD — пропуск (нужно добавить IFD)"); return
    depths, ys = res
    if scan:
        for i in find_ifd(ifds, lambda tags: 34878 in tags):
            set_tag(ifds, i, 34878, 2, scan)
    data = write_full(ifds)
    open(out, "wb").write(data)
    # bck = точная копия (write_bck патчил байт внутри данных — анализ 02.07)
    open(Path(out).with_suffix(".bck"), "wb").write(data)
    m2 = extract(out)
    dg = m2["depth_grid"]
    print(f"Depth Grid: {dg['n']} линий, шаг {dg['step_m']} м")
    print(f"  глубины {depths[0]}..{depths[-1]} (кратные {step})")
    print(f"  ys {ys[:3]} … {ys[-3:]}")
    if gt_ys:
        r = _resid(ys, gt_ys)
        print(f"  vs экспертная сетка: med={r[0]:.1f}px within2={r[1]:.0f}% within3={r[2]:.0f}% within5={r[3]:.0f}%")
    print(f"-> {out} (+bck)")


if __name__ == "__main__":
    main()
