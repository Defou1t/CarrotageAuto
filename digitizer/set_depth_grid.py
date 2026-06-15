r"""
set_depth_grid.py — перегенерировать Depth Grid (тип-8 IFD) на заданный шаг (по умолчанию
4 м для 1:200, канон §4) поверх калибровки Depth Axis. Линии ставятся на КРАТНЫЕ шагу
глубины (…3632, 3636…), как грани рамки в NeuraLOG (НЕ на рукописные подписи). Опционально
патчит скан-путь (тег 34878) на реальный — для QC. Пишет nlgx (+bck).

python set_depth_grid.py --nlgx <in.nlgx> [--step 4] [--scan <jpg>] [--out <out.nlgx>]
"""
import sys, struct
from pathlib import Path
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd
from extract_nlgx import extract


def _tagval(ifd, tag, default=None):
    for t, typ, count, raw in ifd["entries"]:
        if t == tag:
            fmt = {3: "H", 4: "I", 12: "d"}.get(typ, "I")
            return struct.unpack("<" + fmt, raw[:struct.calcsize(fmt)])[0]
    return default


def regrid(ifds, m, step=4.0):
    """Перегенерировать Depth Grid на шаг `step` (м), КОРРЕКТНО с геометрией наклонённых
    сегментов. Каждая линия = сегмент (x_start,y_start)->(x_end,y_end), наклон сохраняется из
    тега 35570. Все per-line массивы (35594/35596/35598/35600/35601) пишутся согласованной длины,
    иначе NeuraLOG рисует неверный угол/веер (баг наивного регрида)."""
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
    depths, xs0, ys0, xs1, ys1 = [], [], [], [], []
    d = d0
    while d <= hi + 1e-6:
        xs = int(round(xt + (d - td) * xpm))           # левый X (по наклону оси)
        ys = int(round(ty + (d - td) * pxm))           # левый Y (глубина)
        depths.append(round(d, 3))
        xs0.append(xs); ys0.append(ys)
        xs1.append(xs + int(width))                    # правый X
        ys1.append(int(round(ys + slope * width)))     # правый Y (наклон горизонтали)
        d += step
    n = len(depths)
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


def main():
    a = sys.argv[1:]
    nlgx = a[a.index("--nlgx")+1]
    step = float(a[a.index("--step")+1]) if "--step" in a else 4.0
    scan = a[a.index("--scan")+1] if "--scan" in a else None
    out = a[a.index("--out")+1] if "--out" in a else nlgx.replace(".nlgx", "_grid.nlgx")
    sys.stdout.reconfigure(encoding="utf-8")
    m = extract(nlgx)
    ifds = read_full(open(nlgx, "rb").read())
    res = regrid(ifds, m, step)
    if res is None:
        print("нет тип-8 Depth Grid IFD — пропуск (нужно добавить IFD)"); return
    depths, ys = res
    if scan:
        for i in find_ifd(ifds, lambda tags: 34878 in tags):
            set_tag(ifds, i, 34878, 2, scan)
    data = write_full(ifds)
    open(out, "wb").write(data)
    open(Path(out).with_suffix(".bck"), "wb").write(write_bck(data))
    m2 = extract(out)
    dg = m2["depth_grid"]
    print(f"Depth Grid: {dg['n']} линий, шаг {dg['step_m']} м")
    print(f"  глубины {depths[0]}..{depths[-1]} (кратные {step})")
    print(f"  ys {ys[:3]} … {ys[-3:]}")
    print(f"-> {out} (+bck)")


if __name__ == "__main__":
    main()
