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


def regrid(ifds, m, step=4.0):
    da = m["depth_axis"]
    ty, by = da["top_y"], da["bottom_y"]
    td, bd = da["top_depth"], da["bottom_depth"]
    lo, hi = min(td, bd), max(td, bd)
    import math
    d0 = math.ceil(lo / step) * step
    depths = []
    d = d0
    while d <= hi + 1e-6:
        depths.append(round(d, 3)); d += step
    ys = [int(round(ty + (dd - td) * (by - ty) / (bd - td))) for dd in depths]
    idxs = find_ifd(ifds, lambda tags: 34768 in tags and
                    struct.unpack("<I", tags[34768][2][:4])[0] == 8)
    if not idxs:
        return None  # тип-8 IFD отсутствует — добавление нового IFD пока не делаем
    i = idxs[0]
    set_tag(ifds, i, 35596, 4, ys)       # Y-координаты горизонталей (LONG[])
    set_tag(ifds, i, 35590, 4, len(ys))  # N линий
    set_tag(ifds, i, 35578, 12, float(step))  # шаг (м, DOUBLE)
    return depths, ys


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
