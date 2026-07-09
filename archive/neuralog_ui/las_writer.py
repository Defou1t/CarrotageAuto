"""
las_writer.py - генерация LAS 2.0 из результата Блока 2 (analyze_log_image).

Берёт result["curves"][i]["trace"]["samples"] ([depth, value]) каждой кривой,
ресемплит на регулярную сетку глубин и пишет валидный LAS 2.0.

Использование:
    from las_writer import write_las
    write_las(result, r"F:\\nds\\output\\well_BK.las", step=0.1)

CLI:
    python las_writer.py <image|analysis.json> [--out file.las] [--step 0.1]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

NULL_VALUE = -999.25


def _curve_samples(curve: dict) -> list[tuple[float, float]]:
    trace = curve.get("trace") or {}
    out = []
    for item in trace.get("samples") or []:
        if len(item) != 2:
            continue
        d, v = item
        if d is None or v is None:
            continue
        out.append((float(d), float(v)))
    out.sort(key=lambda p: p[0])
    return out


def _interp(grid: list[float], samples: list[tuple[float, float]],
            extent: Optional[tuple[float, float]]) -> list[float]:
    """Линейная интерполяция на grid; вне фактического экстента кривой -> NULL."""
    if not samples:
        return [NULL_VALUE] * len(grid)
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    lo, hi = (extent or (xs[0], xs[-1]))
    res = []
    j = 0
    n = len(xs)
    for d in grid:
        if d < lo - 1e-6 or d > hi + 1e-6:
            res.append(NULL_VALUE)
            continue
        if d <= xs[0]:
            res.append(ys[0])
            continue
        if d >= xs[-1]:
            res.append(ys[-1])
            continue
        while j < n - 1 and xs[j + 1] < d:
            j += 1
        # локальный поиск (xs отсортирован, но grid монотонен -> j не сбрасываем)
        k = j
        while k < n - 1 and xs[k + 1] < d:
            k += 1
        x0, x1 = xs[k], xs[k + 1]
        y0, y1 = ys[k], ys[k + 1]
        t = (d - x0) / (x1 - x0) if x1 > x0 else 0.0
        res.append(round(y0 + t * (y1 - y0), 4))
    return res


def write_las(result: dict, out_path: str, step: float = 0.1,
              well: str = "", field: str = "") -> str:
    curves = [c for c in result.get("curves", []) if _curve_samples(c)]
    if not curves:
        raise ValueError("Нет кривых с trace.samples для записи LAS")

    cols = {}
    extents = {}
    lo_all, hi_all = None, None
    for c in curves:
        s = _curve_samples(c)
        cols[c["name"]] = s
        ext = (c.get("depth_start"), c.get("depth_end"))
        if ext[0] is None or ext[1] is None:
            ext = (s[0][0], s[-1][0])
        ext = (min(ext), max(ext))
        extents[c["name"]] = ext
        lo_all = ext[0] if lo_all is None else min(lo_all, ext[0])
        hi_all = ext[1] if hi_all is None else max(hi_all, ext[1])

    # Сетка глубин (округляем к шагу)
    start = round(round(lo_all / step) * step, 4)
    stop = round(round(hi_all / step) * step, 4)
    n = int(round((stop - start) / step)) + 1
    grid = [round(start + i * step, 4) for i in range(n)]

    data = {name: _interp(grid, cols[name], extents[name]) for name in cols}

    names = list(cols.keys())
    img = result.get("image", {})
    well = well or result.get("well_name", "") or Path(img.get("path", "")).stem
    field = field or result.get("field", "")

    lines = []
    a = lines.append
    a("~Version Information Block")
    a(" VERS.                2.00: CWLS LOG ASCII STANDARD - VERSION 2.0")
    a(" WRAP.                  NO: One Line Per Depth Step")
    a("#")
    a("~Well Information Block")
    a("#MNEM.UNIT       Data            Information")
    a("#----------   --------------   --------------")
    a(f" STRT.M       {start:>14.4f}: START DEPTH")
    a(f" STOP.M       {stop:>14.4f}: STOP DEPTH")
    a(f" STEP.M       {step:>14.4f}: STEP")
    a(f" NULL.        {NULL_VALUE:>14.4f}: NULL VALUE")
    a(f" WELL.        {well:>14}: WELL")
    a(f" FLD .        {field:>14}: FIELD")
    a(" SRVC.        NeuraLOG-auto  : SERVICE (Block2 tracer)")
    a("#")
    a("~Curve Information Block")
    a("#MNEM.UNIT      API CODE   Curve Description")
    a("#----------   ----------  -------------------")
    a(" DEPT.M                  : Depth in Meters")
    for c in curves:
        unit = c.get("unit") or ""
        a(f" {c['name']:<4}.{unit:<8}        : {c.get('scale_type','linear')} {c.get('scale_min')}-{c.get('scale_max')}")
    a("#")
    a("~Parameter Information Block")
    a(f" SRC .        Block2: {result.get('source','local')}")
    a(f" CONF.        {result.get('confidence',0)}: pipeline confidence")
    a("#")
    a("~A  DEPTH     " + "  ".join(f"{nm:>12}" for nm in names))
    for i, d in enumerate(grid):
        row = [f"{d:>10.4f}"] + [f"{data[nm][i]:>12.4f}" for nm in names]
        a(" ".join(row))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Генерация LAS из результата Блока 2")
    parser.add_argument("analysis", help="*_analysis.json или путь к изображению")
    parser.add_argument("--out", default=None)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()

    path = Path(args.analysis)
    if path.suffix.lower() != ".json":
        path = path.with_name(path.stem + "_analysis.json")
    result = json.loads(path.read_text(encoding="utf-8"))

    out = args.out or str(Path(r"F:\nds\output") / (path.stem.replace("_analysis", "") + "_auto.las"))
    written = write_las(result, out, step=args.step)
    print("LAS:", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
