r"""
export_las.py — достраивает СДАЧУ: из _auto.nlgx (наша трасса + шкалы/уровни шаблона) пишет
LAS 2.0 (значения в реальных единицах) + валидирует против ЭТАЛОННОГО LAS (corr/log_corr).
End-to-end проверка пайплайна: трасса (форма) → значения (через scale-семейства + перевыносы).

Значение на глубине: уровень из сегментов (тег 35498, наследованы из шаблона) → шкала семейства
(v_left..v_right на x_left..x_right) → линейная интерполяция. Резистив → важен log_corr.

python export_las.py <auto.nlgx | папка> [--step 0.2] [--las ref.las]
Папка → батч (ищет эталонный LAS рядом в ../las/), CSV-сводка corr/log_corr.
"""
import sys, csv, math
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from validate_nlgx import family_of, load_las, resample, stats
import dataset as ds

NULLV = -999.25


def _recon(m, curve):
    """value(depth) из трассы + scale-семейства + уровней; depth робастно (top/bottom, не span_px)."""
    da = m["depth_axis"]
    if not da or (da["bottom_depth"] - da["top_depth"]) == 0 or (da["bottom_y"] - da["top_y"]) == 0:
        return []
    pxm = (da["bottom_y"] - da["top_y"]) / (da["bottom_depth"] - da["top_depth"])
    dof = lambda y: da["top_depth"] + (y - da["top_y"]) / pxm
    fam = family_of(m, curve["name"])
    if not fam:
        return []
    segs = curve["segments"]; ty = curve["top_y"]

    def level_at(y):
        last = None
        for s, e, l in segs:
            if s <= y <= e:
                return l
            if e < y:
                last = l
        return last if last is not None else 0

    out = []
    for i, x in enumerate(curve["xs"]):
        if x == NULL:
            continue
        L = level_at(ty + i)
        if L is None or L >= len(fam):
            continue
        sa = fam[L]
        if sa["x_right"] == sa["x_left"]:
            continue
        frac = (x - sa["x_left"]) / (sa["x_right"] - sa["x_left"])
        out.append((dof(ty + i), sa["v_left"] + frac * (sa["v_right"] - sa["v_left"])))
    return out


def _mnem(name):
    import re
    return re.sub(r"\d+$", "", name.split()[0])


def _units(m, curve):
    fam = family_of(m, curve["name"])
    return (fam[0]["units"] if fam else "") or ""


def export(nlgx, step=0.2, ref_las=None, verbose=True):
    m = extract(nlgx)
    curves = ds.real_curves(m)
    cols = []           # (mnem, units, {depth:value})
    seen = {}
    for c in curves:
        rec = _recon(m, c)
        if len(rec) < 5:
            continue
        mn = _mnem(c["name"]); seen[mn] = seen.get(mn, 0) + 1
        nm = mn if seen[mn] == 1 else f"{mn}{seen[mn]}"
        cols.append((nm, _units(m, c), dict((round(d, 3), v) for d, v in rec)))
    if not cols:
        if verbose: print(f"  {Path(nlgx).stem[:46]}: нет восстановимых кривых");
        return None
    dmin = min(min(d for d in col[2]) for col in cols)
    dmax = max(max(d for d in col[2]) for col in cols)
    grid = np.round(np.arange(dmin, dmax + step / 2, step), 3)
    series = []
    for nm, un, dv in cols:
        rec = sorted(dv.items())
        series.append((nm, un, resample(rec, list(grid))))

    out = Path(nlgx).with_name(Path(nlgx).stem + ".las")
    with open(out, "w", encoding="latin1") as f:
        f.write("~Version\n VERS. 2.0 :\n WRAP. NO :\n")
        f.write(f"~Well\n STRT.M {dmin:.3f} :\n STOP.M {dmax:.3f} :\n STEP.M {step} :\n"
                f" NULL. {NULLV} :\n WELL. {m.get('well','')} :\n")
        f.write("~Curve\n DEPT.M :\n")
        for nm, un, _ in series:
            f.write(f" {nm}.{un} :\n")
        f.write("~ASCII\n")
        for k, d in enumerate(grid):
            row = [f"{d:.3f}"] + [f"{(s[2][k] if s[2][k] is not None else NULLV):.4g}" for s in series]
            f.write(" ".join(row) + "\n")

    res = {"stem": Path(nlgx).stem, "curves": len(series), "las": str(out)}
    if ref_las and Path(ref_las).is_file():
        rcols, rrows = load_las(ref_las)
        rdep = [r[0] for r in rrows]
        line = []
        for nm, un, ser in series:
            j = next((i for i, cn in enumerate(rcols) if _mnem(cn).upper() == _mnem(nm).upper()), None)
            if j is None:
                continue
            ours = resample(sorted({grid[k]: ser[k] for k in range(len(grid)) if ser[k] is not None}.items()), rdep)
            st = stats([rw[j] for rw in rrows], ours)
            if st:
                line.append((nm, st["corr"], st["log_corr"]))
                res[f"corr_{nm}"] = round(st["corr"], 3)
                res[f"logcorr_{nm}"] = round(st["log_corr"], 3) if st["log_corr"] is not None else None
        if verbose and line:
            print(f"  {Path(nlgx).stem[:42]:<42} vs эталон: " +
                  ", ".join(f"{nm} corr={c:.2f}" + (f"/log={lc:.2f}" if lc is not None else "") for nm, c, lc in line))
    elif verbose:
        print(f"  {Path(nlgx).stem[:42]:<42} -> LAS ({len(series)} кривых)")
    return res


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    target = a[0]
    step = float(a[a.index("--step") + 1]) if "--step" in a else 0.2
    ref = a[a.index("--las") + 1] if "--las" in a else None
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = Path(target)
    if p.is_dir():
        files = sorted(f for f in p.glob("*_auto.nlgx"))
        rows = []
        for f in files:
            # эталонный LAS: по имени без _auto в соседней las/ проекта (если структура проектная)
            stem = f.stem.replace("_auto", "")
            cand = list(Path(r"F:\nds\projects").glob(f"*/las/{stem}.las"))
            r = export(f, step, str(cand[0]) if cand else None)
            if r: rows.append(r)
        if rows:
            keys = sorted({k for r in rows for k in r})
            csvp = Path(r"F:\nds\output") / "export_las_summary.csv"
            with open(csvp, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
            cors = [v for r in rows for k, v in r.items() if k.startswith("logcorr_") and v is not None]
            print(f"\nГОТОВО: {len(rows)} файлов; медиана log_corr vs эталон="
                  f"{np.median(cors):.2f}" if cors else f"\nГОТОВО: {len(rows)} файлов")
            print(f"CSV -> {csvp}")
    else:
        export(target, step, ref)


if __name__ == "__main__":
    main()
