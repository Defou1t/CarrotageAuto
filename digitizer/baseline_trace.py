"""
baseline_trace.py — ВАРИАНТ А: эвристика-baseline трассировки.
Меряем КАЧЕСТВО АВТО-ТРАССЕРА в изоляции: калибровка и уровни перевыносов берутся
ОРАКУЛОМ из nlgx (идеальные), автоматическим остаётся только поиск пикселя кривой
по строкам. Два метрики:
  (1) пиксельная ошибка |tracer_x - expert_x| (чистая трассировка, без калибровки);
  (2) значения vs эталон LAS (через оракул-калибровку + level), corr/log_corr/RMSE.

Трассеры (самодостаточные, в духе Блока 2 analyze_log_image._track_band_line):
  • centroid   — среднее x тёмных пикселей в окне (усредняет → срезает пики/наложения);
  • continuity — ближайший «прогон» чернил к предыдущей строке (идёт по одной линии).

Допущение: окно поиска по X = протяжённость кривой ± запас («подсказка колонки»,
имитирует сегментацию/оператора — какая кривая в какой части трека). Без неё на
мультикривных треках (BK+MBK, BKZ) трассер путает соседние линии.

python baseline_trace.py [--well NAME] [--limit N] [--thr 120] [--margin 0.4]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from extract_nlgx import extract, depth_of, depth_axis_ok, NULL
import dataset as ds
from dataset_build import find_image, find_las

Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")


def trace_centroid(mask):
    """Среднее x тёмных пикселей по строкам окна (NaN если пусто)."""
    h, w = mask.shape
    cnt = mask.sum(1)
    cols = np.arange(w)
    s = (mask * cols).sum(1)
    out = np.full(h, np.nan)
    nz = cnt > 0
    out[nz] = s[nz] / cnt[nz]
    return out


def _row_runs(rowmask, min_gap=4, max_width_frac=0.5):
    idx = np.nonzero(rowmask)[0]
    if idx.size == 0:
        return []
    runs = []
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - prev > min_gap:
            runs.append(((start + prev) / 2.0, prev - start + 1)); start = v
        prev = v
    runs.append(((start + prev) / 2.0, prev - start + 1))
    lim = rowmask.shape[0] * max_width_frac
    return [rc for rc in runs if rc[1] < lim]


def trace_continuity(mask, max_jump_frac=0.30, reseed_gap=12):
    """Ближайший к предыдущей позиции прогон чернил (континуити-трекер)."""
    h, w = mask.shape
    max_jump = max(15.0, w * max_jump_frac)
    out = np.full(h, np.nan)
    prev = None; gap = 0
    runs_all = [_row_runs(mask[r]) for r in range(h)]
    for r in range(h):
        runs = runs_all[r]
        if not runs:
            gap += 1
            if gap >= reseed_gap:
                prev = None
            continue
        if prev is None:
            prev = min(runs, key=lambda rc: (rc[1], abs(rc[0] - w/2)))[0]
            out[r] = prev; gap = 0; continue
        cand = min(runs, key=lambda rc: abs(rc[0] - prev))[0]
        if abs(cand - prev) <= max_jump:
            out[r] = cand; prev = cand; gap = 0
        else:
            gap += 1
            if gap >= reseed_gap:
                prev = None
    return out


def family_of(model, curve):
    key = curve["name"].strip().split()[-1]
    return sorted([s for s in model["scale_axes"]
                   if s["name"].strip().split()[-1] == key], key=lambda s: s["idx"])


def level_fn(curve):
    segs = curve["segments"]
    def f(y):
        last = None
        for s, e, l in segs:
            if s <= y <= e:
                return l
            if e < y:
                last = l
        return last
    return f


def values_from_trace(model, curve, trace_x, top_y, fam, lvl):
    """trace_x (по строкам окна) -> (depth, value) через оракул-калибровку+level."""
    out = []
    for i, x in enumerate(trace_x):
        if np.isnan(x):
            continue
        y = top_y + i
        L = lvl(y)
        if L is None or L >= len(fam):
            continue
        sa = fam[L]
        v = sa["v_left"] + (x - sa["x_left"]) / (sa["x_right"] - sa["x_left"]) * (sa["v_right"] - sa["v_left"])
        out.append((depth_of(model, y), v))
    return out


def corr_stats(recon, depths, las_vals, null=-999.25):
    if len(recon) < 5:
        return None
    recon = sorted(recon)
    ds_ = np.array([d for d, _ in recon]); vs = np.array([v for _, v in recon])
    m = (depths >= ds_[0]) & (depths <= ds_[-1]) & (las_vals > null + 1)
    if m.sum() < 5:
        return None
    ri = np.interp(depths[m], ds_, vs); lv = las_vals[m]
    rmse = float(np.sqrt(np.mean((ri - lv) ** 2)))
    corr = float(np.corrcoef(ri, lv)[0, 1]) if ri.std() > 0 and lv.std() > 0 else 0.0
    ok = (ri > 0) & (lv > 0)
    lcorr = None
    if ok.sum() > 5:
        a, b = np.log10(ri[ok]), np.log10(lv[ok])
        if a.std() > 0 and b.std() > 0:
            lcorr = float(np.corrcoef(a, b)[0, 1])
    return {"corr": corr, "log_corr": lcorr, "rmse": rmse}


def process_curve(model, curve, gray, las_depths, las_col, thr, margin):
    H, W = gray.shape
    top_y = curve["top_y"]
    pts = [(i, x) for i, x in enumerate(curve["xs"]) if x != NULL]
    if len(pts) < 50:
        return None
    exp_rows = np.array([i for i, _ in pts]); exp_x = np.array([x for _, x in pts])
    x0 = max(0, int(exp_x.min() - margin * (exp_x.max() - exp_x.min()) - 20))
    x1 = min(W, int(exp_x.max() + margin * (exp_x.max() - exp_x.min()) + 20))
    y0 = top_y; y1 = min(H, top_y + len(curve["xs"]))
    sub = (gray[y0:y1, x0:x1] < thr)

    res = {"curve": curve["name"].split()[0], "n_expert": len(pts)}
    fam = family_of(model, curve); lvl = level_fn(curve)
    for name, fn in (("centroid", trace_centroid), ("continuity", trace_continuity)):
        tr = fn(sub)                      # x в координатах окна
        tr_img = tr + x0                  # -> координаты изображения
        # (1) пиксельная ошибка vs эксперт
        valid = ~np.isnan(tr_img[exp_rows])
        if valid.sum() >= 20:
            dxe = np.abs(tr_img[exp_rows][valid] - exp_x[valid])
            res[name] = {
                "px_median": float(np.median(dxe)), "px_mean": float(np.mean(dxe)),
                "within3": float(np.mean(dxe <= 3)), "within10": float(np.mean(dxe <= 10)),
                "cover": float(valid.mean()),
            }
        else:
            res[name] = {"px_median": None}
        # (2) значения vs LAS
        if las_depths is not None and las_col is not None and depth_axis_ok(model):
            recon = values_from_trace(model, curve, tr, top_y, fam, lvl)
            st = corr_stats(recon, las_depths, las_col)
            if st:
                res[name].update({"v_corr": st["corr"], "v_logcorr": st["log_corr"], "v_rmse": st["rmse"]})
    return res


def main():
    args = sys.argv[1:]
    well = None; limit = 9999; thr = 120; margin = 0.4
    i = 0
    while i < len(args):
        if args[i] == "--well": well = args[i+1]; i += 2
        elif args[i] == "--limit": limit = int(args[i+1]); i += 2
        elif args[i] == "--thr": thr = int(args[i+1]); i += 2
        elif args[i] == "--margin": margin = float(args[i+1]); i += 2
        else: i += 1
    sys.stdout.reconfigure(line_buffering=True)

    wells = sorted([d for d in ARCHIVE.iterdir() if d.is_dir()])
    if well: wells = [w for w in wells if w.name == well]
    wells = wells[:limit]

    agg = {"centroid": {"px": [], "logc": []}, "continuity": {"px": [], "logc": []}}
    print(f"{'curve':<10}{'ctr_px':>8}{'ctr_w3':>7}{'ctr_lc':>8}{'cnt_px':>8}{'cnt_w3':>7}{'cnt_lc':>8}  file")
    for wl in wells:
        wlg = wl / "wlg"
        if not wlg.is_dir(): continue
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem: continue
            try: m = extract(str(nlgx))
            except Exception: continue
            curves = ds.real_curves(m)
            if not curves: continue
            img = find_image(nlgx)
            if not img: continue
            gray = np.asarray(Image.open(img).convert("L"))
            las = find_las(nlgx)
            las_cols, las_arr = (ds.load_las(las) if las else (None, np.zeros((0, 0))))
            matches = ds.match_las(m, las_cols, las_arr) if las_arr.size else {}
            las_depths = las_arr[:, 0] if las_arr.size else None
            for c in curves:
                mm = matches.get(c["name"], {})
                col = las_arr[:, mm["col_idx"]] if mm.get("col_idx") is not None else None
                r = process_curve(m, c, gray, las_depths, col, thr, margin)
                if not r: continue
                ct = r.get("centroid", {}); cn = r.get("continuity", {})
                def g(d, k): return d.get(k)
                def f(v, p=1): return f"{v:.{p}f}" if isinstance(v, (int, float)) else "  -"
                print(f"{r['curve']:<10}{f(g(ct,'px_median')):>8}{f(g(ct,'within3'),2):>7}"
                      f"{f(g(ct,'v_logcorr'),2):>8}{f(g(cn,'px_median')):>8}{f(g(cn,'within3'),2):>7}"
                      f"{f(g(cn,'v_logcorr'),2):>8}  {nlgx.stem[:36]}")
                for nm in ("centroid", "continuity"):
                    d = r.get(nm, {})
                    if isinstance(d.get("px_median"), float): agg[nm]["px"].append(d["px_median"])
                    if isinstance(d.get("v_logcorr"), float): agg[nm]["logc"].append(d["v_logcorr"])

    print("\n=== AGG (variant A: oracle calib+levels, auto tracer only) ===")
    for nm in ("centroid", "continuity"):
        px = agg[nm]["px"]; lc = agg[nm]["logc"]
        if px:
            print(f"{nm:<11} pixel |dx| median-of-medians={np.median(px):.1f}px (n={len(px)}); "
                  f"value log_corr median={np.median(lc):.3f} (n={len(lc)})" if lc else
                  f"{nm:<11} pixel |dx| median={np.median(px):.1f}px (n={len(px)})")


if __name__ == "__main__":
    main()
