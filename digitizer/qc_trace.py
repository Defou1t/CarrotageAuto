r"""
qc_trace.py — НЕЗАВИСИМАЯ аналитическая QC оцифровки: проверяет трассу _auto.nlgx против
ИЗОБРАЖЕНИЯ (не против шаблона!), поэтому ловит ошибки самого оцифровщика. Аналог NeuraLog
Digitizing Quality Index / Curve Overlay / Spike / Gap Check, но глубже и по малозависимым
сигналам. Для каждой кривой:
  • off_ink   — доля точек трассы НЕ на черниле своего цвета (мосты, галлюцинации, заход в шапку);
  • spikes    — резкие скачки x НЕ на границе перевыноса (ложные пики/изломы);
  • gap_m     — суммарные разрывы внутри интервала линии (метры);
  • oof       — точки в зоне A3-маски (текст/линейка/сетка — там линии быть не должно);
  • overlap   — самый длинный участок (м), где кривая идёт вплотную к ДРУГОЙ (потеря идентичности).
Флаги → список подозрительных глубин (для клика) + опц. overlay. Score 0..100 (выше — чище).

python qc_trace.py <auto.nlgx | папка> [--image f.jpg] [--overlay]
"""
import sys, csv
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL, depth_of, depth_axis_ok
from dataset_build import find_image
import dataset as ds
import detect_masks as dm
import extract_instances as ei


def _scan(m, nlgx):
    p = m.get("img_path")
    if p and Path(p).is_file():
        return Path(p)
    return find_image(Path(nlgx))


def _trace(c, H):
    ty = c["top_y"]
    return {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL and 0 <= ty + i < H}


def _level_bnds(c):
    """y-границы смены уровня (перевынос) — там скачок x ОЖИДАЕМ."""
    bs = []
    for (s, e, l), (s2, e2, l2) in zip(c["segments"], c["segments"][1:]):
        bs.append(e)
    return bs


def qc_curve(m, c, gray, chan, exclude, dpx, dof, tol_x=5, vwin=1, spike_thr=45, gap_m=2.0):
    H, W = gray.shape
    tr = _trace(c, H)
    if len(tr) < 30:
        return None
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys])
    n = len(ys)
    # off-ink (векторно): ink своего цвета в окне ±tol_x/±vwin вокруг точки
    ink = (chan > 0) if chan is not None else (gray < 150)
    ink = ink.astype(np.uint8)
    dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tol_x + 1, 2 * vwin + 1)))
    on = dil[np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)] > 0
    off = int((~on).sum())
    # spikes: ИЗОЛИРОВАННЫЙ выброс «туда-обратно» на 1-2 строки (артефакт), а НЕ устойчивый
    # острый пик (реальный пик идёт монотонно и не возвращается сразу). Точка — локальный
    # экстремум, торчащий >thr на ОБЕ стороны. Не у границы перевыноса.
    bnd = _level_bnds(c)
    dy = np.diff(ys)
    spike_ys = []
    for i in range(1, n - 1):
        if ys[i] - ys[i - 1] > 3 or ys[i + 1] - ys[i] > 3:
            continue
        d1 = xs[i] - xs[i - 1]; d2 = xs[i] - xs[i + 1]
        if d1 * d2 > 0 and min(abs(d1), abs(d2)) > spike_thr:
            y = int(ys[i])
            if not any(abs(y - b) < dpx * 0.5 for b in bnd):
                spike_ys.append(y)
    # gaps внутри [start,end]
    gthr = int(gap_m * dpx)
    gap_rows = int(dy[dy > gthr].sum()) if len(dy) else 0
    # out-of-frame: точка в no-line зоне (ruler/frame/text) И НЕ на черниле своего цвета.
    # «И не на черниле» отсекает ложные A3-срабатывания (кривую, помеченную как текст на 1:500):
    # если точка на реальном черниле — это кривая, а не стрэй.
    in_noline = exclude[np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)] > 0
    oof = int((in_noline & ~on).sum())
    return {"name": c["name"].split()[0], "n": n,
            "off_ink_pct": round(off / n * 100, 1), "spikes": len(spike_ys),
            "gap_m": round(gap_rows / dpx, 1), "oof_pct": round(oof / n * 100, 1),
            "start_d": round(dof(int(ys[0])), 1), "end_d": round(dof(int(ys[-1])), 1),
            "spike_d": [round(dof(y), 1) for y in spike_ys[:15]], "_tr": tr}


def overlap_check(curves_qc, dpx, tol=4, lmin_m=3.0):
    """Самый длинный участок (м), где кривая вплотную (|Δx|<tol) к ДРУГОЙ — потеря идентичности."""
    names = [q["name"] for q in curves_qc]
    for q in curves_qc:
        best_run, best_p = 0, "-"
        for q2 in curves_qc:
            if q2 is q:
                continue
            common = sorted(set(q["_tr"]) & set(q2["_tr"]))
            run = mx = 0; prev = None
            for y in common:
                if abs(q["_tr"][y] - q2["_tr"][y]) < tol:
                    run = run + (y - prev) if (prev is not None and y - prev < 5) else 0
                    mx = max(mx, run)
                else:
                    run = 0
                prev = y
            if mx > best_run:
                best_run, best_p = mx, q2["name"]
        q["overlap_m"] = round(best_run / dpx, 1)
        q["overlap_with"] = best_p if best_run / dpx >= lmin_m else "-"


def _score(q):
    """0..100: штрафы за off-ink, spikes, gaps, oof, overlap."""
    s = 100.0
    s -= min(40, q["off_ink_pct"] * 1.5)
    s -= min(15, q["spikes"] * 1.5)
    s -= min(15, q["gap_m"] * 0.5)
    s -= min(20, q["oof_pct"] * 2.0)
    s -= min(15, q.get("overlap_m", 0) * 0.5)
    return round(max(0, s))


def qc_file(nlgx, image=None, overlay=False, verbose=True):
    m = extract(nlgx)
    da = m.get("depth_axis") or {}
    dy = (da.get("bottom_y") or 0) - (da.get("top_y") or 0)
    dd = (da.get("bottom_depth") or 0) - (da.get("top_depth") or 0)
    if not dy or not dd:                       # ось вырождена (в т.ч. span_px=0) — берём top/bottom
        if verbose: print(f"  {Path(nlgx).stem[:46]}: ось глубины вырождена — пропуск")
        return []
    dpx = dy / dd
    def dof(y): return da["top_depth"] + (y - da["top_y"]) / dpx
    img = Path(image) if image else _scan(m, nlgx)
    if not img or not Path(img).is_file():
        if verbose: print(f"  {Path(nlgx).stem[:46]}: нет картинки — пропуск")
        return []
    rgb = np.asarray(Image.open(img).convert("RGB"))
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    masks = dm.compute_masks(rgb, m)
    g = masks["geom"]; xl = max(0, g["x_left"] - 5); xr = min(W, (g["x_right"] or W) + 30)
    # oof / off_ink — только зоны, где линии БЫТЬ НЕ ДОЛЖНО (полоса/рамка/текст), НЕ грид
    # (кривая законно пересекает грид-линии). Цветные маски считаем по тому же body (грид
    # оставлен) — согласованно с gap-fill digitize_b3; чёрный — минус весь exclude (грид чёрный).
    noline = ((masks["ruler"] | masks["frame"] | masks["text"]) > 0).astype(np.uint8)
    body = np.zeros((H, W), np.uint8); body[g["top_y"]:g["bottom_y"], xl:xr] = 1
    chans = ei.classify_ink(rgb, body)                       # ПОЛНОЕ body (как в gap-fill)
    chans["black"] = (chans["black"] & (masks["exclude"] == 0)).astype(np.uint8)
    out = []
    for c in ds.real_curves(m):
        gtc = ei._gt_color(rgb, _trace(c, H))
        q = qc_curve(m, c, gray, chans.get(gtc), noline, dpx, dof)
        if q:
            q["color"] = gtc; q["file"] = Path(nlgx).stem
            out.append(q)
    overlap_check(out, dpx)
    for q in out:
        q["score"] = _score(q)
    if verbose and out:
        print(f"\n{Path(nlgx).stem[:50]}")
        print(f"  {'curve':<6}{'цвет':<6}{'score':>6}{'off_ink':>8}{'spikes':>7}{'gap_m':>7}{'oof%':>6}{'overlap':>9}")
        for q in sorted(out, key=lambda q: q["score"]):
            print(f"  {q['name']:<6}{q['color']:<6}{q['score']:>6}{q['off_ink_pct']:>7}%{q['spikes']:>7}"
                  f"{q['gap_m']:>7}{q['oof_pct']:>5}%  {q['overlap_with']:>5}/{q['overlap_m']}м")
            if q["spike_d"]:
                print(f"         спайки@ {q['spike_d']}")
    if overlay:
        _overlay(rgb, m, out, gray, chans, masks, Path(nlgx).stem[:30])
    for q in out:
        q.pop("_tr", None)
    return out


def _overlay(rgb, m, qc, gray, chans, masks, stem):
    """Подсветка подозрительных точек: off-ink (красный), oof (жёлтый), spike (синий круг)."""
    H, W = rgb.shape[:2]; ov = rgb.copy()
    for c in ds.real_curves(m):
        tr = _trace(c, H)
        if len(tr) < 30:
            continue
        gtc = ei._gt_color(rgb, tr); chan = chans.get(gtc)
        ink = ((chan > 0) if chan is not None else (gray < 150)).astype(np.uint8)
        dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)))
        for y, x in tr.items():
            if not (0 <= y < H and 0 <= x < W):
                continue
            if masks["exclude"][y, x] > 0:
                ov[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = [255, 230, 0]      # oof жёлтый
            elif dil[y, x] == 0:
                ov[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = [255, 0, 0]        # off-ink красный
    Image.fromarray(ov).resize((W // 5, H // 5), Image.LANCZOS).save(rf"F:\nds\output\qc_{stem}.png")
    print(f"  overlay -> qc_{stem}.png")


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    target = a[0]
    image = a[a.index("--image") + 1] if "--image" in a else None
    overlay = "--overlay" in a
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = Path(target)
    files = sorted(f for f in p.glob("*_auto.nlgx") if "framed" not in f.stem) if p.is_dir() else [p]
    if p.is_dir() and not files:
        files = sorted(f for f in p.glob("*.nlgx") if "_auto" not in f.stem and "framed" not in f.stem)
    allq = []
    for f in files:
        allq += qc_file(f, image if not p.is_dir() else None, overlay)
    if allq:
        sc = [q["score"] for q in allq]
        print(f"\n=== ИТОГО {len(allq)} кривых: score медиана={np.median(sc):.0f} "
              f"мин={min(sc)} | off_ink медиана={np.median([q['off_ink_pct'] for q in allq]):.1f}% "
              f"| спайков всего={sum(q['spikes'] for q in allq)} ===")
        if p.is_dir():
            csvp = Path(r"F:\nds\output") / "qc_trace_summary.csv"
            with open(csvp, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=[k for k in allq[0] if k != "spike_d"])
                w.writeheader()
                for q in allq:
                    w.writerow({k: v for k, v in q.items() if k != "spike_d"})
            print(f"CSV -> {csvp}")


if __name__ == "__main__":
    main()
