r"""
behavior_priors.py — B2 (Фаза B роадмапа §6.6.7): ПОВЕДЕНЧЕСКИЕ приоры линий по ground-truth.
Количественно характеризует «SP гладкая / CALI/резистив пиковая» (правило §6.6.7) — признак
ИДЕНТИЧНОСТИ для трекера B3: линия не должна прыгать на чужую с другим поведением (баг QC:
«SP переходит на CALI и доцифровывает её»; «GZ5 прыгает на тонкую GZ4»).

Метрика ИНВАРИАНТНА к плотности экспертных кликов: трасса x(row) интерполируется на равномерную
сетку строк (1px), затем:
  • rough_n  = std(x − сглаженная_1.5м) / x-span   — амплитуда ВЧ-дрожи (гладкость);
  • rev      = доля разворотов направления на сетке ~0.3 м (пиковость).
SP: rough_n≈0.01, rev≈0.2 (гладкая). Резистив/CALI: rough_n>0.03, rev>0.3 (пиковая).

python behavior_priors.py [--limit N]   # по всему Archive, сводка порогов по классам
"""
import sys, re
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds

ARCHIVE = Path(r"F:\nds\projects\Archive")

# класс по мнемонике (для порогов поведения)
CALI = {"DS", "MDS", "DN", "DC"}                       # каверномер (м), пиковая
RES = {"GZ", "PZ", "BK", "MBK", "MGZ", "MPZ", "OGZ", "REZ", "IK", "IKA", "IKR"}  # резистив, пиковая


def mnem(name):
    return re.sub(r"\d+$", "", name.split()[0])


def curve_class(name):
    mm = mnem(name).upper()
    if mm == "SP":
        return "SP"
    if mm in CALI:
        return "CALI"
    if mm in RES:
        return "RES"
    return "OTHER"


def curve_behavior(m, curve, smooth_m=1.5, rev_m=0.3):
    """Поведенческие фичи кривой из пиксель-трассы (инвариантно к плотности кликов).
    Возвращает dict(n, x_span, depth_m, rough, rough_n, rev) или None если мало точек."""
    ty = curve["top_y"]
    pts = [(ty + i, x) for i, x in enumerate(curve["xs"]) if x != NULL]
    if len(pts) < 30:
        return None
    ys = np.array([y for y, _ in pts]); xs = np.array([x for _, x in pts], float)
    da = m["depth_axis"]
    pxm = (da["bottom_y"] - da["top_y"]) / (da["bottom_depth"] - da["top_depth"]) or 1.0
    grid = np.arange(ys.min(), ys.max() + 1)            # равномерная сетка строк (1px)
    xi = np.interp(grid, ys, xs)
    W = max(5, int(round(smooth_m * pxm)) | 1)          # окно сглаживания (нечётное)
    sm = np.convolve(xi, np.ones(W) / W, mode="same")
    hf = (xi - sm)[W:-W] if len(xi) > 2 * W else (xi - sm)
    span = float(np.percentile(xs, 97) - np.percentile(xs, 3)) or 1.0
    rough = float(np.std(hf)) if len(hf) else 0.0
    s = max(1, int(rev_m * pxm)); xd = xi[::s]; dx = np.diff(xd)
    rev = float(np.mean((dx[:-1] * dx[1:]) < 0)) if len(dx) > 2 else 0.0
    return {"n": len(pts), "x_span": round(span, 1), "depth_m": round(len(grid) / pxm, 1),
            "rough": round(rough, 1), "rough_n": round(rough / span, 4), "rev": round(rev, 2)}


def _summ(vals):
    a = np.array(vals, float)
    return f"med={np.median(a):.3f} [q1={np.percentile(a,25):.3f} q3={np.percentile(a,75):.3f}] n={len(a)}"


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    sys.stdout.reconfigure(encoding="utf-8")
    files = []
    for wlg in sorted(ARCHIVE.glob("*/wlg")):
        for n in sorted(wlg.glob("*.nlgx")):
            if "_auto" not in n.stem:
                files.append(n)
    if limit:
        files = files[:limit]

    by_cls = defaultdict(lambda: {"rough_n": [], "rev": [], "x_span": []})
    err = 0
    for n in files:
        try:
            m = extract(str(n))
        except Exception:
            err += 1; continue
        for c in ds.real_curves(m):
            f = curve_behavior(m, c)
            if not f:
                continue
            cl = curve_class(c["name"])
            by_cls[cl]["rough_n"].append(f["rough_n"])
            by_cls[cl]["rev"].append(f["rev"])
            by_cls[cl]["x_span"].append(f["x_span"])

    print(f"файлов={len(files)} (ошибок extract={err})\n")
    print(f"{'класс':<7} {'кривых':>6}   rough_n (гладкость↓)            rev (пиковость↑)")
    for cl in ("SP", "CALI", "RES", "OTHER"):
        d = by_cls.get(cl)
        if not d or not d["rough_n"]:
            continue
        print(f"{cl:<7} {len(d['rough_n']):>6}   {_summ(d['rough_n']):<32} {_summ(d['rev'])}")
    # разделимость SP vs резистив/CALI по rough_n
    sp = np.array(by_cls['SP']['rough_n']) if by_cls['SP']['rough_n'] else np.array([])
    peaky = np.array(by_cls['RES']['rough_n'] + by_cls['CALI']['rough_n'])
    if len(sp) and len(peaky):
        thr = (np.percentile(sp, 90) + np.percentile(peaky, 10)) / 2
        sp_ok = np.mean(sp < thr) * 100; pk_ok = np.mean(peaky >= thr) * 100
        print(f"\nПОРОГ rough_n≈{thr:.3f}: SP<порог {sp_ok:.0f}% | резистив/CALI≥порог {pk_ok:.0f}% "
              f"(разделимость SP↔пиковые)")


if __name__ == "__main__":
    main()
