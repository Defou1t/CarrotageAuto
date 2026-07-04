"""
Этап A плана 95-99% (04.07): ПРОВЕРКА гипотезы «D1/D2 = два независимых прохода
эксперта по одному интервалу» на BKZ-парах Yatskivska_001 + Pn_Zavoda_001.

ВЕРДИКТ (04.07, durable): гипотеза НЕВЕРНА. Шапки сканов (та же скважина, тот же
интервал, та же дата 2.02.94, тот же оператор) показывают РАЗНЫЕ списки зондов:
D1 = A0.4M0.1N / A1.0M0.1N / A2.0M0.5N / ПС (+MDS), D2 = A4.0M0.5N / N0.5M2.0A /
A8.0M1.0N. Это одна БКЗ-серия, разложенная на два бланка — НЕ дубли. Найденные
скриптом матчи GZ31<->GZ41 corr 0.92-0.97 (median|dx| экв. ~20px) — корреляция
СОСЕДНИХ зондов по литологии, НЕ повтор одной кривой. Потолок эксперт-vs-эксперт
по эталону не измерить; вместо него — expert_ink_jitter.py (эксперт-vs-чернила).

Механика (оставлена для переиспользования, напр. кросс-валидация зондов):
сканы D1/D2 разные (px/m различаются) -> сравнение только в глубине/значении
через собственные оси; матчинг кривых по ФОРМЕ (корр. resampled value(depth)),
т.к. мнемоники между бланками не совпадают.
"""
import sys, os, glob, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, depth_axis_ok
from dataset import real_curves, _recon_curve

ARCHIVE = r"F:\nds\projects\Archive"
WELLS = ["Yatskivska_001", "Pn_Zavoda_001"]
OUT_DIR = r"F:\nds\output\expert_dup_ceiling"
MIN_CORR = 0.6          # ниже — не считаем той же физической кривой
MIN_OVERLAP_M = 3.0     # минимум перекрытия по глубине для попытки матчинга
STEP_M = 0.05           # шаг ресемплинга общей сетки глубин


def find_pairs():
    pairs = []
    for well in WELLS:
        wlg = os.path.join(ARCHIVE, well, "wlg")
        for f1 in sorted(glob.glob(os.path.join(wlg, "*_D1.nlgx"))):
            f2 = f1[:-len("D1.nlgx")] + "D2.nlgx"
            if os.path.exists(f2):
                pairs.append((well, f1, f2))
    return pairs


def curve_vrange(model, curve):
    """Наибольший |v_right-v_left| среди осей семейства кривой — для нормировки ошибки."""
    key = curve["name"].strip().split()[-1]
    fam = [s for s in model["scale_axes"] if s["name"].strip().split()[-1] == key]
    if not fam:
        return None
    vals = [abs(s["v_right"] - s["v_left"]) for s in fam]
    return max(vals) if vals else None


def resample_curve(model, curve, grid):
    recon = _recon_curve(model, curve)
    if len(recon) < 5:
        return np.full(len(grid), np.nan)
    recon = sorted(recon)
    ds = np.array([d for d, _ in recon], dtype=float)
    vs = np.array([v for _, v in recon], dtype=float)
    ds, idx = np.unique(ds, return_index=True)
    vs = vs[idx]
    return np.interp(grid, ds, vs, left=np.nan, right=np.nan)


def analyze_pair(well, f1, f2):
    m1, m2 = extract(f1), extract(f2)
    if not (depth_axis_ok(m1) and depth_axis_ok(m2)):
        return None, "depth axis missing/degenerate"
    da1, da2 = m1["depth_axis"], m2["depth_axis"]
    lo = max(min(da1["top_depth"], da1["bottom_depth"]), min(da2["top_depth"], da2["bottom_depth"]))
    hi = min(max(da1["top_depth"], da1["bottom_depth"]), max(da2["top_depth"], da2["bottom_depth"]))
    if hi - lo < MIN_OVERLAP_M:
        return None, f"overlap too small ({hi-lo:.1f}m)"
    grid = np.arange(lo, hi, STEP_M)
    c1s, c2s = real_curves(m1), real_curves(m2)
    if not c1s or not c2s:
        return None, "no real curves on one side"
    r1 = [resample_curve(m1, c, grid) for c in c1s]
    r2 = [resample_curve(m2, c, grid) for c in c2s]

    min_pts = int(MIN_OVERLAP_M / STEP_M)
    cand = []
    for i, a in enumerate(r1):
        for j, b in enumerate(r2):
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() < min_pts or a[mask].std() == 0 or b[mask].std() == 0:
                continue
            c = float(np.corrcoef(a[mask], b[mask])[0, 1])
            cand.append((abs(c), c, i, j, int(mask.sum())))
    cand.sort(key=lambda t: -t[0])

    used1, used2, matches = set(), set(), []
    for absc, c, i, j, n in cand:
        if i in used1 or j in used2 or absc < MIN_CORR:
            continue
        used1.add(i); used2.add(j)
        a, b = r1[i], r2[j]
        mask = ~np.isnan(a) & ~np.isnan(b)
        diff = a[mask] - b[mask]
        vr = max([v for v in (curve_vrange(m1, c1s[i]), curve_vrange(m2, c2s[j])) if v], default=None)
        matches.append({
            "well": well, "plate": os.path.basename(f1),
            "curve_d1": c1s[i]["name"].strip(), "curve_d2": c2s[j]["name"].strip(),
            "corr": round(c, 4), "n_overlap": n,
            "median_abs_diff": round(float(np.median(np.abs(diff))), 4),
            "rmse": round(float(np.sqrt(np.mean(diff ** 2))), 4),
            "vrange": vr,
            "median_abs_diff_pct": round(float(np.median(np.abs(diff)) / vr * 100), 2) if vr else None,
        })
    unmatched_d1 = len(c1s) - len(used1)
    unmatched_d2 = len(c2s) - len(used2)
    return {"matches": matches, "unmatched_d1": unmatched_d1, "unmatched_d2": unmatched_d2,
            "n_curves_d1": len(c1s), "n_curves_d2": len(c2s)}, None


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = find_pairs()
    print(f"Найдено пар D1/D2: {len(pairs)} ({sum(1 for w,_,_ in pairs if w=='Yatskivska_001')} Yatskivska, "
          f"{sum(1 for w,_,_ in pairs if w=='Pn_Zavoda_001')} Pn_Zavoda)")

    all_matches, skipped = [], []
    for well, f1, f2 in pairs:
        plate = os.path.basename(f1)
        try:
            res, err = analyze_pair(well, f1, f2)
        except Exception as e:
            skipped.append({"plate": plate, "reason": f"exception: {e}"})
            print(f"  [SKIP] {plate}: exception {e}")
            continue
        if err:
            skipped.append({"plate": plate, "reason": err})
            print(f"  [SKIP] {plate}: {err}")
            continue
        n_m = len(res["matches"])
        print(f"  {plate}: {n_m} matched / {res['n_curves_d1']} D1 curves "
              f"(unmatched d1={res['unmatched_d1']} d2={res['unmatched_d2']})")
        for m in res["matches"]:
            print(f"      {m['curve_d1']:<16} <-> {m['curve_d2']:<16} corr={m['corr']:+.3f} "
                  f"n={m['n_overlap']:5d} median|diff|={m['median_abs_diff']:.3f} "
                  f"({m['median_abs_diff_pct']}% of range) rmse={m['rmse']:.3f}")
        all_matches.extend(res["matches"])

    def summarize(ms, label):
        if not ms:
            print(f"\n{label}: нет пар.")
            return {"n": 0, "median_abs_diff_pct": None, "p90_abs_diff_pct": None, "median_corr": None}
        pct = np.array([m["median_abs_diff_pct"] for m in ms if m["median_abs_diff_pct"] is not None])
        corr = np.array([m["corr"] for m in ms])
        print(f"\n{label} (n={len(ms)}):")
        print(f"  median_abs_diff_pct: median={np.median(pct):.2f}%  p90={np.percentile(pct,90):.2f}%  mean={pct.mean():.2f}%")
        print(f"  corr: median={np.median(corr):.4f}  min={corr.min():.4f}")
        return {"n": len(ms), "median_abs_diff_pct": float(np.median(pct)),
                "p90_abs_diff_pct": float(np.percentile(pct, 90)), "median_corr": float(np.median(corr))}

    print(f"\n=== ИТОГ ({len(all_matches)} совпавших пар кривых из {len(pairs)} планшетов, "
          f"{len(skipped)} планшетов пропущено) ===")
    print("ВАЖНО: матчинг D1<->D2 идёт по форме кривой (корр.), не по имени/зонду — разные")
    print("зонды резистивиметрии коррелируют между собой просто по литологии. Матчи с")
    print("corr<0.9 НЕ считать надёжным потолком идентичности — вероятно разные физ. кривые.")
    hi_conf = [m for m in all_matches if m["corr"] >= 0.9]
    lo_conf = [m for m in all_matches if m["corr"] < 0.9]
    summary_hi = summarize(hi_conf, "ВЫСОКАЯ уверенность (corr>=0.9) -- рабочая оценка потолка")
    summary_lo = summarize(lo_conf, "низкая уверенность (corr<0.9) -- НЕ использовать как потолок")
    if lo_conf:
        print("\nПодозрительные матчи (corr<0.9):")
        for m in lo_conf:
            print(f"  {m['plate']} {m['curve_d1']}<->{m['curve_d2']} corr={m['corr']:+.3f}")

    out = {
        "n_pairs": len(pairs), "n_matches": len(all_matches), "n_skipped": len(skipped),
        "skipped": skipped, "matches": all_matches,
        "summary_high_confidence_corr_ge_0.9": summary_hi,
        "summary_low_confidence_corr_lt_0.9": summary_lo,
    }
    out_path = os.path.join(OUT_DIR, "ceiling_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {out_path}")


if __name__ == "__main__":
    main()
