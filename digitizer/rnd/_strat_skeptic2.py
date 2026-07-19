r"""_strat_skeptic2.py — ВСКРЫТИЕ единственного выигрыша (BEZLUD_051 CALI1).

Весь прирост 3/24 -> 4/24 держится на ОДНОЙ кривой, перешедшей порог cov 0.786 -> 0.915.
Проверяем, РЕАЛЬНОЕ ли это ведение трассы или инфляция метрики:
  A. качество ДОБИТЫХ строк отдельно (а не медиана по всем) — если добитые строки мусорные,
     а med держится только потому, что их меньшинство, то это игра с метрикой;
  B. NULL-КОНТРОЛИ: даёт ли те же 4/24 любая другая затычка дыр (zero-order hold, константа,
     даже ЗАВЕДОМО ЛОЖНОЕ значение) — если да, механизм не при чём, работает только порог;
  C. запас над порогом: насколько 0.915 близко к 0.9;
  D. не переключился ли ЦВЕТ-победитель (run_strategy берёт лучший по med).
"""
import sys
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
import _relatch_bench as BE

_SHEETS = None
_BE_LOAD = BE.load


def _cached_load():
    global _SHEETS
    if _SHEETS is None:
        _SHEETS = _BE_LOAD()
    return _SHEETS


BE.load = _cached_load


def fill_tracer(max_gap=60, max_dx=30.0, mode="linear"):
    """mode: linear (их идея) | hold (ступенька) | wild (заведомо ложное +200px) — NULL-контроль."""
    def tracer(rec, csr, H):
        tr = BE.trace(rec, csr, H)
        if len(tr) < 2:
            return tr
        ys = sorted(tr)
        for i in range(len(ys) - 1):
            ya, yb = ys[i], ys[i + 1]
            g = yb - ya - 1
            if g <= 0 or g > max_gap:
                continue
            xa, xb = tr[ya], tr[yb]
            if abs(xb - xa) > max_dx:
                continue
            for r in range(ya + 1, yb):
                if mode == "linear":
                    tr[r] = xa + (xb - xa) * (r - ya) / (yb - ya)
                elif mode == "hold":
                    tr[r] = xa
                elif mode == "wild":
                    tr[r] = xa + 200.0
        return tr
    return tracer


def added_row_quality(max_gap=60, max_dx=30.0):
    """A. Ошибка на ДОБИТЫХ строках отдельно от базовых. Только пересечение с экспертом."""
    print("\n" + "=" * 104)
    print("A. КАЧЕСТВО ДОБИТЫХ СТРОК (отдельно от базовых). Только строки, размеченные экспертом.")
    print("=" * 104)
    print(f"{'скважина':<12}{'кривая':<9}{'цвет':<8}"
          f"{'база n':>8}{'база med':>10}"
          f"{'добито':>8}{'добито∩GT':>10}{'ДОБ med':>9}{'ДОБ p90':>9}{'ДОБ >3px':>9}")
    agg_base, agg_add = [], []
    for sheet in _cached_load():
        H = sheet["H"]
        names, mat = BE._gt_matrix(sheet)
        for rec in sheet["curves"]:
            oi = names.index(rec["name"])
            for c in sheet["colors"]:
                csr = rec["runs"][c]
                t0 = BE.trace(rec, csr, H)
                t1 = fill_tracer(max_gap, max_dx)(rec, csr, H)
                new_ys = sorted(set(t1) - set(t0))
                if not new_ys:
                    continue
                # база
                ys0 = np.array(sorted(t0), np.int64)
                x0 = np.array([t0[y] for y in ys0])
                g0 = mat[oi, ys0]; m0 = ~np.isnan(g0)
                d0 = np.abs(x0[m0] - g0[m0])
                # добитые
                ysn = np.array(new_ys, np.int64)
                xn = np.array([t1[y] for y in ysn])
                gn = mat[oi, ysn]; mn = ~np.isnan(gn)
                dn = np.abs(xn[mn] - gn[mn])
                if len(dn) < 5 or len(d0) < 30:
                    continue
                agg_base.append(d0); agg_add.append(dn)
                print(f"{sheet['well']:<12}{rec['short']:<9}{c:<8}"
                      f"{len(d0):>8}{np.median(d0):>10.2f}"
                      f"{len(ysn):>8}{len(dn):>10}"
                      f"{np.median(dn):>9.2f}{np.percentile(dn,90):>9.1f}"
                      f"{100.0*(dn>3).mean():>8.0f}%")
    ab = np.concatenate(agg_base); aa = np.concatenate(agg_add)
    print(f"\n  ИТОГО базовых строк {len(ab)}: med {np.median(ab):.2f}px, >3px {100*(ab>3).mean():.1f}%")
    print(f"  ИТОГО добитых строк {len(aa)}: med {np.median(aa):.2f}px, >3px {100*(aa>3).mean():.1f}%")


def null_controls():
    """B. Если ЛЮБАЯ затычка даёт 4/24 — механизм не важен, работает голый порог cov."""
    print("\n" + "=" * 104)
    print("B. NULL-КОНТРОЛИ: заменяем умную интерполяцию на заведомо тупую/ложную")
    print("=" * 104)
    for mode, label in (("linear", "линейная интерполяция (их идея)"),
                        ("hold", "ступенька: держим левый конец"),
                        ("wild", "★ЛОЖЬ: левый конец + 200px (заведомо не туда)")):
        rows = BE.run_strategy(tracer=fill_tracer(60, 30.0, mode))
        h = sorted((r["well"], r["curve"]) for r in rows if r["med"] <= 3 and r["cov"] >= 0.9)
        print(f"  {label:<48} med(med) {np.median([r['med'] for r in rows]):6.1f}  "
              f"med(cov) {np.median([r['cov'] for r in rows]):.3f}  ЧЕСТНЫХ {len(h)}/{len(rows)}")
        print(f"      честные: {h}")


def margins():
    """C. Запас над порогами. Насколько выигрыш — вопрос третьего знака."""
    print("\n" + "=" * 104)
    print("C. ЗАПАС НАД ПОРОГАМИ (честный = med<=3 И cov>=0.9)")
    print("=" * 104)
    base = BE.run_strategy()
    new = BE.run_strategy(tracer=fill_tracer(60, 30.0))
    nb = {(r["well"], r["curve"]): r for r in base}
    for r in sorted(new, key=lambda r: r["med"]):
        k = (r["well"], r["curve"])
        if r["med"] > 8:
            continue
        b = nb[k]
        h = r["med"] <= 3 and r["cov"] >= 0.9
        print(f"  {k[0]:<12}{k[1]:<9} med {b['med']:6.2f}->{r['med']:6.2f}  "
              f"cov {b['cov']:.3f}->{r['cov']:.3f}  "
              f"запас cov {r['cov']-0.9:+.3f}  запас med {3-r['med']:+.2f}  "
              f"{'ЧЕСТНАЯ' if h else ''}")
    print("\n  Число честных при СДВИНУТОМ пороге cov (проверка хрупкости):")
    for thr in (0.85, 0.88, 0.90, 0.92, 0.95, 0.98):
        hb = sum(1 for r in base if r["med"] <= 3 and r["cov"] >= thr)
        hn = sum(1 for r in new if r["med"] <= 3 and r["cov"] >= thr)
        print(f"    cov>={thr:.2f}:  база {hb}/24   добивка {hn}/24   дельта {hn-hb:+d}")
    print("\n  Число честных при СДВИНУТОМ пороге med (cov>=0.9):")
    for thr in (1.0, 2.0, 3.0, 5.0, 8.0):
        hb = sum(1 for r in base if r["med"] <= thr and r["cov"] >= 0.9)
        hn = sum(1 for r in new if r["med"] <= thr and r["cov"] >= 0.9)
        print(f"    med<={thr:.1f}:  база {hb}/24   добивка {hn}/24   дельта {hn-hb:+d}")


def color_switch():
    """D. Не сменился ли цвет-победитель (выбор best-by-med может перескочить)."""
    print("\n" + "=" * 104)
    print("D. ЦВЕТ-ПОБЕДИТЕЛЬ (run_strategy берёт лучший по med) — не перескочил ли")
    print("=" * 104)
    for sheet in _cached_load():
        H = sheet["H"]
        names, mat = BE._gt_matrix(sheet)
        for rec in sheet["curves"]:
            oi = names.index(rec["name"])
            picks = {}
            for tag, tf in (("база", None), ("добивка", fill_tracer(60, 30.0))):
                bm, bc = 1e9, None
                for c in sheet["colors"]:
                    csr = rec["runs"][c]
                    tr = tf(rec, csr, H) if tf else BE.trace(rec, csr, H)
                    if not tr:
                        continue
                    ys, xs = BE._unpack(tr)
                    gx = mat[oi, ys]; m = ~np.isnan(gx)
                    if int(m.sum()) < 30:
                        continue
                    med = float(np.median(np.abs(xs[m] - gx[m])))
                    if med < bm:
                        bm, bc = med, c
                picks[tag] = (bc, bm)
            if picks["база"][0] != picks["добивка"][0]:
                print(f"  ⚠ {sheet['well']:<12}{rec['short']:<9} ЦВЕТ СМЕНИЛСЯ: "
                      f"{picks['база'][0]}({picks['база'][1]:.2f}) -> "
                      f"{picks['добивка'][0]}({picks['добивка'][1]:.2f})")
    print("  (пусто = ни на одной кривой цвет-победитель не сменился)")


def cali_detail():
    """Вскрытие самой BEZLUD_051 CALI1 — где именно легли добитые строки."""
    print("\n" + "=" * 104)
    print("E. ВСКРЫТИЕ BEZLUD_051 CALI1 — единственного источника выигрыша")
    print("=" * 104)
    for sheet in _cached_load():
        if sheet["well"] != "BEZLUD_051":
            continue
        H = sheet["H"]
        names, mat = BE._gt_matrix(sheet)
        for rec in sheet["curves"]:
            if rec["short"] != "CALI1":
                continue
            oi = names.index(rec["name"])
            ngt = len(sheet["gt"][rec["name"]][0])
            print(f"  строк у эксперта: {ngt}, строки кривой {rec['y0']}..{rec['y1']}")
            for c in sheet["colors"]:
                csr = rec["runs"][c]
                t0 = BE.trace(rec, csr, H)
                t1 = fill_tracer(60, 30.0)(rec, csr, H)
                ys0 = np.array(sorted(t0), np.int64)
                g0 = mat[oi, ys0]
                cov0 = int((~np.isnan(g0)).sum()) / ngt
                ys1 = np.array(sorted(t1), np.int64)
                g1 = mat[oi, ys1]
                cov1 = int((~np.isnan(g1)).sum()) / ngt
                d0 = np.abs(np.array([t0[y] for y in ys0])[~np.isnan(g0)] - g0[~np.isnan(g0)])
                d1 = np.abs(np.array([t1[y] for y in ys1])[~np.isnan(g1)] - g1[~np.isnan(g1)])
                if len(d0) < 30:
                    continue
                print(f"    цвет {c:<8} строк {len(t0)}->{len(t1)}  "
                      f"cov {cov0:.3f}->{cov1:.3f}  med {np.median(d0):.2f}->{np.median(d1):.2f}")
                # сколько ДЫР закрыто и какой длины
                gaps = np.diff(ys0) - 1
                big = gaps[gaps > 0]
                if len(big):
                    print(f"      дыр в базе: {len(big)}, суммарно {int(big.sum())} строк, "
                          f"max {int(big.max())}, med {int(np.median(big))}; "
                          f"закрываемых (<=60): {int((big<=60).sum())}")


if __name__ == "__main__":
    added_row_quality()
    null_controls()
    margins()
    color_switch()
    cali_detail()
