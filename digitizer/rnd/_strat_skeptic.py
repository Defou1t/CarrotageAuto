r"""_strat_skeptic.py — НЕЗАВИСИМАЯ ПРОВЕРКА заявки "_strat_coast.py: ЧЕСТНЫХ 4/24".

Скептик. По умолчанию считаем результат завышенным. Проверяем:
  (1) воспроизводимость чисел;
  (2) главную ловушку: не куплен ли med падением cov (сравнение ПОКРИВОЙ);
  (3) подгонку: разброс по всем 24 кривым и по 5 скважинам, плато параметров;
  (4) переименование: те же ли честные кривые, что у базы;
  (5) вырождение: насколько трасса отличается от базовой вообще.
"""
import sys
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
import _relatch_bench as BE

# кэш листов в своём процессе (файл стенда не трогаем)
_SHEETS = None
_BE_LOAD = BE.load


def _cached_load():
    global _SHEETS
    if _SHEETS is None:
        _SHEETS = _BE_LOAD()
    return _SHEETS


BE.load = _cached_load


# ── СВОЯ, независимая реализация "только добивка" (не импортирую их код) ──
def my_fill_tracer(max_gap=60, max_dx=30.0):
    """Прогнать штатный BE.trace, затем закрыть внутренние дыры линейной интерполяцией."""
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
                tr[r] = xa + (xb - xa) * (r - ya) / (yb - ya)
        return tr
    return tracer


def key(r):
    return (r["well"], r["curve"])


def table(base_rows, new_rows, title):
    """ПОКРИВОЙ: med и cov рядом. Главная ловушка проекта — cov."""
    b = {key(r): r for r in base_rows}
    n = {key(r): r for r in new_rows}
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(f"{'скважина':<12}{'кривая':<9}"
          f"{'med.база':>10}{'med.нов':>10}{'  ':2}"
          f"{'cov.база':>10}{'cov.нов':>10}{'  ':2}"
          f"{'честн.база':>11}{'честн.нов':>10}   пометка")
    cov_down = cov_up = med_down = med_up = 0
    hb_set, hn_set = set(), set()
    for k in sorted(set(b) | set(n)):
        rb, rn = b.get(k), n.get(k)
        if rb is None or rn is None:
            print(f"{k[0]:<12}{k[1]:<9} !! кривая есть только в одном прогоне")
            continue
        hb = rb["med"] <= 3 and rb["cov"] >= 0.9
        hn = rn["med"] <= 3 and rn["cov"] >= 0.9
        if hb: hb_set.add(k)
        if hn: hn_set.add(k)
        mark = ""
        if hn and not hb: mark += "  ★НОВАЯ ЧЕСТНАЯ"
        if hb and not hn: mark += "  ✗ПОТЕРЯНА ЧЕСТНАЯ"
        if rn["cov"] < rb["cov"] - 1e-9:
            cov_down += 1; mark += f"  cov↓{rb['cov']-rn['cov']:+.3f}"
        if rn["cov"] > rb["cov"] + 1e-9: cov_up += 1
        if rn["med"] < rb["med"] - 1e-9: med_down += 1
        if rn["med"] > rb["med"] + 1e-9: med_up += 1
        print(f"{k[0]:<12}{k[1]:<9}"
              f"{rb['med']:10.2f}{rn['med']:10.2f}  "
              f"{rb['cov']:10.3f}{rn['cov']:10.3f}  "
              f"{('ДА' if hb else '-'):>11}{('ДА' if hn else '-'):>10}   {mark}")
    print(f"\n  cov упал у {cov_down} кривых, вырос у {cov_up}; "
          f"med упал у {med_down}, вырос у {med_up}")
    print(f"  ЧЕСТНЫЕ база: {sorted(hb_set)}")
    print(f"  ЧЕСТНЫЕ нов : {sorted(hn_set)}")
    print(f"  общие: {sorted(hb_set & hn_set)}")
    print(f"  ТОЛЬКО у новой: {sorted(hn_set - hb_set)}")
    print(f"  ТОЛЬКО у базы : {sorted(hb_set - hn_set)}")
    return hb_set, hn_set


def by_well(rows, label):
    print(f"\n  --- разбивка по скважинам: {label} ---")
    wells = sorted({r["well"] for r in rows})
    for w in wells:
        rr = [r for r in rows if r["well"] == w]
        h = [r for r in rr if r["med"] <= 3 and r["cov"] >= 0.9]
        print(f"    {w:<12} n={len(rr):<3} med(med) {np.median([r['med'] for r in rr]):8.1f}  "
              f"med(cov) {np.median([r['cov'] for r in rr]):.3f}  честных {len(h)}/{len(rr)}")


def main():
    print("### ШАГ 1. Базовая линия (эталон для сравнения)")
    base = BE.run_strategy()
    rb = BE.report("baseline", base)

    print("\n### ШАГ 2. Заявленный победитель — ИХ КОД, make_tracer3(60, 30.0)")
    import _strat_coast as SC
    SC.BE.load = _cached_load          # общий кэш
    theirs = BE.run_strategy(tracer=SC.make_tracer3(60, 30.0))
    rt = BE.report("ИХ make_tracer3(60,30)", theirs)

    print("\n### ШАГ 3. МОЯ независимая реализация той же идеи")
    mine = BE.run_strategy(tracer=my_fill_tracer(60, 30.0))
    rm = BE.report("моя добивка(60,30)", mine)

    print("\n### ШАГ 4. ПОКРИВОЙ — главная ловушка (cov!)")
    hb, hn = table(base, theirs, "БАЗА vs ИХ make_tracer3(60, 30.0)")

    by_well(base, "база")
    by_well(theirs, "их стратегия")

    print("\n### ШАГ 5. ПЛАТО параметров (проверка подгонки)")
    print(f"{'max_gap':>8}{'max_dx':>8}{'med(med)':>10}{'med(cov)':>10}{'честных':>9}   какие честные")
    for mg in (10, 20, 30, 40, 60, 100, 200, 400):
        for mdx in (30.0,):
            rows = BE.run_strategy(tracer=my_fill_tracer(mg, mdx))
            h = sorted(key(r) for r in rows if r["med"] <= 3 and r["cov"] >= 0.9)
            print(f"{mg:>8}{mdx:>8.0f}"
                  f"{np.median([r['med'] for r in rows]):10.1f}"
                  f"{np.median([r['cov'] for r in rows]):10.3f}"
                  f"{len(h):>9}   {[a[1] for a in h]}")
    for mg in (60,):
        for mdx in (5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 1e9):
            rows = BE.run_strategy(tracer=my_fill_tracer(mg, mdx))
            h = sorted(key(r) for r in rows if r["med"] <= 3 and r["cov"] >= 0.9)
            print(f"{mg:>8}{mdx:>8.0f}"
                  f"{np.median([r['med'] for r in rows]):10.1f}"
                  f"{np.median([r['cov'] for r in rows]):10.3f}"
                  f"{len(h):>9}   {[a[1] for a in h]}")

    print("\n### ШАГ 6. ВЫРОЖДЕНИЕ: насколько трасса вообще отличается от базовой?")
    # считаем, сколько СТРОК добавила добивка на каждой кривой
    tot_base = tot_new = 0
    per = []
    for sheet in _cached_load():
        H = sheet["H"]
        for rec in sheet["curves"]:
            for c in sheet["colors"]:
                csr = rec["runs"][c]
                t0 = BE.trace(rec, csr, H)
                t1 = my_fill_tracer(60, 30.0)(rec, csr, H)
                added = len(t1) - len(t0)
                tot_base += len(t0); tot_new += len(t1)
                if added:
                    per.append((sheet["well"], rec["short"], c, len(t0), added))
    print(f"  всего строк трассы: база {tot_base}, с добивкой {tot_new} "
          f"(+{tot_new-tot_base}, {100.0*(tot_new-tot_base)/max(tot_base,1):.2f}%)")
    per.sort(key=lambda t: -t[4])
    print("  топ-15 кривых по числу добитых строк (well, curve, цвет, строк базы, добито):")
    for p in per[:15]:
        print(f"    {p[0]:<12}{p[1]:<9}{p[2]:<8}{p[3]:>8}{p[4]:>8}")
    print(f"  кривых(цветов) где добивка вообще что-то сделала: {len(per)}")

    print("\n### ШАГ 7. Устойчивость: главное число при разных tol скоринга")
    for tol in (10.0,):
        pass  # tol влияет только на разбор своя/латч, не на честных

    print("\n=== СВОДКА ===")
    for r in (rb, rt, rm):
        print(f"  {r['name']:<28} med(med) {r['med_med']:7.1f}  "
              f"med(cov) {r['med_cov']:.3f}  ЧЕСТНЫХ {r['honest']}/{r['n']}")


if __name__ == "__main__":
    main()
