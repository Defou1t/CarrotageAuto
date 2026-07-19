r"""_strat_skeptic3.py — ДОБИВАЮЩИЙ КОНТРОЛЬ.

Гипотеза скептика: прирост 3->4 даёт не КАЧЕСТВО интерполяции, а САМ ФАКТ добавления строк.
cov в стенде считает ДОЛЮ ПОКРЫТЫХ строк эксперта и НЕ проверяет правильность;
med — МЕДИАНА, устойчивая к 10-20% мусора. Значит добавление даже ЛОЖНЫХ точек
поднимает cov, не трогая med. Проверяем это в лоб, наращивая ложь.
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


def fill_offset(max_gap=60, max_dx=30.0, off=0.0):
    """Заполняем те же дыры, но со СДВИГОМ off от линейной интерполяции."""
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
                tr[r] = xa + (xb - xa) * (r - ya) / (yb - ya) + off
        return tr
    return tracer


print("=" * 100)
print("СКОЛЬКО ЛЖИ ВЫДЕРЖИТ МЕТРИКА: сдвигаем добитые точки от истины на off пикселей")
print("=" * 100)
print(f"{'off, px':>9}{'med(med)':>10}{'med(cov)':>10}{'ЧЕСТНЫХ':>9}   "
      f"{'CALI1 med':>10}{'CALI1 cov':>10}   честные")
base = BE.run_strategy()
hb = sorted((r["well"], r["curve"]) for r in base if r["med"] <= 3 and r["cov"] >= 0.9)
print(f"{'БАЗА':>9}{np.median([r['med'] for r in base]):10.1f}"
      f"{np.median([r['cov'] for r in base]):10.3f}{len(hb):>9}   "
      f"{[r for r in base if r['curve']=='CALI1' and r['well']=='BEZLUD_051'][0]['med']:10.2f}"
      f"{[r for r in base if r['curve']=='CALI1' and r['well']=='BEZLUD_051'][0]['cov']:10.3f}"
      f"   {[a[1] for a in hb]}")
for off in (0.0, 5.0, 20.0, 50.0, 200.0, 1000.0):
    rows = BE.run_strategy(tracer=fill_offset(60, 30.0, off))
    h = sorted((r["well"], r["curve"]) for r in rows if r["med"] <= 3 and r["cov"] >= 0.9)
    cal = [r for r in rows if r["curve"] == "CALI1" and r["well"] == "BEZLUD_051"][0]
    print(f"{off:>9.0f}{np.median([r['med'] for r in rows]):10.1f}"
          f"{np.median([r['cov'] for r in rows]):10.3f}{len(h):>9}   "
          f"{cal['med']:10.2f}{cal['cov']:10.3f}   {[a[1] for a in h]}")

print("\n" + "=" * 100)
print("ВЕДЁТ ЛИ СТРАТЕГИЯ ИДЕНТИЧНОСТЬ ЛУЧШЕ? (разбор своя/ЛАТЧ/между — вот это и есть задача)")
print("=" * 100)
new = BE.run_strategy(tracer=fill_offset(60, 30.0, 0.0))
for tag, rows in (("база", base), ("добивка", new)):
    t = {k: float(np.mean([r[k] for r in rows])) for k in ("своя", "латч", "между")}
    print(f"  {tag:<10} своя {t['своя']:.2f}%   ЛАТЧ {t['латч']:.2f}%   между {t['между']:.2f}%")
print("  -> если 'своя' не выросла, стратегия НЕ улучшила ведение идентичности,")
print("     а только заполнила дыры уже существующей (в т.ч. уже сорванной) трассы.")

print("\n" + "=" * 100)
print("КУДА ЛЕГЛИ ДОБИТЫЕ СТРОКИ: на СВОЮ кривую или на чужую/в пустоту?")
print("=" * 100)
own = latch = between = 0
for sheet in _cached_load():
    H = sheet["H"]
    names, mat = BE._gt_matrix(sheet)
    for rec in sheet["curves"]:
        oi = names.index(rec["name"])
        for c in sheet["colors"]:
            csr = rec["runs"][c]
            t0 = BE.trace(rec, csr, H)
            t1 = fill_offset(60, 30.0, 0.0)(rec, csr, H)
            ny = sorted(set(t1) - set(t0))
            if not ny:
                continue
            ys = np.array(ny, np.int64)
            xs = np.array([t1[y] for y in ys])
            dist = np.abs(mat[:, ys] - xs)
            dist = np.where(np.isnan(dist), np.inf, dist)
            bi = np.argmin(dist, axis=0)
            bd = dist[bi, np.arange(len(ys))]
            seen = np.isfinite(bd); near = seen & (bd <= 10.0)
            own += int((near & (bi == oi)).sum())
            latch += int((near & (bi != oi)).sum())
            between += int((seen & ~near).sum())
tot = own + latch + between or 1
print(f"  добитых строк (размеченных экспертом) {tot}: "
      f"СВОЯ {100*own/tot:.1f}%   ЛАТЧ на чужую {100*latch/tot:.1f}%   между {100*between/tot:.1f}%")
print(f"  (для сравнения, по всей трассе базы: своя 34.9% / латч 31.1% / между 34.0%)")
