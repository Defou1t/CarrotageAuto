r"""_screen_verdict.py — ПРИГОВОР ЭКРАНА НА ПОДМНОЖЕСТВЕ ЛИСТОВ (§6.248, 29.09).

Два файла счёта `_name_cost_prod.py --dump` (база и вариант) и список листов экрана (например, листы фолда, модель которого
переобучена). Критерий §6.248 — задан до прогона: на листах экрана из поля безымянных Δ > 0 при p < 0.05 (парная
перестановка знаков по листам) и именных Δ ≥ −5. Печатает и сорт A (если листы есть) — справочно. Код 0 — пройдено, 2 — нет,
3 — покрытие неполное (листов экрана в обоих счетах < 99.5%).

  _screen_verdict.py --old percurve_scr_base_N.pkl --new percurve_scr_new_N.pkl --sheets screen_f3.txt
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--sheets", required=True)
ap.add_argument("--perm", type=int, default=200000)
a = ap.parse_args()
TS = Path(a.ts)


def load(p):
    d = pickle.load(open(TS / p, "rb"))
    return d[next(iter(d))] if len(d) == 1 and isinstance(next(iter(d.values())), dict) else d


A, B = load(a.old), load(a.new)
scr = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
field = {l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
hold = {l.strip() for l in (TS / "holdoutA_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}


def perm_p(dv):
    nz = dv[dv != 0]
    if not len(nz):
        return 1.0
    rng = np.random.default_rng(0)
    sims = (rng.choice([-1, 1], size=(a.perm, len(nz))) * np.abs(nz)).sum(1)
    return float((np.abs(sims) >= abs(dv.sum()) - 1e-9).mean())


code = 0
for nm, S in (("поле", field), ("сорт A", hold)):
    want = [s for s in scr if s in S]
    if not want:
        continue
    com = [s for s in want if s in A and s in B]
    if len(com) < 0.995 * len(want):
        print(f"⛔ {nm}: сравнимо {len(com)} из {len(want)} листов экрана — покрытие неполное"); code = 3; continue
    du = np.array([B[s][1] - A[s][1] for s in com], float); dn = np.array([B[s][0] - A[s][0] for s in com], float)
    p = perm_p(du)
    ok = du.sum() > 0 and p < 0.05 and dn.sum() >= -5
    print(f"{nm}: листов экрана {len(com)}; безымянных {int(sum(A[s][1] for s in com))} → {int(sum(B[s][1] for s in com))} "
          f"(Δ {int(du.sum()):+d}, листов ↑{int((du > 0).sum())}/↓{int((du < 0).sum())}, p = {p:.4f}); именных Δ {int(dn.sum()):+d}"
          + (f"   {'★ условие выполнено' if ok else '⛔ условие НЕ выполнено'}" if nm == "поле" else "   (справочно)"))
    if nm == "поле" and not ok and code == 0:
        code = 2
print("★ ЭКРАН ПРОЙДЕН" if code == 0 else ("⛔ ЭКРАН НЕ ПРОЙДЕН" if code == 2 else "⛔ ПОКРЫТИЕ НЕПОЛНОЕ"))
sys.exit(code)
