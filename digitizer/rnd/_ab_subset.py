r"""_ab_subset.py — парное сравнение двух режимов повтора мерой приёмки на ОБЩИХ листах по полистным дампам `_name_cost_prod.py --dump`
(поле / вне / сорт A, именных и безымянных, знаковый тест по листам). Для стендов на подмножестве листов (§6.353 M80).

  _ab_subset.py <дамп A> <режим A> <дамп B> <режим B>
"""
import sys, pickle, json
from math import comb
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
TS = Path(r"F:/nds/output/taskS")
dA, mA, dB, mB = sys.argv[1:5]
A = pickle.load(open(dA, "rb"))[mA]; B = pickle.load(open(dB, "rb"))[mB]
lst = lambda f: set(l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip())
FLD = lst("wellmap_sheets.txt"); HLD = lst("holdoutA_sheets.txt")
EVO = set(json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))) - FLD
common = set(A) & set(B)
print(f"общих листов {len(common)} (в {mA} {len(A)}, в {mB} {len(B)})")
for nm, SS in (("поле", FLD), ("вне", EVO), ("поле+вне", FLD | EVO), ("сорт A", HLD)):
    S = sorted(common & SS)
    for j, what in ((0, "именных"), (1, "безымянных")):
        a = sum(A[s][j] for s in S); b = sum(B[s][j] for s in S); t = sum(A[s][2] for s in S)
        up = sum(1 for s in S if B[s][j] > A[s][j]); dn = sum(1 for s in S if B[s][j] < A[s][j]); n = up + dn
        p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(up, dn) + 1)) / 2 ** n) if n else 1.0
        print(f"  {nm:9s} ({len(S):4d} листов, {t:5d} кривых) {what:10s}: {a} → {b} (Δ {b - a:+d}; листов ↑{up}/↓{dn}, знаковый p = {p:.4f})")
