r"""_ab_subset.py — ПАРНОЕ СРАВНЕНИЕ ДВУХ ДАМПОВ СЧЁТА НА ОБЩИХ ЛИСТАХ (10.11, стенд §6.359 на части листов): как `merge_ab3.py`, но
режимы из разных дампов без общей базы. Вывод — те же строки «★ <часть> (N листов), именных …», что разбирает `_ab_decide.py`.

  _ab_subset.py <дамп A> <режим A> <дамп B> <режим B>
"""
import pickle, sys, json
import numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
dA = pickle.load(open(sys.argv[1], "rb"))[sys.argv[2]]; dB = pickle.load(open(sys.argv[3], "rb"))[sys.argv[4]]
A, B = sys.argv[2], sys.argv[4]
TS = Path(r"F:/nds/output/taskS")
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
rng = np.random.default_rng(0)
FLD = set(lst("wellmap_sheets.txt")); HLD = set(lst("holdoutA_sheets.txt"))
EVO = set(json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))) - FLD
print(f"листов: {A} {len(dA)}, {B} {len(dB)}, общих {len(set(dA) & set(dB))}")
for sn, SS in (("поле", FLD), ("вне", EVO), ("поле+вне", FLD | EVO), ("сорт A", HLD)):
    S = [s for s in sorted(SS) if s in dA and s in dB]
    for j, what in ((0, "именных"), (1, "безымянных")):
        a_ = sum(dA[s][j] for s in S); b_ = sum(dB[s][j] for s in S)
        d = np.array([dB[s][j] - dA[s][j] for s in S]); nz = d[d != 0]
        p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
        print(f"★ {sn} ({len(S)} листов), {what} честных в значениях: {A} {a_} → {B} {b_} (Δ {b_ - a_:+d}, "
              f"листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
