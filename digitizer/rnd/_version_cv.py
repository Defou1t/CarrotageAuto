r"""_version_cv.py — ОБУЧЕННЫЙ ВЫБОР ВЕРСИИ СЛОТА ПО ФОЛДАМ СКВАЖИН: СКОЛЬКО ИЗ ПОТОЛКА БЕРЁТ (§6.245, 28.09).

По строкам `_version_ceiling.py` (слоты с обеими версиями; честна ли каждая; признаки обеих версий и уверенность декодера):
модель (градиентный бустинг, неглубокий) учится на слотах, где честна РОВНО одна версия, «честна версия декодера»; фолды —
скважины (5, `rowdec_wellmap.json`), слот оценивается моделью, не видевшей его скважину. Решение на ВСЕХ слотах с обеими
версиями: p ≥ 0.5 → декодер. Счёт (под именем слота) против нынешнего выбора `src`: поле и сорт A, прибыли/потери.
Это оценка на уровне слота — в выдачу её переводит повтор с кэша (правило в `emit`), который и решает.

  _version_cv.py --rows F:/nds/output/taskS/version_rows.pkl
"""
import sys, argparse, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default=r"F:/nds/output/taskS/version_rows.pkl")
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
TS = Path(a.ts)
R = pickle.load(open(a.rows, "rb"))
wm = json.loads((TS / "rowdec_wellmap.json").read_text(encoding="utf-8"))


def well(sh):
    st = Path(sh).stem
    return wm.get(st) or st.split("_")[0]


def feats(r):
    ll = np.log1p(r["nn_alt"]) - np.log1p(r["nn_base"])
    if r["base"] != "prod":
        ll = -ll
    return [r["conf"] if r["conf"] is not None else 0.5, r["cov_d"], r["cov_p"], r["gap_d"], r["gap_p"], r["jmp_d"], r["jmp_p"],
            r["agree"], ll, 1.0 if r["base"] == "dec" else 0.0]


try:
    from sklearn.ensemble import HistGradientBoostingClassifier as GB
    MK = lambda: GB(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0)
except Exception:
    from sklearn.linear_model import LogisticRegression as LR
    MK = lambda: LR(max_iter=500)
X = np.array([feats(r) for r in R], float)
W = [well(r["sheet"]) for r in R]
wells = sorted(set(W)); rng = np.random.default_rng(0); rng.shuffle(wells)
fold_of = {w: i % a.folds for i, w in enumerate(wells)}
F = np.array([fold_of[w] for w in W])
lab_mask = np.array([r["hp"] != r["hd"] for r in R]); y = np.array([r["hd"] for r in R])
P = np.zeros(len(R))
for f in range(a.folds):
    tr = (F != f) & lab_mask; te = F == f
    m = MK(); m.fit(X[tr], y[tr])
    P[te] = m.predict_proba(X[te])[:, 1]
for thr in (0.4, 0.5, 0.6):
    C = defaultdict(Counter)
    for r, p in zip(R, P):
        cur = r["hp"] if r["src"] == "prod" else r["hd"] if r["src"] == "dec" else False
        new = r["hd"] if p >= thr else r["hp"]
        C[r["set"]]["до"] += cur; C[r["set"]]["после"] += new
        C[r["set"]]["прибыль"] += (new and not cur); C[r["set"]]["потеря"] += (cur and not new)
    print(f"★ порог {thr}: " + "; ".join(f"{sn}: {c['до']} → {c['после']} ({c['после'] - c['до']:+d}; +{c['прибыль']}/−{c['потеря']})"
                                       for sn, c in C.items()))
# потолок для справки
for sn in ("поле", "сорт A"):
    rr = [r for r in R if r["set"] == sn]
    cur = sum((r["hp"] if r["src"] == "prod" else r["hd"] if r["src"] == "dec" else False) for r in rr)
    best = sum(r["hp"] or r["hd"] for r in rr)
    print(f"   {sn}: нынешний выбор {cur}, оракул {best} (потолок +{best - cur})")
