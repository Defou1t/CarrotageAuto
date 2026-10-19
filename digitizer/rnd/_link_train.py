r"""_link_train.py — МОДЕЛЬ ПЕРЕХОДА «ПИК → ПИК ТОЙ ЖЕ КРИВОЙ» ДЛЯ ОБУЧЕННОГО ВЕДЕНИЯ (19.10, §6.318).

Выборка `_link_data.py`; 5 фолдов по листам (stem), бустинг на подвыборке. Пишет `--out` = {fold: модель, "folds": stem → fold,
"feats": имена}; печатает AUC вне фолда и AUC простого |Δx| (как цена прод-пути) для сравнения.

  _link_train.py --n 3000000
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"F:/nds/output/taskS/link_data.pkl")
ap.add_argument("--n", type=int, default=3000000)
ap.add_argument("--out", default=r"F:/nds/output/taskS/link_model.pkl")
a = ap.parse_args()
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
D = pickle.load(open(a.data, "rb"))
X, y, st = D["X"], D["y"], D["stem"]
stems = sorted(set(st.tolist())); FOLD = {s: i % 5 for i, s in enumerate(stems)}
fo = np.array([FOLD[s] for s in st])
rng = np.random.default_rng(0)
M = {}; P = np.full(len(y), np.nan)
for k in range(5):
    tr = np.flatnonzero(fo != k); te = np.flatnonzero(fo == k)
    if len(tr) > a.n:
        tr = rng.choice(tr, a.n, replace=False)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=100, random_state=0)
    m.fit(X[tr], y[tr]); M[k] = m
    sub = te if len(te) <= 2000000 else rng.choice(te, 2000000, replace=False)
    P[sub] = m.predict_proba(X[sub])[:, 1]
    print(f"  фолд {k}: обучение {len(tr)}, AUC {roc_auc_score(y[sub], P[sub]):.4f}, |Δx| {roc_auc_score(y[sub], -X[sub, 1]):.4f}")
ok = np.isfinite(P)
print(f"★ AUC вне фолда {roc_auc_score(y[ok], P[ok]):.4f}; по |Δx|/d {roc_auc_score(y[ok], -X[ok, 1]):.4f}; по |Δx/d − v| {roc_auc_score(y[ok], -X[ok, 2]):.4f}")
M["folds"] = FOLD
M["feats"] = ["dx", "adx", "dacc", "v", "pa", "pb", "rank_b", "n_win", "emb_d", "d", "k"]
pickle.dump(M, open(a.out, "wb"))
