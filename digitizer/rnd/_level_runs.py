r"""_level_runs.py — МЕТКА УРОВНЯ НА ОТРЕЗКЕ ВЫДАЧИ: КЛАССИФИКАТОР ПОВЕРХ ПРОДА (11.10, к §6.299).

`_level_ceiling.py`: три четверти потери на уровнях — неверная метка на пробеге уровня выдачи, а не граница (поле +127 из +171,
сорт A +33 из +50). Здесь — бустинг, предсказывающий для пробега выдачи лучший уровень (оракул: больше строк ≤ 3 px от эталона).
Признаки — только то, что есть в проде: цепочка (K, отношение диапазонов, сдвиг, семейство), метка выдачи и соседей, номер
пробега, длина, положение в полосе (медиана, квантили, доли у краёв), скачки x на стыках, невязки значения на стыке при каждом k.
Поле — 5 фолдов по скважинам; сорт A — модель на всём поле. Перемаркировка, если P(лучший ≠ выдача) ≥ τ. Мера — построчная
(как `_level_ceiling.py`): именных честных в значениях.

  _level_runs.py --cache F:/nds/output/taskS/level_bench_nslx4.pkl --taus 0.5 0.6 0.7 0.8
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench_nslx4.pkl")
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--extra-cache", default="", help="кэш `_level_bench` листов вне поля — только в обучение (скважины сорта A исключены)")
ap.add_argument("--prior", default="", help="§6.301: кэш `_level_bench` (эталон вне поля) — приор log10 значений семейства; + признаки правдоподобия значения пробега при каждом k")
ap.add_argument("--dump", default=r"F:/nds/output/taskS/level_runs.pkl")
a = ap.parse_args()
from sklearn.ensemble import HistGradientBoostingClassifier

RES = {"GZ", "OGZ", "BK", "MBK", "BKZ", "BMK", "IK", "IKA", "IKR", "PZ", "MGZ", "MPZ", "BKP", "LL", "LLD", "LLS"}


def grp(root):
    r = re.sub(r"^BKZ_", "", root)
    if r in RES:
        return 0
    if r.startswith("TM") or r in ("T", "TP"):
        return 1
    if r in ("GK", "NGK", "NNK", "NK", "GGK"):
        return 2
    if r in ("SP", "PS", "DS", "MDS", "CALI"):
        return 3
    return 4


def seg_levels(segs, rows):
    out = np.zeros(len(rows), np.int16)
    for y0, y1, lv in segs or []:
        if lv:
            out[(rows >= y0) & (rows <= y1)] = lv
    return out


def val(s, x):
    return s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / ((s["x_right"] - s["x_left"]) or 1)


def pix(s, v):
    return s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / ((s["v_right"] - s["v_left"]) or 1)


KMAX = 4


def curve_runs(cv):
    """→ (общие строки, E[k, строка] или None, покрытие ок, пробеги [(s, e)], метки выдачи, признаки пробегов, оракул пробегов)"""
    gy, gx, ty, tx, ch = cv["gy"], cv["gx"], cv["ty"], cv["tx"], cv["chain"]
    K = len(ch)
    if K < 2 or len(ty) < 30:
        return None
    com, ig, it = np.intersect1d(gy, ty, return_indices=True)
    if len(com) < 30:
        return None
    cov = len(com) / max(1, len(gy)) >= 0.9
    x = tx[it].astype(np.float64); g = gx[ig].astype(np.float64)
    lt = seg_levels(cv["lt"], com); lw = np.minimum(seg_levels(cv["lw"], com), K - 1)
    E = np.full((K, len(com)), 1e4)
    for k in range(K):
        for t in np.unique(lt):
            if t < K:
                m = lt == t
                E[k, m] = np.abs(pix(ch[t], val(ch[k], x[m])) - g[m])
    # пробеги — по ВСЕМ строкам трассы выдачи (как в проде), признаки — по трассе; оракул — по общим с эталоном строкам
    lwa = np.minimum(seg_levels(cv["lw"], ty), K - 1)
    st = np.flatnonzero(np.diff(np.concatenate([[-1], lwa])) != 0); en = np.concatenate([st[1:], [len(lwa)]])
    xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
    W = max(1.0, xr - xl); u = (tx.astype(np.float64) - xl) / W
    r0 = abs(ch[0]["v_right"] - ch[0]["v_left"]) or 1.0; r1 = abs(ch[1]["v_right"] - ch[1]["v_left"])
    f = r1 / r0; shift = (ch[1]["v_left"] - ch[0]["v_left"]) / r0
    F, OR, RUNS = [], [], []
    labs = [int(lwa[s_]) for s_ in st]
    for j, (s_, e_) in enumerate(zip(st, en)):
        uu = u[s_:e_]
        n = e_ - s_
        hd = min(10, n)
        jin = abs(np.median(u[s_:s_ + hd]) - np.median(u[max(0, s_ - hd):s_])) if j > 0 else np.nan
        jout = abs(np.median(u[e_ - hd:e_]) - np.median(u[e_:e_ + hd])) if j < len(st) - 1 else np.nan
        # невязка значения на стыке с предыдущим пробегом (его метка выдачи) при каждом k текущего
        res_ = []
        for k in range(KMAX):
            if j == 0 or k >= K:
                res_.append(np.nan); continue
            vp = val(ch[labs[j - 1]], np.median(tx[max(0, s_ - hd):s_]))
            res_.append(abs(val(ch[k], np.median(tx[s_:s_ + hd])) - vp) / r0)
        zk = []
        pk = PRIOR.get(re.sub(r"^BKZ_", "", cv["root"])) if PRIOR else None
        for k in range(KMAX):
            if pk is None or k >= K:
                zk.append(np.nan); continue
            vm = val(ch[k], float(np.median(tx[s_:e_])))
            zk.append((np.log10(vm) - pk[0]) / pk[1] if vm > 0 else -9.0)
        F.append(zk + [K, np.log(max(1e-3, f)), shift, grp(cv["root"]), labs[j], labs[j - 1] if j > 0 else -1,
                  labs[j + 1] if j < len(st) - 1 else -1, j, len(st) - 1 - j, len(st), np.log1p(n), n / len(tx),
                  np.median(uu), np.percentile(uu, 10), np.percentile(uu, 90), np.mean(uu > 0.9), np.mean(uu < 0.1),
                  np.std(np.diff(uu)) if n > 2 else 0.0, jin, jout] + res_)
        RUNS.append((int(ty[s_]), int(ty[e_ - 1])))
        m = (com >= ty[s_]) & (com <= ty[e_ - 1])
        if m.sum() >= 5:
            gd = (E[:, m] <= 3).mean(1); kb = int(np.argmax(gd))
            OR.append(kb if (gd[kb] >= 0.5 and kb < KMAX) else -1)
        else:
            OR.append(-1)
    return dict(com=com, E=E, cov=cov, lw=lw, runs=RUNS, labs=labs, F=F, OR=OR, K=K)


def hon(R, lev):
    return bool(R["cov"] and np.median(R["E"][lev, np.arange(len(R["com"]))]) <= 3.0)


def relabel(R, newlabs):
    lev = R["lw"].copy()
    for (y0, y1), k in zip(R["runs"], newlabs):
        lev[(R["com"] >= y0) & (R["com"] <= y1)] = k
    return lev


CUR = pickle.load(open(a.cache, "rb"))
PRIOR = {}
if a.prior:                                                   # §6.301: распределение log10 значений семейства по эталону вне поля
    _acc = {}
    for cv in pickle.load(open(a.prior, "rb")):
        ch = cv["chain"]
        if not ch:
            continue
        lt = np.minimum(seg_levels(cv["lt"], cv["gy"]), len(ch) - 1)
        v = np.array([val(ch[l], x) for l, x in zip(lt[::25], cv["gx"][::25].astype(float))])
        v = v[v > 0]
        if len(v):
            _acc.setdefault(re.sub(r"^BKZ_", "", cv["root"]), []).extend(np.log10(v).tolist())
    PRIOR = {k: (float(np.median(v)), float(max(0.1, np.percentile(v, 75) - np.percentile(v, 25)))) for k, v in _acc.items() if len(v) >= 50}
    print(f"приор: семейств {len(PRIOR)}")
EXTRA = pickle.load(open(a.extra_cache, "rb")) if a.extra_cache else []
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else sh
CURVES = []
for cv in CUR + EXTRA:
    R = curve_runs(cv)
    if R is None:
        continue
    R.update(set=cv["set"], sheet=cv["sheet"], name=cv["name"])
    R["out"] = hon(R, R["lw"])
    CURVES.append(R)
field_w = sorted({well(R["sheet"]) for R in CURVES if R["set"] == "поле"})
FOLD = {w: i % a.folds for i, w in enumerate(field_w)}
HOLD_W = {well(R["sheet"]) for R in CURVES if R["set"] == "сорт A"}
X = np.array([f for R in CURVES for f in R["F"]], float)
Y = np.array([o for R in CURVES for o in R["OR"]])
CI = np.array([i for i, R in enumerate(CURVES) for _ in R["F"]])
SET = np.array([CURVES[i]["set"] for i in CI]); WL = np.array([well(CURVES[i]["sheet"]) for i in CI])
OUTL = np.array([l for R in CURVES for l in R["labs"]])
print(f"кривых с цепочкой {len(CURVES)}, пробегов {len(Y)}; оракул ≠ выдаче у {int(((Y >= 0) & (Y != OUTL)).sum())}")
CAT = [3 + (KMAX if a.prior else 0)]


def fit(mask):
    m_ = mask & (Y >= 0)
    cat = np.zeros(X.shape[1], bool); cat[CAT] = True
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                         min_samples_leaf=20, categorical_features=cat, random_state=0)
    clf.fit(X[m_], Y[m_])
    return clf


P = np.full((len(Y), KMAX), np.nan)
ex = (SET == "вне") & ~np.isin(WL, list(HOLD_W))
for k in range(a.folds):
    tr = ((SET == "поле") & np.array([FOLD.get(w, -1) != k for w in WL])) | (ex & np.array([FOLD.get(w, -1) != k for w in WL]))
    te = (SET == "поле") & np.array([FOLD.get(w, -1) == k for w in WL])
    clf = fit(tr)
    pr = clf.predict_proba(X[te]); P[np.ix_(np.flatnonzero(te), clf.classes_)] = pr
clf = fit((SET == "поле") | ex)
te = SET == "сорт A"
pr = clf.predict_proba(X[te]); P[np.ix_(np.flatnonzero(te), clf.classes_)] = pr
P = np.nan_to_num(P, nan=0.0)
off = np.cumsum([0] + [len(R["F"]) for R in CURVES])
for tau in a.taus:
    C = Counter(); PER = {}
    for i, R in enumerate(CURVES):
        if R["set"] == "вне":
            continue
        p = P[off[i]:off[i + 1]]
        new = [int(np.argmax(pp[:R["K"]])) if (pp[:R["K"]].max() >= tau and int(np.argmax(pp[:R["K"]])) != l) else l
               for pp, l in zip(p, R["labs"])]
        h = hon(R, relabel(R, new))
        C[(R["set"], 0)] += R["out"]; C[(R["set"], 1)] += h
        PER.setdefault((R["set"], R["sheet"]), [0, 0]); PER[(R["set"], R["sheet"])][0] += R["out"]; PER[(R["set"], R["sheet"])][1] += h
    rng = np.random.default_rng(0)
    for sn in ("поле", "сорт A"):
        d = np.array([v[1] - v[0] for (s, _), v in PER.items() if s == sn]); nz = d[d != 0]
        pv = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
        print(f"★ τ={tau:g} {sn:6s}: выдача {C[(sn, 0)]:5d} → {C[(sn, 1)]:5d} (Δ {C[(sn, 1)] - C[(sn, 0)]:+d}, листов ↑{int((d > 0).sum())}"
              f"/↓{int((d < 0).sum())}, p = {pv:.4f})")
pickle.dump(dict(P=P, Y=Y, OUTL=OUTL, CI=CI), open(a.dump, "wb"))
