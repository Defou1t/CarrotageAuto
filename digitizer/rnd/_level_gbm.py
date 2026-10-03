r"""_level_gbm.py — УРОВЕНЬ МАСШТАБА ПО СТРОКЕ: ОБУЧЕННАЯ МОДЕЛЬ ПО ТРАССЕ ВЫДАЧИ (03.10, §6.264; критерии — в ROADMAP до обучения).

Кэш — `_level_bench.py build` (массивы: трасса эксперта gy/gx, его сегменты lt, трасса выдачи ty/tx и xy/xx, сегменты выдачи lw,
цепочка масштабов). Признаки по строкам трассы выдачи — в долях шкалы 1× (u); метка — уровень эксперта там, где трасса выдачи
в 5 px от его трассы. Поле — 5 фолдов по скважинам, предсказание вне фолда; сорт A — модель на всём поле. Мера — именные честные
в значениях (та же функция, что в `_level_bench.py`). Парный знаковый тест по листам против прода.

  _level_gbm.py --step 4 --train-step 16
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--step", type=int, default=4)
ap.add_argument("--train-step", type=int, default=16)
ap.add_argument("--min-run", type=int, default=25)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/level_gbm_pred.pkl")
ap.add_argument("--model-out", default=r"F:/nds/output/taskS/level_gbm_model.pkl")
a = ap.parse_args()
from auto import refine
from sklearn.ensemble import HistGradientBoostingClassifier

CUR = pickle.load(open(a.cache, "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
RES = {"GZ", "OGZ", "BK", "MBK", "BKZ", "BMK", "IK", "IKA", "IKR", "PZ", "MGZ", "MPZ", "BKP", "LL", "LLD", "LLS"}


def group(root):
    r = re.sub(r"^BKZ_", "", root)
    if r in RES:
        return 0
    if r == "GK":
        return 1
    if r in ("NGK", "NNK", "NNKB", "NNKM", "NKTB", "NKTM"):
        return 2
    if r.startswith("TM"):
        return 3
    return 4


def seg_levels(segs, rows):
    out = np.zeros(len(rows), np.int16)
    for y0, y1, lv in segs or []:
        if lv:
            out[(rows >= y0) & (rows <= y1)] = lv
    return out


def honest(cv, lo):
    """как в `_level_bench.py`: lo — None (уровни выдачи), {} (всё 0) или dict строка→уровень (наследуется последний)"""
    gy, gx, ty, tx, ch = cv["gy"], cv["gx"], cv["ty"], cv["tx"], cv["chain"]
    com, ig, it = np.intersect1d(gy, ty, return_indices=True)
    if len(com) < 30:
        return False
    x = tx[it].astype(np.float64); g = gx[ig].astype(np.float64)
    lt = seg_levels(cv["lt"], com)
    if lo is None:
        lw = seg_levels(cv["lw"], com)
    elif not lo:
        lw = np.zeros(len(com), np.int16)
    else:
        ks = np.array(sorted(lo), np.int64); vs = np.array([lo[k] for k in ks], np.int16)
        pos = np.searchsorted(ks, com, side="right") - 1
        lw = np.where(pos >= 0, vs[np.clip(pos, 0, len(vs) - 1)], 0).astype(np.int16)
    if len(ch) >= 2:
        diff = lw != lt
        for o in np.unique(lw[diff]):
            for t in np.unique(lt[diff & (lw == o)]):
                m = diff & (lw == o) & (lt == t)
                if o >= len(ch) or t >= len(ch):
                    x[m] = 1e4; continue
                so, sg = ch[o], ch[t]
                v = so["v_left"] + (x[m] - so["x_left"]) * (so["v_right"] - so["v_left"]) / ((so["x_right"] - so["x_left"]) or 1)
                x[m] = sg["x_left"] + (v - sg["v_left"]) * (sg["x_right"] - sg["x_left"]) / ((sg["v_right"] - sg["v_left"]) or 1)
    e = np.abs(x - g)
    return bool(np.median(e) <= 3.0 and len(com) / max(1, len(gy)) >= 0.9)


def features(cv, step):
    """→ (строки, матрица признаков) по трассе выдачи, каждая step-я строка её протяжённости"""
    ch = cv["chain"]; s0, s1 = ch[0], ch[1]
    xl, xr = min(s0["x_left"], s0["x_right"]), max(s0["x_left"], s0["x_right"])
    w = max(1.0, xr - xl)
    ys = cv["xy"].astype(np.int64); xs = cv["xx"].astype(np.float64)
    if len(ys) < 50:
        return None, None
    y0, y1 = int(ys[0]), int(ys[-1])
    R = np.arange(y0, y1 + 1)
    u = (np.interp(R, ys, xs) - xl) / w                                   # u по каждой строке
    has = np.zeros(len(R), bool); has[ys - y0] = True
    near = np.convolve(has.astype(np.int32), np.ones(7, np.int32), mode="same") > 0
    n = len(R)
    # скачки: j = u[t+4] − u[t−4]; события — локальные экстремумы |j| ≥ 0.3
    j = np.zeros(n); j[4:-4] = u[8:] - u[:-8]
    ev = np.flatnonzero(np.abs(j) >= 0.3)
    keep = []
    for i in ev:
        lo, hi = max(0, i - 8), min(n, i + 9)
        if abs(j[i]) >= np.abs(j[lo:hi]).max() - 1e-12:
            keep.append(i)
    ev = np.array(keep, np.int64)
    dn = ev[j[ev] < 0] if len(ev) else ev; up = ev[j[ev] > 0] if len(ev) else ev
    idx = np.arange(0, n, step)
    idx = idx[near[idx]]
    if not len(idx):
        return None, None

    def since(evs, t):
        if not len(evs):
            return np.full(len(t), 5000.0), np.full(len(t), -1)
        p = np.searchsorted(evs, t, side="right") - 1
        d = np.where(p >= 0, t - evs[np.clip(p, 0, len(evs) - 1)], 5000)
        return np.minimum(d, 5000).astype(float), np.where(p >= 0, evs[np.clip(p, 0, len(evs) - 1)], -1)

    def until(evs, t):
        if not len(evs):
            return np.full(len(t), 5000.0)
        p = np.searchsorted(evs, t, side="left")
        d = np.where(p < len(evs), evs[np.clip(p, 0, len(evs) - 1)] - t, 5000)
        return np.minimum(d, 5000).astype(float)

    F = [u[idx]]
    for off in (-400, -100, -25, 25, 100, 400):
        F.append(u[np.clip(idx + off, 0, n - 1)])
    # окна до/после: максимум и минимум
    for W in (50, 200, 800):
        bmax = np.array([u[max(0, i - W):i + 1].max() for i in idx]); bmin = np.array([u[max(0, i - W):i + 1].min() for i in idx])
        amax = np.array([u[i:min(n, i + W + 1)].max() for i in idx]); amin = np.array([u[i:min(n, i + W + 1)].min() for i in idx])
        F += [bmax, bmin, amax, amin]
    sd, pd_ = since(dn, idx); su, pu = since(up, idx)
    F += [sd, su, until(dn, idx), until(up, idx)]
    # откуда и куда последний скачок вниз/вверх
    for p in (pd_, pu):
        F.append(np.where(p >= 0, u[np.clip(p - 6, 0, n - 1)], -1.0))
        F.append(np.where(p >= 0, u[np.clip(p + 6, 0, n - 1)], -1.0))
    r0 = (s0["v_right"] - s0["v_left"]) or 1.0; r1 = (s1["v_right"] - s1["v_left"])
    f = r1 / r0
    shift = float(abs(s1["v_left"] - s0["v_right"]) < 1e-6 * max(1.0, abs(s0["v_right"])))
    gr = group(cv["root"])
    const = [len(ch), f, shift, w] + [float(gr == k) for k in range(5)]
    for c_ in const:
        F.append(np.full(len(idx), c_))
    F.append((idx / max(1, n - 1)).astype(float))
    return R[idx], np.stack(F, 1).astype(np.float32)


# скважина листа и фолды
def well(sheet):
    q = SRC.get(sheet)
    return q.parent.parent.name if q else sheet


chained = [cv for cv in CUR if len(cv["chain"]) >= 2]
field_wells = sorted({well(cv["sheet"]) for cv in chained if cv["set"] == "поле"})
FOLD = {wl: i % a.folds for i, wl in enumerate(field_wells)}
print(f"кривых с цепочкой: {len(chained)} (поле {sum(cv['set'] == 'поле' for cv in chained)}), скважин поля {len(field_wells)}")

DATA = []        # (cv, rows, X, y_mask, y)
for cv in chained:
    rows, X = features(cv, a.step)
    if rows is None:
        DATA.append((cv, None, None, None, None)); continue
    lab = seg_levels(cv["lt"], rows)
    gi = np.searchsorted(cv["gy"], rows); gi = np.clip(gi, 0, len(cv["gy"]) - 1)
    ok = (cv["gy"][gi] == rows)
    xi = np.interp(rows, cv["xy"], cv["xx"])
    close = ok & (np.abs(xi - cv["gx"][gi]) <= 5)
    DATA.append((cv, rows, X, close, np.minimum(lab, 2)))
print("признаки готовы")


def fit(items):
    Xs, Ys = [], []
    for cv, rows, X, close, y in items:
        if rows is None:
            continue
        sel = np.flatnonzero(close)[:: max(1, a.train_step // a.step)]
        if len(sel):
            Xs.append(X[sel]); Ys.append(y[sel])
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, l2_regularization=1.0, random_state=0)
    m.fit(X, Y)
    return m, len(Y), Counter(Y.tolist())


PRED = {}
for k in range(a.folds):
    tr = [d for d in DATA if d[0]["set"] == "поле" and FOLD.get(well(d[0]["sheet"])) != k]
    te = [d for d in DATA if d[0]["set"] == "поле" and FOLD.get(well(d[0]["sheet"])) == k]
    m, n, cnt = fit(tr)
    for cv, rows, X, close, y in te:
        if rows is not None:
            PRED[id(cv)] = (rows, m.predict(X))
    print(f"  фолд {k}: обучение {n} строк {dict(cnt)}, проверка кривых {len(te)}")
mA, n, cnt = fit([d for d in DATA if d[0]["set"] == "поле"])
pickle.dump(mA, open(a.model_out, "wb"))
for cv, rows, X, close, y in DATA:
    if cv["set"] == "сорт A" and rows is not None:
        PRED[id(cv)] = (rows, mA.predict(X))
print(f"  модель на всём поле: {n} строк")

# точность по строкам (размеченным) и мера
acc = defaultdict(list)
for cv, rows, X, close, y in DATA:
    if rows is None or id(cv) not in PRED:
        continue
    p = PRED[id(cv)][1]
    if close.any():
        acc[cv["set"]].append(float(np.mean(p[close] == y[close])))
for sn, v in acc.items():
    print(f"   {sn}: верных уровней по строкам (та же тушь) — медиана по кривым {np.median(v):.2f}, ≥ 0.9 у {np.mean(np.array(v) >= 0.9):.0%}")
C = Counter(); PER = defaultdict(lambda: [0, 0])
OUT = {}
for cv in CUR:
    h0 = honest(cv, None)
    if len(cv["chain"]) >= 2 and id(cv) in PRED:
        rows, p = PRED[id(cv)]
        lv = {int(r): int(v) for r, v in zip(rows, p)}
        lv = refine.enforce_min_run(lv, a.min_run)
        h1 = honest(cv, lv)
        OUT[(cv["sheet"], cv["name"])] = {int(r): int(v) for r, v in sorted(lv.items())}
    else:
        h1 = h0
    C[(cv["set"], "prod")] += h0; C[(cv["set"], "gbm")] += h1
    PER[(cv["set"], cv["sheet"])][0] += h0; PER[(cv["set"], cv["sheet"])][1] += h1
pickle.dump(OUT, open(a.dump, "wb"))
rng = np.random.default_rng(0)
for sn in ("поле", "сорт A"):
    d = np.array([v1 - v0 for (s, _), (v0, v1) in PER.items() if s == sn]); nz = d[d != 0]
    p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
    print(f"★ {sn}: именных честных в значениях — прод {C[(sn, 'prod')]}, модель {C[(sn, 'gbm')]} (Δ {C[(sn, 'gbm')] - C[(sn, 'prod')]:+d}, "
          f"листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
