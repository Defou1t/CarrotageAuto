r"""line_choice.py — ВЫБОР ТРАССЫ ДЛЯ СЛОТА СРЕДИ КАНДИДАТОВ ВЕДЕНИЯ (§6.280, 04.10).

ЗАЧЕМ. §6.279: у трети кривых честная трасса уже посчитана (прод-путь или декодер), но в слот поставлена другая; у 23% эта трасса
вообще никуда не взята. Здесь — после раскладки и перестановки имён (§6.273) — для каждого слота модель оценивает P(кандидат
честен для слота) по признакам кандидата (источник, длина, покрытие строк слота, положение в полосе шкалы, стиль, извилистость),
слота (семейство, номер зонда, десятилетие) и отношений (ранги, занятость, разница с нынешним). Трасса слота заменяется на
СВОБОДНОГО кандидата (не стоит ни в одном слоте), если P(лучший) ≥ τ и P(лучший) − P(нынешний) ≥ δ; жадно по выигрышу.

ЗАМЕРЕНО ДО ВНЕСЕНИЯ (§6.280, стенд `_line_choice.py`, вне фолда по скважинам): τ 0.7 / δ 0.3 — поле +24 (25 стали честными,
1 перестала), сорт A +8 (8 / 0).

⚠ Признаки ПОВТОРЯЮТ `digitizer/rnd/_line_choice_scan.py` + `_line_choice.py` в точности (§6.90). Без sklearn или модели —
раскладка не меняется, и об этом говорится вслух.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"
_CACHE = {}
_SAID = set()


def _announce(msg):
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"  {msg}")


def dense_tr(tr, max_gap=200):
    ys = sorted(tr); out = {}
    for y0, y1 in zip(ys, ys[1:]):
        out[y0] = float(tr[y0])
        if 0 < y1 - y0 <= max_gap:
            for yy in range(y0 + 1, y1):
                out[yy] = tr[y0] + (tr[y1] - tr[y0]) * (yy - y0) / (y1 - y0)
    if ys:
        out[ys[-1]] = float(tr[ys[-1]])
    return out


def rough_robust(x):
    if x is None or len(x) < 400:
        return np.nan
    from scipy.ndimage import median_filter
    x = np.asarray(x, float)
    s_ = median_filter(x, size=41, mode="nearest")
    dev = np.abs(x - s_); jump = np.abs(np.diff(x, prepend=x[0])) > 40
    vals = [np.mean(dev[i:i + 400]) for i in range(0, len(x) - 400, 200) if not jump[i:i + 400].any()]
    if len(vals) < 3:
        return np.nan
    return float(np.median(vals) / max(5.0, np.percentile(x, 95) - np.percentile(x, 5)))


def _probe(name):
    m = re.match(r"^(?:BKZ_)?[A-Z]+?(\d)", name.split()[0])
    return int(m.group(1)) if m else 0


def _decade(sheet):
    m = re.search(r"(19[5-9]\d|20[0-2]\d)", sheet)
    return (int(m.group(1)) // 10 * 10) if m else 0


def resolve(spec, sheet=None):
    if not spec:
        return None
    if spec.startswith("oof:"):
        d = Path(spec[4:])
        mp = d / "line_choice_folds.json"
        if str(mp) not in _CACHE:
            _CACHE[str(mp)] = json.loads(mp.read_text(encoding="utf-8"))
        k = _CACHE[str(mp)].get(sheet)
        return d / (f"line_choice_f{k}.pkl" if k is not None else "line_choice_all.pkl")
    p = Path(spec)
    if not p.is_absolute():
        p = _MODELS / spec
    return p if p.is_file() else None


def seg_resolve(spec, frame_nlgx=None):
    """§6.348: веса сети туши для признака выбора трассы: имя в auto/models, путь, или «oof:<каталог>» — сеть фолда скважины листа
    (`folds.json` по имени скважины = каталог над wlg; иначе `all.pt`) — только для замера."""
    if not spec:
        return None
    if spec.startswith("oof:"):
        d = Path(spec[4:])
        mp = d / "folds.json"
        if str(mp) not in _CACHE:
            _CACHE[str(mp)] = json.loads(mp.read_text(encoding="utf-8"))
        k = _CACHE[str(mp)].get(Path(frame_nlgx).parent.parent.name) if frame_nlgx else None
        return str(d / (f"f{k}.pt" if k is not None else "all.pt"))
    p = Path(spec)
    if not p.is_absolute():
        p = _MODELS / spec
    return str(p) if p.is_file() else None


def seg_features(rgb, net, denses, step=2, fam=False):
    """§6.348 (как `_seg_feat.py`): карта сети туши по рамке всех кандидатов; по кандидату (строки через step) v = max карты в ±1 px
    от x → (среднее v, доля строк с v > 0.5); меньше 10 строк — nan."""
    from . import ink_seg as IS
    ys_all = [y for d in denses for y in d]; xs_all = [x for d in denses for x in d.values()]
    out = [(np.nan, np.nan)] * len(denses)
    if not ys_all:
        return [None] * len(denses) if fam else out
    H, W = rgb.shape[:2]
    y0, y1 = max(0, int(min(ys_all))), min(H, int(max(ys_all)) + 1)
    x0, x1 = max(0, int(min(xs_all)) - 4), min(W, int(max(xs_all)) + 5)
    S = IS.prob_map(rgb, y0, y1, x0, x1, net)
    if fam:                                               # §6.350 (как `_seg_feat.py --fam`): среднее по каждому каналу
        out = [None] * len(denses)
        for k, d in enumerate(denses):
            ys = np.array(sorted(d))[::step]
            ys = ys[(ys >= y0) & (ys < y1)]
            if len(ys) < 10:
                continue
            xs = np.round([d[int(y)] for y in ys]).astype(int) - x0
            v = np.max(np.stack([S[:, ys - y0, np.clip(xs + o, 0, S.shape[2] - 1)] for o in (-1, 0, 1)]), 0)
            out[k] = v.mean(1).astype(np.float16)
        return out
    for k, d in enumerate(denses):
        ys = np.array(sorted(d))[::step]
        ys = ys[(ys >= y0) & (ys < y1)]
        if len(ys) < 10:
            continue
        xs = np.round([d[int(y)] for y in ys]).astype(int) - x0
        v = np.max(np.stack([S[ys - y0, np.clip(xs + o, 0, S.shape[1] - 1)] for o in (-1, 0, 1)]), 0)
        out[k] = (float(v.mean()), float((v > 0.5).mean()))
    return out


def available(spec):
    try:
        import sklearn                                          # noqa: F401
    except Exception:
        return False
    if not spec:
        return False
    if spec.startswith("oof:"):
        return (Path(spec[4:]) / "line_choice_all.pkl").is_file()
    return resolve(spec) is not None


class Bag:
    """§6.297: ансамбль бустингов на подвыборках строк (разные зёрна, доля признаков) — P усредняется. Интерфейс sklearn
    (fit / predict_proba), чтобы стенд и прод брали его вместо одного HistGradientBoostingClassifier."""

    def __init__(self, make, n=5, frac=0.7):
        self.make, self.n, self.frac = make, n, frac

    def fit(self, X, y):
        rng = np.random.default_rng(0)
        self.ms = []
        for s in range(self.n):
            idx = rng.choice(len(y), size=max(1, int(self.frac * len(y))), replace=False)
            m = self.make(s)
            m.fit(X[idx], y[idx])
            self.ms.append(m)
        self.make = None                                   # фабрика не пиклится
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def _load(path):
    key = str(path)
    if key not in _CACHE:
        import pickle
        _CACHE[key] = pickle.load(open(key, "rb"))
    return _CACHE[key]


def _dups(CS):
    """дубли попарно: те же концы (±5 строк) и квантили x (±2 px) → (число дублей, есть ли дубль из другого источника)"""
    out = []
    for i, ci in enumerate(CS):
        d = [j for j, cj in enumerate(CS) if j != i and abs(ci["y0"] - cj["y0"]) <= 5 and abs(ci["y1"] - cj["y1"]) <= 5
             and np.all(np.abs(ci["xs_q"] - cj["xs_q"]) <= 2)]
        out.append((len(d), 1.0 if any(CS[j]["src"] != ci["src"] for j in d) else 0.0))
    return out


def _clusters(CS):
    """кластеры дублей (одна трасса из прод-пути и декодера) — столбцы назначения"""
    lab = list(range(len(CS)))
    for i in range(len(CS)):
        for j in range(i):
            if lab[j] != j:
                continue
            ci, cj = CS[i], CS[j]
            if abs(ci["y0"] - cj["y0"]) <= 5 and abs(ci["y1"] - cj["y1"]) <= 5 and np.all(np.abs(ci["xs_q"] - cj["xs_q"]) <= 2):
                lab[i] = j
                break
    return lab


def _choose_stack(ck, sheet, slots, CS, used, root_of, scales=None, nodup=None):
    """§6.283: два яруса + назначение 1:1 по листу (как `digitizer/rnd/_line_choice_assign.py --feat stack --extra`).
    → {имя слота: индекс кандидата или None (слот опустел: его трассу забрал другой слот)}."""
    from scipy.optimize import linear_sum_assignment
    fams, tau, dlt, nocur = dict(ck["fams"]), ck["tau"], ck["delta"], ck["nocur"]
    DU = _dups(CS) if ck.get("extra") else None
    ns = len(slots)
    ROWS = []                                             # (номер слота, k, признаки, here, elsewhere)
    for si, (name, top_y, n_rows, (xl, xr)) in enumerate(slots):
        wd = max(1.0, xr - xl)
        rows = []
        for k, cs in enumerate(CS):
            u = (cs["xs_q"] - xl) / wd
            if not (-0.1 <= u[1] <= 1.1):
                continue
            span = max(1, n_rows)
            ov = max(0, min(cs["y1"], top_y + span - 1) - max(cs["y0"], top_y) + 1) / span
            rows.append(dict(k=k, u=u.tolist(), cover=ov, here=used.get(k) == name, elsewhere=(k in used and used[k] != name)))
        if not rows:
            continue
        u50 = np.array([r["u"][1] for r in rows])
        wid = np.array([np.nan if CS[r["k"]]["style"] is None else CS[r["k"]]["style"].get("width", np.nan) for r in rows], float)
        rk_u = np.argsort(np.argsort(u50)) / max(1, len(rows) - 1)
        rk_w = np.argsort(np.argsort(np.nan_to_num(wid, nan=-1))) / max(1, len(rows) - 1)
        cur = [r for r in rows if r["here"]]
        cur_n = CS[cur[0]["k"]]["n"] if cur else np.nan; cur_cov = cur[0]["cover"] if cur else np.nan
        root = re.sub(r"^BKZ_", "", root_of(name))
        fc = fams.setdefault(root, len(fams))
        nzv = lambda v: np.nan if v is None else v
        for j, r in enumerate(rows):
            c = CS[r["k"]]; st = c["style"] or {}
            f = [1.0 if c["src"] == "dec" else 0.0, np.log1p(c["n"]), r["cover"], *r["u"],
                 nzv(st.get("width")), nzv(st.get("c1")), nzv(st.get("c2")), nzv(st.get("dark")), nzv(st.get("gap")),
                 c["rough"], fc, _probe(name), _decade(sheet), ns, len(rows), rk_u[j], rk_w[j],
                 1.0 if r["here"] else 0.0, 1.0 if r["elsewhere"] else 0.0,
                 np.log1p(c["n"]) - np.log1p(cur_n) if cur else np.nan, r["cover"] - cur_cov if cur else np.nan]
            if DU is not None:
                nr = max(1, n_rows)
                f += [nzv(st.get("width_h")), nzv(st.get("used")), DU[r["k"]][1], DU[r["k"]][0],
                      (c["y0"] - top_y) / nr, (top_y + nr - 1 - c["y1"]) / nr, c["n"] / nr]
            if ck.get("vfeat"):                           # §6.311: значения кандидата при шкале слота против приора семейства
                _sc = (scales or {}).get(name); _pr = ck["prior"].get(root)
                if _sc and _pr:
                    _vq = [_sc[0] + uq * (_sc[1] - _sc[0]) for uq in r["u"]]
                    f += [float(np.clip((v_ - _pr[0]) / _pr[1], -10, 10)) for v_ in _vq] +                          [float(np.clip((_vq[1] - _pr[2]) / max(_pr[3] - _pr[2], 1e-6), -5, 5))]
                else:
                    f += [np.nan] * 4
            if ck.get("cnn"):                             # §6.325 — после значений, как в стенде
                f += [c.get("cnn", np.nan)]
            if ck.get("segf"):                            # §6.348 — после оценки сети трассы, как в стенде
                f += list(c.get("segf", (np.nan, np.nan)))
            if ck.get("segfam"):                          # §6.350: своё семейство слота, лучшее чужое, разность
                _v = c.get("segfam")
                if _v is None or not np.isfinite(_v[0]):
                    f += [np.nan, np.nan, np.nan]
                else:
                    from .ink_seg import fam_index as _fi_
                    _v = np.asarray(_v, float)[1:]; _fi = _fi_(root); _ot = np.delete(_v, _fi).max()
                    f += [float(_v[_fi]), float(_ot), float(_v[_fi] - _ot)]
            ROWS.append((si, r["k"], f, r["here"], r["elsewhere"]))
    if not ROWS:
        return {}
    X = np.array([r[2] for r in ROWS], float)
    pa = ck["m_all"].predict_proba(X)[:, 1]
    pn = ck["m_nocur"].predict_proba(X[:, nocur])[:, 1]
    pm = 0.5 * (pa + pn)
    cl = _clusters(CS)
    by = {}
    for i, r in enumerate(ROWS):
        by.setdefault(r[0], []).append(i)
    cp = {}
    for si, idxs in by.items():
        for i in idxs:
            cp.setdefault(cl[ROWS[i][1]], []).append((si, pm[i]))
    F = np.full((len(ROWS), 7), np.nan)
    for si, idxs in by.items():
        ps = np.array([pm[i] for i in idxs])
        rank = np.empty(len(ps)); rank[np.argsort(-ps)] = np.arange(len(ps))
        here = [pm[i] for i in idxs if ROWS[i][3]]
        ph = max(here) if here else np.nan
        for j, i in enumerate(idxs):
            oth = [q for s2, q in cp[cl[ROWS[i][1]]] if s2 != si]
            om = max(oth) if oth else 0.0
            bo = max((ps[t] for t in range(len(ps)) if t != j), default=0.0)
            F[i] = [om, len(oth), rank[j] / max(1, len(ps) - 1), ps[j] - bo, ph, ps[j] - ph if here else np.nan,
                    1.0 if ps[j] >= om else 0.0]
    P = ck["m2"].predict_proba(np.hstack([X, pa[:, None], pn[:, None], F]))[:, 1]
    # назначение 1:1: вес P при P ≥ τ, нынешнему кандидату слота — P + δ; пусто — 0 (только если трассу слота забрал другой)
    S = sorted(by)
    curc = {}
    for si in S:
        here = [i for i in by[si] if ROWS[i][3]]
        curc[si] = cl[ROWS[max(here, key=lambda i: P[i])][1]] if here else None
    cols = sorted({cl[ROWS[i][1]] for si in S for i in by[si]})
    ci = {c: j for j, c in enumerate(cols)}
    NEG = -1e6
    W = np.full((len(S), len(cols) + len(S)), NEG)
    for r_, si in enumerate(S):
        best = {}
        for i in by[si]:
            c = cl[ROWS[i][1]]; best[c] = max(best.get(c, 0.0), P[i])
        for c, p in best.items():
            if c == curc[si]:
                W[r_, ci[c]] = p + dlt
            elif p >= tau:
                W[r_, ci[c]] = p
        W[r_, len(cols) + r_] = 0.0
    rr, cc = linear_sum_assignment(-W)
    out = {}
    for r_, c_ in zip(rr, cc):
        si = S[r_]
        new = cols[c_] if c_ < len(cols) else None
        if W[r_, c_] <= NEG / 2:
            new = curc[si]
        if new == curc[si]:
            continue
        if new is None and curc[si] is not None and not any(cols[c2] == curc[si] for r2, c2 in zip(rr, cc) if r2 != r_ and c2 < len(cols)):
            continue
        if new is None:
            out[slots[si][0]] = (None, 0.0)
        else:
            i = max([i for i in by[si] if cl[ROWS[i][1]] == new], key=lambda i: P[i])
            out[slots[si][0]] = (ROWS[i][1], float(P[i]))
    if nodup is not None:
        _nodup(slots, by, ROWS, P, cl, out, *nodup, sheet=sheet)
    return out


def _arr(d):
    ys = np.fromiter(d.keys(), np.int64, len(d)); xs = np.fromiter(d.values(), np.float64, len(d))
    o = np.argsort(ys)
    return ys[o], xs[o]


def _ovl(a, b, tol=3.0):
    """доли строк a и b, где другая трасса ближе tol px (по общим строкам)"""
    cm, ia, ib = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
    if not len(cm):
        return 0.0, 0.0
    k = int(np.sum(np.abs(a[1][ia] - b[1][ib]) <= tol))
    return k / max(1, len(a[0])), k / max(1, len(b[0]))


def _nodup(slots, by, ROWS, P, cl, out, theta, tau2, dense, written, rule="p", info=None, arb=None, sheet=None):
    """§6.359: после назначения 1:1 — слоты, чьи трассы совпадают (≤ 3 px) на доле строк ≥ θ (любой из двух): один из пары берёт
    лучшего своего кандидата с P ≥ τ2, который ни с одним другим слотом не совпадает на ≥ θ и не из занятого кластера. Кого
    переносить: rule "p" — слот с меньшей P; "sum" — вариант с большей суммой P пары. Нет замены — как есть. Меняет `out` (≤ 3 прохода)."""
    A = {}
    def arr_k(k):
        if k not in A:
            A[k] = _arr(dense[k]) if len(dense[k]) else None
        return A[k]
    chg0 = set(out)                                       # §6.363 (журнал): слоты, чью трассу сменило назначение 1:1
    fin = {}                                              # si → (кандидат или None, массивы трассы, P)
    for si in sorted(by):
        name = slots[si][0]
        if name in out:
            k, p = out[name]
            if k is not None:
                fin[si] = (k, arr_k(k), p)
            continue
        here = [i for i in by[si] if ROWS[i][3]]
        if here:
            i = max(here, key=lambda i: P[i]); fin[si] = (ROWS[i][1], arr_k(ROWS[i][1]), float(P[i]))
        elif name in written and len(written[name]) >= 50:
            fin[si] = (None, _arr(written[name]), 0.0)
    for si, (name, *_r) in enumerate(slots):              # слоты без строк назначения — только как препятствие
        if si not in fin and si not in by and name in written and len(written[name]) >= 50:
            fin[si] = (None, _arr(written[name]), 1.0)
    for _ in range(3):
        conf = []
        keys = sorted(fin)
        for a_ in range(len(keys)):
            for b_ in range(a_ + 1, len(keys)):
                si, sj = keys[a_], keys[b_]
                fa, fb = _ovl(fin[si][1], fin[sj][1])
                if max(fa, fb) >= theta:
                    conf.append((max(fa, fb), si, sj, fa, fb))
        if not conf:
            break
        def alt(lo):
            """лучший кандидат слота lo с P ≥ τ2, не совпадающий ни с одним другим слотом и не из занятого кластера"""
            if lo not in by:
                return None
            taken = {cl[fin[s][0]] for s in fin if s != lo and fin[s][0] is not None}
            for i in sorted(by[lo], key=lambda i: -P[i]):
                k = ROWS[i][1]
                if P[i] < tau2:
                    return None
                if k == fin[lo][0] or cl[k] in taken or arr_k(k) is None:
                    continue
                if all(max(_ovl(arr_k(k), fin[s][1])) < theta for s in fin if s != lo):
                    return i
            return None
        moved = False
        for _m, si, sj, fa, fb in sorted(conf, reverse=True):
            fa2, fb2 = _ovl(fin[si][1], fin[sj][1])       # пара могла разойтись после прежних замен этого прохода
            if max(fa2, fb2) < theta:
                continue
            # кого переносить: вариант с большей суммой P пары (как цель назначения 1:1); у трассы без строк назначения P = 0
            opts = []
            if rule in ("sum", "arb", "agree"):           # кого переносить — вариант с большей суммой P пары («arb», «agree» — так же, см. ниже)
                for lo, hi in ((si, sj), (sj, si)):
                    i = alt(lo)
                    if i is not None:
                        opts.append((float(P[i]) + fin[hi][2], lo, i))
            else:                                         # "p": переносится слот с меньшей P (при равенстве — с большей долей совпадения)
                lo = si if (fin[si][2], -fa2) < (fin[sj][2], -fb2) else sj
                i = alt(lo)
                if i is not None:
                    opts.append((float(P[i]), lo, i))
            def _p_on(s_, k_):
                v = [float(P[i]) for i in by.get(s_, []) if ROWS[i][1] == k_]
                return round(max(v), 3) if v else None
            _rec = None; _dec = None
            if os.environ.get("NDS_NODUP_LOG") or (rule == "arb" and arb is not None and opts):
                # запись конфликта: журнал стенда (§6.359 / §6.362) и признаки арбитра §6.363 — одна и та же
                _alts = {}
                for lo_, hi_ in ((si, sj), (sj, si)):
                    i_ = alt(lo_)
                    if i_ is not None:
                        k_ = ROWS[i_][1]
                        _alts[slots[lo_][0]] = [int(k_), round(float(P[i_]), 3), _p_on(hi_, k_), int(len(arr_k(k_)[0])),
                                                round(float(np.median(arr_k(k_)[1])), 1)]
                def _wov(s_):
                    w_ = written.get(slots[s_][0])
                    return round(_ovl(_arr(w_), fin[s_][1])[0], 3) if w_ is not None and len(w_) >= 50 else None
                def _rank(s_):
                    if s_ not in by or fin[s_][0] is None:
                        return None
                    o_ = sorted(by[s_], key=lambda i: -P[i])
                    return next((r for r, i in enumerate(o_) if ROWS[i][1] == fin[s_][0]), None)
                _rec_x = {"chg": [slots[si][0] in chg0, slots[sj][0] in chg0], "wov": [_wov(si), _wov(sj)],
                          "rank": [_rank(si), _rank(sj)],
                          "ncand": [sum(1 for i in by.get(s_, []) if P[i] >= 0.05) for s_ in (si, sj)],
                          "pmax": [round(max([float(P[i]) for i in by.get(s_, [])] or [0.0]), 3) for s_ in (si, sj)]}
                _rec = {"sheet": sheet, "pair": [slots[si][0], slots[sj][0]], "frac": [round(fa2, 3), round(fb2, 3)],
                        "p": [round(fin[si][2], 3), round(fin[sj][2], 3)], "own": [fin[si][0] is not None, fin[sj][0] is not None],
                        "n": [int(len(fin[si][1][0])), int(len(fin[sj][1][0]))],
                        "x": [round(float(np.median(fin[si][1][1])), 1), round(float(np.median(fin[sj][1][1])), 1)],
                        "slot": [[float(v) for v in (slots[si][3] or ())], [float(v) for v in (slots[sj][3] or ())]],
                        "y": [[int(slots[si][1]), int(slots[si][2])], [int(slots[sj][1]), int(slots[sj][2])]],
                        "alts": _alts, "opts": [[round(o[0], 3), slots[o[1]][0], int(ROWS[o[2]][1]), round(float(P[o[2]]), 3)] for o in opts],
                        "moved": slots[max(opts)[1]][0] if opts else None, "rule": rule, **_rec_x}
                if rule == "arb" and arb is not None and opts:
                    from . import nodup_arb as NA
                    _lo0 = max(opts)[1]; _hi0 = sj if _lo0 == si else si
                    # обмен = как в разметке стенда: lo берёт ТЕКУЩУЮ трассу hi (кандидат), hi — альтернативу lo; иначе обмена нет
                    _dec, _rc, _rs = NA.decide(_rec, arb[0], arb[1], can_swp=(_hi0 in by and fin[_hi0][0] is not None))
                if os.environ.get("NDS_NODUP_LOG"):
                    _w = dict(_rec, arb=[_dec, round(_rc, 3), round(_rs, 3)]) if _dec else _rec
                    with open(os.environ["NDS_NODUP_LOG"], "a", encoding="utf-8") as _f:
                        _f.write(json.dumps(_w, ensure_ascii=False) + chr(10))
            if not opts:
                continue
            _s, lo, best = max(opts)
            k = ROWS[best][1]
            if _dec == "keep":                            # §6.363: арбитр — не раздвигать
                continue
            hi = sj if lo == si else si
            if rule == "agree" and fin[lo][2] > fin[hi][2]:   # §6.363: «sum» и «p» не согласны (переносимый — вероятнее владелец) — не раздвигать
                continue
            if _dec == "swp":                             # §6.363: обмен трасс — lo берёт текущую трассу hi, hi — альтернативу lo
                kh = fin[hi][0]
                p_lh = _p_on(lo, kh); p_hk = _p_on(hi, k)
                fin[lo] = (kh, arr_k(kh), float(p_lh if p_lh is not None else fin[hi][2]))
                out[slots[lo][0]] = (kh, fin[lo][2])
                fin[hi] = (k, arr_k(k), float(p_hk if p_hk is not None else P[best]))
                out[slots[hi][0]] = (k, fin[hi][2])
                if info is not None:
                    info.setdefault("pairs", []).append((slots[si][0], slots[sj][0]))
                moved = True
                continue
            fin[lo] = (k, arr_k(k), float(P[best]))
            out[slots[lo][0]] = (k, float(P[best]))
            if info is not None:                          # §6.361: пары, раздвинутые здесь — имена им решит стиль
                info.setdefault("pairs", []).append((slots[si][0], slots[sj][0]))
            moved = True
        if not moved:
            break


def choose(sheet, slots, cands, written, path, root_of, scales=None, look=None, seg=None, segfam=None, nodup=0.0, nodup_tau=0.05, nodup_rule="p", info=None,
           nodup_arb=None):
    """slots — [(имя, top_y, n_rows, (xl, xr))] в порядке рамки (только слоты с цепочкой шкал);
    cands — [dict(src, tr — исходная трасса, dense, style, rough)];
    written — [(имя, плотная записанная трасса)] в порядке рамки (что стоит в слотах сейчас).
    → {имя слота: (индекс кандидата или None — слот опустел, P)} замен."""
    ck = _load(path)
    if ck.get("kind") == "stack":
        used = {}
        for wn, w in written:
            if len(w) < 50:
                continue
            for k, c in enumerate(cands):
                ct = c["dense"]
                com = [y for y in w if y in ct]
                if len(com) >= 0.5 * len(w) and np.mean([abs(w[y] - ct[y]) <= 1.0 for y in com[::5]]) >= 0.9:
                    used.setdefault(k, wn)
        CS = []
        for c in cands:
            ys = np.array(sorted(c["dense"])); xs = np.array([c["dense"][y] for y in ys])
            CS.append(dict(src=c["src"], n=len(c["dense"]), y0=int(ys[0]), y1=int(ys[-1]), xs_q=np.percentile(xs, [10, 50, 90]),
                           style=c["style"], rough=c["rough"]))
        if ck.get("cnn"):                                 # §6.325: оценка сети по кусочкам скана (веса рядом с моделью выбора)
            from . import trace_look as TL
            try:
                sc_ = (TL.scores(look[0], look[1], [c["dense"] for c in cands], TL.cnn_path(path)) if look is not None
                       else np.full(len(cands), np.nan))
            except Exception as e_:                       # нет torch / весов — признак пуст, выбор работает без него
                _announce(f"⚠ оценка трассы по кусочкам скана (§6.325) НЕ включена: {type(e_).__name__}: {e_} — признак пуст")
                sc_ = np.full(len(cands), np.nan)
            for c_, v_ in zip(CS, sc_):
                c_["cnn"] = float(v_)
        if ck.get("segf"):                                # §6.348: сеть туши вдоль кандидата (seg = (выпрямленный rgb, веса))
            try:
                sf_ = seg_features(seg[0], seg[1], [c["dense"] for c in cands]) if seg is not None else [(np.nan, np.nan)] * len(cands)
            except Exception as e_:
                _announce(f"⚠ признак сети туши (§6.348) НЕ включён: {type(e_).__name__}: {e_} — признак пуст")
                sf_ = [(np.nan, np.nan)] * len(cands)
            if seg is None:
                _announce("⚠ модель выбора трассы ждёт признак сети туши (§6.348), а `line_choice_seg` не задан — признак пуст")
            for c_, v_ in zip(CS, sf_):
                c_["segf"] = v_
        if ck.get("segfam"):                              # §6.350: сеть туши по семействам вдоль кандидата (segfam = (rgb, веса))
            try:
                sm_ = seg_features(segfam[0], segfam[1], [c["dense"] for c in cands], fam=True) if segfam is not None else [None] * len(cands)
            except Exception as e_:
                _announce(f"⚠ признак сети туши по семействам (§6.350) НЕ включён: {type(e_).__name__}: {e_} — признак пуст")
                sm_ = [None] * len(cands)
            if segfam is None:
                _announce("⚠ модель выбора трассы ждёт признак по семействам (§6.350), а `line_choice_segfam` не задан — признак пуст")
            for c_, v_ in zip(CS, sm_):
                c_["segfam"] = v_
        out = _choose_stack(ck, sheet, slots, CS, used, root_of, scales=scales,
                            nodup=(nodup, nodup_tau, [c["dense"] for c in cands], dict(written), nodup_rule, info, nodup_arb) if nodup > 0 else None)
        return out
    model, fams, tau, dlt = ck["model"], dict(ck["fams"]), ck["tau"], ck["delta"]
    # занятость: кандидат стоит в слоте, если x совпадает в 1 px на ≥ 90% общих строк (как в стенде по файлам выдачи)
    used = {}
    for wn, w in written:
        if len(w) < 50:
            continue
        for k, c in enumerate(cands):
            ct = c["dense"]
            com = [y for y in w if y in ct]
            if len(com) >= 0.5 * len(w) and np.mean([abs(w[y] - ct[y]) <= 1.0 for y in com[::5]]) >= 0.9:
                used.setdefault(k, wn)
    CS = []
    for c in cands:
        ys = np.array(sorted(c["dense"])); xs = np.array([c["dense"][y] for y in ys])
        CS.append(dict(src=c["src"], n=len(c["dense"]), y0=int(ys[0]), y1=int(ys[-1]), xs_q=np.percentile(xs, [10, 50, 90]),
                       style=c["style"], rough=c["rough"]))
    ns = len(slots)
    props = []
    for name, top_y, n_rows, (xl, xr) in slots:
        wd = max(1.0, xr - xl)
        rows = []
        for k, cs in enumerate(CS):
            u = (cs["xs_q"] - xl) / wd
            if not (-0.1 <= u[1] <= 1.1):
                continue
            span = max(1, n_rows)
            ov = max(0, min(cs["y1"], top_y + span - 1) - max(cs["y0"], top_y) + 1) / span
            rows.append(dict(k=k, u=u.tolist(), cover=ov, here=used.get(k) == name, elsewhere=(k in used and used[k] != name)))
        if not rows:
            continue
        u50 = np.array([r["u"][1] for r in rows])
        wid = np.array([np.nan if CS[r["k"]]["style"] is None else CS[r["k"]]["style"].get("width", np.nan) for r in rows], float)
        rk_u = np.argsort(np.argsort(u50)) / max(1, len(rows) - 1)
        rk_w = np.argsort(np.argsort(np.nan_to_num(wid, nan=-1))) / max(1, len(rows) - 1)
        cur = [r for r in rows if r["here"]]
        cur_n = CS[cur[0]["k"]]["n"] if cur else np.nan; cur_cov = cur[0]["cover"] if cur else np.nan
        root = re.sub(r"^BKZ_", "", root_of(name))
        fc = fams.setdefault(root, len(fams))
        X = []
        for j, r in enumerate(rows):
            c = CS[r["k"]]; st = c["style"] or {}
            nzv = lambda v: np.nan if v is None else v
            X.append([1.0 if c["src"] == "dec" else 0.0, np.log1p(c["n"]), r["cover"], *r["u"],
                      nzv(st.get("width")), nzv(st.get("c1")), nzv(st.get("c2")), nzv(st.get("dark")), nzv(st.get("gap")),
                      c["rough"], fc, _probe(name), _decade(sheet), ns, len(rows), rk_u[j], rk_w[j],
                      1.0 if r["here"] else 0.0, 1.0 if r["elsewhere"] else 0.0,
                      np.log1p(c["n"]) - np.log1p(cur_n) if cur else np.nan, r["cover"] - cur_cov if cur else np.nan])
        P = model.predict_proba(np.array(X, float))[:, 1]
        here = [j for j, r in enumerate(rows) if r["here"]]
        p_here = P[here[0]] if here else 0.0
        free = [j for j, r in enumerate(rows) if not r["here"] and not r["elsewhere"]]
        if not free:
            continue
        b = max(free, key=lambda j: P[j])
        if P[b] >= tau and P[b] - p_here >= dlt:
            props.append((float(P[b] - p_here), name, rows[b]["k"], float(P[b])))
    taken, out = set(), {}
    for gain, name, k, p in sorted(props, reverse=True):
        if k in taken:
            continue
        taken.add(k); out[name] = (k, p)
    return out
