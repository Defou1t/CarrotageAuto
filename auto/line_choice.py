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


def _choose_stack(ck, sheet, slots, CS, used, root_of):
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
    return out


def choose(sheet, slots, cands, written, path, root_of):
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
        return _choose_stack(ck, sheet, slots, CS, used, root_of)
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
