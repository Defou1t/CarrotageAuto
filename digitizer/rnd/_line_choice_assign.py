r"""_line_choice_assign.py — ВЫБОР ТРАССЫ ДЛЯ СЛОТА ПОЛНЫМ НАЗНАЧЕНИЕМ ПО ЛИСТУ (04.10, к §6.283).

`_line_choice_miss.py`: у 229 слотов поля (сорт A — 100) честная трасса стоит в ДРУГОМ слоте выдачи, где она нечестна; нынешний
выбор (§6.280) берёт только свободных кандидатов. Здесь — назначение 1:1 по листу (венгерский алгоритм) по тем же P (бустинг
`_line_choice.py`, поле — 5 фолдов по скважинам, сорт A — модель на всём поле):
- вес пары слот × кандидат = P, если P ≥ τ; нынешнему кандидату слота — P + δ (держит, пока замена не выгоднее на δ);
- кандидаты-дубли (одна трасса из прод-пути и декодера) — один столбец;
- слот может остаться пустым, только если его трассу забрал другой слот.
Мера — честность в слоте до и после по меткам (слоты с эталоном; слот без выдачи — «до» нечестен). Режим `free` — нынешнее правило
(только свободные, жадно) в той же мере, для сравнения.

  _line_choice_assign.py --grid free:0.2:0 hung:0.2:0 hung:0.3:0.1
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--data", default=r"F:/nds/output/taskS/line_choice")
ap.add_argument("--grid", nargs="+", default=["free:0.2:0", "hung:0.2:0", "hung:0.3:0.1", "hung:0.5:0.2"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--save-dir", default="", help="(stack) модели обоих ярусов: фолды и на всём поле, карта лист → фолд, τ/δ первой "
                "точки сетки hung")
ap.add_argument("--trace", default="", help="каталог `_line_choice_trace.py`: + признаки формы трассы (прямизна, размах, дрожание, "
                "скачки, неизменный x)")
ap.add_argument("--bag", type=int, default=1, help="§6.297: ансамбль из N бустингов на 70%% строк (зёрна, доля признаков 0.8)")
ap.add_argument("--hgb", default="300,0.08,31,1.0,20", help="бустинг: итераций, шаг, листьев, l2, мин. в листе")
ap.add_argument("--norm", default="", help="нормировка стиля внутри листа: z — добавить z-оценки толщины, темноты, цвета по "
                "кандидатам листа; zonly — то же, абсолютные значения стиля убрать")
ap.add_argument("--data-extra", default="", help="§6.295: скан листов ВНЕ поля и сорта A — только в обучение (листы скважин поля — "
                "по фолдам поля, скважин сорта A — исключены)")
ap.add_argument("--well-frac", type=float, default=1.0, help="кривая обучения: доля скважин обучения в каждом фолде (случайно)")
ap.add_argument("--well-seed", type=int, default=0)
ap.add_argument("--insample", action="store_true", help="(all/nocur/mix) поле — модель на ВСЁМ поле, без фолдов: потолок признаков")
ap.add_argument("--extra", action="store_true", help="+ признаки: width_h, used, дубль из другого источника, число дублей, "
                "отступ начала / конца от строк слота, длина / строки слота")
ap.add_argument("--feat", default="all", choices=["all", "nocur", "mix", "stack"],
                help="nocur — без признаков отношения к нынешней раскладке (here, elsewhere, разницы с нынешним); mix — среднее P двух моделей; "
                     "stack — второй ярус: исходные признаки + P обеих моделей (вне фолда, вложенно) + конкуренция за кандидата между слотами")
ap.add_argument("--dump", default=r"F:/nds/output/taskS/line_choice_assign.pkl")
a = ap.parse_args()
from sklearn.ensemble import HistGradientBoostingClassifier
TS = Path(a.ts)
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else sh
FAMS = {}
fcode = lambda f: FAMS.setdefault(f, len(FAMS))


def probe(name):
    m = re.match(r"^(?:BKZ_)?[A-Z]+?(\d)", name.split()[0])
    return int(m.group(1)) if m else 0


def decade(sh):
    m = re.search(r"(19[5-9]\d|20[0-2]\d)", sh)
    return (int(m.group(1)) // 10 * 10) if m else 0


def nz(v):
    return np.nan if v is None else v


def dups(CS):
    """дубли попарно: те же концы (±5 строк) и квантили x (±2 px) → (число дублей, есть ли дубль из другого источника)"""
    out = []
    for i, ci in enumerate(CS):
        d = [j for j, cj in enumerate(CS) if j != i and abs(ci["y0"] - cj["y0"]) <= 5 and abs(ci["y1"] - cj["y1"]) <= 5
             and np.all(np.abs(ci["xs_q"] - cj["xs_q"]) <= 2)]
        out.append((len(d), 1.0 if any(CS[j]["src"] != ci["src"] for j in d) else 0.0))
    return out


RECS = [pickle.load(open(f, "rb")) for f in sorted(Path(a.data).glob("*.pkl"))]
OUT = set()
if a.data_extra:                                          # §6.295: листы вне поля и сорта A — только обучение
    _rx = [pickle.load(open(f, "rb")) for f in sorted(Path(a.data_extra).glob("*.pkl"))]
    _rx = [r for r in _rx if r["sheet"] not in FIELD and r["sheet"] not in HOLD]
    OUT = {r["sheet"] for r in _rx}
    RECS += _rx
TRF = {}
if a.trace:
    for f in sorted(Path(a.data).glob("*.pkl")):
        t = pickle.load(open(Path(a.trace) / f.name, "rb"))
        TRF[t["sheet"]] = t["feats"]
SHAPE = ["straight", "xr_px", "mad_dx", "jumps", "flat", "agree_src", "agree_any"]
ROWS = []
for rec in RECS:
    sh = rec["sheet"]; CS = rec["cands"]; ns = len(rec["slots"])
    DU = dups(CS) if a.extra else None
    if a.norm:                                           # нормировка стиля по кандидатам листа
        _st = {}
        for key_ in ("width", "dark", "c1", "c2", "gap"):
            v_ = np.array([nz((c_["style"] or {}).get(key_)) for c_ in CS], float)
            mu_, sd_ = np.nanmedian(v_) if np.isfinite(v_).any() else 0.0, np.nanstd(v_) if np.isfinite(v_).sum() > 1 else 1.0
            _st[key_] = (v_ - mu_) / (sd_ if sd_ and np.isfinite(sd_) else 1.0)
    if a.trace:
        assert len(TRF[sh]) == len(CS), (sh, len(TRF[sh]), len(CS))
    for si, s in enumerate(rec["slots"]):
        cs = s["cands"]
        if not cs:
            continue
        u50 = np.array([r["u"][1] for r in cs]); wid = np.array([nz((CS[r["k"]]["style"] or {}).get("width")) for r in cs], float)
        rk_u = np.argsort(np.argsort(u50)) / max(1, len(cs) - 1)
        rk_w = np.argsort(np.argsort(np.nan_to_num(wid, nan=-1))) / max(1, len(cs) - 1)
        cur = [r for r in cs if r["here"]]
        cur_n = CS[cur[0]["k"]]["n"] if cur else np.nan; cur_cov = cur[0]["cover"] if cur else np.nan
        for j, r in enumerate(cs):
            c = CS[r["k"]]; st = c["style"] or {}
            f = [1.0 if c["src"] == "dec" else 0.0, np.log1p(c["n"]), r["cover"], *r["u"],
                 nz(st.get("width")), nz(st.get("c1")), nz(st.get("c2")), nz(st.get("dark")), nz(st.get("gap")), c["rough"],
                 fcode(s["root"]), probe(s["name"]), decade(sh), ns, len(cs), rk_u[j], rk_w[j],
                 1.0 if r["here"] else 0.0, 1.0 if r["elsewhere"] else 0.0,
                 np.log1p(c["n"]) - np.log1p(cur_n) if cur else np.nan, r["cover"] - cur_cov if cur else np.nan]
            if a.extra:
                nr = max(1, s["n_rows"])
                f += [nz(st.get("width_h")), nz(st.get("used")), DU[r["k"]][1], DU[r["k"]][0],
                      (c["y0"] - s["top_y"]) / nr, (s["top_y"] + nr - 1 - c["y1"]) / nr, c["n"] / nr]
            if a.trace:
                f += [TRF[sh][r["k"]][x] for x in SHAPE]
            if a.norm:
                f += [_st[key_][r["k"]] for key_ in ("width", "dark", "c1", "c2", "gap")]
                if a.norm == "zonly":
                    for j_ in (6, 7, 8, 9, 10):                     # абсолютные width, c1, c2, dark, gap
                        f[j_] = np.nan
            lab = None if r["lab"] is None else int(r["lab"][0])
            ROWS.append((sh, si, r["k"], f, lab, r["here"], r["elsewhere"]))
X_ALL = np.array([r[3] for r in ROWS], float)
CAT = [12]
NOCUR = [j for j in range(X_ALL.shape[1]) if j not in (19, 20, 21, 22)]
X = X_ALL


HGB = [float(v) for v in a.hgb.split(",")]


def _hgb(cat):
    if a.bag > 1:
        from auto.line_choice import Bag
        return Bag(lambda s: _hgb1(cat, s, 0.8), n=a.bag)
    return _hgb1(cat, 0, 1.0)


def _hgb1(cat, seed, mf):
    return HistGradientBoostingClassifier(max_features=mf, max_iter=int(HGB[0]), learning_rate=HGB[1], max_leaf_nodes=int(HGB[2]),
                                          l2_regularization=HGB[3], min_samples_leaf=int(HGB[4]), categorical_features=cat,
                                          random_state=seed)


def fit(sheets):
    idx = [i for i, r in enumerate(ROWS) if r[0] in sheets and r[4] is not None]
    cat = np.zeros(X.shape[1], bool); cat[CAT] = True
    m = _hgb(cat)
    m.fit(X[idx], np.array([ROWS[i][4] for i in idx]))
    return m


wells = sorted({well(r["sheet"]) for r in RECS if r["sheet"] in FIELD})
FOLD = {w: i % a.folds for i, w in enumerate(wells)}
HOLD_W = {well(x) for x in HOLD}


def out_train(k=None):
    """§6.295: листы вне поля, допустимые в обучение: не из скважин сорта A; из скважин поля — только не держанного фолда k"""
    return {x for x in OUT if well(x) not in HOLD_W and (k is None or FOLD.get(well(x)) != k)}


def oof(cols):
    global X
    X = X_ALL[:, cols]
    P = np.full(len(ROWS), np.nan)
    if a.insample:                                       # потолок признаков: поле предсказывается моделью, видевшей его
        m = fit(FIELD)
        te = [i for i, r in enumerate(ROWS) if r[0] in FIELD or r[0] in HOLD]
        P[te] = m.predict_proba(X[te])[:, 1]
        return P
    rs_ = np.random.default_rng(a.well_seed)
    for k in range(a.folds):
        tw = sorted({well(r["sheet"]) for r in RECS if r["sheet"] in FIELD and FOLD[well(r["sheet"])] != k})
        if a.well_frac < 1.0:                              # кривая обучения: часть скважин обучения
            tw = list(rs_.choice(tw, max(1, int(round(a.well_frac * len(tw)))), replace=False))
        tw = set(tw)
        tr = {r["sheet"] for r in RECS if r["sheet"] in FIELD and well(r["sheet"]) in tw} | out_train(k)
        te = [i for i, r in enumerate(ROWS) if r[0] in FIELD and FOLD[well(r[0])] == k]
        P[te] = fit(tr).predict_proba(X[te])[:, 1]
    te = [i for i, r in enumerate(ROWS) if r[0] in HOLD]
    allw = sorted({well(r["sheet"]) for r in RECS if r["sheet"] in FIELD})
    if a.well_frac < 1.0:
        allw = list(rs_.choice(allw, max(1, int(round(a.well_frac * len(allw)))), replace=False))
    P[te] = fit({r["sheet"] for r in RECS if r["sheet"] in FIELD and well(r["sheet"]) in set(allw)} | out_train()).predict_proba(X[te])[:, 1]
    return P


def clusters(CS):
    """дубли: одна трасса из прод-пути и декодера — те же концы (±5 строк) и квантили x (±2 px)"""
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


ALLC = list(range(X_ALL.shape[1]))
Y = np.array([-1 if r[4] is None else r[4] for r in ROWS])


def fit_idx(idx, Xm):
    cat = np.zeros(Xm.shape[1], bool); cat[CAT] = True
    m = _hgb(cat)
    m.fit(Xm[idx], Y[idx])
    return m


ROWS_OF = defaultdict(list)
for i, r in enumerate(ROWS):
    ROWS_OF[r[0]].append(i)


def rows_in(sheets, labeled=False):
    return [i for sh in sheets for i in ROWS_OF.get(sh, []) if not labeled or Y[i] >= 0]


def s1_oof(sheets, cols, nf):
    ws = sorted({well(x) for x in sheets}); fo = {w: i % nf for i, w in enumerate(ws)}
    P_ = np.full(len(ROWS), np.nan)
    for k in range(nf):
        tr = rows_in([x for x in sheets if fo[well(x)] != k], True)
        te = rows_in([x for x in sheets if fo[well(x)] == k])
        P_[te] = fit_idx(tr, X_ALL[:, cols]).predict_proba(X_ALL[te][:, cols])[:, 1]
    return P_


def cross(p, sheets):
    """конкуренция: лучший P того же кандидата (кластера дублей) в других слотах, в скольких слотах он есть, ранг и отрыв в своём
    слоте, P нынешнего кандидата слота и разница с ним, «этот слот для кандидата лучший»"""
    F = np.full((len(ROWS), 7), np.nan)
    for sh in sheets:
        by = defaultdict(list)
        for i in ROWS_OF.get(sh, []):
            by[ROWS[i][1]].append(i)
        cl = clusters(RECD[sh]["cands"])
        cp = defaultdict(list)
        for si, idxs in by.items():
            for i in idxs:
                cp[cl[ROWS[i][2]]].append((si, p[i]))
        for si, idxs in by.items():
            ps = np.array([p[i] for i in idxs])
            rank = np.empty(len(ps)); rank[np.argsort(-ps)] = np.arange(len(ps))
            here = [p[i] for i in idxs if ROWS[i][5]]
            ph = max(here) if here else np.nan
            for j, i in enumerate(idxs):
                oth = [q for s2, q in cp[cl[ROWS[i][2]]] if s2 != si]
                om = max(oth) if oth else 0.0
                bo = max((ps[t] for t in range(len(ps)) if t != j), default=0.0)
                F[i] = [om, len(oth), rank[j] / max(1, len(ps) - 1), ps[j] - bo, ph, ps[j] - ph if here else np.nan,
                        1.0 if ps[j] >= om else 0.0]
    return F


def stage2(train_sheets, test_sheets, nf_inner):
    pa = s1_oof(train_sheets, ALLC, nf_inner); pn = s1_oof(train_sheets, NOCUR, nf_inner)
    trl = rows_in(train_sheets, True); te = rows_in(test_sheets)
    ma = fit_idx(trl, X_ALL[:, ALLC]); mn = fit_idx(trl, X_ALL[:, NOCUR])
    pa[te] = ma.predict_proba(X_ALL[te][:, ALLC])[:, 1]
    pn[te] = mn.predict_proba(X_ALL[te][:, NOCUR])[:, 1]
    Fx = cross(0.5 * (pa + pn), list(train_sheets) + list(test_sheets))
    X2 = np.hstack([X_ALL, pa[:, None], pn[:, None], Fx])
    m2 = fit_idx(trl, X2)
    return m2.predict_proba(X2[te])[:, 1], te, (ma, mn, m2)


RECD = {rec["sheet"]: rec for rec in RECS}
if a.feat == "all":
    P = oof(ALLC)
elif a.feat == "nocur":
    P = oof(NOCUR)
elif a.feat == "mix":
    P = 0.5 * (oof(ALLC) + oof(NOCUR))
else:
    P = np.full(len(ROWS), np.nan)
    fs = [x for x in ROWS_OF if x in FIELD]
    MODELS = {}
    for k in range(a.folds):
        p2, te, MODELS[k] = stage2([x for x in fs if FOLD[well(x)] != k] + sorted(out_train(k)), [x for x in fs if FOLD[well(x)] == k],
                                   a.folds - 1)
        P[te] = p2
        print(f"  второй ярус: фолд {k} готов")
    p2, te, MODELS[None] = stage2(fs + sorted(out_train()), [x for x in ROWS_OF if x in HOLD], a.folds)
    P[te] = p2
print(f"листов {len(RECS)}, пар {len(ROWS)}; P посчитаны вне фолда")


def solve(rec, idx_by_slot, mode, tau, dlt):
    """→ {si: k или None} для слотов, где назначение меняется"""
    CS = rec["cands"]; cl = clusters(CS)
    slots = sorted(idx_by_slot)
    cur = {}
    for si in slots:
        here = [i for i in idx_by_slot[si] if ROWS[i][5]]
        cur[si] = cl[ROWS[max(here, key=lambda i: P[i])][2]] if here else None
    if mode == "free":
        used = {cl[ROWS[i][2]] for si in slots for i in idx_by_slot[si] if ROWS[i][5] or ROWS[i][6]}
        props = []
        for si in slots:
            here = [i for i in idx_by_slot[si] if ROWS[i][5]]
            ph = max((P[i] for i in here), default=0.0)
            free = [i for i in idx_by_slot[si] if cl[ROWS[i][2]] not in used]
            if not free:
                continue
            b = max(free, key=lambda i: P[i])
            if P[b] >= tau and P[b] - ph >= dlt:
                props.append((P[b] - ph, si, cl[ROWS[b][2]]))
        taken, out = set(), {}
        for g, si, c in sorted(props, reverse=True):
            if c in taken:
                continue
            taken.add(c); out[si] = c
        return out
    cols = sorted({cl[ROWS[i][2]] for si in slots for i in idx_by_slot[si]})
    ci = {c: j for j, c in enumerate(cols)}
    NEG = -1e6
    W = np.full((len(slots), len(cols) + len(slots)), NEG)
    for r_, si in enumerate(slots):
        best = defaultdict(float)
        for i in idx_by_slot[si]:
            c = cl[ROWS[i][2]]; best[c] = max(best[c], P[i])
        for c, p in best.items():
            if c == cur[si]:
                W[r_, ci[c]] = p + dlt
            elif p >= tau:
                W[r_, ci[c]] = p
        W[r_, len(cols) + r_] = 0.0                       # пусто — только если нынешнюю трассу забрал другой слот (вес ниже)
    rr, cc = linear_sum_assignment(-W)
    out = {}
    for r_, c_ in zip(rr, cc):
        si = slots[r_]
        new = cols[c_] if c_ < len(cols) else None
        if W[r_, c_] <= NEG / 2:
            new = cur[si]
        if new != cur[si]:
            if new is None and cur[si] is not None and not any(cols[c2] == cur[si] for r2, c2 in zip(rr, cc) if r2 != r_ and c2 < len(cols)):
                continue                                  # никто не забрал — держим
            out[si] = new
    return out


BY = defaultdict(lambda: defaultdict(list))
for i, r in enumerate(ROWS):
    BY[r[0]][r[1]].append(i)
RECD = {rec["sheet"]: rec for rec in RECS}
rng = np.random.default_rng(0)
RES = {}
ALLCH = {}                                                # все замены (и в слотах без эталона) — для сверки с продом
for g in a.grid:
    mode, tau, dlt = g.split(":"); tau = float(tau); dlt = float(dlt)
    D = Counter(); SW = []; AC = []
    for sh, by in BY.items():
        if sh in OUT:
            continue                                      # листы вне поля — только обучение
        rec = RECD[sh]; cl = clusters(rec["cands"])
        ch = solve(rec, by, mode, tau, dlt)
        for si, c in ch.items():
            s = rec["slots"][si]
            km = None if c is None else ROWS[max([i for i in by[si] if cl[ROWS[i][2]] == c], key=lambda i: P[i])][2]
            AC.append((sh, s["name"], km))
            if not s["has_truth"]:
                continue
            before = int(bool(s["out_honest"]))
            after = 0
            if c is not None:                             # пишется член кластера с наибольшим P в этом слоте
                mem = [i for i in by[si] if cl[ROWS[i][2]] == c]
                after = int(ROWS[max(mem, key=lambda i: P[i])][4] or 0)
            D[sh] += after - before
            SW.append(dict(sheet=sh, slot=s["name"], before=bool(before), after=bool(after), empty=c is None))
    RES[g] = SW; ALLCH[g] = AC
    for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
        d = np.array([D.get(x, 0) for x in S]); nzv = d[d != 0]
        p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nzv))) * nzv).sum(1)) >= abs(d.sum()))) if len(nzv) else 1.0
        sw = [x for x in SW if x["sheet"] in S]
        print(f"★ {g:16s} {sn:6s}: замен {len(sw):4d} (стала честной {sum(1 for x in sw if x['after'] and not x['before']):3d}, "
              f"перестала {sum(1 for x in sw if x['before'] and not x['after']):3d}, опустело {sum(1 for x in sw if x['empty']):3d}); "
              f"Δ {int(d.sum()):+4d} (листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
pickle.dump(dict(res=RES, all_changes=ALLCH, P=P, rows=[(r[0], r[1], r[2], r[4], r[5], r[6]) for r in ROWS], X=X_ALL), open(a.dump, "wb"))
if a.save_dir and a.feat == "stack":
    import json
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    g = next(x for x in a.grid if x.startswith("hung:"))
    _, tau, dlt = g.split(":")
    meta = dict(kind="stack", fams=dict(FAMS), cat=CAT, nocur=NOCUR, extra=bool(a.extra), shape=bool(a.trace), tau=float(tau),
                delta=float(dlt))
    for k, (ma, mn, m2) in MODELS.items():
        nm = "line_choice_all.pkl" if k is None else f"line_choice_f{k}.pkl"
        pickle.dump(dict(meta, m_all=ma, m_nocur=mn, m2=m2, fold=k), open(sd / nm, "wb"))
    (sd / "line_choice_folds.json").write_text(json.dumps({r["sheet"]: FOLD[well(r["sheet"])] for r in RECS if r["sheet"] in FIELD},
                                                          ensure_ascii=False), encoding="utf-8")
    print(f"модели сохранены: {sd} (τ {tau}, δ {dlt})")
