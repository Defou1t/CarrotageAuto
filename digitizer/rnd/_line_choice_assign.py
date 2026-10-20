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
ap.add_argument("--bag-frac", type=float, default=0.7, help="§6.301: доля строк в подвыборке ансамбля")
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
ap.add_argument("--pairwise", action="store_true", help="§6.302: + признак второго яруса — доля побед кандидата в парах внутри слота (парная модель по разностям признаков, вложенно вне фолда)")
ap.add_argument("--prod-all", action="store_true", help="§6.302: модель на всём (line_choice_all) — ещё и на сорте A; только для прода")
ap.add_argument("--eval-out", action="store_true", help="§6.308: листы вне поля — в фолды по скважинам вместе с полем и в замер (отдельной строкой «вне»)")
ap.add_argument("--vfeat", default="", help="§6.311: `_slot_scale.py` — + правдоподобие значения кандидата при базовой шкале слота (квантили, нормированные на приор семейства)")
ap.add_argument("--vreg", action="store_true", help="§6.313: (с --vfeat) + те же признаки по приору района (без своей скважины)")
ap.add_argument("--vchain", action="store_true", help="§6.315: (с --vfeat) + значение при каждом уровне цепочки шкал слота: min |z|, его уровень, z при уровне 1")
ap.add_argument("--ovl", default="", help="§6.316: каталог `_cand_overlap.py` — + перекрытие кандидата с другими (лучший, второй, число частичных, покрытие)")
ap.add_argument("--wellcons", action="store_true", help="§6.320: (stack) второй ярус + согласие с соседними листами той же скважины "
                "(без меток): разница положения, стиля, значения кандидата с консенсусом лучших по P трасс того же слота на других листах")
ap.add_argument("--swap", action="store_true", help="§6.324: после назначения — модель обмена трасс двух слотов (пары слотов, где обе трассы — "
                "кандидаты обоих слотов): P четырёх сочетаний, разности стиля трасс, значения при шкалах обоих слотов; вне фолда по скважинам")
ap.add_argument("--cnn", default="", help="§6.325: `_trace_cnn.py` — + оценка сети «настоящая ли трасса» по кусочкам скана (вне фолда)")
ap.add_argument("--miss", action="store_true", help="§6.301: разбор слотов, где честный кандидат есть, а после назначения слот нечестен")
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
FIELD0 = set(FIELD); EVO = set()
if a.eval_out:                                            # §6.308: листы вне поля (кроме скважин сорта A) — в фолды по скважинам и в замер
    _hw = {well(x) for x in HOLD}
    EVO = {x for x in OUT if well(x) not in _hw}
    FIELD = FIELD | EVO
    OUT = OUT - EVO
    print(f"--eval-out: листов вне поля в фолдах и замере {len(EVO)}; поле теперь {len(FIELD)}")
VSC = pickle.load(open(a.vfeat, "rb")) if a.vfeat else None
OVL = {}
if a.ovl:                                                 # §6.316
    import hashlib as _hl
    for _r in RECS:
        _f = Path(a.ovl) / (_hl.md5(_r["sheet"].encode("utf-8")).hexdigest()[:16] + ".pkl")
        if _f.exists():
            OVL[_r["sheet"]] = pickle.load(open(_f, "rb"))
    print(f"перекрытия кандидатов: листов {len(OVL)} из {len(RECS)}")
CNN = pickle.load(open(a.cnn, "rb")) if a.cnn else {}
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
            if a.vfeat:                                  # §6.311: значения кандидата при шкале слота против приора семейства
                _sc = VSC["scale"].get(sh, {}).get(s["name"]); _pr = VSC["prior"].get(re.sub(r"^BKZ_", "", s["root"]))
                if _sc and _pr:
                    _vq = [_sc[0] + uq * (_sc[1] - _sc[0]) for uq in r["u"]]
                    f += [float(np.clip((v_ - _pr[0]) / _pr[1], -10, 10)) for v_ in _vq] +                          [float(np.clip((_vq[1] - _pr[2]) / max(_pr[3] - _pr[2], 1e-6), -5, 5))]
                else:
                    f += [np.nan] * 4
                pass
            if a.ovl:                                    # §6.316: перекрытие кандидата с другими — признак трассы-смеси
                _ov = OVL.get(sh)
                f += (_ov[r["k"]].tolist() if _ov is not None and r["k"] < len(_ov) else [np.nan] * 4)
            if a.vfeat:
                if a.vchain:                             # §6.315: значение при каждом уровне цепочки шкал слота
                    _chn = VSC["chain"].get(sh, {}).get(s["name"])
                    if _chn and _pr and len(_chn) >= 2:
                        zs = [abs((vl_ + r["u"][1] * (vr_ - vl_) - _pr[0]) / _pr[1]) for vl_, vr_ in _chn]
                        z1 = (_chn[1][0] + r["u"][1] * (_chn[1][1] - _chn[1][0]) - _pr[0]) / _pr[1]
                        f += [float(min(min(zs), 10)), float(int(np.argmin(zs))), float(np.clip(z1, -10, 10))]
                    else:
                        f += [np.nan] * 3
                if a.vreg:                               # §6.313: приор района (без своей скважины)
                    _rp = VSC["prior_reg"].get(sh, {}).get(re.sub(r"^BKZ_", "", s["root"]))
                    if _sc and _rp:
                        _vq = [_sc[0] + uq * (_sc[1] - _sc[0]) for uq in r["u"]]
                        f += [float(np.clip((v_ - _rp[0]) / _rp[1], -10, 10)) for v_ in _vq] +                              [float(np.clip((_vq[1] - _rp[2]) / max(_rp[3] - _rp[2], 1e-6), -5, 5))]
                    else:
                        f += [np.nan] * 4
            if a.cnn:                                    # §6.325
                f += [CNN.get((sh, r["k"]), np.nan)]
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
        return Bag(lambda s: _hgb1(cat, s, 0.8), n=a.bag, frac=a.bag_frac)
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


def _slot_groups(sheets):
    g = defaultdict(list)
    for sh in sheets:
        for i in ROWS_OF.get(sh, []):
            g[(sh, ROWS[i][1])].append(i)
    return list(g.values())


def pw_fit(sheets, cols):
    """§6.302: парная модель «i честнее j» по разностям признаков (пары честный × нечестный внутри слота, до 30 на слот)"""
    rs = np.random.default_rng(0); XP, YP = [], []
    for idx in _slot_groups(sheets):
        pos = [i for i in idx if Y[i] == 1]; neg = [i for i in idx if Y[i] == 0]
        pr = [(i, j) for i in pos for j in neg]
        if not pr:
            continue
        if len(pr) > 30:
            pr = [pr[t] for t in rs.choice(len(pr), 30, replace=False)]
        for i, j in pr:
            d = X_ALL[i, cols] - X_ALL[j, cols]
            XP.append(d); YP.append(1); XP.append(-d); YP.append(0)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                       min_samples_leaf=20, random_state=0)
    m.fit(np.array(XP), np.array(YP))
    return m


def pw_score(m, sheets, cols, out):
    """доля побед строки над остальными кандидатами слота"""
    for idx in _slot_groups(sheets):
        if len(idx) < 2:
            out[idx] = 0.5; continue
        A = X_ALL[idx][:, cols]
        D = (A[:, None, :] - A[None, :, :]).reshape(-1, len(cols))
        pp = m.predict_proba(D)[:, 1].reshape(len(idx), len(idx))
        np.fill_diagonal(pp, np.nan)
        out[idx] = np.nanmean(pp, 1)


PWC = [j for j in range(X_ALL.shape[1]) if j not in (12, 13, 14)]   # без категориальных кодов (семейство, зонд, десятилетие)


def pw_oof(sheets, nf):
    ws = sorted({well(x) for x in sheets}); fo = {w: i % nf for i, w in enumerate(ws)}
    out = np.full(len(ROWS), np.nan)
    for k in range(nf):
        m = pw_fit([x for x in sheets if fo[well(x)] != k], PWC)
        pw_score(m, [x for x in sheets if fo[well(x)] == k], PWC, out)
    return out


WC_COLS = [4, 6, 7, 8, 9]                                 # u50, толщина, c1, c2, темнота


def wellcons(p, sheets):
    """§6.320: на других листах той же скважины — лучший по P кандидат слота с тем же именем; консенсус = среднее его признаков,
    взвешенное P. Для строки: разница её признаков с консенсусом, |разница| u50 и z50, число листов, средний P консенсуса.
    Метки не нужны: P вне фолда (обучение) / моделью обучения (проверка); скважины обучения и проверки не пересекаются."""
    cols = WC_COLS + ([31] if a.vfeat else [])
    best = defaultdict(list)
    names = {}
    for sh in sheets:
        by = defaultdict(list)
        for i in ROWS_OF.get(sh, []):
            by[ROWS[i][1]].append(i)
        for si, idxs in by.items():
            b = max(idxs, key=lambda i: p[i])
            nm = RECD[sh]["slots"][si]["name"]; names[(sh, si)] = nm
            best[(well(sh), nm)].append((sh, float(p[b]), X_ALL[b, cols]))
    F = np.full((len(ROWS), len(cols) + 4), np.nan)
    for sh in sheets:
        w = well(sh); cons = {}
        for i in ROWS_OF.get(sh, []):
            si = ROWS[i][1]
            if si not in cons:
                L = [(pp, ff) for s2, pp, ff in best.get((w, names[(sh, si)]), []) if s2 != sh]
                if L:
                    W_ = np.array([pp for pp, _ in L]); FF = np.array([ff for _, ff in L], float)
                    ok = np.isfinite(FF)
                    den = (ok * W_[:, None]).sum(0)
                    cons[si] = (np.where(den > 0, np.nansum(FF * W_[:, None], 0) / np.maximum(den, 1e-9), np.nan), len(L), W_.mean())
                else:
                    cons[si] = None
            if cons[si] is None:
                continue
            c_, n_, pm_ = cons[si]
            d = X_ALL[i, cols] - c_
            F[i] = list(d) + [abs(d[0]), abs(d[-1]) if a.vfeat else np.nan, n_, pm_]
    return F


def stage2(train_sheets, test_sheets, nf_inner):
    pa = s1_oof(train_sheets, ALLC, nf_inner); pn = s1_oof(train_sheets, NOCUR, nf_inner)
    trl = rows_in(train_sheets, True); te = rows_in(test_sheets)
    ma = fit_idx(trl, X_ALL[:, ALLC]); mn = fit_idx(trl, X_ALL[:, NOCUR])
    pa[te] = ma.predict_proba(X_ALL[te][:, ALLC])[:, 1]
    pn[te] = mn.predict_proba(X_ALL[te][:, NOCUR])[:, 1]
    Fx = cross(0.5 * (pa + pn), list(train_sheets) + list(test_sheets))
    X2 = np.hstack([X_ALL, pa[:, None], pn[:, None], Fx])
    if a.wellcons:                                       # §6.320: согласие с листами той же скважины
        X2 = np.hstack([X2, wellcons(0.5 * (pa + pn), list(train_sheets) + list(test_sheets))])
    if a.pairwise:                                       # §6.302: вложенно — обучение по внутренним фолдам, проверка моделью на всём обучении
        pw = pw_oof(train_sheets, nf_inner)
        pw_score(pw_fit(train_sheets, PWC), test_sheets, PWC, pw)
        X2 = np.hstack([X2, pw[:, None]])
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
    if a.prod_all:                                       # §6.302: модель ТОЛЬКО для прода — ещё и сорт A с листами его скважин (сорт A тогда «свой»)
        _all = fs + sorted(out_train()) + [x for x in ROWS_OF if x in HOLD] + sorted(x for x in OUT if well(x) in HOLD_W)
        p2, te, MODELS[None] = stage2(sorted(set(_all)), [x for x in ROWS_OF if x in HOLD], a.folds)
        print("⚠ --prod-all: модель на всём учена и на сорте A — его цифры ниже НЕ вне выборки")
    else:
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
def _swap_pairs(mode, tau, dlt):
    """→ [(лист, si, sj, признаки, выигрыш)]: пары слотов листа после назначения, где трасса каждого — кандидат и другого слота.
    Выигрыш — честных после обмена минус до (по меткам, только слоты с эталоном). Признаки в порядке (si, sj)."""
    out = []
    zc = 31 if a.vfeat else None
    for sh, by in BY.items():
        if sh in OUT:
            continue
        rec = RECD[sh]; cl = clusters(rec["cands"]); ch = solve(rec, by, mode, tau, dlt)
        best, fin = {}, {}
        for si, idx in by.items():
            for i in idx:
                c = cl[ROWS[i][2]]
                if (si, c) not in best or P[i] > P[best[(si, c)]]:
                    best[(si, c)] = i
            cur_ = next((cl[ROWS[i][2]] for i in idx if ROWS[i][5]), None)
            fin[si] = ch.get(si, cur_)
        ss = sorted(si for si in by if fin[si] is not None)
        hon = lambda i: int(ROWS[i][4] or 0)
        for x in range(len(ss)):
            for y in range(x + 1, len(ss)):
                si, sj = ss[x], ss[y]; ci, cj = fin[si], fin[sj]
                if ci == cj:
                    continue
                q = [best.get((si, ci)), best.get((sj, cj)), best.get((si, cj)), best.get((sj, ci))]
                if any(v is None for v in q):
                    continue
                gain = hon(q[2]) + hon(q[3]) - hon(q[0]) - hon(q[1])
                for o in ((0, 1, 2, 3), (1, 0, 3, 2)):             # обе ориентации пары
                    v = [q[t] for t in o]
                    pp = [float(P[t]) for t in v]
                    f = pp + [pp[2] + pp[3] - pp[0] - pp[1], pp[2] - pp[0], pp[3] - pp[1]]
                    f += [X_ALL[t, 4] for t in v] + ([X_ALL[t, zc] for t in v] if zc else [])
                    f += [X_ALL[v[0], c_] - X_ALL[v[1], c_] for c_ in (1, 6, 7, 8, 9, 10, 11)]
                    f += [X_ALL[v[0], 12], X_ALL[v[1], 12], X_ALL[v[0], 13], X_ALL[v[1], 13], X_ALL[v[0], 19], X_ALL[v[1], 19],
                          X_ALL[v[0], 2], X_ALL[v[1], 2]]
                    out.append((sh, (si, sj) if o[0] == 0 else (sj, si), f, gain))
    return out


def _swap_eval(mode, tau, dlt, g):
    PR = _swap_pairs(mode, tau, dlt)
    Xs = np.array([r[2] for r in PR], float); Ys = np.array([r[3] for r in PR]); SH = [r[0] for r in PR]
    ps = np.full(len(PR), np.nan)
    trn = lambda sel: [i for i in range(len(PR)) if sel(SH[i]) and Ys[i] != 0]
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                                min_samples_leaf=40, random_state=0)
    for k in range(a.folds):
        tr = trn(lambda x: x in FIELD and FOLD[well(x)] != k)
        te = [i for i in range(len(PR)) if SH[i] in FIELD and FOLD[well(SH[i])] == k]
        if tr and te:
            ps[te] = mk().fit(Xs[tr], Ys[tr] > 0).predict_proba(Xs[te])[:, 1]
    tr = trn(lambda x: x in FIELD); te = [i for i in range(len(PR)) if SH[i] in HOLD]
    if tr and te:
        ps[te] = mk().fit(Xs[tr], Ys[tr] > 0).predict_proba(Xs[te])[:, 1]
    pos = int((Ys > 0).sum() // 2); neg = int((Ys < 0).sum() // 2)
    print(f"\n★ {g}: обмен — пар {len(PR) // 2}, обмен помог бы {pos}, навредил бы {neg} (потолок оракула +{pos})")
    PAIR = defaultdict(list)                              # (лист, {si, sj}) → средняя P по двум ориентациям
    for i, r in enumerate(PR):
        PAIR[(r[0], frozenset(r[1]))].append(i)
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        D = Counter(); n_sw = Counter()
        by_sh = defaultdict(list)
        for (sh, ss_), idx in PAIR.items():
            pm = float(np.nanmean(ps[idx]))
            if np.isfinite(pm) and pm >= th:
                by_sh[sh].append((pm, ss_, int(Ys[idx[0]])))
        for sh, L in by_sh.items():
            used = set()
            for pm, ss_, gn in sorted(L, key=lambda t: -t[0]):
                if used & ss_:
                    continue
                used |= ss_; D[sh] += gn; n_sw[sh] += 1
        for sn, S in (("поле", FIELD0), ("вне", EVO), ("поле+вне", FIELD), ("сорт A", HOLD)):
            d = np.array([D.get(x, 0) for x in S]); nzv = d[d != 0]
            pv = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nzv))) * nzv).sum(1)) >= abs(d.sum()))) if len(nzv) else 1.0
            print(f"★ обмен P ≥ {th:.1f} {sn:8s}: обменов {sum(n_sw.get(x, 0) for x in S):4d}; Δ {int(d.sum()):+4d} "
                  f"(листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {pv:.4f})")


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
    if a.swap:                                            # §6.324: обмен трасс двух слотов после назначения
        _swap_eval(mode, tau, dlt, g)
    if a.miss:                                            # §6.301: почему честный кандидат не встал
        M = Counter(); RK = Counter(); MISS_EX = []; FM = Counter(); FT = Counter(); FO = Counter()
        for sh, by in BY.items():
            if sh in OUT or sh not in FIELD:
                continue
            rec = RECD[sh]; cl = clusters(rec["cands"]); ch = solve(rec, by, mode, tau, dlt)
            fin = {}
            for si, idx in by.items():
                cur_ = next((cl[ROWS[i][2]] for i in idx if ROWS[i][5]), None)
                fin[si] = ch.get(si, cur_)
            hon = {si: {cl[ROWS[i][2]] for i in idx if ROWS[i][4]} for si, idx in by.items()}
            owner = defaultdict(list)
            for si, c in fin.items():
                if c is not None:
                    owner[c].append(si)
            for si, idx in by.items():
                if rec["slots"][si]["has_truth"]:
                    FT[re.sub(r"^BKZ_", "", rec["slots"][si]["root"])] += 1
                    if hon[si] and fin[si] in hon[si]:
                        FO[re.sub(r"^BKZ_", "", rec["slots"][si]["root"])] += 1
                if not rec["slots"][si]["has_truth"] or not hon[si] or fin[si] in hon[si]:
                    continue
                ph = max(P[i] for i in idx if ROWS[i][4]); pb = max(P[i] for i in idx)
                rk = 1 + sum(1 for c in {cl[ROWS[i][2]] for i in idx} if max(P[i] for i in idx if cl[ROWS[i][2]] == c) > ph)
                RK[min(rk, 4)] += 1
                took = any(owner.get(c) for c in hon[si])
                c = fin[si]
                if c is None:
                    cat = "слот пуст"
                elif any(c in hon[s2] for s2 in by if s2 != si):
                    cat = "стоит трасса, честная для ДРУГОГО слота"
                else:
                    cat = "стоит трасса, ни для кого не честная"
                M[(cat, "честную забрал другой слот" if took else "честная свободна", "P честной < τ" if ph < tau else "P честной ≥ τ")] += 1
                FM[re.sub(r"^BKZ_", "", rec["slots"][si]["root"])] += 1
                if cat.endswith("ни для кого не честная") and not took and ph >= tau:     # §6.316: примеры «мусор вместо честной»
                    _kc = max((i for i in idx if cl[ROWS[i][2]] == c), key=lambda i: P[i])
                    _kh = max((i for i in idx if ROWS[i][4]), key=lambda i: P[i])
                    MISS_EX.append(dict(sheet=sh, slot=rec["slots"][si]["name"], k_chosen=ROWS[_kc][2], p_chosen=float(P[_kc]),
                                        k_honest=ROWS[_kh][2], p_honest=float(P[_kh])))
                # взаимный обмен: честная этого слота стоит в слоте s2, а в этом слоте — честная для s2
                for c_h in hon[si]:
                    for s2 in owner.get(c_h, []):
                        if s2 != si and fin[si] is not None and fin[si] in hon.get(s2, set()):
                            M[("  ↳ из них взаимный обмен двух слотов", rec["slots"][si]["root"] + "↔" + rec["slots"][s2]["root"], "")] += 1
        print(f"\n★ {g}: разбор промахов поля ({sum(M.values())} слотов с честным кандидатом, нечестных после назначения)")
        for k_, v_ in M.most_common():
            print(f"   {v_:5d}  {' | '.join(k_)}")
        print(f"   ранг честной по P среди кластеров слота: {dict(sorted(RK.items()))} (4 = 4 и ниже)")
        pickle.dump(MISS_EX, open(str(a.dump).replace(".pkl", "_missex.pkl"), "wb"))
        print("   по семействам (слотов с эталоном / честных после назначения / промахов при честном кандидате): " +
              "; ".join(f"{f_} {FT[f_]}/{FO[f_]}/{FM[f_]}" for f_, _ in FT.most_common(16)))
    for sn, S in ((("поле", FIELD0), ("вне", EVO), ("поле+вне", FIELD), ("сорт A", HOLD)) if a.eval_out else (("поле", FIELD), ("сорт A", HOLD))):
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
                vfeat=bool(a.vfeat), prior=(VSC["prior"] if a.vfeat else None), cnn=bool(a.cnn),
                delta=float(dlt))
    for k, (ma, mn, m2) in MODELS.items():
        nm = "line_choice_all.pkl" if k is None else f"line_choice_f{k}.pkl"
        pickle.dump(dict(meta, m_all=ma, m_nocur=mn, m2=m2, fold=k), open(sd / nm, "wb"))
    (sd / "line_choice_folds.json").write_text(json.dumps({r["sheet"]: FOLD[well(r["sheet"])] for r in RECS if r["sheet"] in FIELD},
                                                          ensure_ascii=False), encoding="utf-8")
    print(f"модели сохранены: {sd} (τ {tau}, δ {dlt})")
