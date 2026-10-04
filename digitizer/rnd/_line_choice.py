r"""_line_choice.py — ОБУЧЕННЫЙ ВЫБОР ТРАССЫ ДЛЯ СЛОТА СРЕДИ КАНДИДАТОВ КЭША (04.10, §6.280; критерии — в ROADMAP).

Данные — `_line_choice_scan.py` (пары слот × кандидат в полосе шкалы слота, метка — честность для эталона слота на плоскости в обе
стороны по пикселям). Модель — градиентный бустинг P(кандидат честен для слота): признаки кандидата (источник, длина, покрытие строк
слота, квантили положения в полосе, стиль, устойчивая извилистость), слота (семейство, номер зонда, десятилетие, число слотов и
кандидатов), отношения (ранги положения и толщины среди кандидатов полосы, стоит ли кандидат в этом слоте / в другом, разница с
нынешним кандидатом слота). Поле — 5 фолдов по скважинам, сорт A — модель на всём поле.
Замена: трасса слота меняется на СВОБОДНОГО кандидата (не стоит ни в одном слоте), если P(лучший) ≥ τ и P(лучший) − P(нынешний) ≥ δ;
жадно по выигрышу, один свободный кандидат — одному слоту. Мера — честность в слоте до и после (по меткам).

  _line_choice.py --grid 0.7:0.3 0.8:0.4
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--data", default=r"F:/nds/output/taskS/line_choice")
ap.add_argument("--grid", nargs="+", default=["0.7:0.3", "0.8:0.4"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/line_choice_res.pkl")
ap.add_argument("--save-dir", default="", help="модели фолдов, на всём поле, словарь семейств, карта лист → фолд")
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


RECS = [pickle.load(open(f, "rb")) for f in sorted(Path(a.data).glob("*.pkl"))]
ROWS = []        # (лист, слот, k, признаки, метка, here, elsewhere)
for rec in RECS:
    sh = rec["sheet"]; CS = rec["cands"]; ns = len(rec["slots"])
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
            lab = None if r["lab"] is None else int(r["lab"][0])
            ROWS.append((sh, si, r["k"], f, lab, r["here"], r["elsewhere"]))
CAT = [12]                                        # индекс кода семейства в признаках
X = np.array([r[3] for r in ROWS], float)
print(f"листов {len(RECS)}, пар слот × кандидат {len(ROWS)}, с меткой {sum(1 for r in ROWS if r[4] is not None)}, "
      f"честных {sum(1 for r in ROWS if r[4] == 1)}")


def fit(sheets):
    idx = [i for i, r in enumerate(ROWS) if r[0] in sheets and r[4] is not None]
    cat = np.zeros(X.shape[1], bool); cat[CAT] = True
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                       categorical_features=cat, random_state=0)
    m.fit(X[idx], np.array([ROWS[i][4] for i in idx]))
    return m


wells = sorted({well(r["sheet"]) for r in RECS if r["sheet"] in FIELD})
FOLD = {w: i % a.folds for i, w in enumerate(wells)}
P = np.full(len(ROWS), np.nan)
FOLD_MODELS = []
for k in range(a.folds):
    tr = {r["sheet"] for r in RECS if r["sheet"] in FIELD and FOLD[well(r["sheet"])] != k}
    te = [i for i, r in enumerate(ROWS) if r[0] in FIELD and FOLD[well(r[0])] == k]
    m = fit(tr); P[te] = m.predict_proba(X[te])[:, 1]
    FOLD_MODELS.append((k, m))
    print(f"  фолд {k}: обучено")
mA = fit(FIELD)
te = [i for i, r in enumerate(ROWS) if r[0] in HOLD]
P[te] = mA.predict_proba(X[te])[:, 1]
# по листу: слоты → пары
BY = defaultdict(lambda: defaultdict(list))
for i, r in enumerate(ROWS):
    BY[r[0]][r[1]].append(i)
SL = {}
for rec in RECS:
    for si, s in enumerate(rec["slots"]):
        SL[(rec["sheet"], si)] = s
rng = np.random.default_rng(0)
RES = {}
for g in a.grid:
    tau, dlt = (float(x) for x in g.split(":"))
    D = Counter(); SW = []
    for sh, slots in BY.items():
        props = []
        for si, idxs in slots.items():
            s = SL[(sh, si)]
            if not s["has_truth"] and False:
                pass
            here = [i for i in idxs if ROWS[i][5]]
            p_here = P[here[0]] if here else 0.0
            free = [i for i in idxs if not ROWS[i][5] and not ROWS[i][6]]
            if not free:
                continue
            best = max(free, key=lambda i: P[i])
            if P[best] >= tau and P[best] - p_here >= dlt:
                props.append((P[best] - p_here, si, best))
        taken = set()
        for gain, si, best in sorted(props, reverse=True):
            k = ROWS[best][2]
            if k in taken:
                continue
            taken.add(k)
            s = SL[(sh, si)]
            if s["out_honest"] is None:
                continue                                  # нет эталона — не меряется (в проде замена всё равно была бы)
            after = ROWS[best][4] or 0
            D[sh] += int(after) - int(bool(s["out_honest"]))
            SW.append(dict(sheet=sh, slot=s["name"], before=bool(s["out_honest"]), after=bool(after), p=float(P[best])))
    RES[g] = SW
    for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
        d = np.array([D.get(x, 0) for x in S]); nzv = d[d != 0]
        p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nzv))) * nzv).sum(1)) >= abs(d.sum()))) if len(nzv) else 1.0
        sw = [x for x in SW if x["sheet"] in S]
        print(f"★ τ/δ = {g}, {sn}: замен {len(sw)} (стала честной {sum(1 for x in sw if x['after'] and not x['before'])}, "
              f"перестала {sum(1 for x in sw if x['before'] and not x['after'])}); Δ {int(d.sum()):+d} "
              f"(листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
pickle.dump(dict(res=RES, fams=FAMS), open(a.dump, "wb"))
if a.save_dir:
    import json
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    tau, dlt = (float(x) for x in a.grid[0].split(":"))
    meta = dict(fams=dict(FAMS), cat=CAT, tau=tau, delta=dlt)
    for k, m in FOLD_MODELS:
        pickle.dump(dict(meta, model=m, fold=k), open(sd / f"line_choice_f{k}.pkl", "wb"))
    pickle.dump(dict(meta, model=mA, fold=None), open(sd / "line_choice_all.pkl", "wb"))
    (sd / "line_choice_folds.json").write_text(json.dumps({r["sheet"]: FOLD[well(r["sheet"])] for r in RECS if r["sheet"] in FIELD},
                                                          ensure_ascii=False), encoding="utf-8")
    print(f"модели сохранены: {sd}")
