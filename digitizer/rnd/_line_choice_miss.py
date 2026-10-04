r"""_line_choice_miss.py — ПОЧЕМУ ВЫБОР ТРАССЫ НЕ БЕРЁТ СВОБОДНУЮ ЧЕСТНУЮ ТРАССУ (04.10, к §6.280–6.281).

Данные — `_line_choice_scan.py`; модель и признаки — как `_line_choice.py` (поле — 5 фолдов по скважинам, сорт A — модель на всём
поле). Для каждого слота с эталоном, где выдача нечестна:
- есть ли честный кандидат свободный / занятый другим слотом / нет;
- если свободный есть: верхний по P свободный — честный (и тогда P выше или ниже порога) или нет; ранг честного;
- чем отличается верхний нечестный от честного (источник, длина, покрытие, положение в полосе, толщина, извилистость).

  _line_choice_miss.py --tau 0.2
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
ap.add_argument("--tau", type=float, default=0.2)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/line_choice_miss.pkl")
a = ap.parse_args()
from sklearn.ensemble import HistGradientBoostingClassifier
TS = Path(a.ts)
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else sh
FAMS = {}
fcode = lambda f: FAMS.setdefault(f, len(FAMS))
NAMES = ["src_dec", "log_n", "cover", "u10", "u50", "u90", "width", "c1", "c2", "dark", "gap", "rough", "fam", "probe", "decade",
         "n_slots", "n_cands", "rk_u", "rk_w", "here", "elsewhere", "d_logn", "d_cover"]


def probe(name):
    m = re.match(r"^(?:BKZ_)?[A-Z]+?(\d)", name.split()[0])
    return int(m.group(1)) if m else 0


def decade(sh):
    m = re.search(r"(19[5-9]\d|20[0-2]\d)", sh)
    return (int(m.group(1)) // 10 * 10) if m else 0


def nz(v):
    return np.nan if v is None else v


RECS = [pickle.load(open(f, "rb")) for f in sorted(Path(a.data).glob("*.pkl"))]
ROWS = []
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
X = np.array([r[3] for r in ROWS], float)
CAT = [12]


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
for k in range(a.folds):
    tr = {r["sheet"] for r in RECS if r["sheet"] in FIELD and FOLD[well(r["sheet"])] != k}
    te = [i for i, r in enumerate(ROWS) if r[0] in FIELD and FOLD[well(r[0])] == k]
    P[te] = fit(tr).predict_proba(X[te])[:, 1]
te = [i for i, r in enumerate(ROWS) if r[0] in HOLD]
P[te] = fit(FIELD).predict_proba(X[te])[:, 1]

BY = defaultdict(list)
for i, r in enumerate(ROWS):
    BY[(r[0], r[1])].append(i)
SL = {}
RS = {}
for rec in RECS:
    RS[rec["sheet"]] = rec["slots"]
    for si, s in enumerate(rec["slots"]):
        SL[(rec["sheet"], si)] = s
rec_slots = lambda sh: RS[sh]
OUT = []
for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
    C = Counter(); diff = defaultdict(list); rank_h = []
    for (sh, si), idxs in BY.items():
        if sh not in S:
            continue
        s = SL[(sh, si)]
        if not s["has_truth"]:
            continue
        if s["out_honest"] is None:
            hon = [i for i in idxs if ROWS[i][4] == 1]
            C["выдачи нет / коротка" + (", честный кандидат есть" if hon else "")] += 1
            continue
        if s["out_honest"]:
            C["выдача честна"] += 1
            continue
        hon = [i for i in idxs if ROWS[i][4] == 1]
        hf = [i for i in hon if not ROWS[i][5] and not ROWS[i][6]]
        he = [i for i in hon if ROWS[i][6]]
        hh = [i for i in hon if ROWS[i][5]]
        if hh:
            C["нынешний кандидат честен по метке (выдача — нет)"] += 1
            continue
        if not hf:
            C["честный только занят другим слотом" if he else "честного кандидата нет"] += 1
            continue
        free = [i for i in idxs if not ROWS[i][5] and not ROWS[i][6]]
        top = max(free, key=lambda i: P[i])
        here = [i for i in idxs if ROWS[i][5]]
        ph = P[here[0]] if here else 0.0
        bh = max(hf, key=lambda i: P[i])
        rk = sorted(free, key=lambda i: -P[i]).index(bh)
        rank_h.append(rk)
        if ROWS[top][4] == 1:
            C[f"свободный честный — верхний по P, P ≥ {a.tau}" if P[top] >= a.tau else f"свободный честный — верхний по P, P < {a.tau}"] += 1
        else:
            C["свободный честный есть, но верхний по P — нечестный"] += 1
            for j, nm in enumerate(NAMES):
                diff[nm].append((X[top, j], X[bh, j]))
        OUT.append(dict(sheet=sh, slot=s["name"], p_top=float(P[top]), p_hon=float(P[bh]), p_here=float(ph), rank=rk,
                        top_hon=bool(ROWS[top][4] == 1), n_free=len(free), n_hon_free=len(hf), set=sn))
    tot = sum(C.values())
    print(f"\n★ {sn}: слотов с эталоном {tot}")
    for k, v in C.most_common():
        print(f"   {k:62s} {v:5d} ({v / tot:.0%})")
    if rank_h:
        rc = Counter(min(r, 5) for r in rank_h)
        print("   ранг лучшего честного среди свободных по P: " + ", ".join(f"{k if k < 5 else '5+'}: {rc[k]}" for k in sorted(rc)))
    if diff:
        print("   верхний нечестный против честного (медианы; доля, где у верхнего больше):")
        for nm in NAMES:
            v = np.array(diff[nm], float)
            ok = ~np.isnan(v).any(1)
            if ok.sum() < 10:
                continue
            v = v[ok]
            print(f"     {nm:10s} верхний {np.median(v[:, 0]):8.3f}  честный {np.median(v[:, 1]):8.3f}  "
                  f"верхний > честного {np.mean(v[:, 0] > v[:, 1]):.0%}  равны {np.mean(v[:, 0] == v[:, 1]):.0%}")
    # подробности: (а) верхний честный с P ≥ τ — почему не заменён; (б) верхний нечестный — насколько он «рядом»; (в) честный занят
    # другим слотом — честен ли он и там (дубль) и честна ли выдача того слота (тогда обмен ничего не даст)
    blk = Counter(); near = []; occ = Counter()
    for (sh, si), idxs in BY.items():
        if sh not in S:
            continue
        s = SL[(sh, si)]
        if not s["has_truth"] or s["out_honest"] in (None, True):
            continue
        hon = [i for i in idxs if ROWS[i][4] == 1]
        hf = [i for i in hon if not ROWS[i][5] and not ROWS[i][6]]
        free = [i for i in idxs if not ROWS[i][5] and not ROWS[i][6]]
        here = [i for i in idxs if ROWS[i][5]]
        ph = P[here[0]] if here else 0.0
        if hf:
            top = max(free, key=lambda i: P[i])
            if ROWS[top][4] == 1 and P[top] >= a.tau:
                blk["P(нынешний) > P(верхний свободный)" if ph > P[top] else "заменяем (≥ τ и ≥ нынешнего)"] += 1
            elif ROWS[top][4] != 1:
                lab = [r for r in SL[(sh, si)]["cands"] if r["k"] == ROWS[top][2]][0]["lab"]
                near.append(lab[2] if lab else 99.0)
        elif [i for i in hon if ROWS[i][6]]:
            k = ROWS[[i for i in hon if ROWS[i][6]][0]][2]
            other = None
            for sj, s2 in enumerate(rec_slots(sh)):
                if sj != si and any(r["k"] == k and r["here"] for r in s2["cands"]):
                    other = s2
            if other is None:
                occ["занят слотом без пары в записи"] += 1
                continue
            lab2 = [r for r in other["cands"] if r["k"] == k][0]["lab"]
            both = bool(lab2 and lab2[0])
            occ[("тот же кандидат честен и для того слота" if both else "для того слота кандидат нечестен") +
                (", у того слота нет эталона" if not other["has_truth"] else "")] += 1
    print("   (а) свободный честный — верхний по P ≥ τ: " + ", ".join(f"{k}: {v}" for k, v in blk.most_common()))
    if near:
        nv = np.array(near)
        print(f"   (б) верхний нечестный: медиана расстояния до эталона по плоскости {np.median(nv):.1f} px; ≤ 6 px — {np.mean(nv <= 6):.0%}, "
              f"≤ 10 px — {np.mean(nv <= 10):.0%}")
    print("   (в) честный только занят другим слотом: " + ", ".join(f"{k}: {v}" for k, v in occ.most_common()))
pickle.dump(dict(out=OUT, fams=FAMS), open(a.dump, "wb"))
print(f"\nзаписано: {a.dump}")
