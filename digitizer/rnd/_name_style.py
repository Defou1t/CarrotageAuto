r"""_name_style.py — ИМЕНА ПО СТИЛЮ ЛИНИИ: ПАРНАЯ МОДЕЛЬ «КАКОЙ СТИЛЬ КАКОМУ ИМЕНИ» (04.10, §6.273; критерии — в ROADMAP).

Учится по эталону: для двух кривых одного трека (имена A и B) упорядоченная пара стилей (x как A, y как B) — метка 1, если это
их настоящий порядок, 0 — если переставлен. Признаки: толщина, цвет, затемнение, разрывы каждой и разности; семейства A и B;
номера зондов; десятилетие листа. Стиль — `_style_scan.py` (по эталону для обучения, по нашей трассе для применения).
Применение к выдаче: в треке для пары кривых под именами A и B — P(порядок верен); P < 1 − τ ⇒ имена меняются местами (жадно,
самые уверенные первыми, каждая кривая — в одной перестановке). Мера — именные честные по парам 1:1 мерой приёмки
(`_name_cost_prod.py --pairs-dump`): кривая под новым именем честна, если пара 1:1 сопоставила её эталону этого имени.
Поле — 5 фолдов по скважинам (модель без скважин фолда), сорт A — модель на всём поле.

  _name_style.py --taus 0.8 0.9
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--style", default=r"F:/nds/output/taskS/style_scan.pkl")
ap.add_argument("--pairs", default=r"F:/nds/output/taskS/rp_lvl_S_pairs.pkl")
ap.add_argument("--mode", default="S")
ap.add_argument("--taus", nargs="+", type=float, default=[0.8, 0.9])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/name_style_swaps.pkl")
ap.add_argument("--rough", action="store_true", help="§6.276: + извилистость обеих кривых (по трассе)")
ap.add_argument("--rough-robust", action="store_true", help="§6.277: извилистость устойчивая (окна без скачков, медиана)")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--save-dir", default="", help="сохранить модели фолдов, модель на всём поле, словарь семейств и карту лист → фолд")
a = ap.parse_args()
TS = Path(a.ts)
from sklearn.ensemble import HistGradientBoostingClassifier

SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else sh
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
R = pickle.load(open(a.style, "rb"))
P = pickle.load(open(a.pairs, "rb"))[a.mode]
# трек кривой: пары 1:1 дают трек сопоставленных; для остальных — карта слотов (как в `_name_swaps.py`)
smap = pickle.load(open(TS / "slotmap.pkl", "rb"))
for root in ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_heldout", "pools_all"]:
    for f in sorted((TS / root).glob("*.pkl")):
        if f.stem in smap:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        smap.setdefault(d["name"], {s["name"]: s["track"] for s in d["slots"]})
FAMS = {}


def fcode(name):
    f = re.sub(r"^BKZ_", "", M.mnem_root(name))
    return FAMS.setdefault(f, len(FAMS))


def probe(name):
    m = re.match(r"^(?:BKZ_)?[A-Z]+?(\d)", name.split()[0])
    return int(m.group(1)) if m else 0


def decade(sh):
    m = re.search(r"(19[5-9]\d|20[0-2]\d)", sh)
    return (int(m.group(1)) // 10 * 10) if m else 0


def hue(t):
    c1, c2 = t["c1"], t["c2"]
    if max(abs(c1), abs(c2)) < 30:
        return 0
    if c2 > 30 and c2 >= abs(c1):
        return 3
    if c1 > 30:
        return 1
    if c1 < -30:
        return 2
    return 4


def rough(x):
    """§6.276: извилистость — средний отход от медианы по 41 строке в долях размаха (5–95%) кривой;
    §6.277 (`--rough-robust`): медиана по окнам 400 строк (шаг 200), окна со скачком > 40 px мимо, < 3 окон — нет признака"""
    if a.rough_robust:
        if x is None or len(x) < 400:
            return np.nan
        from scipy.ndimage import median_filter
        x = x.astype(float)
        s_ = median_filter(x, size=41, mode="nearest")
        dev = np.abs(x - s_); jump = np.abs(np.diff(x, prepend=x[0])) > 40
        vals = [np.mean(dev[i:i + 400]) for i in range(0, len(x) - 400, 200) if not jump[i:i + 400].any()]
        if len(vals) < 3:
            return np.nan
        return float(np.median(vals) / max(5.0, np.percentile(x, 95) - np.percentile(x, 5)))
    if x is None or len(x) < 200:
        return np.nan
    from scipy.ndimage import median_filter
    s_ = median_filter(x.astype(float), size=41, mode="nearest")
    rng_ = np.percentile(x, 95) - np.percentile(x, 5)
    return float(np.mean(np.abs(x - s_)) / max(5.0, rng_))


def feat(tx, ty, A, B, sh):
    """признаки упорядоченной пары: кривая со стилем tx названа A, со стилем ty — B"""
    if a.rough:
        rx, ry = tx.get("rough", np.nan), ty.get("rough", np.nan)
        extra = [rx, ry, (np.log(rx / ry) if (rx > 1e-6 and ry > 1e-6) else np.nan)]
    else:
        extra = []
    return extra + [tx["width"], ty["width"], tx["width"] - ty["width"], tx["c1"], tx["c2"], ty["c1"], ty["c2"],
            tx["c1"] - ty["c1"], tx["c2"] - ty["c2"], tx["dark"], ty["dark"], tx["gap"], ty["gap"],
            hue(tx), hue(ty), fcode(A), fcode(B), probe(A), probe(B), probe(A) - probe(B), decade(sh)]


CAT = [13, 14, 15, 16]                                      # индексы категориальных признаков (оттенки, семейства)
if a.rough:
    CAT = [c + 3 for c in CAT]                               # три признака извилистости — в начале
ST = {}                                                      # (лист, имя) → запись стиля
for r in R:
    ST[(r["sheet"], r["name"])] = r
if a.rough:                                                  # §6.276: извилистость по эталону и по выдаче
    for cv in pickle.load(open(a.cache, "rb")):
        r = ST.get((cv["sheet"], cv["name"]))
        if r is None:
            continue
        if r["truth"] is not None:
            r["truth"] = dict(r["truth"], rough=rough(cv["gx"]))
        if r["out"] is not None:
            r["out"] = dict(r["out"], rough=rough(cv["tx"]) if len(cv["tx"]) else np.nan)
# пары имён в одном треке, у которых есть эталонный стиль
TR = defaultdict(list)                                       # (лист, трек) → имена
for (sh, nm), r in ST.items():
    t = smap.get(sh, {}).get(nm)
    if t is not None:
        TR[(sh, t)].append(nm)


def train_set(sheets):
    X, Y = [], []
    for (sh, t), names in TR.items():
        if sh not in sheets:
            continue
        for i, A in enumerate(names):
            for B in names[i + 1:]:
                ta, tb = ST[(sh, A)]["truth"], ST[(sh, B)]["truth"]
                if not (ta and tb):
                    continue
                X.append(feat(ta, tb, A, B, sh)); Y.append(1)
                X.append(feat(tb, ta, A, B, sh)); Y.append(0)
                X.append(feat(tb, ta, B, A, sh)); Y.append(1)
                X.append(feat(ta, tb, B, A, sh)); Y.append(0)
    return np.array(X, float), np.array(Y)


def fit(sheets):
    X, Y = train_set(sheets)
    cat = np.zeros(X.shape[1], bool); cat[CAT] = True
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                       categorical_features=cat, random_state=0)
    m.fit(X, Y)
    return m, len(Y)


# пары 1:1 по трекам: имя выдачи → эталон
MT = defaultdict(dict)
for sh, t, g, w in P:
    MT[(sh, t)][w] = g


def apply(model, sheets, tau):
    """→ (Δ по листам, список перестановок)"""
    D = Counter(); SW = []
    for (sh, t), names in TR.items():
        if sh not in sheets:
            continue
        outs = [n for n in names if ST[(sh, n)]["out"]]
        cand = []
        for i, A in enumerate(outs):
            for B in outs[i + 1:]:
                x, y = ST[(sh, A)]["out"], ST[(sh, B)]["out"]
                p = float(model.predict_proba(np.array([feat(x, y, A, B, sh)], float))[0, 1])
                if p < 1 - tau:
                    cand.append((p, A, B))
        used = set()
        mt = MT.get((sh, t), {})
        for p, A, B in sorted(cand):
            if A in used or B in used:
                continue
            used |= {A, B}
            before = (mt.get(A) == A) + (mt.get(B) == B)
            after = (mt.get(A) == B) + (mt.get(B) == A)
            D[sh] += after - before
            SW.append(dict(sheet=sh, track=t, A=A, B=B, p=p, delta=after - before))
    return D, SW


folds_w = sorted({well(sh) for sh in FIELD})
FOLD = {w: i % a.folds for i, w in enumerate(folds_w)}
base = Counter()
for (sh, t), mt in MT.items():
    base[sh] += sum(1 for w, g in mt.items() if w == g)
print(f"база (имя верно по парам 1:1): поле {sum(v for s, v in base.items() if s in FIELD)}, сорт A {sum(v for s, v in base.items() if s in HOLD)}")
MODELS = []
for k in range(a.folds):
    tr = {sh for sh in FIELD if FOLD[well(sh)] != k}
    m, n = fit(tr)
    MODELS.append((k, m))
    print(f"  фолд {k}: обучающих пар {n}")
mA, n = fit(FIELD)
print(f"  модель на всём поле: обучающих пар {n}")
rng = np.random.default_rng(0)
OUT = {}
for tau in a.taus:
    DF, SWF = Counter(), []
    for k, m in MODELS:
        te = {sh for sh in FIELD if FOLD[well(sh)] == k}
        d, sw = apply(m, te, tau); DF.update(d); SWF += sw
    DA, SWA = apply(mA, HOLD, tau)
    OUT[tau] = dict(field=SWF, hold=SWA)
    for sn, Dd, S, sw in (("поле", DF, FIELD, SWF), ("сорт A", DA, HOLD, SWA)):
        d = np.array([Dd.get(s, 0) for s in S]); nz = d[d != 0]
        p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
        good = sum(1 for s in sw if s["delta"] > 0); bad = sum(1 for s in sw if s["delta"] < 0)
        print(f"★ τ={tau:g}, {sn}: перестановок {len(sw)} (лучше {good}, хуже {bad}, без изменения {len(sw) - good - bad}); "
              f"именных Δ {int(d.sum()):+d} (листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
        pr = Counter((M.mnem_root(s["A"]), M.mnem_root(s["B"]), s["delta"]) for s in sw)
        print("   пары (A, B, Δ): " + ", ".join(f"{x}↔{y} {dd:+d}: {c}" for (x, y, dd), c in pr.most_common(10)))
pickle.dump(OUT, open(a.dump, "wb"))
if a.save_dir:
    import json
    sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
    meta = dict(fams=dict(FAMS), cat=CAT, thr=60, step=8, tau=max(a.taus))
    for k, m in MODELS:
        pickle.dump(dict(meta, model=m, fold=k), open(sd / f"name_style_f{k}.pkl", "wb"))
    pickle.dump(dict(meta, model=mA, fold=None), open(sd / "name_style_all.pkl", "wb"))
    (sd / "name_style_folds.json").write_text(json.dumps({sh: FOLD[well(sh)] for sh in FIELD}, ensure_ascii=False), encoding="utf-8")
    print(f"модели сохранены: {sd}")
