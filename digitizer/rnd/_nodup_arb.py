r"""_nodup_arb.py — §6.363 АРБИТР РАЗДВИГАНИЯ ДУБЛЕЙ (13.11): кого из пары переносить и переносить ли вообще.

Данные: журнал конфликтов `_nodup` на стенде (повтор NDY3 = NDS3 с NDS_NODUP_LOG, `taskS/nodupx_stand/NDY3_*.jsonl`) + исход по эталону
каждой раздвинутой пары (`_nodup_selfstyle.py scan` → `taskS/nodup_selfstyle/pairs_NDS3.jsonl`: именных ДО (NSE22), как есть (NDS3) и при
обмене трасс пары). Варианты арбитра: «не раздвигать» (= ДО), «как есть» (переносится lo, правило «sum»), «обмен» (lo остаётся на общей
линии, его новую трассу берёт hi). Модель — две регрессии HGB (как есть − ДО, обмен − ДО) на признаках записи конфликта
(`auto/nodup_arb.py` — те же, что в проде); фолды — по скважинам (5, детерминированно); выбор — argmax {0, r̂_как_есть, r̂_обмен} с запасом m.

  _nodup_arb.py build                 — признаки + исходы → taskS/nodup_arb/data.pkl
  _nodup_arb.py oof [--with-scan]     — вне фолда: Δ именных против «как есть» (NDS3) и против ДО (NSE22) по запасу m
  _nodup_arb.py train --margin M      — модели фолдов + общая → taskS/nodup_arb/ (nodup_arb_f{k}.pkl, _all.pkl, _folds.json)
"""
import sys, argparse, json, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np
from auto import nodup_arb as NA

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "nodup_arb"
ARCH = Path(r"F:\nds\projects\Archive")
SK = ("c1", "c2", "width", "dark", "gap")
SC = dict(c1=15.0, c2=15.0, width=1.0, dark=40.0, gap=0.15)
SCAN = ["d_ownX_alt", "d_ownX_sh", "d_ownY_alt", "d_ownY_sh", "m_style", "n_shared", "ovp"]
K = 5


def _d(s, t):
    if s is None or t is None:
        return np.nan
    return sum(abs(s[k] - t[k]) / SC[k] for k in SK)


def scan_feats(pr):
    oX, oY, a_, s_ = pr["own_X"], pr["own_Y"], pr["alt"], pr["sh"]
    ms = np.nan
    if a_ is not None and s_ is not None and (oX is not None or oY is not None):
        ms = 0.0
        if oX is not None:
            ms += _d(oX, s_) - _d(oX, a_)
        if oY is not None:
            ms += _d(oY, a_) - _d(oY, s_)
    return [_d(oX, a_), _d(oX, s_), _d(oY, a_), _d(oY, s_), ms, pr["n_shared"], max(pr["ov_pre"])]


def fold_of(well):
    return int(hashlib.md5(well.encode("utf-8")).hexdigest(), 16) % K


def build(a):
    OUT.mkdir(parents=True, exist_ok=True)
    logs, seen = [], set()
    for jf in sorted((TS / "nodupx_stand").glob(f"{a.log}_*.jsonl")):
        for l in jf.read_text(encoding="utf-8").splitlines():
            if l.strip() and l not in seen:
                seen.add(l); logs.append(json.loads(l))
    P = {}
    for l in (TS / "nodup_selfstyle" / "pairs_NDS3.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        for p in r["pairs"]:
            P[(r["sheet"], p["X"], p["Y"])] = p
    well = {q.name: q.parent.parent.name for q in ARCH.glob("*/wlg/*.nlgx")}
    rows, nmov, used = [], 0, set()
    for rec in logs:
        if rec.get("moved") is None:
            continue
        nmov += 1
        A, B = rec["pair"]; lo = rec["moved"]; hi = B if lo == A else A
        key = (rec["sheet"], lo, hi)
        pr = P.get(key)
        if pr is None or key in used:              # одна пара — одна запись (повторный конфликт той же пары в след. проходе — нет)
            continue
        used.add(key)
        w = well.get(rec["sheet"], rec["sheet"])
        rows.append(dict(sheet=rec["sheet"], well=w, fold=fold_of(w), x=NA.vec(rec), xs=scan_feats(pr),
                         keep=sum(pr["pre"]), cur=pr["cur"], swp=pr["swp"]))
    stand = [l.strip() for l in (TS / "nodup_stand" / "all.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    folds = {nm: fold_of(well.get(nm, nm)) for nm in stand}
    print(f"★ журнал: строк {len(logs)}, переносов {nmov}, сопоставлено с парами скана {len(rows)} (пар скана {len(P)}); "
          f"фолды листов стенда: {np.bincount(list(folds.values()), minlength=K).tolist()}")
    pickle.dump(dict(rows=rows, feats=NA.FEATS, scan=SCAN, folds=folds), open(OUT / "data.pkl", "wb"))


def _model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


def _choose(rc, rs, mg):
    return np.where((rs > rc + mg) & (rs > mg), 2, np.where(rc < -mg, 0, 1))


def oof(a):
    D = pickle.load(open(OUT / "data.pkl", "rb")); R = D["rows"]
    X = np.array([r["x"] + (r["xs"] if a.with_scan else []) for r in R], float)
    keep = np.array([r["keep"] for r in R]); cur = np.array([r["cur"] for r in R]); swp = np.array([r["swp"] for r in R])
    fo = np.array([r["fold"] for r in R])
    print(f"★ пар {len(R)}, скважин {len(set(r['well'] for r in R))}: ДО {keep.sum()}, как есть {cur.sum()} ({cur.sum() - keep.sum():+d}), "
          f"обмен {swp.sum()} ({swp.sum() - keep.sum():+d}), потолок {np.maximum(np.maximum(keep, cur), swp).sum() - keep.sum():+d}; "
          f"признаков {X.shape[1]}{' (+ стиль скана)' if a.with_scan else ''}")
    rc = np.zeros(len(R)); rs = np.zeros(len(R))
    for k in range(K):
        tr, te = fo != k, fo == k
        if not te.any():
            continue
        for y, out in ((cur - keep, rc), (swp - keep, rs)):
            m = _model(); m.fit(X[tr], y[tr]); out[te] = m.predict(X[te])
    for mg in (0.0, 0.1, 0.2, 0.3, 0.5):
        ch = _choose(rc, rs, mg)
        val = np.choose(ch, [keep, cur, swp])
        print(f"  запас {mg}: не раздвигать {int((ch == 0).sum())}, как есть {int((ch == 1).sum())}, обмен {int((ch == 2).sum())} → "
              f"именных {val.sum()} (против как есть {val.sum() - cur.sum():+d}, против ДО {val.sum() - keep.sum():+d})")
    if a.imp:
        from sklearn.inspection import permutation_importance
        m = _model(); y = cur - keep; m.fit(X, y)
        names = D["feats"] + (D["scan"] if a.with_scan else [])
        pi = permutation_importance(m, X, y, n_repeats=5, random_state=0)
        print("  важность (как есть − ДО):", ", ".join(f"{names[i]} {pi.importances_mean[i]:.3f}" for i in np.argsort(-pi.importances_mean)[:10]))


def train(a):
    D = pickle.load(open(OUT / "data.pkl", "rb")); R = D["rows"]
    X = np.array([r["x"] for r in R], float)
    keep = np.array([r["keep"] for r in R]); cur = np.array([r["cur"] for r in R]); swp = np.array([r["swp"] for r in R])
    fo = np.array([r["fold"] for r in R])
    for k in list(range(K)) + ["all"]:
        tr = fo != k if k != "all" else np.ones(len(R), bool)
        mc = _model(); mc.fit(X[tr], (cur - keep)[tr]); ms = _model(); ms.fit(X[tr], (swp - keep)[tr])
        ck = dict(kind="nodup_arb", feats=NA.FEATS, mc=mc, ms=ms, margin=a.margin, fold=k, n=int(tr.sum()))
        f = OUT / (f"nodup_arb_f{k}.pkl" if k != "all" else "nodup_arb_all.pkl")
        tmp = f.with_suffix(".tmp"); pickle.dump(ck, open(tmp, "wb")); tmp.replace(f)
    (OUT / "nodup_arb_folds.json").write_text(json.dumps(D["folds"], ensure_ascii=False), encoding="utf-8")
    print(f"★ модели: фолды 0–{K - 1} + all, запас {a.margin}; карта фолдов — {len(D['folds'])} листов стенда → {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "oof", "train"])
    ap.add_argument("--with-scan", action="store_true")
    ap.add_argument("--imp", action="store_true")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--log", default="NDY3", help="журнал повтора: NDX3 (первый) / NDY3 (расширенный)")
    a = ap.parse_args()
    {"build": build, "oof": oof, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
