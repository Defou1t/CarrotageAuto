r"""_slot_cache_cv.py — РАСКЛАДКА, ПЕРЕОБУЧЕННАЯ НА КЭШЕ, ЧЕСТНО ПО ФОЛДАМ СКВАЖИН (§6.230, 26.09).

Вход — `_slot_cache_data.py` (пары слот × линия обоих путей из кэша, цель действующей модели). 5 фолдов по скважинам
(md5(скважина) mod 5): модель фолда f учится на остальных скважинах (бустинг как у `slot_model_g250`: 250 деревьев,
глубина 3), выгружается в numpy-деревья формата `auto/slot_model.py` со сверкой против sklearn и против прод-функции
`auto.slot_model.predict` (max|Δ| < 1e-9), и в ПОВТОРЕ С КЭША раскладывает только листы своего фолда (`slot=<вес>`). База —
нынешний прод тем же повтором. Сравнение — `_replay_sweep.py --skip-replay` (тот же счёт и те же пары листов).
Заочная мера до повтора: для слотов, у которых честная линия есть, — доля, где модель ставит честную первой (старая против
новой вне фолда).

  _slot_cache_cv.py [--train] [--replay] [--compare]
"""
import sys, argparse, hashlib, subprocess, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--data", default=r"F:/nds/output/taskS/slotcv/slotdata.npz")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/slotcv")
ap.add_argument("--out", default=r"F:/nds/output/taskS/rp_slotcv")
ap.add_argument("--base", default="rdpick=3,slotlen=0.18,slotall=1,slotgeom=1,slotfill=1")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--trees", type=int, default=250)
ap.add_argument("--train", action="store_true")
ap.add_argument("--replay", action="store_true")
ap.add_argument("--compare", action="store_true")
a = ap.parse_args()
TS = Path(a.ts); D = Path(a.dir); PY = sys.executable; RND = Path(__file__).resolve().parent
wm = json.load(open(TS / "rowdec_wellmap.json", encoding="utf-8"))
fold_of = lambda well: int(hashlib.md5(well.encode("utf-8")).hexdigest(), 16) % a.k
from auto import slot_model as SM

if a.train:
    from sklearn.ensemble import GradientBoostingRegressor
    d = np.load(a.data)
    X, y, W, S, H = d["X"], d["y"], d["well"], d["sheet"], d["hon"]
    F = np.array([fold_of(w) for w in W])
    old = SM.load("slot_model_g250.npz")
    oof = np.zeros(len(y)); sc_old = SM.predict(old, X[:, :SM.NF])  # скор «больше = лучше»; старый вес — по первым 14
    for f in range(a.k):
        tr, te = F != f, F == f
        est = GradientBoostingRegressor(n_estimators=a.trees, max_depth=3, random_state=0).fit(X[tr], y[tr])
        CL, CR, FE, TH, VA, OFF = [], [], [], [], [], [0]
        for stage in est.estimators_:
            t = stage[0].tree_
            CL.append(t.children_left.astype(np.int32)); CR.append(t.children_right.astype(np.int32))
            FE.append(t.feature.astype(np.int32)); TH.append(t.threshold.astype(np.float64))
            VA.append(t.value.reshape(-1).astype(np.float64)); OFF.append(OFF[-1] + t.node_count)
        cl, cr, fe, th, va = (np.concatenate(v) for v in (CL, CR, FE, TH, VA))
        off = np.array(OFF, np.int32); lr = float(est.learning_rate)
        w = dict(children_left=cl, children_right=cr, feature=fe, threshold=th, value=va, offsets=off,
                 lr=np.array([lr]), base=np.array([0.0]), n_features=np.array([X.shape[1]]))
        base = float(est.predict(X[:1])[0] + SM.predict(w, X[:1])[0])       # SM.predict = −(base + lr·Σ)
        w["base"] = np.array([base])
        ref = est.predict(X[te]); got = -SM.predict(w, X[te])
        dmax = float(np.max(np.abs(ref - got))) if te.any() else 0.0
        if dmax > 1e-9:
            sys.exit(f"⛔ фолд {f}: выгрузка расходится с sklearn (max|Δ| = {dmax:.2e}) — ничего не записано")
        np.savez_compressed(D / f"slotcv_f{f}.npz", **w)
        oof[te] = -ref
        sheets_f = sorted({s for s, ff in zip(S, F) if ff == f})
        print(f"★ фолд {f}: обучение {int(tr.sum())} пар, держанных {int(te.sum())} пар / {len(sheets_f)} листов; "
              f"сверка с прод-функцией max|Δ| = {dmax:.1e}")
    # списки листов фолдов — по ВСЕМ листам кэша (у листа без пар фолд — по скважине из карты)
    allsh = [l.strip() for l in (TS / "tcache_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    for f in range(a.k):
        lst = [s for s in allsh if fold_of(wm.get(Path(s).stem, "?")) == f]
        (D / f"slotcv_sheets_f{f}.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")
    # заочная мера: слоты с хотя бы одной честной парой — честная ли пара с лучшим скором
    key = [f"{s}|{p}" for s, p in zip(S, d["path"])]
    # группа слота: лист + путь + индекс слота восстановим по порядку — берём (лист, путь, y-блок) нельзя; используем
    # признак «слот» опосредованно: пары одного слота идут подряд в выгрузке `rows` ⇒ группа = смена (лист, путь) или
    # сброс ранга. Проще и точнее — по листу и пути: доля честных среди лучших K по числу честных пар листа.
    import collections
    grp = collections.defaultdict(list)
    for i, k in enumerate(key):
        grp[k].append(i)
    hit_old = hit_new = tot = 0
    for k, idx in grp.items():
        idx = np.array(idx); nh = int(H[idx].sum())
        if nh == 0:
            continue
        tot += nh
        hit_old += int(H[idx[np.argsort(-sc_old[idx])[:nh]]].sum())
        hit_new += int(H[idx[np.argsort(-oof[idx])[:nh]]].sum())
    print(f"★ заочно: честных пар {tot}; среди лучших по скору (столько же, сколько честных в листе-пути) — "
          f"старая модель {hit_old} ({100*hit_old/max(1,tot):.1f}%), новая вне фолда {hit_new} ({100*hit_new/max(1,tot):.1f}%)")

if a.replay:
    cmd = [PY, str(RND / "_trace_cache.py"), "replay", "--sheets", "tcache_sheets.txt", "--cache", str(TS / "tcache"),
           "--out", a.out, "--mode", f"B:{a.base}"]
    r = subprocess.run(cmd, cwd=str(RND), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-300:]); assert r.returncode == 0, r.stderr[-800:]
    for f in range(a.k):
        spec = f"SC:{a.base},slot={(D / f'slotcv_f{f}.npz').as_posix()}"
        cmd = [PY, str(RND / "_trace_cache.py"), "replay", "--sheets", str(D / f"slotcv_sheets_f{f}.txt"),
               "--cache", str(TS / "tcache"), "--out", a.out, "--mode", spec]
        r = subprocess.run(cmd, cwd=str(RND), capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(f"  повтор фолда {f}: код {r.returncode}; {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
        assert r.returncode == 0, r.stderr[-800:]

if a.compare:
    r = subprocess.run([PY, str(RND / "_replay_sweep.py"), "--skip-replay", "--tag", "slotcv", "--out", a.out,
                        "--base", a.base, "--var", "SC:slot=cv"], cwd=str(RND), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(r.stdout[-2500:]); print(r.stderr[-500:])
