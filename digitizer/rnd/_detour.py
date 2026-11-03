r"""_detour.py — ОБЪЕЗДЫ: КТО ВЕРЕН НА УЧАСТКЕ РАСХОЖДЕНИЯ — ТРАССА ВЫДАЧИ ИЛИ КАНДИДАТ КЭША (03.11, §6.353).

Потолок починки уходов кусками кандидатов кэша (`_repair_ceiling.py`): 29% лишних строк «верных» кривых, чистых (≤ 5%) 222 → 428 из
1223. Починка по прыжкам без туши (`_repair_stand.py`) берёт мало (+2 из 76): главные уходы — на пересечениях, где линии касаются и
трасса переходит на чужую без прыжка. Здесь: для пары «кривая выдачи w — кандидат t» окно — участок, где они расходятся (> 3 px), а
до и после (k строк) совпадают. Признаки обоих кусков (тушь, цвет против всей кривой, прыжки без туши, наклон на стыках, дрожание,
близость других кривых выдачи, вертикаль), метка — эталон (кусок t на линии, w нет → 1; наоборот → 0), бустинг вне фолда по
скважинам (фолды `line_choice_v6e`, сорт A — проверка). Замена: окна с P(t верен) > τ, перекрытия — по наибольшей P.

  _detour.py extract --sheets <список> --workers 2      → taskS/detour/<лист>.pkl
  _detour.py stand [--tau 0.6 0.75 0.9]                   → модель вне фолда, замена, мера приёмки и строгая до / после
"""
import sys, argparse, json, pickle, time, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np

from auto.detour import J, DARK, K, NEAR, PF, FEATS, seg_map, sheet_seg, look, piece_feats, windows, curve_windows  # noqa: F401 (§6.355: один код со стендом)

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "detour"


def one_extract(args):
    nm, part, src, got, field, segdir, outdir = args
    out = Path(outdir) / (sdir(nm) + ".pkl")
    if out.exists():
        return nm, "есть"
    import _sample_review as S, decode_levels as DL, _final_exif
    from _repair_ceiling import cands
    from dataset_build import find_image
    try:
        mt, mw, gts, wr = S.curves(src, got)
        C = cands(S.sheet_dir(nm), field)
        img = find_image(Path(src))
        if C is None or img is None:
            return nm, "нет кэша / скана"
        rgb = _final_exif.load_rgb(str(img))
    except Exception as e:
        return nm, f"{type(e).__name__}"
    lk = look(rgb); H, W = lk[3], lk[4]
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    seg = None
    if segdir:                                              # §: сеть туши вне фолда (скважина листа не в обучении)
        FO = json.loads((Path(segdir) / "folds.json").read_text(encoding="utf-8"))
        k_ = FO.get(Path(src).parent.parent.name)
        seg = sheet_seg(rgb, wr, str(Path(segdir) / (f"f{k_}.pt" if k_ is not None else "all.pt")))
    rec = {"sheet": nm, "part": part, "rows": [], "seg": bool(seg)}
    for g, w in wr.items():
        axis = CH[g][0] if CH.get(g) else None
        gt = gts.get(g)
        for ci, a, b, x in curve_windows(g, w, wr, C, lk, axis, seg):
            t = C[ci]; lab = np.nan
            if gt is not None:
                rr = [y for y in range(a, b + 1) if y in gt and y in w and y in t]
                if len(rr) >= 0.5 * (b - a + 1):
                    won = np.mean([abs(w[y] - gt[y]) <= 3 for y in rr]); woff = np.mean([abs(w[y] - gt[y]) > 10 for y in rr])
                    ton = np.mean([abs(t[y] - gt[y]) <= 3 for y in rr]); toff = np.mean([abs(t[y] - gt[y]) > 10 for y in rr])
                    lab = 1.0 if (ton >= 0.7 and woff >= 0.7) else 0.0 if (won >= 0.7 and toff >= 0.7) else np.nan
            rec["rows"].append((g, ci, a, b, x, lab))
    pickle.dump(rec, open(out, "wb"), protocol=4)
    nl = sum(1 for r in rec["rows"] if not np.isnan(r[5]))
    return nm, f"окон {len(rec['rows'])}, с меткой {nl}"


def sdir(nm):
    import hashlib
    st = Path(nm).stem
    return f"{st[:40]}_{hashlib.md5(st.encode('utf-8')).hexdigest()[:8]}"


def extract(a):
    import _sample_review as S
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    part = {nm: p for p, SS in SETS.items() for nm in SS}
    fld = set(l.strip() for l in (TS / "tcache_sheets.txt").read_text(encoding="utf-8").splitlines())
    names = [l.strip() for l in Path(a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for nm in names:
        got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is not None and nm in SRC and nm in part:
            jobs.append((nm, part[nm], SRC[nm], got, nm in fld, a.seg, a.out_dir))
    print(f"★ листов {len(jobs)}, процессов {a.workers}, сеть туши {a.seg or '—'} → {a.out_dir}")
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (nm, msg) in enumerate(pool.imap_unordered(one_extract, jobs, chunksize=2), 1):
            if k % 25 == 0 or "Error" in msg:
                print(f"  {k}/{len(jobs)} {time.time() - t0:.0f} с — {nm[:40]}: {msg}")
    print("★ извлечение готово")


def stand(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import _sample_review as S, _strict_stand as T, decode_levels as DL
    from _repair_ceiling import cands
    D = {}
    for f in sorted(q for q in Path(a.out_dir).glob("*.pkl") if not q.name.startswith("detour_")):   # модели — не листы
        r = pickle.load(open(f, "rb")); D[r["sheet"]] = r
    folds = json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))
    rows = [(nm, *x, -1 if r["part"] == "сорт A" else folds.get(nm, int(hashlib.md5(nm.encode("utf-8")).hexdigest(), 16) % 5))
            for nm, r in D.items() for x in r["rows"]]
    X = np.array([r[5] for r in rows], np.float32); Y = np.array([r[6] for r in rows], np.float32); FD = np.array([r[7] for r in rows])
    lab = ~np.isnan(Y); P = np.full(len(rows), np.nan)
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    for f in sorted(set(FD) - {-1}):
        tr = lab & (FD != f) & (FD != -1); te = FD == f
        m = mk(); m.fit(X[tr], Y[tr]); P[te] = m.predict_proba(X[te])[:, 1]
    m = mk(); m.fit(X[lab & (FD != -1)], Y[lab & (FD != -1)]); P[FD == -1] = m.predict_proba(X[FD == -1])[:, 1]
    if a.fit_only:                                      # модели пишутся только так: повтор читает их, стенд их не трогает
        od = Path(a.out_dir) / "oof"; od.mkdir(exist_ok=True)
        pickle.dump(m, open(od / "detour_all.pkl", "wb"))                   # §6.355: сорт A и листы вне фолдов
        for f in sorted(set(FD) - {-1}):
            mf = mk(); tr = lab & (FD != f) & (FD != -1); mf.fit(X[tr], Y[tr]); pickle.dump(mf, open(od / f"detour_f{f}.pkl", "wb"))
        (od / "detour_folds.json").write_text(json.dumps(folds, ensure_ascii=False), encoding="utf-8")
        mp = mk(); mp.fit(X[lab], Y[lab]); pickle.dump(mp, open(Path(a.out_dir) / "detour_prod.pkl", "wb"))   # все метки, с сортом A
        print(f"★ модели: {od} (фолды {sorted(set(FD) - {-1})}), прод {Path(a.out_dir) / 'detour_prod.pkl'}"); return
    for pk, msk in (("поле+вне", FD != -1), ("сорт A", FD == -1)):
        mm = msk & lab
        if mm.sum() and len(set(Y[mm])) == 2:
            print(f"★ {pk}: окон с меткой {mm.sum()} (кандидат верен {Y[mm].mean():.0%}), AUC вне фолда {roc_auc_score(Y[mm], P[mm]):.3f}")
    SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    fld = set(l.strip() for l in (TS / "tcache_sheets.txt").read_text(encoding="utf-8").splitlines())
    W_ = {}
    for (nm, g, ci, a_, b_, x, y, fd), p in zip(rows, P):
        W_.setdefault(nm, {}).setdefault(g, []).append((float(p), ci, a_, b_))
    TG = [(t_, g_) if g_ else t_ for t_ in a.tau for g_ in a.guard]
    jobs = [(nm, r["part"], SRC[nm], next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None), nm in fld,
             W_.get(nm, {}), TG) for nm, r in D.items()]
    jobs = [j for j in jobs if j[3] is not None]
    agg = {(t, pk): np.zeros(8) for t in TG for pk in ("поле+вне", "сорт A")}; ps = {k: [] for k in agg}
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (nm, part, out) in enumerate(pool.imap_unordered(eval_sheet, jobs, chunksize=2), 1):
            pk = "сорт A" if part == "сорт A" else "поле+вне"
            for tau, (v, dh, dc) in out.items():
                agg[(tau, pk)] += v; ps[(tau, pk)].append((dh, dc))
            if k % 100 == 0:
                print(f"  мера: {k}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    for tau in TG:
        for pk in ("поле+вне", "сорт A"):
            v = agg[(tau, pk)]; d = np.array(ps[(tau, pk)]) if ps[(tau, pk)] else np.zeros((0, 2))
            sg = lambda col: f"↑{int((d[:, col] > 0).sum())}/↓{int((d[:, col] < 0).sum())}" if len(d) else ""
            print(f"  τ {tau} {pk:9s}: приёмка {int(v[0])} → {int(v[1])} ({int(v[1] - v[0]):+d}, листов {sg(0)}); чистых ≤ 1% {int(v[2])} → "
                  f"{int(v[3])}; ≤ 5% {int(v[4])} → {int(v[5])} ({int(v[5] - v[4]):+d}, листов {sg(1)}); лишних строк {int(v[6])} → {int(v[7])} "
                  f"({(v[7] - v[6]) / max(v[6], 1):+.1%})")


def _measure(S, T, ww, gt, lw, lg, ch):
    S._TREES.clear()
    wl = S.relevel(ww, lw, lg, ch)
    hon = S.HON(*S.err(wl, gt))
    st = T.strict(S, wl, gt, frozenset(y for y in ww if lw.get(y, 0) != lg.get(y, 0)))
    c5 = bool(hon and st and st["off_o"] <= 0.05 * st["n_o"] and st["off_t"] <= 0.05 * st["n_t"])
    c1 = bool(hon and st and st["off_o"] <= 0.01 * st["n_o"] and st["off_t"] <= 0.01 * st["n_t"])
    return hon, c1, c5, st["off_o"] if st else 0


def eval_sheet(args):
    """мера листа до / после замены окон при каждом τ (база считается один раз; кривая без замен — база)"""
    nm, part, src, got, field, Wn, taus = args
    import _sample_review as S, _strict_stand as T, decode_levels as DL
    from _repair_ceiling import cands
    mt, mw, gts, wr = S.curves(src, got)
    C = cands(S.sheet_dir(nm), field)
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    from auto.detour import take, jump_ok
    out = {tau: [np.zeros(8), 0, 0] for tau in taus}
    for g, gt in gts.items():
        if g not in wr:
            continue
        w = wr[g]; lw, lg, ch = LW.get(g, {}), LG.get(g, {}), CH.get(g, [])
        r0 = _measure(S, T, w, gt, lw, lg, ch)
        for tau in taus:                                  # τ или (τ, G) — G: запрет прыжка, `auto.detour.jump_ok`
            t_, G = tau if isinstance(tau, tuple) else (tau, 0)
            w2 = dict(w)
            taken = take(Wn.get(g, []), t_, ok=lambda ci, a_, b_: jump_ok(w, C[ci], a_, b_, G))
            for ci, a_, b_ in taken:
                t = C[ci]
                for yy in range(a_, b_ + 1):
                    if yy in t:
                        w2[yy] = t[yy]
            r1 = _measure(S, T, w2, gt, lw, lg, ch) if taken else r0
            out[tau][0] += [r0[0], r1[0], r0[1], r1[1], r0[2], r1[2], r0[3], r1[3]]
            out[tau][1] += r1[0] - r0[0]; out[tau][2] += r1[2] - r0[2]
    return nm, part, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "stand"])
    ap.add_argument("--sheets", default=str(TS / "segclean_sheets.txt"))
    ap.add_argument("--workers", type=int, default=2, help="процессов извлечения / меры стенда")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seg", default="", help="каталог сетей туши вне фолда (folds.json, f{k}.pt) — признак сети по куску")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--fit-only", action="store_true", help="только модели (фолды, все, прод) — без меры")
    ap.add_argument("--guard", type=float, nargs="+", default=[0], help="запрет прыжка куска кандидата > G px (0 — без)")
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else stand(a)
