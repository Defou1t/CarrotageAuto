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
import sys, argparse, json, pickle, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "detour"
J, DARK, K, NEAR = 8, 45, 6, 3.0
PF = ["n", "ink", "inkrow", "colr", "jit", "maxj", "nj_dry", "dup", "vcol", "edge", "s_in", "s_out"]
FEATS = [f"w_{f}" for f in PF] + [f"t_{f}" for f in PF] + [f"d_{f}" for f in PF] + ["sep", "logn"]


def _ink(gray, paper, y, xa, xb):
    lo, hi = int(min(xa, xb)) + 2, int(max(xa, xb)) - 2
    if hi <= lo or not (0 <= y < gray.shape[0] - 1):
        return 1.0
    lo, hi = max(0, lo), min(gray.shape[1], hi)
    seg = np.minimum(gray[y, lo:hi], gray[y + 1, lo:hi]).astype(float)
    return float(np.mean(seg < paper - DARK)) if len(seg) else 1.0


def _slope(tr, ys):
    ys = [y for y in ys if y in tr]
    if len(ys) < 3:
        return 0.0
    return float(np.polyfit(ys, [tr[y] for y in ys], 1)[0])


def seg_map(rgb, ys_xs, net, pad=6):
    """карта сети туши (`auto.ink_seg`) по рамке всех трасс — полосами по сетке плиток (как `_seg_feat.py`), float16 →
    функция (ys, xs) → значения max в ±1 px"""
    from auto import ink_seg as IS
    H, W = rgb.shape[:2]
    y0 = max(0, min(y for y, _ in ys_xs)); y1 = min(H, max(y for y, _ in ys_xs) + 1)
    x0 = max(0, int(min(x for _, x in ys_xs)) - pad); x1 = min(W, int(max(x for _, x in ys_xs)) + pad + 1)
    M = np.zeros((y1 - y0, x1 - x0), np.float16); band = (1024 - 2 * 64) * 8
    for s0 in range(y0, y1, band):
        s1 = min(y1, s0 + band)
        M[s0 - y0:s1 - y0] = IS.prob_map(rgb, s0, s1, x0, x1, net).astype(np.float16)

    def at(ys, xs):
        yy = np.clip(np.asarray(ys) - y0, 0, M.shape[0] - 1); xx = np.round(np.asarray(xs)).astype(int) - x0
        return np.max(np.stack([M[yy, np.clip(xx + o, 0, M.shape[1] - 1)] for o in (-1, 0, 1)]), 0).astype(np.float32)
    return at


def piece_feats(tr, a, b, ctx, others, axis, seg=None):
    """признаки куска tr на строках [a, b]; ctx — (gray, paper, rgb_s, ref_color, H, W); seg — карта сети туши (или None)"""
    gray, paper, rgb_s, ref, H, W = ctx
    rr = [y for y in range(a, b + 1) if y in tr]
    if len(rr) < 3:
        return None
    xs = np.array([tr[y] for y in rr]); yi = np.clip(np.array(rr), 0, H - 1); xi = np.clip(np.round(xs).astype(int), 1, W - 2)
    under = np.minimum(np.minimum(gray[yi, xi - 1], gray[yi, xi]), gray[yi, xi + 1]).astype(float)
    dk = paper - under
    sel = slice(None, None, max(1, len(rr) // 150))
    colr = float(np.linalg.norm(np.median(rgb_s(yi[sel], xi[sel]), 0) - ref))
    dx = np.abs(np.diff(xs)) if len(xs) > 1 else np.zeros(1)
    dry = 0
    for i in np.flatnonzero(dx > J)[:50]:
        dry += _ink(gray, paper, rr[i], xs[i], xs[i + 1]) < 0.5
    dup = 0.0
    for o in others:
        v = [abs(o[y] - tr[y]) <= NEAR for y in rr[::3] if y in o]
        if v:
            dup = max(dup, float(np.mean(v)))
    xm = float(np.median(xs)); yc = (a + b) // 2
    ca, cb = max(0, yc - 400), min(H, yc + 400); xc = int(np.clip(round(xm), 1, W - 2))
    vcol = float(np.mean(paper - gray[ca:cb:4, xc - 1:xc + 2].min(1) > DARK))
    edge = min(abs(xm - axis["x_left"]), abs(xm - axis["x_right"])) if axis else np.nan
    s_in = abs(_slope(tr, range(a, a + K)) - _slope(tr, range(a - K, a)))
    s_out = abs(_slope(tr, range(b - K + 1, b + 1)) - _slope(tr, range(b + 1, b + 1 + K)))
    f = [len(rr), float(np.mean(dk)), float(np.mean(dk > DARK)), colr, float(np.mean(dx)), float(dx.max()), dry, dup, vcol, edge,
         s_in, s_out]
    if seg is not None:
        sv = seg(rr[::2], xs[::2])
        f += [float(np.mean(sv)), float(np.mean(sv > 0.5)), float(np.min(np.convolve(sv, np.ones(5) / 5, "valid"))) if len(sv) >= 5 else float(np.min(sv))]
    return f


def windows(w, t, k=K, minlen=5, maxlen=3000):
    """окна расхождения w и t (> NEAR px) с совпадением (≤ NEAR) на ≥ k/2 из k строк до и после → [(a, b)]"""
    ys = [y for y in sorted(w) if y in t]
    if len(ys) < 50:
        return []
    agree = {y: abs(w[y] - t[y]) <= NEAR for y in ys}
    out, i = [], 0
    while i < len(ys):
        if agree[ys[i]]:
            i += 1; continue
        j = i
        while j + 1 < len(ys) and not agree[ys[j + 1]] and ys[j + 1] - ys[j] <= 2:
            j += 1
        a, b = ys[i], ys[j]
        if minlen <= b - a + 1 <= maxlen:
            pre = sum(1 for y in range(a - k, a) if agree.get(y)); post = sum(1 for y in range(b + 1, b + 1 + k) if agree.get(y))
            if pre >= k // 2 and post >= k // 2:
                out.append((a, b))
        i = j + 1
    return out


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
    gray = rgb.min(axis=2) if rgb.ndim == 3 else rgb
    rgb_s = (lambda yy, xx: rgb[yy, xx].astype(np.float32)) if rgb.ndim == 3 else (lambda yy, xx: np.repeat(gray[yy, xx][:, None], 3, 1).astype(np.float32))
    paper = float(np.percentile(gray[::7, ::7], 80)); H, W = gray.shape
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    seg = None
    if segdir:                                              # §: сеть туши вне фолда (скважина листа не в обучении)
        FO = json.loads((Path(segdir) / "folds.json").read_text(encoding="utf-8"))
        k_ = FO.get(Path(src).parent.parent.name)
        net = str(Path(segdir) / (f"f{k_}.pt" if k_ is not None else "all.pt"))
        pts = [(y, x) for w in wr.values() for y, x in list(w.items())[::50]]
        if pts:
            seg = seg_map(rgb, pts + [(min(y for y, _ in pts), max(0, min(x for _, x in pts) - 60)),
                                      (max(y for y, _ in pts), min(W - 1, max(x for _, x in pts) + 60))], net)
    rec = {"sheet": nm, "part": part, "rows": [], "seg": bool(seg)}
    for g, w in wr.items():
        if len(w) < 50:
            continue
        ys = sorted(w)[::max(1, len(w) // 3000)]
        yi = np.clip(np.array(ys), 0, H - 1); xi = np.clip(np.round([w[y] for y in ys]).astype(int), 1, W - 2)
        ref = np.median(rgb_s(yi, xi), 0)
        ctx = (gray, paper, rgb_s, ref, H, W)
        others = [o for kk, o in wr.items() if kk != g]
        axis = CH[g][0] if CH.get(g) else None
        gt = gts.get(g)
        seen = set()
        for ci, t in enumerate(C):
            if sum(1 for y in ys if y in t and abs(t[y] - w[y]) <= NEAR) > 0.97 * len(ys):
                continue                                    # сама трасса выдачи (или её копия)
            for a, b in windows(w, t):
                if (a, b, round(t.get(a, 0))) in seen:
                    continue
                seen.add((a, b, round(t.get(a, 0))))
                fw = piece_feats(w, a, b, ctx, others, axis, seg); ft = piece_feats(t, a, b, ctx, others, axis, seg)
                if fw is None or ft is None:
                    continue
                sep = float(np.median([abs(w[y] - t[y]) for y in range(a, b + 1) if y in w and y in t]))
                lab = np.nan
                if gt is not None:
                    rr = [y for y in range(a, b + 1) if y in gt and y in w and y in t]
                    if len(rr) >= 0.5 * (b - a + 1):
                        won = np.mean([abs(w[y] - gt[y]) <= 3 for y in rr]); woff = np.mean([abs(w[y] - gt[y]) > 10 for y in rr])
                        ton = np.mean([abs(t[y] - gt[y]) <= 3 for y in rr]); toff = np.mean([abs(t[y] - gt[y]) > 10 for y in rr])
                        lab = 1.0 if (ton >= 0.7 and woff >= 0.7) else 0.0 if (won >= 0.7 and toff >= 0.7) else np.nan
                rec["rows"].append((g, ci, a, b, np.array(fw + ft + list(np.array(fw) - np.array(ft)) + [sep, np.log1p(b - a + 1)],
                                                         np.float32), lab))
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
    for f in sorted(Path(a.out_dir).glob("*.pkl")):
        r = pickle.load(open(f, "rb")); D[r["sheet"]] = r
    folds = json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))
    rows = [(nm, *x, -1 if r["part"] == "сорт A" else folds.get(nm, hash(nm) % 5)) for nm, r in D.items() for x in r["rows"]]
    X = np.array([r[5] for r in rows], np.float32); Y = np.array([r[6] for r in rows], np.float32); FD = np.array([r[7] for r in rows])
    lab = ~np.isnan(Y); P = np.full(len(rows), np.nan)
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    for f in sorted(set(FD) - {-1}):
        tr = lab & (FD != f) & (FD != -1); te = FD == f
        m = mk(); m.fit(X[tr], Y[tr]); P[te] = m.predict_proba(X[te])[:, 1]
    m = mk(); m.fit(X[lab & (FD != -1)], Y[lab & (FD != -1)]); P[FD == -1] = m.predict_proba(X[FD == -1])[:, 1]
    pickle.dump(m, open(TS / "detour_all.pkl", "wb"))
    for pk, msk in (("поле+вне", FD != -1), ("сорт A", FD == -1)):
        mm = msk & lab
        if mm.sum() and len(set(Y[mm])) == 2:
            print(f"★ {pk}: окон с меткой {mm.sum()} (кандидат верен {Y[mm].mean():.0%}), AUC вне фолда {roc_auc_score(Y[mm], P[mm]):.3f}")
    SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    fld = set(l.strip() for l in (TS / "tcache_sheets.txt").read_text(encoding="utf-8").splitlines())
    W_ = {}
    for (nm, g, ci, a_, b_, x, y, fd), p in zip(rows, P):
        W_.setdefault((nm, g), []).append((p, ci, a_, b_))
    for tau in a.tau:
        agg = {"поле+вне": np.zeros(8), "сорт A": np.zeros(8)}; ps = {"поле+вне": [], "сорт A": []}
        for nm, r in D.items():
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            mt, mw, gts, wr = S.curves(SRC[nm], got)
            C = cands(S.sheet_dir(nm), nm in fld)
            LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
            CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
            LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
            pk = "сорт A" if r["part"] == "сорт A" else "поле+вне"
            dh = dc = 0
            for g, gt in gts.items():
                if g not in wr:
                    continue
                w = wr[g]; w2 = dict(w); taken = []
                for p, ci, a_, b_ in sorted(W_.get((nm, g), []), key=lambda z: -z[0]):
                    if p <= tau:
                        break
                    if any(not (b_ < x0 or a_ > x1) for x0, x1 in taken):
                        continue
                    t = C[ci]
                    for yy in range(a_, b_ + 1):
                        if yy in t:
                            w2[yy] = t[yy]
                    taken.append((a_, b_))
                res = []
                for ww in (w, w2):
                    S._TREES.clear()
                    lw, lg = LW.get(g, {}), LG.get(g, {})
                    wl = S.relevel(ww, lw, lg, CH.get(g, []))
                    hon = S.HON(*S.err(wl, gt))
                    st = T.strict(S, wl, gt, frozenset(y for y in ww if lw.get(y, 0) != lg.get(y, 0)))
                    c5 = bool(hon and st and st["off_o"] <= 0.05 * st["n_o"] and st["off_t"] <= 0.05 * st["n_t"])
                    c1 = bool(hon and st and st["off_o"] <= 0.01 * st["n_o"] and st["off_t"] <= 0.01 * st["n_t"])
                    res.append((hon, c1, c5, st["off_o"] if st else 0))
                agg[pk] += [res[0][0], res[1][0], res[0][1], res[1][1], res[0][2], res[1][2], res[0][3], res[1][3]]
                dh += res[1][0] - res[0][0]; dc += res[1][2] - res[0][2]
            ps[pk].append((dh, dc))
        for pk, v in agg.items():
            d = np.array(ps[pk]) if ps[pk] else np.zeros((0, 2))
            sg = lambda col: f"↑{int((d[:, col] > 0).sum())}/↓{int((d[:, col] < 0).sum())}" if len(d) else ""
            print(f"  τ {tau} {pk:9s}: приёмка {int(v[0])} → {int(v[1])} ({int(v[1] - v[0]):+d}, листов {sg(0)}); чистых ≤ 1% {int(v[2])} → "
                  f"{int(v[3])}; ≤ 5% {int(v[4])} → {int(v[5])} ({int(v[5] - v[4]):+d}, листов {sg(1)}); лишних строк {int(v[6])} → {int(v[7])} "
                  f"({(v[7] - v[6]) / max(v[6], 1):+.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "stand"])
    ap.add_argument("--sheets", default=str(TS / "segclean_sheets.txt"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seg", default="", help="каталог сетей туши вне фолда (folds.json, f{k}.pt) — признак сети по куску")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else stand(a)
