r"""_seg_clean.py — ОБУЧЕННАЯ ВЫЧИСТКА УЧАСТКОВ ТРАССЫ: УХОДЫ НА СЕТКУ, РАМКУ, ЦИФРЫ, ЧУЖИЕ КРИВЫЕ (03.11, §6.353).

Геометрические вычистки (спайк по хорде, дубль, обмен на пересечении, мост по прыжкам без туши) мерой приёмки и строгой — все в минус:
по одной форме трассы не отличить уход с линии от настоящего пика / касания. Здесь трасса выдачи режется на участки по прыжкам
(|Δx| > J между соседними строками), у участка — признаки трассы и скана, метка — эталон (участок на линии / уход), модель —
бустинг вне фолда по скважинам (фолды `line_choice_v6e`; сорт A — только проверка, модель по всему поле + вне).

  _seg_clean.py extract --sheets <список|all> --workers 2   → taskS/segclean/<лист>.pkl (участки, признаки, метки)
  _seg_clean.py stand [--tau 0.5 0.7 0.85]                  → модели вне фолда, вычистка, мера приёмки и строгая до / после

Признаки участка: длина; прыжки на входе / выходе и тушь вдоль их перемычек; тушь и тёмность под трассой; цвет участка против цвета
всей кривой; размах x и дрожание; положение в шкале (доля, расстояние до края); близость других кривых выдачи (дубль); тёмный
вертикальный столбец под участком на ±400 строк (линия сетки / рамка); возврат — насколько соседи участка сходятся без него.
Метка: доля строк участка дальше 10 px от эталона (в пикселях и в значениях — меньшее) > 0.5 → уход, < 0.1 → на линии.
"""
import sys, argparse, json, pickle, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "segclean"
J, DARK = 8, 45
FEATS = ["logn", "jin", "jout", "ink_in", "ink_out", "dark", "inkrow", "colr", "ptp", "jit", "xrel", "edge", "dup", "dmin",
         "vcol", "ret", "far", "share", "lvl"]


def ink_frac(gray, paper, y, xa, xb):
    lo, hi = int(min(xa, xb)) + 2, int(max(xa, xb)) - 2
    if hi <= lo or not (0 <= y < gray.shape[0] - 1):
        return 1.0
    lo, hi = max(0, lo), min(gray.shape[1], hi)
    seg = np.minimum(gray[y, lo:hi], gray[y + 1, lo:hi]).astype(float)
    return float(np.mean(seg < paper - DARK)) if len(seg) else 1.0


def split(w):
    """→ (ys, [(i0, i1)]) — участки без прыжков > J и разрывов > 2 строк"""
    ys = sorted(w); segs, st = [], 0
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] > 2 or abs(w[ys[i]] - w[ys[i - 1]]) > J:
            segs.append((st, i - 1)); st = i
    if ys:
        segs.append((st, len(ys) - 1))
    return ys, segs


def features(g, w, others, gray, rgb_s, paper, axis, lw):
    """признаки участков кривой g (выдача w) → (участки [(y0, y1)], матрица признаков)"""
    ys, segs = split(w)
    H, W = gray.shape
    xs = np.array([w[y] for y in ys], float)
    xi = np.clip(np.round(xs).astype(int), 1, W - 2); yi = np.clip(np.array(ys), 0, H - 1)
    under = np.minimum(np.minimum(gray[yi, xi - 1], gray[yi, xi]), gray[yi, xi + 1]).astype(float)
    darkv = paper - under
    sub = slice(None, None, max(1, len(ys) // 3000))
    col = rgb_s(yi[sub], xi[sub]); ref = np.median(col, 0)
    xl, xr = (axis["x_left"], axis["x_right"]) if axis else (np.nan, np.nan)
    Wsc = abs(xr - xl) if axis else np.nan
    F, R = [], []
    for i0, i1 in segs:
        n = i1 - i0 + 1; y0, y1 = ys[i0], ys[i1]
        seg_x = xs[i0:i1 + 1]
        jin = abs(xs[i0] - xs[i0 - 1]) if i0 > 0 and ys[i0] - ys[i0 - 1] <= 2 else 0.0
        jout = abs(xs[i1 + 1] - xs[i1]) if i1 + 1 < len(ys) and ys[i1 + 1] - ys[i1] <= 2 else 0.0
        ink_in = ink_frac(gray, paper, ys[i0 - 1], xs[i0 - 1], xs[i0]) if jin else 1.0
        ink_out = ink_frac(gray, paper, ys[i1], xs[i1], xs[i1 + 1]) if jout else 1.0
        step = max(1, n // 200)
        sel = np.arange(i0, i1 + 1, step)
        dk = darkv[sel]
        c = rgb_s(yi[sel], xi[sel]); colr = float(np.linalg.norm(np.median(c, 0) - ref))
        xm = float(np.median(seg_x))
        dd, dup = [], []
        for o in others:
            v = [abs(o[ys[k]] - xs[k]) for k in sel if ys[k] in o]
            if v:
                dd.append(np.median(v)); dup.append(np.mean(np.array(v) <= 3))
        dmin = min(dd) if dd else 999.0; dupf = max(dup) if dup else 0.0
        yc = (y0 + y1) // 2; ca, cb = max(0, yc - 400), min(H, yc + 400); xc = int(np.clip(round(xm), 1, W - 2))
        vcol = float(np.mean(paper - gray[ca:cb:4, xc - 1:xc + 2].min(1) > DARK))
        if i0 > 0 and i1 + 1 < len(ys):
            ret = abs(xs[i1 + 1] - xs[i0 - 1]); far = abs(xm - (xs[i1 + 1] + xs[i0 - 1]) / 2)
        else:
            ret, far = -1.0, -1.0
        F.append([np.log1p(n), jin, jout, ink_in, ink_out, float(np.mean(dk)), float(np.mean(dk > DARK)), colr,
                  float(np.ptp(seg_x)), float(np.mean(np.abs(np.diff(seg_x)))) if n > 1 else 0.0,
                  (xm - xl) / Wsc if axis else np.nan, min(abs(xm - xl), abs(xm - xr)) if axis else np.nan,
                  dupf, dmin, vcol, ret, far, n / len(ys), lw.get(ys[i0], 0)])
        R.append((y0, y1))
    return R, np.array(F, np.float32)


def one_extract(args):
    nm, part, src, got = args
    out = OUT / (S_dir(nm) + ".pkl")
    if out.exists():
        return nm, "есть"
    import _sample_review as S, decode_levels as DL, _final_exif
    from dataset_build import find_image
    try:
        mt, mw, gts, wr = S.curves(src, got)
        img = find_image(Path(src))
        if img is None:
            return nm, "нет скана"
        rgb = _final_exif.load_rgb(str(img))
    except Exception as e:
        return nm, f"{type(e).__name__}"
    gray = rgb.min(axis=2) if rgb.ndim == 3 else rgb
    rgb_s = (lambda yy, xx: rgb[yy, xx].astype(np.float32)) if rgb.ndim == 3 else (lambda yy, xx: np.repeat(gray[yy, xx][:, None], 3, 1).astype(np.float32))
    paper = float(np.percentile(gray[::7, ::7], 80))
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    rec = {"sheet": nm, "part": part, "curves": {}}
    for g, w in wr.items():
        if len(w) < 30:
            continue
        others = [o for k, o in wr.items() if k != g]
        ax = CH.get(g, [None])[0] if CH.get(g) else None
        R, F = features(g, w, others, gray, rgb_s, paper, ax, LW.get(g, {}))
        lab = np.full(len(R), np.nan, np.float32)
        if g in gts:
            gt = gts[g]; wl = S.relevel(w, LW.get(g, {}), LG.get(g, {}), CH.get(g, []))
            for k, (y0, y1) in enumerate(R):
                rr = [y for y in range(y0, y1 + 1) if y in w and y in gt]
                if len(rr) >= max(1, (y1 - y0 + 1) // 2):
                    off = np.mean([min(abs(w[y] - gt[y]), abs(wl[y] - gt[y])) > 10 for y in rr])
                    lab[k] = 1.0 if off > 0.5 else 0.0 if off < 0.1 else np.nan
        rec["curves"][g] = {"segs": R, "X": F, "y": lab}
    pickle.dump(rec, open(out, "wb"), protocol=4)
    return nm, f"кривых {len(rec['curves'])}, участков {sum(len(c['segs']) for c in rec['curves'].values())}"


def S_dir(nm):
    import hashlib
    st = Path(nm).stem
    return f"{st[:40]}_{hashlib.md5(st.encode('utf-8')).hexdigest()[:8]}"


def extract(a):
    import _sample_review as S
    OUT.mkdir(parents=True, exist_ok=True)
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    part = {nm: p for p, SS in SETS.items() for nm in SS}
    if a.sheets == "all":
        names = sorted(part)
    else:
        names = [l.strip() for l in Path(a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for nm in names:
        got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is not None and nm in SRC and nm in part:
            jobs.append((nm, part[nm], SRC[nm], got))
    print(f"★ листов {len(jobs)}, процессов {a.workers}")
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (nm, msg) in enumerate(pool.imap_unordered(one_extract, jobs, chunksize=2), 1):
            if k % 25 == 0 or "Error" in msg:
                print(f"  {k}/{len(jobs)} {time.time() - t0:.0f} с — {nm[:40]}: {msg}")
    print("★ извлечение готово")


def load_all():
    D = {}
    for f in sorted(OUT.glob("*.pkl")):
        r = pickle.load(open(f, "rb")); D[r["sheet"]] = r
    return D


def stand(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import _sample_review as S, _strict_stand as T, decode_levels as DL
    D = load_all()
    folds = json.loads((TS / "line_choice_v6e" / "line_choice_folds.json").read_text(encoding="utf-8"))
    rows = []                                                       # (лист, кривая, k, фолд, признаки, метка)
    for nm, r in D.items():
        fd = -1 if r["part"] == "сорт A" else folds.get(nm, hash(nm) % 5)
        for g, c in r["curves"].items():
            for k in range(len(c["segs"])):
                rows.append((nm, g, k, fd, c["X"][k], c["y"][k]))
    X = np.array([x for *_, x, _ in rows], np.float32); Y = np.array([y for *_, y in rows], np.float32)
    FD = np.array([f for _, _, _, f, _, _ in rows])
    P = np.full(len(rows), np.nan)
    lab = ~np.isnan(Y)
    for f in sorted(set(FD) - {-1}):
        tr = lab & (FD != f) & (FD != -1); te = FD == f
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], Y[tr]); P[te] = m.predict_proba(X[te])[:, 1]
    trA = lab & (FD != -1)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    m.fit(X[trA], Y[trA]); P[FD == -1] = m.predict_proba(X[FD == -1])[:, 1]
    pickle.dump(m, open(TS / "seg_clean_all.pkl", "wb"))
    for part_, msk in (("поле+вне", FD != -1), ("сорт A", FD == -1)):
        mm = msk & lab
        if mm.sum() and len(set(Y[mm])) == 2:
            print(f"★ {part_}: участков с меткой {mm.sum()} (уход {Y[mm].mean():.0%}), AUC вне фолда {roc_auc_score(Y[mm], P[mm]):.3f}")
    PR = {}
    for (nm, g, k, *_), p in zip(rows, P):
        PR.setdefault((nm, g), {})[k] = p
    # вычистка по порогам и замер
    SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    for tau in a.tau:
        for mode in a.modes:
            agg = {"поле+вне": np.zeros(6), "сорт A": np.zeros(6)}
            for nm, r in D.items():
                got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
                mt, mw, gts, wr = S.curves(SRC[nm], got)
                LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
                CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
                LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
                pk = "сорт A" if r["part"] == "сорт A" else "поле+вне"
                for g, gt in gts.items():
                    if g not in wr or g not in r["curves"]:
                        continue
                    w = wr[g]; segs = r["curves"][g]["segs"]; pr = PR.get((nm, g), {})
                    bad = [k for k in range(len(segs)) if pr.get(k, 0) > tau]
                    w2 = clean(w, segs, set(bad), mode, a.lmax)
                    res = []
                    for ww in (w, w2):
                        S._TREES.clear()
                        lw, lg = LW.get(g, {}), LG.get(g, {})
                        wl = S.relevel(ww, lw, lg, CH.get(g, []))
                        hon = S.HON(*S.err(wl, gt)) if ww else False
                        st = T.strict(S, wl, gt, frozenset(y for y in ww if lw.get(y, 0) != lg.get(y, 0))) if ww else None
                        cl = bool(hon and st and st["off_o"] <= 0.01 * st["n_o"] and st["off_t"] <= 0.01 * st["n_t"])
                        res.append((hon, cl, st["off_o"] if st else 0, st["n_o"] if st else 0))
                    agg[pk] += [res[0][0], res[1][0], res[0][1], res[1][1], res[0][2], res[1][2]]
            for pk, v in agg.items():
                print(f"  τ {tau} {mode:6s} {pk:9s}: приёмка {int(v[0])} → {int(v[1])} ({int(v[1] - v[0]):+d}), чистых {int(v[2])} → "
                      f"{int(v[3])} ({int(v[3] - v[2]):+d}), лишних строк {int(v[4])} → {int(v[5])} ({(v[5] - v[4]) / max(v[4], 1):+.0%})")


def clean(w, segs, bad, mode, lmax):
    """участки bad → мост по хорде между соседними оставленными (mode bridge, длина ≤ lmax) или пропуск (gap)"""
    if not bad:
        return w
    ys = sorted(w); out = dict(w)
    k = 0
    while k < len(segs):
        if k not in bad:
            k += 1; continue
        j = k
        while j + 1 < len(segs) and (j + 1) in bad:
            j += 1
        y0, y1 = segs[k][0], segs[j][1]
        prev = segs[k - 1][1] if k > 0 else None; nxt = segs[j + 1][0] if j + 1 < len(segs) else None
        if mode == "bridge" and prev is not None and nxt is not None and nxt - prev <= lmax:
            xa, xb = w[prev], w[nxt]
            for y in range(prev + 1, nxt):
                out[y] = xa + (xb - xa) * (y - prev) / (nxt - prev)
        else:
            for y in range(y0, y1 + 1):
                out.pop(y, None)
        k = j + 1
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "stand"])
    ap.add_argument("--sheets", default="all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.7, 0.85])
    ap.add_argument("--modes", nargs="+", default=["bridge", "gap"])
    ap.add_argument("--lmax", type=int, default=400)
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else stand(a)
