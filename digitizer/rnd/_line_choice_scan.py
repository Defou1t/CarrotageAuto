r"""_line_choice_scan.py — ДАННЫЕ ДЛЯ ОБУЧЕННОГО ВЫБОРА ТРАССЫ ДЛЯ СЛОТА (04.10, §6.280).

§6.279: у 31% кривых поля честная трасса уже есть среди кандидатов кэша, но в слоте выдачи другая. Здесь для каждого листа (поле +
сорт A) и каждого слота рамки — все кандидаты кэша (прод-путь `traces`, декодер `alt`), чья медиана x в полосе шкалы слота (±10%):
- источник, длина, покрытие строк слота, положение в полосе (квантили u);
- стиль вдоль трассы (`auto/name_style.style_along`): толщина, цвет, затемнение, разрывы; устойчивая извилистость;
- стоит ли кандидат в этом слоте выдачи NS / в другом слоте;
- метка — честен ли для эталона слота (на плоскости в обе стороны по пикселям, покрытие ≥ 0.9).
Пишет по листу в `--out/<лист>.pkl` (подхват), `--workers` процессов.

  _line_choice_scan.py --workers 4
"""
import sys, argparse, pickle, hashlib, time, os, re
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from _cand_ceiling import _tree, dense_tr


def rough_robust(x):
    if x is None or len(x) < 400:
        return np.nan
    from scipy.ndimage import median_filter
    x = np.asarray(x, float)
    s_ = median_filter(x, size=41, mode="nearest")
    dev = np.abs(x - s_); jump = np.abs(np.diff(x, prepend=x[0])) > 40
    vals = [np.mean(dev[i:i + 400]) for i in range(0, len(x) - 400, 200) if not jump[i:i + 400].any()]
    if len(vals) < 3:
        return np.nan
    return float(np.median(vals) / max(5.0, np.percentile(x, 95) - np.percentile(x, 5)))


def honest_d(ct, tc, gt, tg):
    com = [y for y in gt if y in ct]
    if len(com) < 30:
        return False, 0.0, 99.0
    cov = len(com) / max(1, len(gt))
    d1, _ = tc.query(np.array([[y, gt[y]] for y in com], float))
    d2, _ = tg.query(np.array([[y, ct[y]] for y in com], float))
    m = max(float(np.median(d1)), float(np.median(d2)))
    return bool(m <= 3 and cov >= 0.9), cov, m


def one(job):
    sh, tc_path, got, src, img, P = job
    outp = Path(P["out"]) / (hashlib.md5(sh.encode("utf-8")).hexdigest()[:16] + ".pkl")
    if outp.exists():
        return sh, 0
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from auto import meta as M, name_style as NS, level_ink as LI
    import decode_levels as DL
    c = pickle.load(open(tc_path, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    cands = [("prod", i, dense_tr(un(t))) for i, (L, t) in enumerate(c["traces"] or [])]
    cands += [("dec", i, dense_tr(un(t))) for i, (L, t) in enumerate(c["alt"] or [])]
    cands = [x for x in cands if len(x[2]) >= 50]
    mt = extract(src); mw = extract(got)
    fc = {cc["name"]: cc for cc in mt["curves"] if M.mnem_root(cc["name"]) != "DA"}
    G = {n: dense(cc) for n, cc in fc.items() if sum(1 for x in cc["xs"] if x != NULL) >= 50}
    W = {cc["name"]: dense(cc) for cc in mw["curves"] if M.mnem_root(cc["name"]) != "DA"}
    im = Image.open(img)
    gray = np.asarray(im.convert("L")); rgb = np.asarray(im.convert("RGB"))
    paper = LI.paper_of(gray)
    TC = [_tree(x[2]) for x in cands]
    CS = []
    for src_, i, ct in cands:
        ys = np.array(sorted(ct)); xs = np.array([ct[y] for y in ys])
        st = NS.style_along(rgb, gray, paper, ys[::8], xs[::8])
        CS.append(dict(src=src_, idx=i, n=len(ct), y0=int(ys[0]), y1=int(ys[-1]), xs_q=np.percentile(xs, [10, 50, 90]),
                       style=st, rough=rough_robust(xs)))
    # какой кандидат стоит в каком слоте выдачи (совпадение x в 1 px на ≥ 90% общих строк)
    used = {}
    for wn, w in W.items():
        if len(w) < 50:
            continue
        for k, (_, _, ct) in enumerate(cands):
            com = [y for y in w if y in ct]
            if len(com) >= 0.5 * len(w) and np.mean([abs(w[y] - ct[y]) <= 1.0 for y in com[::5]]) >= 0.9:
                used.setdefault(k, wn)
    slots = []
    for n, cc in fc.items():
        fam = DL.build_family(mt, cc)
        if fam:
            xl = min(min(s_["x_left"], s_["x_right"]) for s_ in fam); xr = max(max(s_["x_left"], s_["x_right"]) for s_ in fam)
        else:
            continue
        wd = max(1.0, xr - xl)
        gt = G.get(n); tg = _tree(gt) if gt else None
        w = W.get(n)
        out_h = None
        if gt and w and len(w) >= 30:
            out_h = honest_d(w, _tree(w), gt, tg)[0]
        rows = []
        for k, cs in enumerate(CS):
            u = (cs["xs_q"] - xl) / wd
            if not (-0.1 <= u[1] <= 1.1):
                continue
            span = max(1, cc["n_rows"])
            ov = max(0, min(cs["y1"], cc["top_y"] + span - 1) - max(cs["y0"], cc["top_y"]) + 1) / span
            lab = None
            if gt:
                h, cov, m = honest_d(cands[k][2], TC[k], gt, tg)
                lab = (h, cov, m)
            rows.append(dict(k=k, u=u.tolist(), cover=ov, here=used.get(k) == n, elsewhere=(k in used and used[k] != n),
                             lab=lab))
        slots.append(dict(name=n, root=re.sub(r"^BKZ_", "", M.mnem_root(n)), top_y=cc["top_y"], n_rows=cc["n_rows"],
                          band=(xl, xr), has_truth=gt is not None, out_honest=out_h, cands=rows))
    rec = dict(sheet=sh, cands=CS, slots=slots)
    tmp = outp.with_suffix(".tmp"); pickle.dump(rec, open(tmp, "wb"), protocol=4); os.replace(tmp, outp)
    return sh, len(slots)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
    ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_lvl/NS")
    ap.add_argument("--out", default=r"F:/nds/output/taskS/line_choice")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import multiprocessing as mp
    TS = Path(a.ts); Path(a.out).mkdir(parents=True, exist_ok=True)
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for sh in lst("wellmap_sheets.txt") + lst("holdoutA_sheets.txt"):
        q = SRC.get(sh)
        if not q:
            continue
        key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        tc = Path(a.cache) / (key + ".pkl"); pd = Path(a.dir) / key
        got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
        if not (tc.exists() and got):
            continue
        cpk = pickle.load(open(tc, "rb"))
        jobs.append((sh, str(tc), str(got), str(q), cpk["image"], dict(out=a.out)))
    print(f"листов {len(jobs)}")
    t0 = time.time(); done = 0
    with mp.Pool(a.workers) as pool:
        for sh, n in pool.imap_unordered(one, jobs):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    print("скан готов")


if __name__ == "__main__":
    main()
