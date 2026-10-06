r"""_pen_pair.py — НА КАКОМ ПЕРЕ ТРАССА: 1× ИЛИ 5× — ПО ПАРТНЁРУ СРЕДИ КАНДИДАТОВ (06.10, к §6.294).

Ответ заказчика (06.10): «если есть одновременно 1× и 5×, то в приоритете 1× (точнее); но если 1× слишком часто прерывается, а 5×
идёт цельной линией, выбираем 5× (быстрее оцифровать)»; границу сегмента ставят «с запасом (желательно) или прямо по переходу».
⇒ оба пера бывают нарисованы одновременно, и уровень трассы = какое это перо. Тушь «близнеца» в одной точке строки этого не
различала (§6.289), а целые трассы-кандидаты листа — могут: перо 5× повторяет перо 1× в r раз «сжатым» к левому краю шкалы.

Для кривых выдачи (прод NSLX), честных по пикселям под своим именем и с цепочкой масштабов (сжатие): ищем среди кандидатов листа
(`tcache` traces + alt + пути второго порога `tcache_pk3`) партнёра:
- трасса — перо 1×, партнёр — перо 5×: x_p ≈ pix(шкала 1, val(шкала 0, x)) на общих строках;
- трасса — перо 5×, партнёр — перо 1×: x_p ≈ pix(шкала 0, val(шкала 1, x)) там, где это в полосе.
Доля строк с |x_p − прогноз| ≤ tol на окне ±W строк → по каждой строке «перо трассы»; уровень = 0 / 1. Сравнение с уровнями
эталона: совпадение по строкам и честность в значениях — против уровней выдачи.

  _pen_pair.py --workers 4
"""
import sys, argparse, pickle, hashlib, time, re
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def one(job):
    sh, src, got, tc, pk, P = job
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    from _level_loss import levels_by_row
    import decode_levels as DL
    from auto import meta as M
    val = lambda s, x: s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / ((s["x_right"] - s["x_left"]) or 1)
    pix = lambda s, v: s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / ((s["v_right"] - s["v_left"]) or 1)
    mt = extract(src); mw = {c["name"]: c for c in extract(got)["curves"]}
    c = pickle.load(open(tc, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    cands = [un(t) for L, t in (c["traces"] or [])] + [un(t) for L, t in (c["alt"] or [])]
    if Path(pk).exists():
        cands += [un(t) for L, t in (pickle.load(open(pk, "rb")).get("alt") or [])]
    cands = [dense_tr(x) for x in cands]
    cands = [x for x in cands if len(x) >= 200]
    out = []
    for g in mt["curves"]:
        n = g["name"]
        if M.mnem_root(n) == "DA" or n not in mw or sum(1 for x in g["xs"] if x != NULL) < 50:
            continue
        fam = DL.build_family(mt, g)
        if len(fam) < 2:
            continue
        r0 = abs(fam[0]["v_right"] - fam[0]["v_left"]); r1 = abs(fam[1]["v_right"] - fam[1]["v_left"])
        if not (r0 > 0 and r1 / r0 > 1.5):
            continue                                           # только цепочки со сжатием
        w = mw[n]
        if sum(1 for x in w["xs"] if x != NULL) < 200:
            continue
        gt, wt = dense(g), dense(w)
        if not honest_d(wt, _tree(wt), gt, _tree(gt))[0]:
            continue
        lg, lw = levels_by_row(g), levels_by_row(w)
        xl = min(min(s["x_left"], s["x_right"]) for s in fam); xr = max(max(s["x_left"], s["x_right"]) for s in fam)
        W = max(1.0, xr - xl); tol = P["tol"] * W
        ys = np.array(sorted(wt)); xs = np.array([wt[y] for y in ys])
        pred5 = np.array([pix(fam[1], val(fam[0], x)) for x in xs])     # где перо 5×, если трасса — перо 1×
        pred1 = np.array([pix(fam[0], val(fam[1], x)) for x in xs])     # где перо 1×, если трасса — перо 5×
        in1 = (pred1 >= xl) & (pred1 <= xr)
        hit5 = np.zeros(len(ys), bool); hit1 = np.zeros(len(ys), bool)
        for ct in cands:
            same = np.array([abs(ct.get(int(y), 1e9) - x) <= 1.5 for y, x in zip(ys[::25], xs[::25])])
            if same.mean() > 0.5:
                continue                                          # это сама трасса (или её дубль)
            cx = np.array([ct.get(int(y), np.nan) for y in ys])
            hit5 |= np.abs(cx - pred5) <= tol
            hit1 |= in1 & (np.abs(cx - pred1) <= tol)
        # окно ±win строк: доля строк, где виден партнёр-5× (трасса — 1×) и где виден партнёр-1× (трасса — 5×)
        win = P["win"]; k = np.ones(2 * win + 1)
        f5 = np.convolve(hit5.astype(float), k, "same") / np.convolve(np.ones(len(ys)), k, "same")
        f1 = np.convolve(hit1.astype(float), k, "same") / np.convolve(np.ones(len(ys)), k, "same")
        lev_out = np.array([lw.get(int(y), 0) for y in ys]); lev_g = np.array([lg.get(int(y), 0) for y in ys])
        # правило: если партнёр-1× виден чаще порога и чаще партнёра-5× — трасса на 5×; если партнёр-5× — на 1×; иначе как в выдаче
        newl = lev_out.copy()
        newl[(f1 >= P["thr"]) & (f1 > f5)] = 1
        newl[(f5 >= P["thr"]) & (f5 >= f1)] = 0
        m = np.isin(ys, list(gt.keys()))
        out.append(dict(sheet=sh, name=n, fam=re.sub(r"^BKZ_", "", M.mnem_root(n)), n=int(m.sum()),
                        agree_out=float(np.mean(lev_out[m] == lev_g[m])) if m.any() else np.nan,
                        agree_new=float(np.mean(newl[m] == lev_g[m])) if m.any() else np.nan,
                        cov5=float(np.mean(f5 >= P["thr"])), cov1=float(np.mean(f1 >= P["thr"])),
                        g5=float(np.mean(lev_g[m] >= 1)) if m.any() else np.nan,
                        hit_on_g1=float(np.mean((f1 >= P["thr"])[m & (lev_g >= 1)])) if (m & (lev_g >= 1)).any() else np.nan,
                        hit_on_g0=float(np.mean((f5 >= P["thr"])[m & (lev_g == 0)])) if (m & (lev_g == 0)).any() else np.nan))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_lvl/NSLX")
    ap.add_argument("--tol", type=float, default=0.02, help="допуск партнёра — доля ширины полосы")
    ap.add_argument("--win", type=int, default=100, help="окно ±строк для доли партнёра")
    ap.add_argument("--thr", type=float, default=0.3, help="доля строк окна с партнёром, чтобы решить «на каком пере»")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/pen_pair.pkl")
    a = ap.parse_args()
    import multiprocessing as mp
    TS = Path(a.ts)
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    FIELD, HOLD = set(lst("wellmap_sheets.txt")), set(lst("holdoutA_sheets.txt"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    P = dict(tol=a.tol, win=a.win, thr=a.thr)
    jobs = []
    for sh in sorted(FIELD | HOLD):
        q = SRC.get(sh)
        if not q:
            continue
        key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        d = Path(a.dir) / key
        got = next(d.glob("*_auto.nlgx"), None) if d.is_dir() else None
        if got:
            jobs.append((sh, str(q), str(got), str(TS / "tcache" / (key + ".pkl")), str(TS / "tcache_pk3" / (key + ".pkl")), P))
    if a.limit:
        jobs = jobs[:a.limit]
    R = []; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            R += r
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
        rs = [r for r in R if r["sheet"] in S]
        if not rs:
            continue
        ao = np.array([r["agree_out"] for r in rs]); an = np.array([r["agree_new"] for r in rs]); w_ = np.array([r["n"] for r in rs])
        print(f"\n★ {sn}: кривых со сжатием, честных по пикселям: {len(rs)}; строк эталона на 5× — {np.average([r['g5'] for r in rs], weights=w_):.0%}")
        print(f"   совпадение уровней по строкам: выдача {np.average(ao, weights=w_):.3f} → по партнёру {np.average(an, weights=w_):.3f}; "
              f"кривых лучше {int((an > ao + 0.02).sum())}, хуже {int((an < ao - 0.02).sum())}")
        h1 = np.array([r["hit_on_g1"] for r in rs if not np.isnan(r["hit_on_g1"])]); h0 = np.array([r["hit_on_g0"] for r in rs if not np.isnan(r["hit_on_g0"])])
        print(f"   партнёр-1× найден на строках эталона 5×: {np.mean(h1):.0%}; партнёр-5× найден на строках эталона 1×: {np.mean(h0):.0%}")


if __name__ == "__main__":
    main()
