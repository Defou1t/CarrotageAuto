r"""_cut_sit.py — СИДЕНИЕ НА КРАЮ ШКАЛЫ У ОБРУБА (08.11, §6.354 «треть потерь уровня — ведение у обруба»): наша трасса на уровне 1×
идёт вдоль края полосы шкалы (перо 1× за краем), а эксперт уже на пере 5×.

Для кривых с цепочкой ≥ 2 уровней (выдача NSE19 вне фолда): строки нашей трассы на уровне 0 в ≤ 4 px от края полосы уровня 0, серии
≥ 10 строк (разрыв ≤ 2) → по эталону: эксперт там же на 1× (честный упор), эксперт на 5× и выше (сидение у обруба), эксперт на 1× в
другом месте (рамка / край вместо кривой), эталона нет. Оракул: строки «сидения у обруба» → x и уровень эталона — сколько кривых стало бы
верными приёмкой (и чистыми ≤ 5%) — потолок правки «переход на близнеца у обруба».

  _cut_sit.py [--limit 400] [--workers 1]
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np

NEAR, MINRUN = 4.0, 10


def runs(ys):
    out, cur = [], []
    for y in ys:
        if cur and y - cur[-1] > 3:
            out.append(cur); cur = []
        cur.append(y)
    if cur:
        out.append(cur)
    return [r for r in out if len(r) >= MINRUN]


def one(args):
    nm, part, src, got = args
    import _sample_review as S, _strict_stand as T, decode_levels as DL
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception as e:
        return nm, part, [], f"{type(e).__name__}"
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    rows = []
    for c in mt["curves"]:
        g = c["name"]
        if g not in gts or g not in wr:
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2:
            continue
        w, gt, lw, lg = wr[g], gts[g], LW.get(g, {}), LG.get(g, {})
        edges = [ch[0]["x_left"], ch[0]["x_right"]]
        sit = [y for y in sorted(w) if lw.get(y, 0) == 0 and min(abs(w[y] - e) for e in edges) <= NEAR]
        cnt = Counter(); fix = set()
        for r in runs(sit):
            for y in r:
                if y not in gt:
                    cnt["нет эталона"] += 1
                elif lg.get(y, 0) >= 1:
                    cnt["эксперт на 5× и выше"] += 1; fix.add(y)
                elif abs(gt[y] - w[y]) <= 10:
                    cnt["эксперт тоже у края (1×)"] += 1
                else:
                    cnt["эксперт на 1× в другом месте"] += 1
        S._TREES.clear()
        def meas(w_, lw_):
            S._TREES.clear()
            wl = S.relevel(w_, lw_, lg, ch)
            hon = S.HON(*S.err(wl, gt))
            st = T.strict(S, wl, gt, frozenset(y for y in w_ if lw_.get(y, 0) != lg.get(y, 0)))
            c5 = bool(hon and st and st["off_o"] <= 0.05 * st["n_o"] and st["off_t"] <= 0.05 * st["n_t"])
            return bool(hon), c5, (st or {}).get("off_o", 0), (st or {}).get("lvl", 0)
        b = meas(w, lw)
        o = b
        if fix:
            w2, lw2 = dict(w), dict(lw)
            for y in fix:
                w2[y] = gt[y]; lw2[y] = lg.get(y, 0)
            o = meas(w2, lw2)
        rows.append((g, len(w), dict(cnt), b, o))
    return nm, part, rows, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mode", default="NSE19")
    a = ap.parse_args()
    import _sample_review as S
    S.OOF = S.TS / "rp_lvl" / a.mode
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = []
    for part, SS in SETS.items():
        if part == "сорт A":
            continue
        for nm in sorted(SS):
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is not None and nm in SRC:
                jobs.append((nm, part, SRC[nm], got))
    rng = np.random.default_rng(11)
    if a.limit:
        jobs = [jobs[i] for i in sorted(rng.permutation(len(jobs))[:a.limit])]
    print(f"★ листов {len(jobs)} (поле + вне, {a.mode}); край — ≤ {NEAR:.0f} px от края полосы уровня 0, серии ≥ {MINRUN} строк")
    from multiprocessing import Pool
    res = []; t0 = time.time()
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            res.append(r)
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    out = S.TS / f"_cut_sit_{a.mode}.json"
    tmp = out.with_name(out.name + ".tmp"); tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8"); tmp.replace(out)
    R = [x for _, _, rows, _ in res for x in rows]
    C = Counter()
    for x in R:
        C.update(x[2])
    tot_lvl = sum(x[3][3] for x in R); tot_off = sum(x[3][2] for x in R)
    print(f"★ кривых с цепочкой {len(R)}; строк выдачи {sum(x[1] for x in R)}; лишних строк {tot_off}, из них уровня {tot_lvl}")
    print(f"★ строк «у края 1×» в сериях ≥ {MINRUN}: {dict(C)}")
    hb, ho = sum(x[3][0] for x in R), sum(x[4][0] for x in R); cb, co = sum(x[3][1] for x in R), sum(x[4][1] for x in R)
    print(f"★ оракул (сидение у обруба → x и уровень эталона): верных приёмкой {hb} → {ho} ({ho - hb:+d}), чистых ≤ 5% {cb} → {co} ({co - cb:+d}); "
          f"лишних строк {tot_off} → {sum(x[4][2] for x in R)}")
    print(f"★ {out}")


if __name__ == "__main__":
    main()
