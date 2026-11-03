r"""_repair_stand.py — ПОЧИНКА УХОДОВ КУСКАМИ КАНДИДАТОВ КЭША (03.11, §6.353).

Потолок (`_repair_ceiling.py`, оракул): 29% лишних строк «верных» кривых чинятся куском кандидата кэша; чистых (≤ 5%) 222 → 428 из
1223 (поле+вне), сорт A 41 → 94. Здесь — без эталона: участок трассы выдачи (между прыжками > J px) — кандидат на починку, если и вход,
и выход — прыжки БЕЗ туши вдоль перемычки (< tau; настоящий пик прочерчен пером). Замена — кусок кандидата, который совпадает с трассой
выдачи (≤ 3 px) на k строках до участка и после него, покрывает участок и проходит его без прыжков без туши; из таких — с наименьшим
разрывом на стыках. Мера приёмки и строгая (≤ 1% / ≤ 5% лишних) до и после.

  _repair_stand.py --sheets F:/nds/output/taskS/segclean_sheets.txt --workers 2 [--tau 0.5] [--k 6]
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np

J = 8


def repair(w, C, gray, paper, tau, k, lw, others=()):
    """→ (новая трасса, число починенных участков)"""
    from _strict_stand import ink_frac
    from _seg_clean import split
    ys, segs = split(w)
    out = dict(w); nfix = 0
    for i0, i1 in segs:
        if i0 == 0 or i1 + 1 >= len(ys):
            continue
        a, b = ys[i0], ys[i1]
        pa, nb = ys[i0 - 1], ys[i1 + 1]
        if a - pa > 2 or nb - b > 2 or lw.get(pa, 0) != lw.get(a, 0) or lw.get(b, 0) != lw.get(nb, 0):
            continue
        if abs(w[a] - w[pa]) <= J or abs(w[nb] - w[b]) <= J:
            continue
        if ink_frac(gray, paper, pa, w[pa], w[a]) >= tau or ink_frac(gray, paper, b, w[b], w[nb]) >= tau:
            continue
        before = [y for y in range(a - k, a) if y in w]; after = [y for y in range(b + 1, b + 1 + k) if y in w]
        if len(before) < k // 2 or len(after) < k // 2:
            continue
        best, bcost = None, None
        for t in C:
            if sum(1 for y in before if y in t and abs(t[y] - w[y]) <= 3) < 0.6 * len(before):
                continue
            if sum(1 for y in after if y in t and abs(t[y] - w[y]) <= 3) < 0.6 * len(after):
                continue
            rr = [y for y in range(a, b + 1)]
            if sum(1 for y in rr if y in t) < 0.9 * len(rr):
                continue
            bad = False
            prev = pa
            for y in rr + [nb]:
                if y in t and prev in t and abs(t[y] - t[prev]) > J and ink_frac(gray, paper, prev, t[prev], t[y]) < tau:
                    bad = True; break
                if y in t:
                    prev = y
            if bad:
                continue
            if any(sum(1 for y in rr[::4] if y in t and y in o and abs(t[y] - o[y]) <= 3) > 0.5 * len(rr[::4]) for o in others):
                continue                                     # кусок — чужая кривая выдачи
            cost = abs(t.get(a, t.get(a + 1, 1e9)) - w[pa]) + abs(t.get(b, t.get(b - 1, 1e9)) - w[nb])
            if bcost is None or cost < bcost:
                best, bcost = t, cost
        if best is not None:
            for y in range(a, b + 1):
                if y in best:
                    out[y] = best[y]
            nfix += 1
    return out, nfix


def one(args):
    nm, part, src, got, field, tau, k = args
    import _sample_review as S, _strict_stand as T, _final_exif, decode_levels as DL
    from _repair_ceiling import cands
    from dataset_build import find_image
    try:
        mt, mw, gts, wr = S.curves(src, got)
        C = cands(S.sheet_dir(nm), field)
        img = find_image(Path(src))
        if C is None or img is None:
            return nm, part, None
        rgb = _final_exif.load_rgb(str(img))
    except Exception as e:
        return nm, part, None
    gray = rgb.min(axis=2) if rgb.ndim == 3 else rgb
    del rgb
    paper = float(np.percentile(gray[::7, ::7], 80))
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    R = Counter()
    for g, gt in gts.items():
        if g not in wr:
            continue
        w = wr[g]
        w2, nf = repair(w, C, gray, paper, tau, k, LW.get(g, {}), [o for kk, o in wr.items() if kk != g])
        R["починок"] += nf
        for tag, ww in (("до", w), ("после", w2)):
            S._TREES.clear()
            lw, lg = LW.get(g, {}), LG.get(g, {})
            wl = S.relevel(ww, lw, lg, CH.get(g, []))
            hon = S.HON(*S.err(wl, gt))
            st = T.strict(S, wl, gt, frozenset(y for y in ww if lw.get(y, 0) != lg.get(y, 0)))
            R[f"приёмка {tag}"] += hon
            if st:
                R[f"лишних {tag}"] += st["off_o"]; R[f"строк {tag}"] += st["n_o"]
                R[f"чистых1 {tag}"] += hon and st["off_o"] <= 0.01 * st["n_o"] and st["off_t"] <= 0.01 * st["n_t"]
                R[f"чистых5 {tag}"] += hon and st["off_o"] <= 0.05 * st["n_o"] and st["off_t"] <= 0.05 * st["n_t"]
        R["кривых"] += 1
    return nm, part, dict(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    import _sample_review as S
    TS = Path(r"F:/nds/output/taskS")
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    part = {nm: p for p, SS in SETS.items() for nm in SS}
    fld = set(l.strip() for l in (TS / "tcache_sheets.txt").read_text(encoding="utf-8").splitlines())
    names = [l.strip() for l in Path(a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for nm in names:
        got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is not None and nm in SRC and nm in part:
            jobs.append((nm, part[nm], SRC[nm], got, nm in fld, a.tau, a.k))
    print(f"★ листов {len(jobs)}, tau {a.tau}, k {a.k}")
    from multiprocessing import Pool
    T = {"поле+вне": Counter(), "сорт A": Counter()}; PS = {"поле+вне": [], "сорт A": []}; t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (nm, p, r) in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            if r:
                kk = "сорт A" if p == "сорт A" else "поле+вне"
                T[kk].update(r); PS[kk].append((r.get("приёмка после", 0) - r.get("приёмка до", 0), r.get("чистых5 после", 0) - r.get("чистых5 до", 0)))
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} {time.time() - t0:.0f} с")
    for kk, c in T.items():
        d = np.array(PS[kk]) if PS[kk] else np.zeros((0, 2))
        def sgn(col):
            x = d[:, col] if len(d) else np.zeros(0); up, dn = int((x > 0).sum()), int((x < 0).sum())
            from math import comb
            n = up + dn; p_ = sum(comb(n, i) for i in range(0, min(up, dn) + 1)) * 2 / 2 ** n if n else 1.0
            return f"↑{up}/↓{dn}, p = {min(1.0, p_):.4f}"
        print(f"★ {kk}: кривых {c['кривых']}, починок {c['починок']}; приёмка {c['приёмка до']} → {c['приёмка после']} ({sgn(0)}); "
              f"чистых ≤ 1% {c['чистых1 до']} → {c['чистых1 после']}; ≤ 5% {c['чистых5 до']} → {c['чистых5 после']} ({sgn(1)}); "
              f"лишних строк {c['лишних до'] / max(1, c['строк до']):.2%} → {c['лишних после'] / max(1, c['строк после']):.2%}")


if __name__ == "__main__":
    main()
