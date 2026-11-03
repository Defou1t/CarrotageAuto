r"""_repair_ceiling.py — ПОТОЛОК ПОЧИНКИ УХОДОВ ЧУЖИМИ КАНДИДАТАМИ (03.11, §6.353).

У «верных» мерой приёмки кривых 70% лишнего — длинные участки на чужой линии. Мост / пропуск на месте ухода приёмку не поднимают —
нужна настоящая линия. Здесь (оракул по эталону): уход выдачи (серия строк > 10 px от эталона при совпавшем уровне, ≥ 5 строк) —
есть ли среди кандидатов кэша (прод `traces`, декодер `alt`, доп. пути NSE19) такой, что на ≥ 80% строк серии лежит на эталоне
(≤ 3 px)? Сколько лишних строк так чинится и сколько кривых стало бы чистыми.

  _repair_ceiling.py --sheets F:/nds/output/taskS/segclean_sheets.txt --workers 2
"""
import sys, argparse, pickle, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np

TS = Path(r"F:/nds/output/taskS")
XF = [("tcache_pk3", "alt"), ("tcache_inert", "extra"), ("tcache_seg", "alt"), ("tcache_seg", "extra"), ("tcache_segraw_f", "alt"),
      ("tcache_segraw_f", "extra")]
XO = [("tcache_out", "extra"), ("tcache_out_inert", "extra"), ("tcache_out_seg", "alt"), ("tcache_out_seg", "extra"),
      ("tcache_out_segraw", "alt"), ("tcache_out_segraw", "extra")]


def cands(dirname, field):
    base = TS / ("tcache" if field else "tcache_out") / (dirname + ".pkl")
    if not base.exists():
        return None
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    c = pickle.load(open(base, "rb"))
    out = [un(t) for _, t in c["traces"]] + [un(t) for _, t in (c.get("alt") or [])]
    for d, fld in (XF if field else XO):
        xp = TS / d / (dirname + ".pkl")
        if xp.exists():
            out += [un(t) for _, t in (pickle.load(open(xp, "rb")).get(fld) or [])]
    return out


def one(args):
    nm, part, src, got, field = args
    import _sample_review as S
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception as e:
        return nm, part, None
    C = cands(S.sheet_dir(nm), field)
    if C is None:
        return nm, part, None
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    R = Counter(); R["кандидатов"] = len(C)
    for g, gt in gts.items():
        if g not in wr:
            continue
        w = wr[g]; lw, lg = LW.get(g, {}), LG.get(g, {})
        rows = [y for y in sorted(w) if y in gt and lw.get(y, 0) == lg.get(y, 0)]
        off = [y for y in rows if abs(w[y] - gt[y]) > 10]
        R["строк"] += len(rows); R["лишних"] += len(off)
        runs, st = [], None
        for i, y in enumerate(off):
            if st is None:
                st = y
            elif y - off[i - 1] > 3:
                runs.append((st, off[i - 1])); st = y
        if st is not None:
            runs.append((st, off[-1]))
        fixed = 0
        for a, b in runs:
            rr = [y for y in range(a, b + 1) if y in gt and y in w]
            if len(rr) < 5:
                continue
            best = 0.0
            for t in C:
                h = sum(1 for y in rr if y in t and abs(t[y] - gt[y]) <= 3)
                best = max(best, h / len(rr))
                if best >= 0.8:
                    break
            if best >= 0.8:
                fixed += len([y for y in rr if abs(w[y] - gt[y]) > 10])
        R["чинится"] += fixed
        if rows:
            R["кривых"] += 1
            R["чистых до"] += len(off) <= 0.05 * len(rows)
            R["чистых после"] += (len(off) - fixed) <= 0.05 * len(rows)
    return nm, part, dict(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import _sample_review as S
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    part = {nm: p for p, SS in SETS.items() for nm in SS}
    fld = set(l.strip() for l in (TS / "tcache_sheets.txt").read_text(encoding="utf-8").splitlines())
    names = [l.strip() for l in Path(a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []
    for nm in names:
        got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
        if got is not None and nm in SRC and nm in part:
            jobs.append((nm, part[nm], SRC[nm], got, nm in fld))
    print(f"★ листов {len(jobs)}")
    from multiprocessing import Pool
    T = {"поле+вне": Counter(), "сорт A": Counter()}; t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (nm, p, r) in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            if r:
                T["сорт A" if p == "сорт A" else "поле+вне"].update(r)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} {time.time() - t0:.0f} с")
    for k, c in T.items():
        print(f"★ {k}: кривых {c['кривых']}, лишних строк (уровень совпал) {c['лишних']} из {c['строк']}; чинится кандидатом кэша "
              f"{c['чинится']} ({c['чинится'] / max(1, c['лишних']):.0%}); чистых (≤ 5%) {c['чистых до']} → {c['чистых после']} "
              f"(кандидатов на лист в среднем {c['кандидатов'] / max(1, len(jobs)):.0f})")


if __name__ == "__main__":
    main()
