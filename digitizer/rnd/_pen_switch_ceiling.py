r"""_pen_switch_ceiling.py — ПОТОЛОК «СКЛЕЙКИ ПО ПЕРУ» ДЛЯ КРИВЫХ СО СМЕНОЙ МАСШТАБА (04.10, к §6.288).

Глазами (`visual_2026-10-03/C_line_choice/gzlev`): у ГЗ без честного кандидата лучший кандидат идёт по перу 1× и при зашкале
доходит до правого края трека (крючок), а эксперт в эти строки берёт перо 5× (левее: «большие значения — в масштабе левее»).
Здесь для слотов без честного кандидата (поле + сорт A, скан `_line_choice_scan.py`) строится склейка двух кандидатов:
C(y) = A(y), пока A не у своего упора (x < 99.5-й процентиль x трассы A − запас, px), иначе B(y) (перо 5×, левее A). A — 3 кандидата с наименьшей
медианой расстояния до эталона, B — кандидаты левее A по медиане x. Мера — на плоскости в обе стороны по пикселям, покрытие ≥ 0.9.
Для сравнения — слоты, где у эталона есть уровни > 0 (переходы масштаба), и прочие.

  _pen_switch_ceiling.py --margin 12 --workers 4
"""
import sys, argparse, pickle, hashlib, time, re
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def one(job):
    rec_path, tc_path, src, margin = job
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    from auto import meta as M
    rec = pickle.load(open(rec_path, "rb"))
    todo = [s for s in rec["slots"] if s["has_truth"] and s["cands"] and not any(r["lab"] and r["lab"][0] for r in s["cands"])]
    if not todo:
        return []
    c = pickle.load(open(tc_path, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    raw = [dense_tr(un(t)) for L, t in (c["traces"] or [])] + [dense_tr(un(t)) for L, t in (c["alt"] or [])]
    raw = [r for r in raw if len(r) >= 50]
    mt = extract(src)
    fc = {cc["name"]: cc for cc in mt["curves"] if M.mnem_root(cc["name"]) != "DA"}
    out = []
    for s in todo:
        cc = fc[s["name"]]
        segs = cc.get("segments") or []
        levs = {sg[2] if isinstance(sg, (list, tuple)) else sg.get("level", 0) for sg in segs}
        multi = len(levs) > 1 or (bool(levs) and max(levs) > 0)
        gt = dense(cc); tg = _tree(gt)
        xl, xr = s["band"]
        labs = sorted([(r["lab"][2], r["k"]) for r in s["cands"] if r["lab"]])
        As = [k for m, k in labs[:3]]
        med = {r["k"]: float(np.median(list(raw[r["k"]].values()))) for r in s["cands"]}
        best = (False, 99.0)
        for ka in As:
            A = raw[ka]
            # край пера 1× — по самой трассе: шкала рамки бывает шире физического хода пера (крючок у упора раньше x_right)
            edge = float(np.percentile(np.fromiter(A.values(), float), 99.5)) - margin
            off = [y for y, x in A.items() if x >= edge]
            if len(off) < 20:
                continue
            for kb in [r["k"] for r in s["cands"] if r["k"] != ka and med[r["k"]] < med[ka]]:
                B = raw[kb]
                C = dict(A)
                hit = 0
                for y in off:
                    if y in B:
                        C[y] = B[y]; hit += 1
                if hit < 10:
                    continue
                h = honest_d(C, _tree(C), gt, tg)
                if h[0] or h[2] < best[1]:
                    best = (bool(h[0]), float(h[2]))
                if h[0]:
                    break
            if best[0]:
                break
        out.append(dict(sheet=rec["sheet"], slot=s["name"], root=re.sub(r"^BKZ_", "", s["root"]), multi=multi,
                        honest=best[0], m=best[1], m0=labs[0][0] if labs else 99.0))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
    ap.add_argument("--scan", default=r"F:/nds/output/taskS/line_choice")
    ap.add_argument("--margin", type=float, default=12.0, help="запас до упора пера 1×, px")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/pen_switch_ceiling.pkl")
    a = ap.parse_args()
    import multiprocessing as mp
    from collections import Counter
    TS = Path(a.ts)
    lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for f in sorted(Path(a.scan).glob("*.pkl")):
        sh = pickle.load(open(f, "rb"))["sheet"]
        q = SRC[sh]
        key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        jobs.append((str(f), str(Path(a.cache) / (key + ".pkl")), str(q), a.margin))
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
        for mv, nm in ((True, "с переходами масштаба"), (False, "без переходов")):
            g = [r for r in rs if r["multi"] == mv]
            fam = Counter(r["root"] for r in g if r["honest"])
            print(f"★ {sn}, слоты без честного кандидата {nm}: {len(g)}; склейка по перу честна у {sum(r['honest'] for r in g)} "
                  f"({', '.join(f'{k} {v}' for k, v in fam.most_common(6))})")


if __name__ == "__main__":
    main()
