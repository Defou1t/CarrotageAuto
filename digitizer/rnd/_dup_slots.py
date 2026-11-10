r"""_dup_slots.py — ДУБЛИ ТРАСС В ВЫДАЧЕ (10.11, §6.358): слот ведёт ту же линию, что другой наш слот того же трека.

Разбор выборки 30 листов NSE22 глазами: 14 из 23 ошибок — наша трасса строка в строку (≤ 3 px) с другим нашим слотом трека на сотнях –
тысячах строк. Здесь — по всем листам: для каждой нашей кривой доля её строк, где другой наш слот того же трека ближе TOL px (`dup`),
партнёр с наибольшей долей, причина ошибки кривой и партнёра из разложения `_sample_review.py --pop` (`pop_curves.tsv`).

  _dup_slots.py [--mode NSE22] [--pop F:/nds/output/sample_2026-11-10/pop_curves.tsv] [--workers 2]
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

TS = Path(r"F:/nds/output/taskS")
TOL = 3.0


def arr(d):
    if not d:
        return None
    ys = np.fromiter(d.keys(), np.int64, len(d)); xs = np.fromiter(d.values(), np.float64, len(d))
    o = np.argsort(ys)
    return ys[o], xs[o]


def one(args):
    sheet, got, tmap = args
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import meta as M
    try:
        mw = extract(str(got))
    except Exception as e:
        return sheet, None, f"{type(e).__name__}"
    C = {c["name"]: arr(dense(c)) for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
    C = {k: v for k, v in C.items() if v is not None and len(v[0]) >= 50}
    by = defaultdict(list)
    for n in C:
        if n in tmap:
            by[tmap[n]].append(n)
    out = {}
    for tr, names in by.items():
        for a in names:
            ya, xa = C[a]; best = (0.0, "", 0)
            for b in names:
                if b == a:
                    continue
                yb, xb = C[b]
                cm, ia, ib = np.intersect1d(ya, yb, assume_unique=True, return_indices=True)
                if not len(cm):
                    continue
                k = int(np.sum(np.abs(xa[ia] - xb[ib]) <= TOL))
                if k / len(ya) > best[0]:
                    best = (k / len(ya), b, k)
            out[a] = {"track": tr, "n": int(len(ya)), "dup": round(best[0], 4), "partner": best[1], "rows": best[2], "k": len(names)}
    return sheet, out, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="NSE22")
    ap.add_argument("--pop", default=r"F:/nds/output/sample_2026-11-10/pop_curves.tsv")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import _sample_review as S
    S.OOF = TS / "rp_lvl" / a.mode
    TRACK = S.load_tracks()
    CAT, PART = {}, {}
    for ln in Path(a.pop).read_text(encoding="utf-8").splitlines()[1:]:
        f = ln.split("\t")
        CAT[(f[1], f[2])] = f[4]; PART[f[1]] = f[0]
    jobs = []
    for sh in sorted(PART):
        got = next(iter(sorted((S.OOF / S.sheet_dir(sh)).glob("*_auto.nlgx"))), None)
        if got is not None:
            jobs.append((sh, got, TRACK.get(sh, {})))
    print(f"★ листов {len(jobs)} (выдача {a.mode}), допуск {TOL} px")
    RES, t0 = {}, time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for k, (sh, o, err) in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            if o is None:
                print(f"  ⚠ {sh}: {err}"); continue
            RES[sh] = o
            if k % 400 == 0:
                print(f"  {k}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    outp = Path(a.out or (TS / f"_dup_slots_{a.mode}.json")); tmp = outp.with_name(outp.name + ".tmp")
    tmp.write_text(json.dumps(RES, ensure_ascii=False), encoding="utf-8"); tmp.replace(outp)
    for part in ("поле+вне", "сорт A"):
        rows = [(sh, n, r) for sh, o in RES.items() for n, r in o.items()
                if (PART.get(sh) != "сорт A") == (part == "поле+вне") and (sh, n) in CAT]
        print(f"\n## {part}: наших кривых с эталоном того же имени {len(rows)}; трек с ≥ 2 слотами {sum(1 for _, _, r in rows if r['k'] >= 2)}")
        for th in (0.1, 0.3, 0.5):
            D = [(sh, n, r) for sh, n, r in rows if r["dup"] >= th]
            c = Counter(CAT[(sh, n)] for sh, n, _ in D)
            bad = sum(v for k, v in c.items() if k != "верно")
            pc = Counter((CAT[(sh, n)] == "верно", CAT.get((sh, r["partner"])) == "верно") for sh, n, r in D)
            print(f"  дубль ≥ {th:.0%} строк: кривых {len(D)}, из них неверных {bad} ({bad / max(len(D), 1):.0%}); пары (сама верна, партнёр "
                  f"верен): да/да {pc[(True, True)]}, да/нет {pc[(True, False)]}, нет/да {pc[(False, True)]}, нет/нет {pc[(False, False)]}")
            print("    причины: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
        tot = Counter(CAT[(sh, n)] for sh, n, _ in rows)
        D3 = Counter(CAT[(sh, n)] for sh, n, r in rows if r["dup"] >= 0.3)
        print("  доля дублей (≥ 30%) по причине: " + ", ".join(f"{k} {D3[k]}/{tot[k]} ({D3[k] / max(tot[k], 1):.0%})" for k, _ in tot.most_common()))


if __name__ == "__main__":
    main()
