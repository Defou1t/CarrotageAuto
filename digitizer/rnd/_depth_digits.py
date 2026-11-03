r"""_depth_digits.py — ЛИШНЕЕ НА ЦИФРАХ ГЛУБИНЫ? (03.11, §6.353, замечание заказчика «линия оцифровывает цифры глубины»).

Серии лишнего (строки выдачи > 10 px от линии эксперта, в значениях) у пар «наша — эталон того же имени»: где центр серии относительно
ближайшей круглой глубины (кратной 10 м и 5 м) в строках. Если серии жмутся к круглым глубинам сильнее, чем случайные строки той же
кривой, — цифры глубины внутри трека — заметный источник лишнего; доля лишних строк в окне ± R строк от круглой глубины — его размер.

  _depth_digits.py --workers 3 [--limit 600]
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np

R = 40


def one(args):
    nm, part, src, got = args
    import _sample_review as S
    import _strict_stand as SS
    import decode_levels as DL
    from extract_nlgx import depth_of
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception as e:
        return nm, part, None, f"{type(e).__name__}: {e}"
    ys_all = sorted({y for t in gts.values() for y in t})
    if len(ys_all) < 100:
        return nm, part, None, "мало строк"
    yy = np.arange(ys_all[0], ys_all[-1] + 1, 50)
    dd = np.array([depth_of(mt, int(y)) for y in yy], float)
    ok = np.isfinite(dd)
    if ok.sum() < 3 or np.ptp(dd[ok]) <= 0:
        return nm, part, None, "нет глубин"
    a, b = np.polyfit(yy[ok], dd[ok], 1)               # глубина = a·y + b; строк на метр 1/|a|
    rpm = 1.0 / abs(a)

    def near(ys, step):
        d = a * np.asarray(ys, float) + b
        return np.abs((d - np.round(d / step) * step) * rpm)  # строк до круглой глубины

    out = Counter()
    for c in mt["curves"]:
        g = c["name"]
        if g not in gts or g not in wr or not wr[g]:
            continue
        cw = next(x for x in mw["curves"] if x["name"] == g)
        lg, lw = S.lv_rows(c), S.lv_rows(cw)
        ch = DL.build_family(mt, c)
        S._TREES.clear()
        w = S.relevel(wr[g], lw, lg, ch); gt = gts[g]
        com = [y for y in sorted(w) if y in gt]
        if len(com) < 30:
            continue
        S._TREES.clear()
        do = S._plane(gt, com, w); off = do > 10.0
        ys = np.array(com)
        for step in (10, 5):
            nr = near(ys, step) <= R
            out[f"rows_{step}"] += len(ys); out[f"rows_near_{step}"] += int(nr.sum())
            out[f"off_{step}"] += int(off.sum()); out[f"off_near_{step}"] += int((off & nr).sum())
            for i0, i1 in SS.runs_of(off, ys):
                n = i1 - i0 + 1
                if n > 120:
                    continue
                cen = (ys[i0] + ys[i1]) / 2
                out[f"runs_{step}"] += 1; out[f"runrows_{step}"] += n
                if near([cen], step)[0] <= R:
                    out[f"runs_near_{step}"] += 1; out[f"runrows_near_{step}"] += n
    return nm, part, dict(out, rpm=rpm), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args()
    import _sample_review as S
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = []
    for part, SS_ in SETS.items():
        for nm in sorted(SS_):
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is not None and nm in SRC:
                jobs.append((nm, part, SRC[nm], got))
    rng = np.random.default_rng(5)
    if a.limit:
        jobs = [jobs[i] for i in sorted(rng.permutation(len(jobs))[:a.limit])]
    print(f"★ листов {len(jobs)}, окно ± {R} строк от круглой глубины")
    from multiprocessing import Pool
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=4):
            res.append(r)
    ok = [r for r in res if r[2]]
    print(f"  листов с глубинами {len(ok)}, без — {len(res) - len(ok)}: " + str(Counter(r[3] for r in res if not r[2]).most_common(3)))
    T = Counter()
    for r in ok:
        T.update({k: v for k, v in r[2].items() if k != "rpm"})
    print(f"  строк на метр: медиана {np.median([r[2]['rpm'] for r in ok]):.1f}")
    for step in (10, 5):
        base = T[f"rows_near_{step}"] / max(1, T[f"rows_{step}"])
        print(f"\n## кратные {step} м: строк выдачи у круглой глубины {base:.1%} (это доля «случайно»)")
        print(f"  лишних строк у круглой глубины {T[f'off_near_{step}'] / max(1, T[f'off_{step}']):.1%} (из {T[f'off_{step}']} лишних)")
        print(f"  коротких серий лишнего (≤ 120 строк) {T[f'runs_{step}']}, центр у круглой глубины {T[f'runs_near_{step}'] / max(1, T[f'runs_{step}']):.1%}; "
              f"их строк {T[f'runrows_near_{step}']} из {T[f'runrows_{step}']} ({T[f'runrows_near_{step}'] / max(1, T[f'runrows_{step}']):.1%}); "
              f"избыток над случайным ≈ {max(0, T[f'runrows_near_{step}'] - base * T[f'runrows_{step}']):.0f} строк "
              f"= {max(0, T[f'runrows_near_{step}'] - base * T[f'runrows_{step}']) / max(1, T[f'off_{step}']):.2%} всего лишнего")


if __name__ == "__main__":
    main()
