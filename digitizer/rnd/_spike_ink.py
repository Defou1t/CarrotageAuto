r"""_spike_ink.py — УКОЛЫ ТРАССЫ: НАСТОЯЩИЙ ОСТРЫЙ ПИК ИЛИ ПРЫЖОК НА ЦИФРУ / СЕТКУ (03.11, §6.355, разбор глазами D07).

Строчная мера почти не видит однострочных уколов на цифры глубины (вклад 0.1% лишних строк), но глазу они заметнее всего. Вычистка
уколов по хорде (`spike3` — §6.353: верных 3110 → 3105) ломает настоящие острые пики (БК, ИК). Здесь: укол = ≤ L строк отлёта
дальше D px от хорды соседей, которые близки друг к другу (≤ E px); тушь вдоль прыжка внутрь и наружу (`auto.detour._ink` — доля
тёмного вдоль строки между x соседних строк); метка эталона — «настоящий» (эталон у точек укола ≤ 10 px), «ложный» (эталон дальше
15 px от всех точек укола и ≤ 10 px от хорды). Сухой укол (туши вдоль обоих прыжков < τ) — кандидат на хорду.

  _spike_ink.py [--limit 400] [--workers 2]
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
import numpy as np

L, D, E = 3, 20.0, 15.0


def spikes(tr, L=L, D=D, E=E):
    """уколы как `_strict_stand.f_spike` → [(i, j)] — индексы строк укола в sorted(tr)"""
    ys = sorted(tr); xs = np.array([tr[y] for y in ys], float); n = len(ys); i = 1; out = []
    while i < n - 1:
        hit = None
        for j in range(i, min(n - 1, i + L)):
            if ys[j + 1] - ys[i - 1] != j - i + 2:
                break
            a, b = xs[i - 1], xs[j + 1]
            if abs(a - b) > E:
                continue
            t = (np.array(ys[i:j + 1]) - ys[i - 1]) / (ys[j + 1] - ys[i - 1])
            if np.all(np.abs(xs[i:j + 1] - (a + (b - a) * t)) > D):
                hit = j
        if hit is None:
            i += 1; continue
        out.append((i, hit)); i = hit + 1
    return ys, xs, out


def spike_rows(tr, gray, paper):
    """уколы трассы с тушью вдоль прыжков → [(y0, y1, амплитуда, тушь внутрь, тушь наружу, хорда-функция)]"""
    from auto.detour import _ink
    ys, xs, sp = spikes(tr)
    out = []
    for i, j in sp:
        a, b = xs[i - 1], xs[j + 1]
        ch = lambda y, a=a, b=b, y0=ys[i - 1], y1=ys[j + 1]: a + (b - a) * (y - y0) / (y1 - y0)
        amp = float(max(abs(xs[k] - ch(ys[k])) for k in range(i, j + 1)))
        ki = _ink(gray, paper, ys[i - 1], xs[i - 1], xs[i]); ko = _ink(gray, paper, ys[j], xs[j], xs[j + 1])
        out.append((ys[i], ys[j], amp, ki, ko, ch))
    return out


def one(args):
    nm, part, src, got = args
    import _sample_review as S, _final_exif
    from dataset_build import find_image
    try:
        mt, mw, gts, wr = S.curves(src, got)
        img = find_image(Path(src))
        if img is None:
            return nm, part, [], "нет скана"
        rgb = _final_exif.load_rgb(str(img))
    except Exception as e:
        return nm, part, [], f"{type(e).__name__}"
    gray = rgb.min(axis=2) if rgb.ndim == 3 else rgb
    del rgb
    paper = float(np.percentile(gray[::7, ::7], 80))
    rows = []
    for g, gt in gts.items():
        if g not in wr or len(wr[g]) < 50:
            continue
        for y0, y1, amp, ki, ko, ch in spike_rows(wr[g], gray, paper):
            yy = [y for y in range(y0, y1 + 1) if y in gt]
            if not yy:
                lab = "нет эталона"
            else:
                dw = [abs(wr[g][y] - gt[y]) for y in yy]; dc = [abs(ch(y) - gt[y]) for y in yy]
                lab = "настоящий" if np.median(dw) <= 10 else "ложный" if (min(dw) > 15 and max(dc) <= 10) else "неясно"
            rows.append((g, y0, y1, amp, ki, ko, lab))
    return nm, part, rows, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mode", default="NSE19")
    a = ap.parse_args()
    import _sample_review as S
    S.OOF = S.TS / "rp_lvl" / a.mode
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = []
    for part, SS in SETS.items():
        for nm in sorted(SS):
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is not None and nm in SRC:
                jobs.append((nm, part, SRC[nm], got))
    rng = np.random.default_rng(7)
    if a.limit:
        jobs = [jobs[i] for i in sorted(rng.permutation(len(jobs))[:a.limit])]
    print(f"★ листов {len(jobs)}; укол — ≤ {L} строк дальше {D:.0f} px от хорды соседей (соседи ≤ {E:.0f} px)")
    from multiprocessing import Pool
    res = []; t0 = time.time()
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            res.append(r)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    (S.TS / "_spike_ink.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    R = [x for _, _, rows, _ in res for x in rows]
    from collections import Counter
    C = Counter(x[6] for x in R)
    print(f"★ уколов {len(R)}: {dict(C)}; кривых с уколом {len({(nm, x[0]) for nm, _, rows, _ in res for x in rows})}")
    for lab in ("настоящий", "ложный"):
        S_ = [x for x in R if x[6] == lab]
        if S_:
            mk = np.array([max(x[4], x[5]) for x in S_]); am = np.array([x[3] for x in S_])
            print(f"  {lab:9s}: {len(S_)}; амплитуда медиана {np.median(am):.0f} px; max(тушь внутрь, наружу) квартили "
                  f"{np.percentile(mk, [10, 25, 50, 75, 90]).round(2)}")
    for tau in (0.1, 0.2, 0.3, 0.5):
        for amin in (20, 40, 80):
            dry = [x for x in R if max(x[4], x[5]) < tau and x[3] > amin]
            c = Counter(x[6] for x in dry)
            tf = C["ложный"] or 1
            print(f"  сухой (тушь < {tau}) и амплитуда > {amin}: {len(dry)} — ложных {c['ложный']} ({c['ложный'] / max(len(dry), 1):.0%}), "
                  f"настоящих {c['настоящий']}, неясно {c['неясно']}, без эталона {c['нет эталона']}; поймано ложных {c['ложный'] / tf:.0%}")


if __name__ == "__main__":
    main()
