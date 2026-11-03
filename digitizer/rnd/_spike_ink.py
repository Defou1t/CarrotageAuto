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
SEG = ""


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


def _init(seg):
    global SEG
    SEG = seg


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
    seg = None
    if SEG:                                             # сеть туши фолда скважины листа (вне обучения) — окошком вокруг укола
        import torch
        torch.set_num_threads(4)
        from auto import ink_seg as IS
        FO = json.loads((Path(SEG) / "folds.json").read_text(encoding="utf-8")); k_ = FO.get(Path(src).parent.parent.name)
        net = str(Path(SEG) / (f"f{k_}.pt" if k_ is not None else "all.pt"))

        def seg(ys, xs, rgb=rgb, net=net):
            y0, y1 = min(ys) - 2, max(ys) + 3; x0, x1 = int(min(xs)) - 4, int(max(xs)) + 5
            pm = IS.prob_map(rgb, y0, y1, x0, x1, net)
            yy = np.clip(np.asarray(ys) - max(0, y0), 0, pm.shape[0] - 1); xx = np.round(np.asarray(xs)).astype(int) - max(0, x0)
            return np.max(np.stack([pm[yy, np.clip(xx + o, 0, pm.shape[1] - 1)] for o in (-1, 0, 1)]), 0)
    else:
        del rgb
    paper = float(np.percentile(gray[::7, ::7], 80))
    import decode_levels as DL
    from extract_nlgx import depth_of
    ys_all = sorted({y for t in gts.values() for y in t}); near = lambda y: np.nan
    if len(ys_all) >= 100:                               # строк до ближайшей глубины, кратной 10 м (как `_depth_digits.py`)
        yy = np.arange(ys_all[0], ys_all[-1] + 1, 50); dd = np.array([depth_of(mt, int(y)) for y in yy], float); ok = np.isfinite(dd)
        if ok.sum() >= 3 and np.ptp(dd[ok]) > 0:
            k_, b_ = np.polyfit(yy[ok], dd[ok], 1)
            near = lambda y, k_=k_, b_=b_: float(abs(((k_ * y + b_) - np.round((k_ * y + b_) / 10) * 10) / k_))
    rows = []
    for g, gt in gts.items():
        if g not in wr or len(wr[g]) < 50:
            continue
        fam = DL.build_family(mt, next(c for c in mt["curves"] if c["name"] == g))
        xl = min(min(s_["x_left"], s_["x_right"]) for s_ in fam) if fam else np.nan
        xr = max(max(s_["x_left"], s_["x_right"]) for s_ in fam) if fam else np.nan
        for y0, y1, amp, ki, ko, ch in spike_rows(wr[g], gray, paper):
            yy = [y for y in range(y0, y1 + 1) if y in gt]
            if not yy:
                lab = "нет эталона"
            else:
                dw = [abs(wr[g][y] - gt[y]) for y in yy]; dc = [abs(ch(y) - gt[y]) for y in yy]
                lab = "настоящий" if np.median(dw) <= 10 else "ложный" if (min(dw) > 15 and max(dc) <= 10) else "неясно"
            xs_ = [wr[g][y] for y in range(y0, y1 + 1)]; xm = xs_[int(np.argmax([abs(x - ch(y)) for y, x in zip(range(y0, y1 + 1), xs_)]))]
            sp_ = sc_ = np.nan
            if seg is not None:
                yr = list(range(y0, y1 + 1))
                v = seg(yr + yr, list(xs_) + [ch(y) for y in yr]); sp_ = float(np.max(v[:len(yr)])); sc_ = float(np.mean(v[len(yr):]))
            rows.append((g, y0, y1, amp, ki, ko, lab, near((y0 + y1) / 2), float((xm - xl) / (xr - xl)) if xr > xl else np.nan, sp_, sc_))
    return nm, part, rows, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mode", default="NSE19")
    ap.add_argument("--seg", default="", help="каталог сетей туши вне фолда (folds.json, f{k}.pt) — тушь сети у укола и у хорды")
    a = ap.parse_args()
    global SEG
    SEG = a.seg
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
    with Pool(a.workers, initializer=_init, initargs=(a.seg,)) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            res.append(r)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    (S.TS / f"_spike_ink{'_seg' if a.seg else ''}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
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
    for lab in ("настоящий", "ложный", "неясно"):
        S_ = [x for x in R if x[6] == lab and np.isfinite(x[7])]
        if S_:
            nr = np.array([x[7] for x in S_]); u = np.array([x[8] for x in S_ if np.isfinite(x[8])])
            print(f"  {lab:9s}: у круглой (кратной 10 м) глубины ± 20 / 40 строк {np.mean(nr <= 20):.1%} / {np.mean(nr <= 40):.1%}; "
                  f"точка укола в полосе шкал (0 — левый край, 1 — правый) квартили {np.percentile(u, [10, 25, 50, 75, 90]).round(2)}, "
                  f"за краями {np.mean((u < 0) | (u > 1)):.0%}")
    Z = [x for x in R if x[6] in ("настоящий", "ложный") and len(x) > 9 and np.isfinite(x[9])]
    if Z:
        from sklearn.metrics import roc_auc_score
        yv = np.array([x[6] == "ложный" for x in Z]); sp = np.array([x[9] for x in Z]); sc = np.array([x[10] for x in Z])
        print(f"  сеть туши: AUC «ложный» по (хорда − укол) {roc_auc_score(yv, sc - sp):.3f}, по 1 − укол {roc_auc_score(yv, 1 - sp):.3f}; "
              f"укол у настоящих квартили {np.percentile(sp[~yv], [10, 25, 50, 75, 90]).round(2)}, у ложных {np.percentile(sp[yv], [10, 25, 50, 75, 90]).round(2)}")
        for th in (0.1, 0.2, 0.3):
            for tc in (0.5, 0.7):
                m = (sp < th) & (sc > tc)
                print(f"    укол < {th}, хорда > {tc}: {m.sum()} — ложных {yv[m].mean() if m.sum() else 0:.0%}, поймано ложных {m[yv].mean():.0%}, "
                      f"задето настоящих {m[~yv].mean():.1%}")
    for tau in (0.1, 0.2, 0.3, 0.5):
        for amin in (20, 40, 80):
            dry = [x for x in R if max(x[4], x[5]) < tau and x[3] > amin]
            c = Counter(x[6] for x in dry)
            tf = C["ложный"] or 1
            print(f"  сухой (тушь < {tau}) и амплитуда > {amin}: {len(dry)} — ложных {c['ложный']} ({c['ложный'] / max(len(dry), 1):.0%}), "
                  f"настоящих {c['настоящий']}, неясно {c['неясно']}, без эталона {c['нет эталона']}; поймано ложных {c['ложный'] / tf:.0%}")


if __name__ == "__main__":
    main()
