r"""_track2.py — ВЕДЕНИЕ ЧЕРЕЗ ПЕРЕСЕЧЕНИЯ: ПУТИ ДЕКОДЕРА ИЗ СОХРАНЁННЫХ ПИКОВ КАРТЫ (13.10, §6.305).

Прод (`rowdec.trace_track._paths`): пути трека по очереди, Витерби по пикам строк (пик ≥ thr·max строки), цена — −log p
(+ эмбеддинг личности при k ≥ 3) + wjump·|Δx|/dy; пики прежних путей недоступны. Штраф только за сдвиг (первый порядок) —
на X-пересечении оба продолжения равноценны, путь уходит на чужую линию.

Здесь — те же пути по пикам, выгруженным `rowdec_peak_dump` (redec), и варианты:
- base   — копия прода (сверка с трассами redec: `--parity`);
- inert  — «инерция»: состояние пика несёт сглаженный наклон своего пути v (EMA, `--alpha`); переход стоит
           wjump·|Δx|/dy + wacc·|Δx/dy − v|; на пересечении путь продолжает свой наклон. O(пиков²) на строку, как прод.
Пишет пути вариантом в каталог-кэш (`<ключ листа>.pkl`, поле `alt` — как redec), чтобы мерить `_extra_cand_ceiling.py`.

  _track2.py --peaks F:/nds/output/taskS/peaks_dev --variant inert --wacc 0.5 --out F:/nds/output/taskS/tk_inert
"""
import sys, argparse, pickle, hashlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--peaks", default=r"F:/nds/output/taskS/peaks_dev")
ap.add_argument("--variant", default="base", choices=["base", "inert"])
ap.add_argument("--thr", type=float, default=0.6, help="порог пиков (доля max строки), как `rowdec_peak_thr`")
ap.add_argument("--wacc", type=float, default=0.5, help="inert: вес |Δx/dy − v|")
ap.add_argument("--alpha", type=float, default=0.3, help="inert: EMA наклона (вес нового шага)")
ap.add_argument("--kplus", type=int, default=0, help="путей на трек больше k (§6.305)")
ap.add_argument("--out", default="")
ap.add_argument("--parity", default="", help="кэш redec (tcache_pkdev): сверить base с его `alt`")
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()


def peaks_at(D, thr):
    k = D["k"]; out = []
    for mx, idx, pr, em in D["rows"]:
        m = pr.astype(np.float32) >= thr * mx
        ii = idx[m].astype(np.int64); pp = pr[m].astype(np.float32); ee = em[m] if em is not None else None
        if len(ii) > 8 * k:
            o = np.sort(np.argsort(pp)[-8 * k:]); ii, pp = ii[o], pp[o]; ee = ee[o] if ee is not None else None
        out.append((ii, pp, ee))
    return out


def paths(D, variant):
    k, y0, x0, wjump, wemb = D["k"], D["y0"], D["x0"], D["wjump"], D["wemb"]
    P = peaks_at(D, a.thr)
    kp = k + a.kplus                                       # прототипы личности — по k (как прод), путей — k + kplus
    use_emb = k >= D["emb_min_k"]; mu = None
    if use_emb:
        pts = [(i, j) for i in range(0, len(P), 7) for j in range(len(P[i][0]))]
        if len(pts) >= k * 8:
            M = np.stack([P[i][2][j] for i, j in pts]).astype(np.float32)
            rng = np.random.default_rng(0)
            mu = M[rng.choice(len(M), k, replace=False)]
            for _ in range(12):
                lab = ((M[:, None, :] - mu[None]) ** 2).sum(-1).argmin(1)
                for j in range(k):
                    if (lab == j).any():
                        mu[j] = M[lab == j].mean(0)
        else:
            use_emb = False
    out = []
    taken = [set() for _ in range(len(P))]
    for j in range(kp):
        rows, cands, locs = [], [], []
        for i, (ii, pp, ee) in enumerate(P):
            keep = np.array([q for q in range(len(ii)) if not any(abs(int(ii[q]) - t) <= 3 for t in taken[i])], int)
            if not len(keep):
                continue
            loc = -np.log(np.clip(pp[keep].astype(np.float64), 1e-6, 1.0))
            if use_emb and mu is not None and j < k:
                loc = loc + wemb * np.sqrt(((ee[keep].astype(np.float32) - mu[j]) ** 2).sum(-1))
            rows.append(i); cands.append(ii[keep].astype(float)); locs.append(loc)
        if len(rows) < 30:
            out.append({}); continue
        dp = [locs[0]]; bp = [np.full(len(cands[0]), -1, int)]
        V = np.zeros(len(cands[0]))
        for t in range(1, len(rows)):
            dy = max(1, rows[t] - rows[t - 1])
            step = (cands[t][:, None] - cands[t - 1][None, :]) / dy
            tot = dp[t - 1][None, :] + wjump * np.abs(step)
            if variant == "inert":
                tot = tot + a.wacc * np.abs(step - V[None, :])
            arg = tot.argmin(1)
            dp.append(locs[t] + tot[np.arange(len(cands[t])), arg]); bp.append(arg)
            if variant == "inert":
                V = (1 - a.alpha) * V[arg] + a.alpha * step[np.arange(len(cands[t])), arg]
        tr, s = {}, int(np.argmin(dp[-1]))
        for t in range(len(rows) - 1, -1, -1):
            x = int(cands[t][s])
            tr[rows[t] + y0] = float(x + x0); taken[rows[t]].add(x)
            s = int(bp[t][s])
            if s < 0:
                break
        out.append(tr)
    return out


def pack(tr):
    ys = np.array(sorted(tr), np.int32)
    return ys, np.array([tr[y] for y in ys], np.float32)


def key_of(stem):
    return f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"


def one(f):
    D = pickle.load(open(f, "rb"))
    return D["stem"], D["ti"], paths(D, a.variant)


def main():
    import multiprocessing as mp
    fs = sorted(Path(a.peaks).glob("*.pkl"))
    BY = defaultdict(dict); t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for n, (stem, ti, trs) in enumerate(pool.imap_unordered(one, fs)):
            BY[stem][ti] = trs
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(fs)} треков, {time.time() - t0:.0f} с")
    if a.parity:
        same = tot = 0
        for stem, by in BY.items():
            pf = Path(a.parity) / (key_of(stem) + ".pkl")
            if not pf.exists():
                continue
            c = pickle.load(open(pf, "rb"))
            ref = {tuple(t[0][:5].tolist()) + tuple(np.round(t[1][:5], 1).tolist()) for L, t in (c["alt"] or []) if len(t[0])}
            for ti, trs in by.items():
                for tr in trs:
                    if not tr:
                        continue
                    ys, xs = pack(tr); tot += 1
                    same += (tuple(ys[:5].tolist()) + tuple(np.round(xs[:5], 1).tolist())) in ref
        print(f"★ сверка base с redec: совпало начало у {same} из {tot} путей")
    if a.out:
        od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
        for stem, by in BY.items():
            alt = [(None, pack(tr)) for ti in sorted(by) for tr in by[ti] if tr]
            pickle.dump(dict(alt=alt), open(od / (key_of(stem) + ".pkl"), "wb"), protocol=4)
        print(f"★ записано листов {len(BY)} → {od}")


if __name__ == "__main__":
    main()
