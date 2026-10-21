r"""_trace_patch.py — КУСОЧКИ СКАНА ВДОЛЬ КАНДИДАТОВ ВЫБОРА ТРАССЫ (20.10, §6.325).

Для каждого листа скана `_line_choice_scan.py` (`--data`) — кандидаты в том же порядке (прод `traces`, декодер `alt`, доп. пути
`--extra` по порядку, как `idx` скана: < 1000 — основной кэш, ≥ 1000 — доп.), по `--np` точек, равномерно по строкам трассы, кусок
`--size`² вокруг точки: канал 0 — темнота (бумага строки − серый), канал 1 — маска самой трассы в куске. Пишет по листу
`--out/<md5 листа>.npz`: k (индексы кандидатов скана), X uint8 [n, np, 2, size, size].

  _trace_patch.py --data F:/nds/output/taskS/line_choice_i --cache F:/nds/output/taskS/tcache
      --extra "F:/nds/output/taskS/tcache_pk3|alt;F:/nds/output/taskS/tcache_inert|extra" --out F:/nds/output/taskS/tpatch
"""
import sys, argparse, pickle, hashlib, os, time
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def one(job):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from _cand_ceiling import dense_tr
    from auto import level_ink as LI
    rec_f, cache, extra, out, NP, S, CROSS = job
    rec = pickle.load(open(rec_f, "rb")); sh = rec["sheet"]
    of = Path(out) / (hashlib.md5(sh.encode("utf-8")).hexdigest()[:16] + ".npz")
    if of.exists():
        return sh, -1
    q = next(Path(r"F:\nds\projects\Archive").glob(f"*/wlg/{sh}"), None)
    if q is None:
        return sh, 0
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    tc = Path(cache) / (key + ".pkl")
    if not tc.exists():
        return sh, 0
    c = pickle.load(open(tc, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    xsrc = []
    for spec in extra.split(";") if extra else []:
        dd, _, fld = spec.partition("|")
        xp = Path(dd) / tc.name
        if xp.exists():
            xsrc += pickle.load(open(xp, "rb")).get(fld or "alt") or []
    try:
        gray = np.asarray(Image.open(c["image"]).convert("L"))
    except OSError:
        return sh, 0
    paper = LI.paper_of(gray).astype(np.int16)
    H, W = gray.shape; h = S // 2
    K, X = [], []
    DD = []
    for k, cs in enumerate(rec["cands"]):
        if cs["src"] == "prod":
            t = c["traces"][cs["idx"]][1]
        elif cs["idx"] < 1000:
            t = c["alt"][cs["idx"]][1]
        else:
            t = xsrc[cs["idx"] - 1000][1]
        DD.append(dense_tr(un(t)))
    if CROSS:                                            # §6.330: строки всех кандидатов на общей сетке (шаг 2)
        T = H // 2 + 1
        MX = np.full((len(DD), T), np.nan, np.float32)
        for k, d in enumerate(DD):
            if d:
                yy = np.array(sorted(d)); MX[k, yy // 2] = [d[y] for y in yy]
    for k, d in enumerate(DD):
        ys = np.array(sorted(d))
        if len(ys) < 50:
            continue
        pts = ys[np.linspace(0, len(ys) - 1, NP + 2).round().astype(int)[1:-1]]
        if CROSS:                                        # центры участков, где кандидат сходится с другим (не дублем) ≤ 3 px
            dx = np.abs(MX - MX[k][None, :])
            close = dx <= 3
            com = np.isfinite(dx).sum(1)
            dup = close.sum(1) >= 0.8 * np.maximum(com, 1)
            dup[k] = True
            cr = close[~dup].any(0)
            runs, i_ = [], 0
            idx_ = np.flatnonzero(cr)
            if len(idx_):
                br = np.flatnonzero(np.diff(idx_) > 3)
                st = np.r_[0, br + 1]; en = np.r_[br, len(idx_) - 1]
                runs = [int(idx_[(a_ + b_) // 2]) * 2 for a_, b_ in zip(st, en)]
            runs = [y for y in runs if y in d]
            if len(runs) > NP:
                runs = [runs[i] for i in np.linspace(0, len(runs) - 1, NP).round().astype(int)]
            pts = np.array(sorted(runs) + list(pts[:NP - len(runs)]))
        P = np.zeros((NP, 2, S, S), np.uint8)
        for j, y in enumerate(pts):
            x = int(round(d[int(y)])); y = int(y)
            y0, x0 = y - h, x - h
            ya, yb, xa, xb = max(0, y0), min(H, y0 + S), max(0, x0), min(W, x0 + S)
            if ya >= yb or xa >= xb:
                continue
            g = gray[ya:yb, xa:xb].astype(np.int16)
            P[j, 0, ya - y0:yb - y0, xa - x0:xb - x0] = np.clip(paper[ya:yb, None] - g, 0, 255).astype(np.uint8)
            for yy in range(ya, yb):
                if yy in d:
                    xx = int(round(d[yy])) - x0
                    if 0 <= xx < S:
                        P[j, 1, yy - y0, max(0, xx - 1):xx + 2] = 255
        K.append(k); X.append(P)
    tmp = of.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, k=np.array(K, np.int32), X=np.stack(X) if X else np.zeros((0, NP, 2, S, S), np.uint8))
    os.replace(tmp, of)
    return sh, len(K)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--np", type=int, default=8)
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cross", action="store_true", help="§6.330: точки — центры сближений с другими кандидатами (не дублями), остаток — равномерно")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), a.cache, a.extra, a.out, a.np, a.size, a.cross) for f in sorted(Path(a.data).glob("*.pkl"))]
    import multiprocessing as mp
    t0 = time.time(); n = 0
    with mp.Pool(a.workers) as pool:
        for sh, m in pool.imap_unordered(one, jobs):
            n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    print(f"★ готово: {n} листов → {a.out}")


if __name__ == "__main__":
    main()
