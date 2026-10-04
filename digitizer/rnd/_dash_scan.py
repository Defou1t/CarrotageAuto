r"""_dash_scan.py — ВЕРТИКАЛЬНЫЙ ПУНКТИР У КРАЯ ШКАЛЫ КАК ПРИЗНАК 5× (04.10, к §6.271).

Заказчик (04.10): если ГЗ/БК на 5× с первой строки листа — «справа от начала линии видны вертикальные полосы (пунктир, который
имеет большую длину)». На картинках (`H_level_ink_v2/P00`, `P05`) это длинные вертикальные штрихи с большими промежутками у
края шкалы 1× со стороны больших значений — похоже на перо 1×, ушедшее за шкалу. Если так, пунктир отмечает 5× на всём участке,
а не только в начале.

Проверка на эталоне (поле + сорт A, кривые с цепочкой масштабов-умножений): зона — край шкалы 1× со стороны больших значений
(доли ширины шкалы `--z0`…`--z1`); тушь — затемнение > `--thr`. Колонка зоны «пунктирна» в строке y, если в окне ±`--win`
строк покрытие тушью в `--cov` и в ней есть ран длиной ≥ `--run` строк, накрывающий окрестность y. Колонки сплошных линий (сетка,
граница трека) отсекаются верхней границей покрытия. По строкам (шаг 8) — доля «пунктир есть» при уровне эксперта 0 и ≥ 1;
то же в первых 400 строках кривой против начального уровня эксперта.

  _dash_scan.py --workers 4
"""
import sys, argparse, pickle, hashlib, time, os
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def seg_levels(segs, rows):
    out = np.zeros(len(rows), np.int16)
    for y0, y1, lv in segs or []:
        if lv:
            out[(rows >= y0) & (rows <= y1)] = lv
    return out


def is_mult(ch):
    """цепочка-умножение: тот же ноль, диапазон растёт (1× → 5× …)"""
    if len(ch) < 2:
        return False
    s0, s1 = ch[0], ch[1]
    r0 = s0["v_right"] - s0["v_left"]; r1 = s1["v_right"] - s1["v_left"]
    return r0 != 0 and abs(r1 / r0) >= 1.5 and abs(s1["v_left"] - s0["v_left"]) <= 1e-6 * max(1.0, abs(s0["v_right"]))


def dash_rows(gray, paper, cv, P):
    """→ (строки шага 8, флаг пунктира) для кривой. v2: штрихи — прямые узкие вертикальные раны длиной run…run_max строк;
    колонка (±2 px) пунктирна, если в ±align строк ≥ n_al таких штрихов; строка y «под пунктиром», если она между первым и
    последним штрихом такой колонки (или в ±win от штриха)."""
    H, W = gray.shape
    s0 = cv["chain"][0]
    xl, xr = float(s0["x_left"]), float(s0["x_right"])
    w = abs(xr - xl)
    if w < 50:
        return None
    hi_side = xr if s0["v_right"] > s0["v_left"] else xl          # край больших значений шкалы 1×
    sgn = 1.0 if hi_side >= (xl + xr) / 2 else -1.0
    za = hi_side - sgn * (1 - P["z0"]) * w; zb = hi_side + sgn * (P["z1"] - 1) * w
    c0, c1 = int(max(0, min(za, zb))), int(min(W, max(za, zb)))
    if c1 - c0 < 5:
        return None
    gy = cv["gy"]
    pad = P["align"]
    y0 = int(max(0, gy[0] - pad)); y1 = int(min(H, gy[-1] + pad))
    m8 = 8
    cc0, cc1 = max(0, c0 - m8), min(W, c1 + m8)
    D = (paper[y0:y1, None] - gray[y0:y1, cc0:cc1].astype(np.int16)) > P["thr"]
    n, wz = D.shape
    rows = np.arange(int(gy[0]), int(gy[-1]) + 1, 8)
    strokes = []                                                       # (колонка, начало, конец) в координатах листа
    for j in range(c0 - cc0, c1 - cc0):
        col = D[:, j]
        if not col.any():
            continue
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
        ln = en - st
        sel = (ln >= P["run"]) & (ln <= P["run_max"])
        for a_, b_ in zip(st[sel], en[sel]):
            # узость: колонки в ±8 px вдоль рана почти пусты
            lft = D[a_:b_, max(0, j - m8)].mean(); rgt = D[a_:b_, min(wz - 1, j + m8)].mean()
            if lft < 0.5 and rgt < 0.5:
                strokes.append((j + cc0, a_ + y0, b_ + y0))
    flag = np.zeros(len(rows), bool)
    if strokes:
        S = np.array(strokes, np.int64)
        # слить соседние колонки одного штриха: ключ — колонка // 3
        S = S[np.argsort(S[:, 0])]
        for key in np.unique(S[:, 0] // 3):
            g = S[np.abs(S[:, 0] // 3 - key) <= 1]
            # уникальные штрихи по перекрытию строк
            g = g[np.argsort(g[:, 1])]
            u = []
            for c_, a_, b_ in g:
                if u and a_ <= u[-1][1]:
                    u[-1][1] = max(u[-1][1], b_)
                else:
                    u.append([a_, b_])
            if len(u) < P["n_al"]:
                continue
            U = np.array(u)
            mids = (U[:, 0] + U[:, 1]) // 2
            for i_ in range(len(U)):
                nb = np.sum(np.abs(mids - mids[i_]) <= P["align"])
                if nb >= P["n_al"]:
                    # строки между соседними штрихами этой колонки (в пределах align) и ± win вокруг штриха
                    lo_ = U[i_, 0] - P["win"]; hi_ = U[i_, 1] + P["win"]
                    if i_ + 1 < len(U) and U[i_ + 1, 0] - U[i_, 1] <= P["align"]:
                        hi_ = max(hi_, U[i_ + 1, 0])
                    flag |= (rows >= lo_) & (rows <= hi_)
    return rows, flag, strokes


def scan_sheet(job):
    sheet, img, curves, P = job
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    gray = np.asarray(Image.open(img).convert("L"))
    H = gray.shape[0]
    paper = np.empty(H, np.int16)
    for r in range(0, H, 2048):
        paper[r:r + 2048] = np.percentile(gray[r:r + 2048, ::4], 90, axis=1)
    out = []
    rgb = np.asarray(Image.open(img).convert("RGB")) if P.get("color") else None
    W = gray.shape[1]
    def chroma_at(ys, xs):
        px = rgb[ys, xs].astype(np.int32)
        return np.stack([px[:, 0] - px[:, 2], px[:, 1] - (px[:, 0] + px[:, 2]) // 2], 1)
    for cv in curves:
        r = dash_rows(gray, paper, cv, P)
        if r is None:
            continue
        rows, near, strokes = r
        lt = seg_levels(cv["lt"], rows)
        rec = dict(ci=cv["ci"], set=cv["set"], root=cv["root"], rows=rows.astype(np.int32), near=near, lt=lt.astype(np.int8))
        if rgb is not None:
            # цвет линии эксперта: самый тёмный пиксель в ±2 px от вершины, каждая 16-я строка
            gy, gx = cv["gy"][::16].astype(np.int64), np.round(cv["gx"][::16]).astype(np.int64)
            ok = (gx >= 2) & (gx < W - 2) & (gy < gray.shape[0])
            gy, gx = gy[ok], gx[ok]
            win = np.stack([gray[gy, np.clip(gx + d, 0, W - 1)] for d in range(-2, 3)], 1)
            bx = gx + win.argmin(1) - 2
            dark = (paper[gy] - gray[gy, bx].astype(np.int16)) > P["thr"]
            cc = chroma_at(gy[dark], bx[dark]) if dark.any() else np.zeros((0, 2), np.int32)
            rec["curve_chroma"] = np.median(cc, 0) if len(cc) >= 20 else None
            # цвет каждого штриха: медиана по пикселям рана
            st_col = []
            for c_, a_, b_ in strokes:
                ys_ = np.arange(a_, b_, 3)
                st_col.append((c_, a_, b_, np.median(chroma_at(ys_, np.full(len(ys_), c_)), 0)))
            # «свой пунктир»: строки рядом со штрихом того же цвета, что линия
            own = np.zeros(len(rows), bool)
            if rec["curve_chroma"] is not None:
                for c_, a_, b_, ch_ in st_col:
                    if np.abs(ch_ - rec["curve_chroma"]).max() <= P["ctol"]:
                        own |= (rows >= a_ - P["win"]) & (rows <= b_ + P["win"])
            rec["own"] = own & near
        out.append(rec)
    return sheet, out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--z0", type=float, default=0.80)
    ap.add_argument("--z1", type=float, default=1.05)
    ap.add_argument("--thr", type=int, default=100)
    ap.add_argument("--win", type=int, default=150)
    ap.add_argument("--cov-lo", type=float, default=0.10)
    ap.add_argument("--cov-hi", type=float, default=0.75)
    ap.add_argument("--run", type=int, default=25)
    ap.add_argument("--run-max", type=int, default=300)
    ap.add_argument("--align", type=int, default=1500)
    ap.add_argument("--n-al", type=int, default=3)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/dash_scan.pkl")
    ap.add_argument("--color", action="store_true", help="цвет линии эксперта и штрихов: «свой пунктир» того же цвета")
    ap.add_argument("--ctol", type=int, default=25)
    ap.add_argument("--colored", type=int, default=30, help="линия «цветная», если max|цвет| ≥ этого")
    a = ap.parse_args()
    import multiprocessing as mp
    CUR = pickle.load(open(a.cache, "rb"))
    IMGS = {}
    for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
        if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            IMGS.setdefault(q.stem, q)
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    by = defaultdict(list)
    for i, cv in enumerate(CUR):
        if is_mult(cv["chain"]) and cv["lt"] and len(cv["gy"]) > 200:
            by[cv["sheet"]].append(dict(ci=i, set=cv["set"], root=cv["root"], chain=cv["chain"], gy=cv["gy"], gx=cv["gx"], lt=cv["lt"]))
    P = dict(z0=a.z0, z1=a.z1, thr=a.thr, win=a.win, cov_lo=a.cov_lo, cov_hi=a.cov_hi, run=a.run, run_max=a.run_max,
             align=a.align, n_al=a.n_al, color=a.color, ctol=a.ctol)
    jobs = [(sh, str(IMGS[SRC[sh].stem]), cs, P) for sh, cs in by.items() if sh in SRC and SRC[sh].stem in IMGS]
    print(f"листов {len(jobs)}, кривых {sum(len(j[2]) for j in jobs)}")
    R = []
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, (sh, out) in enumerate(pool.imap_unordered(scan_sheet, jobs)):
            R.extend(out)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    C = Counter()
    fam = defaultdict(Counter)
    for r in R:
        hi = r["lt"] >= 1
        C["строк 1×"] += int((~hi).sum()); C["строк 5×+"] += int(hi.sum())
        C["пунктир при 1×"] += int((r["near"] & ~hi).sum()); C["пунктир при 5×+"] += int((r["near"] & hi).sum())
        f = r["root"].replace("BKZ_", "")
        fam[f]["1×"] += int((~hi).sum()); fam[f]["5×"] += int(hi.sum())
        fam[f]["п1"] += int((r["near"] & ~hi).sum()); fam[f]["п5"] += int((r["near"] & hi).sum())
        k = min(len(r["rows"]), 50)                                       # первые 400 строк
        st = int(np.bincount(r["lt"][:k].clip(0)).argmax()) >= 1
        dn = bool(r["near"][:k].mean() >= 0.5)
        C[("начало", st, dn)] += 1
    print(f"★ строки: пунктир при уровне эксперта 1× — {C['пунктир при 1×'] / max(1, C['строк 1×']):.1%} "
          f"({C['строк 1×']} строк), при 5× и выше — {C['пунктир при 5×+'] / max(1, C['строк 5×+']):.1%} ({C['строк 5×+']} строк)")
    print(f"★ начало кривой (первые 400 строк): начало 5×+ — пунктир у {C[('начало', True, True)]} из "
          f"{C[('начало', True, True)] + C[('начало', True, False)]}; начало 1× — пунктир у {C[('начало', False, True)]} из "
          f"{C[('начало', False, True)] + C[('начало', False, False)]}")
    if a.color:
        K = Counter()
        for r in R:
            cc = r.get("curve_chroma")
            if cc is None:
                continue
            kind = "цветные" if np.abs(cc).max() >= a.colored else "чёрные"
            hi = r["lt"] >= 1
            K[(kind, "1×")] += int((~hi).sum()); K[(kind, "5×")] += int(hi.sum())
            K[(kind, "свой при 1×")] += int((r["own"] & ~hi).sum()); K[(kind, "свой при 5×")] += int((r["own"] & hi).sum())
            K[(kind, "любой при 1×")] += int((r["near"] & ~hi).sum()); K[(kind, "любой при 5×")] += int((r["near"] & hi).sum())
            K[(kind, "кривых")] += 1
        for kind in ("цветные", "чёрные"):
            print(f"★ {kind} линии ({K[(kind, 'кривых')]} кривых): пунктир ЛЮБОЙ — при 1× {K[(kind, 'любой при 1×')] / max(1, K[(kind, '1×')]):.1%}, "
                  f"при 5× {K[(kind, 'любой при 5×')] / max(1, K[(kind, '5×')]):.1%}; СВОЕГО цвета — при 1× "
                  f"{K[(kind, 'свой при 1×')] / max(1, K[(kind, '1×')]):.1%}, при 5× {K[(kind, 'свой при 5×')] / max(1, K[(kind, '5×')]):.1%}")
    for f, c in sorted(fam.items(), key=lambda kv: -(kv[1]["1×"] + kv[1]["5×"]))[:10]:
        print(f"   {f:6s}: пунктир при 1× {c['п1'] / max(1, c['1×']):.1%}, при 5×+ {c['п5'] / max(1, c['5×']):.1%} "
              f"(строк {c['1×'] + c['5×']})")


if __name__ == "__main__":
    main()
