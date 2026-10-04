r"""_style_scan.py — СТИЛЬ ЛИНИИ (ТОЛЩИНА, ЦВЕТ, ПУНКТИР) КАК ПРИЗНАК ИМЕНИ (04.10, к §6.273).

Заказчик (04.10): кривые трека различает по цвету, толщине и штриху — «GZ1 чёрная и более толстая по штриху, GZ2 более тонкая
чёрная; GZ3 + GZ4 аналогично, но могут быть смещены вместе в сторону (во второй колонке листа); GZ1-6 могут быть разноцветные;
2 из 6 линий могут быть пунктирные».

Здесь — замер стиля вдоль линии (каждая 8-я строка) для кривых кэша `_level_bench.py` (поле + сорт A), по трассе ЭКСПЕРТА и по
трассе ВЫДАЧИ:
- толщина штриха: горизонтальный ран туши (затемнение > --thr), содержащий точку линии или ближайший в ±3 px, приведённый к
  нормали (÷ √(1 + наклон²)); берутся только строки, где ран не длиннее 40 px (не пересечение и не горизонтальный размах);
- цвет: R − B и G − (R + B)/2 самого тёмного пикселя рана; затемнение — бумага минус яркость в нём;
- разрыв: в ±2 px от линии нет туши (доля таких строк — признак пунктира).
По кривой — медианы и доли. Пишет `--dump`; сводка: различает ли стиль пары имён в одном треке (толщина ГЗ1 против ГЗ2 и т. д.).

  _style_scan.py --workers 4
"""
import sys, argparse, pickle, time, re
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def style_along(rgb, gray, paper, ys, xs, P):
    """стиль линии по точкам (ys, xs) — медианы толщины, цвета, затемнения и доля разрывов (векторно)"""
    H, W = gray.shape
    ys = ys.astype(np.int64); xs = np.round(xs).astype(np.int64)
    ok = (ys >= 1) & (ys < H - 1) & (xs >= 45) & (xs < W - 45)
    ys, xs = ys[ok], xs[ok]
    n = len(ys)
    if n < 20:
        return None
    half = 44; NC = 2 * half + 1; c = half
    cols = xs[:, None] + np.arange(-half, half + 1)[None, :]
    D = paper[ys][:, None] - gray[ys[:, None], cols].astype(np.int16)          # [n, 89]
    ink = D > P["thr"]
    gap = ~ink[:, c - 2:c + 3].any(1)
    # j0 — ближайшая к центру тушь в ±3 px (порядок 0, −1, +1, −2, +2, −3, +3)
    j0 = np.full(n, -1)
    for d in (0, -1, 1, -2, 2, -3, 3):
        m = (j0 < 0) & ink[:, c + d]
        j0[m] = c + d
    has = j0 >= 0
    ar = np.arange(NC)[None, :]
    leftF = np.maximum.accumulate(np.where(~ink, ar, -1), axis=1)             # последний «не тушь» слева (включая j)
    rightF = np.minimum.accumulate(np.where(~ink, ar, NC)[:, ::-1], axis=1)[:, ::-1]
    jj = np.clip(j0, 0, NC - 1)
    L = leftF[np.arange(n), jj] + 1; R = rightF[np.arange(n), jj] - 1
    wl = (R - L + 1).astype(float)
    good = has & (L > 0) & (R < NC - 1) & (wl <= 40)
    Dm = np.where((ar >= L[:, None]) & (ar <= R[:, None]), D, -999)
    k = Dm.argmax(1)
    px = rgb[ys, cols[np.arange(n), k]].astype(np.int32)
    c1 = (px[:, 0] - px[:, 2]).astype(float); c2 = (px[:, 1] - (px[:, 0] + px[:, 2]) / 2).astype(float)
    dk = D[np.arange(n), k].astype(float)
    s_ = np.gradient(xs.astype(float), ys.astype(float)) if n > 2 else np.zeros(n)
    wn = wl / np.sqrt(1.0 + s_ * s_)
    if good.sum() < 10:
        return None
    g = good
    return dict(n=int(n), width=float(np.median(wn[g])), width_h=float(np.median(wl[g])),
                c1=float(np.median(c1[g])), c2=float(np.median(c2[g])), dark=float(np.median(dk[g])),
                gap=float(gap.mean()), used=float(g.mean()))


def scan_sheet(job):
    sheet, img, curves, P = job
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(img)
    gray = np.asarray(im.convert("L"))
    rgb = np.asarray(im.convert("RGB"))
    H = gray.shape[0]
    paper = np.empty(H, np.int16)
    for r in range(0, H, 2048):
        paper[r:r + 2048] = np.percentile(gray[r:r + 2048, ::4], 90, axis=1)
    out = []
    for cv in curves:
        rec = dict(ci=cv["ci"], set=cv["set"], sheet=cv["sheet"], name=cv["name"], root=cv["root"])
        st = P["step"]
        rec["truth"] = style_along(rgb, gray, paper, cv["gy"][::st], cv["gx"][::st], P)
        rec["out"] = style_along(rgb, gray, paper, cv["ty"][::st], cv["tx"][::st], P) if len(cv["ty"]) else None
        out.append(rec)
    return sheet, out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--thr", type=int, default=60)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/style_scan.pkl")
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
        by[cv["sheet"]].append(dict(ci=i, set=cv["set"], sheet=cv["sheet"], name=cv["name"], root=cv["root"],
                                    gy=cv["gy"], gx=cv["gx"], ty=cv["ty"], tx=cv["tx"]))
    P = dict(thr=a.thr, step=a.step)
    jobs = [(sh, str(IMGS[SRC[sh].stem]), cs, P) for sh, cs in by.items() if sh in SRC and SRC[sh].stem in IMGS]
    print(f"листов {len(jobs)}, кривых {sum(len(j[2]) for j in jobs)}")
    R = []
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, (sh, out) in enumerate(pool.imap_unordered(scan_sheet, jobs)):
            R.extend(out)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    ok = [r for r in R if r["truth"]]
    print(f"★ кривых со стилем по эталону: {len(ok)} из {len(R)}")
    # толщина по номеру зонда ГЗ: в одном листе пара (GZk, GZm), k < m — доля, где k толще
    bys = defaultdict(dict)
    for r in ok:
        m = re.match(r"^(GZ|BKZ_GZ)(\d)", r["name"])
        if m:
            bys[r["sheet"]][int(m.group(2))] = r["truth"]
    C = Counter()
    for sh, d in bys.items():
        ks = sorted(d)
        for i, k in enumerate(ks):
            for m in ks[i + 1:]:
                dw = d[k]["width"] - d[m]["width"]
                C[(k, m, "пар")] += 1; C[(k, m, "k толще")] += dw > 0.5; C[(k, m, "m толще")] += dw < -0.5
    print("★ ГЗ на одном листе: пара (k, m) — k толще / m толще / пар (разница > 0.5 px)")
    for (k, m) in sorted({(k, m) for k, m, _ in C}):
        n = C[(k, m, "пар")]
        if n >= 15:
            print(f"   ГЗ{k}–ГЗ{m}: {C[(k, m, 'k толще')]} / {C[(k, m, 'm толще')]} / {n}")
    col = Counter(); dash = Counter()
    for r in ok:
        t = r["truth"]
        f = r["root"].replace("BKZ_", "")
        col[(f, "цветная" if max(abs(t["c1"]), abs(t["c2"])) >= 30 else "чёрная")] += 1
        dash[(f, "пунктир" if t["gap"] >= 0.25 else "сплошная")] += 1
    fams = Counter(r["root"].replace("BKZ_", "") for r in ok)
    print("★ по семействам: цветных / пунктирных (разрывов ≥ 25%) / всего")
    for f, n in fams.most_common(14):
        print(f"   {f:7s} {col[(f, 'цветная')]:4d} / {dash[(f, 'пунктир')]:4d} / {n}")


if __name__ == "__main__":
    main()
