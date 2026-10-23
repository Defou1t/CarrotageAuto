r"""_cut_scan.py — «ОБРУБ»: УПОР ПЕРА МЛАДШЕЙ ШКАЛЫ У ПРАВОЙ ГРАНИЦЫ — ДЕТЕКТОР И СОВПАДЕНИЕ С УРОВНЯМИ ЭКСПЕРТА (23.10, §6.338).

Правило заказчика (23.10): вести 1× до правой границы полосы и переходить на 5× там, где появляется «обруб» — перо 1× упирается в
край и рисует вертикаль вдоль него (чуть за границей), прямоугольник. §6.265 искал упор колонкой наибольшего контраста по всей
полосе — она вышла посередине (след пера 5×), у края упор не проверялся.

Детектор по каждой кривой эталона с умножающей цепочкой: окно у края шкалы уровня 0 со стороны v_right (x_right уровня 0),
[x_e − 8, x_e + 40] наружу; темнота D = бумага строки − серый; лишняя темнота E(y, x) = D − медиана D по строкам кривой в этой
колонке (печатная линия трека стоит во всех строках — вычитается); сигнал строки = max_x минимума E(·, x) в вертикальном окне ±W
строк (тушь подряд по вертикали). Мера: AUC сигнала для «эталон на уровне ≥ 1» против «0» по строкам (шаг 4) — по кривым и общая;
доля строк уровня ≥ 1 с сигналом выше порога. Пишет `--out` (pkl по кривым).

  _cut_scan.py --every 3 --out F:/nds/output/taskS/cut_scan.pkl
"""
import sys, argparse, pickle, hashlib, time
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np

W = 8
INK = 70                                   # порог туши (бумага − серый)


def signal(gray, paper, y0, y1, xe, sgn, wl=40, wr=25):
    """сигнал обруба по строкам [y0, y1) у края xe (sgn — направление наружу)"""
    H, Wd = gray.shape
    xs = np.arange(int(round(xe)) - sgn * wl, int(round(xe)) + sgn * wr, sgn)
    xs = xs[(xs >= 0) & (xs < Wd)]
    if len(xs) < 5:
        return None
    ya, yb = max(0, y0), min(H, y1)
    D = np.clip(paper[ya:yb, None].astype(np.int16) - gray[ya:yb][:, xs].astype(np.int16), 0, 255)
    # обруб лежит на печатной линии трека или рядом: колонка становится заметно темнее своего обычного уровня; сглаживание
    # по вертикали (±W) гасит зерно скана и горизонтальные пересечения других кривых
    from scipy.ndimage import uniform_filter1d
    Dm = uniform_filter1d(D.astype(np.float32), size=2 * W + 1, axis=0, mode="nearest")
    E = Dm - np.median(Dm, axis=0)[None, :]
    return ya, E.max(1)


def one(job):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from extract_nlgx import extract, NULL
    from auto import level_ink as LI, meta as M
    import decode_levels as DL
    sh, q, tc = job
    try:
        mt = extract(str(q)); gray = np.asarray(Image.open(pickle.load(open(tc, "rb"))["image"]).convert("L"))
    except Exception:
        return sh, []
    paper = LI.paper_of(gray)
    out = []
    for c in mt["curves"]:
        if M.mnem_root(c["name"]) == "DA":
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2 or LI.is_shift(ch):
            continue
        lv = np.full(c["n_rows"], -1, np.int8)
        for y0, y1, l in c.get("segments") or []:
            a, b = int(y0) - c["top_y"], int(y1) - c["top_y"] + 1
            lv[max(0, a):max(0, b)] = l
        has = np.array([x != NULL for x in c["xs"]])
        if (lv >= 1).sum() < 50 or (lv == 0).sum() < 50:
            continue                                     # только кривые, где эксперт переключает
        s0 = ch[0]; xe = s0["x_right"]; sgn = 1 if s0["x_right"] >= s0["x_left"] else -1
        r = signal(gray, paper, c["top_y"], c["top_y"] + c["n_rows"], xe, sgn)
        if r is None:
            continue
        ya, sg = r
        rows = np.arange(0, c["n_rows"], 4)
        ok = (lv[rows] >= 0) & has[rows]
        rows = rows[ok]
        yy = c["top_y"] + rows - ya
        ok2 = (yy >= 0) & (yy < len(sg))
        rows, yy = rows[ok2], yy[ok2]
        out.append(dict(curve=c["name"], root=M.mnem_root(c["name"]), sig=sg[yy].astype(np.float16), lv=lv[rows], K=len(ch)))
    return sh, out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="F:/nds/output/taskS/cut_scan.pkl")
    a = ap.parse_args()
    TS = Path("F:/nds/output/taskS")
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for f, cache in (("tcache_sheets.txt", "tcache"), ("outside_sheets.txt", "tcache_out")):
        for sh in lst(f)[::a.every]:
            q = SRC.get(sh)
            if q is None:
                continue
            key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
            tc = TS / cache / (key + ".pkl")
            if tc.exists():
                jobs.append((sh, q, tc))
    import multiprocessing as mp
    R = {}; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, (sh, o) in enumerate(pool.imap_unordered(one, jobs)):
            if o:
                R[sh] = o
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.out, "wb"))
    from sklearn.metrics import roc_auc_score
    C = [c for o in R.values() for c in o]
    sig = np.concatenate([c["sig"] for c in C]).astype(float); y = np.concatenate([(c["lv"] >= 1) for c in C]).astype(int)
    print(f"★ кривых с переключением {len(C)}, строк {len(y)} (уровень ≥ 1 — {y.mean():.2f})")
    print(f"★ AUC сигнала обруба (все строки): {roc_auc_score(y, sig):.4f}")
    au = [roc_auc_score((c['lv'] >= 1).astype(int), c['sig'].astype(float)) for c in C if 0 < (c['lv'] >= 1).mean() < 1]
    print(f"★ AUC по кривым: медиана {np.median(au):.3f}; ≥ 0.9 у {np.mean(np.array(au) >= 0.9):.2f}; ≥ 0.8 у {np.mean(np.array(au) >= 0.8):.2f}")
    for th in (10, 20, 40):
        print(f"   порог {th}: строк уровня ≥ 1 с сигналом {np.mean(sig[y == 1] > th):.2f}, уровня 0 — {np.mean(sig[y == 0] > th):.2f}")
    from collections import defaultdict
    by = defaultdict(list)
    for c, a_ in zip([c for c in C if 0 < (c['lv'] >= 1).mean() < 1], au):
        by[c["root"]].append(a_)
    print("   по семействам (медиана AUC, кривых):", "; ".join(f"{k} {np.median(v):.2f}/{len(v)}" for k, v in sorted(by.items(), key=lambda t: -len(t[1]))[:12]))


if __name__ == "__main__":
    main()
