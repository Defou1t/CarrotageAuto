r"""_seg_patch.py — КУСКИ СКАНА ДЛЯ СЕТИ «ТУШЬ ОЦИФРОВАННЫХ КРИВЫХ» (24.10, §6.345).

Оракул §6.345: если карту декодера погасить везде, кроме ±4 px от кривых эталона, честный путь появляется у 31% слотов без честного
кандидата (0% без маски). Декодер видит одно серое «темнее бумаги» без сетки; здесь — цвет (RGB) и кусок крупнее. По каждому листу
поля и вне поля (не скважин сорта A): `--per` кусков S × S — две трети с центром у случайной точки кривой эталона (сдвиг ±S/3),
треть — случайно в прямоугольнике кривых. Цель — пиксели в ±2 px по строке от ломаной эксперта (горизонтальные участки между
соседними строками закрашены), все кривые листа кроме DA. Строки — после `frame.row_shift`, как у декодера.
Пишет по листу `--out/<md5>.npz`: X uint8 [n, 3, S, S], Y uint8 [n, S, S], sheet, well.

  _seg_patch.py --out F:/nds/output/taskS/segpatch
"""
import sys, argparse, pickle, hashlib, time, os
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def curve_mask(G, y0, x0, S, hw=2):
    M = np.zeros((S, S), np.uint8)
    for g in G:
        ys = [y for y in range(y0 - 1, y0 + S + 1) if y in g]
        for y in ys:
            r = y - y0
            xa = xb = g[y]
            if y + 1 in g:
                xa, xb = min(xa, g[y + 1]), max(xb, g[y + 1])
            if 0 <= r < S:
                c0, c1 = int(round(xa)) - x0 - hw, int(round(xb)) - x0 + hw + 1
                if c1 > 0 and c0 < S:
                    M[r, max(0, c0):min(S, c1)] = 1
    return M


# §6.350: семейства кривых для сети туши по семействам (`--fam`): бит k маски — семейство FAMS[k], последний бит — прочие
FAMS = ["GZ", "SP", "GK", "OGZ", "DS", "BK", "ZAT", "CALI", "NGK", "MBK", "IK", "T", "TM", "MGZ", "MPZ", "PZ", "OTHER"]


def fam_index(root):
    import re
    r = re.sub(r"^BKZ_", "", str(root))
    return FAMS.index(r) if r in FAMS[:-1] else len(FAMS) - 1


def one(job):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import meta as Mm, imaging as im, frame as FR
    sh, q, tc, well, out, per, S, seed, fam = job
    of = Path(out) / (hashlib.md5(sh.encode("utf-8")).hexdigest()[:16] + ".npz")
    if of.exists():
        return sh, -1
    try:
        c = pickle.load(open(tc, "rb"))
        rgb = im.load_rgb(c["image"])
        if getattr(c["frame"], "row_shift", None) is not None:
            rgb = FR.apply_row_shift(rgb, c["frame"].row_shift)
        CV = [cv for cv in extract(str(q))["curves"] if Mm.mnem_root(cv["name"]) != "DA"]
        G = [dense(cv) for cv in CV]
    except Exception:
        return sh, 0
    FI = [fam_index(Mm.mnem_root(cv["name"])) for cv, g in zip(CV, G) if g]
    G = [g for g in G if g]
    if not G:
        return sh, 0
    H, W = rgb.shape[:2]
    pts = [(y, x) for g in G for y, x in g.items()]
    ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
    ylo, yhi, xlo, xhi = int(ys.min()), int(ys.max()), int(xs.min()) - 50, int(xs.max()) + 50
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in range(per):
        if i < 2 * per // 3:
            j = rng.integers(len(pts)); yc, xc = pts[j][0] + rng.integers(-S // 3, S // 3 + 1), pts[j][1] + rng.integers(-S // 3, S // 3 + 1)
        else:
            yc, xc = rng.integers(ylo, yhi + 1), rng.integers(xlo, xhi + 1)
        y0 = int(np.clip(yc - S // 2, 0, H - S)); x0 = int(np.clip(xc - S // 2, 0, W - S))
        X.append(rgb[y0:y0 + S, x0:x0 + S].transpose(2, 0, 1).copy())
        if fam:                                           # §6.350: бит семейства у каждого пикселя туши
            M = np.zeros((S, S), np.uint32)
            for k in sorted(set(FI)):
                M |= curve_mask([g for g, f in zip(G, FI) if f == k], y0, x0, S).astype(np.uint32) << np.uint32(k)
            Y.append(M)
        else:
            Y.append(curve_mask(G, y0, x0, S))
    tmp = str(of) + ".tmp.npz"
    if fam:
        np.savez_compressed(tmp, X=np.stack(X), Yb=np.stack(Y), fams=np.array(FAMS), sheet=sh, well=well)
    else:
        np.savez_compressed(tmp, X=np.stack(X), Y=np.stack(Y), sheet=sh, well=well)
    os.replace(tmp, of)
    return sh, len(X)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=r"F:/nds/output/taskS/segpatch")
    ap.add_argument("--fam", action="store_true", help="§6.350: цель — битовая маска семейств (Yb uint32, бит k — FAMS[k]); те же куски")
    a = ap.parse_args()
    TS = Path(r"F:/nds/output/taskS")
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    well = lambda sh: SRC[sh].parent.parent.name
    HOLD = set(lst("holdoutA_sheets.txt")); HW = {well(x) for x in HOLD if x in SRC}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = []
    for f, cache in (("tcache_sheets.txt", "tcache_inert"), ("outside_sheets.txt", "tcache_out")):
        for sh in lst(f):
            q = SRC.get(sh)
            if q is None or sh in HOLD or well(sh) in HW:
                continue
            key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
            tc = TS / cache / (key + ".pkl")
            if tc.exists():
                jobs.append((sh, q, str(tc), well(sh), a.out, a.per, a.size, int(hashlib.md5(sh.encode()).hexdigest()[:8], 16), a.fam))
    import multiprocessing as mp
    t0 = time.time(); n = 0
    with mp.Pool(a.workers) as pool:
        for k, (sh, r) in enumerate(pool.imap_unordered(one, jobs)):
            n += max(r, 0)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(jobs)} листов, {n} кусков, {time.time() - t0:.0f} с")
    print(f"★ листов {len(jobs)}, кусков {n} → {a.out}")


if __name__ == "__main__":
    main()
