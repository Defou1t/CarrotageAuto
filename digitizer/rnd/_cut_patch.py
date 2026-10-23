r"""_cut_patch.py — КУСОЧКИ СКАНА У ПРАВОЙ ГРАНИЦЫ ШКАЛЫ 1× (ГДЕ ПЕРО УПИРАЕТСЯ — «ОБРУБ») С МЕТКОЙ УРОВНЯ ЭКСПЕРТА (23.10, §6.338).

Ручной детектор обруба (`_cut_scan.py`) тонет в штампах, печатных линиях и упорах соседних кривых у края; здесь — кусок 96 × 96 у
края шкалы уровня 0 со стороны v_right: 64 px внутрь полосы, 32 наружу (ориентирован так, что «наружу» — вправо), строки ±48;
канал 0 — темнота, канал 1 — метка колонки края. Метка — уровень эксперта в строке ≥ 1. Кривые с умножающей цепочкой, где эксперт
переключает; `--per` строк на кривую. Формат — как `_level_patch.py` (для `_level_cnn.py`).

  _cut_patch.py --out F:/nds/output/taskS/cpatch.npz
"""
import sys, argparse, pickle, hashlib, time
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np

S, IN, OUT = 96, 64, 32


def one(job):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from extract_nlgx import extract, NULL
    from auto import level_ink as LI, meta as M
    import decode_levels as DL
    sh, q, tc, per, IN, OUT = job
    try:
        mt = extract(str(q)); gray = np.asarray(Image.open(pickle.load(open(tc, "rb"))["image"]).convert("L"))
    except Exception:
        return sh, None
    paper = LI.paper_of(gray).astype(np.int16)
    H, W = gray.shape
    X, L, C = [], [], []
    rng = np.random.default_rng(abs(hash(sh)) % (2 ** 32))
    for c in mt["curves"]:
        if M.mnem_root(c["name"]) == "DA":
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2 or LI.is_shift(ch):
            continue
        lv = {}
        for y0, y1, l in c.get("segments") or []:
            for y in range(int(y0), int(y1) + 1):
                lv[y] = int(l)
        ys = np.array(sorted(lv))
        if len(ys) < 200 or len(set(lv.values())) < 2:
            continue
        s0 = ch[0]; xe = int(round(s0["x_right"])); sgn = 1 if s0["x_right"] >= s0["x_left"] else -1
        xs = xe + sgn * np.arange(-IN, OUT)              # слева направо: внутрь → наружу
        okx = (xs >= 0) & (xs < W)
        for y in rng.choice(ys, min(per, len(ys)), replace=False):
            y = int(y); ya, yb = y - S // 2, y + S // 2
            if ya < 0 or yb > H:
                continue
            P = np.zeros((1, 2, S, IN + OUT), np.uint8)
            g = gray[ya:yb][:, np.clip(xs, 0, W - 1)].astype(np.int16)
            P[0, 0] = np.where(okx[None, :], np.clip(paper[ya:yb, None] - g, 0, 255), 0).astype(np.uint8)
            P[0, 1, :, IN] = 255
            X.append(P); L.append(min(lv[y], 3)); C.append(c["name"])
    return sh, (np.stack(X), np.array(L, np.int8), C) if X else None


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--inn", type=int, default=IN, help="§6.340: px внутрь полосы")
    ap.add_argument("--outw", type=int, default=OUT, help="§6.340: px наружу")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="F:/nds/output/taskS/cpatch.npz")
    a = ap.parse_args()
    TS = Path("F:/nds/output/taskS")
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for f, cache in (("tcache_sheets.txt", "tcache"), ("outside_sheets.txt", "tcache_out")):
        for sh in lst(f):
            q = SRC.get(sh)
            if q is None:
                continue
            key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
            tc = TS / cache / (key + ".pkl")
            if tc.exists():
                jobs.append((sh, q, tc, a.per, a.inn, a.outw))
    import multiprocessing as mp
    X, L, SH, C = [], [], [], []; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, (sh, r) in enumerate(pool.imap_unordered(one, jobs)):
            if r is not None:
                X.append(r[0]); L.append(r[1]); SH += [sh] * len(r[1]); C += r[2]
            if (k + 1) % 300 == 0:
                print(f"  {k + 1}/{len(jobs)} листов, {sum(len(x) for x in L)} кусков, {time.time() - t0:.0f} с")
    np.savez_compressed(a.out, X=np.concatenate(X), lv=np.concatenate(L), sheet=np.array(SH), curve=np.array(C))
    print(f"★ кусков {sum(len(x) for x in L)} → {a.out}")


if __name__ == "__main__":
    main()
