r"""_level_patch.py — ВИДНО ЛИ УРОВЕНЬ МАСШТАБА ПО КУСОЧКУ СКАНА ВОКРУГ ЭТАЛОННОЙ ТРАССЫ (21.10, §6.329).

Кривые эталона с умножающей цепочкой шкал (не сдвиговой), листы поля + вне поля + сорт A (кэш `tcache` / `tcache_out` — путь скана).
По `--per` строк кривой (равномерно среди строк с точкой эталона): кусок `--size`² вокруг (y, x эталона) — темнота + маска трассы
(как `auto.trace_look.patches`, одна точка); метка — уровень эталона в строке. Пишет `--out` (npz: X uint8 [n, 1, 2, S, S],
lv, sheet, curve).

  _level_patch.py --out F:/nds/output/taskS/lpatch.npz
"""
import sys, argparse, pickle, hashlib, time
sys.path.insert(0, "F:/nds/Auto"); sys.path.insert(0, "F:/nds/Auto/digitizer"); sys.path.insert(0, "F:/nds/Auto/digitizer/rnd")
from pathlib import Path
import numpy as np


def one(job):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from auto import level_ink as LI, trace_look as TL, meta as M
    import decode_levels as DL
    sh, q, tc, per, S = job
    try:
        mt = extract(str(q)); img = pickle.load(open(tc, "rb"))["image"]
        gray = np.asarray(Image.open(img).convert("L"))
    except Exception:
        return sh, None
    paper = LI.paper_of(gray)
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
        if len(set(lv.values())) < 2:                    # только кривые, где эталон переключает уровень
            continue
        d = dense(c)
        ys = np.array(sorted(y for y in d if y in lv))
        if len(ys) < 200:
            continue
        for y in rng.choice(ys, min(per, len(ys)), replace=False):
            # одна точка ровно в y (TL.patches берёт точки по квантилям трассы) — та же нарезка, что там
            h = S // 2; x = int(round(d[int(y)])); y = int(y)
            H, W = gray.shape; y0_, x0_ = y - h, x - h
            ya, yb, xa, xb = max(0, y0_), min(H, y0_ + S), max(0, x0_), min(W, x0_ + S)
            if ya >= yb or xa >= xb:
                continue
            Pm = np.zeros((1, 2, S, S), np.uint8)
            g = gray[ya:yb, xa:xb].astype(np.int16)
            Pm[0, 0, ya - y0_:yb - y0_, xa - x0_:xb - x0_] = np.clip(paper[ya:yb, None].astype(np.int16) - g, 0, 255).astype(np.uint8)
            for yy in range(ya, yb):
                if yy in d:
                    xx = int(round(d[yy])) - x0_
                    if 0 <= xx < S:
                        Pm[0, 1, yy - y0_, max(0, xx - 1):xx + 2] = 255
            X.append(Pm); L.append(min(lv[y], 3)); C.append(c["name"])
    return sh, (np.stack(X), np.array(L, np.int8), C) if X else None


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="F:/nds/output/taskS")
    ap.add_argument("--per", type=int, default=40)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="F:/nds/output/taskS/lpatch.npz")
    a = ap.parse_args()
    TS = Path(a.ts)
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
                jobs.append((sh, q, tc, a.per, a.size))
    import multiprocessing as mp
    X, L, SH, C = [], [], [], []
    t0 = time.time(); n = 0
    with mp.Pool(a.workers) as pool:
        for sh, r in pool.imap_unordered(one, jobs):
            n += 1
            if r is not None:
                X.append(r[0]); L.append(r[1]); SH += [sh] * len(r[1]); C += r[2]
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)} листов, {sum(len(x) for x in L)} кусков, {time.time() - t0:.0f} с")
    np.savez_compressed(a.out, X=np.concatenate(X), lv=np.concatenate(L), sheet=np.array(SH), curve=np.array(C))
    print(f"★ кусков {sum(len(x) for x in L)} → {a.out}")


if __name__ == "__main__":
    main()
