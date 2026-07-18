"""ШИРОКИЙ ГЕЙТ «ML НА ФОН» на мульти-листах: сравнить ПЕРЕДНИЙ ПЛАН правил vs правила+recall-модель.
Метрика — не dice, а то, что нужно трассировке: (1) доля точек экспертной трассы, под которыми
ЕСТЬ чернила (recall переднего плана), (2) ранов на строку (цена: модель добавляет и лишнего).

  # 1) выгрузить пути картинок и посчитать prob-карты (venv ComfyUI):
  python digitizer/rnd/_multi_fg_gate.py --dump-images
  <ComfyUI>\\python_embeded\\python.exe digitizer\\rnd\\_prob_batch.py <ckpt> <img...> --out <DIR>
  # 2) сравнить:
  python digitizer/rnd/_multi_fg_gate.py --prob <DIR>

⚠ Экспертная трасса = ВЕРШИНЫ полилинии (4-14% строк) — интерполируем через dense().
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import imaging as im, meta as M, frame as F, emit as E
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=12)
ap.add_argument("--tok", default="BK, IK|GK, NGK|BKZ, DS|MK, MBK, MDS|MBK, MDS, MK|STK+DS|BK+IK")
ap.add_argument("--per-well", type=int, default=1)
ap.add_argument("--prob", default="")
ap.add_argument("--dump-images", action="store_true")
a = ap.parse_args()
TOKS = [t.strip().upper() for t in a.tok.split("|")]
p = Config().cv

cands, per_well = [], {}
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    well = wlg.parent.name
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem or per_well.get(well, 0) >= a.per_well:
            continue
        m = M.parse_filename(n.name, MN)
        if m.curves_token.strip().upper() not in TOKS:
            continue
        img = find_image(n)
        if not img:
            continue
        cands.append((well, n, img))
        per_well[well] = per_well.get(well, 0) + 1
cands = cands[:a.n]

if a.dump_images:
    for _, _, img in cands:
        print(f'"{img}"')
    sys.exit(0)

PROBDIR = Path(a.prob) if a.prob else None
rows = []
for well, n, img in cands:
    mo = extract(str(n))
    m = M.parse_filename(n.name, MN)
    rgb = im.load_rgb(str(img))
    fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
    gts = [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
           and M.mnem_root(c["name"]) != "DA"]
    if not gts:
        continue
    tid = Counter(E._slot_track(mo, c, fr) for c in gts)
    best = tid.most_common(1)[0][0]
    gts = [c for c in gts if E._slot_track(mo, c, fr) == best]
    t = fr.tracks[best if best is not None else 0]
    lo, hi = int(t.x_left) + 3, int(t.x_right) - 3
    prob = None
    if PROBDIR is not None:
        np_path = PROBDIR / f"{Path(img).stem}_prob.npy"
        if np_path.is_file():
            from auto.prob import prob_from_npy
            prob = prob_from_npy(np_path)(rgb)
    variants = {"правила": im.ink_foreground(rgb, p)}
    if prob is not None:
        variants["правила+ML"] = im.ink_foreground(rgb, p, prob=prob)
    ser = [dense(c) for c in gts]
    step = max(1, (int(fr.bottom_y) - int(fr.top_y)) // 2000)
    out = {}
    for tag, fg in variants.items():
        H, W = fg.shape
        hit = miss = 0
        runs_per_row = []
        for y in range(int(fr.top_y), int(fr.bottom_y), step):
            if not (0 <= y < H):
                continue
            rr = im.row_runs(fg[y, lo:hi], gap=4)
            if rr:
                runs_per_row.append(len(rr))
            cents = [lo + r[2] for r in rr]
            for s in ser:
                if y not in s:
                    continue
                if cents and min(abs(c - s[y]) for c in cents) <= 6:
                    hit += 1
                else:
                    miss += 1
        out[tag] = (hit / max(1, hit + miss), float(np.median(runs_per_row or [0])))
    line = f"{well:<14} {m.curves_token:<12} K={len(gts)} "
    for tag, (cov, med) in out.items():
        line += f"| {tag}: GT-покрытие {cov*100:3.0f}% ранов/стр {med:4.1f} "
    print(line)
    rows.append(out)

if rows:
    print("\n=== СВОДКА ===")
    for tag in rows[0]:
        cov = np.mean([r[tag][0] for r in rows if tag in r])
        rpr = np.median([r[tag][1] for r in rows if tag in r])
        print(f"  {tag:<12} GT-покрытие сред {cov*100:.0f}%   ранов/строку медиана {rpr:.1f}   "
              f"(листов {sum(1 for r in rows if tag in r)})")
