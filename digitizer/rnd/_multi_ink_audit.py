"""ЧТО ЛИШНЕГО в чернилах мульти-листа: распределение числа ранов на строку vs K и КУДА
попадают раны, не объяснённые GT-кривыми (гистограмма x лишних ранов по долям ширины трека).
Отвечает на вопрос «почему ранговое присвоение не работает».
  python ink_audit.py <nlgx> [<nlgx> ...]"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from auto import imaging as im, meta as M, frame as F, emit as E
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
p = Config().cv
for arg in sys.argv[1:]:
    n = Path(arg)
    img = find_image(n)
    mo = extract(str(n))
    m = M.parse_filename(n.name, MN)
    rgb = im.load_rgb(str(img))
    fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
    fg = im.ink_foreground(rgb, p)
    gts = [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
           and M.mnem_root(c["name"]) != "DA"]
    tid = Counter(E._slot_track(mo, c, fr) for c in gts)
    best = tid.most_common(1)[0][0]
    gts = [c for c in gts if E._slot_track(mo, c, fr) == best]
    t = fr.tracks[best if best is not None else 0]
    ser = [{c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL} for c in gts]
    K = len(gts)
    lo, hi = int(t.x_left) + 3, int(t.x_right) - 3
    cnt = Counter(); extra_rel = []; unmatched_gt = 0; matched = 0
    step = max(1, (int(fr.bottom_y) - int(fr.top_y)) // 3000)
    for y in range(int(fr.top_y), int(fr.bottom_y), step):
        runs = im.row_runs(fg[y, lo:hi], gap=4)
        if not runs:
            continue
        cnt[len(runs)] += 1
        cents = [lo + r[2] for r in runs]
        gxs = [s[y] for s in ser if y in s]
        used = set()
        for g in gxs:
            k = int(np.argmin([abs(c - g) for c in cents]))
            if abs(cents[k] - g) <= 6:
                used.add(k); matched += 1
            else:
                unmatched_gt += 1
        for k, c in enumerate(cents):
            if k not in used:
                extra_rel.append((c - t.x_left) / max(1, t.width))
    tot = sum(cnt.values())
    print(f"\n{n.stem[:52]}  K={K} ({','.join(c['name'].split()[0] for c in gts)}) трек[{lo}..{hi}] w={hi-lo}")
    print("  ранов/строку:", ", ".join(f"{k}:{100*v/tot:.0f}%" for k, v in sorted(cnt.items())[:9]),
          f"| медиана {np.median([k for k,v in cnt.items() for _ in range(v)]):.0f}")
    print(f"  GT-точек нашли ран ≤6px: {matched}, НЕ нашли: {unmatched_gt} "
          f"({100*unmatched_gt/max(1,matched+unmatched_gt):.0f}% чернил под GT нет)")
    if extra_rel:
        h, _ = np.histogram(extra_rel, bins=10, range=(0, 1))
        print(f"  ЛИШНИХ ранов {len(extra_rel)}, по десятым ширины трека: {list(h*100//max(1,len(extra_rel)))}%")
