"""СЕЛЕКЦИЯ ТУШИ, шаг 2: «лишняя» тушь = кривые ФАЙЛА-БРАТА того же скана.
Замечено 19.07: один физический планшет цифруется НЕСКОЛЬКИМИ nlgx, каждый берёт СВОЙ набор
кривых (BOGAT_011: «BKZ, DS»→GZ41,GZ51,CALI1,SP1 и «BKZ»→GZ11,GZ21,GZ31,OGZ1 — одна и та же
картинка, интервал и дата). Значит «лишние» раны — не мусор, а кривые, которых просто нет
в ЭТОЙ рамке.

Меряем: сколько ранов объясняет свой файл, сколько добавляют братья (та же картинка), сколько
остаётся. Если союз объясняет почти всё — задача селекции переформулируется: не «отличить
кривую от мусора», а «сопоставить K слотов рамки K из N физических кривых».

  python _multi_sibling_probe.py <nlgx> [<nlgx> ...] [--tol 8]
"""
import sys
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
args = [x for x in sys.argv[1:] if not x.startswith("--")]
TOL = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 8.0
p = Config().cv


def real_gts(mo):
    return [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
            and M.mnem_root(c["name"]) != "DA"]


for arg in args:
    n = Path(arg)
    img = find_image(n)
    # БРАТЬЯ = другие nlgx той же папки, указывающие на ТУ ЖЕ картинку
    sibs = [q for q in sorted(n.parent.glob("*.nlgx"))
            if q != n and "_auto" not in q.stem and find_image(q) == img]
    mo = extract(str(n))
    m = M.parse_filename(n.name, MN)
    rgb = im.load_rgb(str(img))
    fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
    fg = im.ink_foreground(rgb, p)
    gts = real_gts(mo)
    tid = Counter(E._slot_track(mo, c, fr) for c in gts)
    best = tid.most_common(1)[0][0]
    t = fr.tracks[best if best is not None else 0]
    lo, hi = int(t.x_left) + 3, int(t.x_right) - 3
    own = [dense(c) for c in gts]
    sib = []
    for q in sibs:
        try:
            sib += [dense(c) for c in real_gts(extract(str(q)))]
        except Exception:
            pass
    print(f"\n{n.stem[:50]}")
    print(f"  свои кривые ({len(own)}): {','.join(c['name'].split()[0] for c in gts)}")
    print(f"  файлов-братьев на ту же картинку: {len(sibs)} → ещё {len(sib)} кривых"
          + (f"  [{', '.join(q.stem[:34] for q in sibs)}]" if sibs else ""))
    n_own = n_sib = n_un = 0
    step = max(1, (int(fr.bottom_y) - int(fr.top_y)) // 2500)
    for y in range(int(fr.top_y), int(fr.bottom_y), step):
        rr = im.row_runs(fg[y, lo:hi], gap=4)
        if not rr:
            continue
        for r in rr:
            c = lo + r[2]
            if any(y in d and abs(d[y] - c) <= TOL for d in own):
                n_own += 1
            elif any(y in d and abs(d[y] - c) <= TOL for d in sib):
                n_sib += 1
            else:
                n_un += 1
    tot = max(1, n_own + n_sib + n_un)
    print(f"  раны: своя рамка {100*n_own/tot:3.0f}% | братья {100*n_sib/tot:3.0f}% | "
          f"НЕ объяснено {100*n_un/tot:3.0f}%   (всего {tot}, tol={TOL:.0f}px)")
