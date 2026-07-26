r"""seq_heldout.py — цифра селектора ТОЛЬКО на листах, которых не было в обучении.

`_decoder_data.HELD` держит вне обучения 5 скважин, а замер §6.66/§6.69 шёл по 66 листам, из
которых 59 попали в обучающую выборку. «Три независимых набора» независимы друг от друга, но не
от модели. Здесь считается то же самое на непересекающейся части.
"""
import sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.argv = ["_pick_gate.py"]
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import _pick_gate as G
from _decoder_data import train_sheets, HELD
from pathlib import Path

ROOT = Path(r"F:\nds\output\taskS\pick_gate")
G.a.bridge = 20
G.a.dedup_tol = 50
TRAIN = {Path(str(s[0] if isinstance(s, (tuple, list)) else s)).stem for s in train_sheets(10**6)}
print(f"обучающих листов {len(TRAIN)}; HELD-скважины {sorted(HELD)}\n")

PAIRS = [("гейт", "cache", "seq_gate"), ("валидация", "holdout", "seq_hold"),
         ("третий", "wide", "seq_wide")]

for pick in ("oracle", "nl+npts"):
    print(f"{'='*96}\nКРИТЕРИЙ {pick}")
    agg = {"в обучении": [0, 0, 0, 0], "НЕ в обучении": [0, 0, 0, 0]}   # greedy, seq, up, down
    for label, dg, dsq in PAIRS:
        names = sorted({f.stem for f in (ROOT / dg).glob("*.pkl")}
                       & {f.stem for f in (ROOT / dsq).glob("*.pkl")})
        for nm in names:
            got = []
            for d in (dg, dsq):
                dd = pickle.load(open(ROOT / d / f"{nm}.pkl", "rb"))
                cand = [G.bridge(t, 20) for t in dd["traces"] if len(t) >= G.MINPTS]
                K = max(1, dd["K"] or len(dd["GM"]))
                got.append(G.score(G.PICKERS[pick](cand, K, G.a), dd["GM"], dd["raw"])[0])
            key = "в обучении" if nm in TRAIN else "НЕ в обучении"
            agg[key][0] += got[0]; agg[key][1] += got[1]
            agg[key][2] += got[1] > got[0]; agg[key][3] += got[1] < got[0]
            if key == "НЕ в обучении":
                print(f"   [{label:<9}] жадный {got[0]:>2} -> селектор {got[1]:>2}"
                      f"  ({got[1]-got[0]:+d})   {nm[:62]}")
    print()
    for key, (g, s, up, dn) in agg.items():
        pct = f"{100*(s-g)/g:+.0f}%" if g else "—"
        print(f"  {key:<14} жадный {g:>3} -> селектор {s:>3}  ({s-g:+d}, {pct})"
              f"   вверх {up}, вниз {dn}")
    print()
