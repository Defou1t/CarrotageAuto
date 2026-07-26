r"""seq_sweep.py — какой комбинацией (критерий, мостик, dedup) воспроизводятся цифры §6.66.

§6.66 заявляет на ГЕЙТЕ: жадный 22 -> обученный 39; на валидации 27 -> 31; на третьем 17 -> 32.
Прогон с nl+npts/bridge=20/dedup=50 дал 12 -> 28. Ищем, что именно давало 22/39.
"""
import sys, pickle, itertools
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.argv = ["_pick_gate.py"]
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")

import _pick_gate as G
from pathlib import Path

ROOT = Path(r"F:\nds\output\taskS\pick_gate")
PAIRS = [("гейт", "cache", "seq_gate", 22, 39), ("валид", "holdout", "seq_hold", 27, 31),
         ("трети", "wide", "seq_wide", 17, 32)]

raw = {}
for _, dg, dsq, _, _ in PAIRS:
    for d in (dg, dsq):
        if d in raw:
            continue
        raw[d] = {f.stem: pickle.load(open(f, "rb")) for f in sorted((ROOT / d).glob("*.pkl"))}

print("загружено:", {k: len(v) for k, v in raw.items()})


def total(dirname, names, pick, bridge, dedup):
    G.a.bridge = bridge
    G.a.dedup_tol = dedup
    s = 0
    for nm in names:
        dd = raw[dirname][nm]
        cand = [G.bridge(t, bridge) for t in dd["traces"] if len(t) >= G.MINPTS]
        K = max(1, dd["K"] or len(dd["GM"]))
        s += G.score(G.PICKERS[pick](cand, K, G.a), dd["GM"], dd["raw"])[0]
    return s


PICKS = ["oracle", "depth", "npts", "dens", "nolatch", "nl+npts", "nl+dens", "nl+loc"]
hits = []
print(f"\n{'критерий':<10}{'мост':>5}{'ddup':>6} | " +
      "".join(f"{lb:>16}" for lb, *_ in PAIRS))
for pick, bridge, dedup in itertools.product(PICKS, [0, 20, 200], [50]):
    line = f"{pick:<10}{bridge:>5}{dedup:>6} | "
    ok = 0
    for lb, dg, dsq, eg, es in PAIRS:
        names = sorted(set(raw[dg]) & set(raw[dsq]))
        g = total(dg, names, pick, bridge, dedup)
        s = total(dsq, names, pick, bridge, dedup)
        mark = "  <<" if (g == eg and s == es) else ""
        ok += (g == eg and s == es)
        line += f"{g:>6}->{s:<6}{mark:<4}"
    print(line + ("   ★СОВПАЛО" if ok else ""))
    if ok:
        hits.append((pick, bridge, dedup, ok))

print("\nсовпадения:", hits or "НЕТ ТОЧНЫХ СОВПАДЕНИЙ")
