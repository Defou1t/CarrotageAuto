r"""_slot_order_x.py — §6.364 РАЗВЕДКА: держится ли порядок слотов трека (номер SA в имени) против порядка линий эталона по x (13.11).

Для каждой пары кривых эталона одного трека (`_sample_review.load_tracks`) — на общих строках доля строк, где кривая с меньшим SA левее
(x меньше) другой. Если доля устойчиво далека от 0.5 (по семействам пары), порядок слотов — признак «чья линия» для дублей выбора трассы.

  _slot_order_x.py
"""
import sys, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import defaultdict
import numpy as np
import _sample_review as SR
from extract_nlgx import extract
from _multi_replica_probe import dense
from auto import meta as M


def sa(n):
    m = re.search(r"SA(\d+)", n)
    return int(m.group(1)) if m else None


def main():
    TR = SR.load_tracks(); S = SR.eval_sets()
    sheets = set().union(*S.values())
    SRC = {q.name: q for q in SR.ARCH.glob("*/wlg/*.nlgx")}
    fr = []; byfam = defaultdict(list); n = 0
    for nm in sorted(sheets):
        if nm not in SRC:
            continue
        try:
            m = extract(str(SRC[nm]))
        except Exception:
            continue
        tm = TR.get(nm, {})
        C = [(c["name"], dense(c)) for c in m["curves"] if M.mnem_root(c["name"]) != "DA" and sa(c["name"]) is not None]
        C = [(a, d) for a, d in C if len(d) >= 200]
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                (na, da), (nb, db) = C[i], C[j]
                if tm.get(na) is None or tm.get(na) != tm.get(nb) or na.split()[1] != nb.split()[1]:   # тот же трек и та же ось DA
                    continue
                com = [y for y in da if y in db]
                if len(com) < 200:
                    continue
                lo, hi = (da, db) if sa(na) < sa(nb) else (db, da)
                f = float(np.mean([lo[y] < hi[y] for y in com]))
                fr.append(f)
                fa, fb = sorted([M.mnem_root(na), M.mnem_root(nb)])
                byfam[(fa, fb) if fa != fb else (fa, "=")].append(f)
        n += 1
    fr = np.array(fr)
    print(f"★ листов {n}, пар кривых одного трека {len(fr)}: доля строк «меньший SA левее» — медиана {np.median(fr):.2f}; "
          f"≥ 0.9: {np.mean(fr >= 0.9):.1%}, ≤ 0.1: {np.mean(fr <= 0.1):.1%}, между: {np.mean((fr > 0.1) & (fr < 0.9)):.1%}")
    print("★ по парам семейств (≥ 20 пар): доля пар, где порядок держится ≥ 0.9 / обратный ≤ 0.1")
    for k, v in sorted(byfam.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 20:
            continue
        v = np.array(v)
        print(f"   {k[0]}–{k[1]:8s} пар {len(v):4d}: ≥ 0.9 {np.mean(v >= 0.9):5.1%}, ≤ 0.1 {np.mean(v <= 0.1):5.1%}, медиана {np.median(v):.2f}")


if __name__ == "__main__":
    main()
