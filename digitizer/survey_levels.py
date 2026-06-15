r"""
survey_levels.py — характеризация ПЕРЕВЫНОСОВ по ground-truth всего Archive.
Для каждой реальной кривой: сегменты-уровни (тег 35498) + цепочка scale-семейства
(idx/next 35272/35273). Цель — понять структуру перед проектированием декодера уровней:
насколько часты перевыносы, сколько переходов, какие множители (×5/×2/×10), по мнемоникам.

python survey_levels.py [--limit N]
"""
import sys, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds

ARCHIVE = Path(r"F:\nds\projects\Archive")


def scale_families(m):
    """{idx: scale} -> список цепочек по next (35273)."""
    by_idx = {s["idx"]: s for s in m["scale_axes"] if s.get("idx") is not None}
    chains = []
    seen = set()
    for s in m["scale_axes"]:
        i = s.get("idx")
        if i is None or i in seen:
            continue
        chain = []
        cur = i
        while cur is not None and cur in by_idx and cur not in seen:
            seen.add(cur); chain.append(by_idx[cur])
            nx = by_idx[cur].get("next")
            cur = nx if (nx is not None and nx != NULL) else None
        if chain:
            chains.append(chain)
    return chains


def mnem(name):
    return re.sub(r"\d+$", "", name.split()[0])  # BK1 -> BK, GZ21 -> GZ


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit")+1])
    sys.stdout.reconfigure(encoding="utf-8")
    files = []
    for wlg in sorted(ARCHIVE.glob("*/wlg")):
        for n in sorted(wlg.glob("*.nlgx")):
            if "_auto" not in n.stem:
                files.append(n)
    if limit:
        files = files[:limit]

    tot_curves = 0
    multilevel = 0
    maxlevel_hist = Counter()
    ntrans_hist = Counter()
    fam_size_hist = Counter()       # размер цепочки scale-семейства
    mult_hist = Counter()           # множитель между соседними шкалами семейства (округл.)
    by_mnem = defaultdict(lambda: [0, 0])  # mnem -> [curves, multilevel]
    err = 0
    for n in files:
        try:
            m = extract(str(n))
        except Exception:
            err += 1; continue
        chains = scale_families(m)
        for ch in chains:
            fam_size_hist[len(ch)] += 1
            for a, b in zip(ch, ch[1:]):
                sa = (a["v_right"] - a["v_left"]) or 1e-9
                sb = (b["v_right"] - b["v_left"])
                r = abs(sb / sa) if sa else 0
                mult_hist[round(r)] += 1
        for c in ds.real_curves(m):
            tot_curves += 1
            mm = mnem(c["name"]); by_mnem[mm][0] += 1
            levels = [l for _, _, l in c["segments"]]
            ml = max(levels) if levels else 0
            maxlevel_hist[ml] += 1
            ntr = sum(1 for a, b in zip(levels, levels[1:]) if a != b)
            ntrans_hist[ntr] += 1
            if ml > 0:
                multilevel += 1; by_mnem[mm][1] += 1

    print(f"файлов={len(files)} (ошибок extract={err}); кривых={tot_curves}")
    print(f"кривых с перевыносами (maxlevel>0): {multilevel} ({100*multilevel/max(1,tot_curves):.1f}%)")
    print(f"\nmaxlevel: " + ", ".join(f"L{k}={v}" for k, v in sorted(maxlevel_hist.items())))
    print(f"переходов уровня/кривую: " + ", ".join(f"{k}:{v}" for k, v in sorted(ntrans_hist.items())[:8]))
    print(f"размер scale-семейства (цепочка next): " + ", ".join(f"{k}:{v}" for k, v in sorted(fam_size_hist.items())))
    print(f"множитель между соседними шкалами: " + ", ".join(f"x{k}:{v}" for k, v in sorted(mult_hist.items()) if v > 2))
    print(f"\nпо мнемоникам (кривых / с перевыносами), топ-20 по доле:")
    rows = [(mm, c, ml) for mm, (c, ml) in by_mnem.items() if c >= 5]
    rows.sort(key=lambda r: -(r[2]/r[1]))
    for mm, c, ml in rows[:20]:
        print(f"   {mm:<8} {c:>5} / {ml:>4}  ({100*ml/c:.0f}%)")


if __name__ == "__main__":
    main()
