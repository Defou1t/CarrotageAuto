r"""_dup_source.py — ОТКУДА ДУБЛИ: ВЕДЕНИЕ ИЛИ РАСКЛАДКА (§6.236, 27.09).

§6.228: 476 пар выдачи, где две кривые трека идут по одной линии и занимают место невзятой. Если дубль рождает ВЕДЕНИЕ
(две линии U1 одного трека и цвета, трассы прод-пути сходятся на одну нарисованную), лечит исключение занятых ранов
(`cv.trace_exclusive`); если трассы путей различны, а дубль собирает `emit` (слот получает версию, лежащую на чужой линии),
ведение ни при чём. По кэшу `tcache`: пары трасс внутри прод-пути и внутри декодера (тот же трек; в 3 px на ≥ `--min-rows`
общих строк) — сколько листов/пар; и на листах `dup_sheets.txt` — есть ли такая пара хотя бы в одном пути.

  _dup_source.py --cache F:/nds/output/taskS/tcache
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--min-rows", type=int, default=200)
ap.add_argument("--dups", default="dup_sheets.txt")
a = ap.parse_args()
TS = Path(a.ts)
dup = {Path(l.strip()).stem for l in (TS / a.dups).read_text(encoding="utf-8").splitlines() if l.strip()}


def pairs(lst):
    out = 0; same_col = 0
    tr = [(L, dict(zip(t[0].tolist(), t[1].tolist()))) for L, t in lst]
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            (La, A), (Lb, B) = tr[i], tr[j]
            if La.track_index != Lb.track_index:
                continue
            ys = np.array([y for y in A if y in B])
            if len(ys) < a.min_rows:
                continue
            close = int(np.sum(np.abs(np.array([A[y] for y in ys]) - np.array([B[y] for y in ys])) <= 3))
            if close >= a.min_rows:
                out += 1; same_col += La.color == Lb.color
    return out, same_col


C = Counter()
for f in sorted(Path(a.cache).glob("*.pkl")):
    v = pickle.load(open(f, "rb"))
    stem = Path(v["frame_nlgx"]).stem
    pp, pc = pairs(v["traces"])
    dp, dc = pairs(v["alt"]) if v.get("alt") else (0, 0)
    C["листов"] += 1
    C["пар в прод-пути"] += pp; C["из них одного цвета"] += pc; C["листов с парой в прод-пути"] += pp > 0
    C["пар в декодере"] += dp; C["листов с парой в декодере"] += dp > 0
    if stem in dup:
        C["листов «место занято»"] += 1
        C["  из них с парой в прод-пути"] += pp > 0
        C["  из них с парой в декодере"] += dp > 0
        C["  из них без пары ни в одном пути (дубль собран раскладкой)"] += (pp == 0 and dp == 0)
print(f"★ ИСТОЧНИК ДУБЛЕЙ (пара = тот же трек, в 3 px на ≥ {a.min_rows} строк):")
for k, v in C.items():
    print(f"   {k}: {v}")
