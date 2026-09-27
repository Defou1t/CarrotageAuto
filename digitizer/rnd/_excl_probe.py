r"""_excl_probe.py — ЧТО ДАЁТ ИСКЛЮЧЕНИЕ ЗАНЯТЫХ РАНОВ НА УРОВНЕ ТРАСС (§6.236, 27.09).

Сравнивает два кэша трасс одних листов (`--old` — прод, `--new` — сборка с `--knob trace_exclusive=1`): по каждой эталонной
кривой — есть ли ЧЕСТНЫЙ кандидат (медиана ≤ 3 px, покрытие ≥ 0.9 с мостом ≤ 30 строк) среди трасс прод-пути и среди всех
трасс (прод + декодер); сколько кривых кандидата получили и сколько потеряли; число пар-дублей прод-пути до/после.
Это потолок на уровне кандидатов — в выдачу его переводит раскладка, окончательный счёт только повтором и A/B.

  _excl_probe.py --old F:/nds/output/taskS/tcache --new F:/nds/output/taskS/tcache_excl [--sheets dup_sheets.txt]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--min-rows", type=int, default=200)
a = ap.parse_args()
SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC.setdefault(q.stem, q)


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def hon(tr, gt, bridge=30):
    com = [y for y in gt if y in tr]
    if len(com) < 30 or np.median([abs(tr[y] - gt[y]) for y in com]) > 3.0:
        return False
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
            ok += 1
    return ok / len(gt) >= 0.9


def ndup(lst):
    n = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            (La, A), (Lb, B) = lst[i], lst[j]
            if La.track_index == Lb.track_index and La.color == Lb.color:
                n += sum(1 for y in A if y in B and abs(A[y] - B[y]) <= 3) >= a.min_rows
    return n


C = Counter()
for f in sorted(Path(a.new).glob("*.pkl")):
    fo = Path(a.old) / f.name
    if not fo.exists():
        continue
    o = pickle.load(open(fo, "rb")); n = pickle.load(open(f, "rb"))
    q = SRC.get(Path(n["frame_nlgx"]).stem)
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    po = [(L, un(t)) for L, t in o["traces"]]; pn = [(L, un(t)) for L, t in n["traces"]]
    do = [un(t) for _, t in o["alt"]] if o.get("alt") else []; dn = [un(t) for _, t in n["alt"]] if n.get("alt") else []
    C["листов"] += 1; C["кривых"] += len(G)
    C["пар-дублей прод-пути: было"] += ndup(po); C["пар-дублей прод-пути: стало"] += ndup(pn)
    for g, gt in G.items():
        a1 = any(hon(t, gt) for _, t in po); b1 = any(hon(t, gt) for _, t in pn)
        a2 = a1 or any(hon(t, gt) for t in do); b2 = b1 or any(hon(t, gt) for t in dn)
        C["прод-путь: честный кандидат был"] += a1; C["прод-путь: стал"] += b1
        C["прод-путь: получили"] += b1 and not a1; C["прод-путь: потеряли"] += a1 and not b1
        C["любой путь: был"] += a2; C["любой путь: стал"] += b2
        C["любой путь: получили"] += b2 and not a2; C["любой путь: потеряли"] += a2 and not b2
print("★ ИСКЛЮЧЕНИЕ ЗАНЯТЫХ РАНОВ — КАНДИДАТЫ:")
for k, v in C.items():
    print(f"   {k}: {v}")
