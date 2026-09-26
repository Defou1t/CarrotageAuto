r"""_lost_honest.py — ПОЧЕМУ 470 КРИВЫХ С ЧЕСТНОЙ ТРАССОЙ-КАНДИДАТОМ НЕ ПОПАДАЮТ В ВЫДАЧУ (§6.229, 26.09).

§6.228: на всём кэше у 470 не взятых кривых честная трасса есть среди кандидатов (прод по линиям U1 или декодер). Здесь —
механизм потери для каждой. Для не взятой кривой g с честным кандидатом c (линия L, путь prod/dec):
  1. «чужой трек» — L.track_index ≠ трек g по `slotmap` (раскладка кладёт линию в слоты ДРУГОГО трека);
  2. «выдана, но не прошла» — в выдаче трека есть кривая в ≤ 3 px от c по медиане (линия выдана; покрытие окна слота < 0.9
     или 1:1 отдало её другой кривой);
  3. «версия другого пути» — выдача трека совпадает с c на ≥ 30% общих строк, но в целом ушла (выдана версия второго пути);
  4. «не выдана — нет слота» — ни одна кривая выдачи трека не идёт по c: раскладка не дала линии слот.
Проверка согласованности индексов треков: для ВЗЯТЫХ кривых — доля, где честный кандидат лежит в треке своей кривой.

  _lost_honest.py --cache F:/nds/output/taskS/tcache --dir rp_fill/NF
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
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--dir", default="rp_fill/NF")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--bridge", type=int, default=30)
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)


def unpack(t):
    rows, xs = t
    return dict(zip(rows.tolist(), xs.tolist()))


def med(tr, gt):
    com = [y for y in gt if y in tr]
    return (float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com)) if len(com) >= 30 else (None, 0)


def hon_raw(tr, gt):
    m, n = med(tr, gt)
    if m is None or m > 3.0:
        return False
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= a.bridge:
            ok += 1
    return ok / len(gt) >= 0.9


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def match(rows, cols, ok):
    pair = {}

    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen:
                continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r
                return True
        return False
    for r in rows:
        try_(r, set())
    return {r: c for c, r in pair.items()}


SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC.setdefault(q.stem, q)
smap = pickle.load(open(TS / a.map, "rb"))
C = Counter(); K = Counter(); REC = []
files = sorted(Path(a.cache).glob("*.pkl"))
for i, f in enumerate(files, 1):
    v = pickle.load(open(f, "rb"))
    q = SRC.get(Path(v["frame_nlgx"]).stem)
    if not q:
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not gts:
        continue
    cands = [("prod", L, unpack(t)) for L, t in v["traces"]] + ([("dec", L, unpack(t)) for L, t in v["alt"]] if v.get("alt") else [])
    pd = TS / a.dir / f.stem
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"} if got else {}
    tm = smap.get(q.name, {})
    ok = {(g, k): HON(*st(W[k], gts[g])) for g in gts for k in W if W[k]}
    mt = match(list(gts), [k for k in W if W[k]], ok)
    for g, gt in gts.items():
        tg = tm.get(g)
        hon = [(pth, L, tr) for pth, L, tr in cands if hon_raw(tr, gt)]
        if g in mt:
            if hon:
                K["взятые с кандидатом"] += 1
                K["из них кандидат в треке своей кривой"] += any(getattr(L, "track_index", None) == tg for _, L, _ in hon)
            continue
        if not hon:
            continue
        C["не взята, честный кандидат есть"] += 1
        same = [(pth, L, tr) for pth, L, tr in hon if getattr(L, "track_index", None) == tg]
        if not same:
            cat = "1 чужой трек (линия кандидата в другом треке)"
        else:
            pth, L, tr = same[0]
            outs = [k for k in W if tm.get(k) == tg and W[k]]
            best = None; part = False
            for k in outs:
                m, n = med(W[k], tr)
                if m is not None and m <= 3.0:
                    best = k
                com = [y for y in tr if y in W[k]]
                if com and sum(1 for y in com if abs(W[k][y] - tr[y]) <= 3) >= 0.3 * len(com) and len(com) >= 30:
                    part = True
            if best:
                cat = "2 выдана, но не прошла (окно слота / 1:1)"
            elif part:
                cat = "3 выдана версия другого пути (частично та же линия)"
            else:
                cat = "4 не выдана — раскладка не дала линии слот"
            C[f"   путь кандидата {pth}"] += 1
            if getattr(L, "flag_reason", None) == "kslots-synth":
                C["   кандидат — синтетическая линия kslots"] += 1
        C[cat] += 1
        n_slots = sum(1 for k in gts if tm.get(k) == tg)
        n_out = sum(1 for k in W if tm.get(k) == tg and W[k])
        REC.append((q.name, g, cat, n_slots, n_out))
    if i % 200 == 0:
        print(f"  … {i}/{len(files)}", file=sys.stderr)
print(f"★ согласованность треков: взятых с кандидатом {K['взятые с кандидатом']}, из них кандидат в треке своей кривой "
      f"{K['из них кандидат в треке своей кривой']} ({100*K['из них кандидат в треке своей кривой']/max(1,K['взятые с кандидатом']):.0f}%)")
n = C["не взята, честный кандидат есть"]
print(f"★★ НЕ ВЗЯТЫ, НО ЧЕСТНЫЙ КАНДИДАТ ЕСТЬ: {n}")
for k in sorted(k for k in C if k[0].isdigit()):
    print(f"   {k}: {C[k]} ({100*C[k]/max(1,n):.0f}%)")
for k in sorted(k for k in C if k.startswith("   ")):
    print(f"{k}: {C[k]}")
fill = Counter((r[2][:1], "слотов > выдано" if r[3] > r[4] else "все слоты заняты") for r in REC)
print("★ по заполнению слотов трека: " + ", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in sorted(fill.items())))
if a.dump:
    pickle.dump(REC, open(a.dump, "wb"))
