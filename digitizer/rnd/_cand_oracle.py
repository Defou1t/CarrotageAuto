r"""_cand_oracle.py — ПОТОЛОК ВЫБОРА: ЕСТЬ ЛИ СРЕДИ ВСЕХ ТРАСС ЛИСТА ТА, ЧТО ЧЕСТНО ИДЁТ ПО ПРОПУЩЕННОЙ КРИВОЙ (§6.219, 25.09).

`_drift_anatomy.py`: не взятые кривые многокривых треков уходят на ДРУГУЮ нарисованную тушь (81%); в 72% прыжков своя
линия продолжалась — выбор, а не разрыв. Вопрос, который решает, куда идти дальше: у пропущенной кривой ЕСТЬ ли честная
трасса среди КАНДИДАТОВ (все трассы прод-пути по линиям U1 и все трассы декодера из кэша `_trace_cache.py`, до раскладки
и отбора K) — тогда потеря в ВЫБОРЕ/раскладке и лечится повтором `emit` на кэше; нет — потеря в ВЕДЕНИИ.

Честность кандидата — как в счёте: медиана |Δx| ≤ 3 px по общим строкам и покрытие ≥ 0.9, где покрытие считается с мостом
пропусков ≤ `--bridge` строк (выдача `emit` пропуски закрывает; сырая трасса — нет).
Счёт выдачи — по файлу `--dir` (по умолчанию прод RA) тем же сопоставлением 1:1.

  _cand_oracle.py --cache F:/nds/output/taskS/tcache [--dir ab_slot/RA] [--every N]
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", required=True)
ap.add_argument("--dir", default="ab_slot/RA")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--bridge", type=int, default=30)
ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()
TS = Path(a.ts)


def unpack(t):
    rows, xs = t
    return dict(zip(rows.tolist(), xs.tolist()))


def hon_raw(tr, gt, bridge):
    com = [y for y in gt if y in tr]
    if len(com) < 30:
        return False
    if np.median([abs(tr[y] - gt[y]) for y in com]) > 3.0:
        return False
    rs = np.array(sorted(tr))
    ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
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
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC.setdefault(q.stem, q)
smap = pickle.load(open(TS / a.map, "rb"))
files = sorted(Path(a.cache).glob("*.pkl"))[::a.every]
C = Counter(); CM = Counter()
for i, f in enumerate(files, 1):
    v = pickle.load(open(f, "rb"))
    stem = Path(v["stem"]).stem if v.get("stem") else None
    q = SRC.get(Path(v["frame_nlgx"]).stem)
    if not q:
        C["нет эталона"] += 1; continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    if not gts:
        continue
    prod = [unpack(t) for _, t in v["traces"]]
    dec = [unpack(t) for _, t in v["alt"]] if v.get("alt") else []
    # выдача
    pd = TS / a.dir / f.stem
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"} if got else {}
    ok = {(g, k): HON(*st(W[k], gts[g])) for g in gts for k in W if W[k]}
    mt = match(list(gts), [k for k in W if W[k]], ok)
    tm = smap.get(q.name, {})
    ntr = Counter(tm.get(g) for g in gts)
    for g, gt in gts.items():
        multi = ntr[tm.get(g)] >= 2 if tm.get(g) is not None else False
        inp = any(hon_raw(t, gt, a.bridge) for t in prod)
        ind = any(hon_raw(t, gt, a.bridge) for t in dec)
        taken = g in mt
        key = ("взята выдачей" if taken else
               "НЕ взята; честный кандидат ЕСТЬ (прод)" if inp and not ind else
               "НЕ взята; честный кандидат ЕСТЬ (декодер)" if ind and not inp else
               "НЕ взята; честный кандидат ЕСТЬ (оба)" if inp and ind else
               "НЕ взята; честного кандидата НЕТ")
        C[key] += 1; C["кривых"] += 1
        if multi:
            CM[key] += 1; CM["кривых"] += 1
    if i % 100 == 0:
        print(f"  … {i}/{len(files)}", file=sys.stderr)

for name, D in (("ВСЕ КРИВЫЕ", C), ("МНОГОКРИВЫЕ ТРЕКИ", CM)):
    n = D["кривых"]
    print(f"\n★★ {name}: {n}")
    for k in ("взята выдачей", "НЕ взята; честный кандидат ЕСТЬ (прод)", "НЕ взята; честный кандидат ЕСТЬ (декодер)",
              "НЕ взята; честный кандидат ЕСТЬ (оба)", "НЕ взята; честного кандидата НЕТ"):
        print(f"   {k}: {D[k]} ({100*D[k]/max(1,n):.1f}%)")
if C["нет эталона"]:
    print(f"   (листов без эталона: {C['нет эталона']})")
