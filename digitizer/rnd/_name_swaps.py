r"""_name_swaps.py — ВЕРНО ПРОВЕДЕНЫ, НО НЕВЕРНО НАЗВАНЫ (03.10, к §6.260).

Поле: безымянных честных 1192, именных 861 — разница ≈ 330 кривых проведена верно (1:1 при 3 px), но в выдаче под другим
именем. Здесь — список таких пар (лист, кривая эталона, кривая выдачи) по 1:1 сопоставлению, как приёмка, и сводка: перестановка
внутри трека (A↔B), имя без пары (выдача названа именем, которого в эталоне нет), прочее. Только чтение выдачи и эталона.

  _name_swaps.py --dir F:/nds/output/taskS/rp_vc --mode N --sheets wellmap_sheets.txt --dump F:/nds/output/taskS/name_swaps_field.pkl
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc")
ap.add_argument("--mode", default="N")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--sheets", default="wellmap_sheets.txt")
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
smap = pickle.load(open(TS / a.map, "rb"))
for root in ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_heldout", "pools_all"]:
    for f in sorted((TS / root).glob("*.pkl")):
        if f.stem in smap:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        smap.setdefault(d["name"], {s["name"]: s["track"] for s in d["slots"]})
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
sheets = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]


def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))


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


C = Counter(); ROWS = []
for sh in sheets:
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem
    pd = Path(a.dir) / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    if not got:
        continue
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    for t in {tm.get(g) for g in G if tm.get(g) is not None}:
        gs = [g for g in G if tm.get(g) == t]; ws = [k for k in W if tm.get(k) == t and W[k]]
        ok = {}
        for g in gs:
            for w in ws:
                m, c = st(W[w], G[g])
                ok[(g, w)] = m is not None and m <= 3.0 and c >= 0.9
        mt = match(gs, ws, ok)
        C["честных 1:1"] += len(mt)
        for g, w in mt.items():
            if g == w:
                C["имя верно"] += 1
                continue
            # честна ли выдача С ТЕМ ЖЕ именем, что эталон (другая версия)? переставлены ли A↔B?
            swap = mt.get(w) == g if w in mt else False
            kind = "перестановка A↔B в треке" if swap else ("имя выдачи есть в эталоне трека" if w in gs else "имени выдачи нет в эталоне")
            C[kind] += 1
            ROWS.append(dict(sheet=sh, truth=g, out=w, kind=kind, fam_t=M.mnem_root(g), fam_o=M.mnem_root(w)))
if a.dump:
    pickle.dump(ROWS, open(a.dump, "wb"))
print(f"★ {a.sheets}: честных 1:1 {C['честных 1:1']}, из них имя верно {C['имя верно']}, неверно {C['честных 1:1'] - C['имя верно']}")
for k in ("перестановка A↔B в треке", "имя выдачи есть в эталоне трека", "имени выдачи нет в эталоне"):
    print(f"   {k}: {C[k]}")
fam = Counter((r["fam_t"], r["fam_o"]) for r in ROWS)
print("   частые пары (эталон → выдача): " + ", ".join(f"{a_}→{b_} {n}" for (a_, b_), n in fam.most_common(15)))
same_root = sum(1 for r in ROWS if r["fam_t"] == r["fam_o"])
print(f"   то же семейство (номер/индекс другой): {same_root} из {len(ROWS)}")
