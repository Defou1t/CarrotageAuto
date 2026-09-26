r"""_dup_traces.py — ДВЕ КРИВЫЕ ВЫДАЧИ НА ОДНОЙ ЛИНИИ: КАК ЧАСТО И СКОЛЬКО ЭТО СТОИТ (§6.228, 26.09).

Для треков с ≥ 2 кривыми выдачи (повтор прода с кэша, `--dir`): пары кривых одного трека, идущие в 3 px друг от друга
на ≥ `--min-rows` общих строк, — «дубли». Для каждой такой пары: сколько эталонов трека взято (честно 1:1), и есть ли в
треке НЕ взятый эталон, — тогда одна из двух кривых занимает чужое место (кандидат для запрета «две кривые на одной туши»).

  _dup_traces.py --dir F:/nds/output/taskS/rp_fill --mode NF
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_fill")
ap.add_argument("--mode", default="NF")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--min-rows", type=int, default=200)
a = ap.parse_args()
TS = Path(a.ts)
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
field = [l.strip() for l in (TS / "wellmap_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


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


C = Counter(); rows_dup = 0; ex = []
for sh in field:
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
    for t in {tm.get(k) for k in W if tm.get(k) is not None}:
        ws = [k for k in W if tm.get(k) == t and W[k]]
        gs = [g for g in G if tm.get(g) == t]
        if len(ws) < 2:
            continue
        C["треков с ≥ 2 кривыми выдачи"] += 1
        ok = {(g, k): HON(*st(W[k], G[g])) for g in gs for k in ws}
        mt = match(gs, ws, ok)
        taken = set(mt.values()); lost = [g for g in gs if g not in mt]
        dup_pairs = []
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                A, B = W[ws[i]], W[ws[j]]
                com = [y for y in A if y in B]
                close = sum(1 for y in com if abs(A[y] - B[y]) <= 3)
                if close >= a.min_rows:
                    dup_pairs.append((ws[i], ws[j], close))
        if not dup_pairs:
            continue
        C["треков с дублями"] += 1
        for k1, k2, n in dup_pairs:
            C["пар-дублей"] += 1; rows_dup += n
            both = k1 in taken and k2 in taken
            one = (k1 in taken) != (k2 in taken)
            C["пара: обе кривые взяты (дубль безвреден/совпадение эталонов)"] += both
            C["пара: взята одна, в треке есть НЕ взятый эталон — место занято"] += one and bool(lost)
            C["пара: взята одна, НЕ взятых эталонов нет"] += one and not lost
            C["пара: не взята ни одна"] += (k1 not in taken and k2 not in taken)
            if one and lost and len(ex) < 12:
                ex.append((sh[:44], t, k1, k2, n, lost[:2]))
print(f"★ поле: {len(field)} листов")
for k, v in C.items():
    print(f"   {k}: {v}")
print(f"   строк в дублях (сумма по парам): {rows_dup}")
print("★ примеры «место занято»:")
for e in ex:
    print("   ", e)
