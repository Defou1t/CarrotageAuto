r"""_ab_diff.py — ОТКУДА РАЗНИЦА ДВУХ ВЫДАЧ: ПОКРИВОЙ РАЗБОР A/B (§6.244, 28.09).

§6.237: на стенде удержание дало +63 честных трассы декодера (+40%), а A/B на кэше — поле 1184 → 1087 (−97). Здесь для
каждой эталонной кривой поля (треки — `slotmap.pkl`): честна ли она 1:1 в выдаче A и в выдаче B (как `_name_cost_prod`:
медиана ≤ 3 px, покрытие ≥ 0.9, выдача мостится `dense`), и из какого пути записана кривая, которой она сопоставлена
(`_pick.json`: `src` слота; без записи — путь трека `tracks[].dec`). Разрез потерь/прибытков по (путь в A → путь в B) и по
тому, что стало с базой трека (`dec`) и переворотами слотов.

  _ab_diff.py --a F:/nds/output/taskS/rp_ab_old/N --b F:/nds/output/taskS/rp_ab_hold/N
"""
import sys, argparse, pickle, hashlib, json
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
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--sheets", default="wellmap_sheets.txt")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
sheets = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()]
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


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


def side(root, stem, G, tm):
    d = Path(root) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(d.glob("*_auto.nlgx"), None) if d.is_dir() else None
    if not got:
        return None
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}
    pj = next(d.glob("*_pick.json"), None)
    pk = json.loads(pj.read_text(encoding="utf-8")) if pj else {}
    tdec = {t["track"]: t["dec"] for t in pk.get("tracks", [])}
    src = {s["name"]: (s.get("src") or ("dec" if (s["base"] == "prod") == bool(s["flip"]) else "prod")) for s in pk.get("slots", [])}
    src.update(pk.get("override", {}))
    flips = {s["name"]: bool(s["flip"]) for s in pk.get("slots", [])}
    res = {}
    for t in {tm.get(g) for g in G if tm.get(g) is not None}:
        gs = [g for g in G if tm.get(g) == t]; ws = [k for k in W if tm.get(k) == t and W[k]]
        ok = {(g, k): HON(*err(W[k], G[g])) for g in gs for k in ws}
        mt = match(gs, ws, ok)
        for g in gs:
            k = mt.get(g)
            res[g] = dict(hon=k is not None, slot=k,
                          src=(src.get(k) or ("dec" if tdec.get(t) else "prod")) if k else None,
                          tdec=tdec.get(t), flip=flips.get(k) if k else None, track=t)
    return res, tdec, src


C = Counter(); trk = Counter(); rows = []
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    A = side(a.a, q.stem, G, tm); B = side(a.b, q.stem, G, tm)
    if A is None or B is None:
        C["лист без выдачи"] += 1; continue
    (ra, tda, sa), (rb, tdb, sb) = A, B
    for t in set(tda) | set(tdb):
        trk[("база трека", "dec" if tda.get(t) else "prod", "→", "dec" if tdb.get(t) else "prod")] += 1
    for g in ra:
        x, y = ra[g], rb.get(g)
        if y is None:
            continue
        C["кривых"] += 1
        if x["hon"] and not y["hon"]:
            C["потеряна"] += 1
            key = ("потеря", x["src"], "→", "?" if y["slot"] is None else y["src"])
        elif y["hon"] and not x["hon"]:
            C["прибыла"] += 1
            key = ("прибыток", x["src"] if x["slot"] else "—", "→", y["src"])
        else:
            continue
        trkchg = "база трека сменилась" if bool(x["tdec"]) != bool(y["tdec"]) else "база та же"
        C[key + (trkchg,)] += 1
        rows.append((sh, g, x, y))
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)
print(f"★ ПОКРИВОЙ РАЗБОР A/B: A = {a.a}, B = {a.b}")
print(f"   кривых {C['кривых']}, потеряно {C['потеряна']}, прибыло {C['прибыла']} (Δ {C['прибыла'] - C['потеряна']:+d}); листов без выдачи {C['лист без выдачи']}")
for k, v in sorted(((k, v) for k, v in C.items() if isinstance(k, tuple)), key=lambda kv: -kv[1]):
    print(f"   {' '.join(str(z) for z in k)}: {v}")
print("   база треков (A → B): " + ", ".join(f"{k[1]}→{k[3]} {v}" for k, v in trk.most_common()))
if a.dump:
    pickle.dump(rows, open(a.dump, "wb"))
