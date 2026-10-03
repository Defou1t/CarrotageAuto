r"""_snap_whatif.py — ПРИТЯЖКА ВЫДАЧИ К ЦЕНТРУ ШТРИХА ПО СТРОКАМ (03.10, §6.259).

Находка ручной оцифровки «своими глазами» (лист Lokachi 27 РК 2009): выдача прода идёт по туши (на плоскости 0.7–1.2 px), но
по строке в 4–6 px от эталона. Простая постобработка — в каждой строке x := центр ПОЛНОГО рана «темнее бумаги строки» на thr,
содержащего x выдачи или ближайшего в ±R px (ран продлевается не дальше 3R от x), — дала на нём |Δx| 6.0 → 1.0 и 4.0 → 1.0.
Здесь — то же на выдаче нынешнего прода (`rp_vc/N`) по всем листам: честные 1:1 при 3 px (как приёмка) и именные (то же имя)
без притяжки и с вариантами притяжки; парный знаковый тест по листам. Бумага строки — 90-й перцентиль яркости (своя
реализация, без кода прода).

  _snap_whatif.py --sheets holdoutA_sheets.txt --variants 8:30 5:30 12:30 8:20 8:45
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc")
ap.add_argument("--mode", default="N")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--sheets", default="wellmap_sheets.txt")
ap.add_argument("--variants", nargs="+", default=["8:30"], help="R:thr — первый вариант основной (задан до прогона)")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
VARS = [tuple(int(v) for v in s.split(":")) for s in a.variants]
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
sheets = [l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()][a.offset::a.every]


def snap(tr, G, paper, R, thr):
    """x строки := центр полного рана туши, содержащего x или ближайшего в ±R (продление не дальше 3R от x)"""
    out = {}
    H, Wd = G.shape
    for y, xm in tr.items():
        if not (0 <= y < H):
            out[y] = xm; continue
        row = G[y]; c = int(round(xm))
        lo, hi = max(0, c - R), min(Wd, c + R + 1)
        if hi <= lo:
            out[y] = xm; continue
        lim = paper[y] - thr
        ink = row[lo:hi] < lim
        if not ink.any():
            out[y] = xm; continue
        idx = np.flatnonzero(ink) + lo
        p0 = int(idx[np.argmin(np.abs(idx - xm))])
        l = r = p0
        while l > 0 and row[l - 1] < lim and xm - l < 3 * R:
            l -= 1
        while r < Wd - 1 and row[r + 1] < lim and r - xm < 3 * R:
            r += 1
        out[y] = (l + r) / 2.0
    return out


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


KEYS = ["без притяжки"] + [f"R {R}, порог {thr}" for R, thr in VARS]
C = Counter(); PER = {k: [] for k in KEYS}; PERN = {k: [] for k in KEYS}
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    stem = q.stem
    pd = Path(a.dir) / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
    img = find_image(q)
    if not got or not img:
        continue
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
         if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    tracks = {tm.get(g) for g in G if tm.get(g) is not None}
    if not tracks:
        continue
    gray = np.asarray(Image.open(img).convert("L"), np.int16)
    paper = np.percentile(gray[:, ::3], 90, axis=1)
    WS = {KEYS[0]: W}
    for (R, thr), k in zip(VARS, KEYS[1:]):
        WS[k] = {n: snap(t, gray, paper, R, thr) if t else t for n, t in W.items()}
    del gray
    cnt = Counter(); cntn = Counter()
    for t in tracks:
        gs = [g for g in G if tm.get(g) == t]; ws = [k for k in W if tm.get(k) == t and W[k]]
        C["кривых"] += len(gs)
        for k in KEYS:
            ok = {}
            for g in gs:
                for w in ws:
                    m, c = st(WS[k][w], G[g])
                    ok[(g, w)] = m is not None and m <= 3.0 and c >= 0.9
            cnt[k] += len(match(gs, ws, ok))
            cntn[k] += sum(1 for g in gs if ok.get((g, g)))
    for k in KEYS:
        C[k] += cnt[k]; C[(k, "имён")] += cntn[k]
        PER[k].append(cnt[k]); PERN[k].append(cntn[k])
    if si % 50 == 0:
        print(f"  … {si}/{len(sheets)}: " + ", ".join(f"{k} {C[k]}" for k in KEYS), file=sys.stderr)


def ptest(d):
    d = np.asarray(d); nz = d[d != 0]
    if not len(nz):
        return 1.0
    rng = np.random.default_rng(0)
    sims = (rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)
    return float(np.mean(np.abs(sims) >= abs(d.sum())))


if a.dump:
    pickle.dump(dict(PER=PER, PERN=PERN, C=dict(C)), open(a.dump, "wb"))
print(f"★ {a.sheets} (каждый {a.every}-й, сдвиг {a.offset}): кривых {C['кривых']}")
b = np.array(PER[KEYS[0]]); bn = np.array(PERN[KEYS[0]])
for k in KEYS:
    d = np.array(PER[k]) - b; dn = np.array(PERN[k]) - bn
    tail = "" if k == KEYS[0] else (f" (Δ {int(d.sum()):+d}, листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {ptest(d):.4f}); "
                                    f"именных Δ {int(dn.sum()):+d} (p = {ptest(dn):.4f})")
    print(f"   {k:18s}: честных 1:1 при 3 px {C[k]}, именных {C[(k, 'имён')]}{tail}")
