# -*- coding: utf-8 -*-
"""«Рядом» на толстых штрихах: где лежат эталон и трасса относительно рана туши в строке (растр)."""
import sys, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im
from auto.config import DEFAULT
TS = Path(r"F:/nds/output/taskS"); DIR = "ab_slot/RA"
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
FAMS = None
def st(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30: return None, 0.0
    return float(np.median([abs(tr[y] - gt[y]) for y in com])), len(com) / max(1, len(gt))
def match(rows, cols, ok):
    pair = {}
    def try_(r, seen):
        for c in cols:
            if not ok.get((r, c)) or c in seen: continue
            seen.add(c)
            if c not in pair or try_(pair[c], seen):
                pair[c] = r; return True
        return False
    for r in rows: try_(r, set())
    return {r: c for c, r in pair.items()}
def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got: return None
    return {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}
trk = pickle.load(open(TS / "pick_learn_honest2.pkl.dump", "rb")); KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / "slotmap.pkl", "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
sheets = sorted({r[0] for r in trk})
agg = defaultdict(list); n_curves = Counter(); MAXC = 10**9; percurve = []
for si, sh in enumerate(sheets, 1):
    if sum(n_curves.values()) >= MAXC: break
    q = SRC.get(sh); W = read_out(DIR, sh)
    if not q or W is None: continue
    G = extract(str(q))
    raw = {c["name"]: c for c in G["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    
    gts = {nm: dense(c) for nm, c in raw.items()}
    tm = smap.get(sh, {}); bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None: bytr[t].append(nm)
    todo = []
    for t, ns in bytr.items():
        if (sh, t) not in KOF: continue
        Wt = [k for k in W if tm.get(k) == t and W[k]]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        for g in ns:
            if g in mt or not Wt or False: continue
            best = None
            for k2 in Wt:
                m2, c2 = st(W[k2], gts[g])
                if m2 is None: continue
                key = (0 if m2 <= 3 else 1, m2)
                if best is None or key < best[0]: best = (key, m2, c2, k2)
            if best and 3 < best[1] <= 100: todo.append((g, best[3], best[1]))
    if not todo: continue
    img = find_image(q)
    if not img: continue
    rgb = im.load_rgb(str(img)); H, Wd = rgb.shape[:2]
    dark = im.dark_mask(rgb, DEFAULT.cv) & ~im.structure_mask(rgb, DEFAULT.cv)
    for g, k, m in todo:
        c = raw[g]; verts = {c["top_y"] + i: float(x) for i, x in enumerate(c["xs"]) if x != NULL}
        w = W[k]; fam = M.mnem_root(g)
        n_curves[fam] += 1
        same = []; wid = []; onink = 0; nrows = 0
        for y, gx in verts.items():
            if y not in w or not (0 <= y < H): continue
            tx = w[y]; row = dark[y]
            gi = int(round(gx)); ti = int(round(tx))
            if not (0 <= gi < Wd and 0 <= ti < Wd): continue
            runs = im.row_runs(row)
            def run_of(xi):
                for a_, b_, c_ in runs:
                    if a_ - 2 <= xi <= b_ + 2: return (a_, b_, c_)
                return None
            rg, rt = run_of(gi), run_of(ti)
            agg[(fam, "строк")].append(1)
            agg[(fam, "эталон на туши")].append(rg is not None)
            agg[(fam, "трасса на туши")].append(rt is not None)
            nrows += 1; onink += (rg is not None)
            if rg and rt:
                same.append(rg == rt); agg[(fam, "один и тот же ран")].append(rg == rt)
                if rg == rt: wid.append(rg[1] - rg[0] + 1)
                if rg == rt:
                    a_, b_, c_ = rg; wdt = b_ - a_ + 1
                    agg[(fam, "ширина рана")].append(wdt)
                    agg[(fam, "позиция эталона в ране (0=лев край,1=прав)")].append((gx - a_) / max(1, wdt - 1))
                    agg[(fam, "позиция трассы в ране")].append((tx - a_) / max(1, wdt - 1))
                    agg[(fam, "|трасса − центр рана|")].append(abs(tx - c_))
                    agg[(fam, "|эталон − центр рана|")].append(abs(gx - c_))
                else:
                    agg[(fam, "разные раны: |центры|")].append(abs(rg[2] - rt[2]))
        percurve.append((fam, 'рядом' if m <= 10 else 'промах 10-100', float(np.mean(same)) if same else 0.0, float(np.median(wid)) if wid else 0.0, onink / max(1, nrows), m))
    if si % 100 == 0: print(f"  … {si}/{len(sheets)}, кривых {sum(n_curves.values())}", file=sys.stderr)
print("кривых «рядом» разобрано:", dict(n_curves))
for fam in sorted(n_curves):
    print(f"\n★ {fam}: строк-вершин {len(agg[(fam,'строк')])}")
    for key in ("эталон на туши", "трасса на туши", "один и тот же ран"):
        v = np.array(agg[(fam, key)], float)
        if len(v): print(f"   {key}: {100*v.mean():.0f}%")
    for key in ("ширина рана", "позиция эталона в ране (0=лев край,1=прав)", "позиция трассы в ране", "|трасса − центр рана|", "|эталон − центр рана|", "разные раны: |центры|"):
        v = np.array(agg[(fam, key)], float)
        if len(v): print(f"   {key}: p25 {np.percentile(v,25):.2f} p50 {np.median(v):.2f} p75 {np.percentile(v,75):.2f} (n={len(v)})")

print("\n★★ ПО КРИВЫМ: доля строк-вершин, где эталон и трасса на ОДНОМ ране туши")
for cat in ("рядом", "промах 10-100"):
    P = [p for p in percurve if p[1] == cat]
    if not P: continue
    same = np.array([p[2] for p in P]); wid = np.array([p[3] for p in P]); onink = np.array([p[4] for p in P])
    print(f"  {cat}: кривых {len(P)}; одна тушь ≥ 85% строк: {int((same >= 0.85).sum())} ({100*(same>=0.85).mean():.0f}%), "
          f"50–85%: {int(((same >= 0.5) & (same < 0.85)).sum())}, < 50% (перескоки): {int((same < 0.5).sum())}; "
          f"ширина рана p50 у «одна тушь»: {np.median(wid[same >= 0.85]) if (same>=0.85).any() else 0:.0f} px; эталон на туши p50 {np.median(onink):.2f}")
    byf = defaultdict(list)
    for p in P: byf[p[0]].append(p)
    for fam, pp in sorted(byf.items(), key=lambda kv: -len(kv[1]))[:10]:
        sm = np.array([p[2] for p in pp]); wd = np.array([p[3] for p in pp])
        print(f"     {fam:<5} n={len(pp):>3}  одна тушь ≥85%: {int((sm>=0.85).sum()):>3}  перескоки <50%: {int((sm<0.5).sum()):>3}  ширина p50 {np.median(wd):.0f}")
pickle.dump(percurve, open(r"F:/nds/output/taskS/near_rows_percurve.pkl", "wb"))
