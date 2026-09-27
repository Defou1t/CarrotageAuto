r"""_ruling_share.py — СКОЛЬКО «УХОДА С ЭТАЛОНА» ЛЕЖИТ НА ВЕРТИКАЛЯХ СЕТКИ, ПЕРЕЖИВШИХ `structure_mask` (§6.218, B7, 25.09).

`_swap_anatomy.py`: 47% не взятых кривых многокривых треков идут по туши вне эталона; на листах — прыжки на прямые вертикали.
Здесь это меряется по растру. Передний план — ТОТ ЖЕ, по которому ведёт прод (`trace2d._color_fg(rgb, 'black', p)`: тёмное,
минус цветные каналы, минус `structure_mask`). Вертикаль-«линейка» в блоке 1000 строк — колонка, где передний план есть
в ≥ `--frac` строк блока (кривая через колонку проходит, а не живёт в ней). Для каждой не взятой кривой (выдача прода RA,
многокривые треки) по строкам, где ближайшая выдача дальше 3 px от ВСЕХ эталонных кривых трека, считается доля строк, чей x
(±2 px) попадает на линейку своего блока.
"""
import sys, argparse, pickle, hashlib, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from dataset_build import find_image
from auto import meta as M, imaging as im, trace2d as T2
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--dir", default="ab_slot/RA")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--frac", type=float, default=0.6)
ap.add_argument("--block", type=int, default=1000)
ap.add_argument("--every", type=int, default=1, help="брать каждый N-й лист")
ap.add_argument("--frac-dot", type=float, default=0.3, help="порог «пунктирной» линейки")
a = ap.parse_args()
TS = Path(a.ts)
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


RAW = {}


def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return None
    cs = [c for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"]
    RAW.clear()
    for c in cs:
        RAW[c["name"]] = {c["top_y"] + i for i, x in enumerate(c["xs"]) if x != NULL}
    return {c["name"]: dense(c) for c in cs}


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})[::a.every]
shares = []; C = Counter(); rows_off = rows_rul = 0; R = Counter()
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    W = read_out(a.dir, sh)
    if W is None:
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    todo = []
    for t, ns in bytr.items():
        if (sh, t) not in KOF or len(ns) < 2:
            continue
        Wt = [k for k in W if tm.get(k) == t and W[k]]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        # ⛔ 25.09 (ревизия агентом): две не взятые кривые трека часто выбирают ОДНУ выдачу — её строки ухода считались
        #   дважды (27% веса). Теперь строка (выдача, y) отдаётся кривой с меньшей медианой. Итог §6.218 «80/18/3»
        #   пересчитан без двойного счёта стендом `_drift_anatomy.py` (§6.219).
        cand = defaultdict(list)
        for g in ns:
            if g in mt or not Wt:
                continue
            best = None
            for k in Wt:
                m, c = st(W[k], gts[g])
                if m is not None and (best is None or m < best[0]):
                    best = (m, k)
            if best and best[0] > 3:
                cand[best[1]].append((best[0], g))
        for kname, lst in cand.items():
            w = W[kname]
            raw = RAW.get(kname, set())
            claimed = set()
            for _m, g in sorted(lst):
                off = [y for y in w if y in raw and y in gts[g] and y not in claimed
                       and all(not (y in gts[o] and abs(w[y] - gts[o][y]) <= 3) for o in ns)]
                claimed.update(off)
                if len(off) >= 30:
                    todo.append((g, w, off, kname))
    if not todo:
        continue
    img = find_image(q)
    if not img:
        continue
    # источник кривой слота в выдаче RA: трек отдан декодеру / проду, слот перевёрнут правилом §6.213
    stem = Path(sh).stem
    pdir = TS / a.dir / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    pj = next(iter(pdir.glob("*_pick.json")), None)
    SRCK = {}
    if pj:
        pk = json.loads(pj.read_text(encoding="utf-8"))
        tdec = {t["track"]: t["dec"] for t in pk.get("tracks", [])}
        for sl in pk.get("slots", []):
            base = sl["base"]
            SRCK[sl["name"]] = sl.get("src") or (("dec" if base == "prod" else "prod") if sl["flip"] else base)
        for k in W:
            if k not in SRCK:
                SRCK[k] = "dec" if tdec.get(tm.get(k)) else "prod"
    rgb = im.load_rgb(str(img))
    fg = T2._color_fg(rgb, "black", DEFAULT.cv)
    H, Wd = fg.shape
    nb = (H + a.block - 1) // a.block
    rul = np.zeros((nb, Wd), bool)
    for b in range(nb):
        blk = fg[b * a.block:(b + 1) * a.block]
        rul[b] = blk.mean(0) >= a.frac
    # расширить ±2 px
    rul2 = rul.copy()
    for s in (-2, -1, 1, 2):
        rul2 |= np.roll(rul, s, axis=1)
    dot = np.zeros((nb, Wd), bool)
    for b in range(nb):
        dot[b] = fg[b * a.block:(b + 1) * a.block].mean(0) >= a.frac_dot
    dot2 = dot.copy()
    for s in (-2, -1, 1, 2):
        dot2 |= np.roll(dot, s, axis=1)
    anyf = fg.copy()
    sm = im.structure_mask(rgb, DEFAULT.cv)
    for cm in im.color_channels(rgb, DEFAULT.cv).values():
        anyf |= (cm & ~sm)
    fg2 = anyf.copy()
    for s in (-2, -1, 1, 2):
        fg2 |= np.roll(anyf, s, axis=1)
    for g, w, off, kname in todo:
        on = 0
        src = SRCK.get(kname, "?")
        for y in off:
            xi = int(round(w[y]))
            if not (0 <= y < H and 0 <= xi < Wd):
                R["вне листа"] += 1; continue
            if rul2[y // a.block, xi]:
                cat = "сплошная линейка (≥ frac)"; on += 1
            elif dot2[y // a.block, xi]:
                cat = "пунктирная линейка (≥ frac-dot)"
            elif fg2[y, xi]:
                cat = "другая тушь (не линейка)"
            else:
                cat = "туши нет (трасса идёт по пустому)"
            R[cat] += 1; R[(src, cat)] += 1; R[(src, "всего")] += 1
        shares.append(on / len(off)); rows_off += len(off); rows_rul += on
        C[f"кривых с ≥ 50% «ухода» на линейке"] += on / len(off) >= 0.5
        C["кривых разобрано"] += 1
    del rgb, fg
    if si % 100 == 0:
        print(f"  … {si}/{len(sheets)}, кривых {C['кривых разобрано']}", file=sys.stderr)

s = np.array(shares)
print(f"\n★★ НЕ ВЗЯТЫЕ КРИВЫЕ МНОГОКРИВЫХ ТРЕКОВ с «уходом» ≥ 30 строк: {C['кривых разобрано']}; строк ухода {rows_off}, "
      f"из них на вертикали-линейке (передний план в ≥ {a.frac:.0%} строк блока {a.block}) {rows_rul} ({100*rows_rul/max(1,rows_off):.0f}%)")
print(f"   доля ухода на линейке по кривым: p25 {np.percentile(s,25):.2f}, p50 {np.median(s):.2f}, p75 {np.percentile(s,75):.2f}; "
      f"кривых, где ≥ 50% ухода на линейке: {C['кривых с ≥ 50% «ухода» на линейке']}")
tot = sum(R.values())
print("★★ СТРОКИ УХОДА (только строки, где в файле есть значение) ПО ТОМУ, ЧТО ПОД ТРАССОЙ (тушь любого цвета ±2 px):")
tot = sum(v for k, v in R.items() if isinstance(k, str))
for k, v in sorted(((k, v) for k, v in R.items() if isinstance(k, str)), key=lambda kv: -kv[1]):
    print(f"   {k}: {v} ({100*v/max(1,tot):.0f}%)")
print("★★ ПО ИСТОЧНИКУ КРИВОЙ В ВЫДАЧЕ (prod = прод-ведение, dec = декодер):")
for src in ("prod", "dec", "?"):
    n = R.get((src, "всего"), 0)
    if not n:
        continue
    parts = ", ".join(f"{c.split(' (')[0]} {100*R.get((src, c), 0)/n:.0f}%" for c in ("другая тушь (не линейка)", "туши нет (трасса идёт по пустому)", "пунктирная линейка (≥ frac-dot)", "сплошная линейка (≥ frac)"))
    print(f"   {src}: строк {n} ({100*n/max(1,tot):.0f}% ухода): {parts}")
