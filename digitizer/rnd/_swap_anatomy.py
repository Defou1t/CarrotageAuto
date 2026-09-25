r"""_swap_anatomy.py — КУДА УХОДИТ ТРАССА, ПОТЕРЯВШАЯ СВОЮ КРИВУЮ (§6.218, B5, 25.09).

ОТКУДА. §6.216: ~35% поля — промахи 10–100 и > 100 px на многокривых треках; по растру 273 из 504 кривых «10–100» идут
по чужой туши больше половины строк. Совместное ведение K кривых с порядковым ограничением (B5) лечит только ПЕРЕСКОК НА
СОСЕДНЮЮ КРИВУЮ ЭТАЛОНА. Если же трасса уходит на тушь, которой в эталоне нет (сетка, подписи, неразмеченная кривая,
перевынос), нужен не порядок, а обнаружение. Здесь это делится, без растра.

Для каждой эталонной кривой g, НЕ взятой 1:1 (выдача прода `ab_slot/RA`), берётся ближайшая выданная кривая того же
трека w (минимум медианы |Δx|) и по строкам общей глубины считается, где она лежит:
  на g (|Δx| ≤ 3), на ДРУГОЙ эталонной кривой трека (≤ 3 px от неё), нигде из эталона.
Кривая относится к классу по большинству строк: «своя, но неточно», «перескок на соседа» (≥ 50% строк на другом эталоне),
«смесь своя/сосед», «не эталонная тушь» (≥ 50% строк вне всех эталонных кривых трека).
Плюс по эталону: как часто кривые трека пересекаются (смена порядка по x на 1000 строк) — годится ли жёсткий порядок.
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--rows", default="pick_learn_honest2.pkl.dump")
ap.add_argument("--map", default="slotmap.pkl")
ap.add_argument("--dir", default="ab_slot/RA")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
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


def read_out(root, sh):
    stem = Path(sh).stem
    pd = TS / root / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    if not got:
        return None
    return {c["name"]: dense(c) for c in extract(str(got))["curves"] if M.mnem_root(c["name"]) != "DA"}


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})
cls = Counter(); byK = defaultdict(Counter); cross = []; pair_never = Counter()
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
    for t, ns in bytr.items():
        if (sh, t) not in KOF or len(ns) < 2:
            continue
        # пересечения эталона: смена знака разности x двух кривых (сглажено окном 25 строк), на 1000 общих строк
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                A, B = gts[ns[i]], gts[ns[j]]
                ys = np.array(sorted(set(A) & set(B)))
                if len(ys) < 200:
                    continue
                d = np.array([A[y] - B[y] for y in ys])
                d = np.convolve(d, np.ones(25) / 25, mode="valid")
                s = np.sign(d[np.abs(d) > 3])
                n_cross = int((np.diff(s) != 0).sum()) if len(s) > 1 else 0
                cross.append(1000.0 * n_cross / len(ys))
                pair_never["никогда не пересекаются" if n_cross == 0 else "пересекаются"] += 1
        Wt = [k for k in W if tm.get(k) == t and W[k]]
        ok = {(g, k): HON(*st(W[k], gts[g])) for g in ns for k in Wt}
        mt = match(ns, Wt, ok)
        K = min(len(ns), 5)
        for g in ns:
            if g in mt or not Wt:
                continue
            best = None
            for k in Wt:
                m, c = st(W[k], gts[g])
                if m is not None and (best is None or m < best[0]):
                    best = (m, k)
            if best is None or best[0] <= 3:
                continue
            w = W[best[1]]
            ys = [y for y in w if y in gts[g]]
            on_own = on_other = nowhere = 0
            for y in ys:
                if abs(w[y] - gts[g][y]) <= 3:
                    on_own += 1
                elif any(y in gts[o] and abs(w[y] - gts[o][y]) <= 3 for o in ns if o != g):
                    on_other += 1
                else:
                    nowhere += 1
            n = max(1, len(ys))
            if on_other / n >= 0.5:
                c_ = "перескок на соседа (≥ 50% строк на другом эталоне)"
            elif nowhere / n >= 0.5:
                c_ = "не эталонная тушь (≥ 50% строк вне всех эталонных кривых)"
            elif on_own / n >= 0.5:
                c_ = "своя, но неточно (≥ 50% строк на своей)"
            else:
                c_ = "смесь"
            cls[c_] += 1; byK[K][c_] += 1
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

tot = sum(cls.values())
print(f"\n★★ НЕ ВЗЯТЫЕ КРИВЫЕ МНОГОКРИВЫХ ТРЕКОВ (ошибка ближайшей выдачи > 3 px): {tot}")
for c_, v in cls.most_common():
    print(f"   {c_}: {v} ({100*v/max(1,tot):.0f}%)")
print("\n| K | " + " | ".join(c_.split(' (')[0] for c_, _ in cls.most_common()) + " |")
print("|---|" + "---|" * len(cls))
for K in sorted(byK):
    print(f"| {K}{'+' if K == 5 else ''} | " + " | ".join(str(byK[K][c_]) for c_, _ in cls.most_common()) + " |")
cr = np.array(cross)
print(f"\n★ ЭТАЛОН: пар кривых на одном треке {len(cr)}; {dict(pair_never)}; пересечений на 1000 строк: "
      f"p50 {np.median(cr):.2f}, p75 {np.percentile(cr,75):.2f}, p90 {np.percentile(cr,90):.2f}")
