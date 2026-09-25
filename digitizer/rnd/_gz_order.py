r"""_gz_order.py — ЧЕМ ОТЛИЧАЮТСЯ ЗОНДЫ ОДНОГО СЕМЕЙСТВА НА ОДНОМ ТРЕКЕ: проверка гипотез порядка по ЭТАЛОНУ (21.09, B4).

ОТКУДА ВОПРОС. §6.214: 119 перестановок внутри семейства, из них GZ 79 (GZ51↔GZ41 33, GZ21↔GZ31 23,
GZ11↔GZ21 13). Индекс зонда — длина зонда (A0.4M0.1N … A8.0M1.0N): длинный зонд сглаживает и
занижает амплитуду на тонких пластах. Если это видно на эталонных кривых, порядок восстанавливается
из формы, без разметки. Здесь — по экспертным кривым шаблона (истина), по каждому треку с ≥ 2 кривыми
одного корня и разными индексами: ПАРНАЯ точность каждого признака-упорядочивателя
(«у зонда с меньшим индексом признак больше») и доля треков, где порядок по признаку совпал целиком.

Признаки: wiggle = медиана |dx| / размах; rough = std(x − скользящее среднее 101) / размах;
rev = доля смен знака dx; medx = медиана x (нынешняя гипотеза sib §6.130: индекс ↔ x); amp = размах
(p90 − p10); hf = доля энергии в высоких частотах (|dx| второй разности).

  <ComfyUI>\python_embeded\python.exe _gz_order.py
"""
import sys, argparse, pickle, re
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
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--fam", default="GZ,T,TM,CAL,I,ZAT")
a = ap.parse_args()
TS = Path(a.ts)


def idx_of(nm):
    """'GZ51 DA1 SA1' → (корень 'GZ', индекс 5); индекс = цифры после корня без последней (номер экземпляра)."""
    short = nm.split()[0]
    root = M.mnem_root(short)
    m = re.match(re.escape(root) + r"(\d+)$", short)
    if not m:
        return root, None
    digits = m.group(1)
    return root, int(digits[:-1]) if len(digits) >= 2 else None


def feats(d):
    ys = sorted(d)
    x = np.array([d[y] for y in ys], float)
    if len(x) < 200:
        return None
    dx = np.diff(x)
    span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
    w = 101
    sm = np.convolve(x, np.ones(w) / w, mode="same")
    d2 = np.abs(np.diff(x, 2))
    return dict(wiggle=float(np.median(np.abs(dx))) / span, rough=float(np.std(x - sm)) / span,
                rev=float(np.mean((dx[:-1] * dx[1:]) < 0)), medx=float(np.median(x)),
                amp=span, hf=float(np.mean(d2)) / span)


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q
sheets = sorted({r[0] for r in trk})
fams = set(a.fam.split(","))
FN = ["wiggle", "rough", "rev", "hf", "amp", "medx"]
pair = defaultdict(lambda: defaultdict(Counter))     # fam → feat → {"lower_idx_bigger": n, "n": n}
full = defaultdict(lambda: defaultdict(Counter))     # fam → feat → {"desc_ok", "asc_ok", "n"}
ntr = Counter()
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    G = extract(str(q))
    gts = {c["name"]: dense(c) for c in G["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None and (sh, t) in KOF:
            bytr[t].append(nm)
    for t, ns in bytr.items():
        byfam = defaultdict(list)
        for nm in ns:
            root, idx = idx_of(nm)
            if root in fams and idx is not None:
                f = feats(gts[nm])
                if f:
                    byfam[root].append((idx, f))
        for root, lst in byfam.items():
            if len(lst) < 2 or len({i for i, _ in lst}) < len(lst):
                continue
            ntr[root] += 1
            for fn in FN:
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        (ia, fa), (ib, fb) = lst[i], lst[j]
                        lo, hi = (fa, fb) if ia < ib else (fb, fa)
                        pair[root][fn]["n"] += 1
                        pair[root][fn]["lower_idx_bigger"] += lo[fn] > hi[fn]
                order = [i for i, _ in sorted(lst, key=lambda z: -z[1][fn])]
                full[root][fn]["n"] += 1
                full[root][fn]["desc_ok"] += order == sorted(order)
                full[root][fn]["asc_ok"] += order == sorted(order, reverse=True)
    if si % 300 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

for root in sorted(ntr, key=lambda r: -ntr[r]):
    print(f"\n★ семейство {root}: треков с ≥ 2 зондами разного индекса — {ntr[root]}")
    print("| признак | пар | «меньший индекс — признак больше» | треков | порядок целиком по убыванию | по возрастанию |")
    print("|---|---|---|---|---|---|")
    for fn in FN:
        p = pair[root][fn]; f = full[root][fn]
        print(f"| {fn} | {p['n']} | {100*p['lower_idx_bigger']/max(1,p['n']):.0f}% | {f['n']} | "
              f"{100*f['desc_ok']/max(1,f['n']):.0f}% | {100*f['asc_ok']/max(1,f['n']):.0f}% |")
