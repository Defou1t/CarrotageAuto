r"""_dec_byk.py — БЕЗЫМЯННЫЙ 1:1 ПО K ТРЕКА ДЛЯ ЛЮБОГО НАБОРА ВЫДАЧ (18.09, к §6.211 п.2-3).

ЗАЧЕМ. §6.211 задал критерий для варианта декодера `bg1` ПО ПЛОТНЫМ ТРЕКАМ: открывать выбор пути
только если bg1 на K ≥ 4 даёт не меньше прода + 20. `_u1_vs_dec.py` считает по исходу U1 и с жёсткой
сверкой на 965/934/1107 — на другом наборе каталогов и подмножестве листов она не сходится по
построению. Здесь: те же `match`/HON/`dense`, любые колонки `--col ИМЯ=каталог`, любое поле `--sheets`,
K трека = число эталонных кривых трека (как в §6.211), плюс сверка: колонка `--check ИМЯ=число`
обязана дать ожидаемую сумму (иначе таблица недействительна).

  <ComfyUI>\python_embeded\python.exe _dec_byk.py --col P965=ab_wellmap/A --col DEC=ab_rdhonest/B --col CUR=ab_pregate/G --check CUR=1107
  <ComfyUI>\python_embeded\python.exe _dec_byk.py --sheets kslots_gated.txt --col G=ab_slot/G --col D0=ab_bg1/D0 --col B1=ab_bg1/B1
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
ap.add_argument("--col", action="append", required=True, help="ИМЯ=каталог выдачи (относительно --ts)")
ap.add_argument("--check", action="append", default=[], help="ИМЯ=ожидаемая сумма (сверка)")
ap.add_argument("--sheets", default="", help="поле: список листов (пусто = все листы честного поля)")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
a = ap.parse_args()
TS = Path(a.ts)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
COLS = [c.split("=", 1) for c in a.col]
CHK = {k: int(v) for k, v in (c.split("=", 1) for c in a.check)}


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
    return [r for r in rows if try_(r, set())]


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
if a.sheets:
    want = {l.strip() for l in (TS / a.sheets).read_text(encoding="utf-8").splitlines() if l.strip()}
    sheets = [s for s in sheets if s in want]
    print(f"★ ПОЛЕ: {len(sheets)} листов из списка {a.sheets} ({len(want)} в списке)")
tab = defaultdict(Counter)          # K → {col: взято, "n": кривых}
empty, done = Counter(), 0
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    outs = {}
    for tag, root in COLS:
        w = read_out(root, sh)
        if w is None:
            empty[tag] += 1; w = {}
        outs[tag] = w
    done += 1
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        K = min(len(ns), 5)
        tab[K]["n"] += len(ns)
        for tag, _ in COLS:
            W = [k for k in outs[tag] if tm.get(k) == t]
            ok = {(g, k): HON(*st(outs[tag][k], gts[g])) for g in ns for k in W}
            tab[K][tag] += len(match(ns, W, ok))
    if si % 200 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)

tot = Counter()
for K in tab:
    for k, v in tab[K].items():
        tot[k] += v
print(f"\nлистов обработано {done} из {len(sheets)}; пустых выдач по колонкам: {dict(empty) or 'нет'}")
bad = 0
for tag, exp in CHK.items():
    got = tot[tag]
    ok = abs(got - exp) <= 2
    bad += not ok
    print(f"  СВЕРКА {tag}: получено {got}, ожидалось {exp} — {'OK' if ok else '⛔ РАСХОЖДЕНИЕ'}")
if bad:
    print("⛔⛔ СВЕРКА НЕ СОШЛАСЬ — таблицу цитировать нельзя")
names = [t for t, _ in COLS]
print("\n★★ БЕЗЫМЯННЫЙ 1:1 ПО K ТРЕКА (K = эталонных кривых трека)")
print("| K | кривых | " + " | ".join(names) + " |")
print("|---|---|" + "---|" * len(names))
for K in sorted(tab):
    print(f"| {K}{'+' if K == 5 else ''} | {tab[K]['n']} | " + " | ".join(str(tab[K][t]) for t in names) + " |")
n4 = sum(tab[K]["n"] for K in tab if K >= 4)
print(f"| **K ≥ 4** | {n4} | " + " | ".join(f"**{sum(tab[K][t] for K in tab if K >= 4)}**" for t in names) + " |")
print(f"| **всего** | {tot['n']} | " + " | ".join(f"**{tot[t]}**" for t in names) + " |")
