r"""_u1_vs_prod.py — ПЕРЕЖИВАЕТ ЛИ ОШИБКА U1 ДОВОДКУ ПРОДОМ (§6.195 → цена правки первого этажа).

ОТКУДА ВОПРОС. §6.195 разложил провалы U1, но U1 — не конец конвейера: прод берёт его линии как
затравку и ведёт заново (`trace2d`), поэтому даёт 35.6% честных против 31.3% у сырых трасс U1.
Значит НЕ КАЖДАЯ ошибка U1 стоит кривой: часть прод чинит сам. Прежде чем вкладываться в U1, надо
знать, какая именно часть.

ЧТО СЧИТАЕТ: перекрёстную таблицу по каждой эталонной кривой — исход у U1 против исхода у прода.
Исход U1 берётся по ЛУЧШЕЙ его трассе на треке: накрыта / короткая (med ≤ 3, cov < 0.9) /
мелкий промах 3-10px / крупный промах / линии нет вовсе.

⇒ Отвечает на единственный вопрос, от которого зависит цена работы: СКОЛЬКО КРИВЫХ ПРОД ТЕРЯЕТ
ТАМ, ГДЕ U1 УЖЕ ДАЛ ПРАВИЛЬНУЮ ЛИНИЮ. Это верхняя оценка выигрыша от доводки U1 и, наоборот, размер
того, что доводкой не лечится.

★ Счёта не тратит: пуловые дампы (линии U1), замороженная выдача прода, разметка.

  <ComfyUI>\python_embeded\python.exe _u1_vs_prod.py
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
ap.add_argument("--prod", default="ab_wellmap/A")
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
    """⚠⚠ ВЕДУЩИЙ СЧЁТ ВЕТКИ — БЕЗЫМЯННЫЙ: максимальное 1:1 внутри трека, кто бы как ни назывался
    (§6.143). Первая редакция этого стенда сверяла кривую по СОВПАДЕНИЮ ИМЕНИ и получила итог 661
    при известных 662 именных — то есть считала не ту величину. Механика взята из
    `_name_cost_prod.py` дословно; возвращает СПИСОК взятых эталонных кривых."""
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


trk = pickle.load(open(TS / a.rows, "rb"))
KOF = {(r[0], r[2]): int(r[3]["n_prod"]) for r in trk}
smap = pickle.load(open(TS / a.map, "rb"))
stem2f = {}
for root in ("pools", "pools_gate", "pools_wide", "pools_more", "pools_div",
             "pools_heldout", "pools_all"):
    for f in sorted((TS / root).glob("*.pkl")):
        stem2f.setdefault(f.stem, f)
SRC = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            SRC[q.name] = q

ORDER = ["★ накрыта", "короткая (med≤3, cov<0.9)", "мелкий промах 3-10px",
         "промах 10-100px", "крупный >100px", "линии нет / нет пересечения"]
tab = defaultdict(Counter)
for sh in sorted({r[0] for r in trk}):
    f, q = stem2f.get(Path(sh).stem), SRC.get(sh)
    if not f or not q:
        continue
    d = pickle.load(open(f, "rb"))
    gts = {c["name"]: dense(c) for c in extract(str(q))["curves"]
           if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    tm = smap.get(sh, {})
    bytr = defaultdict(list)
    for nm in gts:
        t = tm.get(nm)
        if t is not None:
            bytr[t].append(nm)
    lines = defaultdict(list)
    for L in d["lines"]:
        lines[int(L["track"])].append(L["tr"])
    stem = Path(sh).stem
    dn = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    pd = Path(a.ts) / a.prod / dn
    got = next(iter(sorted(pd.glob("*_auto.nlgx"))), None) if pd.is_dir() else None
    wr = ({c["name"]: dense(c) for c in extract(str(got))["curves"]
           if M.mnem_root(c["name"]) != "DA"} if got else {})
    for t, ns in bytr.items():
        if (sh, t) not in KOF:
            continue
        LL = lines.get(t, [])
        # ★ ПРОД — по безымянному 1:1 внутри трека, а не по совпадению имени
        W = [k for k in wr if tm.get(k) == t]
        okp = {(g, w): HON(*st(wr[w], gts[g])) for g in ns for w in W}
        took = set(match(ns, W, okp))
        for nm in ns:
            g = gts[nm]
            sts = [st(tr, g) for tr in LL]
            if any(HON(m, c) for m, c in sts):
                k = ORDER[0]
            else:
                cand = [(m, c) for m, c in sts if m is not None]
                if not cand:
                    k = ORDER[5]
                else:
                    m, c = min(cand)
                    k = (ORDER[1] if m <= 3.0 else ORDER[2] if m <= 10 else
                         ORDER[3] if m <= 100 else ORDER[4])
            tab[k]["прод взял" if nm in took else "прод НЕ взял"] += 1

print("★★ ИСХОД У U1 ПРОТИВ ИСХОДА У ПРОДА (эталонных кривых по честному полю)")
print("| исход у U1 | кривых | прод взял | доля спасённых |")
tot = won = 0
for k in ORDER:
    c = tab[k]
    n = sum(c.values())
    if not n:
        continue
    tot += n
    won += c["прод взял"]
    print(f"| {k} | {n} | {c['прод взял']} | **{100*c['прод взял']/n:.0f}%** |")
print(f"| ВСЕГО | {tot} | {won} | {100*won/max(1,tot):.0f}% |")

good = sum(tab[k]["прод НЕ взял"] for k in ORDER[:3])
goodn = sum(sum(tab[k].values()) for k in ORDER[:3])
print(f"\n★ ПРОД ТЕРЯЕТ {good} кривых ИЗ {goodn} ТАМ, ГДЕ U1 УЖЕ ДАЛ ПРАВИЛЬНУЮ ЛИНИЮ "
      f"(накрыта / короткая / промах ≤10px) — это резерв ВНИЗУ, доводкой U1 он не лечится.")
bad = sum(sum(tab[k].values()) for k in ORDER[3:])
badw = sum(tab[k]["прод взял"] for k in ORDER[3:])
print(f"★ И НАОБОРОТ: там, где U1 ошибся крупно или не нашёл линию ({bad} кривых), прод всё равно "
      f"взял {badw} ({100*badw/max(1,bad):.0f}%) — значит он не привязан к U1 намертво.")
