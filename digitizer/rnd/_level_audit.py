r"""_level_audit.py — ПЕРЕХОДЫ МАСШТАБА (×5 И ДАЛЕЕ): ЭТАЛОН ПРОТИВ ВЫДАЧИ ПРОДА (03.10, вопрос заказчика по LOBACH_032).

Заказчик на C12 (LOBACH_032 РК): «нет перехода на 5х масштаб. Это не учтено?». Проверено: у эталона ГК 68 сегментов,
чередующих шкалу 1× (0–10) и 5× (0–50), у выдачи прода — один сегмент уровня 0. `emit._level_segments` декодирует уровни
ТОЛЬКО у резистивных мнемоник (`decode_levels.RESIST`); у ГК и прочих — всегда уровень 0. Мера честности сравнивает ПИКСЕЛИ
и уровня не видит, хотя значение в LAS в 5×-сегменте занижено впятеро.
Здесь по всему полю и сорту A:
  — кривые эталона с переходами (сегменты уровня > 0): число, доля строк на уровне > 0, по семействам, резистивные или нет;
  — у именно-честных кривых выдачи (то же имя, 1:1 при 3 px) — доля общих строк, где уровень выдачи = уровню эталона;
  — сколько именно-честных кривых остались бы честными, если требовать и уровень (≥ 90% строк с верным уровнем).
Только чтение эталона и выдачи `rp_vc/N`.

  _level_audit.py --dump F:/nds/output/taskS/level_audit.pkl
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
import decode_levels as DL

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def levels(c):
    """строка → уровень по сегментам кривой"""
    out = {}
    for s in c.get("segments") or []:
        y0, y1, lv = s
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


ROWS = []
C = {k: Counter() for k in SETS}
FAM = {k: Counter() for k in SETS}
for sn, names in SETS.items():
    for sh in names:
        q = SRC.get(sh)
        if not q:
            continue
        stem = q.stem
        pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
        mt = extract(str(q))
        Tc = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        Wc = {c["name"]: c for c in extract(str(got))["curves"]} if got else {}
        for nm, c in Tc.items():
            lv = levels(c)
            gt = dense(c)
            rows = [y for y in gt if y in lv]
            up = sum(1 for y in rows if lv[y] > 0)
            chain = len(DL.build_family(mt, c))
            multi = up > 0
            res = DL.is_resistive(nm)
            C[sn]["кривых эталона"] += 1
            C[sn]["с цепочкой масштабов в шаблоне"] += chain >= 2
            C[sn]["с переходами (строки уровня > 0)"] += multi
            if multi:
                FAM[sn][(M.mnem_root(nm), "резистивная" if res else "НЕ резистивная")] += 1
            r = dict(set=sn, sheet=sh, name=nm, root=M.mnem_root(nm), res=res, chain=chain, multi=multi,
                     up_frac=up / max(1, len(rows)), honest=None, lvl_acc=None, out_multi=None)
            w = Wc.get(nm)
            if w is not None:
                tr = dense(w)
                com = [y for y in tr if y in gt]
                if len(com) >= 30:
                    e = np.array([abs(tr[y] - gt[y]) for y in com])
                    r["honest"] = bool(np.median(e) <= 3.0 and len(com) / max(1, len(gt)) >= 0.9)
                    lw = levels(w)
                    both = [y for y in com if y in lv and y in lw]
                    if both:
                        r["lvl_acc"] = float(np.mean([lw[y] == lv[y] for y in both]))
                    r["out_multi"] = any(l > 0 for l in lw.values())
            ROWS.append(r)
if a.dump:
    pickle.dump(ROWS, open(a.dump, "wb"))
for sn in SETS:
    c = C[sn]
    print(f"★ {sn}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    R = [r for r in ROWS if r["set"] == sn]
    mr = [r for r in R if r["multi"]]
    if mr:
        print(f"   кривые с переходами: доля строк на уровне > 0 — медиана {np.median([r['up_frac'] for r in mr]):.2f}")
        print("   по семействам: " + ", ".join(f"{k[0]} ({k[1]}) {v}" for k, v in FAM[sn].most_common(14)))
    H = [r for r in R if r["honest"]]
    Hm = [r for r in H if r["multi"]]
    print(f"   именно-честных (то же имя, 3 px): {len(H)}; из них у эталона есть переходы: {len(Hm)}")
    for grp, sel in (("резистивные", [r for r in Hm if r["res"]]), ("НЕ резистивные", [r for r in Hm if not r["res"]])):
        acc = [r["lvl_acc"] for r in sel if r["lvl_acc"] is not None]
        if acc:
            acc = np.array(acc)
            print(f"      {grp}: {len(sel)}; уровень выдачи верен — медиана {np.median(acc):.2f} строк; ≥ 0.9 у {np.mean(acc >= 0.9):.0%}; "
                  f"выдача вообще с переходами у {np.mean([bool(r['out_multi']) for r in sel]):.0%}")
    keep = sum(1 for r in H if not r["multi"] or (r["lvl_acc"] or 0) >= 0.9)
    print(f"   ⇒ требуя и уровень (≥ 90% строк): именно-честных {len(H)} → {keep} ({keep - len(H):+d})")
