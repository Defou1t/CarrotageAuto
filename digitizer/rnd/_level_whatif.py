r"""_level_whatif.py — ДЕКОДЕР ПЕРЕХОДОВ ДЛЯ ВСЕХ КРИВЫХ С ЦЕПОЧКОЙ МАСШТАБОВ: ЧТО ДАСТ (03.10, к `_level_audit.py`).

`emit._level_segments` зовёт `decode_levels.decode` только для `DL.is_resistive` (GZ/OGZ/BK/IK/PZ/MBK/…). У ГК/НГК/ЗАТ/ТМ/I
и у имён с приставкой («BKZ_GZ41» → «BKZ_GZ» — нет в RESIST) переходов в выдаче нет, хотя шаблон несёт цепочку ×5.
Здесь по выдаче `rp_vc/N` без прогона пайплайна: для каждой кривой выдачи с цепочкой масштабов в шаблоне (≥ 2 шкал)
уровни считаются заново тем же декодером и тем же сглаживанием, что в emit (`level_lam`, `level_min_run`), и сравниваются
с уровнями эталона на общих строках. Варианты:
  «как есть» — уровни из сегментов выдачи;
  «декодер всем» — decode для всех кривых с цепочкой (как для резистивных);
  «декодер всем + оракул формы» — не используется (зарезервировано).
Счёт — именно-честные (то же имя, 1:1 при 3 px) кривые, у которых уровень верен ≥ 90% общих строк; разрез по семействам.

  _level_whatif.py --sets field,A
"""
import sys, argparse, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, refine
from auto.config import DEFAULT
import decode_levels as DL

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--lam", type=float, default=None)
ap.add_argument("--min-run", type=int, default=None)
a = ap.parse_args()
LAM = DEFAULT.cv.level_lam if a.lam is None else a.lam
MINRUN = DEFAULT.cv.level_min_run if a.min_run is None else a.min_run
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def levels(c):
    out = {}
    for y0, y1, lv in c.get("segments") or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


C = {k: Counter() for k in SETS}
FAMC = {k: defaultdict(Counter) for k in SETS}
for sn, names in SETS.items():
    for sh in names:
        q = SRC.get(sh)
        if not q:
            continue
        stem = q.stem
        pd = Path(a.dir) / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
        if not got:
            continue
        mt = extract(str(q))
        Tc = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        for w in extract(str(got))["curves"]:
            nm = w["name"]
            if nm not in Tc:
                continue
            c = Tc[nm]
            gt = dense(c); tr = dense(w)
            com = [y for y in tr if y in gt]
            if len(com) < 30:
                continue
            e = np.array([abs(tr[y] - gt[y]) for y in com])
            if not (np.median(e) <= 3.0 and len(com) / max(1, len(gt)) >= 0.9):
                continue                                         # только именно-честные по пикселям
            lt = levels(c)
            if not any(v > 0 for v in lt.values()):
                C[sn]["честных без переходов в эталоне"] += 1
                continue
            fam = DL.build_family(mt, c)
            group = "резистивные" if DL.is_resistive(nm) else "НЕ резистивные"
            C[sn][f"честных с переходами ({group})"] += 1
            lw = levels(w)
            xs_by_row = {w["top_y"] + i: float(x) for i, x in enumerate(w["xs"]) if x != NULL}
            if len(fam) >= 2:
                raw = DL.decode(xs_by_row, fam, lam=LAM)
                ld = refine.enforce_min_run(raw, MINRUN)
            else:
                ld = {}
            both = [y for y in com if y in lt]
            acc_now = float(np.mean([lw.get(y, 0) == lt[y] for y in both]))
            # незаданные декодером строки (мосты) наследуют последний уровень, как в emit
            lastv, ldf = 0, {}
            for y in range(min(both), max(both) + 1):
                if y in ld:
                    lastv = ld[y]
                ldf[y] = lastv
            acc_dec = float(np.mean([ldf.get(y, 0) == lt[y] for y in both])) if ld else acc_now
            root = M.mnem_root(nm)
            for tag, acc in (("как есть", acc_now), ("декодер всем", acc_dec)):
                C[sn][(group, tag)] += acc >= 0.9
                FAMC[sn][root][tag] += acc >= 0.9
            FAMC[sn][root]["n"] += 1
            C[sn][(group, "без цепочки в шаблоне")] += len(fam) < 2
for sn in SETS:
    c = C[sn]
    print(f"★ {sn} (lam {LAM}, min_run {MINRUN}): именно-честных без переходов в эталоне {c['честных без переходов в эталоне']}")
    for group in ("резистивные", "НЕ резистивные"):
        n = c[f"честных с переходами ({group})"]
        if n:
            print(f"   {group}: с переходами {n} (без цепочки в шаблоне {c[(group, 'без цепочки в шаблоне')]}); уровень верен ≥ 90% строк: "
                  f"как есть {c[(group, 'как есть')]}, декодер всем {c[(group, 'декодер всем')]}")
    tot_now = c["честных без переходов в эталоне"] + sum(c[(g, "как есть")] for g in ("резистивные", "НЕ резистивные"))
    tot_dec = c["честных без переходов в эталоне"] + sum(c[(g, "декодер всем")] for g in ("резистивные", "НЕ резистивные"))
    print(f"   ⇒ именно-честных с верным уровнем: как есть {tot_now}, декодер всем {tot_dec} ({tot_dec - tot_now:+d})")
    print("   по семействам (n: как есть → декодер всем): " + ", ".join(
        f"{r} {v['n']}: {v['как есть']}→{v['декодер всем']}" for r, v in sorted(FAMC[sn].items(), key=lambda kv: -kv[1]["n"])[:14]))
