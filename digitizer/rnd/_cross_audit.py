r"""_cross_audit.py — ЧУЖАЯ ЛИНИЯ НА УЧАСТКАХ: ОБМЕН НА ПЕРЕСЕЧЕНИИ ИЛИ ДУБЛЬ (03.11, §6.353).

Строгая мера (`_strict_stand.py`): у «верных» кривых 69% лишних строк — длинные участки, где трасса стоит на ДРУГОЙ кривой эталона
(Yatskivska ГК на НГК, B_Vol_65 GZ1 на ПС / PZ). Здесь по строкам лишнего (уровень выдачи = уровню эталона, > 10 px от своей линии):
  на чужой кривой эталона g2 (≤ 5 px) — и где в этот момент наша кривая g2: на своей (ДУБЛЬ — две наши кривые на одной туши),
  на линии g (ОБМЕН — кривые поменялись ветками на пересечении), где-то ещё (ЧУЖАЯ);
  не на кривой эталона — СЕТКА / ТЕКСТ / ПРОЧЕЕ.
Потолок обмена (оракул): на строках, где наша g стоит на эталоне g2, а наша g2 — на эталоне g, значения меняются местами → мера
приёмки и чистые до и после. Пиксели (без пересчёта уровней) для «на линии», приёмка — как в `_name_cost_prod.py`.

  _cross_audit.py --workers 2 [--limit 400]   → сводка в stdout, taskS/_cross_<tag>.json
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np

NEAR, OFF = 5.0, 10.0


def one(args):
    nm, part, src, got = args
    import _sample_review as S, decode_levels as DL
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception as e:
        return nm, part, None, f"{type(e).__name__}"
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    names = [g for g in gts if g in wr]
    C = Counter()

    def on(tr, gt_, y):                      # строка y трассы tr на линии gt_ (по строке, ±1 строка эталона)
        if y not in tr:
            return False
        return any(yy in gt_ and abs(tr[y] - gt_[yy]) <= NEAR for yy in (y - 1, y, y + 1))
    swap_rows = {}
    for g in names:
        w, gt = wr[g], gts[g]
        for y in w:
            if y not in gt or LW.get(g, {}).get(y, 0) != LG.get(g, {}).get(y, 0):
                continue
            if abs(w[y] - gt[y]) <= OFF:
                continue
            C["лишних строк"] += 1
            g2 = next((k for k in gts if k != g and on(w, gts[k], y)), None)
            if g2 is None:
                C["не на кривой эталона"] += 1
            elif g2 in wr and on(wr[g2], gts[g2], y):
                C["дубль (наша g2 на своей)"] += 1
            elif g2 in wr and on(wr[g2], gt, y):
                C["обмен (наша g2 на линии g)"] += 1
                swap_rows.setdefault((g, g2), []).append(y)
            else:
                C["на чужой, наша g2 где-то ещё"] += 1

    def accept(tr_map):
        S._TREES.clear()
        return {g: S.HON(*S.err(S.relevel(tr_map[g], LW.get(g, {}), LG.get(g, {}), CH.get(g, [])), gts[g])) for g in names}
    base = accept(wr)
    sw = {g: dict(wr[g]) for g in names}
    for (g, g2), ys in swap_rows.items():
        for y in ys:
            sw[g][y] = wr[g2][y]
    orc = accept(sw)
    C["приёмка база"] = sum(base.values()); C["приёмка обмен-оракул"] = sum(orc.values()); C["кривых"] = len(names)
    return nm, part, dict(C), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="pop")
    a = ap.parse_args()
    import _sample_review as S
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = []
    for part, SS in SETS.items():
        for nm in sorted(SS):
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is not None and nm in SRC:
                jobs.append((nm, part, SRC[nm], got))
    if a.limit:
        jobs = jobs[::max(1, len(jobs) // a.limit)][:a.limit]
    print(f"★ листов {len(jobs)}, процессов {a.workers}")
    from multiprocessing import Pool
    res, t0 = [], time.time()
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            res.append(r)
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    (Path(r"F:/nds/output/taskS") / f"_cross_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    for part in ["поле+вне", "сорт A"]:
        T = Counter()
        for nm, p, c, _ in res:
            if c and (p == "сорт A") == (part == "сорт A"):
                T.update(c)
        L = T["лишних строк"] or 1
        print(f"\n## {part}: кривых {T['кривых']}, лишних строк (уровень совпал) {T['лишних строк']}")
        for k in ["дубль (наша g2 на своей)", "обмен (наша g2 на линии g)", "на чужой, наша g2 где-то ещё", "не на кривой эталона"]:
            print(f"  {k:32s} {T[k]:9d}  {T[k] / L:6.1%}")
        print(f"  приёмка: база {T['приёмка база']}, с обменом-оракулом {T['приёмка обмен-оракул']} "
              f"(+{T['приёмка обмен-оракул'] - T['приёмка база']})")


if __name__ == "__main__":
    main()
