r"""_pool_vs_pop.py — §6.364: ошибки прода (причины по кривым эталона NSE22, `sample_2026-11-10/pop_curves.tsv`) против пула кандидатов
(`line_choice_r` / `_out_r`: есть ли у слота честный кандидат в пикселях). Отбор (честный был) против порождения (честного нет).

  _pool_vs_pop.py
"""
import sys, csv, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter, defaultdict

TS = Path(r"F:/nds/output/taskS")


def main():
    pool = {}
    for d in ("line_choice_r", "line_choice_out_r"):
        for f in (TS / d).glob("*.pkl"):
            r = pickle.load(open(f, "rb"))
            for s in r["slots"]:
                if s.get("has_truth"):
                    pool[(r["sheet"], s["name"])] = any(x.get("lab") and x["lab"][0] for x in s["cands"])
    rows = list(csv.reader(open(r"F:/nds/output/sample_2026-11-10/pop_curves.tsv", encoding="utf-8"), delimiter="\t"))[1:]
    agg = defaultdict(Counter)
    for part, nm, curve, tr, why, *_ in rows:
        if part == "сорт A":
            continue
        k = pool.get((nm, curve))
        agg[why]["нет в пуле" if k is None else ("честный в пуле" if k else "честного нет")] += 1
    print("★ поле+вне, NSE22: причина по эталону × пул кандидатов слота")
    tot = Counter()
    for why, c in sorted(agg.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values()); tot.update(c)
        print(f"   {why:28s} {n:5d}: честный в пуле {c['честный в пуле']:5d}, честного нет {c['честного нет']:4d}, слота нет в скане {c['нет в пуле']:4d}")
    bad = {k: v for k, v in agg.items() if k != "верно"}
    s1 = sum(c["честный в пуле"] for c in bad.values()); s2 = sum(c["честного нет"] for c in bad.values())
    print(f"★ неверных: честный кандидат БЫЛ — {s1} (отбор / уровень / имя), честного НЕ БЫЛО — {s2} (порождение)")


if __name__ == "__main__":
    main()
