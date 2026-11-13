r"""_pool_ceiling.py — РАСКЛАДКА ПОТЕРЬ ВЫБОРА ТРАССЫ ПО ПУЛУ КАНДИДАТОВ (13.11, §6.364): по сканам выбора трассы (`line_choice_r` поле,
`line_choice_out_r` вне — пул нынешнего прода с путями §6.349) — слоты с эталоном: выдача честна (в пикселях) / в пуле есть честный,
но выбран другой / честного кандидата нет вовсе; по семействам. Метка кандидата `lab[0]` — честен ли он для слота.

  _pool_ceiling.py
"""
import sys, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter, defaultdict

TS = Path(r"F:/nds/output/taskS")


def main():
    for part, d in (("поле", "line_choice_r"), ("вне", "line_choice_out_r")):
        c = Counter(); fam = defaultdict(Counter); ncand = []
        for f in sorted((TS / d).glob("*.pkl")):
            r = pickle.load(open(f, "rb"))
            for s in r["slots"]:
                if not s.get("has_truth"):
                    continue
                c["слотов"] += 1
                hon = [x for x in s["cands"] if x.get("lab") and x["lab"][0]]
                k = "выдача честна" if s.get("out_honest") else ("честный в пуле, выбран другой" if hon else "честного в пуле нет")
                c[k] += 1; fam[s["root"]][k] += 1; fam[s["root"]]["всего"] += 1
                ncand.append(len(s["cands"]))
        n = c["слотов"]
        print(f"★ {part}: слотов с эталоном {n}; кандидатов на слот (медиана) {sorted(ncand)[len(ncand) // 2] if ncand else 0}")
        for k in ("выдача честна", "честный в пуле, выбран другой", "честного в пуле нет"):
            print(f"   {k:32s} {c[k]:5d}  {c[k] / max(n, 1):6.1%}")
        top = sorted(fam.items(), key=lambda kv: -kv[1]["честного в пуле нет"])[:12]
        print("   нет честного по семействам: " + ", ".join(f"{f} {v['честного в пуле нет']}/{v['всего']}" for f, v in top))


if __name__ == "__main__":
    main()
