r"""_slot_gate_sweep.py — СВИП ПОРОГА ОТКАЗА РАСКЛАДКИ НА ПОЛНОМ КОРПУСЕ (пересъёмка шага 2 §6.108).

⚠⚠ ЗАЧЕМ. `slot_gate = frac0.2` выбран в §6.108 свипом по **265 листам** держанного набора, и там же
записано «оптимум НЕ на краю: `frac0.0` уже хуже `frac0.2`». С тех пор корпус вырос до **2677 листов**,
а §6.108 сам предупреждал, что прежний порог `frac0.8` был подобран под старый архив и душил механизм
ровно там, где данные свежие. Тот же вопрос теперь законно задать и к `frac0.2`.

ЧТО СЧИТАЕТСЯ. Один проход по дампам, все пороги разом (`assign` дёшев, дорого чтение пулов):
для каждого гейта — сколько кривых записано честно, сколько листов отказано, и лист-в-лист против
ПРАВИЛА (то, что лежит в `written` дампа).
★ КОНТРОЛЬ: колонка правила обязана дать 832 честных на полном корпусе — это же число независимо
получили `_slot_cause` (§6.109) и `_cov_cause` (§6.111). Разойдётся — читается не тот объект.
⚠ Замер ПУЛОВЫЙ: `_pool_oracle` берёт трассу в момент раскладки, выданный файл проходит ещё уровни и
ветку масштаба (§6.108: разница ~4.5 п.п.). Решение о проде принимается только по отгрузке
(`_trace_prod_ab.py` с ключами `gate=`), этот свип лишь называет кандидата.

  python _slot_gate_sweep.py [--model slot_model_g250.npz]
"""
import sys, argparse, pickle, collections
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from auto import slot_model as SM

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--model", default="slot_model_g250.npz")
ap.add_argument("--gates", nargs="+", default=[
    "frac0.0", "frac0.1", "frac0.2", "frac0.3", "frac0.4", "frac0.6", "frac0.8",
    "mean0.0", "mean0.05", "mean0.2", "minx0.0"])
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


class FakeLine:
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


w = SM.load(a.model)
GT = [(g, *SM._parse_gate(g)) for g in a.gates]
print(f"вес {a.model}, порогов {len(GT)}")

acc = {g: dict(hon=0, abst=0, up=0, dn=0) for g, _, _ in GT}
rule_hon = 0
nfile = nsheet = ndup = 0
seen = set()

for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        nfile += 1
        if f.stem in seen:
            ndup += 1; continue
        seen.add(f.stem)
        if a.limit and nsheet >= a.limit:
            continue
        d = pickle.load(open(f, "rb"))
        nsheet += 1
        gts = d["gts"]
        by_track = {}
        for ln in d["lines"]:
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in d["slots"]]
        r_sheet = sum(1 for nm, gt in gts.items()
                      if nm in d["written"] and HON(*err(d["written"][nm], gt)))
        rule_hon += r_sheet
        for g, kind, thr in GT:
            got = SM.assign(slots, by_track, w, kind, thr) if slots else None
            if got is None:                      # отказ ⇒ пишет правило, ровно как в проде
                acc[g]["abst"] += 1
                acc[g]["hon"] += r_sheet
                continue
            h = sum(1 for nm, (L, tr, _) in got.items()
                    if nm in gts and HON(*err(tr, gts[nm])))
            acc[g]["hon"] += h
            acc[g]["up"] += h > r_sheet
            acc[g]["dn"] += h < r_sheet

W = 92
print(f"{'='*W}")
print(f"ВЫБОРКА: файлов {nfile}, прочитано {nsheet}, дублей {ndup}"
      + (f"  ⚠ ОГРАНИЧЕНО --limit {a.limit}" if a.limit else ""))
print(f"  СВЕРКА: {nsheet} + {ndup} = {nsheet+ndup} против {nfile}   "
      + ("★ СОШЛОСЬ" if (nsheet + ndup == nfile) or a.limit else "⛔ РАСХОЖДЕНИЕ"))
print(f"★ КОНТРОЛЬ ПРАВИЛА: честных {rule_hon}"
      + ("   ★ СОШЛОСЬ с §6.109/§6.111 (832)" if rule_hon == 832 and not a.limit else
         "   ⚠ сверить с 832 на полном корпусе" if not a.limit else ""))
print(f"{'='*W}")
print(f"{'гейт':<12}{'честных':>9}{'Δ к правилу':>13}{'листов ↑':>10}{'↓':>6}{'отказано листов':>17}")
print(f"{'правило':<12}{rule_hon:>9}{'—':>13}{'—':>10}{'—':>6}{'—':>17}")
best = max(acc.items(), key=lambda q: q[1]["hon"])
for g, _, _ in GT:
    v = acc[g]
    mark = " ★" if g == best[0] else ("  ← в проде" if g == "frac0.2" else "")
    print(f"{g:<12}{v['hon']:>9}{v['hon']-rule_hon:>+13}{v['up']:>10}{v['dn']:>6}"
          f"{v['abst']:>10} ({100*v['abst']/max(1,nsheet):>3.0f}%){mark}")
print(f"\n⇒ лучший на пулах: {best[0]} ({best[1]['hon']}), в проде frac0.2 "
      f"({acc.get('frac0.2', {}).get('hon', 0)})")
print("⚠ Это ПУЛЫ. Прод меняется только после A/B на отгрузке: "
      '_trace_prod_ab.py --mode "A:...,gate=frac0.2" --mode "B:...,gate=<кандидат>"')
