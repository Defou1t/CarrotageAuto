r"""_u1_map_audit.py — ПРОТИВОРЕЧИЕ §6.29/§6.30 vs ПРОД: КАКУЮ ЛИНИЮ `emit` САЖАЕТ В СЛОТ.

Замеренное противоречие: по §6.30 полоса U1 центрирована на кривой в пределах med 65px и
содержит 96% точек эксперта; по §6.29 такая полоса обязана давать 16-80px. Прод (FLAG снят,
оконный селектор) даёт **586px**. Между «полоса годная» и «результат» есть ещё звено.

Главный подозреваемый — МАППИНГ: §6.30 брал ЛУЧШУЮ линию по полосе, а `emit._map_lines_to_slots`
выбирает по классу/порядку/цвету и может посадить в слот другую. Здесь это проверяется прямо, по
уже записанным `*_auto.nlgx` (пайплайн не перезапускается):

  для каждого слота: медиана x ЗАПИСАННОЙ трассы vs медиана x ЭКСПЕРТНОЙ кривой этого слота
  → и то же расстояние до КАЖДОЙ экспертной кривой листа. Если записанная трасса ближе к ЧУЖОЙ
  кривой, чем к своей — это ошибка маппинга (или латч), и полоса тут ни при чём.

  python _u1_map_audit.py [--dir <каталог с *_auto.nlgx>]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from _multi_replica_probe import dense
from _relatch_bench import SH, ARCH

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\prod_ab_seq\B_seq")
a = ap.parse_args()
D = Path(a.dir)

print(f"каталог: {D}")
print(f"{'скважина':<12}{'слот':<8}{'|наша-своя|':>12}{'|наша-ближ|':>12}{'ближайшая':<10}"
      f"{'вердикт':<28}")
same = other = 0
for rel in SH:
    n = ARCH / rel
    well = n.parent.parent.name
    au = D / (n.stem + "_auto.nlgx")
    if not au.is_file():
        print(f"{well:<12} нет {au.name}"); continue
    gt = extract(str(n)); at = extract(str(au))
    gts = {c["name"]: c for c in ds.real_curves(gt) if any(x != NULL for x in c["xs"])}
    dens = {k: dense(v) for k, v in gts.items()}
    med = {k: float(np.median([v[y] for y in sorted(v)])) for k, v in dens.items() if v}
    for c in at["curves"]:
        nm = c["name"]
        if nm not in med:
            continue
        xs = [x for x in c["xs"] if x != NULL]
        if len(xs) < 50 or list(c["xs"]) == list(gts[nm]["xs"]):     # ловушка §4: эталон не тронут
            continue
        ax = float(np.median(xs))
        d_own = abs(ax - med[nm])
        near = min(med, key=lambda k: abs(ax - med[k]))
        d_near = abs(ax - med[near])
        if near == nm:
            same += 1; verdict = "своя ближайшая"
        else:
            other += 1; verdict = f"БЛИЖЕ К ЧУЖОЙ ({near.split()[0]})"
        print(f"{well:<12}{nm.split()[0]:<8}{d_own:>12.0f}{d_near:>12.0f}"
              f"{near.split()[0]:<10}{verdict:<28}")
t = max(1, same + other)
print(f"\nслотов разобрано: {same+other}")
print(f"  трасса ближе всего к СВОЕЙ кривой: {same} ({100*same/t:.0f}%)")
print(f"  трасса ближе к ЧУЖОЙ (ошибка маппинга/латч): {other} ({100*other/t:.0f}%)")
print("\nЧитать: если большинство «ближе к чужой» — узкое место в ВЫБОРЕ ЛИНИИ ПОД СЛОТ (emit),")
print("а не в полосе (§6.30) и не в трассировке (§6.29). Это чинится без ML.")
