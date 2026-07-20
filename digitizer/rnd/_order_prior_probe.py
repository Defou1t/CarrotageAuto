r"""_order_prior_probe.py — ПОСЛЕДНИЙ НЕПРОВЕРЕННЫЙ ПРИОР ЛИЧНОСТИ: устойчивость ПОРЯДКА (§6.20.7-п.2).

_slot_prior_probe.py закрыл scale-ось и value-калибровку как приоры для КРОССИНГ-семейства GZ
(GZ1/GZ2/GZ3 делят и x-полосу, и шкалу). Остаётся один кандидат: ПОРЯДОК слева-направо.
§5.2 мерил его архив-wide как «ordered/crossing по листам», но НЕ как ПОДАВАЕМЫЙ ПОКРИВОЙ приор.

Здесь — прямо: внутри каждого ТРЕКА (группа кривых, делящих scale-ось) для кривых, делящих
value-калибровку (то самое неразделимое ядро), мерим:
  • ORDER-STABILITY: доля строк, где лево-правый порядок кривых = их МЕДИАННОМУ порядку;
  • SWAPS: сколько раз соседи по порядку реально меняются местами (пересечения);
  • при устойчивом порядке приор реален: «слот k = k-я слева кривая в своём треке» —
    его можно ПОДАТЬ band_rescue (нарезать общий горб на под-полосы по рангу).

Читать так: order-stability высокая (>0.8) на многих кривых → приор порядка ЖИВОЙ, стоит
резать горб по рангу. Низкая (кроссинг) → и этот приор мёртв, ядро GZ неразрешимо приором
рамки в принципе, и честный вывод — закрыть §6.20.7-п.2 для GZ и указывать на P2 (декодер).

  python _order_prior_probe.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from _multi_replica_probe import dense
from _slot_prior_probe import SHEETS


def track_groups(mo, gts):
    """Кривые по треку = по (x_left,x_right) их scale-оси."""
    cd = {c["name"]: c for c in mo.get("curve_desc", [])}
    sas = mo.get("scale_axes", [])
    groups = {}
    for g in gts:
        d = cd.get(g["name"])
        sa = sas[d["axis_idx"]] if d and d.get("axis_idx") is not None else None
        key = (sa["x_left"], sa["x_right"]) if sa else ("?", g["name"])
        vkey = (sa["v_left"], sa["v_right"]) if sa else None
        groups.setdefault(key, []).append((g, vkey))
    return groups


def main():
    tot_stab = []
    for tag, n in SHEETS:
        if not n.is_file():
            continue
        mo = extract(str(n))
        gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
        dens = {g["name"]: dense(g) for g in gts}
        groups = track_groups(mo, gts)
        print(f"\n### {tag}")
        for tk, members in groups.items():
            if len(members) < 2:
                g0 = members[0][0]
                print(f"   трек {tk}: одна кривая {g0['name'].split()[0]} — приор не нужен")
                continue
            names = [g["name"].split()[0] for g, _ in members]
            vks = [vk for _, vk in members]
            # общие строки, где размечены ВСЕ члены трека
            common_ys = set.intersection(*[set(dens[g["name"]]) for g, _ in members])
            common_ys = sorted(common_ys)
            if len(common_ys) < 50:
                print(f"   трек {tk}: {names} — мало общих строк ({len(common_ys)})")
                continue
            M = np.array([[dens[g["name"]][y] for g, _ in members] for y in common_ys])  # (rows, curves)
            med_order = np.argsort(M.mean(0))                    # порядок по средней x
            # доля строк, где локальный порядок == медианному
            row_order = np.argsort(M, axis=1)
            match = np.mean([np.array_equal(row_order[i], med_order) for i in range(len(M))])
            # свопы соседей: для соседних по МЕДИАННОМУ порядку кривых — доля строк, где они местами
            swaps = []
            for a, b in zip(med_order, med_order[1:]):
                sw = float(np.mean(M[:, a] > M[:, b]))            # a должна быть ЛЕВЕЕ b
                swaps.append(sw)
            # разделяет ли value-калибровка внутри группы?
            uniq_v = len(set(vks))
            same_v = [names[i] for i in range(len(names))]
            print(f"   трек {tk}: {names}")
            print(f"      value-калибровок уникальных: {uniq_v}/{len(names)}"
                  f"   ★ order-stability (строк с медианным порядком): {match*100:.0f}%")
            print(f"      свопы соседей (доля строк, где сосед левее правого нарушен): "
                  f"{[round(s*100) for s in swaps]}%")
            tot_stab.append((tag, tuple(names), match, uniq_v == len(names)))
    print(f"\n{'='*70}\n=== СВОДКА: приор ПОРЯДКА по группам-кроссингам ===")
    hard = [(t, nm, st) for t, nm, st, vsep in tot_stab if not vsep]  # value НЕ разделяет
    print(f"  групп, где value-калибровка НЕ разделяет (ядро GZ): {len(hard)}")
    for t, nm, st in sorted(hard, key=lambda z: -z[2]):
        verd = "ПОРЯДОК УСТОЙЧИВ" if st >= 0.7 else ("частично" if st >= 0.4 else "КРОССИНГ")
        print(f"     {t:<14} {'/'.join(nm):<24} order-stability {st*100:>3.0f}%  -> {verd}")


if __name__ == "__main__":
    main()
