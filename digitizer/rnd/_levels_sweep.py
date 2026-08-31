r"""_levels_sweep.py — СВИП ЦЕНЫ ПЕРЕХОДА ПО УЖЕ ВЫДАННЫМ ФАЙЛАМ (Задача 10; из §6.138).

§6.138 намерил на отгрузке: из 1346 кривых, ГОТОВЫХ по геометрии, уровень верен везде у 962
(71.5%); переход у эксперта был у 373 кривых, мы не нашли 162 (43%), а выдумали только 13.
★ Асимметрия 162:13 — это не шум, а СМЕЩЕНИЕ: декодер слишком консервативен, то есть цена перехода
стоит слишком высоко. Проверяется свипом, и свип этот НЕ ТРЕБУЕТ прогонов пайплайна: геометрия
уже лежит в выданных `*_auto.nlgx`, а уровни переигрываются по ней офлайн за секунды.

⚠⚠ ЧТО ЗДЕСЬ НЕЛЬЗЯ ПЕРЕПУТАТЬ. Прод декодирует уровни в `emit._level_segments` (DP `decode_levels`
+ `refine.enforce_min_run`), и свип обязан повторять ОБА шага, иначе он померит не тот механизм.
Семейство шкал строится из ВЫДАННОГО каркаса (`DL.build_family`) — тем же вызовом, что в проде.

МЕТРИКА — та же, что в §6.138: доля кривых, у которых уровень верен ВЕЗДЕ, и рядом обязательный
контроль «всегда level 0» (§6.34: без него построчная точность 90% выглядит отличной, хотя
ничего-не-делание даёт 89.6%).

  <ComfyUI>\python_embeded\python.exe _levels_sweep.py --dir F:/nds/output/taskS/ab_both_fix --mode B --cap 300
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
import decode_levels as DL
from auto import refine
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_both_fix")
ap.add_argument("--mode", default="B")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--cap", type=int, default=300)
ap.add_argument("--lam", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.7])
ap.add_argument("--gate-w", nargs="+", type=float, default=[4.0])
ap.add_argument("--bias", nargs="+", type=float, default=[0.0])
a = ap.parse_args()
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def levels_by_row(curve):
    out = {}
    for s in curve.get("segments", []):
        try:
            y0, y1, lv = int(s[0]), int(s[1]), int(s[2])
        except Exception:
            continue
        for y in range(y0, y1 + 1):
            out[y] = lv
    return out


def main():
    WLG = {}
    for root in a.roots:
        for wlg in Path(root).glob("*/wlg"):
            for q in wlg.glob("*.nlgx"):
                WLG.setdefault(q.stem, q)
    for q in Path(r"F:\nds\projects\Semeguniv_001\wlg").glob("*.nlgx"):
        WLG.setdefault(q.stem, q)

    dirs = sorted(d for d in (Path(a.dir) / a.mode).iterdir() if d.is_dir())[:a.cap or None]
    print(f"каталогов выдачи: {len(dirs)}")

    # ── СНАЧАЛА СОБИРАЕМ ЗАДАЧИ: кривая честная по геометрии + её семейство шкал ────────────
    jobs, n_sheets = [], 0
    for c, d in enumerate(dirs, 1):
        em = list(d.glob("*_auto.nlgx"))
        if not em:
            continue
        q = WLG.get(em[0].stem[:-5])
        if q is None:
            continue
        try:
            ours = extract(str(em[0])); gt = extract(str(q))
        except Exception:
            continue
        n_sheets += 1
        og = [(cu, dense(cu)) for cu in ours["curves"] if any(x != NULL for x in cu["xs"])]
        gg = [(cu, dense(cu)) for cu in gt["curves"] if any(x != NULL for x in cu["xs"])]
        used = set()
        for cu, dn in og:
            best, bj, bcov = None, None, 0.0
            for j, (gcu, gdn) in enumerate(gg):
                if j in used:
                    continue
                com = [y for y in dn if y in gdn]
                if len(com) < 30:
                    continue
                m = float(np.median([abs(dn[y] - gdn[y]) for y in com]))
                if best is None or m < best:
                    best, bj, bcov = m, j, len(com) / max(1, len(gdn))
            if bj is None or not HON(best, bcov):
                continue
            used.add(bj)
            if not DL.is_resistive(cu["name"]):
                continue
            try:
                fam = DL.build_family(ours, cu)
            except Exception:
                continue
            if len(fam) < 2:
                continue
            gl = levels_by_row(gg[bj][0])
            if len(gl) < 30:
                continue
            jobs.append((dn, fam, gl, levels_by_row(cu)))
        if c % 100 == 0:
            print(f"  {c}/{len(dirs)}  задач {len(jobs)}")
    print(f"★ листов {n_sheets}, кривых с цепочкой шкал и честной геометрией: {len(jobs)}")
    if not jobs:
        sys.exit("нечего свипать")

    # ── КОНТРОЛЬ «ВСЕГДА 0» ────────────────────────────────────────────────────────────────
    z_all = sum(1 for j in jobs if all(v == 0 for v in j[2].values()))
    rows_all = sum(len(j[2]) for j in jobs)
    rows_zero = sum(sum(1 for v in j[2].values() if v == 0) for j in jobs)
    # ⚠ ЧТО ФАКТИЧЕСКИ ВЫДАЛ ПРОД на этих же кривых — без него свип не с чем сравнивать:
    # «лучше, чем всегда 0» и «лучше, чем прод» — разные утверждения (§6.78).
    pk = pr = 0
    for dn, fam, gl, ol in jobs:
        com = [y for y in ol if y in gl]
        if len(com) < 30:
            continue
        eq = sum(1 for y in com if ol[y] == gl[y])
        pk += int(eq == len(com)); pr += eq
    print(f"★ ПРОД НА ЭТИХ ЖЕ КРИВЫХ: верно везде {pk} ({100*pk/len(jobs):.1f}%), "
          f"строк верно {100*pr/max(1,rows_all):.1f}%")
    print(f"⚠ КОНТРОЛЬ «всегда level 0»: кривых верных везде {z_all} "
          f"({100*z_all/len(jobs):.1f}%), строк верных {100*rows_zero/max(1,rows_all):.1f}%")

    print(f"\n{'lam':>6}{'gate_w':>8}{'bias':>7}{'верно везде':>14}{'доля':>8}"
          f"{'строк верно':>13}{'пропущено пер.':>16}{'выдумано':>10}")
    prod = (DEFAULT.cv.level_lam, 4.0, 0.0)
    for lam in a.lam:
        for gw in a.gate_w:
            for bs in a.bias:
                ok_all = rows_ok = miss = inv = 0
                for dn, fam, gl, _ol in jobs:
                    xs_by_row = {int(y): float(x) for y, x in dn.items()}
                    try:
                        raw = DL.decode(xs_by_row, fam, lam=lam, gate_w=gw, level_bias=bs)
                        lv = refine.enforce_min_run(raw, DEFAULT.cv.level_min_run)
                    except Exception:
                        continue
                    com = [y for y in lv if y in gl]
                    if len(com) < 30:
                        continue
                    eq = sum(1 for y in com if lv[y] == gl[y])
                    rows_ok += eq
                    ok_all += int(eq == len(com))
                    ng = len({gl[y] for y in com}) - 1
                    no = len({lv[y] for y in com}) - 1
                    miss += int(ng > 0 and no == 0)
                    inv += int(ng == 0 and no > 0)
                mark = "  ← прод" if (lam, gw, bs) == prod else ""
                print(f"{lam:>6.2f}{gw:>8.1f}{bs:>7.2f}{ok_all:>14}"
                      f"{100*ok_all/len(jobs):>7.1f}%{100*rows_ok/max(1,rows_all):>12.1f}%"
                      f"{miss:>16}{inv:>10}{mark}")
    print("\n⚠ Свип идёт по УЖЕ ВЫДАННОЙ геометрии: он показывает потолок правки уровней при"
          "\n  неизменной трассировке. Перенос на прод требует прогона пайплайна (§6.123).")


main()
