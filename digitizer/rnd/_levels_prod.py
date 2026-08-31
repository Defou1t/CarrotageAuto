r"""_levels_prod.py — ЦЕНА ПЕРЕХОДОВ МАСШТАБА НА ОТГРУЖАЕМОМ ПУТИ (Задача 8; §6.34 мерил не там).

§6.34 намерил, что понимание переходов стоит corr с LAS 0.267 → 0.971, а декодер берёт меньше
трети (0.414). ⚠ Но мерилось это по ЭКСПЕРТНОЙ трассе (геометрия идеальна), то есть на входе,
которого в проде нет. На РЕАЛЬНОЙ выдаче переходы не мерились ни разу — и пиксельная метрика ветки
их не видит вовсе: кривая может быть честной по |Δx| и при этом смещённой на целый перевынос ×5.

ЧТО СЧИТАЕТСЯ — по выданным `*_auto.nlgx` против экспертного nlgx того же листа:
  1. геометрия: med|Δx| и cov (та же метрика ветки), матчинг кривых 1:1 БЕЗ ИМЁН;
  2. уровни: для кривых, ЧЕСТНЫХ ПО ГЕОМЕТРИИ, — совпадает ли уровень построчно (тег 35498);
  3. контроль «всегда level 0» — сколько даёт ничего-не-делание (§6.34: acc обманчива без него);
  4. переходы: сколько их у эксперта, сколько у нас, и сколько кривых теряет ХОТЬ ОДИН.

⇒ Отвечает на вопрос, который ветка обходила: сколько ГОТОВЫХ кривых портится не геометрией,
а уровнем.

  <ComfyUI>\python_embeded\python.exe _levels_prod.py --dir F:/nds/output/taskS/ab_both_fix --mode B
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_both_fix")
ap.add_argument("--mode", default="B")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--cap", type=int, default=0)
a = ap.parse_args()
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def levels_by_row(curve):
    """segments [(y0, y1, level)] → dict row→level."""
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

    dirs = sorted(d for d in (Path(a.dir) / a.mode).iterdir() if d.is_dir())
    if a.cap:
        dirs = dirs[:a.cap]
    print(f"каталогов выдачи: {len(dirs)}")

    tot = defaultdict(int); rows_ok = rows_n = rows_zero = 0
    miss = defaultdict(int)
    for c, d in enumerate(dirs, 1):
        em = list(d.glob("*_auto.nlgx"))
        if not em:
            miss["нет выдачи"] += 1; continue
        stem = em[0].stem[:-5]                       # без "_auto"
        q = WLG.get(stem)
        if q is None:
            cand = [k for k in WLG if k.startswith(stem[:40])]
            q = WLG[cand[0]] if len(cand) == 1 else None
        if q is None:
            miss["нет эталона"] += 1; continue
        try:
            ours = extract(str(em[0])); gt = extract(str(q))
        except Exception as e:
            miss[f"разбор: {type(e).__name__}"] += 1; continue

        og = [(cu, dense(cu)) for cu in ours["curves"] if any(x != NULL for x in cu["xs"])]
        gg = [(cu, dense(cu)) for cu in gt["curves"] if any(x != NULL for x in cu["xs"])]
        if not og or not gg:
            continue
        # ── матчинг 1:1 БЕЗ ИМЁН (решение Эдуарда 20.08) ───────────────────────────────────
        used = set()
        for cu, dn in og:
            best, bj = None, None
            for j, (gcu, gdn) in enumerate(gg):
                if j in used:
                    continue
                com = [y for y in dn if y in gdn]
                if len(com) < 30:
                    continue
                m = float(np.median([abs(dn[y] - gdn[y]) for y in com]))
                if best is None or m < best:
                    best, bj, bcov = m, j, len(com) / max(1, len(gdn))
            if bj is None:
                continue
            used.add(bj)
            tot["кривых сопоставлено"] += 1
            if not HON(best, bcov):
                continue
            tot["★ честных по геометрии"] += 1
            # ── УРОВНИ на честных по геометрии ─────────────────────────────────────────────
            ol, gl = levels_by_row(cu), levels_by_row(gg[bj][0])
            com = [y for y in ol if y in gl]
            if len(com) < 30:
                tot["уровней нет"] += 1; continue
            eq = sum(1 for y in com if ol[y] == gl[y])
            z0 = sum(1 for y in com if gl[y] == 0)
            rows_ok += eq; rows_n += len(com); rows_zero += z0
            ntr_g = len({gl[y] for y in com}) - 1
            ntr_o = len({ol[y] for y in com}) - 1
            tot["★ уровень совпал везде"] += int(eq == len(com))
            tot["у эксперта есть переход"] += int(ntr_g > 0)
            tot["переход есть, но мы его не нашли"] += int(ntr_g > 0 and ntr_o == 0)
            tot["перехода нет, а мы выдумали"] += int(ntr_g == 0 and ntr_o > 0)
        if c % 100 == 0:
            print(f"  {c}/{len(dirs)}  честных по геометрии {tot['★ честных по геометрии']}")

    print(f"\n{'':-<70}")
    for k, v in tot.items():
        print(f"{k:<38}{v:>8}")
    for k, v in miss.items():
        print(f"⚠ {k:<36}{v:>8}")
    if rows_n:
        print(f"\nстрок с уровнями {rows_n:,}")
        print(f"★ уровень верен: {100*rows_ok/rows_n:.1f}%")
        print(f"  контроль «всегда 0»: {100*rows_zero/rows_n:.1f}%  "
              f"⚠ без него acc обманчива (§6.34)")
    h = tot["★ честных по геометрии"]
    if h:
        print(f"\n⇒ из {h} кривых, ГОТОВЫХ по геометрии, уровень верен везде у "
              f"{tot['★ уровень совпал везде']} ({100*tot['★ уровень совпал везде']/h:.1f}%)")
        print(f"⇒ ЦЕНА ПЕРЕХОДОВ НА ВЫДАЧЕ: {h - tot['★ уровень совпал везде']} кривых "
              f"({100*(h-tot['★ уровень совпал везде'])/h:.1f}%) правильны геометрически, "
              f"но испорчены уровнем")


main()
