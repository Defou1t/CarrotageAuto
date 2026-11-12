r"""_nodup_pick.py — ВЫБОР ВАРИАНТА СТЕНДА §6.359 ПО ПРЕДРЕГИСТРИРОВАННОМУ КРИТЕРИЮ (10.11).

Читает `taskS/_ab_<режим>.log` (вывод `_ab_subset.py` NSE22 → режим на листах стенда), берёт лучший по поле+вне именных Δ и проверяет:
поле+вне именных Δ ≥ +15 при p < 0.01, безымянных Δ ≥ 0, сорт A именных Δ ≥ 0. Код выхода 0 — в полный A/B (имя режима — в --out),
1 — ни один не прошёл, 2 — логи не разобраны.

  _nodup_pick.py --modes NDP3 NDS3 NDP5 --out F:/nds/output/taskS/nodup_stand/best.txt
"""
import sys, re, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

TS = Path(r"F:/nds/output/taskS")
PAT = re.compile(r"★ (поле\+вне|сорт A|поле|вне) \((\d+) листов\), (именных|безымянных) честных в значениях: \S+ (-?\d+) → \S+ (-?\d+) "
                 r"\(Δ ([+-]?\d+), листов ↑(\d+)/↓(\d+), p = ([\d.]+)\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--need", nargs="*", default=["сорт A"], help="части, где именных Δ ≥ 0 обязательно (§6.361: вне и сорт A)")
    a = ap.parse_args()
    R = {}
    for m in a.modes:
        f = TS / f"_ab_{m}.log"
        if not f.exists():
            print(f"⛔ нет {f}"); sys.exit(2)
        d = {(g[0], g[2]): (int(g[5]), float(g[8]), int(g[1])) for g in PAT.findall(f.read_text(encoding="utf-8"))}
        if ("поле+вне", "именных") not in d or ("сорт A", "именных") not in d:
            print(f"⛔ лог {f} не разобран"); sys.exit(2)
        R[m] = d
        print(f"  {m}: поле+вне именных {d[('поле+вне', 'именных')][0]:+d} (p = {d[('поле+вне', 'именных')][1]}, листов "
              f"{d[('поле+вне', 'именных')][2]}), безымянных {d[('поле+вне', 'безымянных')][0]:+d}, сорт A именных "
              f"{d[('сорт A', 'именных')][0]:+d} (листов {d[('сорт A', 'именных')][2]})")
    best = max(R, key=lambda m: (R[m][("поле+вне", "именных")][0], -R[m][("поле+вне", "именных")][1]))
    d = R[best]
    ok = (d[("поле+вне", "именных")][0] >= 15 and d[("поле+вне", "именных")][1] < 0.01 and d[("поле+вне", "безымянных")][0] >= 0
          and all(d[(pt, "именных")][0] >= 0 for pt in a.need))
    print(f"★ лучший {best}: критерий стенда §6.359 {'ВЗЯТ — в полный A/B' if ok else 'НЕ ВЗЯТ'}")
    if ok:
        tmp = Path(a.out + ".tmp"); tmp.write_text(best + "\n", encoding="utf-8"); tmp.replace(a.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
