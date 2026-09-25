r"""_tcache_parity.py — ПОБАЙТНАЯ СВЕРКА ПОВТОРА С КЭША С НАСТОЯЩИМИ ВЫДАЧАМИ A/B (§6.218).

Код выхода 0 — все пары равны; 2 — есть различия (список в выводе). Пары «повтор=эталон» передаются аргументами:
  _tcache_parity.py --out F:/nds/output/taskS/rp_field G=ab_slot/G RA=ab_slot/RA N=ab_names/N
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--out", required=True)
ap.add_argument("pairs", nargs="+")
a = ap.parse_args()
TS = Path(a.ts)
code = 0
for pr in a.pairs:
    m, ref = pr.split("=", 1)
    same = diff = miss = 0; bad = []
    root = Path(a.out) / m
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        x = next(d.glob("*_auto.nlgx"), None); rd = TS / ref / d.name
        y = next(rd.glob("*_auto.nlgx"), None) if rd.is_dir() else None
        if not x or not y:
            miss += 1; continue
        if x.read_bytes() == y.read_bytes():
            same += 1
        else:
            diff += 1; bad.append(d.name)
    ok = diff == 0 and same > 0
    code = code if ok else 2
    print(f"{'★' if ok else '⛔'} повтор {m} против {ref}: равны {same}, различаются {diff}, без пары {miss}")
    for b in bad[:15]:
        print(f"     различие: {b}")
sys.exit(code)
