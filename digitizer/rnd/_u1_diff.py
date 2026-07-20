r"""_u1_diff.py — A/B двух прогонов _u1_gate.py: что изменилось ПОЛИСТНО и покривой.

  python _u1_diff.py baseline after     # сравнивает u1_all_baseline.json и u1_all_after.json
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

OUT = Path(r"F:\nds\output\taskS\u1")
a_tag, b_tag = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("baseline", "after")
A = {r["sheet"]: r for r in json.loads((OUT / f"u1_all_{a_tag}.json").read_text(encoding="utf-8"))}
B = {r["sheet"]: r for r in json.loads((OUT / f"u1_all_{b_tag}.json").read_text(encoding="utf-8"))}

same = changed = 0
for s in A:
    if s not in B:
        print(f"  ?? {s}: только в {a_tag}"); continue
    ra, rb = A[s], B[s]
    if "err" in ra or "err" in rb:
        if ra.get("err") != rb.get("err"):
            print(f"  !! {s[:50]}: err {ra.get('err')} -> {rb.get('err')}"); changed += 1
        continue
    ka = (ra["class"], ra["n_lines"], ra["band_found"])
    kb = (rb["class"], rb["n_lines"], rb["band_found"])
    if ka == kb:
        same += 1
        continue
    changed += 1
    print(f"  {s[:52]:<54} {ra['class']:<7}->{rb['class']:<7} lines {ra['n_lines']:>3}->{rb['n_lines']:<3} "
          f"полос {ra['band_found']}/{ra['n_curves']} -> {rb['band_found']}/{rb['n_curves']}")
    for ca, cb in zip(ra["curves"], rb["curves"]):
        if (ca["band"], ca["conf"]) != (cb["band"], cb["conf"]):
            print(f"      {ca['name']:<8} band {ca['band']}->{cb['band']}  conf {ca['conf']}->{cb['conf']}")

na = sum(r["band_found"] for r in A.values() if "err" not in r)
nb = sum(r["band_found"] for r in B.values() if "err" not in r)
nc = sum(r["n_curves"] for r in A.values() if "err" not in r)
from collections import Counter
ca = Counter(r["class"] for r in A.values() if "err" not in r)
cb = Counter(r["class"] for r in B.values() if "err" not in r)
print(f"\n  листов без изменений: {same}, изменилось: {changed}")
print(f"  классы: {dict(ca)} -> {dict(cb)}")
print(f"  ★ ПОЛОСА НАЙДЕНА: {na}/{nc} -> {nb}/{nc}")
