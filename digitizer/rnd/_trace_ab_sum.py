r"""_trace_ab_sum.py — сборка шардов `_trace_prod_ab.py` в одну таблицу по отгружаемому пути.

Считает по каждому режиму: честных кривых в ВЫДАННЫХ файлах, листов вверх/вниз против базы и
контроль «режим вообще включился?» — доля листов, где отпечаток выдачи отличается от базы.
⚠ Без этого контроля равные счётчики читаются как «правка нейтральна», хотя правка могла просто
не доехать до кода (§6.71).

  <ComfyUI>\python_embeded\python.exe _trace_ab_sum.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:\nds\output\taskS\prod_ab_trace")
a = ap.parse_args()

# ⚠⚠ ШАРДЫ БЕРУТСЯ ТОЛЬКО ОДНОГО РАЗБИЕНИЯ. Маска `ab_*of*.pkl` однажды загребла дамп ДЫМОВОГО
# прогона (`ab_5of14.pkl`, 3 листа, --shard 5/14): его листы посчитались ВТОРОЙ раз в счётчике `tot`
# и один раз в полистном словаре, отчего две суммы одного и того же прогона разошлись на +1/+3/+3.
# Отсюда два правила: (1) читать один знаменатель `of<N>`, самый свежий; (2) итог считать ПО
# ПОЛИСТНОМУ СЛОВАРЮ, а не по накопителю — словарь идемпотентен к повторам листа.
files = sorted(Path(a.dir).glob("ab_*of*.pkl"))
if not files:
    sys.exit("нет данных")
den = sorted({f.stem.split("of")[1] for f in files},
             key=lambda d: max(f.stat().st_mtime for f in files if f.stem.endswith("of" + d)))[-1]
drop = [f.name for f in files if not f.stem.endswith("of" + den)]
files = [f for f in files if f.stem.endswith("of" + den)]
if drop:
    print(f"⚠ пропущены дампы ДРУГОГО разбиения: {', '.join(drop)}")

RES, FP, modes = {}, {}, None
for f in files:
    d = pickle.load(open(f, "rb"))
    modes = modes or [m[0] for m in d["modes"]]
    for nm in modes:
        t, per = d["res"].get(nm, ({}, {}))
        acc = RES.setdefault(nm, dict(curves=0, leak=0, per={}))
        acc["curves"] += t.get("curves", 0); acc["leak"] += t.get("leak", 0)
        acc["per"].update(per)
        FP.setdefault(nm, {}).update(d["fp"].get(nm, {}))
for nm in RES:
    RES[nm]["hon"] = sum(RES[nm]["per"].values())      # ← по словарю, не по накопителю
    RES[nm]["sheets"] = len(RES[nm]["per"])
print(f"шардов {len(files)} из {den}" + ("  ⚠⚠ ПРОГОН НЕПОЛНЫЙ" if len(files) < int(den) else ""))

base = modes[0]
# ⚠ Сравнение ТОЛЬКО по листам, что есть у ВСЕХ режимов: режим, успевший меньше, иначе выглядел бы
# хуже просто по объёму (§6.71 — «ноль различий значит проверь, что сравниваешь»).
common = set.intersection(*[set(RES[nm]["per"]) for nm in modes])
for nm in modes:
    RES[nm]["hon"] = sum(v for k, v in RES[nm]["per"].items() if k in common)
    RES[nm]["per"] = {k: v for k, v in RES[nm]["per"].items() if k in common}
pa, fa = RES[base]["per"], FP[base]
print(f"листов общих всем режимам {len(pa)}, кривых {RES[base]['curves']}\n")
print(f"{'режим':<18}{'честных':>9}{'Δ':>6}{'листов↑':>9}{'↓':>4}{'выдача≠базы':>13}")
for nm in modes:
    r = RES[nm]
    pb = r["per"]
    up = sum(1 for k in pa if pb.get(k, 0) > pa[k]); dn = sum(1 for k in pa if pb.get(k, 0) < pa[k])
    dif = sum(1 for k in fa if FP[nm].get(k) != fa[k])
    tag = "" if nm == base else (f"{dif}/{len(fa)}" + ("  ⛔ НЕ ВКЛЮЧИЛСЯ" if dif == 0 else ""))
    print(f"{nm:<18}{r['hon']:>9}{r['hon']-RES[base]['hon']:>+6}{up:>9}{dn:>4}{tag:>13}")

for nm in modes[1:]:
    pb = RES[nm]["per"]
    ch = [(k, pa[k], pb.get(k, 0)) for k in pa if pb.get(k, 0) != pa[k]]
    print(f"\n{nm}: изменившихся листов {len(ch)}")
    for k, x, y in sorted(ch, key=lambda q: q[2] - q[1]):
        print(f"  {k[:60]:<62} {x} → {y}")
