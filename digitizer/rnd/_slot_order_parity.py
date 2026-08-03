r"""_slot_order_parity.py — ПАРИТЕТ КЛЮЧА ПОРЯДКА: прод против стенда, побитово.

ЗАЧЕМ. `CVParams.slot_order` (§6.105) вносит в прод (`emit._slot_order_key`) те ключи сортировки,
которые намерены на стенде (`_slot_rules.key_x` / `feats`). Формула продублирована в двух местах,
и это ровно та ситуация, из-за которой ветка уже дважды меряла не то: если `rough_n` в проде считать
хоть чуть иначе (окно свёртки, перцентили, деление на размах), A/B на отгрузке померит ДРУГОЕ
правило, отработает и выдаст правдоподобное число — §6.100 в чистом виде.

ЧТО СРАВНИВАЕТСЯ. Для каждой трассы всех пулов: `x_center`, `med_x`, `rough_n`. Сравнение ТОЧНОЕ
(`==` по float, не `allclose`): обе стороны считают одно и то же одной и той же библиотекой, и любое
расхождение здесь — расхождение формулы, а не арифметики. Расхождения печатаются с примерами.

⚠ Что этот стенд НЕ проверяет: сам цикл раздачи слотов. Его эквивалентность правилу подтверждена
иначе — `_slot_rules.py` в режиме `prod` даёт 147 честных, ровно столько же дают `_slot_cause.py`
(«записана верно») и §6.91 на той же выборке.

  <ComfyUI>\python_embeded\python.exe _slot_order_parity.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path

from auto import emit as E
import _slot_rules as R

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    r"F:/nds/output/taskS/pools", r"F:/nds/output/taskS/pools_gate",
    r"F:/nds/output/taskS/pools_wide", r"F:/nds/output/taskS/pools_more",
    r"F:/nds/output/taskS/pools_div"])
a = ap.parse_args()


class FakeLine:
    """Ровно то, что `_slot_order_key` берёт от линии."""
    __slots__ = ("x_center",)

    def __init__(self, x):
        self.x_center = x


# прод ← → стенд
PAIRS = [("x_center", "center"), ("med_x", "med"), ("rough_n", "rough_n")]

seen, files = set(), []
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem); files.append(f)

bad = {p: 0 for p, _ in PAIRS}
examples = {p: [] for p, _ in PAIRS}
ntr = 0
for i, f in enumerate(files, 1):
    d = pickle.load(open(f, "rb"))
    R._FEAT.clear()          # ⚠ кэш стенда ключуется id(tr) — сбрасывать на лист (см. `_slot_rules`)
    for ln in d["lines"]:
        ntr += 1
        L = FakeLine(ln["x_center"])
        for prod_how, stand_how in PAIRS:
            v_prod = E._slot_order_key(L, ln["tr"], prod_how)
            v_stand = R.key_x(ln, stand_how)
            if v_prod != v_stand:
                bad[prod_how] += 1
                if len(examples[prod_how]) < 5:
                    examples[prod_how].append((d["name"][:38], v_prod, v_stand))
    if i % 100 == 0:
        print(f"  … {i}/{len(files)} листов, трасс {ntr}")

print(f"\n{'='*78}\nПАРИТЕТ КЛЮЧА ПОРЯДКА: листов {len(files)}, трасс {ntr}\n{'='*78}")
for prod_how, stand_how in PAIRS:
    n = bad[prod_how]
    print(f"  {prod_how:<10} против стендового {stand_how:<10} расхождений {n:>6}"
          f"   {'★ ЧИСТО' if n == 0 else '⛔ ФОРМУЛЫ РАЗОШЛИСЬ'}")
    for nm, vp, vs in examples[prod_how]:
        print(f"      {nm:<40} прод {vp!r}  стенд {vs!r}")
sys.exit(1 if any(bad.values()) else 0)
