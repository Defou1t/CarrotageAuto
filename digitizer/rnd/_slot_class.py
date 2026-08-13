r"""_slot_class.py — ЧТО ИМЕННО ЛОМАЕТСЯ В КОРЗИНЕ «КЛАСС» (§6.110).

⚠⚠ ЗАЧЕМ. `_slot_cause.py` на полном корпусе (2677 листов, 2026 кривых с честной трассой в пуле)
разложил потери раскладки так: ПОРЯДОК 976 (82% потерь), **КЛАСС 138 (12%)**, слот пуст 62, ЦВЕТ 18.
Порядок закрыт как направление (§6.105: лучшее из 19 правил берёт 7% резерва) и лечится обученным
ранжировщиком. А корзина КЛАСС — вторая по величине, и она НЕ лечится ранжировщиком: слот с
несовпавшим классом до жадной раздачи просто не доходит. Раньше в ней было 33 кривые, и она терялась
в шуме; на выросшем корпусе её видно.

ЧТО ЗДЕСЬ СЧИТАЕТСЯ (офлайн по дампам пулов, пайплайн не гоняется):
  • НАПРАВЛЕНИЕ ошибки: слот ждёт RES, а линия помечена SP — или наоборот. Это разные дефекты:
    «гладкая резистивная» и «дёрганая SP» ломаются по разным причинам;
  • по МНЕМОНИКЕ слота — страдает ли весь класс кривых или отдельные зонды;
  • ПРИЗНАКИ линии (ВЧ-дрожь, размах, число разворотов) у ошибочно помеченных против верно
    помеченных того же класса. Если распределения расходятся — метку можно чинить порогом;
    если совпадают — `behavior` не различает эти линии в принципе, и чинить надо не порог.

⚠ Объём — ИЗ СЧЁТЧИКОВ, в конце сверка «обработано + пропущено против длины списка» (§6.106).

  <ComfyUI>\python_embeded\python.exe _slot_class.py --pools <каталоги…>
"""
import sys, argparse, pickle, collections
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", required=True)
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def cls_of(ln):                       # ИДЕНТИЧНО `_slot_cause.cls_of` и проду (`emit`)
    return "SP" if ln["behavior"] == "smooth" else "RES"


def shape(tr):
    """ВЧ-дрожь, размах, доля разворотов — те же величины, что у признаков раскладки."""
    x = np.array([tr[y] for y in sorted(tr)], float)
    if len(x) < 5:
        return 0.0, 1.0, 0.0
    dx = np.diff(x)
    span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
    rev = float(np.mean((dx[:-1] * dx[1:]) < 0)) if len(dx) > 2 else 0.0
    return float(np.median(np.abs(dx))), span, rev


FILES = [f for root in a.pools for f in sorted(Path(root).glob("*.pkl"))]
print(f"список: {len(FILES)} дампов из {len(a.pools)} каталогов")

seen, done = set(), 0
SKIP = collections.Counter()
direction = collections.Counter()
by_mnem = collections.Counter()
feat = {"ошибочно": [], "верно": []}

for f in FILES:
    if f.stem in seen:
        SKIP["дубль дампа"] += 1
        continue
    seen.add(f.stem)
    try:
        d = pickle.load(open(f, "rb"))
    except Exception:
        SKIP["дамп не читается"] += 1
        continue
    done += 1
    slots = {s["name"]: s for s in d["slots"]}
    tl = collections.defaultdict(list)
    for ln in d["lines"]:
        tl[ln["track"]].append(ln)
    for nm, gt in d["gts"].items():
        s = slots.get(nm)
        if s is None:
            continue
        # ЛУЧШАЯ честная трасса слота среди линий его трека
        best = None
        for ln in tl.get(s["track"], []):
            m, c = err(ln["tr"], gt)
            if HON(m, c) and (best is None or m < best[0]):
                best = (m, ln)
        if best is None:
            continue
        w = d["written"].get(nm)
        if w is not None and HON(*err(w, gt)):
            # верно записана: копим признаки ВЕРНО помеченных линий того же класса
            _, ln = best
            feat["верно"].append(shape(ln["tr"]))
            continue
        _, ln = best
        if s["class"] not in (cls_of(ln), "OTHER", "CALI"):
            direction[f'слот ждёт {s["class"]} — линия {cls_of(ln)}'] += 1
            by_mnem[nm.split()[0]] += 1
            feat["ошибочно"].append(shape(ln["tr"]))

W = 86
print(f"\n{'='*W}\nКОРЗИНА «КЛАСС»: листов обработано {done}, кривых в корзине {sum(direction.values())}")
_sk = sum(SKIP.values())
print(f"  СВЕРКА СПИСКА: обработано {done} + пропущено {_sk} = {done + _sk} против длины списка "
      f"{len(FILES)}   {'★ СОШЛОСЬ' if done + _sk == len(FILES) else '⛔ НЕ СОШЛОСЬ'}")
for k, v in SKIP.most_common():
    print(f"    пропущено «{k}»: {v}")

print(f"\n★ НАПРАВЛЕНИЕ ОШИБКИ")
for k, v in direction.most_common():
    print(f"    {k:<40}{v:>5}")

print(f"\n★ ПО МНЕМОНИКЕ СЛОТА (топ-12)")
for k, v in by_mnem.most_common(12):
    print(f"    {k:<14}{v:>5}")

print(f"\n★ ПРИЗНАКИ ФОРМЫ: ошибочно помеченные против верно помеченных")
print(f"    {'величина':<16}{'ошибочно':>12}{'верно':>12}")
for i, name in enumerate(("ВЧ-дрожь", "размах", "доля разворотов")):
    for tag in ("ошибочно", "верно"):
        pass
    bad = [q[i] for q in feat["ошибочно"]]; good = [q[i] for q in feat["верно"]]
    if bad and good:
        print(f"    {name:<16}{np.median(bad):>12.3f}{np.median(good):>12.3f}")
print(f"    {'кривых':<16}{len(feat['ошибочно']):>12}{len(feat['верно']):>12}")
print("\n⚠ Читать так: если медианы РАСХОДЯТСЯ — метку `behavior` можно чинить порогом; если\n"
      "  совпадают — признак этих линий не различает, и порогом делу не помочь.")
