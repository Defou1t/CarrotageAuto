r"""_slot_cause.py — ПРИЧИНЫ ПОТЕРЬ РАСКЛАДКИ по всем трём наборам, ОФЛАЙН по сохранённым пулам.

`_slot_why.py` считал то же самое, но гонял ПОЛНЫЙ ПАЙПЛАЙН и потому был прогнан лишь на одном
наборе (§6.80). Пулы сохранены (`_pool_oracle.py --dump`), поэтому разбор становится офлайновым
и идёт сразу по 106 листам.

ВОПРОС. Для каждой экспертной кривой, у которой В ПУЛЕ ЕСТЬ честная трасса, но в файл ушла не она:
что именно помешало? Правило (`auto/emit.py:124-155`) отбраковывает по трём признакам, и они
различимы:
  ЦВЕТ    — верная линия отсеяна фильтром цвета (слот ждёт red, линия чёрная);
  КЛАСС   — верная линия проиграла по совпадению класса (`behavior` SP/RES);
  ПОРЯДОК — оба признака совпали, победила другая линия по `x_center`.

★ ОТДЕЛЬНО СЧИТАЕТСЯ ЦЕНА И ПОЛЬЗА ФИЛЬТРА ЦВЕТА. §6.81 показал, что глобальное снятие фильтра
даёт ноль по сумме (+1/−1/+0) — то есть он и вредит, и помогает. Здесь считается, СКОЛЬКО раз
каждое: сколько честных трасс он отбросил и сколько раз без него в слот попала бы неверная.
Это и решает, возможен ли УСЛОВНЫЙ фильтр вместо глобального включения/выключения.

  <ComfyUI>\python_embeded\python.exe _slot_cause.py
"""
import sys, argparse, collections
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+",
                default=[r"F:/nds/output/taskS/pools", r"F:/nds/output/taskS/pools_gate",
                         r"F:/nds/output/taskS/pools_wide"])
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def cls_of(ln):
    return "SP" if ln["behavior"] == "smooth" else "RES"


reasons = collections.Counter()
color_cost = 0        # честная трасса отсеяна фильтром цвета
color_gain = 0        # без фильтра в слот попала бы НЕ честная, а с ним — честная
examples = []

# ⚠ ДЕДУПЛИКАЦИЯ ПО ИМЕНИ ФАЙЛА. Прежде её не было вовсе: попади один лист в два каталога — он
# считался бы дважды. `_pool_oracle.py` кладёт дамп как `{nlgx.stem[:60]}.pkl`, поэтому дубль виден
# ДО чтения и стоит ноль. ⚠ Здесь, в отличие от `_start_probe`/`_param_sweep`, нет прохода «сначала
# список, потом счёт»: дамп нужен целиком, и пулы читаются по делу. Совпадение ключа с полем `name`
# проверяется при чтении. (На текущих трёх каталогах дублей нет — 20+26+60 = те самые 106 листов.)
seen = set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        if not d["name"].startswith(f.stem[:40]):
            print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
        slots = {s["name"]: s for s in d["slots"]}
        for nm, gt in d["gts"].items():
            s = slots.get(nm)
            if s is None:
                continue
            # лучшая честная трасса пула для этой кривой
            best = None
            for ln in d["lines"]:
                m, c = err(ln["tr"], gt)
                if HON(m, c) and (best is None or m < best[0]):
                    best = (m, ln)
            if best is None:
                continue
            w = d["written"].get(nm)
            if w is not None and HON(*err(w, gt)):
                reasons["записана верно"] += 1
                continue
            _, ln = best
            same_track = [q for q in d["lines"] if q["track"] == ln["track"]]
            colors = {q["color"] for q in same_track}
            strict = s["color"] is not None and s["color"] in colors
            if strict and s["color"] != ln["color"]:
                reasons["ЦВЕТ"] += 1
                color_cost += 1
                examples.append(("ЦВЕТ", nm, s["color"], ln["color"], d["name"][:34]))
            elif s["class"] not in (cls_of(ln), "OTHER", "CALI"):
                reasons["КЛАСС"] += 1
                examples.append(("КЛАСС", nm, s["class"], cls_of(ln), d["name"][:34]))
            elif w is None:
                reasons["слот пуст"] += 1
            else:
                reasons["ПОРЯДОК x_center"] += 1
        # ПОЛЬЗА фильтра: слоты, где он СПАС верное назначение, оценивается косвенно —
        # сколько записанных ЧЕСТНЫХ трасс имеют цвет, совпавший с приором слота, при том что
        # на треке была линия другого цвета, которая по x_center шла бы раньше.
        for nm, gt in d["gts"].items():
            s = slots.get(nm); w = d["written"].get(nm)
            if s is None or w is None or not HON(*err(w, gt)):
                continue
            wl = next((q for q in d["lines"] if q["tr"] is w), None)
            if wl is None or s["color"] is None or s["color"] != wl["color"]:
                continue
            rivals = [q for q in d["lines"]
                      if q["track"] == wl["track"] and q["color"] != wl["color"]
                      and q["x_center"] < wl["x_center"]]
            if rivals:
                color_gain += 1

tot = sum(reasons.values())
# ⚠ ОБЪЁМ ВЫБОРКИ ПЕЧАТАЕТСЯ ИЗ СЧЁТЧИКА, А НЕ ТЕКСТОМ. Прежде здесь стояло «три набора
# (106 листов)» строкой; при вызове с пятью каталогами шапка продолжала утверждать 106, и разбор
# по 702 листам читался как разбор по 106.
print(f"\n{'='*82}\nПРИЧИНЫ ПОТЕРЬ РАСКЛАДКИ: наборов {len(a.pools)}, листов {len(seen)}")
print(f"кривых, у которых в пуле ЕСТЬ честная трасса: {tot}\n")
for k, v in reasons.most_common():
    print(f"  {k:<20} {v:>4} ({100*v/max(1,tot):>3.0f}%)")

lost = tot - reasons["записана верно"]
print(f"\n{'='*82}\nФИЛЬТР ЦВЕТА: цена против пользы")
print(f"  ОТБРОСИЛ честную трассу       {color_cost:>4} раз  ({100*color_cost/max(1,lost):.0f}% всех потерь)")
print(f"  СПАС верное назначение        {color_gain:>4} раз  (оценка сверху: были соперники левее)")
print("\n⇒ Если цена сильно больше пользы, фильтр вреден безусловно; если сопоставимы — нужен")
print("  УСЛОВНЫЙ фильтр, а глобальное включение/выключение (§6.81, сумма 0) обречено на ничью.")
if examples:
    print(f"\nпримеры потерь (слот ждёт — у линии):")
    for w, nm, want, got, sh in examples[:10]:
        print(f"  {w:<7} {nm[:16]:<17} ждёт {str(want):<7} — линия {str(got):<7} {sh}")
