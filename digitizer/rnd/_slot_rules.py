r"""_slot_rules.py — ПЕРЕБОР ПРАВИЛ РАСКЛАДКИ трасс по слотам, ОФЛАЙН по сохранённым пулам.

§6.79: в пуле 15-17 честных кривых, в файл уходит 5. §6.80 (разбор `_slot_why`): 60% потерь даёт
ПОРЯДОК по `x_center`, 30% — класс (`behavior` SP/RES), 10% — фильтр цвета.

⚠ ПОЧЕМУ ОФЛАЙН. Каждый вариант правила требовал бы полного прогона пайплайна по всем листам
(~20 минут на вариант). Но правило — ЧИСТАЯ ФУНКЦИЯ над пулом трасс и списком слотов, поэтому
пул сохраняется один раз (`_pool_oracle.py --dump`), а варианты сравниваются за секунды. Это та же
идея, на которой стоит `_pick_gate` («пайплайн гоняется ОДИН РАЗ на лист»).

ПРАВИЛА (каждое — как выбрать, какая линия идёт в какой слот внутри трека):
  prod        — как сейчас: фильтр цвета, затем сортировка по (класс, x_center);
  no_color    — то же без фильтра цвета;
  no_class    — то же без учёта класса;
  bare        — только порядок по x_center, без цвета и класса;
  med_x       — вместо x_center линии берётся МЕДИАНА x самой трассы (x_center — центр полосы от
                U1, он не обязан совпадать с тем, где реально идёт трасса);
  hungarian   — глобальное назначение (венгерский алгоритм) по стоимости |ранг слота − ранг линии|
                вместо жадной сортировки: слоты рамки упорядочены, линии упорядочены по медиане x,
                и минимизируется суммарное расхождение рангов, а не назначается по одному.
★ ОРАКУЛ — верхняя граница: назначение по РЕАЛЬНОЙ ошибке против эталона (в проде недоступно,
нужен только как потолок).

  <ComfyUI>\python_embeded\python.exe _slot_rules.py [--pools <dir>]
"""
import sys, argparse, pickle, itertools
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pools", default=r"F:\nds\output\taskS\pools")
a = ap.parse_args()

NULL_OK = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def key_x(ln, how):
    if how == "med":
        v = list(ln["tr"].values())
        return float(np.median(v)) if v else ln["x_center"]
    return ln["x_center"]


def assign(d, use_color=True, use_class=True, xhow="center", hungarian=False):
    """Вернуть {имя слота: трасса}. Повторяет прод-правило, кроме явно отключённого."""
    out = {}
    used = set()
    tracks = {s["track"] for s in d["slots"]}
    for ti in tracks:
        tslots = [s for s in d["slots"] if s["track"] == ti]
        tlines = [(i, ln) for i, ln in enumerate(d["lines"]) if ln["track"] == ti]
        if not tslots or not tlines:
            continue
        if hungarian:
            order = sorted(tlines, key=lambda q: key_x(q[1], xhow))
            n = min(len(tslots), len(order))
            for si in range(n):                       # ранг слота ↔ ранг линии, один к одному
                i, ln = order[si]
                if i in used:
                    continue
                used.add(i); out[tslots[si]["name"]] = ln["tr"]
            continue
        colors = {ln["color"] for _, ln in tlines}
        pairs = []
        for s in tslots:
            strict = use_color and s["color"] is not None and s["color"] in colors
            for i, ln in tlines:
                if strict and s["color"] != ln["color"]:
                    continue
                cls = "SP" if ln["behavior"] == "smooth" else "RES"
                ok = (not use_class) or (s["class"] in (cls, "OTHER", "CALI"))
                pairs.append((0 if ok else 1, s, i, ln))
        pairs.sort(key=lambda q: (q[0], key_x(q[3], xhow)))
        taken = set()
        for _, s, i, ln in pairs:
            if s["name"] in taken or i in used:
                continue
            taken.add(s["name"]); used.add(i); out[s["name"]] = ln["tr"]
    return out


def oracle(d):
    """ПОТОЛОК: назначение по реальной ошибке (в проде невозможно)."""
    pairs = []
    for nm, gt in d["gts"].items():
        for i, ln in enumerate(d["lines"]):
            m, c = err(ln["tr"], gt)
            if m is not None:
                pairs.append((c < 0.9, m, nm, i))
    pairs.sort()
    out, used = {}, set()
    for _, _, nm, i in pairs:
        if nm in out or i in used:
            continue
        out[nm] = d["lines"][i]["tr"]; used.add(i)
    return out


RULES = {
    "prod (как сейчас)": dict(use_color=True, use_class=True, xhow="center"),
    "без фильтра цвета": dict(use_color=False, use_class=True, xhow="center"),
    "без класса": dict(use_color=True, use_class=False, xhow="center"),
    "только порядок": dict(use_color=False, use_class=False, xhow="center"),
    "медиана x трассы": dict(use_color=True, use_class=True, xhow="med"),
    "медиана x, без цвета": dict(use_color=False, use_class=True, xhow="med"),
    "медиана x, без цвета и класса": dict(use_color=False, use_class=False, xhow="med"),
    "венгерский по рангу (мед. x)": dict(hungarian=True, xhow="med"),
}

files = sorted(Path(a.pools).glob("*.pkl"))
data = [pickle.load(open(f, "rb")) for f in files]
ncur = sum(len(d["gts"]) for d in data)
print(f"листов {len(data)}, экспертных кривых {ncur}, трасс в пулах "
      f"{sum(len(d['lines']) for d in data)}\n")

base = None
print(f"{'правило':<32}{'честных':>9}{'против прода':>14}")
for name, kw in RULES.items():
    tot = 0
    for d in data:
        got = assign(d, **kw)
        for nm, gt in d["gts"].items():
            tr = got.get(nm)
            if tr and NULL_OK(*err(tr, gt)):
                tot += 1
    if base is None:
        base = tot
    print(f"{name:<32}{tot:>9}{tot - base:>+14d}")

orc = sum(1 for d in data for nm, gt in d["gts"].items()
          if oracle(d).get(nm) and NULL_OK(*err(oracle(d)[nm], gt)))
print(f"{'★ ОРАКУЛ (потолок)':<32}{orc:>9}{orc - base:>+14d}")
