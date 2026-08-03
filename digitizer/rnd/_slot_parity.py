r"""_slot_parity.py — СВЕРКА ПРОД-КОДА РАСКЛАДКИ СО СТЕНДОМ + ЦЕНА «СОПЕРНИЧАЮТ ВСЕ СЛОТЫ».

⚠ ЗАЧЕМ №1. `auto/slot_model.py` считает 14 признаков заново, из объектов Line, а стенд §6.88 считал
их из пулов. Разойдись они хоть в одном признаке — прод делал бы НЕ ТО, что померено, и это не
всплыло бы никогда: обе стороны «работают». Здесь прод-функция получает те же пулы и признаки
сверяются с обучающей матрицей побитово.

⚠⚠ ЗАЧЕМ №2 (важнее). В замерах §6.87-§6.89 за линии соперничали ТОЛЬКО слоты, у которых есть
экспертная кривая. В проде соперничают ВСЕ слоты рамки — обстановка теснее, и часть выигрыша может
съесться. Это единственная известная разница стенда и отгрузки, которую можно посчитать БЕЗ прогона
пайплайна, поэтому она считается здесь, до дорогого A/B (§6.90).

★ Вес `slot_model_g250.npz` обучен БЕЗ 8 аудиторских скважин, поэтому цифры печатаются двумя
группами: аудиторские (честные, вне обучения) и остальные (в обучении — оптимистичные).

  <ComfyUI>\python_embeded\python.exe _slot_parity.py [--gate frac0.8]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
from pathlib import Path
import numpy as np
from auto import slot_model as SM

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--cache", default="F:/nds/output/taskS/_slot_abstain_cache_v2.pkl")
ap.add_argument("--gate", default="frac0.8")
ap.add_argument("--model", default=SM.DEFAULT_MODEL)
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


class FakeLine:
    """Ровно то, что прод-код читает у Line. Трек передаётся ключом by_track."""
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


C = pickle.load(open(a.cache, "rb"))
WELL = dict(zip(C["names"], C["wells"]))
AUDIT = set(__import__("json").loads(
    Path(r"F:\nds\Auto\auto\models\slot_model_g250.json").read_text(encoding="utf-8"))["audit_wells"])
XREF = {(si, nm, i): C["X"][k] for k, (si, nm, i) in enumerate(C["IDX"])}
NAME2SI = {n: i for i, n in enumerate(C["names"])}

w = SM.load(a.model)
kind, thr = SM._parse_gate(a.gate)
print(f"вес {a.model}, гейт {kind}<{thr}, аудиторских скважин {len(AUDIT)}")

seen = set()
worst_feat, nfeat = 0.0, 0
G = {g: dict(prod=0, A=0, B=0, absA=0, absB=0, n=0, upA=0, dnA=0, upB=0, dnB=0, moved=0, slots=0)
     for g in ("аудит (вне обучения)", "прочие (в обучении)")}
# ⚠ КЛЮЧ ДЕДУПЛИКАЦИИ — ИМЯ ФАЙЛА, А НЕ ПОЛЕ `name`: `_pool_oracle.py` кладёт дамп как
# `{nlgx.stem[:60]}.pkl`, поэтому дубль виден ДО чтения и стоит ноль. ⚠ В отличие от
# `_start_probe`/`_param_sweep` здесь нет прохода «сначала список, потом счёт»: дамп нужен целиком,
# и 2.35 ГБ читаются по делу — экономится только повторное чтение дублей.
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        if f.stem in seen:
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        if not d["name"].startswith(f.stem[:40]):
            print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
        si = NAME2SI.get(d["name"])
        grp = "аудит (вне обучения)" if WELL.get(d["name"]) in AUDIT else "прочие (в обучении)"
        g = G[grp]
        gts = d["gts"]
        by_track, lidx = {}, {}
        for i, ln in enumerate(d["lines"]):
            L = FakeLine(ln["color"], ln["behavior"], ln["x_center"])
            lidx[id(L)] = i
            by_track.setdefault(ln["track"], []).append((L, ln["tr"]))
        slots_all = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                     for s in d["slots"]]
        slots_gt = [s for s in slots_all if s["name"] in gts]
        prod = sum(1 for nm, gt in gts.items()
                   if nm in d["written"] and HON(*err(d["written"][nm], gt)))
        g["prod"] += prod; g["n"] += 1; g["slots"] += len(slots_gt)

        # ── признаки прода против обучающей матрицы (только пары, которые стенд строил) ────────
        if si is not None and slots_gt:
            rows, pairs = SM.rows(slots_gt, by_track)
            for r, (nm, L, _) in zip(rows, pairs):
                ref = XREF.get((si, nm, lidx[id(L)]))
                if ref is None:
                    continue
                worst_feat = max(worst_feat, float(np.max(np.abs(np.array(r) - ref))))
                nfeat += 1

        hon = {}
        for tag, slots in (("A", slots_gt), ("B", slots_all)):
            got = SM.assign(slots, by_track, w, kind, thr) if slots else None
            if got is None:
                g["abs" + tag] += 1
                hon[tag] = prod                      # отказ ⇒ раскладка правилом ⇒ как у прода
                continue
            hon[tag] = sum(1 for nm, (L, tr, _) in got.items()
                           if nm in gts and HON(*err(tr, gts[nm])))
        for tag in ("A", "B"):
            g[tag] += hon[tag]
            g["up" + tag] += 1 if hon[tag] > prod else 0
            g["dn" + tag] += 1 if hon[tag] < prod else 0
        g["moved"] += 1 if hon["A"] != hon["B"] else 0

print(f"\n★ ПРИЗНАКИ: сверено {nfeat} пар, max|Δ| = {worst_feat:.3e}"
      + ("  ⇒ прод считает РОВНО то же" if worst_feat < 1e-9 else "  ⛔ РАСХОЖДЕНИЕ"))

W = 92
print(f"\n{'='*W}\nЧЕСТНЫЕ КРИВЫЕ: A = соперничают только слоты с эталоном (как в замере §6.88),\n"
      f"                B = соперничают ВСЕ слоты рамки (как в проде)\n{'='*W}")
print(f"{'группа':<24}{'листов':>7}{'прод':>6}{'A':>6}{'↑/↓ A':>9}{'B':>6}{'↑/↓ B':>9}"
      f"{'отказов A/B':>13}{'листов A≠B':>11}")
tot = dict(n=0, prod=0, A=0, B=0)
for grp, g in G.items():
    print(f"{grp:<24}{g['n']:>7}{g['prod']:>6}{g['A']:>6}"
          f"{f'{g[chr(117)+chr(112)+chr(65)]}/{g[chr(100)+chr(110)+chr(65)]}':>9}{g['B']:>6}"
          f"{f'{g[chr(117)+chr(112)+chr(66)]}/{g[chr(100)+chr(110)+chr(66)]}':>9}"
          f"{f'{g[chr(97)+chr(98)+chr(115)+chr(65)]}/{g[chr(97)+chr(98)+chr(115)+chr(66)]}':>13}"
          f"{g['moved']:>11}")
    for k in ("n", "prod", "A", "B"):
        tot[k] += g[k]
print(f"{'ИТОГО':<24}{tot['n']:>7}{tot['prod']:>6}{tot['A']:>6}{'':>9}{tot['B']:>6}")
print(f"\nΔ к проду: A {tot['A']-tot['prod']:+d}, B {tot['B']-tot['prod']:+d}"
      f"  ⇒ цена «соперничают все слоты» = {tot['B']-tot['A']:+d} кривых")
print("⚠ группа «прочие» в обучении ⇒ её цифры оптимистичны; цитировать надо аудиторскую")
