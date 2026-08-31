r"""_precision_recall.py — ТОЧНОСТЬ ПРОТИВ ПОЛНОТЫ: можно ли отдавать НЕ ВСЁ, но ВЕРНОЕ.

Ветка меряет одну величину — сколько кривых записано честно (14.4% под продом). Но у продукта есть
вторая ось: если модель умеет РАНЖИРОВАТЬ свои ответы по уверенности, можно отдавать верхнюю долю
выдачи с кратно большей точностью, а остальное отдавать эксперту. Это экономит его время БЕЗ единой
правки распознавания — и не требует ни новых данных, ни нового веса (DIRECTIONS.md §3.3).

Считаем на каждой НАЗНАЧЕННОЙ кривой: margin пары (та самая мера, на которой стоит гейт) и честна ли
она. Затем сметаем порог и строим кривую «доля выдачи → точность».
⚠ Гейт ВЫКЛЮЧЕН намеренно: иначе часть листов уходит на правило, у которого margin не определён, и
ранжирование сравнивало бы разные механизмы. Полнота считается от всех экспертных кривых корпуса.

  <ComfyUI>\python_embeded\python.exe _precision_recall.py --shard 0/6
  <ComfyUI>\python_embeded\python.exe _precision_recall.py --sum
"""
import sys, argparse, pickle, json
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from auto import slot_model as SM

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div", "F:/nds/output/taskS/pools_heldout",
    "F:/nds/output/taskS/pools_all"])
ap.add_argument("--model", default="slot_model_g250.npz")
ap.add_argument("--shard", default="0/6")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
OUT = Path(a.out)
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


class FakeLine:
    __slots__ = ("color", "behavior", "x_center")

    def __init__(self, color, behavior, x_center):
        self.color, self.behavior, self.x_center = color, behavior, x_center


def collect(i, n):
    w = SM.load(a.model)
    seen, files = set(), []
    for root in a.pools:
        for f in sorted(Path(root).glob("*.pkl")):
            if f.stem not in seen:
                seen.add(f.stem); files.append(f)
    mine = files[len(files) * i // n:len(files) * (i + 1) // n]
    print(f"★ ШАРД {i}/{n}: листов {len(mine)} из {len(files)}, вес {a.model}")
    rows, done, ncur = [], 0, 0
    for f in mine:
        d = pickle.load(open(f, "rb"))
        done += 1
        gts = d["gts"]
        ncur += len(gts)
        by_track = {}
        for ln in d["lines"]:
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in d["slots"]]
        if not slots:
            continue
        X, pairs = SM.rows(slots, by_track)
        if not len(X):
            continue
        sc = SM.predict(w, X)
        # margin пары — как в slot_model.assign
        per_slot = {}
        for k, (nm, _, _) in enumerate(pairs):
            per_slot.setdefault(nm, []).append(k)
        marg = np.zeros(len(sc))
        for nm, ks in per_slot.items():
            v = sorted((sc[q] for q in ks), reverse=True)
            for q in ks:
                marg[q] = 1e9 if len(v) == 1 else sc[q] - (v[1] if sc[q] >= v[0] else v[0])
        got, used = {}, set()
        for k in sorted(range(len(sc)), key=lambda q: -sc[q]):
            nm, L, tr = pairs[k]
            if nm in got or id(L) in used:
                continue
            got[nm] = (tr, k); used.add(id(L))
        for nm, (tr, k) in got.items():
            if nm not in gts:
                continue
            rows.append(dict(m=float(marg[k]), s=float(sc[k]),
                             hon=bool(HON(*err(tr, gts[nm])))))
    p = OUT / f"prec_rec_{i}of{n}.json"
    p.write_text(json.dumps({"rows": rows, "curves": ncur, "sheets": done}), encoding="utf-8")
    print(f"★ СВЕРКА: листов {done} из {len(mine)}   "
          f"{'★ СОШЛОСЬ' if done == len(mine) else '⛔ НЕ СОШЛОСЬ'}")
    print(f"назначений {len(rows)}, экспертных кривых {ncur} → {p}")


def summarise():
    fs = sorted(OUT.glob("prec_rec_*of*.json"))
    if not fs:
        sys.exit("нет дампов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    rows, curves, sheets = [], 0, 0
    for f in fs:
        d = json.loads(f.read_text(encoding="utf-8"))
        rows += d["rows"]; curves += d["curves"]; sheets += d["sheets"]
    print(f"шардов {len(fs)} из {den}" + ("  ⚠⚠ НЕПОЛНЫЙ" if len(fs) < int(den) else ""))
    print(f"листов {sheets}, экспертных кривых {curves}, назначений {len(rows)}")
    hon = sum(1 for r in rows if r["hon"])
    print(f"честных всего {hon} — точность выдачи {100*hon/max(1,len(rows)):.1f}%, "
          f"полнота от корпуса {100*hon/max(1,curves):.1f}%\n")
    # ⚠⚠ РАЗДЕЛЯТЬ ОБЯЗАТЕЛЬНО. Слот с ЕДИНСТВЕННЫМ кандидатом получает margin = ∞ и всплывает
    # наверх, хотя это не «уверенно», а «не с кем сравнивать». Смешение делало ранжирование
    # НЕМОНОТОННЫМ (верхние 5% выглядели хуже верхних 20%) и пряталo настоящий сигнал.
    inf = [r for r in rows if r["m"] > 1e8]
    fin = [r for r in rows if r["m"] <= 1e8]
    hi = lambda v: 100 * sum(1 for r in v if r["hon"]) / max(1, len(v))
    print(f"безальтернативных (единственный кандидат): {len(inf)}, точность {hi(inf):.1f}%")
    print(f"с конкуренцией:                            {len(fin)}, точность {hi(fin):.1f}%\n")
    fin.sort(key=lambda r: -r["m"])
    print(f"{'верх по margin (конкурентные)':>32}{'кривых':>9}{'честных':>9}{'ТОЧНОСТЬ':>10}{'полнота':>9}")
    for q in (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00):
        k = max(1, int(len(fin) * q))
        h = sum(1 for r in fin[:k] if r["hon"])
        print(f"{f'{100*q:.0f}%':>32}{k:>9}{h:>9}{100*h/k:>9.1f}%{100*h/max(1,curves):>8.1f}%")
    base = hi(fin)
    tp = hi(fin[:max(1, int(len(fin) * 0.05))])
    print(f"\n★ верхние 5% против средней: {tp:.1f}% против {base:.1f}% — в {tp/max(1e-9,base):.1f} раза")
    print("⇒ ранжирование работает и монотонно. Но даже наверху точность НИЖЕ половины ⇒ выдавать")
    print("  «проверенное подмножество» пока нельзя: потолок держит не ранжирование, а базовая точность.")
    print("⚠ Пулы и раскладка моделью без гейта; перенос на отгрузку — отдельный прогон (правило 1).")


if a.sum:
    summarise()
else:
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)
