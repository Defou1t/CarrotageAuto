r"""_slot_ceiling.py — ПОТОЛОК ШАГА НАЗНАЧЕНИЯ: сколько честных дало бы ИДЕАЛЬНОЕ 1:1 по тем же пулам.

⚠⚠ ЗАЧЕМ. Ветка меряет отбор через «сколько кривых, у которых честная трасса в пуле есть, записаны
не той линией». Но это не потолок: назначение 1-к-1, и две кривые могут претендовать на ОДНУ линию —
тогда честной станет только одна, и никакой ранжировщик этого не изменит. Настоящий потолок шага —
максимальное паросочетание в двудольном графе «слот ↔ линия, которая для него честна».

★ ЗАЧЕМ ТОЧНОЕ, А НЕ ЖАДНОЕ. `_pool_oracle.py` считает «ОПТИМУМ 1:1» жадно (по возрастанию ошибки
среди пар). Жадность на паросочетании занижает: классический пример — линия, нужная двум слотам,
достаётся тому, кто подошёл раньше, хотя у второго альтернатив нет. Здесь считается ТОЧНО (алгоритм
Куна, чередующиеся пути), и рядом печатается жадная оценка — чтобы было видно, велика ли разница.
⚠ numpy-only: в `auto/` политика пакета запрещает scipy (`auto/__init__.py`), а стенду ни к чему
зависимость, которой не будет у прода.

ЧТО ЭТО ДАЁТ ПЛАНУ. Разрыв «прод → потолок назначения» — весь остаток, который в принципе может взять
работа над раскладкой (ранжировщик, гейт, алгоритм назначения) при НЕИЗМЕННЫХ пулах. Всё, что выше
потолка, — уже не отбор, а ведение.

  python _slot_ceiling.py
"""
import sys, argparse, pickle, collections
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
ap.add_argument("--gate", default="frac0.2")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

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


def max_matching(adj, nline):
    """Максимальное паросочетание (Кун). adj[i] — линии, честные для слота i. Возврат: размер."""
    match = [-1] * nline
    def try_kuhn(v, used):
        for to in adj[v]:
            if used[to]:
                continue
            used[to] = True
            if match[to] == -1 or try_kuhn(match[to], used):
                match[to] = v
                return True
        return False
    res = 0
    for v in range(len(adj)):
        if adj[v] and try_kuhn(v, [False] * nline):
            res += 1
    return res


w = SM.load(a.model)
kind, thr = SM._parse_gate(a.gate)
print(f"вес {a.model}, гейт {kind}<{thr}")

T = collections.Counter()
nfile = nsheet = ndup = 0
seen = set()
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        nfile += 1
        if f.stem in seen:
            ndup += 1; continue
        seen.add(f.stem)
        if a.limit and nsheet >= a.limit:
            continue
        d = pickle.load(open(f, "rb"))
        nsheet += 1
        gts = d["gts"]
        by_track, tl = {}, {}
        for i, ln in enumerate(d["lines"]):
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
            tl.setdefault(ln["track"], []).append(i)
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in d["slots"]]
        strk = {s["name"]: s["track"] for s in d["slots"]}
        # ── что записано сейчас: прод-конфигурация ────────────────────────────────────────────
        got = SM.assign(slots, by_track, w, kind, thr) if slots else None
        for nm, gt in gts.items():
            # ⚠ откат к правилу ПОЛИСТНЫЙ (`emit.py:274-279`): принятый лист пишет только то, что
            # назначила модель; слот без назначения остаётся пустым. См. §6.115.
            wr = d["written"].get(nm) if got is None else (got[nm][1] if nm in got else None)
            T["прод"] += bool(wr is not None and HON(*err(wr, gt)))
            T["правило"] += bool(nm in d["written"] and HON(*err(d["written"][nm], gt)))
        # ── граф «слот ↔ честная для него линия» ──────────────────────────────────────────────
        names = [nm for nm in gts if strk.get(nm) is not None]
        adj = []
        for nm in names:
            adj.append([i for i in tl.get(strk[nm], [])
                        if HON(*err(d["lines"][i]["tr"], gts[nm]))])
        T["кривых"] += len(gts)
        T["с честной линией"] += sum(1 for v in adj if v)
        T["★ потолок 1:1"] += max_matching(adj, len(d["lines"]))
        # жадная оценка того же (как в `_pool_oracle`): по возрастанию ошибки
        pairs = []
        for k, nm in enumerate(names):
            for i in adj[k]:
                pairs.append((err(d["lines"][i]["tr"], gts[nm])[0], k, i))
        pairs.sort()
        us, ul, gr = set(), set(), 0
        for m, k, i in pairs:
            if k in us or i in ul:
                continue
            us.add(k); ul.add(i); gr += 1
        T["жадная оценка"] += gr

W = 92
print(f"{'='*W}")
print(f"ВЫБОРКА: файлов {nfile}, прочитано {nsheet}, дублей {ndup}"
      + (f"  ⚠ ОГРАНИЧЕНО --limit {a.limit}" if a.limit else ""))
print(f"  СВЕРКА: {nsheet} + {ndup} = {nsheet+ndup} против {nfile}   "
      + ("★ СОШЛОСЬ" if (nsheet + ndup == nfile) or a.limit else "⛔ РАСХОЖДЕНИЕ"))
print(f"{'='*W}")
tot = T["кривых"]
for k in ("правило", "прод", "★ потолок 1:1", "жадная оценка", "с честной линией"):
    print(f"  {k:<20}{T[k]:>8}{100*T[k]/max(1,tot):>7.1f}%")
print(f"\n⇒ ОСТАТОК ШАГА НАЗНАЧЕНИЯ: {T['★ потолок 1:1'] - T['прод']} кривых "
      f"({100*(T['★ потолок 1:1']-T['прод'])/max(1,tot):.1f}% выборки) — это ВСЁ, что может взять "
      f"работа над раскладкой при нынешних пулах")
d1 = T["с честной линией"] - T["★ потолок 1:1"]
print(f"⚠ КОНКУРЕНЦИЯ ЗА ЛИНИЮ: {d1} кривых имеют честную линию, но она нужна ДРУГОМУ слоту — "
      f"их не возьмёт никакой ранжировщик (1:1 по построению)")
print(f"⚠ жадная оценка против точной: {T['жадная оценка']} против {T['★ потолок 1:1']} "
      + ("★ совпали" if T["жадная оценка"] == T["★ потолок 1:1"] else
         f"⇒ жадность занижает потолок на {T['★ потолок 1:1']-T['жадная оценка']}"))
