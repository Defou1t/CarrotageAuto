r"""_slot_match.py — ГЛОБАЛЬНОЕ НАЗНАЧЕНИЕ ПРОТИВ ЖАДНОГО: берётся ли верхняя граница §6.115.

§6.115 нашёл второй адрес остатка: у **172 из 465** промахов модели честный кандидат был ПЕРВЫМ по
скору в своём слоте — линию раньше забрал другой слот, потому что `assign` раздаёт линии ЖАДНО по
убыванию скора («тот же контракт, что у правила»). Жадность на паросочетании не оптимальна, и 172 —
это верхняя граница того, что даст замена. Но она не обязана браться целиком: отдав линию одному
слоту, отнимаешь её у другого. Здесь считается, сколько берётся НА САМОМ ДЕЛЕ.

ЧТО СРАВНИВАЕТСЯ, при одном и том же скоре и одном и том же гейте:
  • ЖАДНОЕ    — ровно прод (`slot_model.assign`);
  • ГЛОБАЛЬНОЕ — паросочетание МАКСИМАЛЬНОГО СУММАРНОГО СКОРА (венгерский алгоритм, numpy-only:
    в `auto/` scipy запрещён политикой пакета, и стенду ни к чему зависимость, которой не будет
    у прода);
  • ПОТОЛОК   — паросочетание, максимизирующее число ЧЕСТНЫХ (оракул, §6.115: 1975).
★ КОНТРОЛЬ ПАРИТЕТА: жадное назначение стенда сверяется с `SM.assign` пара-в-пару; расхождение
печатается счётчиком. Без него сравнение «жадное против глобального» ничего не значит — вдруг стенд
воспроизводит не то жадное.
⚠ Гейт применяется к ОБОИМ одинаково: мера `frac` считает долю назначенных пар с margin ≥ 0.3, а
margin — свойство ПАРЫ (скор минус лучший из остальных кандидатов слота), от способа назначения не
зависит. Значит гейт не даёт ни одной стороне форы.

  python _slot_match.py [--gate frac0.2] [--limit N]
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
BIG = 1e6


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


def hungarian(cost):
    """Минимальное по стоимости паросочетание, O(n²m) (e-maxx). cost — (n строк ≤ m столбцов).
    Возврат: массив длины n, для каждой строки столбец или −1."""
    n, m = cost.shape
    u = np.zeros(n + 1); v = np.zeros(m + 1)
    p = np.zeros(m + 1, int); way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(m + 1, np.inf); used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur; way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]; j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    out = np.full(n, -1, int)
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


w = SM.load(a.model)
kind, thr = SM._parse_gate(a.gate)
print(f"вес {a.model}, гейт {kind}<{thr}")

T = collections.Counter()
mismatch = 0
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
        names = sorted({nm for nm, _, _ in pairs})
        lines = list({id(L): L for _, L, _ in pairs})           # порядок стабилен
        lidx = {k: i for i, k in enumerate({id(L) for _, L, _ in pairs})}
        ni, nj = len(names), len(lidx)
        nmi = {nm: i for i, nm in enumerate(names)}
        S = np.full((ni, nj), -BIG)
        TR = {}
        for k, (nm, L, tr) in enumerate(pairs):
            i, j = nmi[nm], lidx[id(L)]
            if sc[k] > S[i, j]:
                S[i, j] = sc[k]; TR[(i, j)] = (tr, k)
        # ── ЖАДНОЕ: ровно прод ────────────────────────────────────────────────────────────────
        got = SM.assign(slots, by_track, w, kind, thr)
        greedy = {}
        used_l, used_s = set(), set()
        for k in sorted(range(len(sc)), key=lambda q: -sc[q]):
            nm, L, tr = pairs[k]
            if nm in used_s or id(L) in used_l:
                continue
            used_s.add(nm); used_l.add(id(L)); greedy[nm] = tr
        if got is not None:
            if {nm: id(v[1]) for nm, v in got.items()} != {nm: id(t) for nm, t in greedy.items()}:
                mismatch += 1
        # ── ГЛОБАЛЬНОЕ: максимум суммарного скора ─────────────────────────────────────────────
        if ni <= nj:
            asg = hungarian(-S)
            glob = {names[i]: TR[(i, j)][0] for i, j in enumerate(asg)
                    if j >= 0 and (i, j) in TR and S[i, j] > -BIG / 2}
        else:
            asg = hungarian(-S.T)
            glob = {names[i]: TR[(i, j)][0] for j, i in enumerate(asg)
                    if i >= 0 and (i, j) in TR and S[i, j] > -BIG / 2}
        # ── счёт честных: гейт применяется к обоим одинаково (по факту отказа прода) ──────────
        rule_h = sum(1 for nm, gt in gts.items()
                     if nm in d["written"] and HON(*err(d["written"][nm], gt)))
        for tag, mp in (("жадное", greedy), ("глобальное", glob)):
            if got is None:                       # лист отказан ⇒ пишет правило, оба одинаково
                T[tag] += rule_h
                continue
            T[tag] += sum(1 for nm, tr in mp.items()
                          if nm in gts and HON(*err(tr, gts[nm])))
        T["правило"] += rule_h

W = 92
print(f"{'='*W}")
print(f"ВЫБОРКА: файлов {nfile}, прочитано {nsheet}, дублей {ndup}"
      + (f"  ⚠ ОГРАНИЧЕНО --limit {a.limit}" if a.limit else ""))
print(f"★ ПАРИТЕТ ЖАДНОГО С ПРОДОМ: расхождений {mismatch} из {nsheet}   "
      + ("★ СОШЛОСЬ" if mismatch == 0 else "⛔ СТЕНД ВОСПРОИЗВОДИТ НЕ ТО ЖАДНОЕ"))
print(f"{'='*W}")
for k in ("правило", "жадное", "глобальное"):
    print(f"  {k:<14}{T[k]:>8}")
print(f"\n⇒ глобальное против жадного: {T['глобальное']-T['жадное']:+d} кривых")
print("⚠ Пулы, не отгрузка. Верхняя граница §6.115 (честный кандидат был первым по скору, но линию "
      "забрал сосед) — сколько из неё берётся, показывает строка выше.")
