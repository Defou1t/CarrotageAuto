r"""_slot_npts_probe.py — ВЫНИМАЕТСЯ ЛИ СИГНАЛ `npts`, КОТОРЫЙ РАНЖИРОВЩИК НЕ БЕРЁТ.

`_slot_feat_gap.py` (§6.115.2) на 396 промахах модели нашёл ровно один заметный разделитель из 14:
**`npts`** — у ЧЕСТНОГО кандидата точек больше в 68% случаев (медиана 1.445 против 1.249), при этом
модель выбирает более короткий. Остальные признаки дают 0.41-0.63, то есть около случайного.

ВОПРОС: это сигнал, которым можно пользоваться, или он уже учтён скором и добавка только навредит?
Проверяется самым дешёвым способом: к скору добавляется `bonus × npts`, и тем же жадным 1:1 (прод) с
тем же гейтом считается число честных. Свип по `bonus`, включая 0 — контроль, обязан совпасть с продом.

⚠ Это НЕ предложение править прод: добавка к скору — грубая замена переобучению. Замер отвечает на
один вопрос: есть ли в `npts` польза СВЕРХ той, что модель уже извлекла. Если да — признак стоит
усилить при следующем обучении; если нет — направление закрыто, и остаток раскладки требует ДРУГИХ
признаков, а не перевзвешивания нынешних.
⚠ Гейт считается по ИЗМЕНЁННЫМ скорам (так вёл бы себя настоящий механизм), поэтому при bonus>0
меняется и доля отказов — это часть эффекта, а не помеха.

  python _slot_npts_probe.py [--bonus 0 0.1 0.25 0.5 1.0] [--limit N]
"""
import sys, argparse, pickle, json, collections
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
ap.add_argument("--bonus", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
ap.add_argument("--limit", type=int, default=0)
# ★ Отбор по АУДИТОРСКИМ скважинам манифеста: они отложены от обучения этого веса, поэтому эффект,
# найденный на них, — пробел ОБОБЩЕНИЯ, а не подгонка под обучающие данные (урок §6.114).
ap.add_argument("--audit", action="store_true", help="считать только по аудиторским скважинам веса")
# ★ §6.120: полистные счётчики нужны, чтобы у прироста был не только знак, но и ШУМ НАБОРА
# (урок §6.118 — «+13» три недели читалось как довод, пока не посчитали разброс). Дамп
# необязателен и на счёт не влияет: {лист: {bonus: честных}}.
ap.add_argument("--per-sheet", default="", help="куда сложить полистные счётчики (json)")
ap.add_argument("--wells", choices=("all", "trained", "audit", "unseen"), default="all",
                help="unseen — скважины, которых вес НЕ ВИДЕЛ и которые не аудиторские")
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


def assign_with(sc, pairs, kind, thr):
    """Повторяет `slot_model.assign` ОТ МЕСТА, ГДЕ УЖЕ ЕСТЬ СКОР: margin пары, жадное 1:1, отказ
    листом. Паритет с продом проверяется прогоном при bonus=0."""
    per_slot = {}
    for k, (nm, _, _) in enumerate(pairs):
        per_slot.setdefault(nm, []).append(k)
    marg = np.zeros(len(sc)); ncand = np.zeros(len(sc), int)
    for nm, ks in per_slot.items():
        v = sorted((sc[q] for q in ks), reverse=True)
        for q in ks:
            ncand[q] = len(ks)
            marg[q] = np.inf if len(v) == 1 else sc[q] - (v[1] if sc[q] >= v[0] else v[0])
    got, used = {}, set()
    for k in sorted(range(len(sc)), key=lambda q: -sc[q]):
        nm, L, tr = pairs[k]
        if nm in got or id(L) in used:
            continue
        got[nm] = (tr, k); used.add(id(L))
    if not got:
        return None
    ks = [v[1] for v in got.values()]
    if kind == "frac":
        ok = float(np.mean([marg[k] >= 0.3 for k in ks])) >= thr
    elif kind == "mean":
        ok = float(np.mean(np.clip([marg[k] for k in ks], -5.0, 5.0))) >= thr
    elif kind == "off":
        ok = False
    else:
        ok = min((marg[k] if ncand[k] > 1 else sc[k] + 2.0) for k in ks) >= thr
    return got if ok else None


w = SM.load(a.model)
kind, thr = SM._parse_gate(a.gate)
man = json.loads((Path(r"F:\nds\Auto\auto\models") / a.model.replace(".npz", ".json"))
                 .read_text(encoding="utf-8"))
NPTS = man["features"].index("npts")
AUDIT = set(man["audit_wells"])
TRAINED = set(man["trained_wells"])
SEL = "audit" if a.audit else a.wells
WELL = {}
if SEL != "all":
    for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            WELL[q.name] = wlg.parent.name
print(f"вес {a.model}, гейт {kind}<{thr}, столбец npts = {NPTS}"
      + (f"; ★ ГРУППА СКВАЖИН: {SEL}" if SEL != "all" else ""))


def take(name):
    if SEL == "all":
        return True
    w = WELL.get(name)
    if SEL == "audit":
        return w in AUDIT
    if SEL == "trained":
        return w in TRAINED
    return w not in AUDIT and w not in TRAINED      # unseen

T = collections.Counter()
PS = {}
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
        if not take(d["name"]):
            continue
        nsheet += 1
        gts = d["gts"]
        by_track = {}
        for ln in d["lines"]:
            by_track.setdefault(ln["track"], []).append(
                (FakeLine(ln["color"], ln["behavior"], ln["x_center"]), ln["tr"]))
        slots = [dict(name=s["name"], track=s["track"], color=s["color"], cls=s["class"])
                 for s in d["slots"]]
        rule_h = sum(1 for nm, gt in gts.items()
                     if nm in d["written"] and HON(*err(d["written"][nm], gt)))
        T["правило"] += rule_h
        ps = PS.setdefault(d["name"], {}) if a.per_sheet else None
        if not slots:
            for b in a.bonus:
                T[b] += rule_h
                if ps is not None:
                    ps[str(b)] = rule_h
            continue
        X, pairs = SM.rows(slots, by_track)
        if not len(X):
            for b in a.bonus:
                T[b] += rule_h
                if ps is not None:
                    ps[str(b)] = rule_h
            continue
        base = SM.predict(w, X)
        prod = SM.assign(slots, by_track, w, kind, thr)
        for b in a.bonus:
            got = assign_with(base + b * X[:, NPTS], pairs, kind, thr)
            if b == 0.0 and prod is not None and got is not None:
                if {nm: id(v[1]) for nm, v in prod.items()} != {nm: id(v[0]) for nm, v in got.items()}:
                    mismatch += 1
            if got is None:
                T[b] += rule_h                      # отказ ⇒ пишет правило, как в проде
                T[("отказ", b)] += 1
                if ps is not None:
                    ps[str(b)] = rule_h
                continue
            hb = sum(1 for nm, (tr, _) in got.items()
                     if nm in gts and HON(*err(tr, gts[nm])))
            T[b] += hb
            if ps is not None:
                ps[str(b)] = hb

W = 92
print(f"{'='*W}")
print(f"ВЫБОРКА: файлов {nfile}, прочитано {nsheet}, дублей {ndup}"
      + (f"  ⚠ ОГРАНИЧЕНО --limit {a.limit}" if a.limit else ""))
print(f"★ ПАРИТЕТ ПРИ bonus=0: расхождений с продом {mismatch} из {nsheet}   "
      + ("★ СОШЛОСЬ" if mismatch == 0 else "⛔ СТЕНД СЧИТАЕТ НЕ ТО"))
print(f"{'='*W}")
print(f"  {'правило':<10}{T['правило']:>9}")
b0 = T[a.bonus[0]] if a.bonus else 0
for b in a.bonus:
    mark = "  ← прод" if b == 0.0 else f"{T[b]-T[0.0]:+d}"
    print(f"  bonus {b:<4}{T[b]:>9}   отказано листов {T[('отказ', b)]:>5}   {mark}")
best = max(a.bonus, key=lambda b: T[b])
print(f"\n⇒ лучший bonus {best} ({T[best]}) против прода {T[0.0]}: {T[best]-T[0.0]:+d} кривых")
print("⚠ Пулы, не отгрузка. Положительный результат означает лишь, что признак недоиспользован —"
      " лечится переобучением, а не добавкой в прод.")
if a.per_sheet:
    Path(a.per_sheet).write_text(json.dumps(PS), encoding="utf-8")
    print(f"★ полистные счётчики ({len(PS)} листов) → {a.per_sheet}")
