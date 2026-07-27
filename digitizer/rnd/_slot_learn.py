r"""_slot_learn.py — МОЖЕТ ЛИ ОБУЧЕНИЕ ЗАКРЫТЬ РАЗРЫВ РАСКЛАДКИ. Только замер, прод не трогается.

§6.84: перебор ~20 рукописных правил исчерпан, ни одно не проходит гейт; прод берёт 36 честных
кривых из 417, оракул 106. Вывод «это задача для обучения» опирался на ИСЧЕРПАНИЕ, а не на
доказательство, что обучение справится. Здесь оно проверяется дёшево и обратимо: ничего не
вносится, считается только достижимое обучаемым ранжировщиком.

ПОСТАНОВКА. Пара (слот, трасса) → признаки; метка = «эта трасса честна для этого слота» (по
эталону). Назначение — жадно по убыванию скора, взаимно-однозначно внутри листа: ровно тот же
контракт, что у прода.

⚠⚠ ПРОТОКОЛ ОБЪЯВЛЕН ДО ПРОГОНА (иначе подгонка, §6.49): обучение на ОДНОМ наборе, замер на ДВУХ
ОСТАЛЬНЫХ, и так все три раза. Цитируется ТОЛЬКО невиданный набор.
⚠ ПРИЗНАКИ — РОВНО ТЕ, ЧТО ЕСТЬ У ПРОДА В МОМЕНТ РАСКЛАДКИ. Никакой ошибки против эталона: иначе
получится оракул, переодетый в модель.

  <ComfyUI>\python_embeded\python.exe _slot_learn.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+",
                default=["F:/nds/output/taskS/pools:валидация",
                         "F:/nds/output/taskS/pools_gate:гейт",
                         "F:/nds/output/taskS/pools_wide:третий"])
a = ap.parse_args()

HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None, 0.0
    d = np.array([abs(tr[y] - gt[y]) for y in com], float)
    return float(np.median(d)), len(com) / max(1, len(gt))


def tfeat(tr):
    ys = sorted(tr)
    x = np.array([tr[y] for y in ys], float)
    if len(x) < 5:
        return (0., 0., 1., 0., float(len(x)), 0., 0.)
    dx = np.diff(x)
    span = float(np.percentile(x, 90) - np.percentile(x, 10)) or 1.0
    w = min(101, len(x) // 2 * 2 + 1)
    sm = np.convolve(x, np.ones(w) / w, mode="same") if w >= 3 else x
    return (float(np.median(np.abs(dx))),
            float(np.mean((dx[:-1] * dx[1:]) < 0)) if len(dx) > 2 else 0.,
            span, float(np.std(x - sm) / span), float(len(x)),
            float(np.median(x)), float(ys[-1] - ys[0]))


def build(pool_dir):
    """X, y и индекс (лист, слот, линия) — чтобы потом собрать назначение полистно."""
    X, Y, IDX, sheets = [], [], [], []
    for f in sorted(Path(pool_dir).glob("*.pkl")):
        d = pickle.load(open(f, "rb"))
        si = len(sheets); sheets.append(d)
        slots = {s["name"]: s for s in d["slots"]}
        tl = {}
        for i, ln in enumerate(d["lines"]):
            tl.setdefault(ln["track"], []).append(i)
        for nm, gt in d["gts"].items():
            s = slots.get(nm)
            if s is None:
                continue
            same = tl.get(s["track"], [])
            if not same:
                continue
            meds = sorted(float(np.median(list(d["lines"][i]["tr"].values()))) for i in same)
            for i in same:
                ln = d["lines"][i]
                wig, rev, span, rough, n, med, dy = tfeat(ln["tr"])
                cls = "SP" if ln["behavior"] == "smooth" else "RES"
                rank = meds.index(min(meds, key=lambda v: abs(v - med))) / max(1, len(meds) - 1)
                X.append([1.0 if (s["color"] and s["color"] == ln["color"]) else 0.0,
                          1.0 if s["color"] is None else 0.0,
                          1.0 if s["class"] in (cls, "OTHER", "CALI") else 0.0,
                          1.0 if s["class"] == "SP" else 0.0,
                          rank, ln["x_center"] / 1000.0, med / 1000.0,
                          wig, rev, rough, span / 1000.0, n / 10000.0,
                          len(same) / 10.0, dy / 10000.0])
                m, c = err(ln["tr"], gt)
                Y.append(1 if HON(m, c) else 0)
                IDX.append((si, nm, i))
    return np.array(X, float), np.array(Y, int), IDX, sheets


def count_by_score(scores, IDX, sheets):
    """Жадно 1:1 внутри листа по убыванию скора, затем счёт честных."""
    per = {}
    for k, (si, nm, i) in enumerate(IDX):
        per.setdefault(si, []).append((scores[k], nm, i))
    tot = 0
    for si, cand in per.items():
        d = sheets[si]
        cand.sort(key=lambda q: -q[0])
        used, got = set(), {}
        for sc, nm, i in cand:
            if nm in got or i in used:
                continue
            got[nm] = i; used.add(i)
        for nm, i in got.items():
            if HON(*err(d["lines"][i]["tr"], d["gts"][nm])):
                tot += 1
    return tot


SETS = [(s.rsplit(":", 1)[0], s.rsplit(":", 1)[1]) for s in a.sets]
D = {}
for p, lbl in SETS:
    D[lbl] = build(p)
    X, Y, IDX, sh = D[lbl]
    orc = count_by_score(np.array([1.0 if y else 0.0 for y in Y]), IDX, sh)
    print(f"{lbl:<11} листов {len(sh):>3}  пар (слот×трасса) {len(Y):>5}  честных пар {int(Y.sum()):>4}"
          f"  оракул {orc:>3}")

PROD = {"валидация": 5, "гейт": 9, "третий": 22}      # §6.79-§6.84, тот же контракт 1:1
print(f"\n{'='*74}\nПРОТОКОЛ: обучен на ОДНОМ наборе, замер на ДВУХ ОСТАЛЬНЫХ (невиданных)\n")
print(f"{'обучен на':<12}{'замер на':<12}{'обученный':>11}{'прод':>7}{'оракул':>8}{'':>4}")
for ptr, ltr in SETS:
    Xtr, Ytr, _, _ = D[ltr]
    if Ytr.sum() < 5:
        print(f"{ltr:<12} мало положительных ({int(Ytr.sum())}) — пропуск"); continue
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
    clf.fit(Xtr, Ytr)
    for pte, lte in SETS:
        if lte == ltr:
            continue
        Xte, Yte, IDXte, shte = D[lte]
        got = count_by_score(clf.predict_proba(Xte)[:, 1], IDXte, shte)
        orc = count_by_score(np.array([1.0 if y else 0.0 for y in Yte]), IDXte, shte)
        mark = "  ★" if got > PROD[lte] else ("  ✗" if got < PROD[lte] else "  =")
        print(f"{ltr:<12}{lte:<12}{got:>11}{PROD[lte]:>7}{orc:>8}{mark}")
