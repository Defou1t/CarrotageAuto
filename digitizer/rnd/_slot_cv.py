r"""_slot_cv.py — ЧЕСТНАЯ ПЕРЕКРЁСТНАЯ ПРОВЕРКА ОБУЧЕННОЙ РАСКЛАДКИ, разбиение ПО СКВАЖИНАМ.

⚠ ЗАЧЕМ ОТДЕЛЬНЫЙ СТЕНД. `_slot_learn.py` обучался на одном ГОТОВОМ наборе и мерял на других, но
наборы (гейт/валидация/третий) нарезались по листам, а не по скважинам, и почти все их скважины
встречаются в архиве. После исключения утечки на замер оставалось 1-7 листов — мерить нечего.
Здесь разбиение делается там, где ему место: **скважина целиком идёт либо в обучение, либо в
замер**. Листы одной скважины — тот же бланк, почерк и цвета, поэтому только так проверяется
обобщение, а не запоминание бланка.

ПРОТОКОЛ (объявлен до прогона, §6.49): все собранные пулы объединяются, скважины делятся на K
фолдов, каждый фолд по очереди — замер, остальные — обучение. Считается ПОЛИСТНО (§6.83:
агрегат маскирует немонотонность) и сравнивается с продом на тех же листах.

  <ComfyUI>\python_embeded\python.exe _slot_cv.py [--folds 4]
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

ap = argparse.ArgumentParser()
ap.add_argument("--pools", nargs="+", default=[
    "F:/nds/output/taskS/pools", "F:/nds/output/taskS/pools_gate",
    "F:/nds/output/taskS/pools_wide", "F:/nds/output/taskS/pools_more",
    "F:/nds/output/taskS/pools_div"])
ap.add_argument("--folds", type=int, default=4)
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


WELL = {}
for wlg in Path(r"F:\nds\projects\Archive").glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        WELL[q.name] = wlg.parent.name
for e in (r"F:\nds\projects\Semeguniv_001\wlg", r"F:\nds\projects\Semeguniv_020\wlg"):
    if Path(e).is_dir():
        for q in Path(e).glob("*.nlgx"):
            WELL[q.name] = Path(e).parent.name

# ⚠ КЛЮЧ ДЕДУПЛИКАЦИИ — ИМЯ ФАЙЛА, А НЕ ПОЛЕ `name`: `_pool_oracle.py` кладёт дамп как
# `{nlgx.stem[:60]}.pkl`, поэтому дубль (один лист в двух каталогах) виден ДО чтения и стоит ноль.
# ⚠ В отличие от `_start_probe`/`_param_sweep` здесь нет прохода «сначала список, потом счёт»: дамп
# нужен целиком, и 2.35 ГБ читаются по делу — экономится только повторное чтение дублей.
sheets, seen = [], set()
FOUND, DUPS = 0, 0              # §6.106: объём — ИЗ СЧЁТЧИКА, и сверка в конце
for root in a.pools:
    for f in sorted(Path(root).glob("*.pkl")):
        FOUND += 1
        if f.stem in seen:
            DUPS += 1
            continue
        seen.add(f.stem)
        d = pickle.load(open(f, "rb"))
        if not d["name"].startswith(f.stem[:40]):
            print(f"  ⚠ имя дампа {f.stem[:40]!r} расходится с полем name {d['name'][:40]!r}")
        sheets.append(d)

print(f"дампов найдено {FOUND} в {len(a.pools)} каталогах, дублей {DUPS}, листов {len(sheets)}"
      f"   {'★ СОШЛОСЬ' if len(sheets) + DUPS == FOUND else '⛔ НЕ СОШЛОСЬ'}")

X, Y, R, IDX = [], [], [], []
prod = [0] * len(sheets)
for si, d in enumerate(sheets):
    slots = {s["name"]: s for s in d["slots"]}
    tl = {}
    for i, ln in enumerate(d["lines"]):
        tl.setdefault(ln["track"], []).append(i)
    for nm, gt in d["gts"].items():
        s = slots.get(nm)
        if s is None:
            continue
        w = d["written"].get(nm)
        if w is not None and HON(*err(w, gt)):
            prod[si] += 1
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
            R.append(np.log1p(m if m is not None else 5000.0) + 3.0 * max(0.0, 0.9 - c))
            IDX.append((si, nm, i))
X = np.array(X, float); Y = np.array(Y, int); R = np.array(R, float)

wells = sorted({WELL.get(d["name"], "?") for d in sheets})
fold_of = {w: k % a.folds for k, w in enumerate(wells)}
sheet_fold = np.array([fold_of[WELL.get(d["name"], "?")] for d in sheets])
print(f"листов {len(sheets)}, скважин {len(wells)}, пар {len(Y)}, честных пар {int(Y.sum())}")
print(f"прод берёт {sum(prod)} честных кривых\n")


def assign_count(scores, mask_sheets):
    per = {}
    for k, (si, nm, i) in enumerate(IDX):
        if mask_sheets[si]:
            per.setdefault(si, []).append((scores[k], nm, i))
    out = {}
    for si, cand in per.items():
        d = sheets[si]
        cand.sort(key=lambda q: -q[0])
        used, got = set(), {}
        for sc, nm, i in cand:
            if nm in got or i in used:
                continue
            got[nm] = i; used.add(i)
        out[si] = sum(1 for nm, i in got.items()
                      if HON(*err(d["lines"][i]["tr"], d["gts"][nm])))
    return out


print(f"{'фолд':<6}{'листов':>7}{'прод':>6}{'классиф.':>10}{'↑/↓':>9}{'регресс.':>10}{'↑/↓':>9}{'оракул':>8}")
tot = dict(prod=0, clf=0, reg=0, cu=0, cd=0, ru=0, rd=0, orc=0, n=0)
FOLD_SKIP = 0
pair_fold = np.array([sheet_fold[si] for si, _, _ in IDX])
for k in range(a.folds):
    te_s = sheet_fold == k
    tr_p = pair_fold != k
    if Y[tr_p].sum() < 10 or not te_s.any():
        FOLD_SKIP += 1          # §6.106: молча пропущенный фолд = молча урезанная выборка
        continue
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)
    clf.fit(X[tr_p], Y[tr_p])
    reg = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=0)
    reg.fit(X[tr_p], R[tr_p])
    sc = np.zeros(len(Y)); sc[pair_fold == k] = clf.predict_proba(X[pair_fold == k])[:, 1]
    sr = np.zeros(len(Y)); sr[pair_fold == k] = -reg.predict(X[pair_fold == k])
    so = np.array([1.0 if y else 0.0 for y in Y])
    gc = assign_count(sc, te_s); gr = assign_count(sr, te_s); go = assign_count(so, te_s)
    idxs = sorted(gc)
    p = sum(prod[i] for i in idxs)
    cu = sum(1 for i in idxs if gc[i] > prod[i]); cd = sum(1 for i in idxs if gc[i] < prod[i])
    ru = sum(1 for i in idxs if gr[i] > prod[i]); rd = sum(1 for i in idxs if gr[i] < prod[i])
    print(f"{k:<6}{len(idxs):>7}{p:>6}{sum(gc.values()):>10}{f'{cu}/{cd}':>9}"
          f"{sum(gr.values()):>10}{f'{ru}/{rd}':>9}{sum(go.values()):>8}")
    tot["prod"] += p; tot["clf"] += sum(gc.values()); tot["reg"] += sum(gr.values())
    tot["cu"] += cu; tot["cd"] += cd; tot["ru"] += ru; tot["rd"] += rd
    tot["orc"] += sum(go.values()); tot["n"] += len(idxs)

# ⚠⚠ СВЕРКА ВЫБОРКИ (§6.106): фолд, пропущенный из-за нехватки положительных, уносит свои листы
# из ИТОГО молча — и таблица продолжает выглядеть как проверка по всей выборке.
# ⚠ Второй источник расхождения нашла сама эта сверка при первом же прогоне: лист, у которого НЕТ
# НИ ОДНОЙ пары (слот × трасса), в `assign_count` не попадает, а значит не попадает и в столбец
# «листов». На дельты он не влияет (даёт ноль всем сторонам), но число в таблице — это листы
# С КАНДИДАТАМИ, а не размер выборки, и читать его надо именно так.
_no_pairs = len(sheets) - len({si for si, _, _ in IDX})
_acc = tot["n"] + _no_pairs
print(f"  СВЕРКА ВЫБОРКИ: листов в тестах {tot['n']} + без пар {_no_pairs} = {_acc} против "
      f"{len(sheets)} загруженных; фолдов {a.folds - FOLD_SKIP} из {a.folds}"
      f"   {'★ СОШЛОСЬ' if _acc == len(sheets) and not FOLD_SKIP else '⛔ НЕ СОШЛОСЬ'}")

pc = lambda v: f"{100*(v-tot['prod'])/max(1,tot['prod']):+.0f}%"
print(f"{'-'*66}\n{'ИТОГО':<6}{tot['n']:>7}{tot['prod']:>6}{tot['clf']:>10}"
      f"{f'{tot[chr(99)+chr(117)]}/{tot[chr(99)+chr(100)]}':>9}{tot['reg']:>10}"
      f"{f'{tot[chr(114)+chr(117)]}/{tot[chr(114)+chr(100)]}':>9}{tot['orc']:>8}")
print(f"\nклассификатор {pc(tot['clf'])}, регрессия {pc(tot['reg'])} к проду; "
      f"потолок {tot['orc']} ({100*tot['orc']/max(1,tot['prod']):.0f}% от прода)")
