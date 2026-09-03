r"""_combo_sum.py — СЛОЖЕНИЕ ПРЕДГЕЙТА И ОБУЧЕННОГО ВЫБОРА НА ЧЕСТНОМ КОРПУСЕ (§6.175).

ОТКУДА ВОПРОС. Две прибавки измерены порознь и НИ РАЗУ ВМЕСТЕ:
  * ПРЕДГЕЙТ по листу (§6.161) решает, гнать ли второй путь на этом листе ВООБЩЕ, — он режет цену;
  * ОБУЧЕННЫЙ ВЫБОР по треку (§6.145, §6.169) решает, какой путь взять на треке, — он режет ошибку.
Они не исключают друг друга и работают на РАЗНЫХ уровнях. Сложатся ли прибавки, съедят ли друг
друга — не знал никто.

ЧТО ДЕЛАЕТ. Собирает ПОЛИСТНЫЕ честные счёты для обученного выбора, ДЕРЖАННО ПО СКВАЖИНАМ, из
потрековых честных счётов `pick_learn_honest.pkl` (пара `ab_wellmap/A` + `ab_rdhonest/B`, обе
стороны с честной скважиной, §6.153/§6.162), и выгружает их в виде дампа `percurve_*`, который
дальше читает `_pregate.py` — тот же гейт, та же вложенная выборка порога, те же три контроля.
Счёта не тратит.

★★ ГЛАВНАЯ ПРОВЕРКА, И ОНА НЕ ТАВТОЛОГИЯ. Та же композиция применяется к ПОРОГУ `n_dec ≤ 3` и
сверяется с РЕАЛЬНЫМ прогоном `ab_wellmap/H`. Прогон считался отдельно, композиция берётся из
потрековых счётов ⇒ совпадение здесь — настоящее подтверждение механизма, в отличие от §6.169,
где сверялись два чтения одного и того же. Не сойдись — композиции нельзя верить и для модели.

⚠ ВОСПРОИЗВОДИМОСТЬ ВАЖНЕЕ КРАСОТЫ: обучение повторено ПОБУКВЕННО за `_pick_learn.py` (та же
логистическая на numpy, те же 400 итераций, тот же вес |разность|, та же раскладка `i % 5` по
отсортированным скважинам) — иначе числа двух стендов спорили бы вместо того, чтобы сравниваться.
★ Отдельно считается вариант с ЧЕСТНОЙ НОРМИРОВКОЙ (среднее и разброс — только по обучающим
фолдам): в `_pick_learn.py` они берутся по ВСЕМ строкам, и это утечка распределения — та же, что
§6.161 нашёл у своей первой редакции в сетке порогов.

  <ComfyUI>\python_embeded\python.exe _combo_sum.py
  … --out F:/nds/output/taskS/percurve_pickhonest.pkl
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/pick_learn_honest.pkl")
ap.add_argument("--prod", default=r"F:/nds/output/taskS/percurve_wellmap.pkl")
ap.add_argument("--prod-mode", default="A")
ap.add_argument("--rule", default=r"F:/nds/output/taskS/percurve_wellmap.pkl")
ap.add_argument("--rule-mode", default="H", help="реальный прогон ПОРОГА — для сверки композиции")
ap.add_argument("--out", default=r"F:/nds/output/taskS/percurve_pickhonest.pkl")
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()

rows = pickle.load(open(a.cache, "rb"))
FEATS = sorted(rows[0][3])
wells = sorted({r[1] for r in rows})
fold = {w: i % a.folds for i, w in enumerate(wells)}          # ★ как в `_pick_learn.py`, буква в букву
X = np.array([[r[3][k] for k in FEATS] for r in rows], float)
hA = np.array([r[4] for r in rows], float)
hB = np.array([r[5] for r in rows], float)
F = np.array([fold[r[1]] for r in rows])
sheets = [r[0] for r in rows]
thr = np.array([r[3]["n_dec"] for r in rows])
y = (hB > hA).astype(float)
w = np.abs(hB - hA)
print(f"★ ЧЕСТНАЯ ПАРА: треков {len(rows)}, листов {len(set(sheets))}, скважин {len(wells)}, "
      f"признаков {len(FEATS)}")
print(f"★ опоры по трекам: прод {hA.sum():.0f}, декодер {hB.sum():.0f}, "
      f"оракул {np.maximum(hA, hB).sum():.0f}")


def fit(Z, idx, iters=400, lr=0.3, l2=1e-3):
    b = np.zeros(Z.shape[1] + 1)
    Zi = np.c_[Z[idx], np.ones(len(idx))]
    yi, wi = y[idx], w[idx]
    if wi.sum() == 0:
        return b
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Zi @ b))
        b -= lr * (Zi.T @ (wi * (p - yi)) / max(1e-9, wi.sum()) + l2 * b)
    return b


def decisions(honest_norm):
    """→ булев вектор по трекам: взять ли ДЕКОДЕР. Держанно по скважинам.

    `honest_norm=False` повторяет `_pick_learn.py` (нормировка по ВСЕМ строкам);
    `True` — нормировка внутри обучающих фолдов, без утечки распределения."""
    take = np.zeros(len(rows), bool)
    if not honest_norm:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    for f in range(a.folds):
        tr, te = np.where(F != f)[0], np.where(F == f)[0]
        if not len(te):
            continue
        if honest_norm:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        b = fit(Z, tr)
        pr = 1 / (1 + np.exp(-(np.c_[Z[te], np.ones(len(te))] @ b)))
        take[te] = pr >= 0.5
    return take


def thr_take():
    """Держанный ПОРОГ: значение подбирается на 4/5 скважин, применяется к держанной пятой."""
    take = np.zeros(len(rows), bool)
    for f in range(a.folds):
        tr, te = np.where(F != f)[0], np.where(F == f)[0]
        if not len(te):
            continue
        t = max(range(0, 8), key=lambda q: float(np.where(thr[tr] <= q, hB[tr], hA[tr]).sum()))
        take[te] = thr[te] <= t
    return take


def per_sheet(take):
    """Потрековый выбор → ПОЛИСТНЫЙ счёт."""
    out = {}
    for s, t, va, vb in zip(sheets, take, hA, hB):
        out[s] = out.get(s, 0.0) + (vb if t else va)
    return out


lead = lambda f, m: {k: v[1] for k, v in pickle.load(open(f, "rb"))[m].items()}
run_prod = lead(a.prod, a.prod_mode)
run_rule = lead(a.rule, a.rule_mode)

fixed3 = thr <= 3
comp_prod = per_sheet(np.zeros(len(rows), bool))
comp_rule3 = per_sheet(fixed3)

# ★★ СВЕРКА КОМПОЗИЦИИ С РЕАЛЬНЫМИ ПРОГОНАМИ — до любых выводов
print("\n★★ СВЕРКА КОМПОЗИЦИИ С РЕАЛЬНЫМИ ПРОГОНАМИ (композиция считается из ПОТРЕКОВЫХ счётов,")
print("   прогон — отдельный проход по листам; совпадение здесь НЕ тавтология, в отличие от §6.169)")
bad = 0
for nm, comp, run in (("прод", comp_prod, run_prod), ("порог n_dec≤3", comp_rule3, run_rule)):
    K = sorted(set(comp) & set(run))
    d = np.array([comp[k] - run[k] for k in K], float)
    ne = int((d != 0).sum())
    bad += ne > 0.02 * len(K)
    print(f"   {nm:<14} листов {len(K):>5}: композиция {sum(comp[k] for k in K):.0f}, "
          f"прогон {sum(run[k] for k in K):.0f}, расходится на {ne} листах "
          f"(сумма |разности| {np.abs(d).sum():.0f})   {'★' if ne <= 0.02*len(K) else '⛔'}")
if bad:
    # ⚠ печать, а не `sys.exit(str)`: текст ошибки уходит в stderr, который здесь не перенастроен
    # на utf-8, и при cp1251/cp866 отказ читается кракозябрами — то есть самый важный вывод теряется.
    print("⛔⛔ КОМПОЗИЦИЯ НЕ ВОСПРОИЗВОДИТ ПРОГОН — верить ей нельзя и для модели, замер прерван")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────── варианты, всё на одних листах
takes = {
    "прод (один путь)": np.zeros(len(rows), bool),
    "декодер целиком": np.ones(len(rows), bool),
    "порог n_dec≤3 (в §6.144)": fixed3,
    "порог, ДЕРЖАННО подобранный": thr_take(),
    "★ обученный выбор (как в _pick_learn)": decisions(False),
    "★ обученный выбор, честная нормировка": decisions(True),
    "оракул по треку (знает эталон)": hB > hA,
}
sheetsets = {nm: per_sheet(t) for nm, t in takes.items()}
K = sorted(set.intersection(*(set(v) for v in sheetsets.values())))
base = sum(sheetsets["прод (один путь)"][k] for k in K)
print(f"\n★★★ ЧЕСТНЫЙ КОРПУС, ВСЁ НА ОДНИХ И ТЕХ ЖЕ {len(K)} ЛИСТАХ (ведущий безымянный счёт)")
print("| вариант | честных | Δ к проду | декодер на треках |")
for nm, per in sheetsets.items():
    tot = sum(per[k] for k in K)
    use = 100.0 * takes[nm].mean()
    print(f"| {nm} | {tot:.0f} | {tot-base:+.0f} | {use:.1f}% |")

# ─────────────────────────────────────────────── выгрузка полистных счётов модели для `_pregate.py`
best = "★ обученный выбор (как в _pick_learn)"
per = sheetsets[best]
# ⚠ ВЕДУЩИЙ СЧЁТ ЕДИНСТВЕННЫЙ, ЧТО ЗДЕСЬ ЕСТЬ: именной по трекам не собирался. Ставлю −1 в
# позиции [0] и [2] НАМЕРЕННО — читатель, взявший именной счёт по ошибке, увидит бессмыслицу
# сразу, а не тихо посчитает по нему.
dump = {"M": {k: (-1, per[k], -1) for k in per}}
Path(a.out).write_bytes(pickle.dumps(dump))
print(f"\n★ полистные счёты обученного выбора выгружены: {a.out} (режим M, листов {len(per)})")
print("  ⚠ в дампе ЗАПОЛНЕН ТОЛЬКО ведущий безымянный счёт [1]; [0] и [2] равны −1 намеренно.")
print("  дальше — предгейт поверх этого выбора:")
print(f"    _pregate.py --prod {a.prod} --prod-mode {a.prod_mode} \\\n"
      f"                --both {a.out} --both-mode M --dumps F:/nds/output/taskS/ab_wellmap/H")
