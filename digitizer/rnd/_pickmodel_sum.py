r"""_pickmodel_sum.py — СВОДКА ЗАМЕРА «ОБУЧЕННЫЙ ВЫБОР ПУТИ НА ОТГРУЖАЕМОМ ПУТИ» (§6.145 → §6.154)

ОТКУДА ВОПРОС. §6.150 подтвердил на отгрузке ПОРОГ выбора пути: 971 → 1075, +104 (+10.7%),
p < 0.00001. Держанно по скважинам обученный выбор даёт 1126 против 1079 у порога, проверенного
ТАК ЖЕ (§6.145), то есть 51% доступного против 35%. Вопрос один: доживает ли это преимущество до
ВЫДАННЫХ ФАЙЛОВ. Пуловое преимущество до файла доживает не всегда (§6.123).

⚠⚠ ПОЧЕМУ ПЯТЬ КАТАЛОГОВ, А НЕ ОДИН. Полная подгонка `pick_model_v2.npz` видела ВСЕ скважины;
мерить ею те же листы — вес НА СВОЁМ ПОЛЕ, ошибка §6.114. Поэтому прогон идёт пятью кусками, у
каждого СВОЙ вес, не видевший скважин своего куска (`_pick_learn.py --export-folds`; выгрузка
сверена: пять весов на своих держанных фолдах дают ровно то же число, что `cv_model`).
⇒ Схема ИЗМЕРИТЕЛЬНАЯ, как `auto5` (§6.70): в проде на НОВОЙ скважине так не будет.

ЧТО СЧИТАЕТ. Сводит 5 фолдов в один полистный словарь и сравнивает с ДВУМЯ замороженными
прогонами на ТЕХ ЖЕ листах: прод (`ab_rowdec_pair/A`) и порог (`ab_rdpick_c/D`). Оба сравнения —
парные по листу, с перестановочным и знаковым тестом.

⚠ СВЕРКА ОБЪЁМА — ПЕРВОЕ, ЧТО ПЕЧАТАЕТСЯ. Три прогона обязаны сойтись по множеству листов; если
не сошлись, дельты считаются по ПЕРЕСЕЧЕНИЮ, и это говорится вслух (§6.71).

  python _pickmodel_sum.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/ab_pickmodel", help="каталог с f0..f4")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--prod", default=r"F:/nds/output/taskS/ab_rowdec_pair", help="замороженный прод")
ap.add_argument("--prod-mode", default="A")
ap.add_argument("--rule", default=r"F:/nds/output/taskS/ab_rdpick_c", help="замороженный порог")
ap.add_argument("--rule-mode", default="D")
ap.add_argument("--perm", type=int, default=20000)
a = ap.parse_args()


def load(d, mode):
    """→ {лист: честных} по одному разбиению шардов (правило §6 из `_trace_ab_sum`: один
    знаменатель `of<N>`, итог по ПОЛИСТНОМУ словарю, а не по накопителю)."""
    files = sorted(Path(d).glob("ab_*of*.pkl"))
    if not files:
        return {}, 0, 0
    den = sorted({f.stem.split("of")[1] for f in files},
                 key=lambda q: max(f.stat().st_mtime for f in files if f.stem.endswith("of" + q)))[-1]
    files = [f for f in files if f.stem.endswith("of" + den)]
    per, curves = {}, 0
    for f in files:
        dd = pickle.load(open(f, "rb"))
        t, p = dd["res"].get(mode, ({}, {}))
        per.update(p); curves += t.get("curves", 0)
    return per, len(files), int(den)


print("★ СБОРКА ПЯТИ ФОЛДОВ (у каждого свой держанный вес)")
mod, bad = {}, []
for f in range(a.folds):
    p, got, want = load(Path(a.dir) / f"f{f}", "M")
    print(f"   фолд {f}: листов {len(p):>4}, шардов {got} из {want}"
          f"{'' if got == want and want else '   ⚠⚠ НЕПОЛНЫЙ'}")
    if not want or got != want:
        bad.append(f)
    inter = set(p) & set(mod)
    if inter:
        print(f"   ⛔ ФОЛД {f} ПЕРЕСЕКАЕТСЯ С ПРЕДЫДУЩИМИ на {len(inter)} листах — "
              f"фолды обязаны быть непересекающимися")
        sys.exit(1)
    mod.update(p)
if bad:
    print(f"⚠⚠ НЕПОЛНЫЕ ФОЛДЫ: {bad} — числа ниже НЕ ОКОНЧАТЕЛЬНЫ")
print(f"   ★ всего листов у модели: {len(mod)}")

prod, _, _ = load(a.prod, a.prod_mode)
rule, _, _ = load(a.rule, a.rule_mode)
print(f"\n★ ЗАМОРОЖЕННЫЕ БАЗЫ: прод {len(prod)} листов ({a.prod}/{a.prod_mode}), "
      f"порог {len(rule)} листов ({a.rule}/{a.rule_mode})")

common = set(mod) & set(prod) & set(rule)
print(f"★ СВЕРКА ОБЪЁМА: общих листов {len(common)} из {len(mod)}/{len(prod)}/{len(rule)}"
      f"   {'★ СОШЛОСЬ' if len(common) == len(mod) == len(prod) == len(rule) else '⚠ СЧИТАЮ ПО ПЕРЕСЕЧЕНИЮ'}")
if not common:
    sys.exit("нет общих листов")
K = sorted(common)
M = np.array([mod[k] for k in K], float)
P = np.array([prod[k] for k in K], float)
R = np.array([rule[k] for k in K], float)
print(f"\n{'путь':<34}{'честных':>9}")
print(f"{'прод (замороженный)':<34}{int(P.sum()):>9}")
print(f"{'порог §6.150 (замороженный)':<34}{int(R.sum()):>9}")
print(f"{'★ ОБУЧЕННЫЙ ВЫБОР (держанно)':<34}{int(M.sum()):>9}")


def paired(x, y, nm):
    """Парный тест по полистным разностям: перестановка = случайные знаки (§6 правило 3 —
    без разброса набора прирост не число, а разговор)."""
    d = x - y
    obs = float(d.sum())
    nz = d[d != 0]
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    rng = np.random.default_rng(20260829)
    if len(nz):
        sg = rng.integers(0, 2, size=(a.perm, len(nz))) * 2 - 1
        null = (sg * np.abs(nz)).sum(1)
        p = float((np.abs(null) >= abs(obs)).mean())
    else:
        p = 1.0
    # знаковый: биномиальный на листах, где есть разница
    from math import comb
    n, k = up + dn, min(up, dn)
    ps = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n and n <= 1000 else float("nan")
    print(f"\n★ {nm}: Δ = {obs:+.0f} кривых ({100*obs/max(1,y.sum()):+.1f}%)")
    print(f"   листов ↑ {up} / ↓ {dn}, SD полистной разности {float(np.std(d))*np.sqrt(len(d)):.1f}")
    print(f"   перестановочный p = {p:.5f}   знаковый p = {ps:.6f}")
    return obs, p


d_rule, p_rule = paired(M, R, "ОБУЧЕННЫЙ ВЫБОР против ПОРОГА")
d_prod, p_prod = paired(M, P, "ОБУЧЕННЫЙ ВЫБОР против ПРОДА")

# ★★★ ЧТЕНИЕ. Порог зафиксирован в `_pickmodel_run.ps1` ДО прогона: перенос состоялся при
# Δ ≥ +20 к ПОРОГУ при p < 0.01; Δ ≤ 0 — не состоялся, и это тоже результат.
print(f"\n★★★ ЧТЕНИЕ (правило зафиксировано ДО прогона: Δ ≥ +20 к порогу при p < 0.01)")
if d_rule >= 20 and p_rule < 0.01:
    print(f"   Δ = {d_rule:+.0f} при p = {p_rule:.5f} ⇒ ★ ПЕРЕНОС СОСТОЯЛСЯ: обученный выбор")
    print("   обгоняет порог и на выданных файлах.")
elif d_rule <= 0:
    print(f"   Δ = {d_rule:+.0f} ⇒ ⛔ ПЕРЕНОС НЕ СОСТОЯЛСЯ. Держанное преимущество (+47 на")
    print("   раскладке) до файла не доживает — значит оно жило в раскладке, а не в ведении.")
else:
    print(f"   Δ = {d_rule:+.0f} при p = {p_rule:.5f} ⇒ ⚠ НИ ТО НИ ДРУГОЕ: направление верное,")
    print("   но предрегистрированный порог не взят. Цитировать как «не доказано», а не как выигрыш.")
print(f"\n⚠ Схема ИЗМЕРИТЕЛЬНАЯ (вес на фолд + `auto5`): на НЕЗНАКОМОЙ площади так не будет.")
print(f"⚠ Цена вдвое: считаются ОБА пути, декодер втрое дороже прода по листу.")
