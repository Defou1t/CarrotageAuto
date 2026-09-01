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
ap.add_argument("--sheets", default=r"F:/nds/output/taskS/pickfolds/sheets_f{f}.txt",
                help="списки листов фолдов: по ним считается ОЖИДАЕМЫЙ объём")
ap.add_argument("--prod", default=r"F:/nds/output/taskS/ab_rowdec_pair", help="замороженный прод")
ap.add_argument("--prod-mode", default="A")
ap.add_argument("--rule", default=r"F:/nds/output/taskS/ab_rdpick_c", help="замороженный порог")
ap.add_argument("--rule-mode", default="D")
# ★★ ВЕДУЩИЙ СЧЁТ ВЕТКИ — БЕЗЫМЯННЫЙ (решение Эдуарда 20.08), и дампы `_trace_prod_ab` его НЕ
# ДЕРЖАТ: там ИМЕННОЙ (§6.126, сказано в шапке `_name_cost_prod.py`). Предрегистрированное правило
# «Δ ≥ +20 к порогу (1075)» написано на БЕЗЫМЯННОЙ шкале: 1075 — это она. На именной тот же
# замороженный прогон даёт 696 против 670, то есть весь ход прод→порог там +26, а не +99, и
# «+20» означало бы совсем другую высоту. ⇒ правило читаем по ведущему счёту, у ТОЙ ЖЕ реализации,
# что дала 976 → 1075 (`_name_cost_prod.py --dump`), а именной печатаем вторым числом.
ap.add_argument("--per-model", default=r"F:/nds/output/taskS/percurve_pickf{f}.pkl",
                help="выгрузка ведущего счёта фолда, {f} — номер фолда")
ap.add_argument("--per-prod", default=r"F:/nds/output/taskS/percurve_pair.pkl")
ap.add_argument("--per-rule", default=r"F:/nds/output/taskS/percurve_rule.pkl")
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



def lead(path, mode):
    """→ {лист: БЕЗЫМЯННЫХ} из выгрузки `_name_cost_prod.py --dump` (значение — кортеж
    «с именем, без имени, экспертных»). None, если выгрузки нет: молча подставить именной счёт
    вместо ведущего нельзя — правило чтения написано не на нём."""
    f = Path(path)
    if not f.exists():
        return None
    d = pickle.load(open(f, "rb"))
    return {k: v[1] for k, v in d[mode].items()} if mode in d else None


print("★ СБОРКА ПЯТИ ФОЛДОВ (у каждого свой держанный вес)")
mod, bad, short = {}, [], []
for f in range(a.folds):
    p, got, want = load(Path(a.dir) / f"f{f}", "M")
    # ★★ ДОЗАПУСК УПАВШИХ ЛИСТОВ ЖИВЁТ В ОТДЕЛЬНОМ КАТАЛОГЕ `fN_fix` — и это не прихоть.
    # `load()` берёт ОДИН знаменатель `of<N>` (самый свежий), поэтому дозапуск на 2 шардах,
    # положенный рядом с восемью, ПОДМЕНИЛ БЫ выборку двумя листами. Сливаем явно.
    fx = Path(a.dir) / f"f{f}_fix"
    if fx.is_dir():
        pf, gf, wf = load(fx, "M")
        inter = set(pf) & set(p)
        p = dict(p); p.update(pf)
        print(f"   фолд {f}: + дозапуск {len(pf)} листов (шардов {gf} из {wf})"
              f"{f', пересечение {len(inter)} — перекрыто' if inter else ''}")
    # ★★ ПОЛНОТА СЧИТАЕТСЯ ПО ЛИСТАМ, А НЕ ПО ШАРДАМ (§6.166). Шард может доработать до конца и
    # записать дамп, УРОНИВ по дороге отдельные листы: так фолд 2 оказался на 206 из 208, и ни
    # один контролёр этого не увидел. Память кончается на КРУПНЫХ листах ⇒ потеря смещает выборку.
    n_want = 0
    try:
        n_want = sum(1 for l in open(a.sheets.format(f=f), encoding="utf-8") if l.strip())
    except OSError:
        pass
    flag = ""
    if n_want and len(p) != n_want:
        flag = f"   ⛔ ЛИСТОВ {len(p)} ИЗ {n_want} — НЕПОЛНЫЙ ПО ЛИСТАМ"
        short.append((f, len(p), n_want))
    print(f"   фолд {f}: листов {len(p):>4}, шардов {got} из {want}"
          f"{'' if got == want and want else '   ⚠⚠ НЕПОЛНЫЙ ПО ШАРДАМ'}{flag}")
    if not want or got != want:
        bad.append(f)
    inter = set(p) & set(mod)
    if inter:
        print(f"   ⛔ ФОЛД {f} ПЕРЕСЕКАЕТСЯ С ПРЕДЫДУЩИМИ на {len(inter)} листах — "
              f"фолды обязаны быть непересекающимися")
        sys.exit(1)
    mod.update(p)
if bad:
    print(f"⚠⚠ НЕПОЛНЫЕ ПО ШАРДАМ ФОЛДЫ: {bad} — числа ниже НЕ ОКОНЧАТЕЛЬНЫ")
if short:
    print(f"⛔⛔ НЕПОЛНЫЕ ПО ЛИСТАМ: " +
          ", ".join(f"фолд {f}: {g} из {w}" for f, g, w in short))
    print(f"   Это НЕ мелочь: память кончается на КРУПНЫХ листах, значит выборка смещена по")
    print(f"   размеру бланка. Лечение — дозапуск списком `_fold_missing.py --out` в каталог")
    print(f"   `f<N>_fix` с МЕНЬШИМ числом шардов; тик делает это сам.")
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

# ★★ ОЖИДАЕМЫЙ ОБЪЁМ ЗАМЕРА. Полнота шардов говорит лишь «фолд досчитан», а правило чтения
# написано на ВЕСЬ корпус (1130 листов = 221+263+208+300+138). Замер по части фолдов — законный
# промежуточный взгляд, но правило к нему НЕ ПРИМЕНЯЕТСЯ, и стенд обязан это сказать сам.
want = 0
for _f in range(5):
    try:
        want += sum(1 for _l in open(a.sheets.format(f=_f), encoding="utf-8") if _l.strip())
    except OSError:
        want = 0
        break
if want:
    print(f"★ ОЖИДАЕМЫЙ ОБЪЁМ (списки листов пяти фолдов): {want}; посчитано {len(common)}"
          f"   {'★ ПОЛНЫЙ' if len(common) >= want * 0.98 else '⚠ ЧАСТИЧНЫЙ — правило чтения не применяется'}")
else:
    print("⚠ списки листов фолдов не прочитаны — ожидаемый объём неизвестен")
full_volume = bool(want) and len(common) >= want * 0.98
K = sorted(common)
M = np.array([mod[k] for k in K], float)
P = np.array([prod[k] for k in K], float)
R = np.array([rule[k] for k in K], float)
# ★ ВЕДУЩИЙ (безымянный) счёт тех же листов — из выгрузок `_name_cost_prod.py --dump`
lead_mod, miss = {}, []
for f in range(a.folds):
    q = lead(a.per_model.format(f=f), "M")
    if q is None:
        miss.append(a.per_model.format(f=f))
    else:
        lead_mod.update(q)
lead_prod = lead(a.per_prod, a.prod_mode)
lead_rule = lead(a.per_rule, a.rule_mode)
if lead_prod is None:
    miss.append(a.per_prod)
if lead_rule is None:
    miss.append(a.per_rule)
uncov = set(K) - (set(lead_mod) & set(lead_prod) & set(lead_rule)) if not miss else set(K)
have_lead = not miss and not uncov
if miss:
    print(f"\n⚠⚠ ВЕДУЩЕГО (БЕЗЫМЯННОГО) СЧЁТА НЕТ — не выгружено: {', '.join(miss)}")
elif uncov:
    print(f"\n⚠⚠ ВЕДУЩИЙ СЧЁТ НЕ ПОКРЫВАЕТ ЗАМЕР: {len(uncov)} листов из {len(K)} без счёта")


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


def block(mod_, prod_, rule_, title):
    m = np.array([mod_[k] for k in K], float)
    q = np.array([prod_[k] for k in K], float)
    r = np.array([rule_[k] for k in K], float)
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"{'путь':<34}{'честных':>9}")
    print(f"{'прод (замороженный)':<34}{int(q.sum()):>9}")
    print(f"{'порог §6.150 (замороженный)':<34}{int(r.sum()):>9}")
    print(f"{'★ ОБУЧЕННЫЙ ВЫБОР (держанно)':<34}{int(m.sum()):>9}")
    d_r, p_r = paired(m, r, "ОБУЧЕННЫЙ ВЫБОР против ПОРОГА")
    paired(m, q, "ОБУЧЕННЫЙ ВЫБОР против ПРОДА")
    paired(r, q, "ПОРОГ против ПРОДА (сверка с §6.150)")
    return d_r, p_r


d_rule = p_rule = None
if have_lead:
    d_rule, p_rule = block(lead_mod, lead_prod, lead_rule,
                           "★★★ ВЕДУЩИЙ СЧЁТ — БЕЗ ИМЕНИ (критерий Эдуарда; шкала правила чтения)")
block(mod, prod, rule, "СПРАВОЧНЫЙ СЧЁТ — С ИМЕНЕМ (дампы `_trace_prod_ab`)")

# ★★★ ЧТЕНИЕ. Порог зафиксирован в `_pickmodel_run.ps1` ДО прогона: перенос состоялся при
# Δ ≥ +20 к ПОРОГУ при p < 0.01; Δ ≤ 0 — не состоялся, и это тоже результат.
# ⚠ Правило написано на БЕЗЫМЯННОЙ шкале (порог там 1075) — на именной оно не читается.
print(f"\n★★★ ЧТЕНИЕ (правило зафиксировано ДО прогона: Δ ≥ +20 к порогу при p < 0.01,")
print(f"    по ВЕДУЩЕМУ безымянному счёту — той шкале, на которой порог равен 1075)")
if not have_lead:
    print("   ⛔ ПРАВИЛО НЕ ПРИМЕНЯЕТСЯ: ведущего счёта нет. Выгрузить и повторить:")
    print("      python _name_cost_prod.py --dir <…>/ab_pickmodel/f<N> --mode M \\")
    print("          --dump F:/nds/output/taskS/percurve_pickf<N>.pkl")
elif not full_volume:
    print(f"   ⛔ ПРАВИЛО НЕ ПРИМЕНЯЕТСЯ: посчитано {len(common)} листов из {want or '?'} —\n"
          f"   это часть корпуса, а правило писано на весь. Числа выше читать как промежуточные.")
elif short:
    print(f"   ⛔ ПРАВИЛО НЕ ПРИМЕНЯЕТСЯ: фолды неполны ПО ЛИСТАМ {short} —\n"
          f"   выборка смещена по размеру бланка, а не просто мала.")
elif bad:
    print(f"   ⛔ ПРАВИЛО НЕ ПРИМЕНЯЕТСЯ: фолды {bad} неполны — объём не тот, на котором оно писано.")
elif d_rule >= 20 and p_rule < 0.01:
    print(f"   Δ = {d_rule:+.0f} при p = {p_rule:.5f} ⇒ ★ ПЕРЕНОС СОСТОЯЛСЯ: обученный выбор")
    print("   обгоняет порог и на выданных файлах.")
elif d_rule <= 0:
    print(f"   Δ = {d_rule:+.0f} ⇒ ⛔ ПЕРЕНОС НЕ СОСТОЯЛСЯ. Держанное преимущество (+47 на")
    print("   раскладке) до файла не доживает — значит оно жило в раскладке, а не в ведении.")
else:
    print(f"   Δ = {d_rule:+.0f} при p = {p_rule:.5f} ⇒ ⚠ НИ ТО НИ ДРУГОЕ: направление верное,")
    print("   но предрегистрированный порог не взят. Цитировать как «не доказано», а не как выигрыш.")
print(f"\n⚠ Схема ИЗМЕРИТЕЛЬНАЯ (вес на фолд + `auto5`): на НЕЗНАКОМОЙ площади так не будет.")
print(f"⚠ Цена второго пути ЗАМЕРЕНА (§6.160, `_path_cost.py`): ×1.5 от прода на GPU, "
      f"×2 на CPU. Прежнее «декодер втрое дороже» — не замер и неверно: сам декодер ×0.17 от прода.")
