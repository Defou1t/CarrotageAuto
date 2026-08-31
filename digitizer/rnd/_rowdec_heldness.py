r"""_rowdec_heldness.py — СКОЛЬКО ИЗ +104 (§6.150) ДЕРЖИТСЯ НА ЧЕСТНО ДЕРЖАННЫХ СКВАЖИНАХ (§6.156)

ОТКУДА ВОПРОС. §6.153 измерил: `auto5` выбирает чекпойнт по скважине, ВЫЧИСЛЕННОЙ ИЗ ИМЕНИ ФАЙЛА,
и промахивается на 52.8% листов отгрузки; скважины нет в обучении лишь у 0.6% ⇒ **51.0% листов
декодирует модель, ВИДЕВШАЯ эту скважину**. Утечка работает В ПОЛЬЗУ ДЕКОДЕРА ⇒ §6.144, §6.148 и
§6.150 (+104) смещены в его пользу. Вопрос: НА СКОЛЬКО.

★★ НИ ОДНОГО НОВОГО ПРОГОНА НЕ НУЖНО. Все три выдачи заморожены на ОДНИХ И ТЕХ ЖЕ листах: прод
(`ab_rowdec_pair/A`), декодер ВСЕГДА (`ab_rowdec_pair/B`), порог §6.150 (`ab_rdpick_c/D`).
Признак «держан или с утечкой» считается офлайн по имени листа и манифесту обучения.

⚠⚠ СЧЁТ БЕРЁТСЯ ВЕДУЩИЙ — БЕЗЫМЯННЫЙ. Первая редакция этого стенда разложила ИМЕННОЙ счёт из
дампов `_trace_prod_ab` (638 против 659) и назвала это «разложением +104». Это разные метрики:
+104 снят БЕЗ ИМЕНИ (§6.150, `_name_cost_prod.py`), и решение Эдуарда 20.08 — вести по безымянному.
⇒ счёты берутся из `_name_cost_prod.py --dump`, то есть у ТОЙ ЖЕ реализации, что дала 971→1075,
а не у второй копии метрики.

⚠ ЧЕСТНОСТЬ ЧТЕНИЯ. Разложение ПОСТФАКТУМ, а не новый предрегистрированный прогон. Но НАПРАВЛЕНИЕ
предсказано §6.153 ДО просмотра чисел: утечка помогает декодеру ⇒ на ДЕРЖАННЫХ его преимущество
обязано быть МЕНЬШЕ. Проверяется это предсказание.
⚠ Подмножества НЕ СЛУЧАЙНЫ: держанность зависит от того, как назван файл, а значит от эпохи и
семейства бланка ⇒ разрыв может быть СОСТАВОМ, а не утечкой. Названо в чтении, за доказанное
не выдаётся.

  <ComfyUI>\python_embeded\python.exe _name_cost_prod.py --dir …/ab_rowdec_pair --mode A --mode2 B \
      --dump F:/nds/output/taskS/percurve_pair.pkl
  <ComfyUI>\python_embeded\python.exe _name_cost_prod.py --dir …/ab_rdpick_c --mode D \
      --dump F:/nds/output/taskS/percurve_rule.pkl
  python _rowdec_heldness.py
"""
import sys, argparse, pickle, json, glob
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter
import numpy as np
from auto import meta as M, rowdec

ap = argparse.ArgumentParser()
ap.add_argument("--pair", default=r"F:/nds/output/taskS/percurve_pair.pkl")
ap.add_argument("--rule", default=r"F:/nds/output/taskS/percurve_rule.pkl")
ap.add_argument("--crops", default=r"F:/nds/output/taskS/rowdec_crops")
ap.add_argument("--arch", default=r"F:\nds\projects\Archive")
ap.add_argument("--perm", type=int, default=20000)
ap.add_argument("--named", action="store_true", help="считать ИМЕННОЙ счёт (справочный)")
a = ap.parse_args()
J = 0 if a.named else 1          # 0 = с именем, 1 = БЕЗ имени (ведущий)
WHAT = "С ИМЕНЕМ (справочный)" if a.named else "БЕЗ ИМЕНИ (ведущий, решение 20.08)"

for f in (a.pair, a.rule):
    if not Path(f).is_file():
        sys.exit(f"⛔ нет {f} — сперва выгрузить полистные счёты (`_name_cost_prod.py --dump`)")
dp, dr = pickle.load(open(a.pair, "rb")), pickle.load(open(a.rule, "rb"))
A, B, D = dp["A"], dp["B"], dr["D"]
print(f"счёт: {WHAT}")
print(f"замороженные выдачи: прод {len(A)}, декодер-всегда {len(B)}, порог §6.150 {len(D)}")

# ── держанность листа: та же арифметика, что в §6.153 ────────────────────────────────────────
tr_well, wells = {}, set()
for f in sorted(glob.glob(str(Path(a.crops) / "man_*of*.json"))):
    for t in json.loads(Path(f).read_text(encoding="utf-8"))["tracks"]:
        tr_well[t["sheet"]] = t["well"]; wells.add(t["well"])
FOLD = {w.upper(): i % 5 for i, w in enumerate(sorted(wells))}
mism = [w for w in wells if rowdec.fold_of_well(w) != FOLD[w.upper()]]
print(f"★ СВЕРКА КАРТЫ ФОЛДОВ с rowdec.fold_of_well: скважин {len(wells)}, расхождений {len(mism)}"
      f"   {'★ СОШЛАСЬ' if not mism else '⛔ НЕ СОШЛАСЬ — раздел недействителен'}")
if mism:
    sys.exit(1)

ARCH = {}
for wlg in Path(a.arch).glob("*/wlg"):
    for q in wlg.glob("*.nlgx"):
        ARCH.setdefault(q.name, wlg.parent.name)
d2w = {}
for s, w in tr_well.items():
    if ARCH.get(s):
        d2w.setdefault(ARCH[s], Counter())[w] += 1
d2w = {d: c.most_common(1)[0][0] for d, c in d2w.items()}

K = sorted(set(A) & set(B) & set(D))
print(f"★ СВЕРКА ОБЪЁМА: листов во ВСЕХ трёх выдачах {len(K)}")
held, leak, unk = [], [], []
for s in K:
    tw = d2w.get(ARCH.get(s))
    if tw is None:
        unk.append(s); continue
    used = rowdec.fold_of_well(M.parse_filename(Path(s)).well)
    (held if (0 if used is None else used) == FOLD[tw.upper()] else leak).append(s)
print(f"   ★ ДЕРЖАННЫХ {len(held)}, С УТЕЧКОЙ {len(leak)}, скважины нет в обучении вовсе {len(unk)}")
if not held or not leak:
    sys.exit("одно из подмножеств пусто")

g = lambda src, ks: np.array([src[k][J] for k in ks], float)
KA = held + leak
base = g(A, KA); dD = g(D, KA) - base; dB = g(B, KA) - base
isH = np.r_[np.ones(len(held), bool), np.zeros(len(leak), bool)]

print(f"\n★★ ВЕДУЩИЙ СЧЁТ ({WHAT}), Δ к ПРОДУ")
print(f"   {'подмножество':<34}{'листов':>7}{'прод':>7}{'порог':>7}{'Δ':>6}{'Δ%':>8}"
      f"{'декодер':>9}{'Δ':>6}")
for m, nm in ((isH, "★ ДЕРЖАННЫЕ (чекпойнт не видел)"), (~isH, "⛔ С УТЕЧКОЙ (чекпойнт видел)")):
    b = base[m].sum()
    print(f"   {nm:<34}{int(m.sum()):>7}{int(b):>7}{int(b+dD[m].sum()):>7}"
          f"{dD[m].sum():>+6.0f}{100*dD[m].sum()/max(1,b):>+7.1f}%"
          f"{int(b+dB[m].sum()):>9}{dB[m].sum():>+6.0f}")
print(f"   {'ВСЕГО (сверка с §6.150)':<34}{len(KA):>7}{int(base.sum()):>7}"
      f"{int(base.sum()+dD.sum()):>7}{dD.sum():>+6.0f}{100*dD.sum()/max(1,base.sum()):>+7.1f}%"
      f"{int(base.sum()+dB.sum()):>9}{dB.sum():>+6.0f}")

# ⚠⚠ ПЕРЕСТАНОВКА С ПЕРЕСЧЁТОМ ЗНАМЕНАТЕЛЯ. Первая редакция перемешивала ярлыки, но проценты
# считала по ЗАФИКСИРОВАННЫМ базам исходных групп — нулевое распределение выходило не про ту
# величину. Здесь каждая перестановка пересчитывает и числитель, и знаменатель.
rng = np.random.default_rng(20260829)
nH = len(held)
for d, nm in ((dD, "ПОРОГ §6.150"), (dB, "ДЕКОДЕР ВСЕГДА")):
    ph = 100 * d[isH].sum() / max(1, base[isH].sum())
    pl = 100 * d[~isH].sum() / max(1, base[~isH].sum())
    obs = pl - ph
    null = np.empty(a.perm)
    for i in range(a.perm):
        idx = rng.permutation(len(KA))
        h, l = idx[:nH], idx[nH:]
        null[i] = (100 * d[l].sum() / max(1, base[l].sum())
                   - 100 * d[h].sum() / max(1, base[h].sum()))
    p = float((np.abs(null) >= abs(obs)).mean())
    print(f"\n★ {nm}: прирост к проду {ph:+.1f}% на ДЕРЖАННЫХ против {pl:+.1f}% С УТЕЧКОЙ")
    print(f"   разрыв {obs:+.1f} п.п., перестановочный p = {p:.4f}"
          f"   {'★ различимо' if p < 0.05 else '⚠ НЕ РАЗЛИЧИМО на этом объёме'}")

# ═══ РАССЛОЕНИЕ ПО ТРУДНОСТИ ЛИСТА ═══════════════════════════════════════════════════════════
# ⚠⚠ ГЛАВНОЕ ВОЗРАЖЕНИЕ К ЭТОМУ РАЗДЕЛУ — «разрыв это СОСТАВ, а не утечка», и оно не пустое:
# подмножества РЕАЛЬНО разной трудности (честных у прода на лист: держанные против с утечкой —
# печатается ниже). Если весь разрыв объясняется трудностью, то ВНУТРИ слоя одинаковой трудности
# он обязан исчезнуть. Проверяется прямо здесь, на тех же данных.
# ⚠ Расслоение по ПРОДУ, а не по числу экспертных кривых: прод — общий знаменатель обеих ветвей и
# не зависит от того, какой чекпойнт выбран.
print("\n★★ РАССЛОЕНИЕ ПО ТРУДНОСТИ (слой = сколько честных дал ПРОД на листе)")
print(f"   средних честных прода на лист: держанные {base[isH].mean():.2f}, "
      f"с утечкой {base[~isH].mean():.2f}"
      f"   {'⚠ СОСТАВ РАЗЛИЧАЕТСЯ' if abs(base[isH].mean()-base[~isH].mean()) > 0.1 else '★ близки'}")
print(f"   {'слой (честных у прода)':<26}{'держ.':>7}{'Δ%':>8}{'утечка':>8}{'Δ%':>8}{'разрыв':>9}")
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 10**9)]
for lo, hi in BINS:
    m = (base >= lo) & (base <= hi)
    mh, ml = m & isH, m & ~isH
    if mh.sum() < 5 or ml.sum() < 5:
        continue
    bh, bl = base[mh].sum(), base[ml].sum()
    # ⚠ В слое «прод дал 0» относительный прирост неопределён (знаменатель 0) — печатаем АБСОЛЮТ.
    if bh == 0 or bl == 0:
        # ⚠ Знаменатель нулевой (прод не дал ничего) ⇒ проценты неопределены. Сравнивать абсолюты
        # НЕЛЬЗЯ: в слое разное число листов. Считаем прирост НА ЛИСТ — это и сопоставимо, и
        # честно, потому что база у обеих сторон одна и та же (ноль).
        ah, al = dD[mh].sum() / mh.sum(), dD[ml].sum() / ml.sum()
        print(f"   {f'{lo}' if lo == hi else f'{lo}-{hi if hi < 10**9 else 'и выше'}':<26}"
              f"{int(mh.sum()):>7}{ah:>+7.2f}/л{int(ml.sum()):>7}{al:>+7.2f}/л"
              f"{al-ah:>+8.2f}")
        continue
    ph_, pl_ = 100 * dD[mh].sum() / bh, 100 * dD[ml].sum() / bl
    print(f"   {f'{lo}' if lo == hi else f'{lo}-{hi if hi < 10**9 else 'и выше'}':<26}"
          f"{int(mh.sum()):>7}{ph_:>+7.1f}%{int(ml.sum()):>8}{pl_:>+7.1f}%{pl_-ph_:>+8.1f}")
print("   ⇒ если разрыв держится ВНУТРИ слоёв — «это просто разная трудность» не объясняет его;")
print("     если исчезает — объясняет, и §6.156 надо читать как замер СОСТАВА, а не утечки.")

print("\n★★★ ЧТЕНИЕ (направление предсказано §6.153 ДО просмотра чисел):")
ph = 100 * dD[isH].sum() / max(1, base[isH].sum())
pl = 100 * dD[~isH].sum() / max(1, base[~isH].sum())
print(f"   Порог даёт {ph:+.1f}% на держанных против {pl:+.1f}% с утечкой.")
print("   ⇒ Читать §6.150 как «+104 на выборке, половина которой декодеру знакома». Честная")
print("     оценка для НЕЗНАКОМОЙ площади — держанная строка, и она НИЖЕ.")
print("⚠ Разрыв может быть СОСТАВОМ подмножеств, а не утечкой: держанность зависит от того, как")
print("  назван файл. Разделить может только прогон `auto5` с ПРАВИЛЬНО определённой скважиной")
print("  на тех же листах — это и есть следующий замер, а не этот.")
