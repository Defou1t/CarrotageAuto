r"""_degen_ab_sum.py — СВОДКА A/B ЗАЩИТЫ ОТ ВЫРОЖДЕННОГО ЦВЕТОВОГО КАНАЛА (§6.152 → §6.155)

ОТКУДА ВОПРОС. `color_max_frac = 0.28` («канал, покрывающий десятки процентов листа, — БУМАГА, а
не тушь») стоит РОВНО В ОДНОМ месте — `imaging.ink_foreground` (`imaging.py:186`). У
`trace2d._color_fg`, по которому прод ВЕДЁТ КАЖДУЮ ЛИНИЮ, этой защиты НЕТ: чёрный считается как
`dark_mask & ~ВСЕ_цветовые_каналы`, поэтому на пожелтевшем скане чёрный штрих вычитается из СВОЕЙ
ЖЕ маски, ран исчезает, и трассировщик садится на соседа (класс промаха ~70px, §6.152).
Распространённость: 64 листа из 1130 отгрузки (5.7%), red 35 / orange 29, медиана доли 0.598.

★★ ЧТО ДЕЛАЕТ ЭТОТ СТЕНД, ЧЕГО НЕ ДЕЛАЕТ `_trace_ab_sum.py`. Тот считает одну общую строку на
режим. Здесь выборка СОСТАВНАЯ, и в этом весь замысел:
  `sheets_bad` (64 задетых)   — сигнал: только тут условие `mean > color_max_frac` срабатывает;
  `sheets_ctl` (64 НЕзадетых) — КОНТРОЛЬ: правка обязана быть БИТ-В-БИТ продом, потому что
                                 условие не срабатывает вовсе.
Общая строка смешала бы их и спрятала бы главный вопрос: «а не трогает ли ручка то, что не
обещала». Поэтому контроль читается ПЕРВЫМ и умеет объявить весь замер недействительным.

⚠ Отпечаток выдачи — не отладка, а условие годности (§6.71): равные счётчики сами по себе не
доказывают ни что режим включился, ни что он не включился там, где не должен был.

  python _degen_ab_sum.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=r"F:/nds/output/taskS/degen_ab/run")
ap.add_argument("--lists", default=r"F:/nds/output/taskS/degen_ab")
ap.add_argument("--base", default="A")
ap.add_argument("--test", default="B")
ap.add_argument("--perm", type=int, default=20000)
a = ap.parse_args()

files = sorted(Path(a.dir).glob("ab_*of*.pkl"))
if not files:
    sys.exit("нет данных — прогон ещё не дал ни одного дампа")
den = sorted({f.stem.split("of")[1] for f in files},
             key=lambda q: max(f.stat().st_mtime for f in files if f.stem.endswith("of" + q)))[-1]
files = [f for f in files if f.stem.endswith("of" + den)]
RES, FP = {}, {}
for f in files:
    d = pickle.load(open(f, "rb"))
    for nm in (a.base, a.test):
        t, per = d["res"].get(nm, ({}, {}))
        RES.setdefault(nm, {}).update(per)
        FP.setdefault(nm, {}).update(d["fp"].get(nm, {}))
print(f"шардов {len(files)} из {den}"
      f"{'' if len(files) == int(den) else '   ⚠⚠ ПРОГОН НЕПОЛНЫЙ, числа НЕ ОКОНЧАТЕЛЬНЫ'}")

# ⚠⚠ КЛЮЧ ДАМПА — ПРОСТОЕ ИМЯ nlgx (`per[n.name]`, `_trace_prod_ab.py:448`), А НЕ каталог листа
# `<stem[:40]>_<хэш8>`. Первая редакция строила хэш и сопоставляла 0 ключей из 139 — то есть
# напечатала бы «задетых 0» и выглядела бы как честный нулевой результат. ⇒ сверка сопоставления
# ниже ОБЯЗАТЕЛЬНА и падает, а не предупреждает.
L = {}
for tag in ("bad", "ctl"):
    names = [x.strip() for x in (Path(a.lists) / f"sheets_{tag}.txt").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    L[tag] = set(names)
    print(f"   список {tag}: {len(names)} имён")
_all = set(RES.get(a.base, {}))
_hit = _all & (L["bad"] | L["ctl"])
if _all and not _hit:
    sys.exit("⛔ НИ ОДИН ключ дампа не сопоставился со списками — сверять нечего, "
             "стенд остановлен (иначе напечатал бы правдоподобный ноль)")
common = set(RES.get(a.base, {})) & set(RES.get(a.test, {}))
print(f"★ СВЕРКА ОБЪЁМА: листов у обоих режимов {len(common)}, "
      f"из них задетых {len(common & L['bad'])}, контрольных {len(common & L['ctl'])}, "
      f"вне списков {len(common - L['bad'] - L['ctl'])}")
if not common:
    sys.exit("нет общих листов")

# ═══ КОНТРОЛЬ ЧИТАЕТСЯ ПЕРВЫМ ════════════════════════════════════════════════════════════════
ctl = sorted(common & L["ctl"])
dif_ctl = [k for k in ctl if FP[a.test].get(k) != FP[a.base].get(k)]
print(f"\n★★★ КОНТРОЛЬ (НЕзадетые листы — выдача обязана совпасть БИТ-В-БИТ):")
print(f"   листов {len(ctl)}, выдача отличается на {len(dif_ctl)}")
if dif_ctl:
    print("   ⛔⛔ ЗАМЕР НЕДЕЙСТВИТЕЛЕН: ручка меняет выдачу там, где условие "
          "`mean > color_max_frac` не срабатывает. Первые расхождения:")
    for k in dif_ctl[:5]:
        print(f"     {k}")
    sys.exit(1)
print("   ★ СОШЛОСЬ — ручка трогает ровно то, что обещала")

bad = sorted(common & L["bad"])
dif_bad = [k for k in bad if FP[a.test].get(k) != FP[a.base].get(k)]
print(f"\n★★ ЗАДЕТЫЕ ЛИСТЫ: {len(bad)}, выдача отличается на {len(dif_bad)}"
      f"{'   ⛔ РЕЖИМ НЕ ВКЛЮЧИЛСЯ' if not dif_bad else ''}")
if not dif_bad:
    sys.exit(1)

A = np.array([RES[a.base][k] for k in bad], float)
B = np.array([RES[a.test][k] for k in bad], float)
d = B - A
obs, up, dn = float(d.sum()), int((d > 0).sum()), int((d < 0).sum())
rng = np.random.default_rng(20260829)
nz = d[d != 0]
if len(nz):
    sg = rng.integers(0, 2, size=(a.perm, len(nz))) * 2 - 1
    p = float((np.abs((sg * np.abs(nz)).sum(1)) >= abs(obs)).mean())
else:
    p = 1.0
print(f"\n{'режим':<28}{'честных':>9}")
print(f"{'прод (A)':<28}{int(A.sum()):>9}")
print(f"{'★ с защитой (B)':<28}{int(B.sum()):>9}")
print(f"\n★ Δ = {obs:+.0f} кривых ({100*obs/max(1, A.sum()):+.1f}%) на {len(bad)} задетых листах")
print(f"   листов ↑ {up} / ↓ {dn}, перестановочный p = {p:.5f}")

# ★★★ ЧТЕНИЕ. Правило зафиксировано в `_degen_ab_run.ps1` ДО прогона.
print("\n★★★ ЧТЕНИЕ (правило зафиксировано ДО прогона):")
if obs >= 8 and p < 0.05:
    print(f"   Δ = {obs:+.0f} при p = {p:.5f} ⇒ ★ ЕСТЬ О ЧЁМ ГОВОРИТЬ: защита даёт кривые,")
    print("   правка в одну строку, нести Эдуарду.")
elif obs <= 0:
    print(f"   Δ = {obs:+.0f} ⇒ ⛔ ДЕФЕКТ РЕАЛЕН, НО КРИВЫХ НЕ ДАЁТ. Закрыть и не возвращаться:")
    print("   маска чинится, а линия всё равно теряется дальше по пути.")
else:
    print(f"   Δ = {obs:+.0f} при p = {p:.5f} ⇒ ⚠ НАПРАВЛЕНИЕ ВЕРНОЕ, ВЕЛИЧИНА НЕ ДОКАЗАНА.")
    print("   Цитировать как «не доказано», а не как выигрыш (§6.114 durable).")
print(f"\n⚠ Мерено на 64 задетых листах отгрузки из 1130; по корпусу доля задетых 5.7%.")
print("⚠ Правится ТОЛЬКО чёрная ветка `_color_fg`; цветная линия на вырожденном канале — "
      "отдельный вопрос со своим замером.")
