r"""_pregate_curve.py — ПОЛКА ПРЕДГЕЙТА: НАСКОЛЬКО РЕЗУЛЬТАТ ДЕРЖИТСЯ ЗА КОНКРЕТНЫЙ ПОРОГ (§6.178).

ОТКУДА ВОПРОС. §6.161 и §6.177 честно мерили ПРОЦЕДУРУ выбора правила (вложенный отбор,
согласованный нуль, 200 раскладок), но конкретное правило `color_top_frac >= 0.8333` — АПОСТЕРИОРНОЕ
чтение. В прод же уходит не процедура, а ОДНО ЧИСЛО. ⇒ Надо знать, стоит ли результат на счастливом
пороге или на полке: если соседние значения дают то же самое, апостериорность почти ничего не стоит;
если результат — пик, цитировать его нельзя.

ЧТО ДЕЛАЕТ. Свип порога по замороженным выдачам, счёта не тратит. По каждому порогу: сколько
честных кривых даёт «прод там, где гейт отказал, второй путь там, где пропустил», и на скольких
листах при этом считается декодер (это и есть цена).

★★ ПОЧЕМУ ТАКОЙ ВЫБОР ВЫДАЧ ЗАКОНЕН, И ЭТО ПРОВЕРЕНО, А НЕ ПРЕДПОЛОЖЕНО (§6.178):
  1. `<лист>_understanding.json` — источник признака гейта — ПОБАЙТОВО одинаков у прогона «прод»
     и прогона «оба пути» на **1123 из 1123** листов ⇒ решение гейта определено ДО ведения и не
     зависит от того, какой прогон его считал.
  2. Там, где второй путь не взят ни на одном треке, выдача прогона «оба пути» ПОБАЙТОВО равна
     прод-выдаче на **187 из 187** таких листов ⇒ само присутствие декодера выдачу не портит.
⇒ Гейтованный прогон выдал бы ровно те же файлы, что выбираются здесь. Это не пуловая оценка
(§6.123), а отгрузочное число.

  <ComfyUI>\python_embeded\python.exe _pregate_curve.py
"""
import sys, argparse, json, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--prod", default=r"F:/nds/output/taskS/percurve_wellmap.pkl")
ap.add_argument("--prod-mode", default="A")
ap.add_argument("--rule", default=r"F:/nds/output/taskS/percurve_wellmap.pkl")
ap.add_argument("--rule-mode", default="H")
ap.add_argument("--model", nargs="*", default=[f"F:/nds/output/taskS/percurve_hpickf{f}.pkl"
                                               for f in range(5)])
ap.add_argument("--model-mode", default="M")
ap.add_argument("--dumps", nargs="+", default=[r"F:/nds/output/taskS/ab_wellmap/H"])
ap.add_argument("--feat", default="color_top_frac")
a = ap.parse_args()

lead = lambda f, m: {k: v[1] for k, v in pickle.load(open(f, "rb"))[m].items()}
A = lead(a.prod, a.prod_mode)
H = lead(a.rule, a.rule_mode)
M = {}
for f in a.model:
    M.update(lead(f, a.model_mode))

# ★ признак считается ТОЧНО так же, как в `_pregate.py`: доля самого частого значения `color`
#   среди линий листа. Разойдись формула — свип описывал бы другое правило, чем замеренное.
src = {}
for root in a.dumps:
    for d in Path(root).iterdir():
        if not d.is_dir():
            continue
        u = next(iter(d.glob("*_understanding.json")), None)
        if not u:
            continue
        L = (json.loads(u.read_text(encoding="utf-8")).get("lines") or [])
        if not L:
            continue
        c = Counter(str(x.get("color")) for x in L)
        src[u.name[:-len("_understanding.json")] + ".nlgx"] = max(c.values()) / len(L)

K = sorted(set(A) & set(H) & set(src) & (set(M) if M else set(A)))
if len(K) < 100:
    print(f"⛔ общих листов всего {len(K)} — свип бессмыслен")
    sys.exit(1)
ct = np.array([src[k] for k in K])
va = np.array([A[k] for k in K], float)
vh = np.array([H[k] for k in K], float)
vm = np.array([M[k] for k in K], float) if M else None
print(f"★ листов {len(K)}; {a.feat}: мин {ct.min():.3f}, медиана {np.median(ct):.3f}, макс {ct.max():.3f}")
print(f"★ опоры: прод {va.sum():.0f}, порог {vh.sum():.0f}"
      + (f", обученный выбор {vm.sum():.0f}" if vm is not None else ""))

grid = sorted({0.0, 0.5, 0.6, 2/3, 0.7, 0.75, 0.8, 5/6, 0.875, 0.9, 1.0} | set(np.round(np.unique(ct), 4)))
rows = []
for t in grid:
    take = ct >= t
    r = [t, float(np.where(take, vh, va).sum()), 100.0 * take.mean()]
    if vm is not None:
        r.append(float(np.where(take, vm, va).sum()))
    rows.append(r)

best_h = max(rows, key=lambda r: r[1])
print(f"\n★★ СВИП ПОРОГА (гейт пропускает лист ко второму пути при {a.feat} ≥ порога)")
print("| порог | гейт+ПОРОГ | декодер на | " + ("гейт+МОДЕЛЬ |" if vm is not None else ""))
show = [r for r in rows if r[0] in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.875, 0.9, 1.0)] + [best_h]
for r in sorted({tuple(x) for x in show}):
    mark = "  ← лучший" if r[1] == best_h[1] else ""
    print(f"| {r[0]:.4f} | {r[1]:.0f} | {r[2]:.1f}% | " + (f"{r[3]:.0f} |" if vm is not None else "") + mark)

# ★★ ПОЛКА: диапазон порогов, где результат в пределах `tol` кривых от лучшего. Это и есть ответ
#    на «не стоит ли число на счастливом пороге».
for tol in (5, 10):
    ok = [r[0] for r in rows if best_h[1] - r[1] <= tol and r[2] > 0]
    if ok:
        lo, hi = min(ok), max(ok)
        inside = [r for r in rows if lo <= r[0] <= hi and r[2] > 0]
        worst = min(r[1] for r in inside)
        print(f"★ ПОЛКА ±{tol} кривых: пороги {lo:.3f}…{hi:.3f} "
              f"(в ней {len(inside)} значений, худшее {worst:.0f} против лучшего {best_h[1]:.0f}, "
              f"декодер на {min(r[2] for r in inside):.1f}-{max(r[2] for r in inside):.1f}% листов)")
print("⚠ Свип показывает УСТОЙЧИВОСТЬ, но сам по себе он апостериорен: держанная оценка правила — "
      "во вложенном отборе `_pregate.py`, здесь её нет и быть не может.")
