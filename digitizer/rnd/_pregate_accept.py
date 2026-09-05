r"""_pregate_accept.py — ПРИЁМКА ПРОД-ПРАВКИ ПРЕДГЕЙТА: ДВА ЧИСЛА ИЛИ ОТКАТ (§6.184, §6.186).

ОТКУДА КРИТЕРИЙ. Заказчик задал его дословно: «прогнать A/B и сверить с двумя числами: 1109 без
имени и 732 с именем. Не увидели — откатывать». Этот стенд — единственное место, где приговор
выносится: сверять глазами по логу нельзя, потому что глаз считает то, что хочет увидеть.

★★ ПОЧЕМУ ЧИСЕЛ ДВА, А НЕ ОДНО. Ведущий счёт — БЕЗЫМЯННЫЙ (элемент [1] кортежа
`(named, UNNAMED, expert)`), именной справочный. Но §6.143 требует снимать оба в один заход:
правка, которая поднимает безымянные кривые и роняет именные, — это не победа, а перекладывание.
Поэтому именной счёт здесь не «до кучи», а ВТОРОЕ УСЛОВИЕ приёмки.

★ ЧТО ИМЕННО ПРИНИМАЕТСЯ. Режим G прогона задаёт `rowdec_pregate=0.8` через ПРОД-КОНФИГ, а решение
принимает `trace2d._pregate_ok` — прод-код, а не симуляция стенда. Режим A присваивает
`row_decoder = ""` БЕЗУСЛОВНО, поэтому включённый в проде декодер базу не отравляет: A остаётся
прежним продом. Обе предпосылки проверены 05.09 чтением `_trace_prod_ab.py:448-453`.

⚠ ЧЕГО СТЕНД НЕ ДЕЛАЕТ: не считает ни одного листа. Всё из готовых дампов `percurve_*`.

  <ComfyUI>\python_embeded\python.exe _pregate_accept.py
  … --exp-unnamed 1109 --exp-named 732     # ожидания можно переопределить
  … --tol 15                               # допуск на «то же число» (см. ниже, почему он есть)
"""
import sys, pickle, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:\nds\output\taskS")
ap.add_argument("--a", default="percurve_pregate_A.pkl")
ap.add_argument("--g", default="percurve_pregate_G.pkl")
ap.add_argument("--base-unnamed", type=int, default=965, help="прод без декодера, безымянные")
ap.add_argument("--base-named", type=int, default=661, help="прод без декодера, именные")
ap.add_argument("--exp-unnamed", type=int, default=1109)
ap.add_argument("--exp-named", type=int, default=732)
ap.add_argument("--tol", type=int, default=15)
ap.add_argument("--perm", type=int, default=200000)
a = ap.parse_args()
TS = Path(a.ts)

PA, PG = TS / a.a, TS / a.g
gone = [str(p) for p in (PA, PG) if not p.exists()]
if gone:
    sys.exit("⛔ ПРОГОН ЕЩЁ НЕ ЗАКОНЧЕН: нет " + "; ".join(gone) +
             "\n   Ход замера: nds")


def load(pkl, mode):
    d = pickle.load(open(pkl, "rb"))
    if mode not in d:
        sys.exit(f"⛔ в {pkl.name} нет режима {mode}: есть {sorted(d)}")
    return d[mode]


A, G = load(PA, "A"), load(PG, "G")
common = sorted(set(A) & set(G))
print(f"★ ИСТОЧНИКИ: A {len(A)} листов, G {len(G)}, ОБЩИХ {len(common)}")
if not common:
    sys.exit("⛔ пересечение пусто — сверять нечего")
# ⚠ §6.166: неполнота — это не «почти то же самое». Считаем ТОЛЬКО по общим листам и говорим,
#   сколько выпало: разница в 30 листов сама по себе стоит больше, чем весь ожидаемый прирост.
if len(common) < max(len(A), len(G)):
    print(f"⚠⚠ НЕПОЛНОЕ ПЕРЕСЕЧЕНИЕ: только в A {len(set(A)-set(G))}, только в G {len(set(G)-set(A))}."
          f" Приговор выносится по общим {len(common)} — числа НЕ сравнимы с ожиданием напрямую,"
          f" если выпало больше {a.tol} листов.")

au = np.array([A[k][1] for k in common], float)   # [1] — безымянные, ВЕДУЩИЙ счёт
gu = np.array([G[k][1] for k in common], float)
an = np.array([A[k][0] for k in common], float)   # [0] — именные, второе условие (§6.143)
gn = np.array([G[k][0] for k in common], float)


def perm(x, y, n):
    """Парный перестановочный: знак разницы на каждом листе меняем случайно. Лист — единица
    независимости, кривая — нет (кривые одного бланка делят его дефекты)."""
    d = y - x
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0, 0
    obs = abs(nz.sum())
    sg = np.random.default_rng(0).choice([-1.0, 1.0], size=(n, nz.size))
    return float((np.abs((sg * np.abs(nz)).sum(1)) >= obs - 1e-9).mean()), int(nz.size)


rows = []
for tag, x, y, base, exp in (("БЕЗЫМЯННЫЕ (ведущий)", au, gu, a.base_unnamed, a.exp_unnamed),
                             ("именные (2-е условие)", an, gn, a.base_named, a.exp_named)):
    p, nz = perm(x, y, a.perm)
    rows.append((tag, int(x.sum()), int(y.sum()), base, exp, p, nz))

print(f"\n★★ ПРИЁМКА ПРЕДГЕЙТА (листов {len(common)}, перестановок {a.perm})")
print("| счёт | прод A | предгейт G | прирост | ожидалось | p | листов с разницей |")
for tag, xs, ys, base, exp, p, nz in rows:
    print(f"| {tag} | {xs} | {ys} | {ys-xs:+d} | {exp} (было {base}) | {p:.5f} | {nz} |")

# ─────────────────────────────────────────────────────────────────────── ПРИГОВОР
# ★ ДОПУСК ЕСТЬ, И ВОТ ПОЧЕМУ. Ожидание 1109/732 снято на том же поле тем же кодом, но прогон
#   заново трассирует листы, а трассировка зависит от порядка и от того, какой чекпойнт подобрал
#   `auto5`. Требовать побайтового совпадения счёта — значит откатывать здоровую правку из-за
#   единиц. Но допуск односторонне мягким быть не должен: ПАДЕНИЕ ниже ожидания на тот же допуск
#   — тоже «не увидели».
u_obs, n_obs = rows[0][2], rows[1][2]
ok_u = abs(u_obs - a.exp_unnamed) <= a.tol
ok_n = abs(n_obs - a.exp_named) <= a.tol
ok_up = rows[0][2] > rows[0][1] and rows[0][5] < 0.05
print()
for name, ok, obs, exp in (("безымянные", ok_u, u_obs, a.exp_unnamed),
                           ("именные", ok_n, n_obs, a.exp_named)):
    print(f"  {'✔' if ok else '✘'} {name}: {obs} против ожидания {exp} "
          f"(отклонение {obs-exp:+d}, допуск ±{a.tol})")
print(f"  {'✔' if ok_up else '✘'} прирост безымянных значим: {rows[0][2]-rows[0][1]:+d}, p={rows[0][5]:.5f}")

if ok_u and ok_n and ok_up:
    print("\n★★★ ПРИНЯТО. Прод-правка воспроизвела оба числа — оставляем как есть.")
    print("   Записать в §6.184 подтверждение с этими числами.")
else:
    print("\n⛔⛔ НЕ ПРИНЯТО — ОТКАТ, как условлено заранее ('не увидели — откатывать').")
    print("   В `auto/config.py`: row_decoder = \"\"  ⇒ прод возвращается к прежнему ведению.")
    print("   ⚠ Откатывать ИМЕННО так, а не гасить предгейт: rowdec_pregate = 0.0 оставит декодер")
    print("     включённым на КАЖДОМ листе — это не прежний прод, а самый дорогой из режимов.")
    sys.exit(2)
