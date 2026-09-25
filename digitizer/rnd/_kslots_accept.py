r"""_kslots_accept.py — ПРИГОВОР ПРАВКЕ «K ДЕКОДЕРА НЕ НИЖЕ ЧИСЛА СЛОТОВ» ПО КРИТЕРИЮ §6.204.

КРИТЕРИЙ ЗАДАН ДО ПРАВКИ (§6.204, 11.09), здесь только исполняется:
  ВЗЯТЬ, если безымянный прирост K − G ≥ +20 при p < 0.01 (парный перестановочный по листам,
  как §6.169) И именной счёт K − G не ниже −15. Сторона G обязана дать 1107 ± 15 (§6.188) —
  иначе прогон не воспроизвёл опору и приговор не выносится (код 3).
Код возврата: 0 = ПРИНЯТО (включать `rowdec_k_slots = True`), 2 = НЕ ПРИНЯТО (ручка остаётся
выключенной), 3 = опора не воспроизведена / прогон неполон.

★ Вызывается ДРАЙВЕРОМ после `_name_cost_prod.py --dump` обоих режимов (образец —
`_pregate_ab_run.ps1` → `_pregate_accept.py`).
★ Отдельно печатает, ОТКУДА пришёл прирост: с треков, где K_слотов > K_U1, или с прочих — §6.204
назвал это предпосылкой, которую A/B обязан проверить сам (§6.175: сумма не доказывает механизм).

  <ComfyUI>\python_embeded\python.exe _kslots_accept.py
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:\nds\output\taskS")
ap.add_argument("--g", default="percurve_kslots_G.pkl")
ap.add_argument("--k", default="percurve_kslots_K.pkl")
ap.add_argument("--g-mode", default="G", help="имя режима-опоры в дампе")
ap.add_argument("--k-mode", default="K", help="имя режима-кандидата в дампе (§6.207: K3 = порог 0.3, K2 = 0.2)")
ap.add_argument("--exp-g", type=int, default=1107, help="опора: нынешний прод, §6.188")
ap.add_argument("--exp-from", default="", help="взять ожидание опоры из дампа другого прогона (сумма по --sheets)")
ap.add_argument("--exp-mode", default="K", help="режим в --exp-from")
ap.add_argument("--tol", type=int, default=15)
ap.add_argument("--min-gain", type=int, default=20)
ap.add_argument("--max-p", type=float, default=0.01)
ap.add_argument("--min-named", type=int, default=-15)
# ★ §6.215: правка ИМЁН — ведущим становится именной счёт (≥ +min-gain при p < max-p), а безымянный —
#   вторым условием (≥ min-named). Опора по-прежнему сверяется по безымянному.
ap.add_argument("--lead", default="unnamed", choices=["unnamed", "named"])
ap.add_argument("--perm", type=int, default=200000)
ap.add_argument("--sheets", default="wellmap_sheets.txt", help="поле прогона: список листов (§6.166)")
ap.add_argument("--tol-sheets", type=int, default=15, help="допустимая недостача листов в ОБОИХ режимах")
a = ap.parse_args()
TS = Path(a.ts)

PG, PK = TS / a.g, TS / a.k
gone = [str(p) for p in (PG, PK) if not p.exists()]
if gone:
    print("⛔ ПРОГОН ЕЩЁ НЕ ЗАКОНЧЕН: нет " + "; ".join(gone))
    sys.exit(3)


def load(pkl, mode):
    d = pickle.load(open(pkl, "rb"))
    if mode not in d:
        sys.exit(f"⛔ в {pkl.name} нет режима {mode}: есть {sorted(d)}")
    return d[mode]


G, K = load(PG, a.g_mode), load(PK, a.k_mode)
common = sorted(set(G) & set(K))
print(f"★ ИСТОЧНИКИ: G {len(G)} листов, K {len(K)}, ОБЩИХ {len(common)}")
if not common:
    sys.exit(3)
if len(common) < max(len(G), len(K)):
    print(f"⚠⚠ НЕПОЛНОЕ ПЕРЕСЕЧЕНИЕ: только в G {len(set(G)-set(K))}, только в K {len(set(K)-set(G))}.")
# ★ ПОЛНОТА ПРОТИВ ПОЛЯ, А НЕ ПРОТИВ ДРУГ ДРУГА (§6.166; ревизия §6.205 нашла, что первая редакция
#   сравнивала только G∩K — пропажа ≤15 листов в обоих режимах прошла бы через допуск опоры молча).
FIELD = TS / a.sheets
# ★ §6.207: ожидание опоры — не число из головы, а сумма по полю из дампа предыдущего прогона той же
#   конфигурации (K §6.204 == G §6.207); иначе для подполя (723 гейтованных листа) ожидать нечего.
if a.exp_from:
    E = load(TS / a.exp_from, a.exp_mode)
    fs = {l.strip() for l in FIELD.read_text(encoding="utf-8").splitlines() if l.strip()} if FIELD.exists() else set(E)
    a.exp_g = int(sum(E[k][1] for k in E if k in fs))
    print(f"★ ОЖИДАНИЕ ОПОРЫ из {a.exp_from}[{a.exp_mode}] по {len([k for k in E if k in fs])} листам поля: {a.exp_g}")
if FIELD.exists():
    field = {l.strip() for l in FIELD.read_text(encoding="utf-8").splitlines() if l.strip()}
    common = [k for k in common if k in field]        # ★ считаем СТРОГО по полю (подполе §6.207)
    lost = sorted(field - set(common))
    print(f"★ ПОЛНОТА ПО ПОЛЮ: общих {len(common)} из {len(field)}; недостаёт {len(lost)}"
          + (f": {', '.join(x[:40] for x in lost[:6])}{' …' if len(lost) > 6 else ''}" if lost else ""))
    if len(lost) > a.tol_sheets:
        print(f"⛔ НЕДОСТАЁТ БОЛЬШЕ {a.tol_sheets} ЛИСТОВ — досчитать (`_fold_missing.py`, меньше шардов), "
              f"удалить percurve-дампы и повторить (код 3)")
        sys.exit(3)
else:
    print(f"⚠ списка поля {FIELD.name} нет — полнота проверена только пересечением G∩K")

gu = np.array([G[k][1] for k in common], float)   # безымянные — ВЕДУЩИЙ счёт
ku = np.array([K[k][1] for k in common], float)
gn = np.array([G[k][0] for k in common], float)   # именные — второе условие
kn = np.array([K[k][0] for k in common], float)


def perm(x, y, n):
    """Парный перестановочный: знак разницы на каждом листе меняем случайно (лист — единица
    независимости, кривые одного бланка делят его дефекты)."""
    d = y - x
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0, 0
    obs = abs(nz.sum())
    sg = np.random.default_rng(0).choice([-1.0, 1.0], size=(n, nz.size))
    return float((np.abs((sg * np.abs(nz)).sum(1)) >= obs - 1e-9).mean()), int(nz.size)


pu, nzu = perm(gu, ku, a.perm)
pn, nzn = perm(gn, kn, a.perm)
print(f"\n★★ ПРИЁМКА §6.204 (листов {len(common)}, перестановок {a.perm})")
print("| счёт | G (нынешний прод) | K (K ≥ слотов) | прирост | p | листов с разницей |")
print(f"| БЕЗЫМЯННЫЕ (ведущий) | {int(gu.sum())} | {int(ku.sum())} | {int(ku.sum()-gu.sum()):+d} | {pu:.5f} | {nzu} |")
print(f"| именные (2-е условие) | {int(gn.sum())} | {int(kn.sum())} | {int(kn.sum()-gn.sum()):+d} | {pn:.5f} | {nzn} |")
up = sum(1 for x, y in zip(gu, ku) if y > x); dn = sum(1 for x, y in zip(gu, ku) if y < x)
print(f"   листов ↑{up} / ↓{dn} (безымянные)")

ok_base = abs(int(gu.sum()) - a.exp_g) <= a.tol
gain = int(ku.sum() - gu.sum()); named = int(kn.sum() - gn.sum())
if a.lead == "named":
    ok_gain = named >= a.min_gain and pn < a.max_p
    ok_named = gain >= a.min_named
else:
    ok_gain = gain >= a.min_gain and pu < a.max_p
    ok_named = named >= a.min_named
print()
print(f"  {'✔' if ok_base else '✘'} опора G: {int(gu.sum())} против {a.exp_g} ± {a.tol}")
if a.lead == "named":
    print(f"  {'✔' if ok_gain else '✘'} ИМЕННОЙ прирост (ведущий, §6.215) {named:+d} ≥ +{a.min_gain} при p = {pn:.5f} < {a.max_p}")
    print(f"  {'✔' if ok_named else '✘'} безымянный (2-е условие) {gain:+d} ≥ {a.min_named}")
else:
    print(f"  {'✔' if ok_gain else '✘'} безымянный прирост {gain:+d} ≥ +{a.min_gain} при p = {pu:.5f} < {a.max_p}")
    print(f"  {'✔' if ok_named else '✘'} именной {named:+d} ≥ {a.min_named}")
if not ok_base:
    print("\n⛔ ОПОРА НЕ ВОСПРОИЗВЕДЕНА — приговор не выносится (код 3)")
    sys.exit(3)
if ok_gain and ok_named:
    print("\n★★★ ПРИНЯТО по критерию §6.204 — включать `rowdec_k_slots = True` в auto/config.py")
    sys.exit(0)
print("\n⛔ НЕ ПРИНЯТО по критерию §6.204 — ручка остаётся выключенной (код 2)")
sys.exit(2)
