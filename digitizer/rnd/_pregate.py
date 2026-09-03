r"""_pregate.py — ПРЕДГЕЙТ ПО ЛИСТУ: НУЖЕН ЛИ ВТОРОЙ ПУТЬ НА ЭТОМ ЛИСТЕ (§6.161, переписан 01.09).

ОТКУДА ВОПРОС. §6.160 показал по исходнику, что цена второго пути порогом `rowdec_pick` НЕ
регулируется: декодер считается безусловно на каждом листе (`trace2d.py:145`), а порог только
выбирает результат по треку. ⇒ Удешевить может лишь гейт ПО ЛИСТУ, ДО вызова декодера.

⛔⛔ ПЕРВАЯ РЕДАКЦИЯ ОТВЕТИЛА «НЕ РАБОТАЕТ» — И ЭТО БЫЛ ОТВЕТ ПРО ЕЁ СОБСТВЕННЫЙ СПИСОК ПРИЗНАКОВ,
А НЕ ПРО ИДЕЮ. Она читала только `<лист>_pick.json`, где прод-только величин ровно три (`n_prod`,
`cov_prod`, число треков). В ТОМ ЖЕ каталоге лежит `<лист>_understanding.json` — тоже прод-только
и доступный РАНЬШЕ: `pipeline.py:46-52` фиксирует `sheet` (understand → confidence.classify) ДО
вызова `trace2d.trace_auto`, `emit.py:32` пишет `sheet.to_dict()`, а не трассы; ни `trace2d.py`,
ни `rowdec.py` в `sheet` не пишут ни строки. Проверено ПОБАЙТОВО: файл совпадает у прогона «прод
один путь» и «оба пути» на всех сверенных листах. ⇒ признаки оттуда не требуют даже перестановки
порядка в `trace_auto`, которую первая редакция сама себе вменяла как условие.

⚠⚠ И У ПЕРВОЙ РЕДАКЦИИ БЫЛА УТЕЧКА В СЕТКЕ ПОРОГОВ: кандидаты брались `np.percentile` по ВСЕМ
листам, а не по обучающим. Это утечка распределения, не меток, но она в пользу гейта. Здесь
сетка строится ТОЛЬКО на обучающих фолдах.

ЧТО СЧИТАЕТ. По замороженным выдачам, без нового счёта: прод (`ab_rowdec_pair/A`) против «оба
пути» (`ab_rdpick_c/D`), счёт ВЕДУЩИЙ безымянный (`_name_cost_prod.py --dump`). Признаки —
механически из `_understanding.json` плюс три прод-величины из `_pick.json`. Признак, направление
И порог выбираются ВНУТРИ обучающих фолдов (выбор признака по зачёту — это §6.114 на уровень
выше), зачёт — на держанной пятой ПО СКВАЖИНАМ.

ТРИ КОНТРОЛЯ, И ПЕРВЫЙ ГЛАВНЫЙ:
  1. СОГЛАСОВАННЫЙ НУЛЬ — те же признаки, но связь «лист → исход» разорвана перестановкой.
     Сохраняет распределения и связки признаков, рвёт только проверяемое. Случайные равномерные
     признаки для этого НЕ ГОДЯТСЯ: у них больше различных порогов, они переобучаются сильнее,
     нуль проседает ниже «всегда оба», и SD над нулём завышается.
  2. УСТОЙЧИВОСТЬ К РАЗБИЕНИЮ — то же на N случайных раскладках скважин. Одна раскладка `i % 5` —
     рядовая выборка из распределения шириной в десяток кривых.
  3. ЗНАКОВЫЙ ПЕРЕСТАНОВОЧНЫЙ ТЕСТ по листам, где гейт разошёлся с «всегда оба».

⚠⚠ СТАТУС ЧИСЛА: ОЦЕНКА ПО ЗАМОРОЖЕННЫМ ВЫДАЧАМ, а не прогон (тот же статус, что у §6.144, и там
он на отгрузке менялся). Корпус `ab_rdpick_c` УТЁКШИЙ (§6.153): половина листов знакома чекпойнту
декодера. Сравнение вариантов внутри стенда равнообъёмное и равнопутевое, абсолютные числа — не
отгрузочные.
⚠ КОДИРОВКА: `rowdec_wellmap.json` — UTF-8, и 325 из 2736 ключей не-ASCII при локали cp1251.
Открытый без `encoding="utf-8"`, он читается БЕЗ ошибки, но скважина не находится и разбор молча
идёт по 990 листам вместо 1121 — с другим составом и ДРУГИМ ЗНАКОМ вывода.

  <ComfyUI>\python_embeded\python.exe _pregate.py
"""
import sys, argparse, json, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--prod", default=r"F:/nds/output/taskS/percurve_pair.pkl")
ap.add_argument("--prod-mode", default="A")
ap.add_argument("--both", nargs="+", default=[r"F:/nds/output/taskS/percurve_rule.pkl"],
                help="дампы второго пути; их НЕСКОЛЬКО, когда прогон шёл по фолдам — склеиваются по листам")
ap.add_argument("--both-mode", default="D")
ap.add_argument("--dumps", nargs="+", default=[r"F:/nds/output/taskS/ab_rdpick_c/D"],
                help="каталоги выдачи с `_pick.json`/`_understanding.json`; их НЕСКОЛЬКО, когда прогон шёл по фолдам")
ap.add_argument("--wellmap", default=r"F:/nds/output/taskS/rowdec_wellmap.json")
ap.add_argument("--roots", nargs="+", default=[r"F:\nds\projects\Archive"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--null", type=int, default=50, help="перестановок согласованного нуля")
ap.add_argument("--splits", type=int, default=200, help="случайных раскладок скважин")
ap.add_argument("--grid", type=int, default=2, help="шаг перцентилей сетки порогов")
ap.add_argument("--only-pick", action="store_true",
                help="только три прод-величины из _pick.json — воспроизводит ПЕРВУЮ редакцию")
a = ap.parse_args()

# ⚠ Оговорка про утечку берётся из ФАКТИЧЕСКОГО каталога дампов, а не пишется рукой:
# `ab_wellmap` снят с картой скважин (честная держанность, §6.157), `ab_rdpick_c` — без неё.
_DP = " ".join(a.dumps).replace("\\", "/")
LEAK = ("корпус ЧЕСТНО ДЕРЖАННЫЙ, wellmap" if "wellmap" in _DP or "honest" in _DP
        else "корпус УТЁКШИЙ: половина листов знакома чекпойнту декодера, §6.153")
print(f"★ корпус дампов: {', '.join(a.dumps)}  ⇒ {LEAK}")

lead = lambda f, m: {k: v[1] for k, v in pickle.load(open(f, "rb"))[m].items()}
A = lead(a.prod, a.prod_mode)
# ★ несколько дампов склеиваются СЛОВАРЁМ, и пересечение ключей — ОШИБКА, а не мелочь: фолды
# не пересекаются по листам по построению, и задвоение значило бы, что склеено не то.
D = {}
for _f in a.both:
    _d = lead(_f, a.both_mode)
    _dup = set(_d) & set(D)
    if _dup:
        sys.exit(f"⛔ дампы второго пути пересекаются на {len(_dup)} листах ({_f}) — склеено не то")
    D.update(_d)

WM = json.load(open(a.wellmap, encoding="utf-8"))
na = sum(1 for k in WM if not k.isascii())
print(f"★ карта скважин: {len(WM)} ключей, НЕ-ASCII {na} ({100*na/len(WM):.0f}%) — "
      f"читается только с encoding='utf-8'")

BY = {}
for r in a.roots:
    for wlg in Path(r).glob("*/wlg"):
        for q in wlg.glob("*.nlgx"):
            BY[f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"] = (q.name, q.stem)

NUM = ("x_center", "x_band", "thickness", "rough_n", "n_strokes", "n_runs_med", "row_cov")
CAT = ("color", "behavior", "confidence", "flag_reason")


def feats(u, pick):
    """Признаки ЛИСТА строятся механически: агрегаты по линиям `_understanding.json` (прод-только,
    зафиксирован ДО ведения) плюс три прод-величины из `_pick.json`. Ни один не смотрит в эталон
    и ни один не требует декодера."""
    f = {"n_prod": float(sum(x["n_prod"] for x in pick)),
         "cov_prod": float(sum(x["cov_prod"] for x in pick)),
         "n_tracks_pick": float(len(pick))}
    if a.only_pick:
        return f
    L = u.get("lines") or []
    f["n_lines"] = float(len(L))
    f["n_tracks_frame"] = float((u.get("diag") or {}).get("n_tracks") or 0)
    f["n_lines_total"] = float(u.get("n_lines_total") or 0)
    ptc = [float(v) for v in (u.get("per_track_line_count") or {}).values()] or [0.0]
    f["ptc_max"], f["ptc_mean"] = max(ptc), float(np.mean(ptc))
    f["n_expected"] = float(len(u.get("expected_curves") or []))
    fr = u.get("frame") or {}
    for k in ("top_y", "bottom_y", "px_per_m", "grid_period_px"):
        try:
            f["fr_" + k] = float(fr.get(k) or 0)
        except (TypeError, ValueError):
            f["fr_" + k] = 0.0
    f["fr_height"] = f["fr_bottom_y"] - f["fr_top_y"]
    for k in NUM:
        v = [float(x[k]) for x in L if isinstance(x.get(k), (int, float))]
        if v:
            f[k + "_med"], f[k + "_max"] = float(np.median(v)), float(max(v))
            f[k + "_min"], f[k + "_sum"] = float(min(v)), float(sum(v))
            f[k + "_spread"] = float(max(v) - min(v))
        else:
            for s in ("_med", "_max", "_min", "_sum", "_spread"):
                f[k + s] = 0.0
    for k in CAT:
        c = Counter(str(x.get(k)) for x in L)
        f["n_" + k + "s"] = float(len(c))
        f[k + "_top_frac"] = float(max(c.values()) / len(L)) if L else 0.0
    f["frac_flagged"] = float(sum(1 for x in L if x.get("flag_reason")) / len(L)) if L else 0.0
    f["frac_auto"] = float(sum(1 for x in L if x.get("confidence") == "AUTO") / len(L)) if L else 0.0
    f["frac_bunched"] = float(sum(1 for x in L if x.get("flag_reason") == "bunched_crossing")
                              / len(L)) if L else 0.0
    return f


F, skip = {}, Counter()
for dd in [q for root in a.dumps for q in Path(root).iterdir()]:
    if not dd.is_dir():
        continue
    if dd.name not in BY:
        skip["каталог не узнан"] += 1
        continue
    j = next(iter(dd.glob("*_pick.json")), None)
    u = next(iter(dd.glob("*_understanding.json")), None)
    if j is None or u is None:
        skip["нет дампа признаков"] += 1
        continue
    pk = json.load(open(j, encoding="utf-8"))["tracks"]
    if not pk:
        skip["пустые треки"] += 1
        continue
    nm, stem = BY[dd.name]
    F[nm] = (feats(json.load(open(u, encoding="utf-8")), pk), WM.get(stem))

K = [k for k in sorted(set(A) & set(D) & set(F)) if F[k][1]]
print(f"★ СВЕРКА ОБЪЁМА: разобрано {len(F)}, пропущено {sum(skip.values())} "
      f"({dict(skip) if skip else 'нет'}); со счётом и скважиной {len(K)}")
if len(K) < 100:
    sys.exit("⛔ мало листов — держанный разбор невозможен")

pa = np.array([A[k] for k in K], float)
pdd = np.array([D[k] for k in K], float)
w = np.array([F[k][1] for k in K])
NAMES = sorted(F[K[0]][0])
X = np.array([[F[k][0].get(n, 0.0) for n in NAMES] for k in K], float)
print(f"★ признаков {len(NAMES)} на {len(K)} листах, скважин {len(set(w))}"
      f"{'   (режим --only-pick: ПЕРВАЯ редакция)' if a.only_pick else ''}")
up, dn = pdd > pa, pdd < pa
print(f"\n★ ЧТО ДАЁТ ВТОРОЙ ПУТЬ: лучше на {up.sum()} листах (+{pdd[up].sum()-pa[up].sum():.0f}), "
      f"ХУЖЕ на {dn.sum()} (−{pa[dn].sum()-pdd[dn].sum():.0f}); всегда прод {pa.sum():.0f}, "
      f"всегда оба {pdd.sum():.0f}, оракул по листу {np.maximum(pa,pdd).sum():.0f}")


def nested(Xm, y_a, y_d, fw, ret_sel=False):
    """Признак, направление, порог И СЕТКА порогов — только на обучающих фолдах."""
    tot, gat, chosen = 0.0, 0, []
    sel = np.zeros(len(y_a), bool)
    for f in range(a.folds):
        tr, te = fw != f, fw == f
        if not tr.any() or not te.any():
            continue
        best = (-1e18, None)
        for c in range(Xm.shape[1]):
            for sgn in (1.0, -1.0):
                v = sgn * Xm[:, c]
                for t in np.unique(np.percentile(v[tr], np.arange(0, 101, a.grid))):
                    s = np.where(v[tr] <= t, y_d[tr], y_a[tr]).sum()
                    if s > best[0]:
                        best = (s, (c, sgn, t))
        c, sgn, t = best[1]
        m = (sgn * Xm[:, c])[te] <= t
        sel[te] = m
        tot += float(np.where(m, y_d[te], y_a[te]).sum())
        gat += int(m.sum())
        chosen.append(f"{NAMES[c]}{'<=' if sgn > 0 else '>='}{abs(t):.4g}")
    return (tot, gat, chosen, sel) if ret_sel else (tot, gat, chosen)


wells = sorted(set(w))
fold = {x: i % a.folds for i, x in enumerate(wells)}
fw = np.array([fold[x] for x in w])
tot, gat, chosen, gsel = nested(X, pa, pdd, fw, ret_sel=True)
print(f"\n★★ ВЛОЖЕННЫЙ ОТБОР (признак+направление+порог И СЕТКА — на обучающих фолдах)")
print(f"   всегда прод {pa.sum():.0f} · всегда оба пути {pdd.sum():.0f} · ★ ГЕЙТ {tot:.0f}"
      f"   (Δ = {tot-pdd.sum():+.0f}, декодер на {100*gat/len(K):.1f}% листов)")
print(f"   выбрано по фолдам: {chosen}")

rng = np.random.default_rng(20260901)
nulls = np.array([nested(X, pa[o], pdd[o], fw)[0]
                  for o in (rng.permutation(len(K)) for _ in range(a.null))])
z = (tot - nulls.mean()) / max(1e-9, nulls.std())
print(f"\n★ КОНТРОЛЬ 1 — СОГЛАСОВАННЫЙ НУЛЬ ({a.null} перестановок исхода при тех же признаках):")
print(f"   {nulls.mean():.0f} ± {nulls.std():.0f} (разброс {nulls.min():.0f}-{nulls.max():.0f})"
      f"   ⇒ гейт на {z:.1f} SD выше; нулевых прогонов не ниже гейта: {int((nulls >= tot).sum())} из {a.null}")

ds = []
for _ in range(a.splits):
    perm = rng.permutation(len(wells))
    fm = {wells[perm[i]]: i % a.folds for i in range(len(wells))}
    ds.append(nested(X, pa, pdd, np.array([fm[x] for x in w]))[0] - pdd.sum())
ds = np.array(ds)
print(f"\n★ КОНТРОЛЬ 2 — {a.splits} случайных раскладок скважин по фолдам:")
print(f"   Δ к «всегда оба» = {ds.mean():+.1f} ± {ds.std():.1f} "
      f"(разброс {ds.min():+.0f}..{ds.max():+.0f}); положителен в {int((ds > 0).sum())} из {a.splits}")

diff = np.where(gsel, pdd, pa) - pdd
nz = diff[diff != 0]
obs = float(diff.sum())
sg = rng.integers(0, 2, size=(20000, len(nz))) * 2 - 1 if len(nz) else None
p = float((np.abs((sg * np.abs(nz)).sum(1)) >= abs(obs)).mean()) if len(nz) else 1.0
print(f"\n★ КОНТРОЛЬ 3 — ЗНАКОВЫЙ ПЕРЕСТАНОВОЧНЫЙ по листам: гейт разошёлся с «всегда оба» на "
      f"{len(nz)} листах, ↑{int((diff>0).sum())} / ↓{int((diff<0).sum())}, Δ = {obs:+.0f}, p = {p:.4f}")

print(f"\n★★★ ЧТЕНИЕ")
ok = tot > pdd.sum() and z >= 3 and (ds > 0).mean() > 0.9 and p < 0.05
if ok:
    print(f"   Гейт даёт {tot:.0f} против {pdd.sum():.0f} у «всегда оба» ({tot-pdd.sum():+.0f}) и проходит")
    print(f"   ВСЕ ТРИ контроля ⇒ долистовый предгейт на ПРОД-признаках работает по КАЧЕСТВУ.")
    print(f"   ⚠ Но экономит мало: декодер на {100*gat/len(K):.1f}% листов вместо 100%.")
    print(f"   ⚠ И это оценка по ЗАМОРОЖЕННЫМ выдачам ({LEAK}) — не отгрузочное число.")
    print(f"   ⚠ Конкретное выбранное правило — АПОСТЕРИОРНОЕ чтение; честно измерена ПРОЦЕДУРА.")
else:
    print(f"   Δ = {tot-pdd.sum():+.0f}, {z:.1f} SD над согласованным нулём, положителен в "
          f"{100*(ds>0).mean():.0f}% раскладок, p = {p:.4f}")
    print(f"   ⇒ хотя бы один контроль не пройден: читать как «не доказано», а не как выигрыш.")
