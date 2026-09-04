r"""_pregate_model.py — ПРЕДГЕЙТ НА НЕСКОЛЬКИХ ПРИЗНАКАХ ВМЕСТО ОДНОГО ПОРОГА (§6.180).

ОТКУДА ВОПРОС. §6.179 разложил зазор гейта до его собственного потолка (1215 − 1109 = 106 кривых)
и показал, что держат его ДВЕ ПРОТИВОПОЛОЖНЫЕ ошибки одного порядка: «не пустил, а помогло бы»
(37 листов, −40) и «пустил, а навредило» (52 листа, −66). Сдвиг порога лечит одну ровно настолько,
насколько портит другую ⇒ **подбор порога исчерпан, и нужен ДРУГОЙ признак.** Один признак
`_pregate.py` уже перебрал исчерпывающе (60 штук, вложенный отбор) ⇒ остаётся их КОМБИНАЦИЯ.

ЧТО СЧИТАЕТ. Логистическая по листу на тех же 60 признаках, ДЕРЖАННО ПО СКВАЖИНАМ, вес строки —
|второй путь − прод| (лист, где пути равны, ничего не стоит и учить на нём нечему).

★★ САМОПРОВЕРКА ПЕРЕД ЛЮБЫМ ЧИСЛОМ. Признаки здесь считаются СВОИМ кодом, а сравниваться должны с
`_pregate.py`. Поэтому первым делом воспроизводится ОДНОПРИЗНАКОВЫЙ гейт `color_top_frac >= 0.80`:
он обязан дать ровно то, что даёт `_pregate_curve.py` (1109 на честном корпусе). Не сошлось —
признаки разъехались, и всё дальнейшее сравнивало бы разное. Это проверка, а не обещание.

⚠⚠ НОРМИРОВКА — ВНУТРИ ОБУЧАЮЩИХ ФОЛДОВ. В `_pick_learn.py` среднее и разброс берутся по ВСЕМ
строкам, и это утечка распределения (не меток, но в пользу модели). Здесь её нет: §6.161 уже
поймал ровно такую утечку у своей первой редакции в сетке порогов, и повторять её нельзя.

ТРИ КОНТРОЛЯ, ТЕ ЖЕ ЧТО В §6.161, И ПЕРВЫЙ ГЛАВНЫЙ:
  1. СОГЛАСОВАННЫЙ НУЛЬ — те же признаки, но связь «лист → исход» разорвана перестановкой. У
     модели на 60 признаках переобучение куда вероятнее, чем у одного порога, и виден он будет
     именно здесь: если нуль подтянется к результату, весь прирост — вместимость, а не сигнал.
  2. УСТОЙЧИВОСТЬ К РАЗБИЕНИЮ — то же на N случайных раскладках скважин.
  3. ЗНАКОВЫЙ ПЕРЕСТАНОВОЧНЫЙ по листам, где решение разошлось с «всегда оба пути».

  <ComfyUI>\python_embeded\python.exe _pregate_model.py
"""
import sys, argparse, json, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--prod", default=r"F:/nds/output/taskS/percurve_wellmap.pkl")
ap.add_argument("--prod-mode", default="A")
ap.add_argument("--both", nargs="+", default=[r"F:/nds/output/taskS/percurve_wellmap.pkl"])
ap.add_argument("--both-mode", default="H")
ap.add_argument("--dumps", nargs="+", default=[r"F:/nds/output/taskS/ab_wellmap/H"])
ap.add_argument("--wellmap", default=r"F:/nds/output/taskS/rowdec_wellmap.json")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--null", type=int, default=50)
ap.add_argument("--splits", type=int, default=200)
ap.add_argument("--l2", type=float, default=0.05)
ap.add_argument("--check-thr", type=float, default=0.80)
ap.add_argument("--check-want", type=float, default=1109.0)
a = ap.parse_args()

lead = lambda f, m: {k: v[1] for k, v in pickle.load(open(f, "rb"))[m].items()}
A = lead(a.prod, a.prod_mode)
D = {}
for f in a.both:
    D.update(lead(f, a.both_mode))
# ⚠ КОДИРОВКА ЯВНО: карта скважин — UTF-8 с 325 не-ASCII ключами; открытая по локали cp1251 она
# читается БЕЗ ошибки и молча теряет треть корпуса, меняя даже ЗНАК вывода (§6.161).
WM = json.loads(Path(a.wellmap).read_text(encoding="utf-8"))

NUM = ("x_center", "x_band", "thickness", "rough_n", "n_strokes", "n_runs_med", "row_cov")
CAT = ("color", "behavior", "confidence", "flag_reason")


def feats(u, pick):
    """★ ДОЛЖНА СОВПАДАТЬ С `_pregate.py.feats` ДО ЗНАКА. Совпадение не обещается, а проверяется
    ниже воспроизведением однопризнакового гейта."""
    f = {"n_prod": float(sum(x["n_prod"] for x in pick)),
         "cov_prod": float(sum(x["cov_prod"] for x in pick)),
         "n_tracks_pick": float(len(pick))}
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
for root in a.dumps:
    for dd in Path(root).iterdir():
        if not dd.is_dir():
            continue
        j = next(iter(dd.glob("*_pick.json")), None)
        u = next(iter(dd.glob("*_understanding.json")), None)
        if j is None or u is None:
            skip["нет дампа признаков"] += 1
            continue
        pk = json.loads(j.read_text(encoding="utf-8"))["tracks"]
        if not pk:
            skip["пустые треки"] += 1
            continue
        stem = u.name[:-len("_understanding.json")]
        F[stem + ".nlgx"] = (feats(json.loads(u.read_text(encoding="utf-8")), pk), WM.get(stem))

K = [k for k in sorted(set(A) & set(D) & set(F)) if F[k][1]]
print(f"★ листов со счётом, признаками и скважиной: {len(K)}; пропущено {dict(skip) or 'нет'}")
if len(K) < 200:
    print("⛔ мало листов — держанный разбор невозможен")
    sys.exit(1)

NAMES = sorted(F[K[0]][0])
X = np.array([[F[k][0].get(n, 0.0) for n in NAMES] for k in K], float)
pa = np.array([A[k] for k in K], float)
pd_ = np.array([D[k] for k in K], float)
wells = np.array([F[k][1] for k in K])
print(f"★ признаков {len(NAMES)} на {len(K)} листах, скважин {len(set(wells))}")
print(f"★ опоры: всегда прод {pa.sum():.0f} · всегда оба пути {pd_.sum():.0f} · "
      f"потолок решения по листу {np.maximum(pa, pd_).sum():.0f}")

# ── САМОПРОВЕРКА: однопризнаковый гейт на СВОИХ признаках обязан дать известное число ──────────
ci = NAMES.index("color_top_frac")
one = np.where(X[:, ci] >= a.check_thr, pd_, pa).sum()
ok = abs(one - a.check_want) < 0.5
print(f"\n★★ САМОПРОВЕРКА ПРИЗНАКОВ: гейт `color_top_frac >= {a.check_thr}` на своих признаках даёт "
      f"{one:.0f}, ожидалось {a.check_want:.0f}   {'★ СОШЛОСЬ' if ok else '⛔ РАЗОШЛОСЬ'}")
if not ok:
    print("⛔⛔ признаки разъехались с `_pregate.py` — сравнивать было бы разное, замер прерван")
    sys.exit(1)

uw = sorted(set(wells))
fold = {w: i % a.folds for i, w in enumerate(uw)}
FD = np.array([fold[w] for w in wells])


def fit(Xi, yi, wi, iters=500, lr=0.4, l2=None):
    l2 = a.l2 if l2 is None else l2
    b = np.zeros(Xi.shape[1] + 1)
    Z = np.c_[Xi, np.ones(len(Xi))]
    if wi.sum() == 0:
        return b
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
        b -= lr * (Z.T @ (wi * (p - yi)) / max(1e-9, wi.sum()) + l2 * b)
    return b


def run(y, w, fd):
    """Держанно по скважинам: нормировка, обучение и решение — всё внутри обучающих фолдов."""
    take = np.zeros(len(K), bool)
    for f in range(a.folds):
        tr, te = np.where(fd != f)[0], np.where(fd == f)[0]
        if not len(te) or not len(tr):
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        b = fit((X[tr] - mu) / sd, y[tr], w[tr])
        z = np.c_[(X[te] - mu) / sd, np.ones(len(te))] @ b
        take[te] = 1 / (1 + np.exp(-np.clip(z, -30, 30))) >= 0.5
    return take


y = (pd_ > pa).astype(float)
w = np.abs(pd_ - pa)
take = run(y, w, FD)
got = float(np.where(take, pd_, pa).sum())
print(f"\n★★★ МОДЕЛЬ ПО ЛИСТУ, держанно по скважинам: **{got:.0f}** "
      f"(второй путь на {100*take.mean():.1f}% листов)")
print(f"   против {pa.sum():.0f} у прода ({got-pa.sum():+.0f}), {pd_.sum():.0f} у «всегда оба» "
      f"({got-pd_.sum():+.0f}), {one:.0f} у ОДНОПРИЗНАКОВОГО гейта ({got-one:+.0f})")

# ── КОНТРОЛЬ 1: согласованный нуль ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(20260904)
nul = []
for _ in range(a.null):
    idx = rng.permutation(len(K))
    nul.append(float(np.where(run(y[idx], w[idx], FD), pd_, pa).sum()))
nul = np.array(nul)
sdv = nul.std() or 1e-9
print(f"\n★ КОНТРОЛЬ 1 — СОГЛАСОВАННЫЙ НУЛЬ ({a.null} перестановок исхода при тех же признаках):")
print(f"   {nul.mean():.0f} ± {sdv:.0f} (разброс {nul.min():.0f}-{nul.max():.0f})   "
      f"⇒ модель на {(got-nul.mean())/sdv:.1f} SD выше; нулевых не ниже модели: "
      f"{int((nul >= got).sum())} из {a.null}")

# ── КОНТРОЛЬ 2: устойчивость к раскладке скважин ───────────────────────────────────────────────
d2 = []
for _ in range(a.splits):
    perm = rng.permutation(len(uw))
    fmap = {w2: perm[i] % a.folds for i, w2 in enumerate(uw)}
    d2.append(float(np.where(run(y, w, np.array([fmap[x] for x in wells])), pd_, pa).sum()) - pd_.sum())
d2 = np.array(d2)
print(f"\n★ КОНТРОЛЬ 2 — {a.splits} случайных раскладок скважин:")
print(f"   Δ к «всегда оба» = {d2.mean():+.1f} ± {d2.std():.1f} "
      f"(разброс {d2.min():+.0f}..{d2.max():+.0f}); положителен в {int((d2 > 0).sum())} из {a.splits}")

# ── КОНТРОЛЬ 3: знаковый перестановочный ───────────────────────────────────────────────────────
dif = np.where(take, pd_, pa) - pd_
nz = dif[dif != 0]
up, dn = int((dif > 0).sum()), int((dif < 0).sum())
if len(nz):
    sg = rng.integers(0, 2, size=(100000, len(nz))) * 2 - 1
    p3 = float((np.abs((sg * np.abs(nz)).sum(1)) >= abs(dif.sum())).mean())
else:
    p3 = 1.0
print(f"\n★ КОНТРОЛЬ 3 — ЗНАКОВЫЙ ПЕРЕСТАНОВОЧНЫЙ: разошлась с «всегда оба» на {len(nz)} листах, "
      f"↑{up} / ↓{dn}, Δ = {dif.sum():+.0f}, p = {p3:.4f}")

# ── КОНТРОЛЬ 4: ПРЯМОЕ СРАВНЕНИЕ С ОДНИМ ПРИЗНАКОМ ────────────────────────────────────────────
# ⚠⚠ БЕЗ ЭТОГО ЧИСЛО НЕЧИТАЕМО. Первые три контроля говорят лишь, что модель лучше «всегда оба
# пути». А решается другое: стоит ли комбинация шестидесяти признаков того, что она даёт ПОВЕРХ
# одного порога. Сравнение ПАРНОЕ, по листам, и оно же отвечает на «не вместимость ли это».
one_v = np.where(X[:, ci] >= a.check_thr, pd_, pa)
mod_v = np.where(take, pd_, pa)
d4 = mod_v - one_v
nz4 = d4[d4 != 0]
u4, n4 = int((d4 > 0).sum()), int((d4 < 0).sum())
if len(nz4):
    sg4 = rng.integers(0, 2, size=(100000, len(nz4))) * 2 - 1
    p4 = float((np.abs((sg4 * np.abs(nz4)).sum(1)) >= abs(d4.sum())).mean())
else:
    p4 = 1.0
print(f"\n★★ КОНТРОЛЬ 4 — МОДЕЛЬ ПРОТИВ ОДНОГО ПРИЗНАКА (парно по листам): "
      f"разошлись на {len(nz4)} листах, ↑{u4} / ↓{n4}, Δ = {d4.sum():+.0f}, p = {p4:.4f}")

print("\n★★★ ЧТЕНИЕ")
c1 = (got - nul.mean()) / sdv >= 3 and (nul >= got).sum() == 0
c2 = (d2 > 0).sum() >= 0.95 * a.splits
c3 = p3 < 0.01
c4 = p4 < 0.05 and d4.sum() > 0
if c1 and c2 and c3 and c4:
    print(f"   Модель по листу даёт {got:.0f} против {one:.0f} у одного признака, проходит ВСЕ "
          f"КОНТРОЛИ и обгоняет один признак ЗНАЧИМО (p = {p4:.4f}) ⇒ комбинация берёт то, чего "
          f"порог не брал.")
else:
    fail = [n for n, c in (("согласованный нуль", c1), ("устойчивость к раскладке", c2),
                           ("знаковый", c3), ("ЗНАЧИМО лучше одного признака", c4)) if not c]
    print(f"   ⚠ НЕ ПРОЙДЕНО: {', '.join(fail)} ⇒ читать как «не доказано», а не как выигрыш.")
    print(f"   Одного признака хватало на {one:.0f}; модель на 60 признаках даёт {got:.0f}.")
