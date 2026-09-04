r"""_pick_sheetfeat.py — ДАТЬ ПОТРЕКОВОМУ ВЫБОРУ КОНТЕКСТ ЛИСТА (§6.181, ЭКРАН, не замер).

ОТКУДА ВОПРОС. §6.180 закрыл полистное направление: комбинация всех 60 признаков ЛИСТА не обгоняет
один порог (+14, p = 0.13). Оставшееся указание §6.179 — ПОТРЕКОВЫЕ сведения: потолок по треку
(1257) выше потолка по листу (1215), а смешанный путь даёт кривые, которых нет ни у одного чистого.
⇒ Самая дешёвая проверка этого указания: потрековая модель §6.177 знает восемь признаков ТРЕКА и
ничего не знает о листе. Дать ей те же 60 признаков листа — и посмотреть, есть ли сигнал.

⚠⚠ ЭТО ЭКРАН, А НЕ ЗАМЕР, И ВОТ ПОЧЕМУ ЭТО ЧЕСТНО СКАЗАТЬ. Счёт здесь СКЛАДЫВАЕТСЯ из потрековых
величин, а §6.175 такую композицию ОТКАЗАЛ для полистных выводов: она разошлась с реальным прогоном
на 69 листах из 1123. Но там решалась судьба ПРЕДГЕЙТА, которому нужны верные ПОЛИСТНЫЕ числа, —
а здесь сравниваются ИТОГИ двух потрековых моделей на одном и том же наборе треков, и по итогу
композиция ошибалась на 3 кривые из 1084 (§6.177: композиция 1084, прогон 1081).
⇒ Годится решить «есть ли сигнал», НЕ годится назвать цену. Прогон стоит ~5 ч и запускается
ТОЛЬКО если экран показал заметный отрыв.

★ ОПОРА — ТЕКУЩЕЕ ЛУЧШЕЕ, А НЕ СЛАБОЕ (урок §6.180). Сравнение идёт с той же моделью на восьми
признаках трека, обученной ТЕМ ЖЕ кодом на тех же фолдах; отдельно печатается парный тест.

  <ComfyUI>\python_embeded\python.exe _pick_sheetfeat.py
"""
import sys, argparse, json, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from pathlib import Path
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default=r"F:/nds/output/taskS/pick_learn_honest2.pkl.dump")
ap.add_argument("--dumps", nargs="+", default=[r"F:/nds/output/taskS/ab_wellmap/H"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--l2", type=float, default=0.02)
ap.add_argument("--perm", type=int, default=100000)
a = ap.parse_args()

rows = pickle.load(open(a.rows, "rb"))
TF = sorted(rows[0][3])
print(f"★ треков {len(rows)}, признаков ТРЕКА {len(TF)}: {', '.join(TF)}")

NUM = ("x_center", "x_band", "thickness", "rough_n", "n_strokes", "n_runs_med", "row_cov")
CAT = ("color", "behavior", "confidence", "flag_reason")


def sheet_feats(u, pick):
    """★ ТОТ ЖЕ набор, что в `_pregate.py` и `_pregate_model.py` — иначе сравнивалось бы разное."""
    f = {"s_n_prod": float(sum(x["n_prod"] for x in pick)),
         "s_cov_prod": float(sum(x["cov_prod"] for x in pick)),
         "s_n_tracks_pick": float(len(pick))}
    L = u.get("lines") or []
    f["s_n_lines"] = float(len(L))
    f["s_n_tracks_frame"] = float((u.get("diag") or {}).get("n_tracks") or 0)
    f["s_n_lines_total"] = float(u.get("n_lines_total") or 0)
    ptc = [float(v) for v in (u.get("per_track_line_count") or {}).values()] or [0.0]
    f["s_ptc_max"], f["s_ptc_mean"] = max(ptc), float(np.mean(ptc))
    f["s_n_expected"] = float(len(u.get("expected_curves") or []))
    fr = u.get("frame") or {}
    for k in ("top_y", "bottom_y", "px_per_m", "grid_period_px"):
        try:
            f["s_fr_" + k] = float(fr.get(k) or 0)
        except (TypeError, ValueError):
            f["s_fr_" + k] = 0.0
    f["s_fr_height"] = f["s_fr_bottom_y"] - f["s_fr_top_y"]
    for k in NUM:
        v = [float(x[k]) for x in L if isinstance(x.get(k), (int, float))]
        if v:
            f["s_" + k + "_med"], f["s_" + k + "_max"] = float(np.median(v)), float(max(v))
            f["s_" + k + "_min"], f["s_" + k + "_sum"] = float(min(v)), float(sum(v))
            f["s_" + k + "_spread"] = float(max(v) - min(v))
        else:
            for s2 in ("_med", "_max", "_min", "_sum", "_spread"):
                f["s_" + k + s2] = 0.0
    for k in CAT:
        c = Counter(str(x.get(k)) for x in L)
        f["s_n_" + k + "s"] = float(len(c))
        f["s_" + k + "_top_frac"] = float(max(c.values()) / len(L)) if L else 0.0
    f["s_frac_flagged"] = float(sum(1 for x in L if x.get("flag_reason")) / len(L)) if L else 0.0
    f["s_frac_auto"] = float(sum(1 for x in L if x.get("confidence") == "AUTO") / len(L)) if L else 0.0
    f["s_frac_bunched"] = float(sum(1 for x in L if x.get("flag_reason") == "bunched_crossing")
                                / len(L)) if L else 0.0
    return f


SF = {}
for root in a.dumps:
    for dd in Path(root).iterdir():
        if not dd.is_dir():
            continue
        j = next(iter(dd.glob("*_pick.json")), None)
        u = next(iter(dd.glob("*_understanding.json")), None)
        if j is None or u is None:
            continue
        pk = json.loads(j.read_text(encoding="utf-8"))["tracks"]
        if not pk:
            continue
        stem = u.name[:-len("_understanding.json")]
        SF[stem + ".nlgx"] = sheet_feats(json.loads(u.read_text(encoding="utf-8")), pk)

keep = [r for r in rows if r[0] in SF]
if len(keep) < 0.95 * len(rows):
    print(f"⛔ признаки листа нашлись только у {len(keep)} треков из {len(rows)} — замер прерван")
    sys.exit(1)
SN = sorted(next(iter(SF.values())))
print(f"★ признаков ЛИСТА {len(SN)}; треков с обоими наборами {len(keep)} из {len(rows)}")

Xt = np.array([[r[3][k] for k in TF] for r in keep], float)
Xs = np.array([[SF[r[0]].get(k, 0.0) for k in SN] for r in keep], float)
hA = np.array([r[4] for r in keep], float)
hB = np.array([r[5] for r in keep], float)
wells = sorted({r[1] for r in keep})
fold = {w: i % a.folds for i, w in enumerate(wells)}
FD = np.array([fold[r[1]] for r in keep])
y = (hB > hA).astype(float)
w = np.abs(hB - hA)
print(f"★ опоры по трекам: прод {hA.sum():.0f}, декодер {hB.sum():.0f}, "
      f"оракул {np.maximum(hA, hB).sum():.0f}")


def fit(Z, idx, iters=500, lr=0.4):
    b = np.zeros(Z.shape[1] + 1)
    Zi = np.c_[Z[idx], np.ones(len(idx))]
    yi, wi = y[idx], w[idx]
    if wi.sum() == 0:
        return b
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Zi @ b, -30, 30)))
        b -= lr * (Zi.T @ (wi * (p - yi)) / max(1e-9, wi.sum()) + a.l2 * b)
    return b


def run(X, leak_norm=False):
    """Держанно по скважинам; нормировка ВНУТРИ обучающих фолдов (без утечки распределения).

    ⚠⚠ `leak_norm=True` повторяет `_pick_learn.py`: среднее и разброс берутся по ВСЕМ строкам,
    включая держанные. Это утечка РАСПРЕДЕЛЕНИЯ (не меток) — и она уехала в проде: выгруженные
    веса `pick_model_h_f*.npz` несут `mu`/`sd`, посчитанные по всему набору. Опция нужна, чтобы
    ИЗМЕРИТЬ её цену, а не рассуждать о ней."""
    take = np.zeros(len(keep), bool)
    gmu, gsd = X.mean(0), X.std(0) + 1e-9
    for f in range(a.folds):
        tr, te = np.where(FD != f)[0], np.where(FD == f)[0]
        if not len(te) or not len(tr):
            continue
        mu, sd = (gmu, gsd) if leak_norm else (X[tr].mean(0), X[tr].std(0) + 1e-9)
        Z = (X - mu) / sd
        b = fit(Z, tr)
        z = np.c_[Z[te], np.ones(len(te))] @ b
        take[te] = 1 / (1 + np.exp(-np.clip(z, -30, 30))) >= 0.5
    return take


def total(take):
    return float(np.where(take, hB, hA).sum())


# ★★★ ТРЕТИЙ НАБОР — АГРЕГАТЫ ПО ТРЕКАМ ЛИСТА, И ОН ЕДИНСТВЕННЫЙ РЕАЛИЗУЕМЫЙ СЕГОДНЯ.
# `emit` листа не видит (§6.146, §6.181), но ВСЕ ТРЕКИ листа у него на руках в момент решения:
# `emit.py:392` строит `_pick.json` по всему `by_t`. ⇒ Сводки потрековых величин ПО ЛИСТУ ему
# доступны без единой новой строки плумбинга — в отличие от признаков `_understanding.json`.
# ⚠ В агрегат входит и сам трек: `emit` тоже видит его в общей куче, скрывать нечего.
by_sheet = {}
for i, r in enumerate(keep):
    by_sheet.setdefault(r[0], []).append(i)
AG = np.zeros((len(keep), 3 * len(TF) + 2), float)
for idx in by_sheet.values():
    blk = Xt[idx]
    agg = np.concatenate([blk.sum(0), blk.mean(0), blk.max(0),
                          [float(len(idx)), 0.0]])
    for j in idx:
        agg[-1] = float(len(idx))          # число треков листа — тоже сводка
        AG[j] = agg
AGN = ([f"a_sum_{k}" for k in TF] + [f"a_mean_{k}" for k in TF] + [f"a_max_{k}" for k in TF]
       + ["a_n_tracks", "a_n_tracks2"])

t_only = run(Xt)
t_both = run(np.c_[Xt, Xs])
t_sheet = run(Xs)
t_agg = run(np.c_[Xt, AG])
print(f"\n★★ ПОТРЕКОВЫЙ ВЫБОР, держанно по скважинам (композиция — ЭКРАН, не замер)")
print("| что знает модель | признаков | честных | декодер на треках |")
for nm, tk, n in (("только ТРЕК (как §6.177)", t_only, len(TF)),
                  ("только ЛИСТ (нереализуемо)", t_sheet, len(SN)),
                  ("ТРЕК + ЛИСТ (нереализуемо)", t_both, len(TF) + len(SN)),
                  ("★ ТРЕК + СВОДКИ ПО ТРЕКАМ ЛИСТА (РЕАЛИЗУЕМО)", t_agg, len(TF) + AG.shape[1])):
    print(f"| {nm} | {n} | {total(tk):.0f} | {100*tk.mean():.1f}% |")


def paired(x, yv, nm):
    d = np.where(x, hB, hA) - np.where(yv, hB, hA)
    obs = float(d.sum())
    nz = d[d != 0]
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    rng = np.random.default_rng(20260904)
    if len(nz):
        sg = rng.integers(0, 2, size=(a.perm, len(nz))) * 2 - 1
        p = float((np.abs((sg * np.abs(nz)).sum(1)) >= abs(obs)).mean())
    else:
        p = 1.0
    print(f"★ {nm}: Δ = {obs:+.0f}, треков ↑{up}/↓{dn}, p = {p:.4f}")
    return obs, p


print()
obs, p = paired(t_both, t_only, "ТРЕК+ЛИСТ против ТОЛЬКО ТРЕК (опора — текущее лучшее)")
paired(t_sheet, t_only, "ТОЛЬКО ЛИСТ против ТОЛЬКО ТРЕК")
obs_a, p_a = paired(t_agg, t_only, "★ ТРЕК+СВОДКИ против ТОЛЬКО ТРЕК (РЕАЛИЗУЕМЫЙ вариант)")
paired(t_agg, t_both, "сводки против полного контекста листа (сколько теряем реализуемостью)")

# ── ЦЕНА УТЕЧКИ НОРМИРОВКИ: тот же код, отличается ТОЛЬКО областью среднего и разброса ─────────
# ⚠ Иначе разницу с §6.177 нельзя приписать нормировке: там отличались ещё l2, шаг и число итераций.
lk_t = run(Xt, leak_norm=True)
lk_b = run(np.c_[Xt, Xs], leak_norm=True)
print(f"\n★★ ЦЕНА УТЕЧКИ НОРМИРОВКИ (всё прочее тождественно, меняется ТОЛЬКО область mu/sd):")
print(f"   только ТРЕК: честно {total(t_only):.0f} против {total(lk_t):.0f} с утечкой "
      f"({total(lk_t)-total(t_only):+.0f})")
print(f"   ТРЕК+ЛИСТ:   честно {total(t_both):.0f} против {total(lk_b):.0f} с утечкой "
      f"({total(lk_b)-total(t_both):+.0f})")
print("   ⚠ Выгруженные веса `pick_model_h_f*.npz` несут mu/sd по ВСЕМУ набору ⇒ на эту величину "
      "число §6.177 завышено, если разница положительна.")

print("\n★★★ ЧТЕНИЕ")
if obs_a > 0 and p_a < 0.05:
    print(f"   ★★ РЕАЛИЗУЕМЫЙ вариант (сводки по трекам листа) даёт {obs_a:+.0f} при p = {p_a:.4f} — "
          f"и он НЕ требует правки прода: `emit` видит все треки листа в момент решения.")
else:
    print(f"   ⚠ Реализуемый вариант даёт {obs_a:+.0f} при p = {p_a:.4f} ⇒ сводок по трекам мало; "
          f"сигнал §6.181 держится на том, чего `emit` не видит.")
if obs > 0 and p < 0.05:
    print(f"   Контекст листа даёт потрековому выбору {obs:+.0f} кривых при p = {p:.4f} ⇒ сигнал "
          f"ЕСТЬ, и это основание потратить ~5 ч на настоящий прогон. Композиция цену НЕ называет.")
else:
    print(f"   Контекст листа даёт {obs:+.0f} при p = {p:.4f} ⇒ сигнала нет; прогон запускать не на что.")
    print("   ⚠ Экран отвечает только на «есть ли сигнал». Отрицательный ответ здесь надёжнее "
          "положительного: композиция завышать не склонна, она ошибается в РАСПРЕДЕЛЕНИИ (§6.175).")
