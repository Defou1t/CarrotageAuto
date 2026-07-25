r"""_levels_ml.py — ОБУЧАЕМАЯ ЦЕНА ПЕРЕХОДА ДЛЯ DP (первый ML-шаг по уровням).

Почему ML, а не ещё одно правило. За 22.07 опровергнуто ПЯТЬ рукотворных механизмов подряд
(§6.35/§6.38/§6.40): плоская цена, линейная непрерывность глобально, минимальная длительность
уровня, гейт по упору, жёсткий порог прыжка. Перебор 288 комбинаций параметров исчерпан
(0.751 против 0.737 — шум). Остаток — 0.73 из 0.96 возможных.

Диагноз последнего отказа: ошибка в ПРОТЯЖЁННОСТИ сегментов, а не в их числе (Semeguniv_020 GZ31:
эксперт держит 56% строк на верхнем уровне, декодер 92%, переходов 4 против 2). Значит правило
«когда переходить» надо не угадывать, а УЧИТЬ — как в §6.26 для выбора рана.

СХЕМА (минимальная, по образцу §6.24): логистика оценивает P(переход в этой строке) по признакам
строки; DP получает цену перехода −log p вместо рукотворной. Структура DP не меняется.

★ ЧЕСТНОСТЬ: сплит ПО СКВАЖИНАМ; признаки на train и inference считает ОДНА функция; метрика —
corr значений с LAS (LAS в признаки НЕ входит, на инференсе его нет).

  python _levels_ml.py [--min-jump 0.0]
"""
import sys, pickle, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import decode_levels as DL
from _levels_decode2 import decode2, is_additive

CACHE = Path(r"F:\nds\output\taskS\decoder\vsweep_items.pkl")
FEAT = ["|dx|/w", "dx_sign", "x/w", "скорость", "|dx| p95-норм", "с_прошлого_перехода",
        "K", "аддитивная", "лог_отн_ширин", "смена_знака"]


def row_features(it, i, rows, xs, w, last_tr):
    """Признаки строки i для решения «здесь переход». БЕЗ LAS и без уровней — только геометрия
    трассы и рамка. ИДЕНТИЧНА на train и inference."""
    dx = (xs[i] - xs[i - 1]) / w
    sp = [abs(s["v_right"] - s["v_left"]) or 1e-9 for s in it["fam"]]
    wr = float(np.median([sp[j + 1] / sp[j] for j in range(len(sp) - 1)]))
    vel = (xs[i] - xs[max(0, i - 5)]) / (5.0 * w)
    return [abs(dx), np.sign(dx), (xs[i] - min(xs)) / w, vel, min(1.0, abs(dx) / 0.2),
            min(1.0, (rows[i] - last_tr) / 500.0), it["K"] / 3.0,
            1.0 if it["add"] else 0.0, np.log(max(wr, 1e-6)) / 2.0,
            1.0 if (i > 1 and np.sign(xs[i] - xs[i - 1]) != np.sign(xs[i - 1] - xs[i - 2])) else 0.0]


def build_xy(items):
    X, y, g = [], [], []
    for gi, it in enumerate(items):
        rows = it["rows"].tolist(); xs = it["xs"].tolist()
        w = abs(it["fam"][0]["x_right"] - it["fam"][0]["x_left"]) or 1
        gl = it["gl"]
        tr_rows = [rows[i] for i in range(1, len(rows)) if gl.get(rows[i], 0) != gl.get(rows[i - 1], 0)]
        last = rows[0]
        step = max(1, len(rows) // 2000)                 # прореживание: строк десятки тысяч
        for i in range(1, len(rows), step):
            lab = 1.0 if any(abs(rows[i] - t) <= 25 for t in tr_rows) else 0.0
            X.append(row_features(it, i, rows, xs, w, last)); y.append(lab); g.append(gi)
            if lab:
                last = rows[i]
    return np.array(X, float), np.array(y, float), np.array(g)


class Logistic:
    def fit(self, X, y, l2=1e-3, lr=0.5, iters=400):
        n, d = X.shape
        self.w = np.zeros(d); self.b = 0.0
        pos = max(1.0, y.sum()); neg = max(1.0, n - y.sum())
        sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(X @ self.w + self.b, -30, 30)))
            gr = (p - y) * sw
            self.w -= lr * (X.T @ gr / n + l2 * self.w); self.b -= lr * gr.mean()
        return self

    def prob(self, X):
        return 1 / (1 + np.exp(-np.clip(X @ self.w + self.b, -30, 30)))


def decode_ml(it, model, lam_scale=1.0, min_jump=0.0, pfloor=1e-3):
    """DP как в decode2, но цена перехода = lam_scale * (−log p) из модели."""
    fam = it["fam"]; rows = it["rows"].tolist(); xs = it["xs"].tolist()
    K = it["K"]; maps = [DL.scale_map(s) for s in fam]
    w = abs(fam[0]["x_right"] - fam[0]["x_left"]) or 1
    lin = is_additive(fam)
    vspan = abs(fam[0]["v_right"] - fam[0]["v_left"]) or 1.0

    def val(k, x):
        v = maps[min(k, K - 1)](x)
        return v / vspan if lin else np.log(max(abs(v), 1e-6))

    Xf = np.array([row_features(it, i, rows, xs, w, rows[0]) for i in range(1, len(rows))])
    P = np.clip(model.prob(Xf), pfloor, 1 - pfloor)
    cost = -np.log(P) * lam_scale
    dp = [0.0] * K; back = []
    for i in range(1, len(rows)):
        x = xs[i]; xp = xs[i - 1]; dx = x - xp
        ndp = [1e18] * K; bk = [0] * K
        lvp = [val(kp, xp) for kp in range(K)]
        for k in range(K):
            vk = val(k, x); best = 1e18; bki = 0
            for kp in range(K):
                if k == kp:
                    tc = 0.0
                elif min_jump and abs(dx) < min_jump * w:
                    tc = 1e17
                else:
                    want = -1 if k > kp else 1
                    tc = cost[i - 1] * (1.0 if (dx * want) > 0 else 3.0) * abs(k - kp)
                c = dp[kp] + (vk - lvp[kp]) ** 2 + tc
                if c < best:
                    best, bki = c, kp
            ndp[k] = best; bk[k] = bki
        dp = ndp; back.append(bk)
    k = int(np.argmin(dp)); lv = {rows[-1]: k}
    for i in range(len(rows) - 1, 0, -1):
        k = back[i - 1][k]; lv[rows[i - 1]] = k
    return lv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-jump", type=float, default=0.0)
    a = ap.parse_args()
    items = pickle.loads(CACHE.read_bytes())
    wells = sorted({it["well"] for it in items})
    TR = [it for it in items if it["well"] in set(wells[::2])]
    VA = [it for it in items if it["well"] in set(wells[1::2])]
    print(f"кривых {len(items)}: подбор {len(TR)} / проверка {len(VA)} (сплит по скважинам)")

    Xt, yt, _ = build_xy(TR)
    Xv, yv, _ = build_xy(VA)
    print(f"строк-решений: train {len(Xt)} (положительных {yt.sum():.0f}), val {len(Xv)}")
    mdl = Logistic().fit(Xt, yt)
    pv = mdl.prob(Xv)
    print("веса: " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(FEAT, mdl.w)))
    thr = 0.5
    acc = np.mean((pv > thr) == (yv > 0.5))
    tp = ((pv > thr) & (yv > 0.5)).sum(); fp = ((pv > thr) & (yv < 0.5)).sum()
    fn = ((pv <= thr) & (yv > 0.5)).sum()
    print(f"на отложенных скважинах: acc {100*acc:.0f}%  полнота {100*tp/max(1,tp+fn):.0f}%  "
          f"точность {100*tp/max(1,tp+fp):.0f}%")

    def corr(it, lv):
        mp = [DL.scale_map(s) for s in it["fam"]]; K = it["K"]
        V = np.array([mp[min(int(lv.get(y, 0)), K - 1)](x)
                      for y, x in zip(it["rows"].tolist(), it["xs"].tolist())])
        return float(np.corrcoef(V, it["las"])[0, 1]) if V.std() > 0 else float("nan")

    print(f"\n{'вариант':<28}{'подбор':>9}{'ПРОВЕРКА':>11}{'хуже чем ничего':>18}")
    for tag, fn_ in (("правило §6.35", lambda it: decode2(dict(zip(it["rows"].tolist(), it["xs"].tolist())),
                                                          it["fam"], 0.05, 0.01, 8.0, auto=True)),
                     ("ОБУЧАЕМАЯ цена", lambda it: decode_ml(it, mdl, 1.0, a.min_jump)),
                     ("обучаемая ×0.3", lambda it: decode_ml(it, mdl, 0.3, a.min_jump)),
                     ("обучаемая ×3", lambda it: decode_ml(it, mdl, 3.0, a.min_jump))):
        out = []
        for S in (TR, VA):
            cc = []; bad = 0; n = 0
            for it in S:
                d = corr(it, fn_(it)); b = corr(it, {})
                if d == d:
                    cc.append(d); n += 1
                    if b == b and d < b - 0.02:
                        bad += 1
            out.append((float(np.nanmedian(cc)), 100 * bad / max(1, n)))
        print(f"{tag:<28}{out[0][0]:>9.3f}{out[1][0]:>11.3f}{out[1][1]:>17.0f}%")
    print(f"\nопоры: база0 0.395, прод 0.537, оракул 0.961 (проверочные скважины)")
