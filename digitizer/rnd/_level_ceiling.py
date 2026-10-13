r"""_level_ceiling.py — ЧТО ЧИНИТЬ В УРОВНЯХ МАСШТАБА: ГРАНИЦЫ ИЛИ МЕТКИ ОТРЕЗКОВ (11.10, к §6.299).

Кэш `_level_bench.py build` (трассы выдачи прода + эталон). Для кривой с цепочкой масштабов ≥ 2 на общих строках трасса / эталон
считается ошибка e[k, строка] = |x выдачи, пересчитанный с уровня k на уровень эталона − x эталона|. Варианты уровней выдачи:
- выдача — уровни прода как есть;
- метки отрезков — границы прода (пробеги уровня выдачи), на каждом пробеге лучший один k (больше строк ≤ 3 px);
- одна метка — один k на всю кривую;
- Витерби-оракул — свободные границы, цена строки [e > 3], смена уровня стоит `--pen` (не дробит на шум);
- оракул по строкам — лучший k на каждой строке (верхняя граница).
Честность — как в мере приёмки: медиана ошибки ≤ 3 px, покрытие ≥ 0.9. Печатает именных честных в значениях по вариантам.

  _level_ceiling.py --cache F:/nds/output/taskS/level_bench_nslx4.pkl --pen 5 20
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench_nslx4.pkl")
ap.add_argument("--pen", nargs="+", type=float, default=[5.0, 20.0])
ap.add_argument("--jump", type=float, default=0.4, help="§6.304: разворот — |Δx| ≥ этой доли ширины полосы за ≤ 10 строк")
ap.add_argument("--edge", type=float, default=0.85, help="§6.304: разворот — перед прыжком трасса у края (u ≥ edge / ≤ 1 − edge)")
ap.add_argument("--dump", default=r"F:/nds/output/taskS/level_ceiling.pkl")
a = ap.parse_args()


def seg_levels(segs, rows):
    out = np.zeros(len(rows), np.int16)
    for y0, y1, lv in segs or []:
        if lv:
            out[(rows >= y0) & (rows <= y1)] = lv
    return out


def val(s, x):
    return s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / ((s["x_right"] - s["x_left"]) or 1)


def pix(s, v):
    return s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / ((s["v_right"] - s["v_left"]) or 1)


def wrap_levels(cv, K):
    """§6.304 (заказчик 13.10): уровень по направлению разворотов трассы — ушла за правый край и появилась левее (Δx < 0) — +1,
    обратный разворот (Δx > 0) — −1; старт — так, чтобы уровень нигде не был < 0. → (уровни по строкам трассы, есть ли развороты)"""
    ty, tx, ch = cv["ty"], cv["tx"].astype(np.float64), cv["chain"]
    xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
    W = max(1.0, xr - xl)
    d = np.zeros(len(ty), np.int16)
    if len(ty) > 1:
        dy = np.diff(ty); dx = np.diff(tx)
        J = np.flatnonzero((dy <= 10) & (np.abs(dx) >= a.jump * W))
        u = (tx - xl) / W
        for j in J:
            ub = np.median(u[max(0, j - 4):j + 1]); ua = np.median(u[j + 1:j + 21])
            if len(u[j + 1:j + 21]) < 20:
                continue
            if dx[j] < 0 and ub >= a.edge and ua <= 0.5:
                d[j + 1] = 1                                   # ушла за правый край — появилась левее: уровень вверх
            elif dx[j] > 0 and ub <= 1 - a.edge and ua >= 0.5:
                d[j + 1] = -1                                  # обратный разворот

    cum = np.cumsum(d)
    lev = cum - min(0, int(cum.min()))
    return np.clip(lev, 0, K - 1).astype(np.int16), bool(np.any(d != 0))


def viterbi(bad, pen):
    K, n = bad.shape
    cost = bad[:, 0].astype(np.float64).copy(); bp = np.zeros((K, n), np.int16)
    for t in range(1, n):
        best = cost.min(); bk = int(cost.argmin())
        stay = cost; sw = best + pen
        bp[:, t] = np.where(stay <= sw, np.arange(K), bk)
        cost = np.minimum(stay, sw) + bad[:, t]
    s = int(cost.argmin()); path = np.empty(n, np.int16)
    for t in range(n - 1, -1, -1):
        path[t] = s; s = bp[s, t]
    return path


CUR = pickle.load(open(a.cache, "rb"))
VAR = ["выдача", "развороты", "выдача + старт по разворотам", "метки отрезков", "непрерывность", "вып. ИЛИ непр.", "одна метка"] + [f"Витерби {p:g}" for p in a.pen] + ["оракул по строкам"]
C = Counter(); OUT = []
for cv in CUR:
    gy, gx, ty, tx, ch = cv["gy"], cv["gx"], cv["ty"], cv["tx"], cv["chain"]
    com, ig, it = np.intersect1d(gy, ty, return_indices=True)
    if len(com) < 30:
        continue
    cov = len(com) / max(1, len(gy)) >= 0.9
    x = tx[it].astype(np.float64); g = gx[ig].astype(np.float64)
    lt = seg_levels(cv["lt"], com); lw = seg_levels(cv["lw"], com)
    K = max(1, len(ch))
    E = np.full((K, len(com)), 1e4)
    if K >= 2:
        for k in range(K):
            for t in np.unique(lt):
                m = lt == t
                if t >= K:
                    continue
                E[k, m] = np.abs(pix(ch[t], val(ch[k], x[m])) - g[m])
    else:
        E[0] = np.abs(x - g)
    hon = lambda lev: bool(cov and np.median(E[np.clip(lev, 0, K - 1), np.arange(len(com))]) <= 3.0)
    res = {"выдача": hon(np.minimum(lw, K - 1))}
    bad = (E > 3.0).astype(np.float64)
    if K >= 2:
        lev = np.minimum(lw, K - 1).copy()
        st = np.flatnonzero(np.diff(np.concatenate([[-1], lw])) != 0); en = np.concatenate([st[1:], [len(lw)]])
        for s_, e_ in zip(st, en):
            lev[s_:e_] = int(np.argmin(bad[:, s_:e_].sum(1)))
        res["метки отрезков"] = hon(lev)
        wl, has_w = wrap_levels(cv, K)
        wl_c = wl[it]                                          # на общих строках
        res["развороты"] = hon(wl_c)
        # гибрид: уровни выдачи, но до первого разворота — стартовый уровень по правилу
        hyb = np.minimum(lw, K - 1).copy()
        if has_w:
            fj = int(np.flatnonzero(np.diff(np.concatenate([[wl[0]], wl])) != 0)[0]) if np.any(np.diff(wl) != 0) else len(wl)
            y_fj = cv["ty"][min(fj, len(wl) - 1)]
            hyb[com < y_fj] = wl[0]
        res["выдача + старт по разворотам"] = hon(hyb)
        # непрерывность значения: якорь — самый длинный пробег с уровнем выдачи; соседние пробеги — k с ближайшим значением
        # на стыке (медианы 10 крайних строк), в долях диапазона шкалы 0
        rng0 = abs(ch[0]["v_right"] - ch[0]["v_left"]) or 1.0
        if len(st) >= 2:
            lab = [int(min(lw[s_], K - 1)) for s_ in st]
            j0 = int(np.argmax(en - st))
            ok_runs = [None] * len(st); ok_runs[j0] = lab[j0]
            for order in (range(j0 + 1, len(st)), range(j0 - 1, -1, -1)):
                for j in order:
                    pj = j - 1 if j > j0 else j + 1
                    s_p, e_p = st[pj], en[pj]
                    xp = np.median(x[e_p - min(10, e_p - s_p):e_p]) if j > j0 else np.median(x[s_p:s_p + min(10, e_p - s_p)])
                    xc = np.median(x[st[j]:st[j] + min(10, en[j] - st[j])]) if j > j0 else np.median(x[en[j] - min(10, en[j] - st[j]):en[j]])
                    vp = val(ch[ok_runs[pj]], xp)
                    ok_runs[j] = int(np.argmin([abs(val(ch[k], xc) - vp) for k in range(K)]))
            lv2 = np.minimum(lw, K - 1).copy()
            for j, (s_, e_) in enumerate(zip(st, en)):
                lv2[s_:e_] = ok_runs[j]
            res["непрерывность"] = hon(lv2)
            res["вып. ИЛИ непр."] = res["выдача"] or res["непрерывность"]
        else:
            res["непрерывность"] = res["выдача"]; res["вып. ИЛИ непр."] = res["выдача"]
        res["первый отрезок"] = bool(lev[0] != min(lw[0], K - 1))
        res["прочие отрезки"] = bool(np.any(lev[1:] != np.minimum(lw[1:], K - 1)))
        res["одна метка"] = max(hon(np.full(len(com), k, np.int16)) for k in range(K))
        for p in a.pen:
            res[f"Витерби {p:g}"] = hon(viterbi(bad, p))
        res["оракул по строкам"] = bool(cov and np.median(E.min(0)) <= 3.0)
    else:
        for v in VAR[1:]:
            res[v] = res["выдача"]
        res["первый отрезок"] = res["прочие отрезки"] = False
    for v in VAR:
        C[(cv["set"], v)] += res[v]
    OUT.append(dict(set=cv["set"], sheet=cv["sheet"], name=cv["name"], root=cv["root"], K=K, **res))
pickle.dump(OUT, open(a.dump, "wb"))
for sn in ("поле", "сорт A"):
    fx = [r for r in OUT if r["set"] == sn and r["метки отрезков"] and not r["выдача"]]
    print(f"{sn}: чинятся метками отрезков {len(fx)} — неверна метка первого отрезка у {sum(r['первый отрезок'] for r in fx)}, "
          f"прочих у {sum(r['прочие отрезки'] for r in fx)}; ломает непрерывность честных выдачи: "
          f"{sum(1 for r in OUT if r['set'] == sn and r['выдача'] and not r['непрерывность'])}")
for sn in ("поле", "сорт A"):
    print(f"\n★ {sn}: именных честных в значениях (кривых в кэше {sum(1 for r in OUT if r['set'] == sn)}, с цепочкой "
          f"{sum(1 for r in OUT if r['set'] == sn and r['K'] >= 2)})")
    for v in VAR:
        print(f"   {v:18s} {C[(sn, v)]:5d}  ({C[(sn, v)] - C[(sn, 'выдача')]:+d})")
