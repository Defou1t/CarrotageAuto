r"""_level_tm.py — СДВИГОВЫЕ ЦЕПОЧКИ (ТЕРМОМЕТРИЯ): УРОВЕНЬ ПО ГЛАДКИМ УЧАСТКАМ ТРАССЫ (03.10, §6.267; критерии — в ROADMAP
до прогона).

Переход термометрии: кривая доходит до правого края, пунктир идёт справа налево (вдоль него — новая шкала), кривая
продолжается от левого края. Около перехода трасса выдачи мечется по пунктиру, подписям и сетке, но между переходами она
гладкая и длинная. Поэтому уровень решается по участкам, а не по строкам:
1. Гладкие участки трассы выдачи: подряд идущие строки с |Δx| ≤ 0.03 ширины шкалы на строку, длина ≥ 150 строк.
2. Первый участок — уровень 0. Следующий — уровень предыдущего или на 1 больше: какой даёт меньший разрыв значения между
   концом предыдущего и началом этого (медианы 20 крайних строк). Уровень не убывает.
3. Строки между участками — уровень последнего начавшегося участка.

Сдвиговая цепочка: у каждой пары соседних уровней v_left растёт, ширина диапазона та же (±20%). Остальные кривые — уровни
§6.266 v2 (дамп `_level_ink.py`). Мера — именные честные в значениях; парный знаковый тест по листам против v2.

  _level_tm.py --pred F:/nds/output/taskS/level_ink_pred_v2r.pkl
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from collections import Counter, defaultdict
import numpy as np
from _level_ink import honest, val

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--pred", default=r"F:/nds/output/taskS/level_ink_pred_v2r.pkl")
ap.add_argument("--step", type=float, default=0.03)
ap.add_argument("--min-len", type=int, default=150)
ap.add_argument("--edge", type=int, default=20)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/level_tm_levels.pkl")
a = ap.parse_args()
CUR = pickle.load(open(a.cache, "rb"))
LV = pickle.load(open(a.pred, "rb"))["levels"]


def is_shift(ch):
    if len(ch) < 2:
        return False
    for p, q in zip(ch, ch[1:]):
        wp = p["v_right"] - p["v_left"]; wq = q["v_right"] - q["v_left"]
        if not (q["v_left"] > p["v_left"]) or wp == 0 or abs(wq - wp) > 0.2 * abs(wp):
            return False
    return True


def stretches(ty, tx, w):
    """→ [(i, j)] — индексы начала и конца гладких участков плотной трассы"""
    ok = (np.diff(ty) == 1) & (np.abs(np.diff(tx)) <= a.step * w)
    out, i, n = [], 0, len(ty)
    while i < n - 1:
        if not ok[i]:
            i += 1; continue
        j = i
        while j < n - 1 and ok[j]:
            j += 1
        if ty[j] - ty[i] + 1 >= a.min_len:
            out.append((i, j))
        i = j + 1
    return out


def decode_tm(cv):
    ch = cv["chain"]; K = len(ch)
    w = abs(ch[0]["x_right"] - ch[0]["x_left"]) or 1.0
    ty, tx = cv["ty"], cv["tx"].astype(np.float64)
    R = stretches(ty, tx, w)
    if not R:
        return None
    lv, k, pv = {}, 0, None
    for i, j in R:
        xs = float(np.median(tx[i:i + a.edge])); xe = float(np.median(tx[max(i, j - a.edge + 1):j + 1]))
        if pv is not None and k + 1 < K:
            if abs(val(ch[k + 1], xs) - pv) < abs(val(ch[k], xs) - pv):
                k += 1
        lv[int(ty[i])] = k
        pv = val(ch[k], xe)
    return lv


C = Counter(); PER = defaultdict(lambda: [0, 0]); FH = Counter(); n_shift = Counter()
OUT = {}
for ci, cv in enumerate(CUR):
    base_lo = LV.get(ci)                                   # v2; нет — уровни прода
    h0 = honest(cv, base_lo)
    h1 = h0
    if is_shift(cv["chain"]):
        n_shift[(cv["set"], re.sub(r"^BKZ_", "", cv["root"]))] += 1
        lo = decode_tm(cv)
        if lo is not None:
            h1 = honest(cv, lo); OUT[ci] = lo
            FH[(cv["set"], re.sub(r"^BKZ_", "", cv["root"]), 0)] += h0; FH[(cv["set"], re.sub(r"^BKZ_", "", cv["root"]), 1)] += h1
    C[(cv["set"], 0)] += h0; C[(cv["set"], 1)] += h1
    PER[(cv["set"], cv["sheet"])][0] += h0; PER[(cv["set"], cv["sheet"])][1] += h1
pickle.dump(OUT, open(a.dump, "wb"))
for sn in ("поле", "сорт A"):
    print(f"   {sn}: кривых со сдвиговой цепочкой — " + ", ".join(f"{f} {v}" for (s, f), v in n_shift.most_common() if s == sn))
rng = np.random.default_rng(0)
for sn in ("поле", "сорт A"):
    d = np.array([v1 - v0 for (s, _), (v0, v1) in PER.items() if s == sn]); nz = d[d != 0]
    p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
    print(f"★ {sn}: именных честных в значениях — v2 {C[(sn, 0)]}, v2 + участки {C[(sn, 1)]} (Δ {C[(sn, 1)] - C[(sn, 0)]:+d}, "
          f"листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
    fams = sorted({k[1] for k in FH if k[0] == sn}, key=lambda f_: -FH[(sn, f_, 1)])
    print("   по семействам (v2 → участки): " + "; ".join(f"{f_} {FH[(sn, f_, 0)]} → {FH[(sn, f_, 1)]}" for f_ in fams[:12]))
