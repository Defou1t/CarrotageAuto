r"""_strat_beam.py — ЛУЧЕВОЙ ПОИСК (beam search) вместо жадного выбора рана.

ИДЕЯ. Базовый трассировщик ЖАДНЫЙ: ошибся строкой — назад дороги нет. Держим K гипотез
(x, v, ширина пера, накопленная стоимость). На каждой строке каждую гипотезу разворачиваем
по NC ближайшим к её pred ранам ПЛЮС «коаст» (пропуск строки с затуханием скорости), отсекаем
до K, в конце берём минимальную по сумме и восстанавливаем путь.

ДВА УРОКА ПЕРВОЙ ИТЕРАЦИИ (обе — реальные прогоны, см. историю ниже):
 1. Стоимость НЕЛЬЗЯ мерить как |nx-pred| за каждую строку: тогда самый дешёвый путь на листе —
    ПРЯМАЯ ЛИНИЯ РАМКИ (|nx-pred|=0 всегда), и луч честно на неё садится (med 660px). Платить
    надо только за РАЗРЫВ СВЯЗНОСТИ: ран перекрывает pred → почти даром (штрих физически
    непрерывен), не перекрывает → дорого, коаст → дёшево, но за каждую строку.
 2. Глобальный argmin по всей кривой — тоже ловушка (тот же эффект рамки на длинном плече).
    Решение — ГОРИЗОНТ: каждые D строк луч ФИКСИРУЕТ решение D-строчной давности (оставляем
    только потомков лучшей гипотезы). Сравнение «переждать обрыв vs латчнуться на соседа»
    становится локальным: коаст*D против одного jump.

  python _strat_beam.py            # база + текущий лучший вариант
  python _strat_beam.py --sweep A  # серия прогонов
"""
import sys, argparse, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import numpy as np
import _relatch_bench as BE

SLMAX = BE.SLMAX
WIDE = BE.WIDE_RUN

P0 = dict(
    K=12,            # ширина луча
    NC=3,            # сколько ближайших ранов разворачивать на гипотезу
    D=120,           # ГОРИЗОНТ: через столько строк решение фиксируется
    w_cont=0.02,     # цена |nx-pred|, когда ран ПЕРЕКРЫВАЕТ pred (связность — почти даром)
    w_step=0.50,     # цена |nx-pred| при прыжке (сверх мёртвой зоны)
    dz=12.0,         # МЁРТВАЯ ЗОНА: законный перезахват своей кривой идёт на med 15px (замер),
                     # это нормальная кривизна, а не смена личности. Без неё дешевле СТОЯТЬ,
                     # чем идти за быстрым выносом кривой — луч замирал на 160px, пока GT уходил на 750
    jump=1.0,        # фиксированная цена разрыва связности
    jmax=40.0,       # ВОРОТА: за этим расстоянием прыжок считается сменой личности
    far=400.0,       # цена прыжка за воротами (замер: 93% законных перезахватов ближе 50px,
                     # а из ЧУЖИХ ранов на тех же строках внутри 50px лишь 15%)
    lam_w=1.0,       # цена несогласованности ширины штриха (только при прыжке)
    coast=1.50,      # цена одной строки коаста
    tau=30.0,        # РОСТ цены коаста: coast*(1+k/tau). Без роста «потеряться» — поглощающее
                     # состояние: коаст 1.5/строку навсегда дешевле любого перезахвата (cov→0.04)
    cmax=200,        # жёсткий предел строк подряд в коасте
    decay=0.85,      # затухание скорости при коасте
    aw=0.05,         # EMA ширины пера
    w_pr=0.0,        # ПРИОР ЛИЧНОСТИ: w_pr * max(0, |x-base| - pr_r) за строку.
    pr_f=0.0,        # ...радиус как доля ПОЛУШИРИНЫ полосы (0 = брать абсолютный pr_r).
                     # Фиксированный радиус не годится всем: GZ51 четверть строк живёт дальше
                     # 230px от base, а SP1 не отходит и на 50 — один и тот же r их не мирит.
    pr_r=150.0,      # Без него самый дешёвый путь на листе — самая ГЛАДКАЯ тушь в полосе, а это
                     # почти всегда не наша кривая: замер на GZ51 — луч встал на прямую у края
                     # полосы (x~986 при медиане GT 331) и прошёл так весь лист, 3 шага >30px
                     # на 23088 строк. Связность+расстояние личность НЕ задают; base (x_center
                     # линии, он же используется базовым алгоритмом при инициализации) — задаёт.
    mg=0.0,          # СКЛЕЙКА РАНОВ в строке через разрыв <= mg. Быстрый вынос пера идёт почти
                     # горизонтально и в маске рассыпается на клочки (замер на GZ41: строка 3539
                     # = 9 огрызков от 295 до 537). Склеенный «сверхран» даёт связности шанс
                     # перенести нить вдоль строки к вершине выноса. 0 = выкл.
    dedup=2.0,       # px, слияние гипотез с близким x
    init_w=1.0,      # цена рождения: init_w*|c - base|
    birth=2.0,       # цена ОЖИДАНИЯ рождения за строку (луч сам выбирает, где стартовать)
    bwin=300,        # ...но только в первых bwin строках с чернилами. Без окна луч «рождается»
                     # в середине листа, пропуская трудное начало: med 5px при cov 0.34 — брак
    fill=400,        # макс. длина коаст-дыры, достраиваемой интерполяцией
)


def _beam(rec, csr, H, P):
    base = rec["base"]
    A_all, B_all, C_all, ptr = csr
    y0 = max(0, rec["y0"]); y1 = min(H, rec["y1"] + 1)
    K, NC, D = P["K"], P["NC"], P["D"]
    off = np.arange(-NC, NC, dtype=np.int64)
    w_cont, w_step, jump = P["w_cont"], P["w_step"], P["jump"]
    jmax, far_p = P["jmax"], P["far"]
    pr_r = P["pr_f"] * (rec["hi"] - rec["lo"]) / 2.0 if P["pr_f"] else P["pr_r"]
    lam_w, coast_c, decay, aw, dedup = P["lam_w"], P["coast"], P["decay"], P["aw"], P["dedup"]

    x = np.zeros(0); v = np.zeros(0); w = np.zeros(0); cost = np.zeros(0); nco = np.zeros(0)
    unborn = 0.0; nink = 0                             # цена «ещё не стартовали»
    par_h, x_h, y_h = [], [], []                       # история: родитель, x (NaN=коаст), строка
    since = 0                                          # строк с последней фиксации

    for y in range(y0, y1):
        i, j = ptr[y], ptr[y + 1]
        m = int(j - i)
        if m == 0:
            if len(x):                                 # чернил нет вовсе — коаст без штрафа
                x = x + np.clip(v, -SLMAX, SLMAX)
                par_h.append(np.arange(len(x)))
                x_h.append(np.full(len(x), np.nan)); y_h.append(y); since += 1
            continue
        A = A_all[i:j].astype(np.float64)
        B = B_all[i:j].astype(np.float64)
        C = C_all[i:j]
        n = len(x)

        # ── РОЖДЕНИЕ: ран, ближайший к base. Ожидание стоит birth за строку, поэтому луч сам
        # решает, на какой строке стартовать — иначе первая же строка с одиноким чужим раном
        # (реальный случай BOGAT_011: единственный ран в 592px от base) намертво задаёт личность.
        nink += 1
        if nink <= P["bwin"] or not n:
            kb = int(np.argmin(np.abs(C - base)))
            wb = float(B[kb] - A[kb])
            xb = (float(B[kb]) if abs(B[kb] - base) >= abs(A[kb] - base) else float(A[kb])) \
                if wb >= WIDE else float(C[kb])
            b_cost = unborn + P["init_w"] * abs(float(C[kb]) - base)
            unborn += P["birth"]
        else:
            b_cost = np.inf; xb = 0.0; wb = 1.0

        if n:
            pred = x + np.clip(v, -SLMAX, SLMAX)
            pos = np.searchsorted(C, pred)
            idx = np.clip(pos[:, None] + off[None, :], 0, m - 1)       # (n, 2NC)
            c = C[idx]; a = A[idx]; b = B[idx]
            d = np.abs(c - pred[:, None])
            if idx.shape[1] > NC:
                sel = np.argsort(d, axis=1)[:, :NC]
                r = np.arange(n)[:, None]
                c = c[r, sel]; a = a[r, sel]; b = b[r, sel]
            wid = b - a
            far = np.where(np.abs(b - base) >= np.abs(a - base), b, a)
            # ШИРОКИЙ РАН ДВОЯК: либо ВЕРШИНА горизонтального выноса (перо ушло к упору), либо
            # просто клякса-пересечение, сквозь которую нить идёт прямо. Жадный алгоритм всегда
            # читает его как вершину. Луч держит ОБЕ версии и даёт решить тому, что будет дальше.
            nx1 = np.where(wid >= WIDE, far, c)
            nx2 = np.where(wid >= WIDE, np.clip(pred[:, None], a, b), c)
            nx = np.concatenate([nx1, nx2], axis=1)
            a2 = np.concatenate([a, a], axis=1); b2 = np.concatenate([b, b], axis=1)
            c2 = np.concatenate([c, c], axis=1); wid2 = np.concatenate([wid, wid], axis=1)
            step = np.abs(nx - pred[:, None])
            dc = np.abs(c2 - pred[:, None])                             # расстояние по центру
            ov = (a2 - 2 <= pred[:, None]) & (pred[:, None] <= b2 + 2)
            wp = lam_w * np.abs(wid2 - w[:, None])
            gate = np.where(dc > jmax, far_p, jump)                     # ВОРОТА смены личности
            pen = w_step * np.maximum(dc - P["dz"], 0.0) + gate + wp
            cl = cost[:, None] + np.where(ov, w_cont * step, pen)
            wl = np.where(wid2 < WIDE, (1 - aw) * w[:, None] + aw * wid2, w[:, None])
            if P["mg"] and m > 1:
                # ── «сверхран»: цепочка ранов строки, разорванная не более чем на mg.
                # Предлагается ТОЛЬКО когда pred уже внутри цепочки — это чистый ход по
                # связности («доехать по горизонтальному штриху до его вершины»), а не прыжок.
                cut = np.nonzero((A[1:] - B[:-1]) > P["mg"])[0]
                MA = A[np.concatenate([[0], cut + 1])]
                MB = B[np.concatenate([cut, [m - 1]])]
                q = np.clip(np.searchsorted(MA, pred) - 1, 0, len(MA) - 1)
                ma = MA[q]; mb = MB[q]
                inside = (ma - 2 <= pred) & (pred <= mb + 2)
                mfar = np.where(np.abs(mb - base) >= np.abs(ma - base), mb, ma)
                nx = np.concatenate([nx, mfar[:, None]], axis=1)
                cl = np.concatenate(
                    [cl, np.where(inside, cost + w_cont * np.abs(mfar - pred), np.inf)[:, None]],
                    axis=1)
                wl = np.concatenate([wl, w[:, None]], axis=1)
            vl = 0.6 * v[:, None] + 0.4 * (nx - x[:, None])
            NB = nx.shape[1]
            if P["w_pr"]:                                   # приор личности (см. P0)
                cl = cl + P["w_pr"] * np.maximum(np.abs(nx - base) - pr_r, 0.0)
            cc = np.where(nco < P["cmax"], cost + coast_c * (1.0 + nco / P["tau"]), np.inf)
            if P["w_pr"]:
                cc = cc + P["w_pr"] * np.maximum(np.abs(pred - base) - pr_r, 0.0)
            all_cost = np.concatenate([cl.ravel(), cc, [b_cost]])
            all_x = np.concatenate([nx.ravel(), pred, [xb]])
            all_v = np.concatenate([vl.ravel(), v * decay, [0.0]])
            all_w = np.concatenate([wl.ravel(), w, [max(wb, 1.0)]])
            all_p = np.concatenate([np.repeat(np.arange(n), NB), np.arange(n), [-1]])
            all_nc = np.concatenate([np.zeros(n * NB), nco + 1, [0.0]])
            is_coast = np.concatenate([np.zeros(n * NB, bool), np.ones(n, bool), [False]])
        else:
            all_cost = np.array([b_cost]); all_x = np.array([xb])
            all_v = np.zeros(1); all_w = np.array([max(wb, 1.0)])
            all_p = np.array([-1]); all_nc = np.zeros(1); is_coast = np.zeros(1, bool)

        order = np.argsort(all_cost, kind="stable")
        keys = np.round(all_x[order] / dedup).astype(np.int64)
        _, first = np.unique(keys, return_index=True)
        keep = order[np.sort(first)][:K]

        off0 = float(all_cost[keep].min())
        cost = all_cost[keep] - off0; unborn -= off0
        x = all_x[keep]; v = all_v[keep]; w = all_w[keep]; nco = all_nc[keep]
        par_h.append(all_p[keep]); x_h.append(np.where(is_coast[keep], np.nan, all_x[keep]))
        y_h.append(y); since += 1

        # ── ГОРИЗОНТ: фиксируем решение D-строчной давности, режем несогласных
        if D and since >= D and len(x) > 1:
            t = len(par_h) - 1
            anc = np.arange(len(x))
            for s in range(t, max(0, t - D), -1):
                anc = np.where(anc < 0, -1, par_h[s][np.maximum(anc, 0)])
            kb_ = anc[int(np.argmin(cost))]
            good = (anc == kb_) if kb_ >= 0 else np.ones(len(anc), bool)
            if good.any() and not good.all():
                sub = np.nonzero(good)[0]
                cost = cost[sub] - cost[sub].min(); x = x[sub]; v = v[sub]; w = w[sub]
                nco = nco[sub]
                par_h[t] = par_h[t][sub]; x_h[t] = x_h[t][sub]
            since = 0

    if not len(x) or not par_h:
        return {}

    k = int(np.argmin(cost))
    ys, xs = [], []
    for t in range(len(par_h) - 1, -1, -1):
        ys.append(y_h[t]); xs.append(x_h[t][k]); k = int(par_h[t][k])
        if k < 0:
            break
    ys = np.array(ys[::-1]); xs = np.array(xs[::-1])

    tr = {int(yy): float(xx) for yy, xx in zip(ys, xs) if not np.isnan(xx)}
    if P["fill"] and len(tr) >= 2:                      # коаст-дыры → интерполяция (иначе cov)
        gi = np.nonzero(~np.isnan(xs))[0]
        for s, e in zip(gi[:-1], gi[1:]):
            if e - s <= 1 or (ys[e] - ys[s]) > P["fill"]:
                continue
            dy = float(ys[e] - ys[s])
            for t in range(s + 1, e):
                f = (ys[t] - ys[s]) / dy
                tr[int(ys[t])] = float(xs[s] + f * (xs[e] - xs[s]))
    BE._extend_ends(tr, csr, H, SLMAX)
    return tr


def make(**kw):
    P = dict(P0); P.update(kw)
    return (lambda rec, csr, H: _beam(rec, csr, H, P)), P


def run(name, **kw):
    tr, P = make(**kw)
    t = time.time()
    res = BE.report(name, BE.run_strategy(tracer=tr))
    print(f"   ({time.time()-t:.0f}s)  " + ", ".join(f"{k}={v}" for k, v in kw.items()))
    return res


SWEEP = {
    "A": [dict(w_pr=0.0), dict(w_pr=0.02, pr_r=200.0), dict(w_pr=0.05, pr_r=150.0),
          dict(w_pr=0.05, pr_r=250.0), dict(w_pr=0.2, pr_r=150.0), dict(w_pr=0.5, pr_r=80.0)],
    "B": [dict(pr_f=0.3), dict(pr_f=0.45), dict(pr_f=0.6), dict(pr_f=0.45, w_pr=0.15),
          dict(pr_f=0.45, jmax=20.0), dict(pr_f=0.45, jmax=80.0), dict(pr_f=0.45, coast=0.5),
          dict(pr_f=0.45, coast=4.0), dict(pr_f=0.45, D=40), dict(pr_f=0.45, D=300)],
    "C": [dict(), dict(lam_w=0.0), dict(lam_w=3.0), dict(far=150.0), dict(far=1000.0),
          dict(tau=10.0), dict(tau=100.0), dict(cmax=60), dict(fill=0), dict(w_cont=0.0)],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="")
    a = ap.parse_args()
    if a.sweep == "":
        BE.report("baseline (как в проде)", BE.run_strategy())
        run("beam P0")
    else:
        BE.report("baseline (как в проде)", BE.run_strategy())
        for kw in SWEEP[a.sweep]:
            run("beam " + (str(kw) if kw else "P0"), **kw)
