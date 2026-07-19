r"""_strat_beamwidth.py — ЛУЧЕВОЙ ПОИСК (beam / Viterbi по ранам) + ПРИОР ШИРИНЫ ШТРИХА.

ИДЕЯ. Базовый алгоритм жаден: на каждой строке он необратимо выбирает один ран. Ошибка на
одной строке (обрыв туши, пересечение) уводит трассу на соседа НАВСЕГДА. Здесь вместо жадного
выбора ведётся K гипотез одновременно; стоимость перехода накапливается, и в конце выбирается
дешевейший ПУТЬ ЦЕЛИКОМ. Гипотеза, которая «сползла» на соседа, платит за вход и за выход;
гипотеза, которая ПРОКОАСТИЛА разрыв с затуханием скорости, платит только c_coast за строку.

Состояние = индекс рана на строке (Viterbi-слияние: для каждого рана держим лучшего родителя),
плюс K «коаст»-состояний (по одному на родителя) — коаст с ЗАТУХАНИЕМ скорости v*=decay,
чтобы x не убегал по инерции (это и был баг прежнего эксперимента с запретом прыжка).

СТОИМОСТЬ ПЕРЕХОДА (это и есть «идентичность штриха»):
  w_gap  * dist(pred, [a,b])        — pred вне рана: разрыв связности (0 при перекрытии = бонус)
  w_pred * |nx - pred|              — ускорение (отклонение от постоянной скорости)
  w_wid  * min(|width - wm|, cap_w) — ПРИОР ШИРИНЫ: перо имеет ~постоянную толщину;
                                      wm — EMA собственной ширины, не обновляется на спайках
  c_coast                           — за строку коаста

w_wid=0 даёт ГОЛЫЙ beam — так измеряется вклад приора ширины отдельно.

Запуск:  python _strat_beamwidth.py [base|bare|width|grid|final]
"""
import sys, time
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
import _relatch_bench as BE

SLMAX = 30.0
WIDE_RUN = 14


STAT = {"coast": 0, "run": 0}


def make_tracer(K=8, win=45.0, win_grow=4.0, win_max=400.0, w_gap=1.5, w_pred=0.3,
                w_wid=0.0, cap_w=10.0, c_coast=4.0, decay=0.75, w_init=1.0, n_init=6,
                write_coast=True, wid_ema=0.05, horizon=0, slmax=SLMAX, wide_run=WIDE_RUN):
    """Фабрика tracer(rec, csr, H) -> {y: x} для BE.run_strategy(tracer=...)."""
    NST = max(K, n_init) + 4

    def tracer(rec, csr, H):
        A_all, B_all, C_all, ptr = csr
        lo = float(rec["lo"]); hi = float(rec["hi"]); base = rec["base"]
        y0 = max(0, rec["y0"]); y1 = min(H, rec["y1"] + 1)
        nrows = max(1, y1 - y0)
        cap = (nrows + 2) * NST + 16
        P_par = np.full(cap, -1, np.int32)
        P_y = np.zeros(cap, np.int32)
        P_x = np.zeros(cap, np.float64)
        P_co = np.zeros(cap, np.bool_)
        nn = 0
        x = None; v = None; cost = None; wm = None; nc = None; node = None

        def coast_all(y, nn, x, v, cost, nc, node):
            """Все гипотезы коастят: x=pred (в пределах полосы!), v затухает, ПЛАТИМ c_coast.
            Без клампа и без оплаты состояние уходило за полосу, где ранов нет вовсе, —
            и там коаст был БЕСПЛАТНЫМ и ВЕЧНЫМ (ровно та ловушка, что убивала прогон)."""
            nk = len(x)
            x = np.clip(x + np.clip(v, -slmax, slmax), lo, hi)
            P_par[nn:nn + nk] = node; P_y[nn:nn + nk] = y
            P_x[nn:nn + nk] = x; P_co[nn:nn + nk] = True
            return (nn + nk, x, v * decay, cost + c_coast, nc + 1,
                    np.arange(nn, nn + nk, dtype=np.int32))

        for y in range(y0, y1):
            i = ptr[y]; j = ptr[y + 1]

            # ── скользящий горизонт: раз в horizon строк КОММИТИМ лучшую гипотезу.
            # Без этого beam оптимизирует ПУТЬ ЦЕЛИКОМ, а глобально самый «гладкий»
            # путь на каротажке — это прямая (линейка сетки), а не кривая. ──────
            if horizon and x is not None and len(x) > 1 and (y - y0) % horizon == 0:
                kb2 = int(np.argmin(cost))
                x = x[kb2:kb2 + 1]; v = v[kb2:kb2 + 1]; wm = wm[kb2:kb2 + 1]
                nc = nc[kb2:kb2 + 1]; node = node[kb2:kb2 + 1]; cost = np.zeros(1)

            # ── старт: K гипотез на ближайших к base ранах ──────────────────
            if x is None:
                if i == j:
                    continue
                A = A_all[i:j]; B = B_all[i:j]; Cc = C_all[i:j]
                d = np.abs(Cc - base)
                o = np.argsort(d)[:n_init]
                m = len(o)
                x = Cc[o].astype(np.float64)
                v = np.zeros(m)
                cost = w_init * d[o].astype(np.float64)
                wm = (B[o] - A[o] + 1).astype(np.float64)
                nc = np.zeros(m)
                P_par[nn:nn + m] = -1; P_y[nn:nn + m] = y
                P_x[nn:nn + m] = x; P_co[nn:nn + m] = False
                node = np.arange(nn, nn + m, dtype=np.int32); nn += m
                continue

            pred = x + np.clip(v, -slmax, slmax)
            nk = len(x)
            # ПЕРЕЗАХВАТ: чем дольше гипотеза коастит, тем шире окно поиска рана.
            winv = np.minimum(win + win_grow * nc, win_max)

            if i == j:                                   # строка без чернил вовсе
                nn, x, v, cost, nc, node = coast_all(y, nn, x, v, cost, nc, node)
                continue

            A = A_all[i:j]; B = B_all[i:j]
            wmx = float(winv.max())
            pmin = pred.min() - wmx; pmax = pred.max() + wmx
            s = int(np.searchsorted(B, pmin, "left")); e = int(np.searchsorted(A, pmax, "right"))
            if e <= s:                                   # рядом ранов нет → коаст
                nn, x, v, cost, nc, node = coast_all(y, nn, x, v, cost, nc, node)
                continue

            a = A[s:e].astype(np.float64); b = B[s:e].astype(np.float64); c = C_all[i + s:i + e]
            wr = b - a + 1.0
            wide = (b - a) >= wide_run
            far = np.where(np.abs(b - base) >= np.abs(a - base), b, a)
            NX = np.where(wide, far, c)                  # точка съёма (вершина спайка / центр)
            M = len(a)

            pc = pred[:, None]
            gapd = np.maximum(0.0, np.maximum(a[None, :] - 2.0 - pc, pc - b[None, :] - 2.0))
            dpred = np.abs(NX[None, :] - pc)
            add = w_gap * gapd + w_pred * dpred
            if w_wid:
                add = add + w_wid * np.minimum(np.abs(wr[None, :] - wm[:, None]), cap_w)
            tot = cost[:, None] + add
            tot[np.minimum(gapd, dpred) > winv[:, None]] = np.inf

            kb = np.argmin(tot, axis=0)                  # лучший родитель для каждого рана
            cb = tot[kb, np.arange(M)]

            # кандидаты: M ран-состояний + nk коаст-состояний
            cand_cost = np.concatenate([cb, cost + c_coast])
            keep = np.argsort(cand_cost)[:K]
            keep = keep[np.isfinite(cand_cost[keep])]
            if not len(keep):
                nn, x, v, cost, nc, node = coast_all(y, nn, x, v, cost, nc, node)
                continue

            isrun = keep < M
            ri = keep[isrun]                             # индексы ранов
            ci = keep[~isrun] - M                        # индексы родителей-коастов
            par_r = kb[ri]
            nx_r = NX[ri]
            v_r = 0.6 * v[par_r] + 0.4 * (nx_r - x[par_r])
            wm_r = np.where(wide[ri], wm[par_r],
                            (1.0 - wid_ema) * wm[par_r] + wid_ema * wr[ri])
            nx_c = np.clip(pred[ci], lo, hi)

            newx = np.concatenate([nx_r, nx_c])
            newv = np.concatenate([v_r, v[ci] * decay])
            newc = np.concatenate([cb[ri], cost[ci] + c_coast])
            newwm = np.concatenate([wm_r, wm[ci]])
            newnc = np.concatenate([np.zeros(len(ri)), nc[ci] + 1])
            newpar = np.concatenate([node[par_r], node[ci]])
            newco = np.concatenate([np.zeros(len(ri), np.bool_), np.ones(len(ci), np.bool_)])

            m = len(newx)
            P_par[nn:nn + m] = newpar; P_y[nn:nn + m] = y
            P_x[nn:nn + m] = newx; P_co[nn:nn + m] = newco
            node = np.arange(nn, nn + m, dtype=np.int32); nn += m
            x = newx; v = newv; cost = newc - newc.min(); wm = newwm; nc = newnc

        if x is None or not len(x):
            return {}
        # ── обратный ход по лучшему пути ───────────────────────────────────
        tr = {}
        p = int(node[int(np.argmin(cost))])
        while p >= 0:
            if P_co[p]:
                STAT["coast"] += 1
            else:
                STAT["run"] += 1
            if write_coast or not P_co[p]:
                tr[int(P_y[p])] = float(P_x[p])
            p = int(P_par[p])
        BE._extend_ends(tr, csr, H, slmax)
        return tr

    return tracer


# ───────────────────────────── прогоны ─────────────────────────────

NOCOAST = 1e9

CFG_BARE = dict(K=8, win=45.0, w_gap=1.5, w_pred=0.3, w_wid=0.0,
                c_coast=4.0, decay=0.75, w_init=1.0, n_init=6)

# Опорная конфигурация после гейта: окно НЕ ограничено (ограничение расстоянием — тупик,
# §диагностика: ошибочные прыжки близкие, med 15px), коаст только на пустых строках.
CFG_OPEN = dict(K=1, n_init=1, horizon=1, win=1e9, win_grow=0.0, win_max=1e9,
                c_coast=NOCOAST, w_gap=1.0, w_pred=0.0, w_wid=0.0, cap_w=10.0)


def run(name, verbose=False, _base=None, **kw):
    cfg = dict(_base or CFG_BARE); cfg.update(kw)
    STAT["coast"] = STAT["run"] = 0
    t0 = time.time()
    rows = BE.run_strategy(tracer=make_tracer(**cfg))
    r = BE.report(name, rows)
    tt = STAT["coast"] + STAT["run"]
    print("   коаст в выбранных путях %.1f%%   %.1fs" % (100.0 * STAT["coast"] / max(1, tt),
                                                         time.time() - t0))
    print("   " + " ".join("%s=%s" % kv for kv in sorted(cfg.items())))
    if verbose:
        table(rows)
    r["cfg"] = cfg
    return r


def table(rows):
    print("   %-12s %-8s %8s %6s %6s %6s %6s %6s" %
          ("well", "curve", "med", "cov", "<=3", "своя", "латч", "между"))
    for x in rows:
        mark = "★" if (x["med"] <= 3 and x["cov"] >= 0.9) else " "
        print("  %s%-12s %-8s %8.1f %6.2f %6.2f %6.1f %6.1f %6.1f" %
              (mark, x["well"], x["curve"], x["med"], x["cov"], x["le3"],
               x["своя"], x["латч"], x["между"]))


def sweep(tag, variants, base=None):
    res = []
    for nm, kw in variants:
        res.append(run("%s | %s" % (tag, nm), _base=base, **kw))
    print("\n##### ИТОГ СВОДКА %s #####" % tag)
    for r in sorted(res, key=lambda z: (-z["honest"], z["med_med"])):
        print("   ★%2d/%d  med %8.1f  cov %.2f   %s" %
              (r["honest"], r["n"], r["med_med"], r["med_cov"], r["name"]))
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "base":
        rows = BE.run_strategy(); BE.report("baseline (как в проде)", rows); table(rows)
    elif mode == "bare":
        run("beam ГОЛЫЙ (w_wid=0)", verbose=True)
    elif mode == "width":
        run("beam + приор ширины", w_wid=2.0, verbose=True)
    elif mode == "s1":
        sweep("s1 цена коаста", [
            ("c_coast=8",   dict(c_coast=8.0)),
            ("c_coast=20",  dict(c_coast=20.0)),
            ("c_coast=40",  dict(c_coast=40.0)),
            ("без коаста",  dict(c_coast=NOCOAST)),
        ])
    elif mode == "gate":
        # ГЕЙТ РЕАЛИЗАЦИИ: K=1, окно ∞, коаст запрещён, стоимость = расстояние до рана.
        # Это почти дословно жадный baseline; если тут не ~68px — баг в машинерии, а не в парам.
        sweep("gate", [
            ("мимикрия baseline", dict(K=1, n_init=1, horizon=1, win=1e9, win_grow=0.0,
                                       win_max=1e9, c_coast=NOCOAST, w_gap=1.0, w_pred=0.0)),
            ("мимикрия + w_pred", dict(K=1, n_init=1, horizon=1, win=1e9, win_grow=0.0,
                                       win_max=1e9, c_coast=NOCOAST, w_gap=1.0, w_pred=1.0)),
        ])
    elif mode == "s3":
        sweep("s3 окно прыжка (K=1, жадно, коаст+затухание+перезахват)", [
            ("win=10",  dict(K=1, n_init=1, horizon=1, win=10.0, w_gap=1.0, w_pred=0.0, c_coast=1e6)),
            ("win=20",  dict(K=1, n_init=1, horizon=1, win=20.0, w_gap=1.0, w_pred=0.0, c_coast=1e6)),
            ("win=35",  dict(K=1, n_init=1, horizon=1, win=35.0, w_gap=1.0, w_pred=0.0, c_coast=1e6)),
            ("win=60",  dict(K=1, n_init=1, horizon=1, win=60.0, w_gap=1.0, w_pred=0.0, c_coast=1e6)),
        ])
    elif mode == "s4":
        sweep("s4 ПРИОР ШИРИНЫ поверх открытого выбора (K=1)", [
            ("w_wid=0 (контроль)", dict(w_wid=0.0)),
            ("w_wid=3",            dict(w_wid=3.0)),
            ("w_wid=10",           dict(w_wid=10.0)),
            ("w_wid=30",           dict(w_wid=30.0)),
        ], base=CFG_OPEN)
    elif mode == "s5":
        sweep("s5 ЛУЧ поверх открытого выбора (w_wid по s4)", [
            ("K=1 H=1 (контроль)", dict(K=1, n_init=1, horizon=1)),
            ("K=6 H=60",           dict(K=6, n_init=6, horizon=60)),
            ("K=6 H=200",          dict(K=6, n_init=6, horizon=200)),
            ("K=12 H=60",          dict(K=12, n_init=12, horizon=60)),
        ], base=CFG_OPEN)
    elif mode == "s6":
        B = dict(CFG_OPEN); B.update(K=12, n_init=12, horizon=60)
        sweep("s6 добивка вокруг K=12 H=60", [
            ("H=30",              dict(horizon=30)),
            ("K=24 H=60",         dict(K=24, n_init=24)),
            ("+ приор ширины 3",  dict(w_wid=3.0)),
            ("+ w_pred=0.5",      dict(w_pred=0.5)),
        ], base=B)
    elif mode == "best":
        B = dict(CFG_OPEN); B.update(K=12, n_init=12, horizon=60)
        run("beam K=12 H=60 (лучший)", verbose=True, _base=B)
    elif mode == "s2":
        sweep("s2 горизонт (c_coast=40)", [
            ("H=1 (почти жадно)", dict(c_coast=40.0, horizon=1)),
            ("H=8",               dict(c_coast=40.0, horizon=8)),
            ("H=25",              dict(c_coast=40.0, horizon=25)),
            ("H=80",              dict(c_coast=40.0, horizon=80)),
        ])
