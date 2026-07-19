r"""_strat_width.py — СТРАТЕГИЯ «ПРИОР ШИРИНЫ ШТРИХА» (§6.19).

ЧТО ПОКАЗАЛ ЗАМЕР (_strat_width_diag.py, 8050 моментов ИЗБЕЖНОГО ухода со своей кривой):
  ширина рана :  СВОЙ med 10.0 px      ложный (взятый базой) med  2.0 px
  зазор до pred: СВОЙ med  9.0 px      ложный med  1.9 px
  приор ширины (бегущая медиана принятых узких ранов): med 5.0 px

★ Это ОПРОВЕРГАЕТ наивную формулировку стратегии «бери ран, чья ширина ПОХОЖА на приор»:
  симметричный штраф |w - w_med| уводит С кривой (свой ран в момент ухода ШИРЕ приора —
  это пересечение с соседом или спайк пера, ровно та ситуация, где трасса и срывается),
  а ложный кандидат — ТОНКИЙ обрывок, который к приору как раз ближе.
  Симметричное правило спасает 18.8% уходов, «ближайший» (база) 11.1%,
  АСИММЕТРИЧНОЕ (узкий — подозрителен, широкий — свой) до 59.8%.

Отсюда форма стоимости:   cost = gap(pred,run) + bn*max(0, w_prior - w) - rw*min(w, cap)
  gap — зазор до предсказания (не расстояние центров: у широкого рана центр врёт);
  bn  — штраф за то, что ран ТОНЬШЕ приора (обрывок/шум);
  rw  — награда за ширину (пересечение/спайк своей кривой), с ПОТОЛКОМ cap:
        ран в 200px — это рамка или горизонтальная линовка, а не штрих.

Запуск:  python _strat_width.py [--round 1|2|3]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
import numpy as np
import _relatch_bench as BE

SLMAX = BE.SLMAX
WIDE = BE.WIDE_RUN


def _prior(st, win, wide_only_excl=True):
    """Бегущая медиана ширин принятых ранов. Считается ИНКРЕМЕНТАЛЬНО: st['widths']
    ведёт сам стенд, мы лишь дочитываем хвост (иначе O(n) на каждую строку)."""
    h = st.setdefault("_h", [])
    i = st.get("_i", 0)
    W = st["widths"]
    if i < len(W):
        for w in W[i:]:
            if (not wide_only_excl) or w < WIDE:
                h.append(w)
        st["_i"] = len(W)
    if not h:
        return None
    return float(np.median(h[-win:] if len(h) > win else h))


def make_chooser(bn=6.0, rw=5.0, cap=40.0, win=64, maxcost=None, incl_wide=False):
    """Подменяет ТОЛЬКО ветку else. maxcost=None — никогда не коастим."""
    def chooser(A, B, Cc, pred, x, v, st):
        p = _prior(st, win, not incl_wide)
        if p is None:
            return int(np.argmin(np.abs(Cc - pred)))
        w = (B - A).astype(np.float64)
        gap = np.maximum(np.maximum(A - pred, pred - B), 0.0).astype(np.float64)
        cost = gap + bn * np.maximum(0.0, p - w) - rw * np.minimum(w, cap)
        k = int(np.argmin(cost))
        if maxcost is not None and gap[k] > maxcost:
            return None
        return k
    return chooser


def make_tracer(bn=6.0, rw=5.0, cap=40.0, win=64, jump=None, decay=0.7,
                write_coast=True, coast_max=30, wide_mode="edge", capmul=None,
                slmax=SLMAX, wide_run=WIDE):
    """Копия BE.trace + асимметричный приор ширины в else.

    jump       — потолок зазора; выше него прыжок не делаем, а КОАСТИМ с затуханием;
    wide_mode  — что брать на ШИРОКОМ ране, на который мы прыгнули в else:
                 'edge' (как в базе: дальний край от базлайна — вершина спайка),
                 'clip' (ближайшая к pred точка внутри рана — держим непрерывность:
                 если ран широк из-за ПЕРЕСЕЧЕНИЯ, вершина спайка — чужая точка).
    """
    def tracer(rec, csr, H):
        lo, hi, base = rec["lo"], rec["hi"], rec["base"]
        x = None; v = 0.0; tr = {}; hist = []; coast = 0
        for y in range(max(0, rec["y0"]), min(H, rec["y1"] + 1)):
            A, B, Cc = BE._runs_at(csr, y)
            if not len(A):
                if x is not None:
                    v *= decay
                    x = x + float(np.clip(v, -slmax, slmax))
                    coast += 1
                    if write_coast:
                        tr[y] = float(x)
                continue
            if x is None:
                k = int(np.argmin(np.abs(Cc - base)))
                x = float(Cc[k]); v = 0.0; tr[y] = x
                if int(B[k] - A[k]) < wide_run:
                    hist.append(int(B[k] - A[k]))
                continue
            pred = x + float(np.clip(v, -slmax, slmax))
            cont = np.nonzero((A - 2 <= pred) & (pred <= B + 2))[0]
            wide_jump = False
            if len(cont):
                k = int(cont[np.argmin(np.abs(Cc[cont] - pred))]); coast = 0
            else:
                p = float(np.median(hist[-win:])) if hist else None
                w = (B - A).astype(np.float64)
                gap = np.maximum(np.maximum(A - pred, pred - B), 0.0).astype(np.float64)
                if p is None:
                    k = int(np.argmin(np.abs(Cc - pred)))
                else:
                    cp = (capmul * p) if capmul else cap   # потолок награды: абс. или ОТ ПРИОРА
                    cost = gap + bn * np.maximum(0.0, p - w) - rw * np.minimum(w, cp)
                    k = int(np.argmin(cost))
                lim = 1e9 if (jump is None or coast >= coast_max) else jump
                if gap[k] > lim:
                    v *= decay; x = pred; coast += 1
                    if write_coast:
                        tr[y] = float(x)
                    continue
                coast = 0; wide_jump = True
            a, bb, c = int(A[k]), int(B[k]), float(Cc[k])
            if (bb - a) >= wide_run:
                if wide_mode == "clipall" or (wide_jump and wide_mode == "clip"):
                    nx = float(min(max(pred, a), bb))
                else:
                    nx = float(bb if abs(bb - base) >= abs(a - base) else a)
            else:
                nx = c
            v = 0.6 * v + 0.4 * (nx - x); x = float(nx); tr[y] = float(nx)
            if (bb - a) < wide_run:
                hist.append(bb - a)
        BE._extend_ends(tr, csr, H, slmax)
        return tr
    return tracer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=2)
    a = ap.parse_args()
    res = [BE.report("baseline (контроль)", BE.run_strategy())]

    if a.round == 2:
        for nm, P in [
            ("A bn6 rw5 cap40",          dict(bn=6, rw=5, cap=40)),
            ("B bn6 rw3 cap40",          dict(bn=6, rw=3, cap=40)),
            ("C bn3 rw2 cap20",          dict(bn=3, rw=2, cap=20)),
            ("D bn6 rw0 (только штраф)", dict(bn=6, rw=0, cap=40)),
            ("E bn0 rw5 (только награда)", dict(bn=0, rw=5, cap=40)),
            ("F bn6 rw5 cap20",          dict(bn=6, rw=5, cap=20)),
            ("G bn6 rw5 cap40 win=16",   dict(bn=6, rw=5, cap=40, win=16)),
        ]:
            res.append(BE.report(nm, BE.run_strategy(chooser=make_chooser(**P))))

    if a.round == 3:
        for nm, P in [
            ("T0 tracer=chooser F (контроль)", dict(bn=6, rw=5, cap=20, jump=None)),
            ("T1 F + clip на широком",         dict(bn=6, rw=5, cap=20, wide_mode="clip")),
            ("T2 F + jump60 коаст",            dict(bn=6, rw=5, cap=20, jump=60)),
            ("T3 F + jump30 коаст",            dict(bn=6, rw=5, cap=20, jump=30)),
            ("T4 F + jump30 + clip",           dict(bn=6, rw=5, cap=20, jump=30, wide_mode="clip")),
            ("T5 F + jump15 + clip",           dict(bn=6, rw=5, cap=20, jump=15, wide_mode="clip")),
            ("T6 T4 без записи коаста",        dict(bn=6, rw=5, cap=20, jump=30,
                                                   wide_mode="clip", write_coast=False)),
        ]:
            res.append(BE.report(nm, BE.run_strategy(tracer=make_tracer(**P))))

    if a.round == 4:      # сужение вокруг T1
        B0 = dict(bn=6, rw=5, cap=20, wide_mode="clip")
        for nm, P in [
            ("T1 опорный",          B0),
            ("clipall",             {**B0, "wide_mode": "clipall"}),
            ("decay 0.5",           {**B0, "decay": 0.5}),
            ("decay 0.9",           {**B0, "decay": 0.9}),
            ("decay 1.0 (без затух)", {**B0, "decay": 1.0}),
            ("cap 14",              {**B0, "cap": 14}),
            ("cap 30",              {**B0, "cap": 30}),
            ("bn 3",                {**B0, "bn": 3}),
            ("bn 10",               {**B0, "bn": 10}),
            ("rw 3",                {**B0, "rw": 3}),
            ("rw 8",                {**B0, "rw": 8}),
            ("win 16",              {**B0, "win": 16}),
            ("win 256",             {**B0, "win": 256}),
        ]:
            res.append(BE.report(nm, BE.run_strategy(tracer=make_tracer(**P))))

    if a.round == 5:      # комбинации лучших ручек
        for nm, P in [
            ("K1 clipall+rw8",        dict(bn=6, rw=8, cap=20, wide_mode="clipall")),
            ("K2 clipall+rw8+bn10",   dict(bn=10, rw=8, cap=20, wide_mode="clipall")),
            ("K3 clipall+rw12",       dict(bn=10, rw=12, cap=20, wide_mode="clipall")),
            ("K4 clipall+rw20",       dict(bn=10, rw=20, cap=20, wide_mode="clipall")),
            ("K5 K2 cap30",           dict(bn=10, rw=8, cap=30, wide_mode="clipall")),
            ("K6 K2 cap14",           dict(bn=10, rw=8, cap=14, wide_mode="clipall")),
        ]:
            res.append(BE.report(nm, BE.run_strategy(tracer=make_tracer(**P))))

    if a.round == 6:      # покривой разбор победителя
        for nm, P in [("K7 rw50 (почти «самый широкий»)",
                       dict(bn=10, rw=50, cap=20, wide_mode="clipall")),
                      ("K8 rw20 cap30", dict(bn=10, rw=20, cap=30, wide_mode="clipall"))]:
            res.append(BE.report(nm, BE.run_strategy(tracer=make_tracer(**P))))
        best = dict(bn=10, rw=20, cap=20, wide_mode="clipall")
        rw_ = BE.run_strategy(tracer=make_tracer(**best))
        res.append(BE.report("K4 (кандидат-победитель)", rw_))
        print(f"\n{'кривая':<22}{'база med':>10}{'cov':>6}{'|':>3}{'K4 med':>10}{'cov':>6}")
        for b_, f_ in zip(res[0]["rows"], rw_):
            fl = "  ★" if (f_["med"] <= 3 and f_["cov"] >= 0.9) else ""
            if not fl and f_["med"] <= 12:
                fl = "  ~близко"
            print(f"{b_['well'][:9]+' '+b_['curve']:<22}{b_['med']:>10.1f}{b_['cov']:>6.2f}"
                  f"{'|':>3}{f_['med']:>10.1f}{f_['cov']:>6.2f}{fl}")

    if a.round == 7:      # rw50 x режим широкого рана
        for nm, P in [
            ("W1 rw50 clipall", dict(bn=10, rw=50, cap=20, wide_mode="clipall")),
            ("W2 rw50 clip",    dict(bn=10, rw=50, cap=20, wide_mode="clip")),
            ("W3 rw50 edge",    dict(bn=10, rw=50, cap=20, wide_mode="edge")),
            ("W4 rw50 cap30 clipall", dict(bn=10, rw=50, cap=30, wide_mode="clipall")),
            ("W5 rw100 clipall", dict(bn=10, rw=100, cap=20, wide_mode="clipall")),
        ]:
            res.append(BE.report(nm, BE.run_strategy(tracer=make_tracer(**P))))

    if a.round == 10:     # ★ ЧЕСТНАЯ ПРОВЕРКА: а нужен ли ПРИОР вообще?
        for nm, P in [
            ("W5 bn10 rw100 cap20 (приор есть)", dict(bn=10, rw=100, cap=20,
                                                      wide_mode="clipall")),
            ("P1 bn0  rw100 cap20 (БЕЗ приора)", dict(bn=0, rw=100, cap=20,
                                                      wide_mode="clipall")),
            ("P2 cap=3*приор (адаптивный)",      dict(bn=10, rw=100, capmul=3.0,
                                                      wide_mode="clipall")),
            ("P3 cap=4*приор (адаптивный)",      dict(bn=10, rw=100, capmul=4.0,
                                                      wide_mode="clipall")),
            ("P4 cap=2*приор (адаптивный)",      dict(bn=10, rw=100, capmul=2.0,
                                                      wide_mode="clipall")),
        ]:
            res.append(BE.report(nm, BE.run_strategy(tracer=make_tracer(**P))))

    if a.round == 11:     # ★ АБЛЯЦИЯ: чей вклад в выигрыш?
        for nm, P in [
            ("A0 только зазор, без коаста, edge", dict(bn=0, rw=0, wide_mode="edge",
                                                       write_coast=False)),
            ("A1 только КОАСТ (зазор, edge)",     dict(bn=0, rw=0, wide_mode="edge")),
            ("A2 КОАСТ + clipall",                dict(bn=0, rw=0, wide_mode="clipall")),
            ("A3 КОАСТ + ШИРИНА (edge)",          dict(bn=10, rw=100, cap=20,
                                                       wide_mode="edge")),
            ("A4 ШИРИНА без коаста (clipall)",    dict(bn=10, rw=100, cap=20,
                                                       wide_mode="clipall",
                                                       write_coast=False)),
            ("A5 ВСЁ (W5)",                       dict(bn=10, rw=100, cap=20,
                                                       wide_mode="clipall")),
        ]:
            res.append(BE.report(nm, BE.run_strategy(tracer=make_tracer(**P))))

    if a.round == 8:      # покривой разбор финалиста
        best = dict(bn=10, rw=100, cap=20, wide_mode="clipall")
        rw_ = BE.run_strategy(tracer=make_tracer(**best))
        res.append(BE.report("W5 ФИНАЛ bn10 rw100 cap20 clipall", rw_))
        print(f"\n{'кривая':<22}{'база med':>10}{'cov':>6}{'|':>3}{'W5 med':>10}{'cov':>6}")
        for b_, f_ in zip(res[0]["rows"], rw_):
            fl = "  ★" if (f_["med"] <= 3 and f_["cov"] >= 0.9) else (
                 "  ~близко" if f_["med"] <= 12 else "")
            print(f"{b_['well'][:9]+' '+b_['curve']:<22}{b_['med']:>10.1f}{b_['cov']:>6.2f}"
                  f"{'|':>3}{f_['med']:>10.1f}{f_['cov']:>6.2f}{fl}")

    if a.round == 9:      # покривой разбор двух лучших
        rows_b = res[0]["rows"]
        rw_ = BE.run_strategy(chooser=make_chooser(bn=6, rw=5, cap=20))
        res.append(BE.report("F bn6 rw5 cap20", rw_))
        print(f"\n{'кривая':<22}{'база med':>10}{'cov':>6}{'|':>3}{'F med':>10}{'cov':>6}")
        for b_, f_ in zip(rows_b, rw_):
            fl = "  ★" if (f_["med"] <= 3 and f_["cov"] >= 0.9) else (
                 "  +" if f_["med"] < b_["med"] * 0.7 else "")
            print(f"{b_['well'][:9]+' '+b_['curve']:<22}{b_['med']:>10.1f}{b_['cov']:>6.2f}"
                  f"{'|':>3}{f_['med']:>10.1f}{f_['cov']:>6.2f}{fl}")

    print("\n" + "=" * 80)
    print(f"{'стратегия':<34}{'med(med)':>10}{'med(cov)':>10}{'ЧЕСТНЫХ':>10}"
          f"{'своя%':>8}{'латч%':>8}")
    for r in res:
        sv = float(np.mean([q["своя"] for q in r["rows"]]))
        lt = float(np.mean([q["латч"] for q in r["rows"]]))
        print(f"{r['name']:<34}{r['med_med']:>10.1f}{r['med_cov']:>10.2f}"
              f"{r['honest']:>7}/{r['n']}{sv:>8.1f}{lt:>8.1f}")


if __name__ == "__main__":
    main()
