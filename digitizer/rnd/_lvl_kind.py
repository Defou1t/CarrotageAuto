r"""_lvl_kind.py — ИЗ ЧЕГО СОСТОИТ ЛИШНЕЕ УРОВНЯ (08.11, §6.354): у кривых с цепочкой масштабов лишние строки уровня (наш уровень ≠
уровню эксперта и после перевода в его шкалу строка дальше 10 px) — 62% всех лишних строк этих кривых (`_cut_sit.py`, 200 листов).

Разбор строк с несовпавшим уровнем: (a) ТА ЖЕ ТУШЬ — наша трасса в пикселях ≤ 10 px от линии эксперта (перо то же, ошибка только
метки уровня → правка модели уровня); (b) ДРУГАЯ ТУШЬ — трасса на другом пере / линии. Пары уровней (наш → эксперт), длины серий,
положение в полосе шкалы (доля ширины от края уровня 0), в т. ч. у нуля центрированных цепочек.

  _lvl_kind.py [--limit 200] [--workers 1]
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np


def one(args):
    nm, part, src, got = args
    import _sample_review as S, decode_levels as DL
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception as e:
        return nm, part, None, f"{type(e).__name__}"
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    C = Counter(); runlen = []; pos = {"a": [], "b": []}; zero = Counter(); TY = Counter()
    ORC = {k: np.zeros(2, int) for k in ("метка", "перо 5×→1×", "перо 1×→5×", "всё")}
    for c in mt["curves"]:
        g = c["name"]
        if g not in gts or g not in wr:
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2:
            continue
        w, gt, lw, lg = wr[g], gts[g], LW.get(g, {}), LG.get(g, {})
        wl = S.relevel(w, lw, lg, ch)
        _r = abs(ch[1]["v_right"] - ch[1]["v_left"]) / (abs(ch[0]["v_right"] - ch[0]["v_left"]) or 1)
        typ = "сдвиг" if _r < 1.5 else "перья"
        s0 = ch[0]; x0, x1 = min(s0["x_left"], s0["x_right"]), max(s0["x_left"], s0["x_right"])
        # центрированная цепочка: ноль значения внутри шкалы уровня 0
        v0, v1 = s0["v_left"], s0["v_right"]
        zx = s0["x_left"] + (0 - v0) * (s0["x_right"] - s0["x_left"]) / ((v1 - v0) or 1) if min(v0, v1) < 0 < max(v0, v1) else None
        cur = None; ln = 0
        for y in sorted(w):
            if y not in gt:
                continue
            lo, lt = lw.get(y, 0), lg.get(y, 0)
            if lo == lt or abs(wl[y] - gt[y]) <= 10:
                if ln:
                    runlen.append(ln); ln = 0
                continue
            ln += 1
            k = "a" if abs(w[y] - gt[y]) <= 10 else "b"
            TY[(typ, k)] += 1
            C[(k, lo, lt)] += 1
            pos[k].append((w[y] - x0) / ((x1 - x0) or 1))
            if zx is not None:
                zero[(k, abs(w[y] - zx) <= 0.1 * (x1 - x0))] += 1
        if ln:
            runlen.append(ln)
        # оракулы по классам строк: метка (та же тушь — уровень эксперта), перо 1→0 / 0→1 (другая тушь — x и уровень эксперта)
        cls = {}
        for y in w:
            if y in gt and lw.get(y, 0) != lg.get(y, 0) and abs(wl[y] - gt[y]) > 10:
                cls[y] = ("a" if abs(w[y] - gt[y]) <= 10 else "b", lw.get(y, 0), lg.get(y, 0))
        def meas(w_, lw_):
            S._TREES.clear()
            return bool(S.HON(*S.err(S.relevel(w_, lw_, lg, ch), gt)))
        ORC_last = meas
        r0 = meas(w, lw)
        for nm_, sel in (("метка", lambda t: t[0] == "a"), ("перо 5×→1×", lambda t: t[0] == "b" and t[1] > t[2]),
                         ("перо 1×→5×", lambda t: t[0] == "b" and t[1] < t[2]), ("всё", lambda t: True)):
            ys = [y for y, t in cls.items() if sel(t)]
            if not ys:
                ORC[nm_] += [r0, r0]; continue
            w2, lw2 = dict(w), dict(lw)
            for y in ys:
                lw2[y] = lg.get(y, 0)
                if cls[y][0] == "b":
                    w2[y] = gt[y]
            ORC[nm_] += [r0, meas(w2, lw2)]
            ORC.setdefault(f"{typ}: {nm_}", np.zeros(2, int)); ORC[f"{typ}: {nm_}"] += [r0, ORC_last(w2, lw2) if ys else r0]
    return nm, part, (dict((f"{k}|{a}|{b}", v) for (k, a, b), v in C.items()), runlen, pos, {f"{k}|{z}": v for (k, z), v in zero.items()},
                      {k: v.tolist() for k, v in ORC.items()}, {f"{t}|{k}": v for (t, k), v in TY.items()}), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mode", default="NSE19")
    a = ap.parse_args()
    import _sample_review as S
    S.OOF = S.TS / "rp_lvl" / a.mode
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = [(nm, part, SRC[nm], next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None))
            for part, SS in SETS.items() if part != "сорт A" for nm in sorted(SS) if nm in SRC]
    jobs = [j for j in jobs if j[3] is not None]
    rng = np.random.default_rng(11)
    if a.limit:
        jobs = [jobs[i] for i in sorted(rng.permutation(len(jobs))[:a.limit])]
    from multiprocessing import Pool
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=2):
            res.append(r)
    C = Counter(); RL = []; P = {"a": [], "b": []}; Z = Counter()
    for _, _, r, _ in res:
        if not r:
            continue
        C.update(r[0]); RL += r[1]; P["a"] += r[2]["a"]; P["b"] += r[2]["b"]; Z.update(r[3])
    tot = sum(C.values()) or 1
    ta = sum(v for k, v in C.items() if k.startswith("a|")); tb = tot - ta
    print(f"★ листов {len(jobs)}; строк с несовпавшим уровнем и лишних {tot}: ТА ЖЕ ТУШЬ (ошибка метки) {ta} ({ta / tot:.0%}), "
          f"ДРУГАЯ ТУШЬ {tb} ({tb / tot:.0%})")
    for k, v in C.most_common(12):
        kind, lo, lt = k.split("|")
        print(f"  {'та же тушь' if kind == 'a' else 'другая тушь'}: наш уровень {lo} → у эксперта {lt}: {v} ({v / tot:.0%})")
    RL = np.array(RL)
    if len(RL):
        print(f"★ серии несовпадения: {len(RL)}; длина квартили {np.percentile(RL, [25, 50, 75, 90]).round(0)}; строк в сериях ≥ 500 — "
              f"{RL[RL >= 500].sum() / RL.sum():.0%}, ≥ 2000 — {RL[RL >= 2000].sum() / RL.sum():.0%}")
    for k in "ab":
        u = np.array(P[k])
        if len(u):
            print(f"★ положение в полосе уровня 0 ({'та же' if k == 'a' else 'другая'} тушь): квартили {np.percentile(u, [10, 25, 50, 75, 90]).round(2)}")
    O = {}
    for _, _, r, _ in res:
        if r:
            for k, v in r[4].items():
                O[k] = O.get(k, np.zeros(2, int)) + np.array(v)
    for k, v in sorted(O.items()):
        print(f"★ оракул «{k}»: верных приёмкой {v[0]} → {v[1]} ({v[1] - v[0]:+d})")
    T_ = Counter()
    for _, _, r, _ in res:
        if r and len(r) > 5:
            T_.update(r[5])
    print("★ строки лишнего уровня по типу цепочки (сдвиг — ширина уровня 1 ≈ уровня 0; перья — ×5 / ×10):", dict(T_))
    za, zb = Z.get("a|True", 0), Z.get("b|True", 0)
    print(f"★ у нуля центрированной цепочки (± 10% ширины): та же тушь {za} из {Z.get('a|True', 0) + Z.get('a|False', 0)}, "
          f"другая {zb} из {Z.get('b|True', 0) + Z.get('b|False', 0)}")


if __name__ == "__main__":
    main()
