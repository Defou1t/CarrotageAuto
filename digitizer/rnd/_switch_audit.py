r"""_switch_audit.py — РАЗБОР ПЕРЕХОДОВ МАСШТАБА ВЫДАЧИ ПРОТИВ ЭТАЛОНА (03.11, §6.353, просьба заказчика «перепроверяй переход масштабов»).

По паре «наша кривая — эталон того же имени» с цепочкой масштабов ≥ 2 уровней:
  смены эталона (строка, уровень до → после) против смен выдачи: совпала (сдвиг ≤ 5 / 20 / 50 / 200 строк), пропущена, лишняя смена
  выдачи, смена не на тот уровень; строки с чужим уровнем; потолок «уровни как у эталона» (оракул меток при той же трассе) — мерой
  приёмки и строгой (лишнее ≤ 5%).

  _switch_audit.py --workers 3 [--mode NSE19]  → taskS/_switch_audit_<mode>.json, сводка в stdout, худшие случаи в .tsv
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter
import numpy as np

BUCKETS = (5, 20, 50, 200)


def switches(lv, rows):
    """смены уровня вдоль строк rows → [(y, до, после)]"""
    out, prev = [], None
    for y in rows:
        v = lv.get(y, 0)
        if prev is not None and v != prev:
            out.append((y, prev, v))
        prev = v
    return out


def one(args):
    nm, part, src, got = args
    import _sample_review as S
    import _strict_stand as SS
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception as e:
        return nm, part, [], f"{type(e).__name__}: {e}"
    import decode_levels as DL
    rows = []
    for c in mt["curves"]:
        g = c["name"]
        if g not in gts or g not in wr or not wr[g]:
            continue
        ch = DL.build_family(mt, c)
        if len(ch) < 2:
            continue
        cw = next(x for x in mw["curves"] if x["name"] == g)
        lg, lw = S.lv_rows(c), S.lv_rows(cw)
        gt, w = gts[g], wr[g]
        ty, wy = sorted(gt), sorted(w)
        st_ = switches(lg, ty); sw_ = switches(lw, wy)
        com = [y for y in wy if y in gt]
        mis = frozenset(y for y in com if lw.get(y, 0) != lg.get(y, 0))
        S._TREES.clear()
        wl = S.relevel(w, lw, lg, ch)
        hv = bool(S.HON(*S.err(wl, gt))); sv = SS.strict(S, wl, gt, mis)
        S._TREES.clear()
        hp = bool(S.HON(*S.err(w, gt))); sp = SS.strict(S, w, gt, frozenset())
        ev, used = [], set()
        for y, a, b in st_:
            if not (wy[0] <= y <= wy[-1]):
                ev.append(("вне выдачи", None)); continue
            best = min(((abs(y2 - y), k) for k, (y2, a2, b2) in enumerate(sw_) if (a2, b2) == (a, b) and k not in used), default=None)
            if best is not None and best[0] <= BUCKETS[-1]:
                used.add(best[1]); ev.append(("совпала", best[0]))
            else:
                near = min((abs(y2 - y) for y2, a2, b2 in sw_ if a2 == a and b2 != b), default=None)
                ev.append(("не на тот уровень", None) if near is not None and near <= BUCKETS[-1] else ("пропущена", None))
        extra = [s for k, s in enumerate(sw_) if k not in used and not any(abs(s[0] - y) <= BUCKETS[-1] for y, _, _ in st_)]
        rows.append({"curve": g, "nlev": len(ch), "t_sw": len(st_), "w_sw": len(sw_), "ev": ev, "extra": len(extra),
                     "n_com": len(com), "mis": len(mis), "lvl0_t": all(v == 0 for v in lg.values()),
                     "hv": hv, "hp": hp, "sv": sv and (sv["off_o"] / max(sv["n_o"], 1)), "sp": sp and (sp["off_o"] / max(sp["n_o"], 1)),
                     "lv_set_t": sorted(set(lg.get(y, 0) for y in ty)), "lv_set_w": sorted(set(lw.get(y, 0) for y in wy))})
    return nm, part, rows, ""


def report(res):
    for nmp, parts in (("поле+вне", ("поле", "вне")), ("сорт A", ("сорт A",))):
        R = [r for nm, part, rows, e in res if part in parts for r in rows]
        if not R:
            continue
        ev = Counter(); dd = Counter()
        for r in R:
            for k, d in r["ev"]:
                ev[k] += 1
                if k == "совпала":
                    dd[next(f"≤ {b}" for b in BUCKETS if d <= b)] += 1
        cl = lambda x: x is not None and x is not False and x <= 0.05
        print(f"\n## {nmp}: кривых с цепочкой ≥ 2 уровней {len(R)}; со сменами в эталоне {sum(1 for r in R if r['t_sw'])}, в выдаче "
              f"{sum(1 for r in R if r['w_sw'])}; с чужим уровнем хоть на строке {sum(1 for r in R if r['mis'])}")
        print(f"  смен эталона {sum(ev.values())}: " + ", ".join(f"{k} {v}" for k, v in ev.most_common()))
        print(f"  сдвиг совпавших: " + ", ".join(f"{k} {dd[k]}" for k in (f'≤ {b}' for b in BUCKETS)))
        print(f"  лишних смен выдачи {sum(r['extra'] for r in R)} (у {sum(1 for r in R if r['extra'])} кривых); строк с чужим уровнем "
              f"{sum(r['mis'] for r in R)} из {sum(r['n_com'] for r in R)} ({sum(r['mis'] for r in R) / max(1, sum(r['n_com'] for r in R)):.1%})")
        print(f"  верных приёмкой {sum(r['hv'] for r in R)}; с уровнями эталона (оракул меток) {sum(r['hp'] for r in R)} "
              f"(+{sum(1 for r in R if r['hp'] and not r['hv'])} / −{sum(1 for r in R if r['hv'] and not r['hp'])})")
        print(f"  чистых ≤ 5% {sum(1 for r in R if r['hv'] and cl(r['sv']))}; с уровнями эталона "
              f"{sum(1 for r in R if r['hp'] and cl(r['sp']))}")
        wt = Counter((tuple(r["lv_set_t"]), tuple(r["lv_set_w"])) for r in R if r["mis"] and not r["hv"] and r["hp"])
        print(f"  потеряно из-за уровня (оракул верен, сейчас нет) по наборам уровней эталон → выдача: "
              + ", ".join(f"{a}→{b} {v}" for (a, b), v in wt.most_common(8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mode", default="NSE19")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import _sample_review as S
    S.OOF = S.TS / "rp_lvl" / a.mode
    SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    jobs = []
    for part, SS_ in SETS.items():
        for nm in sorted(SS_):
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is not None and nm in SRC:
                jobs.append((nm, part, SRC[nm], got))
    if a.limit:
        jobs = jobs[::max(1, len(jobs) // a.limit)][:a.limit]
    print(f"★ листов {len(jobs)}, режим {a.mode}, процессов {a.workers}")
    t0 = time.time(); res = []
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            res.append(r)
            if k % 300 == 0:
                print(f"  {k}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    out = S.TS / f"_switch_audit_{a.mode}.json"
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"  ошибок чтения {sum(1 for r in res if r[3])}")
    report(res)
    W = sorted(((nm, part, r) for nm, part, rows, e in res for r in rows if r["mis"] and r["hp"] and not r["hv"]),
               key=lambda t: -t[2]["mis"])
    (S.TS / f"_switch_audit_{a.mode}.tsv").write_text("\n".join(
        "\t".join(map(str, (nm, part, r["curve"], r["mis"], r["n_com"], r["lv_set_t"], r["lv_set_w"], r["t_sw"], r["w_sw"],
                            [k for k, _ in r["ev"]]))) for nm, part, r in W) + "\n", encoding="utf-8")
    print(f"★ {out}")


if __name__ == "__main__":
    main()
