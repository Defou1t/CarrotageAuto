r"""_level_loss.py — ГДЕ ПОСЛЕ ВЫБОРА ТРАССЫ v2 ТЕРЯЮТСЯ КРИВЫЕ НА УРОВНЯХ МАСШТАБА (04.10, к §6.289).

NSL3 (прод с v2, модели фолдов): поле 1368 именных честных по пикселям против 1171 в значениях (−197), сорт A 360 → 307 (−53).
Для каждой кривой выдачи, честной по пикселям под своим именем (на плоскости в обе стороны, покрытие ≥ 0.9): совпадение уровней
масштаба выдачи и эталона по строкам (доля строк, где уровень тот же); кривые с совпадением < 0.9 — «потеря на уровнях».
Разбивка: семейство; уровень с первой строки неверен; эталон весь на 1× (а выдача нет) или выдача вся на 1× (а эталон нет);
число переходов у эталона и у выдачи.

  _level_loss.py --dir F:/nds/output/taskS/rp_lvl/NSL3 --workers 4
"""
import sys, argparse, pickle, hashlib, time, re
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def levels_by_row(c):
    out = {}
    for sg in c.get("segments") or []:
        a, b, l = (sg[0], sg[1], sg[2]) if isinstance(sg, (list, tuple)) else (sg["start"], sg["end"], sg["level"])
        for y in range(int(a), int(b) + 1):
            out[y] = int(l)
    return out


def one(job):
    sh, src, got = job
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree
    from _line_choice_scan import honest_d
    from auto import meta as M
    mt = {c["name"]: c for c in extract(src)["curves"] if M.mnem_root(c["name"]) != "DA"}
    mw = {c["name"]: c for c in extract(got)["curves"] if M.mnem_root(c["name"]) != "DA"}
    out = []
    for n, g in mt.items():
        if n not in mw or sum(1 for x in g["xs"] if x != NULL) < 50 or sum(1 for x in mw[n]["xs"] if x != NULL) < 30:
            continue
        gt = dense(g); w = dense(mw[n])
        if not honest_d(w, _tree(w), gt, _tree(gt))[0]:
            continue
        lg, lw = levels_by_row(g), levels_by_row(mw[n])
        rows = [y for y in gt if y in w]
        if not rows:
            continue
        ag = np.mean([lg.get(y, 0) == lw.get(y, 0) for y in rows])
        y0 = rows[0]
        tg = sum(1 for a_, b_ in zip(sorted(lg), sorted(lg)[1:]) if lg[b_] != lg[a_])
        tw = sum(1 for a_, b_ in zip(sorted(lw), sorted(lw)[1:]) if lw[b_] != lw[a_])
        out.append(dict(sheet=sh, name=n, fam=re.sub(r"^BKZ_", "", M.mnem_root(n)), agree=float(ag),
                        start_ok=lg.get(y0, 0) == lw.get(y0, 0), g_all1=max(lg.values(), default=0) == 0,
                        w_all1=max(lw.values(), default=0) == 0, tg=tg, tw=tw))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_lvl/NSL3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/level_loss_nsl3.pkl")
    a = ap.parse_args()
    import multiprocessing as mp
    from collections import Counter, defaultdict
    TS = Path(a.ts)
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    FIELD, HOLD = set(lst("wellmap_sheets.txt")), set(lst("holdoutA_sheets.txt"))
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for sh in sorted(FIELD | HOLD):
        q = SRC.get(sh)
        if not q:
            continue
        d = Path(a.dir) / f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        got = next(d.glob("*_auto.nlgx"), None) if d.is_dir() else None
        if got:
            jobs.append((sh, str(q), str(got)))
    R = []; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            R += r
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
        rs = [r for r in R if r["sheet"] in S]
        bad = [r for r in rs if r["agree"] < 0.9]
        print(f"\n★ {sn}: честных по пикселям под своим именем {len(rs)}; уровни совпадают < 90% строк у {len(bad)}")
        C = Counter()
        for r in bad:
            if r["g_all1"] and not r["w_all1"]:
                C["эталон весь 1×, у выдачи есть 5×"] += 1
            elif r["w_all1"] and not r["g_all1"]:
                C["выдача вся 1×, у эталона есть 5×"] += 1
            elif not r["start_ok"]:
                C["уровень с первой строки неверен"] += 1
            else:
                C["переходы не там"] += 1
        print("   " + ", ".join(f"{k}: {v}" for k, v in C.most_common()))
        F = Counter(r["fam"] for r in bad)
        print("   по семействам: " + ", ".join(f"{k} {v}" for k, v in F.most_common(12)))
        ag = np.array([r["agree"] for r in bad])
        if len(ag):
            print(f"   доля строк с верным уровнем у них: медиана {np.median(ag):.2f}; < 0.5 у {np.mean(ag < 0.5):.0%}")


if __name__ == "__main__":
    main()
