r"""_stitch_ceiling.py — ПОТОЛОК СКЛЕЙКИ ТРАССЫ ИЗ КУСКОВ КАНДИДАТОВ ДЛЯ СЛОТОВ БЕЗ ЧЕСТНОГО КАНДИДАТА (04.10, к §6.284).

У 27% слотов поля ни один кандидат кэша (прод-путь `traces`, декодер `alt`) не честен целиком (`_line_choice_miss.py`), и кусков,
честных «где есть», почти нет (медиана по всему кандидату > 3 px). Вопрос: идут ли по эталону КУСКИ разных кандидатов, так что
склейка по строкам дала бы честную трассу. Для каждого такого слота (поле + сорт A):
- покрытие строк эталона, где хоть один кандидат в ≤ 3 px по строке (нижняя оценка: по строке строже, чем на плоскости);
- сколько переключений между кандидатами нужно (жадно: держим кандидата, пока он в ≤ 3 px; иначе — ближайший);
- длина кусков.

  _stitch_ceiling.py --workers 6
"""
import sys, argparse, pickle, hashlib, time
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def one(job):
    rec_path, tc_path, src = job
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import dense_tr
    from auto import meta as M
    rec = pickle.load(open(rec_path, "rb"))
    todo = [s for s in rec["slots"] if s["has_truth"] and not any(r["lab"] and r["lab"][0] for r in s["cands"])]
    if not todo:
        return []
    c = pickle.load(open(tc_path, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    raw = [dense_tr(un(t)) for L, t in (c["traces"] or [])] + [dense_tr(un(t)) for L, t in (c["alt"] or [])]
    raw = [r for r in raw if len(r) >= 50]
    mt = extract(src)
    fc = {cc["name"]: cc for cc in mt["curves"] if M.mnem_root(cc["name"]) != "DA"}
    out = []
    for s in todo:
        gt = dense(fc[s["name"]])
        ys = np.array(sorted(gt)); gx = np.array([gt[y] for y in ys])
        ks = sorted({r["k"] for r in s["cands"]})          # кандидаты в полосе слота
        if not ks:
            out.append(dict(sheet=rec["sheet"], slot=s["name"], root=s["root"], cover=0.0, sw=0, n=len(ys), nk=0))
            continue
        D = np.full((len(ks), len(ys)), np.inf)
        for j, k in enumerate(ks):
            ct = raw[k]
            D[j] = [abs(ct[y] - x) if y in ct else np.inf for y, x in zip(ys, gx)]
        ok = D <= 3.0
        cover = float(ok.any(0).mean())
        # жадная склейка: держим кандидата, пока он в ≤ 3 px; иначе берём того, кто дальше всех продержится
        sw, cur, i, pieces = 0, -1, 0, []
        while i < len(ys):
            if cur >= 0 and ok[cur, i]:
                i += 1
                continue
            cand = np.flatnonzero(ok[:, i])
            if not len(cand):
                i += 1
                continue
            best, run = -1, -1
            for j in cand:
                r_ = np.argmin(ok[j, i:]) if not ok[j, i:].all() else len(ys) - i
                if r_ > run:
                    best, run = j, r_
            if cur >= 0:
                sw += 1
            cur = best; pieces.append(run)
        out.append(dict(sheet=rec["sheet"], slot=s["name"], root=s["root"], cover=cover, sw=sw, n=len(ys), nk=len(ks),
                        med_piece=float(np.median(pieces)) if pieces else 0.0))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
    ap.add_argument("--scan", default=r"F:/nds/output/taskS/line_choice")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/stitch_ceiling.pkl")
    a = ap.parse_args()
    import multiprocessing as mp
    TS = Path(a.ts)
    lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    FIELD, HOLD = lst("wellmap_sheets.txt"), lst("holdoutA_sheets.txt")
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    jobs = []
    for f in sorted(Path(a.scan).glob("*.pkl")):
        sh = pickle.load(open(f, "rb"))["sheet"]
        q = SRC[sh]
        key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
        jobs.append((str(f), str(Path(a.cache) / (key + ".pkl")), str(q)))
    if a.limit:
        jobs = jobs[:a.limit]
    R = []; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            R += r
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
        rs = [r for r in R if r["sheet"] in S]
        cv = np.array([r["cover"] for r in rs]); sw = np.array([r["sw"] for r in rs])
        print(f"\n★ {sn}: слотов без честного кандидата {len(rs)}")
        for t in (0.9, 0.8, 0.5):
            m = cv >= t
            print(f"   покрытие кусками ≥ {t}: {int(m.sum())} ({m.mean():.0%}); переключений (медиана) {np.median(sw[m]) if m.any() else 0:.0f}")
        import collections
        by = collections.defaultdict(list)
        for r in rs:
            by[r["root"]].append(r["cover"] >= 0.9)
        print("   по семействам (≥ 0.9): " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by.items(), key=lambda x: -len(x[1]))[:10]))


if __name__ == "__main__":
    main()
