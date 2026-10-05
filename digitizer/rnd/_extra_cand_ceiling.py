r"""_extra_cand_ceiling.py — ДАЮТ ЛИ ТРАССЫ ДРУГОГО ДЕКОДА ЧЕСТНЫХ КАНДИДАТОВ СЛОТАМ, ГДЕ ИХ НЕТ (04.10, к §6.287).

Выбор трассы v2 (§6.283) берёт трассу слота среди всех кандидатов кэша. Значит, пополнение пула кандидатов (пути декодера с
другим порогом пиков, `tcache_pk3`) может дать честную трассу слотам, у которых в `tcache` её нет (27% слотов поля).
Для каждого слота с эталоном (поле + сорт A; скан `_line_choice_scan.py`) без честного кандидата в `tcache`: есть ли честный
среди путей декодера `--extra` (мера — на плоскости в обе стороны по пикселям, покрытие ≥ 0.9). И наоборот — сколько слотов
с честным кандидатом в `tcache` его там имеют (пересечение).

  _extra_cand_ceiling.py --extra F:/nds/output/taskS/tcache_pk3 --workers 4
"""
import sys, argparse, pickle, hashlib, time
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np


def one(job):
    rec_path, ex_path, src, field, base_x = job
    from extract_nlgx import extract
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree, dense_tr
    from _line_choice_scan import honest_d
    from auto import meta as M
    rec = pickle.load(open(rec_path, "rb"))
    if not Path(ex_path).exists():
        return []
    ex = pickle.load(open(ex_path, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    alt = [dense_tr(un(t)) for L, t in (ex.get(field) or [])]
    alt = [a for a in alt if len(a) >= 50]
    bx = []
    for bpath, bfield in (base_x or []):
        if Path(bpath).exists():
            bx += [dense_tr(un(t)) for L, t in (pickle.load(open(bpath, "rb")).get(bfield) or [])]
    bx = [b for b in bx if len(b) >= 50]
    mt = extract(src)
    fc = {cc["name"]: cc for cc in mt["curves"] if M.mnem_root(cc["name"]) != "DA"}
    TA = [_tree(a) for a in alt]
    out = []
    for s in rec["slots"]:
        if not s["has_truth"]:
            continue
        had = any(r["lab"] and r["lab"][0] for r in s["cands"])
        gt = dense(fc[s["name"]]); tg = _tree(gt)
        xl, xr = s["band"]; wd = max(1.0, xr - xl)
        if not had and bx:                      # честный уже есть среди путей базового пополнения пула
            for b in bx:
                u50 = (np.median(list(b.values())) - xl) / wd
                if -0.1 <= u50 <= 1.1 and honest_d(b, _tree(b), gt, tg)[0]:
                    had = True
                    break
        got = False
        for a, ta in zip(alt, TA):
            u50 = (np.median(list(a.values())) - xl) / wd
            if not (-0.1 <= u50 <= 1.1):
                continue
            if honest_d(a, ta, gt, tg)[0]:
                got = True
                break
        out.append(dict(sheet=rec["sheet"], slot=s["name"], root=s["root"], had=had, extra=got))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default=r"F:/nds/output/taskS")
    ap.add_argument("--scan", default=r"F:/nds/output/taskS/line_choice")
    ap.add_argument("--extra", default=r"F:/nds/output/taskS/tcache_pk3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--field", default="alt", help="поле кэша `--extra` с путями: alt (кэш redec с другим порогом) или extra "
                    "(кэш redec с `rowdec_extra_thr`)")
    ap.add_argument("--base-extra", nargs="*", default=[], help="кэши, чьи пути уже в пуле (`каталог|поле`, поле по умолчанию alt; "
                    "напр. tcache_pk3 tcache_x15|extra): слоты с честным кандидатом среди них считаются «уже есть»")
    ap.add_argument("--dump", default=r"F:/nds/output/taskS/extra_cand_ceiling.pkl")
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
        jobs.append((str(f), str(Path(a.extra) / (key + ".pkl")), str(q), a.field,
                     [(str(Path(b.partition("|")[0]) / (key + ".pkl")), b.partition("|")[2] or "alt") for b in a.base_extra]))
    R = []; t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            R += r
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(jobs)} листов, {time.time() - t0:.0f} с")
    pickle.dump(R, open(a.dump, "wb"))
    for sn, S in (("поле", FIELD), ("сорт A", HOLD)):
        rs = [r for r in R if r["sheet"] in S]
        no = [r for r in rs if not r["had"]]; yes = [r for r in rs if r["had"]]
        print(f"★ {sn}: слотов с эталоном {len(rs)}; без честного кандидата в основном кэше {len(no)} — из них честный есть среди путей "
              f"`{Path(a.extra).name}`: {sum(r['extra'] for r in no)}; со честным кандидатом {len(yes)} — у {sum(r['extra'] for r in yes)} он есть и там")


if __name__ == "__main__":
    main()
