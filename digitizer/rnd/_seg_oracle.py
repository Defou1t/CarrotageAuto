r"""_seg_oracle.py — §6.365 РАЗВЕДКА: ПОТОЛОК СБОРКИ КРИВОЙ ИЗ КУСКОВ КАНДИДАТОВ (13.11).

Разбор выборок 10.11 / 13.11: лишняя трасса дубля часто составная — часть по своей линии, часть по соседней (перескок на пересечении);
у части слотов честного ЦЕЛОГО кандидата нет. Если резать кандидатов на пересечениях и собирать кривую из кусков, сколько кривых эталона
станут достижимы? Пул — как у прода (`traces` + `alt` + доп. кэши, как в повторе NSE22/26). Для каждой кривой эталона:
  целый — есть кандидат, честный на плоскости (`_line_choice_scan.honest_d`: медиана ≤ 3 px в обе стороны, покрытие ≥ 0.9);
  куски — доля строк эталона, где хоть один кандидат ≤ 3 px (на плоскости, 13.11 — было по строке: склоны), и минимум смен кандидата, чтобы её набрать (DP);
  «достижим кусками» — покрытие кусками ≥ 0.9 при ≤ K сменах (K = 2, 4, 8).

  _seg_oracle.py --n 150 [--seed 1]   → taskS/seg_oracle/<часть>.jsonl (продолжаемо), сводка
"""
import sys, argparse, json, pickle, hashlib, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np

TS = Path(r"F:/nds/output/taskS")
OUT = TS / "seg_oracle"
EXTRA = {"f": ["tcache_pk3|alt", "tcache_inert|extra", "tcache_seg|alt", "tcache_seg|extra", "tcache_segraw_f|alt", "tcache_segraw_f|extra"],
         "o": ["tcache_out|extra", "tcache_out_inert|extra", "tcache_out_seg|alt", "tcache_out_seg|extra", "tcache_out_segraw|alt",
               "tcache_out_segraw|extra"]}
BASE = {"f": "tcache", "o": "tcache_out"}


def key_of(q):
    return f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"


def cands_of(kind, key):
    from _cand_ceiling import dense_tr
    c = pickle.load(open(TS / BASE[kind] / (key + ".pkl"), "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    C = [dense_tr(un(t)) for _, t in (c["traces"] or [])] + [dense_tr(un(t)) for _, t in (c["alt"] or [])]
    for spec in EXTRA[kind]:
        dd, _, fld = spec.partition("|")
        xp = TS / dd / (key + ".pkl")
        if xp.exists():
            C += [dense_tr(un(t)) for _, t in (pickle.load(open(xp, "rb")).get(fld) or [])]
    return [d for d in C if len(d) >= 50]


def min_switches(M):
    """M — bool [строки × кандидаты] (кандидат ≤ 3 px на строке); строки без кандидата пропускаются. → мин. число смен кандидата"""
    if M.shape[0] == 0:
        return 0
    INF = 10 ** 9
    cost = np.where(M[0], 0, INF).astype(np.int64)
    for r in range(1, M.shape[0]):
        best = cost.min()
        stay = cost
        cost = np.where(M[r], np.minimum(stay, best + 1), INF)
    return int(cost.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from extract_nlgx import extract, NULL
    from _multi_replica_probe import dense
    from _cand_ceiling import _tree
    from _line_choice_scan import honest_d
    from auto import meta as M
    OUT.mkdir(parents=True, exist_ok=True)
    SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    for kind, f in (("f", "tcache_sheets.txt"), ("o", "outside_sheets.txt")):
        L = [s for s in lst(f) if s in SRC and (TS / BASE[kind] / (key_of(SRC[s]) + ".pkl")).exists()]
        random.Random(a.seed).shuffle(L); L = L[:a.n]
        jf = OUT / f"{kind}.jsonl"
        done = {json.loads(l)["sheet"] for l in jf.read_text(encoding="utf-8").splitlines()} if jf.exists() else set()
        for i, nm in enumerate(L):
            if nm in done:
                continue
            q = SRC[nm]; key = key_of(q)
            try:
                C = cands_of(kind, key)
                mt = extract(str(q))
            except Exception as e:
                print(f"  ⚠ {nm}: {type(e).__name__}"); continue
            TC = [_tree(d) for d in C]
            recs = []
            for cc in mt["curves"]:
                if M.mnem_root(cc["name"]) == "DA" or sum(1 for x in cc["xs"] if x != NULL) < 50:
                    continue
                gt = dense(cc)
                if len(gt) < 50:
                    continue
                tg = _tree(gt)
                whole = any(honest_d(C[k], TC[k], gt, tg)[0] for k in range(len(C)))
                ys = np.array(sorted(gt)); gx = np.array([gt[y] for y in ys])
                Mx = np.zeros((len(ys), len(C)), bool)
                pts = np.stack([ys.astype(float), gx], 1)
                for k, d in enumerate(C):                       # расстояние НА ПЛОСКОСТИ до ломаной кандидата (как мера приёмки)
                    dk, _ = TC[k].query(pts, distance_upper_bound=3.01)
                    Mx[:, k] = np.isfinite(dk) & (dk <= 3.0)
                anyc = Mx.any(1)
                cov = float(anyc.mean())
                sw = min_switches(Mx[anyc][::5]) if cov >= 0.9 else None
                recs.append(dict(curve=cc["name"], whole=bool(whole), cov=round(cov, 3), sw=sw, ncand=len(C)))
            with open(jf, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(sheet=nm, curves=recs), ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {kind}: {i + 1}/{len(L)}")
    for kind in ("f", "o"):
        jf = OUT / f"{kind}.jsonl"
        R = [c for l in jf.read_text(encoding="utf-8").splitlines() for c in json.loads(l)["curves"]]
        n = len(R); w = sum(c["whole"] for c in R)
        print(f"★ {'поле' if kind == 'f' else 'вне'}: кривых эталона {n}; честный ЦЕЛЫЙ кандидат есть у {w} ({w / max(n, 1):.1%})")
        for K in (2, 4, 8):
            s = sum(1 for c in R if c["sw"] is not None and c["sw"] <= K)
            sn = sum(1 for c in R if not c["whole"] and c["sw"] is not None and c["sw"] <= K)
            print(f"   кусками (покрытие ≥ 0.9, смен ≤ {K}): {s} ({s / max(n, 1):.1%}); из них без целого — {sn}")


if __name__ == "__main__":
    main()
