"""Потолок «очищенных копий» кандидатов (§6.345 разведка): слоты без честного кандидата (поле + вне, без сорта A) — станет ли честным
(plane2 ≤ 3 px, покрытие ≥ 0.9, в пикселях) хоть один кандидат слота после вычистки длинных экскурсий: точки дальше D px от бегущей
медианы окна W строк выкидываются (строки остаются за соседями — dense). Контроль вреда: честные кандидаты честных слотов — остаются
ли честными после той же вычистки. despike_oracle.py [шаг по листам]"""
import sys, pickle, glob, hashlib
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from _cand_ceiling import dense_tr
src = open(Path(__file__).with_name("lvl_loss.py"), encoding="utf-8").read()
_argv = sys.argv; sys.argv = sys.argv[:1] + ["F:/nds/output/taskS/rp_lvl/__none__", "nul"]
exec(src[src.index("def tree"):src.index("def lv_rows")])           # tree, err
sys.argv = _argv
HON = lambda e: e[0] <= 3.0 and e[1] >= 0.9
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TS = Path("F:/nds/output/taskS")
HLD = {l.strip() for l in (TS / "holdoutA_sheets.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
SRC = {q.name: q for q in Path("F:/nds/projects/Archive").glob("*/wlg/*.nlgx")}
VAR = [(21, 10), (41, 10), (81, 10), (41, 20), (81, 20), (161, 20)]


def despike(d, W, D):
    ys = np.array(sorted(d)); xs = np.array([d[y] for y in ys], float)
    if len(ys) < W:
        return d
    m = median_filter(xs, size=W, mode="nearest")
    ok = np.abs(xs - m) <= D
    if ok.sum() < 10:
        return d
    xi = np.interp(ys, ys[ok], xs[ok])
    return dict(zip(ys.tolist(), xi.tolist()))


NEW = Counter(); LOST = Counter(); NMISS = 0; NHON = 0
files = [(f, "tcache", "tcache_pk3|alt;tcache_inert|extra") for f in sorted(glob.glob(str(TS / "line_choice_i" / "*.pkl")))][::STEP] + \
        [(f, "tcache_out", "tcache_out|extra;tcache_out_inert|extra") for f in sorted(glob.glob(str(TS / "line_choice_out_i" / "*.pkl")))][::STEP]
for n_, (f, cache, extra) in enumerate(files):
    rec = pickle.load(open(f, "rb")); sh = rec["sheet"]
    if sh in HLD or sh not in SRC:
        continue
    q = SRC[sh]
    key = f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"
    tc = TS / cache / (key + ".pkl")
    if not tc.exists():
        continue
    c = pickle.load(open(tc, "rb"))
    un = lambda t: dict(zip(t[0].tolist(), t[1].tolist()))
    xsrc = []
    for spec in extra.split(";"):
        dd, _, fld = spec.partition("|")
        xp = TS / dd / tc.name
        if xp.exists():
            xsrc += pickle.load(open(xp, "rb")).get(fld or "alt") or []
    gts = None; DD = {}

    def cand(k):
        if k not in DD:
            cs = rec["cands"][k]
            t = c["traces"][cs["idx"]][1] if cs["src"] == "prod" else (c["alt"][cs["idx"]][1] if cs["idx"] < 1000 else xsrc[cs["idx"] - 1000][1])
            DD[k] = dense_tr(un(t))
        return DD[k]
    for s in rec["slots"]:
        if not s["has_truth"]:
            continue
        labs = [(sc["k"], sc["lab"]) for sc in s["cands"] if sc["lab"] is not None]
        hon = [k for k, l in labs if l[0]]
        if gts is None:
            gts = {cc["name"]: cc for cc in extract(str(q))["curves"]}
        if s["name"] not in gts:
            continue
        g = dense(gts[s["name"]])
        if not hon:
            NMISS += 1
            pool = [k for k, l in labs if l[1] >= 0.9 and l[2] <= 60]
            for W, D in VAR:
                if any(HON(err(despike(cand(k), W, D), g)) for k in pool):
                    NEW[(W, D)] += 1
        else:
            NHON += 1
            k = hon[0]
            for W, D in VAR:
                if not HON(err(despike(cand(k), W, D), g)):
                    LOST[(W, D)] += 1
    if (n_ + 1) % 50 == 0:
        print(f"  {n_ + 1}/{len(files)}", flush=True)
print(f"★ листов {len(files)}; слотов без честного {NMISS}, с честным {NHON}")
for v in VAR:
    print(f"   W {v[0]:3d}, D {v[1]}: стали честными {NEW[v]} ({NEW[v] / max(1, NMISS):.1%}); честный кандидат перестал быть честным {LOST[v]} ({LOST[v] / max(1, NHON):.1%})")
