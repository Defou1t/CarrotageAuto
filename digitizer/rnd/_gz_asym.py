r"""_gz_asym.py — АСИММЕТРИЯ ПИКОВ: ПОДОШВЕННЫЙ ГРАДИЕНТ-ЗОНД (A…M…N) ПРОТИВ КРОВЕЛЬНОГО (N…M…A, OGZ) (03.10, к §6.260).

Физика БКЗ: у подошвенного градиент-зонда (A2.0M0.5N и прочие GZ, парные электроды ниже непарного) максимум кажущегося
сопротивления — у ПОДОШВЫ высокоомного пласта: кривая растёт по пласту плавно и падает у подошвы резко. У кровельного
(обращённого, N0.5M2.0A = OGZ) — зеркально: резкий подъём у кровли, плавный спад. §6.215 закрыл признаки формы для ПОРЯДКА
GZ1…GZ5 (wiggle), но асимметрию пиков не проверял, а на поле OGZ↔GZ — 48 перепутанных имён (§6.260).
Здесь — на ЭТАЛОНЕ (поле + сорт A): для каждой кривой пики x (вправо = выше сопротивление), у каждого заметного пика длина
подъёма (строк от предыдущего минимума) и спада (до следующего); признак — медиана log(подъём/спад). GZ → > 0, OGZ → < 0?
AUC «OGZ против GZ»; разрез по треку, где есть пара GZ и OGZ, — доля треков, где признак ставит их в верном порядке.

  _gz_asym.py --prom 20
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--prom", type=float, default=20.0, help="заметный пик: возвышение над обоими соседними минимумами, px")
ap.add_argument("--smooth", type=int, default=5)
ap.add_argument("--dump", default="")
ap.add_argument("--feat", default="len", choices=["len", "slope"], help="len — log(длина подъёма/спада); slope — log(макс. крутизна спада / подъёма)")
a = ap.parse_args()
FEAT = a.feat
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
smap = pickle.load(open(TS / "slotmap.pkl", "rb"))
for root in ["pools", "pools_gate", "pools_wide", "pools_more", "pools_div", "pools_heldout", "pools_all"]:
    for f in sorted((TS / root).glob("*.pkl")):
        if f.stem in smap:
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        smap.setdefault(d["name"], {s["name"]: s["track"] for s in d["slots"]})


def kind(name):
    r = M.mnem_root(name)
    r = r[4:] if r.startswith("BKZ_") else r
    return "OGZ" if r == "OGZ" else ("GZ" if r == "GZ" else None)


def asym(tr, prom, sm):
    """медиана log(подъём/спад) по заметным пикам; None — пиков мало"""
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys], float)
    if len(xs) < 200:
        return None, 0
    if sm > 1:
        k = np.ones(sm) / sm
        xs = np.convolve(xs, k, mode="same")
    # экстремумы: смена знака производной
    d = np.diff(xs)
    sgn = np.sign(d)
    for i in range(1, len(sgn)):                          # плато наследует знак
        if sgn[i] == 0:
            sgn[i] = sgn[i - 1]
    ext = np.flatnonzero(np.diff(sgn) != 0) + 1
    mx = [i for i in ext if sgn[i - 1] > 0]
    mn = [i for i in ext if sgn[i - 1] < 0]
    if not mx or not mn:
        return None, 0
    mn = np.array(mn)
    vals = []
    for p in mx:
        lo = mn[mn < p]; hi = mn[mn > p]
        if not len(lo) or not len(hi):
            continue
        a0, b0 = lo[-1], hi[0]
        if xs[p] - xs[a0] < prom or xs[p] - xs[b0] < prom:
            continue
        rise = ys[p] - ys[a0]; fall = ys[b0] - ys[p]
        if rise <= 0 or fall <= 0:
            continue
        if FEAT == "slope":
            sr = np.max(np.abs(np.diff(xs[a0:p + 1])) / np.maximum(1, np.diff(ys[a0:p + 1])))
            sf = np.max(np.abs(np.diff(xs[p:b0 + 1])) / np.maximum(1, np.diff(ys[p:b0 + 1])))
            if sr <= 0 or sf <= 0:
                continue
            vals.append(np.log(sf / sr))                 # подошвенный: спад круче → > 0, как и для «len»
        else:
            vals.append(np.log(rise / fall))
    if len(vals) < 3:
        return None, len(vals)
    return float(np.median(vals)), len(vals)


R = []
for sn, f in (("поле", "wellmap_sheets.txt"), ("сорт A", "holdoutA_sheets.txt")):
    for sh in lst(f):
        q = SRC.get(sh)
        if not q:
            continue
        tm = smap.get(sh, {})
        for c in extract(str(q))["curves"]:
            k = kind(c["name"])
            if k is None or sum(1 for x in c["xs"] if x != NULL) < 200:
                continue
            v, n = asym(dense(c), a.prom, a.smooth)
            if v is None:
                continue
            R.append(dict(set=sn, sheet=sh, name=c["name"], kind=k, asym=v, n=n, track=tm.get(c["name"])))
if a.dump:
    pickle.dump(R, open(a.dump, "wb"))


def auc(sc, y):
    sc = np.asarray(sc, float); y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(sc)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


for sn in ("поле", "сорт A"):
    G = [r for r in R if r["set"] == sn]
    g = np.array([r["asym"] for r in G if r["kind"] == "GZ"]); o = np.array([r["asym"] for r in G if r["kind"] == "OGZ"])
    print(f"★ {sn}: GZ {len(g)} кривых — медиана {np.median(g):+.3f} (> 0 у {np.mean(g > 0):.0%}); "
          f"OGZ {len(o)} — медиана {np.median(o):+.3f} (< 0 у {np.mean(o < 0):.0%}); "
          f"AUC «OGZ: признак ниже» {auc([-r['asym'] for r in G], [r['kind'] == 'OGZ' for r in G]):.3f}")
    # пары в треке: есть OGZ и хотя бы один GZ — верно ли OGZ ниже всех GZ трека
    from collections import defaultdict
    T = defaultdict(list)
    for r in G:
        if r["track"] is not None:
            T[(r["sheet"], r["track"])].append(r)
    ok = n = 0; okp = np_ = 0
    for k, rs in T.items():
        os_ = [r for r in rs if r["kind"] == "OGZ"]; gs = [r for r in rs if r["kind"] == "GZ"]
        if os_ and gs:
            n += 1; ok += all(o_["asym"] < min(x["asym"] for x in gs) for o_ in os_)
            for o_ in os_:
                for x in gs:
                    np_ += 1; okp += o_["asym"] < x["asym"]
    if n:
        print(f"   треков с OGZ и GZ: {n}; OGZ ниже всех GZ трека — {ok} ({100 * ok / n:.0f}%); парно {okp} из {np_} ({100 * okp / max(1, np_):.0f}%)")
