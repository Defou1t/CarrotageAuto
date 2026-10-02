r"""_ink_oracle.py — ПОТОЛОК «ИДЕАЛЬНОГО ВЕДЕНИЯ ПО ТУШИ» ПРИ НЫНЕШНЕЙ МЕТРИКЕ (02.10, к §6.257).

§6.257: кривые без честной трассы вдвое круче, карта декодера на крутых строках видит лишь половину. Но эталон — вершины
полилинии с линейной интерполяцией (шаг 6–23 строки), и на частоколе крутых штрихов интерполяция может лежать не на туши.
Тогда кривую не проведёт честно НИКАКОЙ трассировщик, идущий по туши. Оракул: в каждой строке эталона берётся ран туши
(«темнее бумаги своей строки», без структуры — вход декодера `rowdec._band`), ближайший к x эталона в ±R px, и его центр.
Сколько кривых такой оракул проводит честно (медиана ≤ 3 px, покрытие ≥ 0.9 с мостом 30 — как `hon` стендов), и то же при
допуске толщины max(3, 0.35·ширина рана) — для решения заказчика о допуске. Разрез: кривые с честным кандидатом в кэше и без.

  _ink_oracle.py --every 9
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, imaging as im, rowdec as RD
from auto.config import DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--every", type=int, default=9)
ap.add_argument("--radius", type=int, default=12)
ap.add_argument("--thr", type=int, default=20, help="тушь: темнее бумаги своей строки на столько")
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts); P = DEFAULT.cv
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def cover(tr, gt, bridge=30):
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
            ok += 1
    return ok / max(1, len(gt))


def hon(tr, gt, tol=None):
    """tol: None — 3 px; иначе dict y → допуск строки (max(3, 0.35·ширина рана))"""
    com = [y for y in gt if y in tr]
    if len(com) < 30:
        return False
    if tol is None:
        ok = np.median([abs(tr[y] - gt[y]) for y in com]) <= 3.0
    else:
        ok = np.median([abs(tr[y] - gt[y]) / tol[y] for y in com]) <= 1.0
    return bool(ok) and cover(tr, gt) >= 0.9


C = {k: Counter() for k in SETS}
ROWS = []
for sn, names in SETS.items():
    for sh in names[::a.every]:
        q = SRC.get(sh)
        if not q:
            continue
        stem = q.stem; key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        f = Path(a.cache) / f"{key}.pkl"
        if not f.exists():
            continue
        v = pickle.load(open(f, "rb"))
        G = {c["name"]: dense(c) for c in extract(str(q))["curves"]
             if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        if not G:
            continue
        T = [un(t) for _, t in v["traces"]] + ([un(t) for _, t in v["alt"]] if v.get("alt") else [])
        rgb = im.load_rgb(v["image"])
        dark = RD._band(rgb, P, 0, rgb.shape[1])
        H, W = dark.shape
        for g, gt in G.items():
            cand = any(hon(t, gt) for t in T)
            orc, tol = {}, {}
            ys = sorted(gt)
            steep = 0
            for y in ys:
                if not (0 <= y < H):
                    continue
                xt = gt[y]; c = int(round(xt))
                lo, hi = max(0, c - a.radius), min(W, c + a.radius + 1)
                if hi - lo < 3:
                    continue
                m = dark[y, lo:hi] >= a.thr
                if not m.any():
                    continue
                # раны туши в окне; берётся ближайший к x эталона (0, если содержит)
                d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
                st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)       # [st, en)
                dist = np.where((st + lo <= xt) & (xt < en + lo), 0.0, np.minimum(np.abs(st + lo - xt), np.abs(en - 1 + lo - xt)))
                i = int(np.argmin(dist))
                orc[y] = (st[i] + en[i] - 1) / 2.0 + lo
                tol[y] = max(3.0, 0.35 * (en[i] - st[i]))
                sl = abs(gt.get(y + 1, xt) - gt.get(y - 1, xt)) / 2.0
                steep += sl > 2.0
            ho = hon(orc, gt); ht = hon(orc, gt, tol) if orc else False
            cls = "с кандидатом" if cand else "без кандидата"
            C[sn][(cls, "кривых")] += 1
            C[sn][(cls, "оракул честен")] += ho
            C[sn][(cls, "оракул честен с допуском толщины")] += ht
            ROWS.append(dict(set=sn, sheet=sh, name=g, cand=cand, ho=ho, ht=ht, steep=steep / max(1, len(ys)),
                             cov=cover(orc, gt) if orc else 0.0))
        del rgb, dark
    print(f"  {sn}: готово", file=sys.stderr)
if a.dump:
    pickle.dump(ROWS, open(a.dump, "wb"))
for sn in SETS:
    c = C[sn]
    print(f"★ {sn} (каждый {a.every}-й лист):")
    for cls in ("с кандидатом", "без кандидата"):
        n = c[(cls, "кривых")]
        if n:
            print(f"   {cls}: кривых {n}; оракул по туши честен {c[(cls, 'оракул честен')]} ({100 * c[(cls, 'оракул честен')] / n:.0f}%), "
                  f"с допуском толщины {c[(cls, 'оракул честен с допуском толщины')]} "
                  f"({100 * c[(cls, 'оракул честен с допуском толщины')] / n:.0f}%)")
    R = [r for r in ROWS if r["set"] == sn and not r["cand"]]
    if R:
        for lo_, hi_ in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)):
            rr = [r for r in R if lo_ <= r["steep"] < hi_]
            if rr:
                print(f"   без кандидата, крутых строк {lo_:.1f}–{min(hi_, 1):.1f}: {len(rr)}; оракул честен "
                      f"{np.mean([r['ho'] for r in rr]):.0%}, с допуском {np.mean([r['ht'] for r in rr]):.0%}; "
                      f"покрытие оракула — медиана {np.median([r['cov'] for r in rr]):.2f}")
