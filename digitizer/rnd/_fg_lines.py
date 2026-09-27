r"""_fg_lines.py — ЧТО ПРОД ДЕЛАЕТ С КРИВЫМИ, КОТОРЫХ НЕ ВИДИТ ЕГО МАСКА (§6.234, 27.09).

По дампу `_fg_why.py` (кривые без честного кандидата, покрытие прод-маской < 0.5) и кэшу трасс: есть ли у кривой линия U1
(полоса линии накрывает медиану x эталона на своих строках), какого она цвета, и где трассы обоих путей относительно эталона
(доля строк эталона в 3 / 10 px от ЛУЧШЕЙ трассы; длина трассы к длине эталона). Отвечает, что чинить: поиск линии (U1),
цвет линии или маску трассировки.

  _fg_lines.py --dump F:/nds/output/taskS/fg_why.pkl
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default=r"F:/nds/output/taskS/fg_why.pkl")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
REC = pickle.load(open(a.dump, "rb"))
SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC.setdefault(q.stem, q)


def unpack(t):
    rows, xs = t
    return dict(zip(rows.tolist(), xs.tolist()))


def near(tr, gt, d):
    com = [y for y in gt if y in tr]
    return sum(1 for y in com if abs(tr[y] - gt[y]) <= d) / max(1, len(gt))


C = Counter(); col = Counter(); lcol = Counter(); fam = defaultdict(Counter)
best3, best10, lens = [], [], []
by_sheet = defaultdict(list)
for r in REC:
    if r["cls"] == "кандидата нет" and r["cp"] < a.thr:
        by_sheet[r["sheet"]].append(r)
for sh, rs in by_sheet.items():
    v = pickle.load(open(Path(a.cache) / f"{sh}.pkl", "rb"))
    q = SRC.get(Path(v["frame_nlgx"]).stem)
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]}
    lines = [(L, unpack(t)) for L, t in v["traces"]]
    alts = [unpack(t) for _, t in v["alt"]] if v.get("alt") else []
    for r in rs:
        gt = G.get(r["g"])
        if not gt:
            continue
        ys = sorted(gt); gx = float(np.median([gt[y] for y in ys])); y0, y1 = ys[0], ys[-1]
        cand = [(L, tr) for L, tr in lines if L.x_lo - 5 <= gx <= L.x_hi + 5 and L.y1 >= y0 and L.y0 <= y1]
        col[r["col"]] += 1
        if not cand:
            C["нет линии U1 в полосе кривой"] += 1; fam[r["root"]]["нет линии"] += 1
        else:
            C["линия U1 есть"] += 1
            for L, _ in cand:
                lcol[(r["col"], L.color)] += 1
        trs = [tr for _, tr in lines] + alts
        b3 = max((near(tr, gt, 3) for tr in trs), default=0.0)
        b10 = max((near(tr, gt, 10) for tr in trs), default=0.0)
        best3.append(b3); best10.append(b10)
        if cand:
            k = max(cand, key=lambda c: near(c[1], gt, 10))
            lens.append(len(k[1]) / max(1, len(gt)))
            fam[r["root"]]["линия есть, лучшая в 10 px ≥ 0.5" if near(k[1], gt, 10) >= 0.5 else "линия есть, трасса мимо"] += 1
n = len(best3)
print(f"★ НЕВИДИМЫЕ ДЛЯ ПРОД-МАСКИ (покрытие < {a.thr}) БЕЗ КАНДИДАТА: {n} кривых на {len(by_sheet)} листах")
print("   " + ", ".join(f"{k} {v} ({100*v/max(1,n):.0f}%)" for k, v in C.items()))
print("   цвет в разметке: " + ", ".join(f"{k} {v}" for k, v in col.most_common()))
print("   (цвет разметки → цвет линии U1): " + ", ".join(f"{k[0]}→{k[1]} {v}" for k, v in lcol.most_common(8)))
b3, b10 = np.array(best3), np.array(best10)
print(f"   лучшая трасса любого пути: доля строк эталона в 3 px — медиана {np.median(b3):.2f}; в 10 px — {np.median(b10):.2f}; "
      f"≥ 0.5 строк в 10 px у {100*np.mean(b10 >= 0.5):.0f}%, < 0.1 у {100*np.mean(b10 < 0.1):.0f}%")
if lens:
    print(f"   длина трассы ближайшей линии U1 к длине эталона: медиана {np.median(lens):.2f}")
print("   по семействам: " + "; ".join(f"{k}: " + ", ".join(f"{a2} {b}" for a2, b in c.items())
                                   for k, c in sorted(fam.items(), key=lambda kv: -sum(kv[1].values()))[:8]))
