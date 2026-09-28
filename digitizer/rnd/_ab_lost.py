r"""_ab_lost.py — ПОТЕРЯННЫЕ В A/B КРИВЫЕ: ЕСТЬ ЛИ ЧЕСТНАЯ ТРАЕКТОРИЯ В КЭШЕ Б, И ЧТО ЗАПИСАНО (§6.244, 28.09).

По дампу `_ab_diff.py --dump` (кривые «потеряна»: честная в выдаче A, нечестная в B) и двум кэшам трасс: у кривой — лучшая
траектория декодера в кэше A и в кэше Б (медиана |Δ|, покрытие с мостом ≤ 30 — как стенд `_dec_joint`), покрытие без моста,
длина; и что записано в слот выдачи B, сопоставленный кривой в A (медиана, покрытие). Отвечает: трасса испорчена ведением или
её не довезла раскладка/выбор версии.

  _ab_lost.py --dump F:/nds/output/taskS/ab_diff_hold.pkl --ca F:/nds/output/taskS/tcache --cb F:/nds/output/taskS/tcache_hold \
      --b F:/nds/output/taskS/rp_ab_hold/N
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--ca", required=True)
ap.add_argument("--cb", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
rows = [r for r in pickle.load(open(a.dump, "rb")) if r[2]["hon"] and not r[3]["hon"]]
if a.n:
    rows = rows[:a.n]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


def un(t):
    return dict(zip(t[0].tolist(), t[1].tolist()))


def stats(tr, gt, bridge=30):
    com = [y for y in gt if y in tr]
    if len(com) < 30:
        return None
    med = float(np.median([abs(tr[y] - gt[y]) for y in com]))
    rs = np.array(sorted(tr)); ok = 0
    for y in gt:
        if y in tr:
            ok += 1; continue
        i = np.searchsorted(rs, y)
        if 0 < i < len(rs) and rs[i] - rs[i - 1] <= bridge:
            ok += 1
    return med, ok / len(gt), len(com) / len(gt), len(tr)


def best(alt, gt):
    b = None
    for t in alt:
        s = stats(t, gt)
        if s and (b is None or s[0] < b[0]):
            b = s
    return b


C = Counter(); ex = []
for sh, g, x, y in rows:
    q = SRC[sh]; stem = q.stem
    key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
    va = pickle.load(open(Path(a.ca) / f"{key}.pkl", "rb")); vb = pickle.load(open(Path(a.cb) / f"{key}.pkl", "rb"))
    G = {c["name"]: dense(c) for c in extract(str(q))["curves"]}
    gt = G[g]
    ba = best([un(t) for _, t in (va["alt"] or [])], gt); bb = best([un(t) for _, t in (vb["alt"] or [])], gt)
    hon_a = ba is not None and ba[0] <= 3 and ba[1] >= 0.9
    hon_b = bb is not None and bb[0] <= 3 and bb[1] >= 0.9
    C[("честная траектория декодера: в A", hon_a, "в B", hon_b)] += 1
    # что записано в слот выдачи B, где в A стояла честная кривая
    d = Path(a.b) / key
    got = next(d.glob("*_auto.nlgx"), None)
    W = {c["name"]: dense(c) for c in extract(str(got))["curves"]} if got else {}
    wb = W.get(x["slot"])
    sw = stats(wb, gt) if wb else None
    if hon_b:
        C[("траектория в B честна, записано в слот", "нет" if sw is None else ("рядом ≤ 3" if sw[0] <= 3 else "мимо"))] += 1
    if bb is not None and not hon_b:
        C[("лучшая траектория B: медиана ≤ 3" if bb[0] <= 3 else "лучшая траектория B: медиана > 3",
           "покрытие (мост 30) ≥ 0.9" if bb[1] >= 0.9 else "покрытие < 0.9")] += 1
    if len(ex) < 12:
        ex.append((sh[:34], g.split()[0], ba and tuple(round(v, 2) for v in ba[:3]), bb and tuple(round(v, 2) for v in bb[:3]),
                   sw and tuple(round(v, 2) for v in sw[:3])))
print(f"★ ПОТЕРЯННЫЕ КРИВЫЕ: {len(rows)}")
for k, v in sorted(C.items(), key=lambda kv: -kv[1]):
    print(f"   {' '.join(str(z) for z in k)}: {v}")
print("   примеры (лист, кривая, лучшая A (мед, покр мост, покр), лучшая B, записано в слот B):")
for e in ex:
    print("     ", e)
