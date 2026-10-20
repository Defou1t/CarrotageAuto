"""Потери уровней масштаба на выдаче: кривые, честные в пикселях (plane2), но не в значениях — как именно не совпал уровень.
lvl_loss.py <каталог режима> <out.pkl>"""
import sys, pickle, hashlib, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M, level_ink as LI
import decode_levels as DL

MD = Path(sys.argv[1])
TS = Path(r"F:/nds/output/taskS")
lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
HLD = lst("holdoutA_sheets.txt")
SRC = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    SRC[f"{q.stem[:40]}_{hashlib.md5(q.stem.encode('utf-8')).hexdigest()[:8]}"] = q


def tree(tr):
    ys = np.array(sorted(tr), float); xs = np.array([tr[int(y)] for y in ys], float)
    P = [np.stack([ys, xs], 1)]
    dy = np.diff(ys); dx = np.diff(xs)
    for i in np.flatnonzero((dy <= 30) & (np.abs(dx) > 1) & (np.abs(dx) <= 3000)):
        n = int(np.ceil(abs(dx[i]))); t = np.arange(1, n) / n
        P.append(np.stack([ys[i] + dy[i] * t, xs[i] + dx[i] * t], 1))
    return cKDTree(np.concatenate(P))


def err(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return 99.0, 0.0
    d1, _ = tree(tr).query(np.array([[y, gt[y]] for y in com], float))
    d2, _ = tree(gt).query(np.array([[y, tr[y]] for y in com], float))
    return max(float(np.median(d1)), float(np.median(d2))), len(com) / max(1, len(gt))


def lv_rows(c):
    out = {}
    for y0, y1, lv in c.get("segments") or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


def relevel(tr, lw, lg, chain):
    if len(chain) < 2:
        return tr
    out = {}
    for y, x in tr.items():
        lo, lt = lw.get(y, 0), lg.get(y, 0)
        if lo == lt:
            out[y] = x; continue
        if lo >= len(chain) or lt >= len(chain):
            out[y] = x + 1e4; continue
        so, sg = chain[lo], chain[lt]
        v = so["v_left"] + (x - so["x_left"]) * (so["v_right"] - so["v_left"]) / ((so["x_right"] - so["x_left"]) or 1)
        out[y] = sg["x_left"] + (v - sg["v_left"]) * (sg["x_right"] - sg["x_left"]) / ((sg["v_right"] - sg["v_left"]) or 1)
    return out


HON = lambda e: e[0] <= 3.0 and e[1] >= 0.9
C = Counter(); FAM = Counter(); EX = []; n_px = n_val = 0
for d in sorted(x for x in MD.iterdir() if x.is_dir()):
    q = SRC.get(d.name)
    got = next(iter(sorted(d.glob("*_auto.nlgx"))), None)
    if q is None or got is None:
        continue
    mt = extract(str(q)); mw = extract(got.as_posix())
    gts = {c["name"]: c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
    wr = {c["name"]: c for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
    for n, cg in gts.items():
        if n not in wr:
            continue
        gt = dense(cg); w = dense(wr[n])
        if not w:
            continue
        ep = err(w, gt)
        if not HON(ep):
            continue
        n_px += 1
        ch = DL.build_family(mt, cg)
        lg = lv_rows(cg); lw = lv_rows(wr[n])
        ev = err(relevel(w, lw, lg, ch), gt)
        if HON(ev):
            n_val += 1; continue
        com = [y for y in w if y in gt]
        mis = np.array([lw.get(y, 0) != lg.get(y, 0) for y in com])
        g_lv = Counter(lg.get(y, 0) for y in com); w_lv = Counter(lw.get(y, 0) for y in com)
        shift = LI.is_shift(ch) if len(ch) >= 2 else False
        fr = float(mis.mean())
        if len(g_lv) == 1 and len(w_lv) == 1:
            cat = "оба одного уровня, разного (весь лист)"
        elif len(g_lv) == 1:
            cat = f"эталон одного уровня, выдача переключает (неверно {'<30%' if fr < 0.3 else '30–70%' if fr < 0.7 else '>70%'} строк)"
        elif len(w_lv) == 1:
            cat = f"эталон переключает, выдача одного уровня (неверно {'<30%' if fr < 0.3 else '30–70%' if fr < 0.7 else '>70%'} строк)"
        else:
            cat = f"оба переключают (неверно {'<30%' if fr < 0.3 else '30–70%' if fr < 0.7 else '>70%'} строк)"
        C[(cat, "сдвиг" if shift else "умножение")] += 1
        FAM[re.sub(r"^BKZ_", "", M.mnem_root(n))] += 1
        EX.append(dict(dir=d.name, sheet=q.name, curve=n, cat=cat, shift=shift, frac=fr, g_lv=dict(g_lv), w_lv=dict(w_lv),
                       chain=[(s["v_left"], s["v_right"]) for s in ch], hold=q.name in HLD))
print(f"★ {MD.name}: честных в пикселях (по имени) {n_px}, из них в значениях {n_val}; потеряно на уровнях {n_px - n_val}")
for k, v in C.most_common():
    print(f"   {v:5d}  {k[0]} | {k[1]}")
print("   по семействам:", dict(FAM.most_common(14)))
pickle.dump(EX, open(sys.argv[2], "wb"))
