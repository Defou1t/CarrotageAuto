r"""_fail_plane2.py — ЧЕМ НЕЧЕСТНЫ КРИВЫЕ В МЕРЕ ПРИЁМКИ (НА ПЛОСКОСТИ В ОБЕ СТОРОНЫ) (04.10).

Аналог `_fail_kind.py` для меры с 04.10. Для каждой кривой эталона (не DA, ≥ 50 точек) и выдачи под ТЕМ ЖЕ именем (именная мера) —
честна ли (обе медианы расстояний на плоскости ≤ 3 px, покрытие ≥ 0.9; уровни масштаба — пересчёт как в счёте). Нечестные —
по виду провала:
  «нет выдачи»  — кривой с этим именем нет или общих строк < 30;
  «короткая»    — обе медианы ≤ 3 px, но покрытие < 0.9;
  «не то имя»   — выдача под ДРУГИМ именем трека честна для этой кривой (именная ошибка, трасса есть);
  «чужая»       — ≥ половины точек выдачи в 3 px от ДРУГОЙ кривой эталона листа;
  «рядом»       — медиана расстояний эталон → выдача 3–10 px (та же линия, смещение / край штриха / уровень);
  «мимо»        — прочее (сетка, текст, шум, неоцифрованная кривая).
Только чтение выдачи повтора и эталона.

  _fail_plane2.py --dir F:/nds/output/taskS/rp_lvl --mode NS --every 3
"""
import sys, argparse, pickle, hashlib, re
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M
import decode_levels as DL

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_lvl")
ap.add_argument("--mode", default="NS")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/fail_plane2.pkl")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}


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


def tree(tr):
    ys = np.array(sorted(tr), float); xs = np.array([tr[int(y)] for y in ys], float)
    P = [np.stack([ys, xs], 1)]
    if len(ys) > 1:
        dy = np.diff(ys); dx = np.diff(xs)
        for i in np.flatnonzero((dy <= 30) & (np.abs(dx) > 1) & (np.abs(dx) <= 3000)):
            n = int(np.ceil(abs(dx[i]))); t = np.arange(1, n) / n
            P.append(np.stack([ys[i] + dy[i] * t, xs[i] + dx[i] * t], 1))
    return cKDTree(np.concatenate(P))


def dists(tr, gt):
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return None
    d1, _ = tree(tr).query(np.array([[y, gt[y]] for y in com], float))
    d2, _ = tree(gt).query(np.array([[y, tr[y]] for y in com], float))
    return com, d1, d2, len(com) / max(1, len(gt))


R = []
for sn, S in SETS.items():
    for sh in S[::a.every]:
        q = SRC.get(sh)
        if not q:
            continue
        stem = q.stem
        pd = Path(a.dir) / a.mode / f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
        if not got:
            continue
        mt = extract(str(q)); mw = extract(str(got))
        G = {c["name"]: dense(c) for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and sum(1 for x in c["xs"] if x != NULL) >= 50}
        LG = {c["name"]: lv_rows(c) for c in mt["curves"] if c["name"] in G}
        CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in G}
        W = {c["name"]: dense(c) for c in mw["curves"] if M.mnem_root(c["name"]) != "DA"}
        LW = {c["name"]: lv_rows(c) for c in mw["curves"] if c["name"] in W}
        GT = {g: tree(G[g]) for g in G}
        for g in G:
            kind = None
            w = W.get(g)
            r = dists(relevel(w, LW.get(g, {}), LG.get(g, {}), CH.get(g, [])), G[g]) if w else None
            if r is None:
                kind = "нет выдачи"
            else:
                com, d1, d2, cov = r
                m1, m2 = float(np.median(d1)), float(np.median(d2))
                if m1 <= 3 and m2 <= 3 and cov >= 0.9:
                    kind = "честна"
                elif m1 <= 3 and m2 <= 3:
                    kind = "короткая"
            if kind is None:
                # другая выдача трека честна для этой кривой? (именная ошибка)
                for w2, tr2 in W.items():
                    if w2 == g or not tr2:
                        continue
                    r2 = dists(relevel(tr2, LW.get(w2, {}), LG.get(g, {}), CH.get(g, [])), G[g])
                    if r2 and np.median(r2[1]) <= 3 and np.median(r2[2]) <= 3 and r2[3] >= 0.9:
                        kind = "не то имя"; break
            if kind is None:
                pts = np.array([[y, w[y]] for y in sorted(w)][::4], float)
                if len(pts):
                    best = 0.0
                    for g2, T2 in GT.items():
                        if g2 == g:
                            continue
                        d, _ = T2.query(pts)
                        best = max(best, float(np.mean(d <= 3)))
                    if best >= 0.5:
                        kind = "чужая"
                if kind is None:
                    kind = "рядом" if 3 < m1 <= 10 else "мимо"
            R.append(dict(set=sn, sheet=sh, name=g, root=re.sub(r"^BKZ_", "", M.mnem_root(g)), kind=kind))
pickle.dump(R, open(a.dump, "wb"))
for sn in SETS:
    C = Counter(r["kind"] for r in R if r["set"] == sn)
    n = sum(C.values())
    print(f"★ {sn} (каждый {a.every}-й лист, кривых {n}): " + ", ".join(f"{k} {C[k]} ({C[k] / max(1, n):.0%})" for k in
          ("честна", "не то имя", "короткая", "рядом", "чужая", "мимо", "нет выдачи")))
    F = defaultdict(Counter)
    for r in R:
        if r["set"] == sn:
            F[r["root"]][r["kind"]] += 1
    for f, c in sorted(F.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
        tot = sum(c.values())
        print(f"   {f:6s} {tot:4d}: честна {c['честна']}, не то имя {c['не то имя']}, короткая {c['короткая']}, рядом {c['рядом']}, "
              f"чужая {c['чужая']}, мимо {c['мимо']}, нет выдачи {c['нет выдачи']}")
