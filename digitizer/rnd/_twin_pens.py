r"""_twin_pens.py — СБОРКА КРИВОЙ ИЗ ДВУХ ПЕРЬЕВ 1× И 5× (03.10, §6.263). Что даст, по кэшу трасс, без прогона пайплайна.

§6.262: на переходах эксперта трасса прода не прыгает в 53% случаев — кривая записана ДВУМЯ перьями сразу. Перо 1× уходит на упор
у правого края, перо 5× идёт отдельной линией. Эксперт берёт перо 1×, пока оно не на упоре, иначе перо 5× (правило заказчика
§6.41). Трасса прода ведёт одно перо целиком.
Здесь для каждой кривой выдачи с цепочкой масштабов (то же имя в эталоне) ищется перо-близнец среди всех трасс кэша её трека
(прод-путь + декодер):
  S — трасса выдачи; близнец C — на общих строках, где перо 1× не на упоре, |(x1 − x0)/f − (x5 − x0)| мало (медиана ≤ tol px),
  где f — отношение диапазонов шкал 5×/1×, x0 — x нуля шкалы 1×. Проверяются оба случая (S — перо 1× или S — перо 5×).
Сборка: в строке берётся перо 1× с уровнем 0, если оно есть и не на упоре (x < упор − margin, упор = 99-й перцентиль x пера 1×,
но не дальше края шкалы); иначе перо 5× с уровнем 1; иначе что есть. Кривые без близнеца не меняются.
Счёт — именные честные В ЗНАЧЕНИЯХ (как `_level_bench.py` / `_name_cost_prod.py --levels`), поле и сорт A, парный тест по листам.

  _twin_pens.py --every 1 --tol 4 --margin 4
"""
import sys, argparse, pickle, hashlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--dir", default=r"F:/nds/output/taskS/rp_vc/N")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/tcache")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--tol", type=float, default=4.0)
ap.add_argument("--margin", type=float, default=4.0)
ap.add_argument("--min-rows", type=int, default=150)
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SETS = {"поле": lst("wellmap_sheets.txt"), "сорт A": lst("holdoutA_sheets.txt")}
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
import decode_levels as DL


def lv_rows(segs):
    out = {}
    for y0, y1, lv in segs or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


def un(t):
    return dict(zip(np.asarray(t[0]).tolist(), np.asarray(t[1]).tolist()))


def honest_val(tr, lo, gt, lt, ch):
    """именная честность в значениях: tr — x по строкам, lo — уровень выдачи по строкам (нет — 0)"""
    com = [y for y in tr if y in gt]
    if len(com) < 30:
        return False
    e = []
    for y in com:
        x = tr[y]; o = lo.get(y, 0); t = lt.get(y, 0)
        if o != t and len(ch) >= 2:
            if o >= len(ch) or t >= len(ch):
                e.append(1e4); continue
            so, sg = ch[o], ch[t]
            v = so["v_left"] + (x - so["x_left"]) * (so["v_right"] - so["v_left"]) / ((so["x_right"] - so["x_left"]) or 1)
            x = sg["x_left"] + (v - sg["v_left"]) * (sg["x_right"] - sg["x_left"]) / ((sg["v_right"] - sg["v_left"]) or 1)
        e.append(abs(x - gt[y]))
    return bool(np.median(e) <= 3.0 and len(com) / max(1, len(gt)) >= 0.9)


def twin_err(p1, p5, x0, f, rail):
    """медиана |(x1 − x0)/f − (x5 − x0)| на общих строках, где перо 1× не на упоре; → (ошибка, строк)"""
    rows = [y for y in p1 if y in p5 and p1[y] < rail - a.margin]
    if len(rows) < a.min_rows:
        return None, len(rows)
    e = np.array([abs((p1[y] - x0) / f - (p5[y] - x0)) for y in rows])
    return float(np.median(e)), len(rows)


C = {k: Counter() for k in SETS}
PER = {k: defaultdict(lambda: [0, 0]) for k in SETS}
EX = []
for sn, names in SETS.items():
    for sh in names[a.offset::a.every]:
        q = SRC.get(sh)
        if not q:
            continue
        stem = q.stem
        key = f"{stem[:40]}_{hashlib.md5(stem.encode('utf-8')).hexdigest()[:8]}"
        pd = Path(a.dir) / key
        got = next(pd.glob("*_auto.nlgx"), None) if pd.is_dir() else None
        cf = Path(a.cache) / f"{key}.pkl"
        if not got or not cf.exists():
            continue
        mt = extract(str(q)); mw = {c["name"]: c for c in extract(str(got))["curves"]}
        v = None
        for c in mt["curves"]:
            nm = c["name"]
            if M.mnem_root(nm) == "DA" or sum(1 for x in c["xs"] if x != NULL) < 50 or nm not in mw:
                continue
            ch = DL.build_family(mt, c)
            if len(ch) < 2:
                continue
            gt = dense(c); lt = lv_rows(c.get("segments"))
            w = mw[nm]
            S = dense(w); lw = lv_rows(w.get("segments"))
            if len(S) < 30:
                continue
            h0 = honest_val(S, lw, gt, lt, ch)
            C[sn]["кривых с цепочкой"] += 1
            C[sn]["честно как есть"] += h0
            PER[sn][sh][0] += h0
            s0, s1 = ch[0], ch[1]
            f = ((s1["v_right"] - s1["v_left"]) / ((s0["v_right"] - s0["v_left"]) or 1))
            if not (1.5 <= abs(f) <= 30):
                PER[sn][sh][1] += h0; C[sn]["честно со сборкой"] += h0; continue
            x0 = s0["x_left"] - s0["v_left"] * (s0["x_right"] - s0["x_left"]) / ((s0["v_right"] - s0["v_left"]) or 1)
            xr = max(s0["x_left"], s0["x_right"])
            if v is None:
                v = pickle.load(open(cf, "rb"))
                cands = [un(t) for _, t in v["traces"]] + ([un(t) for _, t in v["alt"]] if v.get("alt") else [])
            lo_x, hi_x = min(s0["x_left"], s0["x_right"]) - 20, xr + 20
            cs = [d for d in cands if d and lo_x <= np.median(list(d.values())) <= hi_x]
            best = None
            for d in cs:
                if d is S:
                    continue
                # S — перо 1×, d — 5×
                rail1 = min(xr, np.percentile(list(S.values()), 99))
                e1, n1 = twin_err(S, d, x0, f, rail1)
                # S — перо 5×, d — 1×
                rail2 = min(xr, np.percentile(list(d.values()), 99))
                e2, n2 = twin_err(d, S, x0, f, rail2)
                for e, n, role, rail in ((e1, n1, "S=1×", rail1), (e2, n2, "S=5×", rail2)):
                    if e is not None and e <= a.tol and (best is None or e < best[0]):
                        best = (e, n, role, d, rail)
            if best is None:
                PER[sn][sh][1] += h0; C[sn]["честно со сборкой"] += h0; continue
            e, n, role, d, rail = best
            p1, p5 = (S, d) if role == "S=1×" else (d, S)
            tr, lo = {}, {}
            for y in set(p1) | set(p5):
                if y in p1 and p1[y] < rail - a.margin:
                    tr[y] = p1[y]; lo[y] = 0
                elif y in p5:
                    tr[y] = p5[y]; lo[y] = 1
                else:
                    tr[y] = p1[y]; lo[y] = 0
            h1 = honest_val(tr, lo, gt, lt, ch)
            C[sn]["близнец найден"] += 1
            C[sn][f"близнец: {role}"] += 1
            C[sn]["честно со сборкой"] += h1
            C[sn]["стало честно"] += (h1 and not h0); C[sn]["перестало"] += (h0 and not h1)
            PER[sn][sh][1] += h1
            EX.append(dict(set=sn, sheet=sh, name=nm, role=role, err=e, n=n, h0=h0, h1=h1))
if a.dump:
    pickle.dump(EX, open(a.dump, "wb"))
rng = np.random.default_rng(0)
for sn in SETS:
    c = C[sn]
    d = np.array([v1 - v0 for v0, v1 in PER[sn].values()]); nz = d[d != 0]
    p = float(np.mean(np.abs((rng.choice([-1, 1], size=(20000, len(nz))) * nz).sum(1)) >= abs(d.sum()))) if len(nz) else 1.0
    print(f"★ {sn} (каждый {a.every}-й лист, tol {a.tol}, margin {a.margin}): кривых с цепочкой {c['кривых с цепочкой']}, "
          f"близнец найден у {c['близнец найден']} (S=1× {c['близнец: S=1×']}, S=5× {c['близнец: S=5×']})")
    print(f"   именных честных в значениях среди них: как есть {c['честно как есть']} → со сборкой {c['честно со сборкой']} "
          f"(Δ {c['честно со сборкой'] - c['честно как есть']:+d}: стало {c['стало честно']}, перестало {c['перестало']}; "
          f"листов ↑{int((d > 0).sum())}/↓{int((d < 0).sum())}, p = {p:.4f})")
