r"""_style_analysis.py — ЗАКОНОМЕРНОСТИ СТИЛЯ ЛИНИЙ В КОРПУСЕ (04.10, к §6.273; по описанию заказчика).

Читает `_style_scan.py --dump` (стиль вдоль трассы эксперта и выдачи) и, если дан, `--pairs` (пары 1:1 без имени мерой приёмки,
`_name_cost_prod.py --pairs-dump`). Разделы:
  A. надёжность: стиль по выдаче против стиля по эталону на кривых, где выдача на той же туши (медиана |Δx| ≤ 3 px);
  B. по семействам: толщина (медиана), доля цветных, доля пунктирных;
  C. ГЗ на одном листе: толщина по номеру зонда (k < m — кто толще), цвет;
  D. по годам (из имени листа) и по скважинам — устойчивость соглашений;
  E. перепутанные имена (по `--pairs`): различает ли стиль выдачи пару, которую раскладка перепутала.

  _style_analysis.py --style F:/nds/output/taskS/style_scan.pkl --pairs F:/nds/output/taskS/rp_lvl_S_pairs.pkl
"""
import sys, argparse, pickle, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from auto import meta as M

ap = argparse.ArgumentParser()
ap.add_argument("--style", default=r"F:/nds/output/taskS/style_scan.pkl")
ap.add_argument("--cache", default=r"F:/nds/output/taskS/level_bench.pkl")
ap.add_argument("--pairs", default="")
ap.add_argument("--mode", default="S")
ap.add_argument("--colored", type=float, default=30.0)
ap.add_argument("--dash", type=float, default=0.25)
a = ap.parse_args()
R = pickle.load(open(a.style, "rb"))
CUR = pickle.load(open(a.cache, "rb"))
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
well = lambda sh: SRC[sh].parent.parent.name if sh in SRC else "?"
fam = lambda r: re.sub(r"^BKZ_", "", r["root"])
iscol = lambda t: max(abs(t["c1"]), abs(t["c2"])) >= a.colored
isdash = lambda t: t["gap"] >= a.dash


def year(sh):
    m = re.search(r"(19[5-9]\d|20[0-2]\d)", sh)
    return int(m.group(1)) if m else None


# A. надёжность замера по выдаче
same = []
for r in R:
    if not (r["truth"] and r["out"]):
        continue
    cv = CUR[r["ci"]]
    com, ig, it = np.intersect1d(cv["gy"], cv["ty"], return_indices=True)
    if len(com) < 100:
        continue
    if np.median(np.abs(cv["gx"][ig] - cv["tx"][it])) <= 3.0:
        same.append(r)
if same:
    wt = np.array([r["truth"]["width"] for r in same]); wo = np.array([r["out"]["width"] for r in same])
    ct = np.array([[r["truth"]["c1"], r["truth"]["c2"]] for r in same]); co = np.array([[r["out"]["c1"], r["out"]["c2"]] for r in same])
    gt_ = np.array([r["truth"]["gap"] for r in same]); go = np.array([r["out"]["gap"] for r in same])
    print(f"★ A. Надёжность (кривых, где выдача на той же туши: {len(same)}): толщина — корреляция {np.corrcoef(wt, wo)[0, 1]:.2f}, "
          f"|Δ| медиана {np.median(np.abs(wt - wo)):.2f} px; цвет R−B — корр. {np.corrcoef(ct[:, 0], co[:, 0])[0, 1]:.2f}; "
          f"разрывы — корр. {np.corrcoef(gt_, go)[0, 1]:.2f}")
# B. по семействам
ok = [r for r in R if r["truth"]]
F = defaultdict(list)
for r in ok:
    F[fam(r)].append(r["truth"])
print(f"\n★ B. По семействам (эталон; {len(ok)} кривых): толщина по нормали — медиана [квартили]; цветных; пунктирных")
for f, L in sorted(F.items(), key=lambda kv: -len(kv[1]))[:22]:
    w = np.array([t["width"] for t in L])
    print(f"   {f:8s} {len(L):4d}: {np.median(w):4.1f} [{np.percentile(w, 25):.1f}–{np.percentile(w, 75):.1f}] px; "
          f"цветных {np.mean([iscol(t) for t in L]):4.0%}; пунктирных {np.mean([isdash(t) for t in L]):4.0%}")
# C. ГЗ на одном листе: номер зонда → толщина
bys = defaultdict(dict)
for r in ok:
    m = re.match(r"^(?:BKZ_)?GZ(\d)", r["name"])
    if m:
        bys[r["sheet"]][int(m.group(1))] = r["truth"]
C = Counter(); colpair = Counter()
for sh, d in bys.items():
    ks = sorted(d)
    for i, k in enumerate(ks):
        for m in ks[i + 1:]:
            dw = d[k]["width"] - d[m]["width"]
            C[(k, m, "n")] += 1; C[(k, m, "k")] += dw > 0.5; C[(k, m, "m")] += dw < -0.5
            dc = np.hypot(d[k]["c1"] - d[m]["c1"], d[k]["c2"] - d[m]["c2"])
            colpair[(k, m, "n")] += 1; colpair[(k, m, "разный цвет")] += dc >= 30
            colpair[(k, m, "разный штрих")] += isdash(d[k]) != isdash(d[m])
print("\n★ C. ГЗ на одном листе, пара (k < m): k толще / m толще / всего (разница > 0.5 px); разный цвет; разный штрих")
for (k, m) in sorted({(k, m) for k, m, _ in C}):
    n = C[(k, m, "n")]
    if n >= 12:
        print(f"   ГЗ{k}–ГЗ{m}: {C[(k, m, 'k')]:3d} / {C[(k, m, 'm')]:3d} / {n:3d}; цвет {colpair[(k, m, 'разный цвет')]:3d}; штрих {colpair[(k, m, 'разный штрих')]:3d}")
# различимость пары по стилю вообще (хоть что-то: толщина > 1 px, цвет, штрих)
dist = Counter()
for sh, d in bys.items():
    ks = sorted(d)
    for i, k in enumerate(ks):
        for m in ks[i + 1:]:
            tk, tm = d[k], d[m]
            diff = (abs(tk["width"] - tm["width"]) > 1.0, np.hypot(tk["c1"] - tm["c1"], tk["c2"] - tm["c2"]) >= 30, isdash(tk) != isdash(tm))
            dist["пар"] += 1; dist["различимы хоть чем-то"] += any(diff)
            dist["толщиной > 1 px"] += diff[0]; dist["цветом"] += diff[1]; dist["штрихом"] += diff[2]
print(f"   все пары ГЗ на листе: {dist['пар']}; различимы стилем хоть чем-то — {dist['различимы хоть чем-то'] / max(1, dist['пар']):.0%} "
      f"(толщиной {dist['толщиной > 1 px'] / max(1, dist['пар']):.0%}, цветом {dist['цветом'] / max(1, dist['пар']):.0%}, "
      f"штрихом {dist['штрихом'] / max(1, dist['пар']):.0%})")
# D. по годам и по скважинам
Y = defaultdict(list)
for r in ok:
    y = year(r["sheet"])
    Y["до 1970" if y and y < 1970 else ("1970-е" if y and y < 1980 else ("1980-е" if y and y < 1990 else ("1990-е" if y and y < 2000 else ("2000+" if y else "год не указан"))))].append(r["truth"])
print("\n★ D. По годам: кривых; цветных; пунктирных; толщина (медиана)")
for k in ("до 1970", "1970-е", "1980-е", "1990-е", "2000+", "год не указан"):
    L = Y.get(k, [])
    if L:
        print(f"   {k:14s} {len(L):4d}; цветных {np.mean([iscol(t) for t in L]):4.0%}; пунктирных {np.mean([isdash(t) for t in L]):4.0%}; "
              f"толщина {np.median([t['width'] for t in L]):.1f} px")
WG = defaultdict(lambda: Counter())
for sh, d in bys.items():
    if 1 in d and 2 in d:
        dw = d[1]["width"] - d[2]["width"]
        WG[well(sh)]["n"] += 1; WG[well(sh)]["1 толще"] += dw > 0.5; WG[well(sh)]["2 толще"] += dw < -0.5
cons = [(w_, c) for w_, c in WG.items() if c["n"] >= 4]
if cons:
    agree = sum(max(c["1 толще"], c["2 толще"]) for _, c in cons); tot = sum(c["n"] for _, c in cons)
    print(f"   ГЗ1–ГЗ2 по скважинам (≥ 4 листов, {len(cons)} скважин): преобладающий порядок толщины держится в {agree / max(1, tot):.0%} листов")
# E. перепутанные имена: различает ли стиль выдачи
if a.pairs:
    P = pickle.load(open(a.pairs, "rb"))[a.mode]
    ST = {}
    for r in R:
        ST[(r["sheet"], r["name"])] = r
    E = Counter(); ex = []
    by = defaultdict(dict)
    for sh, t, g, w in P:
        by[(sh, t)][g] = w
    for (sh, t), mt in by.items():
        for g, w in mt.items():
            if g == w or mt.get(w) != g or g > w:          # только перестановки A↔B, каждую один раз
                continue
            # выдача «w» лежит на линии эталона g, выдача «g» — на линии эталона w. Эталонные стили имён: S_t(g), S_t(w)
            sg, sw = ST.get((sh, g)), ST.get((sh, w))
            if not (sg and sw and sg["truth"] and sw["truth"]):
                continue
            tg, tw = sg["truth"], sw["truth"]
            E["перестановок"] += 1
            dw = abs(tg["width"] - tw["width"]) > 1.0
            dc = np.hypot(tg["c1"] - tw["c1"], tg["c2"] - tw["c2"]) >= 30
            dd = isdash(tg) != isdash(tw)
            E["различимы стилем"] += dw or dc or dd
            E["толщиной"] += dw; E["цветом"] += dc; E["штрихом"] += dd
            if len(ex) < 12:
                ex.append(f"{sh[:34]} {g.split()[0]}↔{w.split()[0]}: толщина {tg['width']:.1f}/{tw['width']:.1f}, "
                          f"цвет ({tg['c1']:.0f},{tg['c2']:.0f})/({tw['c1']:.0f},{tw['c2']:.0f}), разрывы {tg['gap']:.2f}/{tw['gap']:.2f}")
    n = max(1, E["перестановок"])
    print(f"\n★ E. Перестановки A↔B мерой приёмки (пары кривых в одном треке): {E['перестановок']}; линии пары различимы стилем — "
          f"{E['различимы стилем'] / n:.0%} (толщиной > 1 px {E['толщиной'] / n:.0%}, цветом {E['цветом'] / n:.0%}, штрихом {E['штрихом'] / n:.0%})")
    for e in ex:
        print("   " + e)
