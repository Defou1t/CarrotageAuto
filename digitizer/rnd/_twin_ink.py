r"""_twin_ink.py — НА КАКОМ ПЕРЕ ТРАССА: ТУШЬ ПЕРА-БЛИЗНЕЦА НА СКАНЕ (03.10, §6.265).

Одна величина пишется двумя перьями: 1× и 5× (отношение диапазонов шкал f, общий ноль x0). Если трасса в строке на пере 5× в x,
перо 1× рисует ту же величину в x_up = x0 + f·(x − x0); если трасса на пере 1×, близнец 5× — в x_dn = x0 + (x − x0)/f.
Правило по строке: тушь у x_up и нет у x_dn → уровень 1; тушь у x_dn и нет у x_up → 0; иначе — не решено. По эталону
(трасса и уровни эксперта, каждый N-й лист поля и сорта A, кривые с переходами и цепочкой; каждая 3-я строка): доля решённых
строк и их точность; разрез по семействам. Это — проверка признака на трассе эксперта, до использования на выдаче.

  _twin_ink.py --every 3 --tol 3
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
from _multi_replica_probe import dense
from auto import meta as M
import decode_levels as DL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--tol", type=int, default=3)
ap.add_argument("--thr", type=int, default=40)
ap.add_argument("--dump", default=r"F:/nds/output/taskS/twin_ink.pkl")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
        IMGS.setdefault(q.stem, q)
R = []
for sh in (lst("wellmap_sheets.txt") + lst("holdoutA_sheets.txt"))[::a.every]:
    q = SRC.get(sh)
    if not q or q.stem not in IMGS:
        continue
    mt = extract(str(q))
    cur = [c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and any(l > 0 for _, _, l in (c.get("segments") or []))
           and len(DL.build_family(mt, c)) >= 2]
    if not cur:
        continue
    gray = np.asarray(Image.open(IMGS[q.stem]).convert("L"), np.int16)
    paper = np.percentile(gray[:, ::4], 90, axis=1)
    H, W = gray.shape
    for c in cur:
        fam = DL.build_family(mt, c)
        s0, s1 = fam[0], fam[1]
        r0 = (s0["v_right"] - s0["v_left"]) or 1.0; f = (s1["v_right"] - s1["v_left"]) / r0
        if not (1.5 <= abs(f) <= 30) or abs(s1["v_left"] - s0["v_left"]) > 1e-6 * max(1.0, abs(s0["v_right"])):
            R.append(dict(sheet=sh, name=c["name"], root=M.mnem_root(c["name"]), kind="не умножение", n=0)); continue
        x0 = s0["x_left"] - s0["v_left"] * (s0["x_right"] - s0["x_left"]) / r0
        xl, xr = min(s0["x_left"], s0["x_right"]), max(s0["x_left"], s0["x_right"])
        gt = dense(c)
        lv = {}
        for y0, y1, l in c["segments"]:
            for y in range(int(y0), int(y1) + 1):
                lv[y] = int(l)
        ys = sorted(y for y in gt if y in lv)[::3]
        res = Counter()
        for y in ys:
            if not (0 <= y < H):
                continue
            x = gt[y]; L = lv[y]
            if L > 1:
                continue
            xs_ = x if L == 0 else x                                         # трасса эксперта в этой строке
            xu = x0 + f * (xs_ - x0); xd = x0 + (xs_ - x0) / f
            row = gray[y] < paper[y] - a.thr

            def ink(xx):
                if xx < xl - 5 or xx > xr + 5:
                    return None
                i = int(round(xx))
                return bool(row[max(0, i - a.tol):min(W, i + a.tol + 1)].any())
            iu, id_ = ink(xu), ink(xd)
            if iu and not id_:
                p = 1
            elif id_ and not iu:
                p = 0
            else:
                res["не решено"] += 1; continue
            res["решено"] += 1; res["верно"] += (p == L)
            res[f"решено, эксперт {L}"] += 1; res[f"верно, эксперт {L}"] += (p == L)
        R.append(dict(sheet=sh, name=c["name"], root=M.mnem_root(c["name"]), kind="умножение", n=len(ys), **res))
    del gray
pickle.dump(R, open(a.dump, "wb"))
mul = [r for r in R if r["kind"] == "умножение" and r["n"]]
tot = Counter()
for r in mul:
    for k in ("решено", "верно", "не решено", "решено, эксперт 0", "верно, эксперт 0", "решено, эксперт 1", "верно, эксперт 1"):
        tot[k] += r.get(k, 0)
n_all = tot["решено"] + tot["не решено"]
print(f"★ кривых с переходами {len(R)}: шкалы-умножения {len(mul)}, прочие (сдвиг и т. п.) {len(R) - len(mul)}")
print(f"   строк {n_all}: решено правилом {tot['решено']} ({tot['решено'] / max(1, n_all):.0%}), из них верно {tot['верно'] / max(1, tot['решено']):.1%}")
print(f"   эксперт на 1×: решено {tot['решено, эксперт 0']}, верно {tot['верно, эксперт 0'] / max(1, tot['решено, эксперт 0']):.1%}; "
      f"эксперт на 5×: решено {tot['решено, эксперт 1']}, верно {tot['верно, эксперт 1'] / max(1, tot['решено, эксперт 1']):.1%}")
fam = Counter(r["root"] for r in mul)
for fm, n in fam.most_common(12):
    rs = [r for r in mul if r["root"] == fm]
    dec = sum(r.get("решено", 0) for r in rs); ok = sum(r.get("верно", 0) for r in rs); un = sum(r.get("не решено", 0) for r in rs)
    print(f"   {fm:8s} {n:4d}: решено {dec / max(1, dec + un):.0%}, верно {ok / max(1, dec):.1%}")
