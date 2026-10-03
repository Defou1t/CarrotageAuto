r"""_level_dec_truth.py — ДЕКОДЕР ПЕРЕХОДОВ НА ТРАССЕ ЭКСПЕРТА (03.10, §6.262): ПЛОХ САМ ДЕКОДЕР ИЛИ ТРАССА?

`decode_levels.decode` (DP «оборотов») на трассе эксперта — изоляция качества декодера от качества трассы. Для каждой
кривой эталона с переходами и цепочкой масштабов в шаблоне: уровни декодера (с `refine.enforce_min_run`, как emit) против
уровней эксперта — доля верных строк; разрез по семействам и по механизму перехода на рисунке:
  «оборот»  — в строках уровня > 0 у правого края шкалы 1× туши нет (одно перо ушло влево);
  «два пера» — в строках уровня > 0 у правого края шкалы 1× есть тушь (перо 1× стоит на упоре, кривую ведёт второе перо).
Признак по картинке: доля строк уровня > 0, где в ±4 px от правого края шкалы 1× есть тёмный пиксель.

  _level_dec_truth.py --every 2
"""
import sys, argparse, pickle
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
from auto import meta as M, refine
from auto.config import DEFAULT
import decode_levels as DL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--rail-thr", type=float, default=0.5, help="доля строк уровня > 0 с тушью у края шкалы 1× ⇒ «два пера»")
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
        IMGS.setdefault(q.stem, q)


def lv_rows(c):
    out = {}
    for y0, y1, lv in c.get("segments") or []:
        for y in range(int(y0), int(y1) + 1):
            out[y] = int(lv)
    return out


R = []
sheets = (lst("wellmap_sheets.txt") + lst("holdoutA_sheets.txt"))[::a.every]
for si, sh in enumerate(sheets, 1):
    q = SRC.get(sh)
    if not q:
        continue
    mt = extract(str(q))
    cur = [c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and any(l > 0 for _, _, l in (c.get("segments") or []))]
    if not cur:
        continue
    gray = None
    for c in cur:
        fam = DL.build_family(mt, c)
        if len(fam) < 2:
            continue
        lt = lv_rows(c)
        xs = {c["top_y"] + i: float(x) for i, x in enumerate(c["xs"]) if x != NULL}
        if len(xs) < 50:
            continue
        raw = DL.decode(xs, fam, lam=DEFAULT.cv.level_lam)
        ld = refine.enforce_min_run(raw, DEFAULT.cv.level_min_run)
        rows = [y for y in xs if y in lt]
        acc = float(np.mean([ld.get(y, 0) == lt[y] for y in rows])) if rows else float("nan")
        acc0 = float(np.mean([lt[y] == 0 for y in rows])) if rows else float("nan")       # «всё на 1×»
        # механизм: тушь у правого края шкалы 1× в строках уровня > 0
        up = [y for y in rows if lt[y] > 0][::7]
        rail = float("nan")
        if up and q.stem in IMGS:
            if gray is None:
                gray = np.asarray(Image.open(IMGS[q.stem]).convert("L"), np.int16)
            xr = int(round(fam[0]["x_right"]))
            hits = 0
            for y in up:
                if 0 <= y < gray.shape[0]:
                    row = gray[y]
                    paper = np.percentile(row[::4], 90)
                    seg = row[max(0, xr - 4):xr + 5]
                    hits += bool((seg < paper - 40).any())
            rail = hits / len(up)
        R.append(dict(sheet=sh, name=c["name"], root=M.mnem_root(c["name"]), res=DL.is_resistive(c["name"]), K=len(fam),
                      acc=acc, acc0=acc0, rail=rail, up=float(np.mean([lt[y] > 0 for y in rows])) if rows else 0.0))
    del gray
    if si % 100 == 0:
        print(f"  … {si}/{len(sheets)}", file=sys.stderr)
if a.dump:
    pickle.dump(R, open(a.dump, "wb"))
R = [r for r in R if r["acc"] == r["acc"]]
acc = np.array([r["acc"] for r in R]); acc0 = np.array([r["acc0"] for r in R])
print(f"★ кривых эталона с переходами и цепочкой: {len(R)} (каждый {a.every}-й лист поля и сорта A)")
print(f"   декодер оборотов на ТРАССЕ ЭКСПЕРТА: верных строк — медиана {np.median(acc):.2f}, ≥ 0.9 у {np.mean(acc >= 0.9):.0%}; "
      f"«всё на 1×» — медиана {np.median(acc0):.2f}, ≥ 0.9 у {np.mean(acc0 >= 0.9):.0%}")
for grp, sel in (("резистивные", [r for r in R if r["res"]]), ("НЕ резистивные", [r for r in R if not r["res"]])):
    if sel:
        aa = np.array([r["acc"] for r in sel])
        print(f"   {grp}: {len(sel)}; медиана {np.median(aa):.2f}, ≥ 0.9 у {np.mean(aa >= 0.9):.0%}")
rl = [r for r in R if r["rail"] == r["rail"]]
two = [r for r in rl if r["rail"] >= a.rail_thr]; one = [r for r in rl if r["rail"] < a.rail_thr]
for grp, sel in (("«два пера» (тушь у края 1× в строках 5×)", two), ("«оборот»", one)):
    if sel:
        aa = np.array([r["acc"] for r in sel])
        fam = Counter(r["root"] for r in sel)
        print(f"   {grp}: {len(sel)}; декодер верен — медиана {np.median(aa):.2f}, ≥ 0.9 у {np.mean(aa >= 0.9):.0%}; семейства: "
              + ", ".join(f"{k} {v}" for k, v in fam.most_common(10)))
print("   по семействам (n, медиана верных строк, доля «два пера»): " + ", ".join(
    f"{k} {len(v)}: {np.median([r['acc'] for r in v]):.2f} / {np.mean([r['rail'] >= a.rail_thr for r in v if r['rail'] == r['rail']] or [0]):.0%}"
    for k, v in sorted(((k, [r for r in R if r['root'] == k]) for k in {r['root'] for r in R}), key=lambda kv: -len(kv[1]))[:14]))
