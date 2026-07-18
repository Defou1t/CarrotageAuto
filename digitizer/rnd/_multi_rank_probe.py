"""ПОТОЛОК разбора ПО СЧЁТУ (ранговое присвоение) на ordered-мульти-листах, БЕЗ кода пайплайна.
Гипотеза: если кривые листа нигде не меняют порядок слева-направо (класс ordered), то k-й ран
чернил в строке = k-я кривая. Меряем ПРЯМО на чернилах: в каждой строке трека берём раны
ink_foreground, если их ровно K (=число GT-кривых) — присваиваем по рангу и сравниваем с
экспертной трассой (px). Печатаем cov (доля строк с ровно K ранами) и med/<=3px по кривой.

  python rank_probe.py [N] [--tok ...]   (листы берутся из cross.json, класс ordered)
"""
import sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from auto import imaging as im, meta as M, frame as F
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
SCR = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=10)
ap.add_argument("--tok", default="")
ap.add_argument("--per-well", type=int, default=1)
ap.add_argument("--cross", default="", help="json от _multi_crossings.py (иначе cross.json рядом)")
a = ap.parse_args()

rows = json.loads(Path(a.cross or (SCR / "cross.json")).read_text(encoding="utf-8"))
sel, per_well = [], {}
for r in rows:
    if r["cls"] != "ordered" or per_well.get(r["well"], 0) >= a.per_well:
        continue
    if a.tok and r["tok"].strip().upper() not in [t.strip().upper() for t in a.tok.split("|")]:
        continue
    sel.append(r); per_well[r["well"]] = per_well.get(r["well"], 0) + 1
sel = sel[:a.n]
print(f"ordered-листов в пробе: {len(sel)}")

cfg = Config(); p = cfg.cv
allm = []
for r in sel:
    n = ARCHIVE / r["well"] / "wlg" / (r["stem"] + ".nlgx")
    if not n.is_file():
        cands = list((ARCHIVE / r["well"] / "wlg").glob(r["stem"][:30] + "*.nlgx"))
        if not cands:
            print(f"{r['well']}: нет {r['stem']}"); continue
        n = cands[0]
    img = find_image(n)
    mo = extract(str(n))
    m = M.parse_filename(n.name, MN)
    rgb = im.load_rgb(str(img))
    fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
    if getattr(fr, "row_shift", None) is not None:
        rgb = F.apply_row_shift(rgb, fr.row_shift)
    fg = im.ink_foreground(rgb, p)
    gts = [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
           and M.mnem_root(c["name"]) != "DA"]
    ser = [{c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL} for c in gts]
    K = len(gts)
    if len(fr.tracks) != 1:
        # многотрековый лист: берём трек, к которому эмиттер относит БОЛЬШИНСТВО слотов
        from auto import emit as E
        tid = {}
        for c in gts:
            ti = E._slot_track(mo, c, fr)
            tid[ti] = tid.get(ti, 0) + 1
        best = max(tid, key=tid.get)
        keep = [i for i, c in enumerate(gts) if E._slot_track(mo, c, fr) == best]
        gts = [gts[i] for i in keep]; ser = [ser[i] for i in keep]; K = len(gts)
        if best is None or K < 2:
            print(f"{r['well']}/{r['tok']:<12} треков={len(fr.tracks)}, слоты не легли на трек — пропуск"); continue
        t = fr.tracks[best]
    else:
        t = fr.tracks[0]
    lo, hi = int(t.x_left) + 3, int(t.x_right) - 3
    y0, y1 = int(fr.top_y), int(fr.bottom_y)
    err = [[] for _ in range(K)]
    nk = nink = 0
    for y in range(y0, y1):
        runs = im.row_runs(fg[y, lo:hi], gap=4)
        if not runs:
            continue
        nink += 1
        if len(runs) != K:
            continue
        nk += 1
        cent = [lo + run[2] for run in runs]        # ЦЕНТР ТЕЛА рана (canon prefer_body)
        gx = [(s[y], k) for k, s in enumerate(ser) if y in s]
        if len(gx) != K:
            continue
        order = [k for _, k in sorted(gx)]          # GT-кривые слева-направо в этой строке
        gsorted = sorted(v for v, _ in gx)
        for rank, k in enumerate(order):
            err[k].append(abs(cent[rank] - gsorted[rank]))
    covK = nk / max(1, nink)
    line = f"{r['well']:<14} {r['tok']:<12} K={K} cov(K ранов)={covK:.2f}  "
    for k, c in enumerate(gts):
        e = np.array(err[k], float)
        if len(e) < 50:
            line += f"{c['name'].split()[0]}:мало "; continue
        line += f"{c['name'].split()[0]}:med{np.median(e):.0f}/≤3px{(e<=3).mean()*100:.0f}% "
        allm.append((float(np.median(e)), float((e <= 3).mean())))
    print(line)

if allm:
    med = np.array([x[0] for x in allm]); p3 = np.array([x[1] for x in allm])
    print(f"\nИТОГ по {len(allm)} кривым: med(med)={np.median(med):.1f}px  "
          f"кривых med<=3px: {(med<=3).sum()}/{len(med)}  <=3px сред {p3.mean()*100:.0f}%")
