"""ПРИРОДА ОСТАТОЧНОЙ ОШИБКИ ТРАССИРОВЩИКА: латч на соседнюю кривую или блуждание между?

Контекст (замеры 19.07). FLAG-гейт узким местом НЕ является: снятие барьера даёт 8->28
записанных слотов, но честных (med<=3px И cov>=0.9) 0 в обоих прогонах, а кросс-матч трасс
против всех экспертных кривых листа — 0/28 в пределах 10px. Стенд «оракульная полоса» (полоса
из ЭКСПЕРТНОЙ кривой ±20px, т.е. идеальный U1) даёт med 1048.8 -> 71.7px, но честных 3/24.
Значит потолок ограничен самим 2D-обходом.

ПОДОЗРЕВАЕМЫЙ УЧАСТОК — auto/trace2d.py::trace_line, ветка else:
    cont = [r for r in runs if r[0] - 2 <= pred <= r[1] + 2]
    a, b, c = (min(cont, ...) if cont else min(runs, key=lambda r: abs(r[2] - pred)))
Ветка else НЕ ограничена расстоянием: если чернила своей кривой на строке нет, трасса садится
на ближайший ран хоть за сотни px и остаётся там. В _extend_ends такой предохранитель ЕСТЬ
(`if abs(c - x) > slmax + (b - a): break`), в основном цикле — нет.

НО прежде чем чинить, надо знать ТИП ошибки. Меряем по строкам оракульной трассы:
  «своя»   — ближайшая экспертная кривая листа = та, которую ведём;
  «ЛАТЧ»   — ближайшая ЧУЖАЯ экспертная кривая (и мы к ней ближе, чем к своей);
  «между»  — ни к одной кривой не ближе TOL px (трасса болтается по пустому месту).
Если преобладает ЛАТЧ — правка ветки else бьёт в цель. Если «между» — виновата маска/выбор
точки в ране, и правка не поможет.

  python _latch_probe.py [--tol 10] [--pad 20]
"""
import sys, io, json, contextlib, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import cv2
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import frame as F, understand as U, confidence as C, trace2d as T, meta as M
from auto.config import DEFAULT

ARCH = Path(r"F:\nds\projects\Archive")
SH = [r"BOGAT_011\wlg\BOGAT_011_BKZ, DS_2800-3190_200_1984-03-02_D_1_B_1.nlgx",
      r"BOGAT_015\wlg\BOGAT_015_BKZ, DS_3300-3700_200_1990-01-08_D_1.nlgx",
      r"LEVEN_023\wlg\LEVEN_023_BKZ, DS_1010-1500_200_1996-12-02_D_1.nlgx",
      r"BEZLUD_051\wlg\BEZLUD_051_RK, AK, DS_2688-3142_500_1998-08-05_D_1.nlgx",
      r"YULIIV_055\wlg\YULIIV_055_MK, MBK, MDS_2084-3060_200_1998-07-18_D_1.nlgx"]
MN = r"F:\nds\Auto\mnemonics.json"
ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=10.0)
ap.add_argument("--pad", type=int, default=20)
a = ap.parse_args()
p = DEFAULT.cv


class FakeLine:
    pass


tot = {"своя": 0, "ЛАТЧ на чужую": 0, "между кривыми": 0}
per_curve = []
for rel in SH:
    n = ARCH / rel
    well = n.parent.parent.name
    img = find_image(n)
    if not img:
        continue
    rgb = cv2.cvtColor(cv2.imread(str(img), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = M.parse_filename(n.name, MN)
        fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
        if getattr(fr, "row_shift", None) is not None:
            rgb = F.apply_row_shift(rgb, fr.row_shift)
        sheet = U.understand(rgb, fr, m, p)
        C.classify(sheet)
    colors = sorted({L.color for L in sheet.lines}) or ["black"]
    fgs = {c: T._color_fg(rgb, c, p) for c in colors}
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    dens = {g["name"]: dense(g) for g in gts}          # ВСЕ кривые листа, плотно
    print(f"\n### {well}  кривых {len(gts)}, цвета U1 {colors}")
    for g in gts:
        nm = g["name"].split()[0]
        own = dens[g["name"]]
        if len(own) < 50:
            continue
        ys = sorted(own); xs = [own[y] for y in ys]
        L = FakeLine()
        L.x_lo = min(xs) - a.pad; L.x_hi = max(xs) + a.pad
        L.x_center = float(np.median(xs)); L.y0 = min(ys); L.y1 = max(ys)
        best, best_med = None, 1e9
        for c in colors:                                # как в oracle_band: лучший цвет
            tr = T.trace_line(fgs[c], L, fr, p)
            common = [y for y in tr if y in own]
            if len(common) < 30:
                continue
            d = float(np.median([abs(tr[y] - own[y]) for y in common]))
            if d < best_med:
                best_med, best = d, tr
        if best is None:
            print(f"   {nm:<8} нет трассы"); continue
        cnt = {"своя": 0, "ЛАТЧ на чужую": 0, "между кривыми": 0}
        for y, x in best.items():
            dists = [(abs(x - dd[y]), nmk) for nmk, dd in dens.items() if y in dd]
            if not dists:
                continue
            dmin, who = min(dists)
            if dmin > a.tol:
                cnt["между кривыми"] += 1
            elif who == g["name"]:
                cnt["своя"] += 1
            else:
                cnt["ЛАТЧ на чужую"] += 1
        s = sum(cnt.values()) or 1
        for k in tot:
            tot[k] += cnt[k]
        per_curve.append({"well": well, "curve": nm, "med": round(best_med, 1),
                          **{k: round(100 * v / s) for k, v in cnt.items()}})
        print(f"   {nm:<8} med={best_med:7.1f}  своя {100*cnt['своя']//s:>3}%  "
              f"ЛАТЧ {100*cnt['ЛАТЧ на чужую']//s:>3}%  между {100*cnt['между кривыми']//s:>3}%")

s = sum(tot.values()) or 1
print(f"\n=== ИТОГО по {len(per_curve)} кривым (tol={a.tol:.0f}px) ===")
for k, v in tot.items():
    print(f"   {k:<16} {100*v/s:5.1f}%   ({v} строк)")
print("\nЧитать так: преобладает ЛАТЧ -> правка ветки else в trace_line бьёт в цель;")
print("преобладает «между кривыми» -> виновата маска/выбор точки в ране, правка НЕ поможет.")
Path(r"F:\nds\output\taskS").mkdir(parents=True, exist_ok=True)
Path(r"F:\nds\output\taskS\latch.json").write_text(
    json.dumps({"per_curve": per_curve, "total": tot}, ensure_ascii=False, indent=1),
    encoding="utf-8")
