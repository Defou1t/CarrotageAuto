r"""_twin_probe.py — ЛИНИЯ-ДВОЙНИК В ИЗОБРАЖЕНИИ: подтверждение перехода из картинки.

Эдуард (23.07): «часто одна и та же линия одновременно находится в 1х и 5х масштабе на одной
глубине». §6.41 подтвердил это на уровне ГЕОМЕТРИИ (85% переходов объясняются формулой «то же
значение, другая шкала», медиана ошибки 2.5% ширины). Здесь проверяется, видно ли двойника
В ТУШИ — то есть можно ли подтверждать переход НЕЗАВИСИМЫМ признаком из изображения, которого у
декодера нет вообще.

ПОЧЕМУ ПРОШЛАЯ ПОПЫТКА ПРОВАЛИЛАСЬ (замер 22.07 на Semeguniv: 5% против 5% случайных): она шла
по ИНТЕРПОЛИРОВАННОЙ трассе — §6.41 показал, что интерполяция убивает прыжок, и позиции у неё
смещены. Здесь всё считается по СЫРЫМ ВЕРШИНАМ.

МЕТРИКА. Для строки y и уровня k: значение v = scale_k(x). Для каждого ДРУГОГО уровня k2
считаем x2 = inv(scale_k2, v) и смотрим, есть ли тушь в ±tol px. Сравниваем три популяции:
  ON  — строки, где эксперт РЕАЛЬНО держит два уровня рядом (окрестность перехода ±200 строк);
  OFF — строки далеко от любого перехода (двойника быть не должно или он не важен);
  RND — случайные строки (общий фон).
Если ON ≫ OFF/RND — признак рабочий и его можно подать в DP.

  python _twin_probe.py [--sheets 25] [--tol 10]
"""
import sys, io, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import cv2
from extract_nlgx import extract, NULL
import dataset as ds
import decode_levels as DL
from dataset_build import find_image
from auto import trace2d as T
from auto.config import DEFAULT
from _decoder_data import train_sheets

ap = argparse.ArgumentParser()
ap.add_argument("--sheets", type=int, default=25)
ap.add_argument("--tol", type=int, default=10)
ap.add_argument("--near", type=int, default=200, help="окрестность перехода, строк")
a = ap.parse_args()
P = DEFAULT.cv
COLORS = ("black", "red", "green", "blue", "orange")


def inv(s, v):
    dv = (s["v_right"] - s["v_left"]) or 1e-9
    return s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / dv


ON, OFF, RND = [], [], []
per_curve = []
for n in train_sheets(a.sheets):
    img = find_image(n)
    if not img:
        continue
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = extract(str(n))
        cs = [c for c in ds.real_curves(m) if len(DL.build_family(m, c)) > 1 and DL.gt_levels(c)]
        if not cs:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(img), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        fgs = {c: T._color_fg(rgb, c, P) for c in COLORS}
    except Exception:
        continue
    H, W = rgb.shape[:2]
    for c in cs:
        fam = DL.build_family(m, c); gl = DL.gt_levels(c)
        vy = [c["top_y"] + i for i, x in enumerate(c["xs"]) if x != NULL]
        vx = [float(x) for x in c["xs"] if x != NULL]
        if len(vy) < 80:
            continue
        maps = [DL.scale_map(s) for s in fam]
        # цвет кривой: где под её ВЕРШИНАМИ больше туши
        best_c, best_h = "black", -1
        for col in COLORS:
            fg = fgs[col]
            h = sum(1 for k in range(0, len(vy), 5)
                    if 0 <= vy[k] < H and fg[vy[k], max(0, int(vx[k]) - 3):int(vx[k]) + 4].any())
            if h > best_h:
                best_h, best_c = h, col
        fg = fgs[best_c]
        trs = [vy[i] for i in range(1, len(vy)) if gl.get(vy[i], 0) != gl.get(vy[i - 1], 0)]
        if not trs:
            continue
        loc = {"ON": [], "OFF": []}
        for i in range(len(vy)):
            y = vy[i]
            if not (0 <= y < H):
                continue
            k = gl.get(y, 0)
            if k >= len(fam):
                continue
            v = maps[k](vx[i])
            hit = 0; tot = 0
            for k2 in range(len(fam)):
                if k2 == k:
                    continue
                x2 = inv(fam[k2], v)
                if 0 <= x2 < W:
                    tot += 1
                    lo = max(0, int(x2) - a.tol); hi = min(W, int(x2) + a.tol + 1)
                    if fg[y, lo:hi].any():
                        hit += 1
            if not tot:
                continue
            near = any(abs(y - t) <= a.near for t in trs)
            loc["ON" if near else "OFF"].append(hit / tot)
        if len(loc["ON"]) >= 10 and len(loc["OFF"]) >= 10:
            ON += loc["ON"]; OFF += loc["OFF"]
            per_curve.append((n.parent.parent.name, c["name"].split()[0],
                              float(np.mean(loc["ON"])), float(np.mean(loc["OFF"]))))
        # фон: те же вершины, но значение берётся у СЛУЧАЙНОЙ строки (разрушает связь)
        rg = np.random.default_rng(0)
        for i in rg.choice(len(vy), size=min(60, len(vy)), replace=False):
            y = vy[i]
            if not (0 <= y < H):
                continue
            j = int(rg.integers(len(vy)))
            k = gl.get(y, 0)
            if k >= len(fam):
                continue
            v = maps[k](vx[j])                       # значение ЧУЖОЙ строки
            hit = 0; tot = 0
            for k2 in range(len(fam)):
                if k2 == k:
                    continue
                x2 = inv(fam[k2], v)
                if 0 <= x2 < W:
                    tot += 1
                    lo = max(0, int(x2) - a.tol); hi = min(W, int(x2) + a.tol + 1)
                    if fg[y, lo:hi].any():
                        hit += 1
            if tot:
                RND.append(hit / tot)

on = np.array(ON); off = np.array(OFF); rnd = np.array(RND)
print(f"кривых: {len(per_curve)}   строк: ON {len(on)}, OFF {len(off)}, фон {len(rnd)}   tol=±{a.tol}px\n")
print(f"ТУШЬ НА ПОЗИЦИИ ДВОЙНИКА (то же значение, другая шкала):")
print(f"  ON  — рядом с переходом (±{a.near} строк): {100*on.mean():.1f}%")
print(f"  OFF — вдали от переходов               : {100*off.mean():.1f}%")
print(f"  ФОН — значение от чужой строки          : {100*rnd.mean():.1f}%")
if len(per_curve):
    d = np.array([p[2] - p[3] for p in per_curve])
    print(f"\nпокривой ON−OFF: med {100*np.median(d):+.1f} п.п.,  "
          f"кривых с ON>OFF: {100*(d > 0).mean():.0f}% из {len(d)}")
    print("\nтоп-6 по разнице ON−OFF:")
    for p in sorted(per_curve, key=lambda q: q[2] - q[3], reverse=True)[:6]:
        print(f"   {p[0]:<14}{p[1]:<8} ON {100*p[2]:>5.1f}%  OFF {100*p[3]:>5.1f}%")
print("\nЧитать: ON ≫ ФОН ⇒ двойник виден в туши и может ПОДТВЕРЖДАТЬ переход независимо от DP.")
