"""МУЛЬТИ-ВЕТКА, шаг 1: ловят ли НИТИ (link_strands по связности ранов) кривые мульти-листа?
Полосы по x опровергнуты (§5.2), разбор по счёту тоже (§5.3). Нить — третий вариант: вести
штрих по связности, как в BKZ M1 (там детект нитей покрывал GT на 100%, med 1.0px).

Меряем ПОТОЛОК подхода: для каждой GT-кривой берём ЛУЧШУЮ нить (оракул по med|dx|) и печатаем
med/≤3px/покрытие. Это НЕ метрика продукта — это ответ на вопрос «есть ли что выбирать».
Плюс N (сколько длинных нитей) против K (сколько кривых в рамке) — цена задачи выбора.

  python _multi_strand_probe.py [N листов] [--tok ...] [--prob DIR] [--min-cov 0.2]
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from bkz_track import link_strands
from _multi_replica_probe import dense
from auto import imaging as im, meta as M, frame as F, emit as E
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")


def chain_fragments(strands, min_len=5, max_gap_rows=40, tol=25.0):
    """СБОРКА фрагментов в кривую по НЕПРЕРЫВНОСТИ. Замер 19.07: линкер даёт куски, попадающие
    в кривую с med 1px, но покрывающие лишь ~10% её длины — значит проблема не в геометрии, а в
    том, что нить рвётся. Здесь куски сшиваются: следующий начинается ниже конца предыдущего
    (в пределах max_gap_rows) и его старт согласован с ЭКСТРАПОЛЯЦИЕЙ конца по наклону (tol).
    Жадно от самого длинного куска. Возвращает список цепочек (dict row->x)."""
    frs = [s for s in strands if len(s) >= min_len]
    frs.sort(key=len, reverse=True)
    used = [False] * len(frs)
    ends = []
    for s in frs:
        rows = sorted(s)
        k = max(1, len(rows) // 10)
        slope = (s[rows[-1]] - s[rows[-1 - k]]) / max(1, rows[-1] - rows[-1 - k])
        ends.append((rows[0], s[rows[0]], rows[-1], s[rows[-1]], slope))
    out = []
    for i in range(len(frs)):
        if used[i]:
            continue
        used[i] = True
        cur = dict(frs[i])
        y_end, x_end, sl = ends[i][2], ends[i][3], ends[i][4]
        while True:
            best, bd = -1, 1e9
            for j in range(len(frs)):
                if used[j]:
                    continue
                y0j, x0j = ends[j][0], ends[j][1]
                dy = y0j - y_end
                if not (0 < dy <= max_gap_rows):
                    continue
                d = abs(x0j - (x_end + sl * dy))         # отклонение от экстраполяции
                if d < bd:
                    bd, best = d, j
            if best < 0 or bd > tol:
                break
            used[best] = True
            cur.update(frs[best])
            y_end, x_end, sl = ends[best][2], ends[best][3], ends[best][4]
        out.append(cur)
    out.sort(key=len, reverse=True)
    return out
ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=12)
ap.add_argument("--tok", default="BK, IK|GK, NGK|BKZ, DS|MBK, MDS|STK+DS|BK+IK|MK, MDS")
ap.add_argument("--per-well", type=int, default=1)
ap.add_argument("--prob", default="")
ap.add_argument("--min-cov", type=float, default=0.2, help="доля высоты рамки для «длинной» нити")
ap.add_argument("--chain", action="store_true", help="сшивать фрагменты по непрерывности")
ap.add_argument("--chain-gap", type=int, default=60, help="макс. разрыв сшивки в строках скана")
ap.add_argument("--chain-tol", type=float, default=25.0, help="допуск на отклонение от экстраполяции")
ap.add_argument("--gap", type=int, default=12, help="max_gap_x НА СТРОКУ (умножается на step)")
ap.add_argument("--skip", type=int, default=12, help="max_skip_y в строках скана")
ap.add_argument("--step", type=int, default=4, help="прореживание строк: link_strands на чистом "
                "Python идёт по строкам и на скане 47000x3161 не считается за разумное время; "
                "для ЗАМЕРА ПОТОЛКА геометрия при шаге 3-4 не теряется")
a = ap.parse_args()
TOKS = [t.strip().upper() for t in a.tok.split("|")]
p = Config().cv
PROBDIR = Path(a.prob) if a.prob else None

cands, per_well = [], {}
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    well = wlg.parent.name
    for n in sorted(wlg.glob("*.nlgx")):
        if "_auto" in n.stem or per_well.get(well, 0) >= a.per_well:
            continue
        m = M.parse_filename(n.name, MN)
        if m.curves_token.strip().upper() not in TOKS or not find_image(n):
            continue
        cands.append((well, n, find_image(n), m))
        per_well[well] = per_well.get(well, 0) + 1
# link_strands — чистый Python по строкам, и число АКТИВНЫХ нитей растёт с числом ранов
# (каждый неиспользованный ран порождает нить). На 47000x3161 с 60 ранами/строку это не считается.
# Для замера потолка берём САМЫЕ МЕЛКИЕ листы — вывод о геометрии от размера скана не зависит.
from PIL import Image
cands.sort(key=lambda c: Image.open(c[2]).size[0] * Image.open(c[2]).size[1])
cands = cands[:a.n]
print(f"листов: {len(cands)}")

allm = []
for well, n, img, m in cands:
    mo = extract(str(n))
    rgb = im.load_rgb(str(img))
    fr = F.frame_from_nlgx(str(n), m, p, rgb=rgb)
    prob = None
    if PROBDIR is not None:
        q = PROBDIR / f"{Path(img).stem}_prob.npy"
        if q.is_file():
            from auto.prob import prob_from_npy
            prob = prob_from_npy(q)(rgb)
    fg = im.ink_foreground(rgb, p, prob=prob)
    gts = [c for c in mo["curves"] if sum(1 for x in c["xs"] if x != NULL) >= 100
           and M.mnem_root(c["name"]) != "DA"]
    tid = Counter(E._slot_track(mo, c, fr) for c in gts)
    best_t = tid.most_common(1)[0][0]
    gts = [c for c in gts if E._slot_track(mo, c, fr) == best_t]
    if not gts:
        continue
    t = fr.tracks[best_t if best_t is not None else 0]
    y0, y1 = int(fr.top_y), int(fr.bottom_y)
    lo, hi = int(t.x_left) + 3, int(t.x_right) - 3
    H = max(1, y1 - y0)
    st = max(1, a.step)
    sub = np.ascontiguousarray(fg[y0:y1:st].astype(bool))       # прореженные строки
    # ПАРАМЕТРЫ ЛИНКЕРА масштабируются шагом: они заданы «на строку», а при прореживании скачок
    # x между соседними обработанными строками в step раз больше. Плюс перо мульти-листа
    # свипует на десятки px за строку — БКЗ-шный max_gap_x=10 рвёт нить на каждом взмахе.
    raw = link_strands(sub, lo, hi, 0, sub.shape[0],
                       max_gap_x=a.gap * st, max_skip_y=max(2, a.skip // st))
    strands = [{y0 + r * st: x for r, x in s.items()} for s in raw]   # обратно в строки скана
    long_s = [s for s in strands if len(s) * st >= a.min_cov * H]
    if a.chain:
        long_s = chain_fragments(strands, min_len=max(3, int(0.01 * H / st)),
                                 max_gap_rows=a.chain_gap // st, tol=a.chain_tol)
        long_s = [s for s in long_s if len(s) * st >= a.min_cov * H]
    K = len(gts)
    line = f"{well:<14} {m.curves_token:<12} K={K} нитей={len(strands):>4} длинных={len(long_s):>3}  "
    for c in gts:
        d = dense(c)
        best, bmed, bcov = None, 1e9, 0.0
        for s in long_s:
            common = [y for y in s if y in d]
            if len(common) < 50:
                continue
            e = np.array([abs(s[y] - d[y]) for y in common], float)
            if np.median(e) < bmed:
                bmed, best, bcov = float(np.median(e)), s, len(common) / max(1, len(d))
        nm = c["name"].split()[0]
        if best is None:
            line += f"{nm}:НЕТ "; continue
        e = np.array([abs(best[y] - d[y]) for y in best if y in d], float)
        line += f"{nm}:med{bmed:.0f}/≤3px{(e<=3).mean()*100:.0f}%/cov{bcov:.2f} "
        allm.append((bmed, float((e <= 3).mean()), bcov, len(long_s), K))
    print(line)

if allm:
    med = np.array([x[0] for x in allm]); p3 = np.array([x[1] for x in allm])
    cov = np.array([x[2] for x in allm])
    print(f"\nПОТОЛОК НИТЕЙ (оракул выбирает лучшую) по {len(allm)} кривым:")
    print(f"  med(med)={np.median(med):.1f}px  кривых med<=3px: {(med<=3).sum()}/{len(med)}  "
          f"≤3px сред {p3.mean()*100:.0f}%  покрытие сред {cov.mean():.2f}")
    print(f"  цена выбора: длинных нитей на лист {np.median([x[3] for x in allm]):.0f} против K={np.median([x[4] for x in allm]):.0f}")
