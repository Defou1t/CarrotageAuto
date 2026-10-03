r"""_return_stroke.py — ВИДЕН ЛИ НА СКАНЕ ВОЗВРАТНЫЙ ШТРИХ ПЕРА В МЕСТАХ ПЕРЕХОДОВ МАСШТАБА (03.10, §6.262).

На переходе (оборот: перо ушло за край шкалы и продолжило на следующем масштабе) регистратор рисует быстрый почти
горизонтальный штрих от точки до перехода к точке после. Трасса прода — функция строки и такой штрих не ведёт; декодер уровней
по трассе (`decode_levels`) поэтому слеп. Здесь — можно ли видеть штрих прямо на скане.
Для переходов эталона (смена уровня между соседними сегментами, x до и x после из вершин эксперта): в окне ±W строк — доля
колонок между x до и x после, где есть тушь (темнее бумаги строки на thr) хотя бы в одной строке окна. Контроль — та же
величина для «ложного скачка» той же длины от случайной строки той же кривой вдали от переходов (тушь между ними — сетка,
чужие кривые). AUC «переход против контроля» и доля с покрытием ≥ 0.9.

  _return_stroke.py --every 3 --win 12 --thr 40
"""
import sys, argparse, random
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto"); sys.path.insert(0, r"F:\nds\Auto\digitizer"); sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
from auto import meta as M
import decode_levels as DL

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--every", type=int, default=3)
ap.add_argument("--win", type=int, default=12)
ap.add_argument("--thr", type=int, default=40)
ap.add_argument("--dump", default="")
a = ap.parse_args()
TS = Path(a.ts)
lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
SRC = {q.name: q for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx")}
IMGS = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/img/*"):
    if q.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
        IMGS.setdefault(q.stem, q)
rng = random.Random(0)


def coverage(gray, paper, y, xa, xb, win, thr):
    x0, x1 = int(round(min(xa, xb))), int(round(max(xa, xb)))
    if x1 - x0 < 10:
        return None
    y0, y1 = max(0, y - win), min(gray.shape[0], y + win + 1)
    blk = gray[y0:y1, x0:x1 + 1]
    pp = paper[y0:y1, None]
    ink = (blk < pp - thr).any(axis=0)
    return float(ink.mean())


T, Cn = [], []
for sh in (lst("wellmap_sheets.txt") + lst("holdoutA_sheets.txt"))[::a.every]:
    q = SRC.get(sh)
    if not q or q.stem not in IMGS:
        continue
    mt = extract(str(q))
    curves = [c for c in mt["curves"] if M.mnem_root(c["name"]) != "DA" and len(c.get("segments") or []) > 1
              and len(DL.build_family(mt, c)) >= 2]
    if not curves:
        continue
    gray = np.asarray(Image.open(IMGS[q.stem]).convert("L"), np.int16)
    paper = np.percentile(gray[:, ::4], 90, axis=1)
    for c in curves:
        pts = sorted((c["top_y"] + i, float(x)) for i, x in enumerate(c["xs"]) if x != NULL)
        if len(pts) < 50:
            continue
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        segs = sorted(c["segments"], key=lambda s: s[0])
        trans = []
        for s0, s1 in zip(segs, segs[1:]):
            if s0[2] == s1[2]:
                continue
            ia = np.searchsorted(ys, s0[1], side="right") - 1; ib = np.searchsorted(ys, s1[0])
            if ia < 0 or ib >= len(ys) or ys[ib] - ys[ia] > 40:
                continue
            trans.append((int(ys[ib]), xs[ia], xs[ib], s1[2] - s0[2]))
        for y, xa, xb, d in trans:
            cv = coverage(gray, paper, y, xa, xb, a.win, a.thr)
            if cv is not None:
                T.append((M.mnem_root(c["name"]), d, abs(xb - xa), cv))
                # контроль: случайная строка этой кривой не ближе 200 строк к переходам, «скачок» той же длины внутрь трека
                far = [i for i in range(len(ys)) if all(abs(ys[i] - t[0]) > 200 for t in trans)]
                if far:
                    i = rng.choice(far)
                    sa = mt["scale_axes"]
                    xl = min(s["x_left"] for s in sa); xr = max(s["x_right"] for s in sa)
                    L = abs(xb - xa)
                    xc = xs[i] - L if xs[i] - L >= xl else xs[i] + L
                    if xl <= xc <= xr:
                        cc = coverage(gray, paper, int(ys[i]), xs[i], xc, a.win, a.thr)
                        if cc is not None:
                            Cn.append((M.mnem_root(c["name"]), 0, L, cc))
    del gray
if a.dump:
    import pickle
    pickle.dump(dict(T=T, C=Cn), open(a.dump, "wb"))
t = np.array([r[3] for r in T]); cn = np.array([r[3] for r in Cn])


def auc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(bool)
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


print(f"★ переходов эталона {len(T)}, контролей {len(Cn)} (каждый {a.every}-й лист, окно ±{a.win} строк, порог туши {a.thr})")
print(f"   покрытие тушью между x до и после: переходы — медиана {np.median(t):.2f}, ≥ 0.9 у {np.mean(t >= 0.9):.0%}; "
      f"контроль — медиана {np.median(cn):.2f}, ≥ 0.9 у {np.mean(cn >= 0.9):.0%}; AUC {auc(t, cn):.3f}")
up = np.array([r[3] for r in T if r[1] > 0]); dn = np.array([r[3] for r in T if r[1] < 0])
print(f"   вверх (1× → 5×): {len(up)}, медиана {np.median(up):.2f}; вниз: {len(dn)}, медиана {np.median(dn):.2f}")
fam = Counter(r[0] for r in T)
for f, n in fam.most_common(10):
    v = np.array([r[3] for r in T if r[0] == f])
    print(f"   {f:8s} {n:4d}: медиана {np.median(v):.2f}, ≥ 0.9 у {np.mean(v >= 0.9):.0%}")
