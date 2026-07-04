r"""Зигзаг-скелет, ШАГ 1 (04.07): скелет туши + ВЕРШИНЫ штрихов (x-экстремумы пути
пера). Домен: в зигзаг-режиме (3752 целиком, хвост 3130 3724-3760, качели 3853)
клики эксперта сидят на ВЕРШИНАХ почти-горизонтальных штрихов; per-row экстракция
кривую там не описывает. Прежде чем строить нитевание двух путей (идентичность),
меряем ДЕТЕКЦИЮ: какая доля кликов эксперта лежит возле найденной вершины.

  python mk_pen_path.py <scan.jpg> <etalon.nlgx> [--win d0 d1] [--render]

Выход: CDF расстояния клик→ближайшая вершина + (опц.) рендер окна с вершинами.
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, NULL

DARK = 110
PROM = 8          # прominence вершины: насколько путь уходит назад по x
DEDUP = 4         # радиус дедупликации вершин, px


def zhang_suen(img, max_iter=60):
    """Скелетизация Zhang-Suen, векторно на numpy. img: bool HxW."""
    I = img.astype(np.uint8).copy()
    I = np.pad(I, 1)

    def neighbors(A):
        P2 = A[:-2, 1:-1]; P3 = A[:-2, 2:]; P4 = A[1:-1, 2:]; P5 = A[2:, 2:]
        P6 = A[2:, 1:-1]; P7 = A[2:, :-2]; P8 = A[1:-1, :-2]; P9 = A[:-2, :-2]
        return P2, P3, P4, P5, P6, P7, P8, P9

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P2, P3, P4, P5, P6, P7, P8, P9 = neighbors(I)
            C = I[1:-1, 1:-1]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros_like(B)
            for k in range(8):
                A += ((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8)
            if step == 0:
                cond = (C == 1) & (B >= 2) & (B <= 6) & (A == 1) & \
                       (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (C == 1) & (B >= 2) & (B <= 6) & (A == 1) & \
                       (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            if cond.any():
                C[cond] = 0
                changed = True
        if not changed:
            break
    return I[1:-1, 1:-1].astype(bool)


def find_vertices(skel):
    """Вершины = скелетные пиксели-локальные x-экстремумы: справа (нет соседа
    правее, и не вертикальный проход) и слева. Возврат (rights, lefts) списками (y,x)."""
    S = np.pad(skel, 1)
    C = S[1:-1, 1:-1]
    ys, xs = np.nonzero(C)
    rights, lefts = [], []
    for y, x in zip(ys, xs):
        n = S[y:y + 3, x:x + 3]  # 3x3 вокруг (в паддинге центр = y+1,x+1)
        right_col = n[:, 2].any()
        left_col = n[:, 0].any()
        vert_through = n[0, 1] and n[2, 1]
        if not right_col and not vert_through:
            rights.append((y, x))
        if not left_col and not vert_through:
            lefts.append((y, x))
    return rights, lefts


def prominence_filter(skel, verts, side, prom=PROM, walk=120):
    """Вершина реальна, если путь по скелету уходит от неё назад по x на ≥prom.
    Дешёвая аппроксимация: в окне строк ±3 скелет не появляется за вершиной
    (уже гарантировано), а в ±walk строк есть скелет на ≥prom НАЗАД по x."""
    H, W = skel.shape
    out = []
    for y, x in verts:
        y0, y1 = max(0, y - 3), min(H, y + 4)
        if side == "R":
            back = skel[y0:y1, max(0, x - walk):max(0, x - prom + 1)]
        else:
            back = skel[y0:y1, x + prom:min(W, x + walk)]
        if back.any():
            out.append((y, x))
    return out


def thick_filter(dark, verts, min_th=3):
    """Вершина на КРИВОЙ, не на линии сетки: вертикальная толщина туши ≥ min_th."""
    H, W = dark.shape
    out = []
    for y, x in verts:
        th = 1
        while y - th >= 0 and dark[y - th, x] and th < 12: th += 1
        d = 0
        while y + d + 1 < H and dark[y + d + 1, x] and d < 12: d += 1
        if th + d >= min_th:
            out.append((y, x))
    return out


def tip_extend(dark, verts, side, max_ext=14):
    """Скелет-вершина сидит на медиальной оси = на полтолщины ДО кончика туши
    (систематика med 5px в шаге 1). Дотяжка вершины вдоль строки до границы."""
    H, W = dark.shape
    out = []
    for y, x in verts:
        step = 1 if side == "R" else -1
        xx = x
        for _ in range(max_ext):
            nx = xx + step
            if 0 <= nx < W and dark[y, nx]:
                xx = nx
            else:
                break
        out.append((y, xx))
    return out


def dedup(verts, r=DEDUP):
    out = []
    seen = set()
    for y, x in sorted(verts):
        key = (y // r, x // r)
        if key in seen:
            continue
        seen.add(key)
        out.append((y, x))
    return out


def trace_dict(curve):
    ty = curve["top_y"]
    return {ty + i: x for i, x in enumerate(curve["xs"]) if x != NULL}


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    scan, et_path = a[0], a[1]
    render = "--render" in a
    me = extract(et_path)
    da = me["depth_axis"]
    def yof(d): return int(da["top_y"] + (d - da["top_depth"]) * da["span_px"] / da["span_depth"])
    if "--win" in a:
        i = a.index("--win")
        y0, y1 = yof(float(a[i + 1])), yof(float(a[i + 2]))
    else:
        y0, y1 = int(da["top_y"]), int(da["bottom_y"])

    gts = {c["name"].split()[0]: trace_dict(c) for c in me["curves"]}
    clicks = [(y, x, mn) for mn in ("MGZ1", "MPZ1") for y, x in gts.get(mn, {}).items() if y0 <= y < y1]
    if not clicks:
        print("нет кликов в окне"); return
    xs_all = [x for _, x, _ in clicks]
    x0, x1 = max(0, int(min(xs_all)) - 40), int(max(xs_all)) + 40

    rgb = np.asarray(Image.open(scan).convert("RGB"))
    band = rgb[y0:y1, x0:x1]
    dark = band.max(2) < DARK
    print(f"окно y[{y0}..{y1}] x[{x0}..{x1}], тушь {dark.mean()*100:.1f}%")
    skel = zhang_suen(dark)
    print(f"скелет: {skel.sum()} px")
    R, L = find_vertices(skel)
    # tip_extend НЕ применять: эксперт кликает середину разворота (медиальную ось),
    # не внешний край туши — гейт 04.07: с дотяжкой med 5.0→7.1px, ≤3px 36→18%.
    # thick_filter НЕ применять глобально: убивает вершины тонких штрихов (>15px
    # 14→19%); ложные вершины сетки не мешают recall-метрике, отсев — на нитевании.
    R = dedup(prominence_filter(skel, R, "R"))
    L = dedup(prominence_filter(skel, L, "L"))
    print(f"вершин: правых {len(R)}, левых {len(L)}")

    verts = np.array([(y, x) for y, x in R + L], dtype=float)
    dists = []
    for y, x, mn in clicks:
        vy, vx = y - y0, x - x0
        d = np.sqrt(((verts[:, 0] - vy) ** 2) + ((verts[:, 1] - vx) ** 2)).min() if len(verts) else 1e9
        dists.append(d)
    dists = np.array(dists)
    print(f"клики→вершина ({len(dists)} кликов): med {np.median(dists):.1f}px | "
          f"≤3px {(dists<=3).mean()*100:.0f}% | ≤5px {(dists<=5).mean()*100:.0f}% | "
          f"≤8px {(dists<=8).mean()*100:.0f}% | >15px {(dists>15).mean()*100:.0f}%")

    # метрика 2: клик → скелет в ТОЙ ЖЕ строке (±1) — верхняя граница качества
    # пер-строчной реконструкции после нитевания (скелет = путь пера)
    dx2 = []
    for y, x, mn in clicks:
        vy, vx = y - y0, x - x0
        rows = skel[max(0, vy - 1):vy + 2]
        cols = np.nonzero(rows.any(0))[0]
        dx2.append(np.abs(cols - vx).min() if len(cols) else 1e9)
    dx2 = np.array(dx2)
    print(f"клики→скелет в строке: med {np.median(dx2):.1f}px | "
          f"≤3px {(dx2<=3).mean()*100:.0f}% | ≤5px {(dx2<=5).mean()*100:.0f}% | "
          f">15px {(dx2>15).mean()*100:.0f}%")

    if render:
        h = min(y1 - y0, 1000)
        crop = Image.fromarray(band[:h]).convert("RGB")
        dr = ImageDraw.Draw(crop)
        sy, sx = np.nonzero(skel[:h])
        for yy, xx in zip(sy, sx):
            dr.point([(int(xx), int(yy))], fill=(0, 180, 0))
        for y, x in R:
            if y < h: dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 140, 0), width=2)
        for y, x in L:
            if y < h: dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(160, 0, 200), width=2)
        for y, x, mn in clicks:
            vy, vx = y - y0, x - x0
            if vy < h:
                col = (220, 30, 30) if mn == "MGZ1" else (30, 80, 230)
                dr.ellipse([vx - 5, vy - 5, vx + 5, vy + 5], outline=col, width=2)
        stem = os.path.splitext(os.path.basename(scan))[0]
        fn = rf"F:\nds\output\mk_data\export_qc\{stem}_penpath_step1.png"
        crop.save(fn)
        print("рендер:", fn)


if __name__ == "__main__":
    main()
