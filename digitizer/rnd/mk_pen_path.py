r"""Зигзаг-скелет: ШАГ 1 (04.07) — скелет туши + ВЕРШИНЫ штрихов (x-экстремумы пути
пера). Домен: в зигзаг-режиме (3752 целиком, хвост 3130 3724-3760, качели 3853)
клики эксперта сидят на ВЕРШИНАХ почти-горизонтальных штрихов; per-row экстракция
кривую там не описывает. Потолок доказан: клики→скелет same-row ≤3px 76-77%.

ШАГ 2 (04.07) — НИТЕВАНИЕ. Итоговый алгоритм SCJP (skeleton-candidates joint pick
+ plateau): per-row кандидаты = концы+центры скелетных интервалов строки → joint-
назначение пары (x_mgz,x_mpz) по prob-каналам ens3 с бонусами толщины (th_w=0.3,
MGZ жирнее) и длины штриха (sl_w=0.2, амплитуда MGZ больше: med 52 vs 16px) →
ПЛАТО вокруг разворотов трассы (±2 строки держат x вершины: клик эксперта
джиттерит на строку от скелет-дуги — потолок same-row 47/56% ≤3px, ±1-строка
70/81%). Гейт 3752: MGZ 22.0 / MPZ 41.0 eff≤3 vs prod ens3 23.0/23.5.

Испытано и ОТБРОШЕНО на 3752 (вердикты, не повторять):
  • жадный обход графа ветвей (walk) и сшивка фрагментов (thread) — глобальный
    путь пера рвётся в столбике слипшихся разворотов у левого упора, одна ошибка
    уводит нить (med 54px); клики закрываются и без пути пера;
  • маятник канальных вершин (pendulum) — 9/13%: детекция вершин med 5-6px
    ЕВКЛИД (dy!), а наклон штрихов ~20px/строку ⇒ dy-погрешность вершины
    фатальна для хорд; метка канала на вершинах MPZ врёт (жирная тушь рядом);
  • пила по огибающим rM/lM per-row (локальные экстремумы) — 15/18%;
  • DP-пила по вершинам с чередованием R/L — 14/27%: цепь дребезжит (1460
    вершин на ~640 реальных), skip-штраф не лечит;
  • толщина туши НЕ разделяет каналы на вершинах (разворот жирнит обоих:
    med 10px у обеих) — работает только как per-row бонус вне разворотов.

  шаг 1:  python mk_pen_path.py <scan.jpg> <etalon.nlgx> [--win d0 d1] [--render]
  шаг 2:  python mk_pen_path.py <scan.jpg> <etalon.nlgx> --step2 --prob <mk_prob2.npy>
            [--win d0 d1] [--ab <traces.npz>] [--render]

--prob: 2-канальная карта infer_mk --save-prob (scale-px, float16); scale выводится
из отношения к скану. --ab: prod-трассы (night_gate ens3 raw) для A/B на тех же кликах.
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


# ---------- ШАГ 2: граф ветвей скелета + нитевание двух путей пера ----------

def skel_degrees(skel):
    S = np.pad(skel, 1).astype(np.uint8)
    d = np.zeros_like(S)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                d[1:-1, 1:-1] += S[1 + dy:S.shape[0] - 1 + dy, 1 + dx:S.shape[1] - 1 + dx]
    return (d[1:-1, 1:-1] * skel).astype(np.int8)


NBR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def build_graph(skel):
    """Граф ветвей: узлы = 8-связные кластеры пикселей deg!=2 (концы и развилки),
    рёбра = цепочки deg=2 между узлами. Возврат nodes, edges, adj, node_of (px->id)."""
    deg = skel_degrees(skel)
    H, W = skel.shape
    node_mask = skel & (deg != 2)
    node_of = {}
    nodes = []
    ys, xs = np.nonzero(node_mask)
    for y, x in zip(ys, xs):
        if (y, x) in node_of:
            continue
        nid = len(nodes)
        stack, px = [(y, x)], []
        node_of[(y, x)] = nid
        while stack:
            cy, cx = stack.pop()
            px.append((cy, cx))
            for dy, dx in NBR8:
                q = (cy + dy, cx + dx)
                if 0 <= q[0] < H and 0 <= q[1] < W and node_mask[q] and q not in node_of:
                    node_of[q] = nid
                    stack.append(q)
        ay = sum(p[0] for p in px) / len(px)
        ax = sum(p[1] for p in px) / len(px)
        nodes.append({"id": nid, "px": px, "cy": ay, "cx": ax})
    edges = []
    adj = {n["id"]: [] for n in nodes}
    visited = np.zeros_like(skel, bool)
    zero_pairs = set()
    for n in nodes:
        for (py, px_) in n["px"]:
            for dy, dx in NBR8:
                q = (py + dy, px_ + dx)
                if not (0 <= q[0] < H and 0 <= q[1] < W) or not skel[q]:
                    continue
                if node_mask[q]:
                    v = node_of[q]
                    if v != n["id"]:
                        key = (min(n["id"], v), max(n["id"], v))
                        if key not in zero_pairs:
                            zero_pairs.add(key)
                            eid = len(edges)
                            edges.append({"u": n["id"], "v": v, "px": [(py, px_), q]})
                            adj[n["id"]].append((eid, True))
                            adj[v].append((eid, False))
                    continue
                if visited[q]:
                    continue
                chain = [(py, px_)]
                prev, cur = (py, px_), q
                while not node_mask[cur]:
                    visited[cur] = True
                    chain.append(cur)
                    nxt = None
                    for dy2, dx2 in NBR8:
                        r = (cur[0] + dy2, cur[1] + dx2)
                        if (0 <= r[0] < H and 0 <= r[1] < W and skel[r]
                                and r != prev and not (visited[r] and not node_mask[r])):
                            nxt = r
                            break
                    if nxt is None:
                        break
                    prev, cur = cur, nxt
                if node_mask[cur]:
                    chain.append(cur)
                    v = node_of[cur]
                else:
                    v = None  # оборванная цепочка (кольцо/дефект) — вешаем как тупик
                eid = len(edges)
                edges.append({"u": n["id"], "v": v, "px": chain})
                adj[n["id"]].append((eid, True))
                if v is not None:
                    adj[v].append((eid, False))
    return nodes, edges, adj


def _dir(px, k=7, from_end=False):
    if len(px) < 2:
        return (0.0, 0.0)
    if from_end:
        a, b = px[max(0, len(px) - 1 - k)], px[-1]
    else:
        a, b = px[0], px[min(k, len(px) - 1)]
    vy, vx = b[0] - a[0], b[1] - a[1]
    n = (vy * vy + vx * vx) ** 0.5
    return (vy / n, vx / n) if n else (0.0, 0.0)


def _edge_prob(probw, px, ch, k=25):
    h, w = probw.shape[1:]
    vals = [probw[ch, min(h - 1, y), min(w - 1, x)] for y, x in px[:k]]
    return float(np.mean(vals)) if vals else 0.0


W_PROB, W_DIR, W_DOWN = 2.0, 1.0, 0.3


def prune_spurs(edges, adj, max_len=5, rounds=3):
    """Спуры скелетизации: короткое ребро с тупиковым/оборванным концом у развилки.
    Гасим (dead) итеративно — жадный проход перестаёт нырять в отростки."""
    def live_deg(nid):
        return sum(1 for eid, _ in adj[nid] if not edges[eid].get("dead"))
    killed = 0
    for _ in range(rounds):
        ch_ = 0
        for e in edges:
            if e.get("dead") or len(e["px"]) > max_len:
                continue
            u, v = e["u"], e["v"]
            du = live_deg(u)
            dv = live_deg(v) if v is not None else 0
            if (v is None and du >= 2) or (v is not None and
                    ((du == 1 and dv >= 3) or (dv == 1 and du >= 3))):
                e["dead"] = True
                ch_ += 1
        killed += ch_
        if not ch_:
            break
    return killed


def make_fragments(edges, probw, min_len=6, min_prob=0.12):
    """Рёбра → фрагменты-полилинии с prob-атрибутами. Фильтр по prob гасит сетку,
    подписи и помарки ДО обхода (у них оба канала ~0) — мусор в графе не мешает."""
    frs = []
    for e in edges:
        if e.get("dead") or len(e["px"]) < min_len:
            continue
        px = e["px"]
        p0 = _edge_prob(probw, px, 0, k=len(px))
        p1 = _edge_prob(probw, px, 1, k=len(px))
        if max(p0, p1) < min_prob:
            continue
        frs.append({"px": px, "p0": p0, "p1": p1})
    return frs


GAP_MAX, PERP_MAX = 150.0, 50.0
W_GAP, W_PERP = 1.0 / 60, 1.0 / 25


def thread(frs, ch, y_stop, start_i):
    """Нить пера: жадная сшивка концов фрагментов (конус вперёд + разворотный конус).
    ВЕРДИКТ 04.07 (3752): НЕ работает — у левого упора развороты обоих перьев
    сливаются в сплошной столбик туши, фрагменты-обрезки там неразличимы, одна
    ошибка сшивки уводит нить навсегда (med 54px, обрыв y=732). Глобальный путь
    пера через месиво не нужен: гейт по кликам закрывается маятником вершин
    (pendulum ниже) — right-вершины чистые, канал решает prob."""
    used = {start_i}
    fr = frs[start_i]
    px = fr["px"] if fr["px"][0][0] <= fr["px"][-1][0] else fr["px"][::-1]
    path = list(px)
    stop = "?"
    def scan(p_end, d_end):
        best = None
        for i, f in enumerate(frs):
            if i in used:
                continue
            for rev in (False, True):
                q = f["px"][-1 if rev else 0]
                vy, vx = q[0] - p_end[0], q[1] - p_end[1]
                proj = vy * d_end[0] + vx * d_end[1]
                gap = (vy * vy + vx * vx) ** 0.5
                if proj < -5 or gap > GAP_MAX:
                    continue
                perp = abs(-vy * d_end[1] + vx * d_end[0])
                if perp > PERP_MAX:
                    continue
                pxf = f["px"][::-1] if rev else f["px"]
                d2 = _dir(pxf, k=10)
                cosv = d_end[0] * d2[0] + d_end[1] * d2[1]
                sc = (W_PROB * (f["p0"] if ch == 0 else f["p1"]) + W_DIR * cosv
                      - W_GAP * gap - W_PERP * perp)
                if best is None or sc > best[0]:
                    best = (sc, i, rev)
        return best
    while True:
        p_end = path[-1]
        d_end = _dir(path, k=10, from_end=True)
        if p_end[0] >= y_stop:
            stop = "y_stop"; break
        best = scan(p_end, d_end)
        if best is None:
            # РАЗВОРОТ: прямой+обратный штрихи сливаются в одну линию скелета —
            # обрыв на вершине продолжается назад по x (вниз по y не отпускаем)
            n = (d_end[0] * d_end[0] * 0.25 + d_end[1] * d_end[1]) ** 0.5 or 1.0
            d_rev = (max(0.2, abs(d_end[0]) * 0.5), -d_end[1] / n)
            m = (d_rev[0] ** 2 + d_rev[1] ** 2) ** 0.5
            best = scan(p_end, (d_rev[0] / m, d_rev[1] / m))
        if best is None:
            stop = f"обрыв@y={p_end[0]}"; break
        _, i, rev = best
        used.add(i)
        path.extend(frs[i]["px"][::-1] if rev else frs[i]["px"])
    return path, stop


TH_W, SL_W = 0.3, 0.2          # бонусы joint-скоринга: толщина (MGZ жирнее), длина штриха


def scjp_candidates(skel, dark, probw):
    """Per-row кандидаты: концы ВСЕХ скелетных интервалов строки + центры длинных
    (вершина часто на конце короткого интервала — центров недостаточно: оракул
    45→47/55→56%). Скоринг (e0,e1) = prob-канал (окно 3×5 max) + бонусы."""
    H, W = skel.shape

    def thick(y, x):
        t = 1; yy = y
        while yy - 1 >= 0 and dark[yy - 1, x] and t < 16: t += 1; yy -= 1
        yy = y
        while yy + 1 < H and dark[yy + 1, x] and t < 16: t += 1; yy += 1
        return t

    def stroke(y, x, cap=80):
        n = 1; xx = x
        while xx - 1 >= 0 and dark[y, xx - 1] and n < cap: xx -= 1; n += 1
        xx = x
        while xx + 1 < W and dark[y, xx + 1] and n < cap: xx += 1; n += 1
        return n

    cands = []
    for y in range(H):
        xs = np.nonzero(skel[y])[0]
        row = []
        if len(xs):
            segs = np.split(xs, np.nonzero(np.diff(xs) > 1)[0] + 1)
            cc = set()
            for seg in segs:
                cc.add(int(seg[0])); cc.add(int(seg[-1]))
                if len(seg) > 4:
                    cc.add(int(seg.mean()))
            for x in sorted(cc):
                p0 = float(probw[0, max(0, y - 1):y + 2, max(0, x - 2):x + 3].max())
                p1 = float(probw[1, max(0, y - 1):y + 2, max(0, x - 2):x + 3].max())
                tn = min(thick(y, x), 10) / 10.0
                sn = min(stroke(y, x), 60) / 60.0
                row.append((x, p0 + TH_W * tn + SL_W * sn,
                               p1 + TH_W * (1 - tn) + SL_W * (1 - sn)))
        cands.append(row)
    return cands


def scjp_joint(cands, thr=0.3, merge_pen=0.15):
    """Совместное назначение пары на кандидатов строки (урок trace_assign:
    независимый argmax берёт один пик / уходит на спурионный)."""
    trm, trp = {}, {}
    for y, E in enumerate(cands):
        if not E:
            continue
        best = None
        for i, (xi, e0i, _) in enumerate(E):
            for j, (xj, _, e1j) in enumerate(E):
                s = e0i + e1j - (merge_pen if i == j else 0.0)
                if best is None or s > best[0]:
                    best = (s, xi, xj, e0i, e1j)
        _, xm, xp, e0, e1 = best
        if e0 > thr:
            trm[y] = float(xm)
        if e1 > thr:
            trp[y] = float(xp)
    return trm, trp


def plateau(tr, prom=5, halfwin=2):
    """Плато вокруг разворотов: клик эксперта джиттерит на ±1 строку от скелет-
    дуги разворота (потолок same-row 47/56%, ±1 70/81%) — строки y*±2 держат x
    вершины вместо хорды с наклоном ~20px/строку."""
    ys = sorted(tr)
    out = dict(tr)
    xs = [tr[y] for y in ys]
    for k in range(1, len(ys) - 1):
        x = xs[k]
        loc = [xs[j] for j in range(max(0, k - 4), min(len(xs), k + 5))]
        if (x >= max(loc) and x - min(loc) >= prom) or \
           (x <= min(loc) and max(loc) - x >= prom):
            for j in range(max(0, k - halfwin), min(len(xs), k + halfwin + 1)):
                if abs(ys[j] - ys[k]) <= halfwin + 1:
                    out[ys[j]] = x
    return out


def pendulum(R, L, probw, ch, dy_max=70):
    """[ОТБРОШЕН, вердикт в шапке] Маятник одного пера: канальные ПРАВЫЕ вершины
    (метка prob с margin) задают период пилы; между соседними правыми — левая
    вершина канала. Провал: dy-погрешность вершин × наклон штрихов."""
    h, w = probw.shape[1:]
    def plab(y, x, side):
        xa, xb = (max(0, x - 8), x + 2) if side == "R" else (max(0, x - 1), min(w, x + 9))
        ya, yb = max(0, y - 3), min(h, y + 4)
        seg = probw[:, ya:yb, xa:xb]
        return float(seg[0].max()), float(seg[1].max())
    mineR = []
    for y, x in R:
        p0, p1 = plab(y, x, "R")
        pm, po = (p0, p1) if ch == 0 else (p1, p0)
        if pm > 0.35 and pm - po > 0.08:
            mineR.append((y, x))
    mineR.sort()
    Llab = []
    for y, x in L:
        p0, p1 = plab(y, x, "L")
        Llab.append((y, x, p0, p1))
    pts = []
    for (ya, xa), (yb, xb) in zip(mineR, mineR[1:]):
        pts.append((ya, xa))
        if yb - ya > dy_max:
            continue                       # далёкая пара — не тот период, без левой
        cands = [(y, x, (p0 if ch == 0 else p1)) for y, x, p0, p1 in Llab
                 if ya < y < yb and (p0 if ch == 0 else p1) > 0.3]
        if cands:
            y, x, _ = max(cands, key=lambda t: t[2])
            pts.append((y, x))
    if mineR:
        pts.append(mineR[-1])
    return pts


def path_turning_points(path, prom=PROM):
    """Вершины пути = развороты по x с prominence>=prom, в порядке прохождения."""
    if len(path) < 3:
        return list(range(len(path)))
    verts = [0]
    ext_i, dirx = 0, 0
    for i in range(1, len(path)):
        x = path[i][1]
        xe = path[ext_i][1]
        if dirx >= 0 and x >= xe:
            ext_i = i
            if dirx == 0 and x - path[verts[-1]][1] >= prom:
                dirx = 1
        elif dirx <= 0 and x <= xe:
            ext_i = i
            if dirx == 0 and path[verts[-1]][1] - x >= prom:
                dirx = -1
        elif dirx > 0 and xe - x >= prom:
            verts.append(ext_i)
            dirx, ext_i = -1, i
        elif dirx < 0 and x - xe >= prom:
            verts.append(ext_i)
            dirx, ext_i = 1, i
    verts.append(len(path) - 1)
    return verts


def chord_trace(pts):
    """Перепараметризация на строки: ломаная по вершинам (как эксперт: клик-вершины,
    NeuraLOG соединяет хордами). Поздний сегмент перезаписывает строку."""
    tr = {}
    for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
        if y1 == y0:
            tr[y0] = float((x0 + x1) / 2)
            continue
        step = 1 if y1 > y0 else -1
        for y in range(y0, y1 + step, step):
            tr[y] = float(x0 + (x1 - x0) * (y - y0) / (y1 - y0))
    return tr


def gate_clicks(clicks, tr, y_off, x_off, tag):
    """Гейт по кликам эксперта (native): |dx| на строках кликов. tr — оконные коорд."""
    dx, miss = [], 0
    for y, x, _ in clicks:
        xo = tr.get(y - y_off)
        if xo is None:
            miss += 1
            continue
        dx.append(abs(xo + x_off - x))
    dx = np.array(dx)
    n = len(clicks)
    if not n:
        print(f"  {tag}: нет кликов")
        return 0.0
    eff = float((dx <= 3).sum() / n) if len(dx) else 0.0
    print(f"  {tag:<22} n={n} дыр={miss} ({miss/n*100:.0f}%) | "
          + (f"med {np.median(dx):.1f}px | ≤3px {(dx<=3).mean()*100:.0f}% | ≤5px {(dx<=5).mean()*100:.0f}% "
             f"| >15px {(dx>15).mean()*100:.0f}% | " if len(dx) else "") + f"eff≤3 {eff*100:.1f}%")
    return eff


def load_prob_window(prob_path, Hs, y0, y1, x0, x1):
    """mk_prob2.npy (2,H*S,W*S) → окно native (2,h,w) max-pool'ом блоков SxS."""
    p = np.load(prob_path, mmap_mode="r")
    S = max(1, round(p.shape[1] / Hs))
    pw = np.asarray(p[:, y0 * S:y1 * S, x0 * S:x1 * S], np.float32)
    if S > 1:
        h, w = y1 - y0, x1 - x0
        pw = pw[:, :h * S, :w * S].reshape(2, h, S, w, S).max((2, 4))
    return pw, S


def step2(scan, et_path, prob_path, args):
    render = "--render" in args
    me = extract(et_path)
    da = me["depth_axis"]
    def yof(d): return int(da["top_y"] + (d - da["top_depth"]) * da["span_px"] / da["span_depth"])
    if "--win" in args:
        i = args.index("--win")
        y0, y1 = yof(float(args[i + 1])), yof(float(args[i + 2]))
    else:
        y0, y1 = int(da["top_y"]), int(da["bottom_y"])
    gts = {c["name"].split()[0]: trace_dict(c) for c in me["curves"]}
    cl_m = [(y, x, "MGZ1") for y, x in gts.get("MGZ1", {}).items() if y0 <= y < y1]
    cl_p = [(y, x, "MPZ1") for y, x in gts.get("MPZ1", {}).items() if y0 <= y < y1]
    if not cl_m and not cl_p:
        print("нет кликов в окне"); return
    xs_all = [x for _, x, _ in cl_m + cl_p]
    x0, x1 = max(0, int(min(xs_all)) - 40), int(max(xs_all)) + 40

    rgb = np.asarray(Image.open(scan).convert("RGB"))
    band = rgb[y0:y1, x0:x1]
    dark = band.max(2) < DARK
    skel = zhang_suen(dark)
    probw, S = load_prob_window(prob_path, rgb.shape[0], y0, y1, x0, x1)
    print(f"окно y[{y0}..{y1}] x[{x0}..{x1}] | скелет {skel.sum()}px | prob S={S} "
          f"MGZ>0.4 {(probw[0]>0.4).mean()*100:.1f}% MPZ>0.4 {(probw[1]>0.4).mean()*100:.1f}%")

    cands = scjp_candidates(skel, dark, probw)
    tm, tp = scjp_joint(cands)
    ta, tb = plateau(tm), plateau(tp)
    cov_a = len(ta) / max(1, y1 - y0)
    cov_b = len(tb) / max(1, y1 - y0)
    print(f"SCJP: кандидатов {sum(len(r) for r in cands)} | "
          f"покрытие строк {cov_a*100:.0f}% / {cov_b*100:.0f}%")

    print("ГЕЙТ (клики эксперта, окно):")
    ea = gate_clicks(cl_m, ta, y0, x0, "skel MGZ")
    eb = gate_clicks(cl_p, tb, y0, x0, "skel MPZ")
    gate_clicks(cl_m, tb, y0, x0, "swap MGZ<-B")
    gate_clicks(cl_p, ta, y0, x0, "swap MPZ<-A")
    if "--ab" in args:
        ab = np.load(args[args.index("--ab") + 1])
        am = {int(y): float(x) for y, x in zip(ab["mgz_y"], ab["mgz_x"])}
        ap = {int(y): float(x) for y, x in zip(ab["mpz_y"], ab["mpz_x"])}
        print("A/B prod (те же клики):")
        gate_clicks(cl_m, {y - y0: x - x0 for y, x in am.items()}, y0, x0, "prod MGZ")
        gate_clicks(cl_p, {y - y0: x - x0 for y, x in ap.items()}, y0, x0, "prod MPZ")

    if render:
        h = min(y1 - y0, 1400)
        crop = Image.fromarray(band[:h]).convert("RGB")
        dr = ImageDraw.Draw(crop)
        sy, sx = np.nonzero(skel[:h])
        for yy, xx in zip(sy, sx):
            dr.point([(int(xx), int(yy))], fill=(0, 170, 0))
        for tr, col in ((ta, (230, 40, 40)), (tb, (40, 80, 230))):
            for y, x in tr.items():
                if 0 <= y < h:
                    dr.point([(int(x), y), (int(x) + 1, y)], fill=col)
        for y, x, mn in cl_m + cl_p:
            vy, vx = y - y0, x - x0
            if vy < h:
                col = (220, 30, 30) if mn == "MGZ1" else (30, 80, 230)
                dr.ellipse([vx - 5, vy - 5, vx + 5, vy + 5], outline=col, width=2)
        stem = os.path.splitext(os.path.basename(scan))[0]
        fn = rf"F:\nds\output\mk_data\export_qc\{stem}_penpath_step2.png"
        crop.save(fn)
        print("рендер:", fn)
    return {"eff_mgz": ea, "eff_mpz": eb}


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    scan, et_path = a[0], a[1]
    if "--step2" in a:
        step2(scan, et_path, a[a.index("--prob") + 1], a)
        return
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
