r"""§6.355: ОБЪЕЗДЫ — кусок кандидата ведения вместо ухода трассы выдачи на чужую линию.

Окно — участок, где кривая выдачи w и кандидат t (прод-трасса, декодер, доп. пути) расходятся (> NEAR px), а до и после (K строк)
совпадают. Признаки обоих кусков (тушь, цвет против всей кривой, прыжки без туши, наклон на стыках, дрожание, близость других кривых
выдачи, вертикаль, край шкалы; сеть туши — среднее, доля > 0.5, минимум скользящего среднего), бустинг P(кусок t верен). Замена окон
с P > τ, перекрытия — по наибольшей P. Стенд и обучение — `digitizer/rnd/_detour.py` (берёт окна и признаки отсюда).
"""
import json
import pickle
from pathlib import Path

import numpy as np

_MODELS = Path(__file__).resolve().parent / "models"
J, DARK, K, NEAR = 8, 45, 6, 3.0
PF = ["n", "ink", "inkrow", "colr", "jit", "maxj", "nj_dry", "dup", "vcol", "edge", "s_in", "s_out"]
FEATS = [f"w_{f}" for f in PF] + [f"t_{f}" for f in PF] + [f"d_{f}" for f in PF] + ["sep", "logn"]
_CACHE = {}


def _ink(gray, paper, y, xa, xb):
    lo, hi = int(min(xa, xb)) + 2, int(max(xa, xb)) - 2
    if hi <= lo or not (0 <= y < gray.shape[0] - 1):
        return 1.0
    lo, hi = max(0, lo), min(gray.shape[1], hi)
    seg = np.minimum(gray[y, lo:hi], gray[y + 1, lo:hi]).astype(float)
    return float(np.mean(seg < paper - DARK)) if len(seg) else 1.0


def _slope(tr, ys):
    ys = [y for y in ys if y in tr]
    if len(ys) < 3:
        return 0.0
    return float(np.polyfit(ys, [tr[y] for y in ys], 1)[0])


def seg_map(rgb, ys_xs, net, pad=6):
    """карта сети туши (`auto.ink_seg`) по рамке всех трасс — полосами по сетке плиток (как `_seg_feat.py`), float16 →
    функция (ys, xs) → значения max в ±1 px"""
    from . import ink_seg as IS
    H, W = rgb.shape[:2]
    y0 = max(0, min(y for y, _ in ys_xs)); y1 = min(H, max(y for y, _ in ys_xs) + 1)
    x0 = max(0, int(min(x for _, x in ys_xs)) - pad); x1 = min(W, int(max(x for _, x in ys_xs)) + pad + 1)
    M = np.zeros((y1 - y0, x1 - x0), np.float16); band = (1024 - 2 * 64) * 8
    for s0 in range(y0, y1, band):
        s1 = min(y1, s0 + band)
        M[s0 - y0:s1 - y0] = IS.prob_map(rgb, s0, s1, x0, x1, net).astype(np.float16)

    def at(ys, xs):
        yy = np.clip(np.asarray(ys) - y0, 0, M.shape[0] - 1); xx = np.round(np.asarray(xs)).astype(int) - x0
        return np.max(np.stack([M[yy, np.clip(xx + o, 0, M.shape[1] - 1)] for o in (-1, 0, 1)]), 0).astype(np.float32)
    return at


def sheet_seg(rgb, wr, net):
    """карта сети туши по трассам выдачи листа (точки через 50 строк + поля 60 px по x) — как при извлечении стенда"""
    H, W = rgb.shape[:2]
    pts = [(y, x) for w in wr.values() for y, x in list(w.items())[::50]]
    if not pts:
        return None
    return seg_map(rgb, pts + [(min(y for y, _ in pts), max(0, min(x for _, x in pts) - 60)),
                               (max(y for y, _ in pts), min(W - 1, max(x for _, x in pts) + 60))], net)


def look(rgb):
    """(gray = min по каналам, paper, rgb_s, H, W) — как при извлечении стенда"""
    gray = rgb.min(axis=2) if rgb.ndim == 3 else rgb
    if rgb.ndim == 3:
        rgb_s = lambda yy, xx: rgb[yy, xx].astype(np.float32)
    else:
        rgb_s = lambda yy, xx: np.repeat(gray[yy, xx][:, None], 3, 1).astype(np.float32)
    paper = float(np.percentile(gray[::7, ::7], 80)); H, W = gray.shape
    return gray, paper, rgb_s, H, W


def piece_feats(tr, a, b, ctx, others, axis, seg=None):
    """признаки куска tr на строках [a, b]; ctx — (gray, paper, rgb_s, ref_color, H, W); seg — карта сети туши (или None)"""
    gray, paper, rgb_s, ref, H, W = ctx
    rr = [y for y in range(a, b + 1) if y in tr]
    if len(rr) < 3:
        return None
    xs = np.array([tr[y] for y in rr]); yi = np.clip(np.array(rr), 0, H - 1); xi = np.clip(np.round(xs).astype(int), 1, W - 2)
    under = np.minimum(np.minimum(gray[yi, xi - 1], gray[yi, xi]), gray[yi, xi + 1]).astype(float)
    dk = paper - under
    sel = slice(None, None, max(1, len(rr) // 150))
    colr = float(np.linalg.norm(np.median(rgb_s(yi[sel], xi[sel]), 0) - ref))
    dx = np.abs(np.diff(xs)) if len(xs) > 1 else np.zeros(1)
    dry = 0
    for i in np.flatnonzero(dx > J)[:50]:
        dry += _ink(gray, paper, rr[i], xs[i], xs[i + 1]) < 0.5
    dup = 0.0
    for o in others:
        v = [abs(o[y] - tr[y]) <= NEAR for y in rr[::3] if y in o]
        if v:
            dup = max(dup, float(np.mean(v)))
    xm = float(np.median(xs)); yc = (a + b) // 2
    ca, cb = max(0, yc - 400), min(H, yc + 400); xc = int(np.clip(round(xm), 1, W - 2))
    vcol = float(np.mean(paper - gray[ca:cb:4, xc - 1:xc + 2].min(1) > DARK))
    edge = min(abs(xm - axis["x_left"]), abs(xm - axis["x_right"])) if axis else np.nan
    s_in = abs(_slope(tr, range(a, a + K)) - _slope(tr, range(a - K, a)))
    s_out = abs(_slope(tr, range(b - K + 1, b + 1)) - _slope(tr, range(b + 1, b + 1 + K)))
    f = [len(rr), float(np.mean(dk)), float(np.mean(dk > DARK)), colr, float(np.mean(dx)), float(dx.max()), dry, dup, vcol, edge,
         s_in, s_out]
    if seg is not None:
        sv = seg(rr[::2], xs[::2])
        f += [float(np.mean(sv)), float(np.mean(sv > 0.5)), float(np.min(np.convolve(sv, np.ones(5) / 5, "valid"))) if len(sv) >= 5 else float(np.min(sv))]
    return f


def windows(w, t, k=K, minlen=5, maxlen=3000):
    """окна расхождения w и t (> NEAR px) с совпадением (≤ NEAR) на ≥ k/2 из k строк до и после → [(a, b)]"""
    ys = [y for y in sorted(w) if y in t]
    if len(ys) < 50:
        return []
    agree = {y: abs(w[y] - t[y]) <= NEAR for y in ys}
    out, i = [], 0
    while i < len(ys):
        if agree[ys[i]]:
            i += 1; continue
        j = i
        while j + 1 < len(ys) and not agree[ys[j + 1]] and ys[j + 1] - ys[j] <= 2:
            j += 1
        a, b = ys[i], ys[j]
        if minlen <= b - a + 1 <= maxlen:
            pre = sum(1 for y in range(a - k, a) if agree.get(y)); post = sum(1 for y in range(b + 1, b + 1 + k) if agree.get(y))
            if pre >= k // 2 and post >= k // 2:
                out.append((a, b))
        i = j + 1
    return out


def curve_windows(g, w, wr, C, lk, axis, seg):
    """окна кривой выдачи g (трасса w) со всеми кандидатами C → [(ci, a, b, вектор признаков)]"""
    gray, paper, rgb_s, H, W = lk
    if len(w) < 50:
        return []
    ys = sorted(w)[::max(1, len(w) // 3000)]
    yi = np.clip(np.array(ys), 0, H - 1); xi = np.clip(np.round([w[y] for y in ys]).astype(int), 1, W - 2)
    ref = np.median(rgb_s(yi, xi), 0)
    ctx = (gray, paper, rgb_s, ref, H, W)
    others = [o for kk, o in wr.items() if kk != g]
    seen, out = set(), []
    for ci, t in enumerate(C):
        if sum(1 for y in ys if y in t and abs(t[y] - w[y]) <= NEAR) > 0.97 * len(ys):
            continue                                    # сама трасса выдачи (или её копия)
        for a, b in windows(w, t):
            if (a, b, round(t.get(a, 0))) in seen:
                continue
            seen.add((a, b, round(t.get(a, 0))))
            fw = piece_feats(w, a, b, ctx, others, axis, seg); ft = piece_feats(t, a, b, ctx, others, axis, seg)
            if fw is None or ft is None:
                continue
            sep = float(np.median([abs(w[y] - t[y]) for y in range(a, b + 1) if y in w and y in t]))
            out.append((ci, a, b, np.array(fw + ft + list(np.array(fw) - np.array(ft)) + [sep, np.log1p(b - a + 1)], np.float32)))
    return out


def take(scored, tau, ok=None):
    """[(p, ci, a, b)] → окна к замене: P > τ, ok(ci, a, b) (если задан), без перекрытий, по убыванию P"""
    taken = []
    for p, ci, a, b in sorted(scored, key=lambda z: -z[0]):
        if p <= tau:
            break
        if any(not (b < x0 or a > x1) for _, x0, x1 in taken):
            continue
        if ok is not None and not ok(ci, a, b):
            continue
        taken.append((ci, a, b))
    return taken


def max_jump(w, t, a, b):
    """наибольший |Δx| соседних строк (разрыв ≤ 2) на [a − 1, b + 1] у w и у w с куском t на [a, b] → (до, после)"""
    ys = [y for y in range(a - 1, b + 2) if y in w or (a <= y <= b and y in t)]
    x0 = [w.get(y, t.get(y)) for y in ys]
    x1 = [t[y] if (a <= y <= b and y in t) else w.get(y, t.get(y)) for y in ys]
    m0 = m1 = 0.0
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] <= 2:
            if x0[i] is not None and x0[i - 1] is not None:
                m0 = max(m0, abs(x0[i] - x0[i - 1]))
            m1 = max(m1, abs(x1[i] - x1[i - 1]))
    return m0, m1


def jump_ok(w, t, a, b, G):
    """§6.355 (разбор глазами D19): кусок кандидата не вносит прыжка больше G px, которого у выдачи там не было; G ≤ 0 — без запрета"""
    if G <= 0:
        return True
    m0, m1 = max_jump(w, t, a, b)
    return m1 <= max(G, m0)


def resolve(spec, sheet=None):
    """модель: имя в auto/models, путь, или «oof:<каталог>» — модель фолда листа (`detour_folds.json`; иначе `detour_all.pkl`)"""
    if not spec:
        return None
    if spec.startswith("oof:"):
        d = Path(spec[4:])
        mp = d / "detour_folds.json"
        if str(mp) not in _CACHE:
            _CACHE[str(mp)] = json.loads(mp.read_text(encoding="utf-8"))
        k = _CACHE[str(mp)].get(sheet)
        return d / (f"detour_f{k}.pkl" if k is not None else "detour_all.pkl")
    p = Path(spec)
    if not p.is_absolute():
        p = _MODELS / spec
    return p if p.is_file() else None


def load(path):
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = pickle.load(open(path, "rb"))
    return _CACHE[key]
