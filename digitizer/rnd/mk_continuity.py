r"""Continuity-постобработка разделителя MK (v2, ХИРУРГИЧЕСКАЯ): база = peak-экстракция
(следование 3-15px, её НЕ трогаем), continuity чинит только два известных провала:

  A. СХЛОПЫВАНИЕ (диагностика 29.06: оба канала argmax на одной кривой, GT-разделение 78-90px)
     → мост: в схлопнутом ране ищем вторую прядь по пикам combined prob, марш от позиций
     покинутой кривой на границах рана (вход/выход) с проверкой попаданий и стыковки;
  B. ЧЕРЕДОВАНИЕ идентичности (LEVEN) → идентичность per-BLOCK (блоки между слияниями)
     голосом канальных prob вместо per-row argmax; слабый голос → наследуем от соседнего
     блока по непрерывности (min-jerk сквозь слияние, [[gz-crossing-behavioral-continuity]]).

Урок v1 (полная замена DP-трекингом, отвергнута eval 02.07): глобальный голос переворачивает
хорошо-размеченные скважины (BEZLUD 8.6→30% swap), identity-free DP уводит прядь на паразитную
тушь при реальных слияниях (BOGAT MPZ 8.7→53px). Хирургия монотоннее: sep-строки не трогаем.
"""
import numpy as np


def _peak(pr, x0, thr=0.4, win=8):
    """Пик канала в строке: argmax + субпиксельный центроид ±win (как trace_ch в infer_mk)."""
    if pr.max() <= thr:
        return None
    a = int(np.argmax(pr)); lo, hi = max(0, a - win), min(len(pr), a + win + 1)
    w = pr[lo:hi]
    return float(((np.arange(lo, hi) + x0) * w).sum() / w.sum())


def row_candidates(p0r, p1r, x0, thr=0.25, min_sep=3, kmax=4, sub=3):
    """Пики max(p0,p1) в строке: [(x_subpx, pcm)], сильные первыми, NMS min_sep."""
    pm = np.maximum(p0r, p1r)
    n = len(pm)
    if n < 3 or pm.max() <= thr:
        return []
    loc = np.nonzero((pm[1:-1] >= thr) & (pm[1:-1] >= pm[:-2]) & (pm[1:-1] > pm[2:]))[0] + 1
    if len(loc) == 0:
        return []
    loc = loc[np.argsort(-pm[loc])]
    keep = []
    for a in loc:
        if all(abs(a - b) >= min_sep for b in keep):
            keep.append(int(a))
        if len(keep) >= kmax:
            break
    ps = p0r + p1r
    out = []
    for a in keep:
        lo, hi = max(0, a - sub), min(n, a + sub + 1)
        w = ps[lo:hi]
        out.append((x0 + float((np.arange(lo, hi) * w).sum() / max(w.sum(), 1e-6)), float(pm[a])))
    return out


def _bridge(run_ys, xc_of, u_in, y_in, u_out, y_out, cand_of, S=1):
    """Мост покинутой кривой через схлопнутый ран: марш от u_in к u_out по кандидатам.
    Возврат {y: u} или None (ран = реальное слияние). Гейты: ≥50% попаданий, стыковка с u_out,
    мост реально отделён от схлопнутой линии (иначе нашли ту же тушь)."""
    slope = (u_out - u_in) / max(y_out - y_in, 1)
    if abs(slope) > 8.0:                                          # вход/выход не соединить кривой — не мост
        return None
    u, py = u_in, y_in
    hits, path, seps = 0, {}, []
    for y in run_ys:
        g = y - py
        pred = u + slope * g
        tol = 6.0 * S + 2.0 * g
        best = None
        for x, p in cand_of(y):
            if abs(x - pred) <= tol and (best is None or abs(x - pred) < abs(best - pred)):
                best = x
        if best is not None:
            u = best; hits += 1
        else:
            u = pred
        path[y] = u; seps.append(abs(u - xc_of(y))); py = y
    if hits < 0.5 * len(run_ys):
        return None
    if abs(u + slope * (y_out - py) - u_out) > 12.0 * S:          # не состыковался с выходом
        return None
    if np.median(seps) <= 3.0 * S:                                # «мост» лёг на ту же линию
        return None
    return path


def link_strands(rows, XL, XR, S=1, ema=0.3, vmax=26.0, merge_eps=1.5):
    """Пары (xl,xr) → пряди A/B: straight/cross по предсказанию скоростью; скорость
    сохраняется (с затуханием) сквозь слияния → min-jerk выход из наложения."""
    R = len(rows)
    xa = np.zeros(R); xb = np.zeros(R)
    xa[0], xb[0] = XL[0], XR[0]
    va = vb = 0.0
    for r in range(1, R):
        g = rows[r] - rows[r - 1]
        pa = xa[r - 1] + np.clip(va, -vmax, vmax) * g
        pb = xb[r - 1] + np.clip(vb, -vmax, vmax) * g
        l, h = XL[r], XR[r]
        if abs(h - l) <= merge_eps * S:
            xa[r] = xb[r] = (l + h) / 2
            va *= 0.9; vb *= 0.9
            continue
        if abs(l - pa) + abs(h - pb) <= abs(h - pa) + abs(l - pb):
            xa[r], xb[r] = l, h
        else:
            xa[r], xb[r] = h, l
        va = (1 - ema) * va + ema * np.clip((xa[r] - xa[r - 1]) / g, -vmax, vmax)
        vb = (1 - ema) * vb + ema * np.clip((xb[r] - xb[r - 1]) / g, -vmax, vmax)
    merged = np.abs(np.asarray(XR) - np.asarray(XL)) <= merge_eps * S
    return xa, xb, merged


def continuity_traces(prob, y0, y1, x0, x1, S=1, thr=0.4, thr2=0.25, min_conf=1.0):
    """Главный вход: prob (2×H×W, scaled) → (mgz {y:x}, mpz {y:x}, info) в scaled-координатах."""
    p0, p1 = prob[0], prob[1]
    eps = 3.0 * S
    # --- 1. per-row пики каналов (как peak-база) ---
    rows, PM, PP = [], {}, {}
    for y in range(y0, y1):
        m = _peak(p0[y, x0:x1], x0, thr)
        p = _peak(p1[y, x0:x1], x0, thr)
        if m is None and p is None:
            continue
        rows.append(y); PM[y] = m; PP[y] = p
    if not rows:
        return {}, {}, {"rows": 0}
    sep = {y: (PM[y] is not None and PP[y] is not None and abs(PM[y] - PP[y]) > eps) for y in rows}
    # --- 2. мосты через схлопнутые раны (только между sep-границами) ---
    cand_cache = {}
    def cand_of(y):
        if y not in cand_cache:
            cand_cache[y] = row_candidates(p0[y, x0:x1], p1[y, x0:x1], x0, thr=thr2, min_sep=int(3 * S))
        return cand_cache[y]
    def xc_at(y):
        vals = [v for v in (PM[y], PP[y]) if v is not None]
        return sum(vals) / len(vals)
    bridges = {}                                                  # y -> x второй пряди
    n_runs = n_bridged = 0
    i = 0
    while i < len(rows):
        if sep[rows[i]]:
            i += 1; continue
        j = i
        while j < len(rows) and not sep[rows[j]]:
            j += 1
        run = rows[i:j]                                           # схлопнутый/одиночный ран
        n_runs += 1
        if i > 0 and j < len(rows) and len(run) >= 3:
            ei, eo = rows[i - 1], rows[j]                         # sep-границы рана
            x_first, x_last = xc_at(run[0]), xc_at(run[-1])
            u_in = PM[ei] if abs(PM[ei] - x_first) > abs(PP[ei] - x_first) else PP[ei]
            u_out = PM[eo] if abs(PM[eo] - x_last) > abs(PP[eo] - x_last) else PP[eo]
            if abs(u_in - x_first) > 2 * eps and abs(u_out - x_last) > 2 * eps:
                path = _bridge(run, xc_at, u_in, ei, u_out, eo, cand_of, S=S)
                if path:
                    bridges.update(path); n_bridged += 1
        i = j
    # --- 3. геометрические пары + линкер прядей ---
    XL, XR = [], []
    for y in rows:
        if sep[y]:
            a, b = PM[y], PP[y]
        elif y in bridges:
            a, b = xc_at(y), bridges[y]
        else:
            a = b = xc_at(y)
        XL.append(min(a, b)); XR.append(max(a, b))
    xa, xb, merged = link_strands(rows, XL, XR, S=S)
    # --- 4. идентичность per-BLOCK (блоки между слияниями/большими разрывами) ---
    def ev(y, x, ch):
        xi = int(round(x)); lo, hi = max(x0, xi - 2), min(x1, xi + 3)
        return float((p0 if ch == 0 else p1)[y, lo:hi].mean())
    blocks = []                                                   # (i0, i1) полуинтервалы индексов
    st = 0
    for r in range(1, len(rows)):
        if merged[r] or rows[r] - rows[r - 1] > 20:
            if r > st:
                blocks.append((st, r))
            st = r
    blocks.append((st, len(rows)))
    mgz, mpz = {}, {}
    prev_a_is_mgz = True
    n_flip = 0
    for i0, i1 in blocks:
        score = 0.0
        for r in range(i0, i1):
            if merged[r] or abs(xa[r] - xb[r]) <= 2 * eps:
                continue
            y = rows[r]
            score += (ev(y, xa[r], 0) + ev(y, xb[r], 1)) - (ev(y, xb[r], 0) + ev(y, xa[r], 1))
        a_is_mgz = prev_a_is_mgz if abs(score) < min_conf else (score >= 0)
        if a_is_mgz != prev_a_is_mgz:
            n_flip += 1
        prev_a_is_mgz = a_is_mgz
        for r in range(i0, i1):
            y = rows[r]
            mgz[y], mpz[y] = (float(xa[r]), float(xb[r])) if a_is_mgz else (float(xb[r]), float(xa[r]))
    info = {"rows": len(rows), "sep_frac": round(float(np.mean([sep[y] for y in rows])), 3),
            "runs": n_runs, "bridged": n_bridged, "bridge_rows": len(bridges),
            "blocks": len(blocks), "flips": n_flip,
            "merged_frac": round(float(merged.mean()), 3)}
    return mgz, mpz, info
