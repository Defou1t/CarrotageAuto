"""§6.356: уколы трассы на цифры глубины / клетки сетки — ≤ L строк отлёта дальше D px от хорды соседей, которые близки друг к другу
(≤ E px); под уколом туши по сети нет (`auto.ink_seg`, max < ts), на хорде есть (среднее > tc) ⇒ строки укола по хорде.
Настоящий острый пик (БК, ИК) сеть видит тушью — его не трогаем (`digitizer/rnd/_spike_ink.py`: 150 листов, ловит 41% ложных при
1.1% задетых настоящих). Один код у стенда (`_strict_stand.py`, вычистка `spikeseg`) и у прода (`emit._spike_pass`)."""
import numpy as np

L, D, E, TS, TC = 3, 20.0, 15.0, 0.1, 0.5


def spikes(tr, L=L, D=D, E=E):
    """уколы трассы {y: x} → (ys, xs, [(i, j)]) — индексы строк укола в sorted(tr) (как `_strict_stand.f_spike`)"""
    ys = sorted(tr); xs = np.array([tr[y] for y in ys], float); n = len(ys); i = 1; out = []
    while i < n - 1:
        hit = None
        for j in range(i, min(n - 1, i + L)):
            if ys[j + 1] - ys[i - 1] != j - i + 2:
                break
            a, b = xs[i - 1], xs[j + 1]
            if abs(a - b) > E:
                continue
            t = (np.array(ys[i:j + 1]) - ys[i - 1]) / (ys[j + 1] - ys[i - 1])
            if np.all(np.abs(xs[i:j + 1] - (a + (b - a) * t)) > D):
                hit = j
        if hit is None:
            i += 1; continue
        out.append((i, hit)); i = hit + 1
    return ys, xs, out


def seg_at(rgb, net, ys, xs):
    """тушь сети в точках (ys, xs) — окно по рамке точек, плитка 192 (± 64 px контекста), max в ±1 px по x"""
    from . import ink_seg as IS
    y0, y1 = min(ys) - 2, max(ys) + 3; x0, x1 = int(min(xs)) - 4, int(max(xs)) + 5
    pm = IS.prob_map(rgb, y0, y1, x0, x1, net, tile=192)
    yy = np.clip(np.asarray(ys) - max(0, y0), 0, pm.shape[0] - 1); xx = np.round(np.asarray(xs)).astype(int) - max(0, x0)
    return np.max(np.stack([pm[yy, np.clip(xx + o, 0, pm.shape[1] - 1)] for o in (-1, 0, 1)]), 0)


def fix(tr, rgb, net, ts=TS, tc=TC, L=L, D=D, E=E, Lmin=0):
    """трасса {y: x} → (трасса с уколами без туши по хорде, [(y0, y1, туши у укола, туши у хорды)] замен); Lmin — только длиннее"""
    ys, xs, sp = spikes(tr, L, D, E)
    out = dict(tr); log = []
    for i, j in sp:
        if j - i + 1 <= Lmin:
            continue
        a, b, ya, yb = xs[i - 1], xs[j + 1], ys[i - 1], ys[j + 1]
        yr = list(ys[i:j + 1]); ch = [a + (b - a) * (y - ya) / (yb - ya) for y in yr]
        v = seg_at(rgb, net, yr + yr, list(xs[i:j + 1]) + ch)
        s_, c_ = float(np.max(v[:len(yr)])), float(np.mean(v[len(yr):]))
        if s_ < ts and c_ > tc:
            for y, c in zip(yr, ch):
                out[y] = c
            log.append((yr[0], yr[-1], round(s_, 3), round(c_, 3)))
    return out, log


def fix_all(tr, rgb, net, ts=TS, tc=TC, long_=0):
    """уколы ≤ 3 строк, затем (long_ > 3) отлёты 4…long_ строк — обвод цифр глубины (`_spike_ink.py --L 15 --Lmin 3`: 150 листов,
    правило то же — ложных 100%, задето настоящих 0.0%, поймано ложных 25%)"""
    out, log = fix(tr, rgb, net, ts, tc)
    if long_ > L:
        out, lg2 = fix(out, rgb, net, ts, tc, L=long_, Lmin=L)
        log += lg2
    return out, log
