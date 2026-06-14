r"""
resolve_swaps.py — v2 пост-хок развязка свопов идентичности на пересечениях
(реализация идеи пользователя: у линии есть постоянный ХАРАКТЕР — гладкая/спайковая).
Работает поверх ВЫХОДОВ v1-трекера (кэш cache_tracks.py), на CPU:
  1) строим x(строка) для каждой линии;
  2) находим пересечения = смена знака (xᵢ−xⱼ);
  3) на каждом сравниваем шероховатость сегментов ДО/ПОСЛЕ; если перестановка хвостов
     даёт более само-консистентный характер (rough_before ≈ rough_after для каждой
     линии), переставляем хвосты. Несколько проходов (каскадные пересечения).
Оценка: медиана |dx| к GT ДО и ПОСЛЕ развязки.

python resolve_swaps.py [--cache DIR] [--win 60] [--margin 0.5]
"""
import sys, pickle, glob
from pathlib import Path
import numpy as np


def to_array(tr, y0, y1):
    a = np.full(y1 - y0, np.nan)
    for y, x in tr.items():
        if y0 <= y < y1:
            a[y - y0] = x
    return a


def rough(a, lo, hi, k=15):
    lo = max(0, lo); hi = min(len(a), hi)
    seg = a[lo:hi]
    seg = seg[~np.isnan(seg)]
    if len(seg) < k + 2:
        return None
    sm = np.convolve(seg, np.ones(k)/k, mode="same")
    return float(np.std(seg - sm))


def resolve(arrs, win=60, margin=0.5, passes=4):
    n = len(arrs)
    for _ in range(passes):
        crossings = []
        for i in range(n):
            for j in range(i+1, n):
                d = arrs[i] - arrs[j]
                idx = np.where(~np.isnan(d))[0]
                for k in range(1, len(idx)):
                    r0, r1 = idx[k-1], idx[k]
                    if r1 - r0 > 5 or d[r0] == 0:
                        continue
                    if np.sign(d[r0]) != np.sign(d[r1]):
                        crossings.append((r1, i, j))
        crossings.sort()
        did = False
        for yc, i, j in crossings:
            ai, aj = arrs[i], arrs[j]
            rib, ria = rough(ai, yc-win, yc), rough(ai, yc, yc+win)
            rjb, rja = rough(aj, yc-win, yc), rough(aj, yc, yc+win)
            if None in (rib, ria, rjb, rja):
                continue
            keep = abs(rib-ria) + abs(rjb-rja)
            swap = abs(rib-rja) + abs(rjb-ria)
            if swap + margin < keep:
                tmp = ai[yc:].copy(); ai[yc:] = aj[yc:]; aj[yc:] = tmp
                did = True
        if not did:
            break
    return arrs


def match_err(arrs, gts, y0):
    """для каждой GT — лучшая линия (min медиана |dx|) на общих строках."""
    out = []
    for name, gd in gts:
        best = None
        for a in arrs:
            errs = [abs(a[y-y0]-x) for y, x in gd.items()
                    if 0 <= y-y0 < len(a) and not np.isnan(a[y-y0])]
            if len(errs) < 15:
                continue
            e = float(np.median(errs))
            if best is None or e < best:
                best = e
        out.append((name, best))
    return out


def main():
    args = sys.argv[1:]
    cache = Path(r"F:\nds\output\unet_data\trackcache"); win = 60; margin = 0.5
    i = 0
    while i < len(args):
        if args[i] == "--cache": cache = Path(args[i+1]); i += 2
        elif args[i] == "--win": win = int(args[i+1]); i += 2
        elif args[i] == "--margin": margin = float(args[i+1]); i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8")
    pkls = sorted(glob.glob(str(cache / "*.pkl")))
    if not pkls:
        print(f"нет кэша в {cache} — сначала cache_tracks.py"); return

    tot_before, tot_after = [], []
    for p in pkls:
        rec = pickle.load(open(p, "rb"))
        y0, y1 = rec["y0"], rec["y1"]
        arrs0 = [to_array(tr, y0, y1) for tr in rec["lines"]]
        before = match_err([a.copy() for a in arrs0], rec["gts"], y0)
        arrs = resolve([a.copy() for a in arrs0], win=win, margin=margin)
        after = match_err(arrs, rec["gts"], y0)
        print(f"\n{rec['stem'][:48]}  (N={rec['N']}, GT={len(rec['gts'])})")
        print(f"   {'curve':<9}{'before':>9}{'after':>9}")
        for (nm, b), (_, a) in zip(before, after):
            mark = ""
            if b is not None and a is not None:
                if a < b - 3: mark = "  ✓ лучше"
                elif a > b + 3: mark = "  ✗ хуже"
                tot_before.append(b); tot_after.append(a)
            print(f"   {nm:<9}{('%.1f'%b) if b else '-':>9}{('%.1f'%a) if a else '-':>9}{mark}")
    if tot_before:
        import numpy as _np
        print(f"\n=== ИТОГ по {len(tot_before)} кривым ===")
        print(f"медиана |dx|: ДО {_np.median(tot_before):.1f}px → ПОСЛЕ {_np.median(tot_after):.1f}px")
        print(f"кривых улучшено (>3px): {sum(a<b-3 for a,b in zip(tot_after,tot_before))}, "
              f"ухудшено: {sum(a>b+3 for a,b in zip(tot_after,tot_before))}")


if __name__ == "__main__":
    main()
