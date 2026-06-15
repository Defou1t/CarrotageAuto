r"""
eval_swaps.py — метрика свопов: трасса _auto vs экспертная трасса orig, ТОЛЬКО на
строках, где у экспертной кривой ЕСТЬ данные (убирает искажение от разреженного GT).
Кривые группируются в треки по медиане x; своп = наша точка ближе к ДРУГОЙ экспертной
кривой того же трека, чем к своей.

python eval_swaps.py --orig <o.nlgx> --auto <a.nlgx>
"""
import sys
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL


def traces(m):
    out = {}
    for c in m["curves"]:
        nm = c["name"].split()[0]
        if nm.startswith("DA"):
            continue
        ty = c["top_y"]
        d = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL}
        if len(d) >= 20:
            out[nm] = d
    return out


def main():
    a = sys.argv[1:]
    orig = a[a.index("--orig")+1]; auto = a[a.index("--auto")+1]
    margin = 3.0
    sys.stdout.reconfigure(encoding="utf-8")
    EO = traces(extract(orig)); AU = traces(extract(auto))
    names = [n for n in EO if n in AU]
    med = {n: np.median(list(EO[n].values())) for n in names}
    # треки: кластеризация по медиане x (зазор > 250px = новый трек)
    order = sorted(names, key=lambda n: med[n])
    track = {}; t = 0
    for k, n in enumerate(order):
        if k and med[n] - med[order[k-1]] > 250:
            t += 1
        track[n] = t
    print(f"{'curve':<7}{'track':>6}{'rows':>7}{'own_px':>8}{'swap%':>7}  partner")
    tot_rows = tot_swap = 0
    for nm in order:
        rows = sorted(EO[nm])
        own = []; swap = 0; partner_cnt = {}
        peers = [p for p in names if p != nm and track[p] == track[nm]]
        for y in rows:
            ax = AU[nm].get(y)
            if ax is None:
                continue
            d_own = abs(EO[nm][y] - ax)
            own.append(d_own)
            best_peer = None; best_d = d_own - margin
            for p in peers:
                if y in EO[p] and abs(EO[p][y] - ax) < best_d:
                    best_d = abs(EO[p][y] - ax); best_peer = p
            if best_peer:
                swap += 1; partner_cnt[best_peer] = partner_cnt.get(best_peer, 0) + 1
        nr = len(own)
        if nr == 0:
            print(f"{nm:<7}{track[nm]:>6}{0:>7}     -      -"); continue
        sf = swap / nr
        pp = max(partner_cnt, key=partner_cnt.get) if partner_cnt else "-"
        print(f"{nm:<7}{track[nm]:>6}{nr:>7}{np.median(own):>8.1f}{sf*100:>6.0f}%  {pp}")
        tot_rows += nr; tot_swap += swap
    print(f"\nИТОГО свопнутых строк: {tot_swap}/{tot_rows} ({100*tot_swap/max(1,tot_rows):.1f}%)")


if __name__ == "__main__":
    main()
