r"""_vertical_stat.py — §6.364 (б) РАЗВЕДКА: «вертикальность» наших кривых и их верность (13.11).

Вертикальность трассы — средняя по окнам 400 строк доля строк, лежащих ≤ 4 px от медианы x окна (у кривой ~0.02–0.05, у линии сетки /
границы трека / нулевой линии 0.6–0.9; выборка 13.11, лист 09). Причины по кривым эталона NSE22 — `sample_2026-11-10/pop_curves.tsv`
(часть, лист, кривая, трек, причина…). Выход: распределение верности по корзинам вертикальности (наша кривая под тем же именем).

  _vertical_stat.py [--mode NSE22]
"""
import sys, argparse, csv
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from auto import meta as M

TS = Path(r"F:/nds/output/taskS")


def vert(xs, win=400, tol=4.0):
    x = np.array([v for v in xs if v != NULL], float)
    if len(x) < 100:
        return np.nan
    f = []
    for i in range(0, len(x), win):
        w = x[i:i + win]
        if len(w) >= 50:
            f.append(np.mean(np.abs(w - np.median(w)) <= tol))
    return float(np.mean(f)) if f else np.nan


def main():
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="NSE22")
    ap.add_argument("--pop", default=r"F:/nds/output/sample_2026-11-10/pop_curves.tsv")
    a = ap.parse_args()
    rows = list(csv.reader(open(a.pop, encoding="utf-8"), delimiter="\t"))[1:]
    cat = {(r[1], r[2]): (r[0], r[4]) for r in rows}
    sheets = sorted({r[1] for r in rows})
    sd = lambda nm: f"{Path(nm).stem[:40]}_{hashlib.md5(Path(nm).stem.encode('utf-8')).hexdigest()[:8]}"
    B = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
    agg = defaultdict(lambda: defaultdict(int)); ex = []
    for i, nm in enumerate(sheets):
        g = next(iter(sorted((TS / "rp_lvl" / a.mode / sd(nm)).glob("*_auto.nlgx"))), None)
        if g is None:
            continue
        for c in extract(str(g))["curves"]:
            if M.mnem_root(c["name"]) == "DA" or (nm, c["name"]) not in cat:
                continue
            v = vert(c["xs"])
            if not np.isfinite(v):
                continue
            part, why = cat[(nm, c["name"])]
            b = np.digitize(v, B) - 1
            agg[b][why] += 1; agg[b]["всего"] += 1
            if v >= 0.5:
                ex.append((round(v, 2), why, nm[:50], c["name"]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} листов")
    print(f"★ вертикальность наших кривых ({a.mode}) и причина по эталону (поле+вне+сорт A, кривые эталона с нашей одноимённой)")
    for b in sorted(agg):
        d = agg[b]; n = d["всего"]
        print(f"  [{B[b]:.1f}, {B[b + 1]:.1f}): кривых {n:5d}, верно {d['верно'] / n:6.1%}, не та линия {d['не та линия'] / n:6.1%}, "
              f"на другой {d['на другой кривой эталона'] / n:6.1%}, уровень {d['уровень масштаба'] / n:6.1%}")
    print("★ примеры ≥ 0.5:")
    for e in sorted(ex, reverse=True)[:25]:
        print("   ", e)


if __name__ == "__main__":
    main()
