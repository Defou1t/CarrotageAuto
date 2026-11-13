r"""_switch_edge.py — §6.364 (г) РАЗВЕДКА: где стоит перо эталона в момент смены уровня 0 → k относительно края шкалы уровня 0 (13.11).

Канал обруба (`auto.level_ink.cut_fn`, §6.340) смотрит окно у x_right шкалы уровня 0 (S строк × inn внутрь + outw наружу). Выборка 13.11:
у 10, 11, 01, 28 реальный упор пера на 160–1100 px левее x_right. Здесь — по всему архиву: для каждой смены 0 → k у кривой эталона
(сегменты уровня, цепочка шкал ≥ 2) x эталона на последних строках уровня 0 и расстояние до x_right в сторону шкалы (> 0 — внутри).

  _switch_edge.py [--part поле|вне|все]
"""
import sys, json, argparse
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path[:0] = [r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"]
from pathlib import Path
import numpy as np
from extract_nlgx import extract
from _multi_replica_probe import dense
from auto import meta as M
import decode_levels as DL

TS = Path(r"F:/nds/output/taskS")
ARCH = Path(r"F:\nds\projects\Archive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="все")
    a = ap.parse_args()
    lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    keep = None
    if a.part == "поле":
        keep = lst("wellmap_sheets.txt")
    elif a.part == "вне":
        keep = lst("outside_sheets.txt")
    D, R, W_, J, fam = [], [], [], [], []
    ncur = nsh = 0
    for q in sorted(ARCH.glob("*/wlg/*.nlgx")):
        if keep is not None and q.name not in keep:
            continue
        try:
            m = extract(str(q))
        except Exception:
            continue
        nsh += 1
        for c in m["curves"]:
            if M.mnem_root(c["name"]) == "DA" or not c.get("segments"):
                continue
            ch = DL.build_family(m, c)
            if len(ch) < 2:
                continue
            segs = sorted(c["segments"])
            tr = dense(c)
            if len(tr) < 50:
                continue
            ncur += 1
            s0 = ch[0]; xe = s0["x_right"]; sgn = 1 if s0["x_right"] >= s0["x_left"] else -1
            w0 = abs(s0["x_right"] - s0["x_left"]) or 1.0
            for (y0, y1, l0), (z0, z1, l1) in zip(segs, segs[1:]):
                if l0 != 0 or l1 < 1:
                    continue
                ys = [y for y in range(int(y1) - 4, int(y1) + 1) if y in tr]
                if not ys:
                    continue
                x0 = tr[ys[-1]]
                xa = [tr[y] for y in range(int(z0), int(z0) + 5) if y in tr]
                d = sgn * (xe - x0)
                D.append(d); W_.append(w0); R.append(d / w0)
                J.append(abs(xa[0] - x0) / w0 if xa else np.nan)
                fam.append(M.mnem_root(c["name"]))
    D = np.array(D); R = np.array(R); J = np.array(J)
    print(f"★ {a.part}: листов {nsh}, кривых с цепочкой и сменами {ncur}, смен 0 → k: {len(D)}")
    for q in (10, 25, 50, 75, 90):
        print(f"  p{q}: до x_right {np.percentile(D, q):7.1f} px ({np.percentile(R, q):5.2f} ширины шкалы)")
    for t in (30, 64, 96, 160, 300):
        print(f"  дальше {t:3d} px внутрь: {np.mean(D > t):6.1%};  наружу дальше {t} px: {np.mean(D < -t):6.1%}")
    print(f"  скачок x на смене > 0.3 ширины: {np.nanmean(J > 0.3):6.1%}")
    from collections import Counter
    far = Counter(f for f, d in zip(fam, D) if d > 160); allc = Counter(fam)
    print("  дальше 160 px по семействам:", ", ".join(f"{k} {far[k]}/{allc[k]}" for k, _ in allc.most_common(12)))


if __name__ == "__main__":
    main()
