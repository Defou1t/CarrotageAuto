r"""_switch_rail.py — §6.376 РАЗВЕДКА: стоит ли смена уровня 0 → k у ФИЗИЧЕСКОГО края хода пера («рельса»), а не у объявленного x_right (15.11).

§6.364 (г) (`_switch_edge.py`): 83% смен 0 → k у эталона дальше 160 px от x_right шкалы уровня 0 — канал обруба (`level_ink.cut_fn`, окно у
x_right) видит меньшинство смен. Здесь рельс кривой = p99.5 x её трассы в пикселях по ВСЕМ строкам (ход пера не зависит от уровня; в
пикселях наша трасса в основном совпадает с эталоном — проверка «на нашей» отдельно). Для каждой смены 0 → k (как §6.364 г): x эталона на
последних строках уровня 0; доля смен в ±W px от рельса и от x_right; расстояние рельс − x_right.
Вариант `--ours`: рельс по НАШЕЙ трассе (выдача повтора `rp_lvl/<mode>`, кривая того же имени), смены — эталона.

  _switch_rail.py [--part поле|вне|все] [--w 60] [--ours NSE26]
"""
import sys, argparse, hashlib
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


def sheet_dir(n):
    return f"{n.stem[:40]}_{hashlib.md5(n.stem.encode('utf-8')).hexdigest()[:8]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="все")
    ap.add_argument("--w", type=float, default=60)
    ap.add_argument("--ours", default="")
    a = ap.parse_args()
    lst = lambda f: {l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()}
    keep = None
    if a.part == "поле":
        keep = lst("wellmap_sheets.txt")
    elif a.part == "вне":
        keep = lst("outside_sheets.txt")
    nearR, nearX, both, dRX, nsw, ncur, nsh, nmiss = 0, 0, 0, [], 0, 0, 0, 0
    per_curve = []
    for q in sorted(ARCH.glob("*/wlg/*.nlgx")):
        if keep is not None and q.name not in keep:
            continue
        ours = None
        if a.ours:
            od = TS / "rp_lvl" / a.ours / sheet_dir(q)
            fs = list(od.glob("*_auto.nlgx")) if od.exists() else []
            if not fs:
                continue
            try:
                ours = {c["name"]: c for c in extract(str(fs[0]))["curves"]}
            except Exception:
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
            s0 = ch[0]; xe = s0["x_right"]; sgn = 1 if s0["x_right"] >= s0["x_left"] else -1
            if ours is not None:
                oc = ours.get(c["name"])
                if oc is None:
                    nmiss += 1; continue
                otr = dense(oc)
                if len(otr) < 50:
                    nmiss += 1; continue
                xs = np.array(list(otr.values()))
            else:
                xs = np.array(list(tr.values()))
            rail = np.percentile(sgn * xs, 99.5) * sgn
            ncur += 1; k = 0
            for (y0, y1, l0), (z0, z1, l1) in zip(segs, segs[1:]):
                if l0 != 0 or l1 < 1:
                    continue
                ys = [y for y in range(int(y1) - 4, int(y1) + 1) if y in tr]
                if not ys:
                    continue
                x0 = tr[ys[-1]]
                r_ = abs(x0 - rail) <= a.w; x_ = abs(x0 - xe) <= a.w
                nearR += r_; nearX += x_; both += r_ and x_; nsw += 1; k += 1
            if k:
                dRX.append(sgn * (xe - rail)); per_curve.append(k)
    dRX = np.array(dRX)
    print(f"★ {a.part}{' (рельс по нашей ' + a.ours + ')' if a.ours else ' (рельс по эталону)'}: листов {nsh}, кривых с цепочкой {ncur}"
          f"{f' (нет нашей кривой того же имени {nmiss})' if a.ours else ''}, смен 0 → k {nsw}")
    print(f"   в ±{a.w:.0f} px от рельса: {nearR / max(nsw, 1):.1%};  от x_right: {nearX / max(nsw, 1):.1%};  от обоих: {both / max(nsw, 1):.1%};"
          f"  отношение {nearR / max(nearX, 1):.2f}×")
    for qq in (10, 25, 50, 75, 90):
        print(f"   x_right − рельс (внутрь шкалы), p{qq}: {np.percentile(dRX, qq):7.1f} px")


if __name__ == "__main__":
    main()
