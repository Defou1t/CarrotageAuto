r"""Этап B, итерационный драйвер: сырые трассы (peak, до refine) + prob-карта →
mk_refine (правки 04.07: reseat + лепестковый допуск) → гейт vs эталонные клики.
Позволяет крутить дотяжку БЕЗ повторного GPU-инференса.

  python mk_knots_iter.py <work_dir> <scan.jpg> <etalon.nlgx> [--no-refine]

work_dir: где лежат <stem>_traces.npz (сырые) и <stem>_prob.npy от infer_mk --save-prob.
Гейт печатается для сырых и для refined трасс (сравнение до/после).
"""
import sys, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_nlgx import extract, NULL
from mk_refine import refine_traces


def trace_dict(curve):
    ty = curve["top_y"]
    return {ty + i: x for i, x in enumerate(curve["xs"]) if x != NULL}


def gate(traces, etalon_model, label):
    """traces: {'MGZ1': {y:x}, 'MPZ1': {y:x}} vs клики эталона. Печать CDF."""
    print(f"--- гейт: {label} ---")
    res = {}
    for ce in etalon_model["curves"]:
        mn = ce["name"].split()[0]
        if mn.rstrip("0123456789") == "DA":
            continue
        gt = trace_dict(ce)
        if len(gt) < 30:
            continue
        our = traces.get(mn)
        if our is None:
            continue
        dx, miss = [], 0
        for y, xg in gt.items():
            xo = our.get(y)
            if xo is None:
                miss += 1
                continue
            dx.append(abs(xo - xg))
        dx = np.array(dx)
        n = len(gt)
        r = {"n": n, "miss": miss,
             "med": float(np.median(dx)) if len(dx) else None,
             "le3": float((dx <= 3).mean()) if len(dx) else 0,
             "le5": float((dx <= 5).mean()) if len(dx) else 0,
             "gt15": float((dx > 15).mean()) if len(dx) else 1}
        res[mn] = r
        print(f"  [{mn}] med {r['med']:.1f}px | ≤3px {r['le3']*100:.0f}% | ≤5px {r['le5']*100:.0f}% "
              f"| >15px {r['gt15']*100:.1f}% | дыр {miss/n*100:.1f}%")
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    work, scan, et_path = a[0], a[1], a[2]
    stem = os.path.splitext(os.path.basename(scan))[0]
    tr_path = os.path.join(work, stem + "_traces.npz")
    pr_path = os.path.join(work, "mk_prob2.npy")     # infer_mk --save-prob (float16, 2×H×W)
    d = np.load(tr_path)
    mgz = {int(y): float(x) for y, x in zip(d["mgz_y"], d["mgz_x"])}
    mpz = {int(y): float(x) for y, x in zip(d["mpz_y"], d["mpz_x"])}
    prob = np.load(pr_path).astype(np.float32) if os.path.exists(pr_path) else None
    print(f"трассы: MGZ {len(mgz)} / MPZ {len(mpz)} строк; prob: "
          f"{prob.shape if prob is not None else 'нет'}")
    me = extract(et_path)
    gate({"MGZ1": mgz, "MPZ1": mpz}, me, "сырые (peak)")

    if "--no-refine" not in a:
        rgb = np.asarray(Image.open(scan).convert("RGB"))
        configs = [
            ("H только recenter",            dict(do_fill=0, do_bridge=0, do_ext=0, do_smooth=0, do_reseat=0, do_recenter=1)),
            ("I recenter+fill+bridge",       dict(do_fill=1, do_bridge=1, do_ext=0, do_smooth=0, do_reseat=1, do_recenter=1)),
            ("J recenter+всё-D",             dict(do_fill=1, do_bridge=1, do_ext=1, do_smooth=0, do_reseat=1, petal=0, do_recenter=1)),
            ("D эталонный лучший (без recenter)", dict(do_fill=1, do_bridge=1, do_ext=1, do_smooth=0, do_reseat=1, petal=0)),
            ("K J+smooth (сдача)",           dict(do_fill=1, do_bridge=1, do_ext=1, do_smooth=1, do_reseat=1, petal=0, do_recenter=1)),
        ]
        best = None
        for label, kw in configs:
            m2, p2, st = refine_traces(rgb, mgz, mpz, prob=prob, **kw)
            r = gate({"MGZ1": m2, "MPZ1": p2}, me, label)
            # эффективный скор: доля кликов ≤3px с дырами как fail
            eff = np.mean([v["le3"] * (1 - v["miss"] / v["n"]) for v in r.values()])
            print(f"    eff(≤3px с дырами-fail) = {eff*100:.1f}%")
            if best is None or eff > best[0]:
                best = (eff, label, m2, p2)
        eff, label, m2, p2 = best
        print(f"\nЛУЧШИЙ: {label} (eff {eff*100:.1f}%)")
        out = os.path.join(work, stem + "_traces_refined.npz")
        np.savez(out,
                 mgz_y=np.array(sorted(m2)), mgz_x=np.array([m2[y] for y in sorted(m2)]),
                 mpz_y=np.array(sorted(p2)), mpz_x=np.array([p2[y] for y in sorted(p2)]))
        print(f"refined (лучший) → {out}")


if __name__ == "__main__":
    main()
