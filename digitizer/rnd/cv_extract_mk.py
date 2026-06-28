r"""CV-базлайн MGZ/MPZ: обобщённое ИЗВЛЕЧЬ-ВЫЧЕСТЬ (из _mk_extract) на ЛЮБОЙ scan+nlgx → трассы npz.
Рамка (TY/BY) из nlgx depth_axis; x-полоса бандла автономно по плотности тёмного. Для CV vs NN на GT.
  python cv_extract_mk.py <scan> <nlgx> [--out DIR]
"""
import sys
from pathlib import Path
import numpy as np, cv2
_ROOT = Path(__file__).resolve().parent.parent.parent       # F:\nds\Auto (для auto.*)
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "digitizer"))
import auto.imaging as im
from auto.config import DEFAULT
from extract_nlgx import extract
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
p = DEFAULT.cv


def bundle_band(fg, TY, BY):
    """x-полоса бандла = самый плотный непрерывный диапазон тёмного (метки слева — разрежены)."""
    col = fg[TY:BY].sum(0).astype(float)
    if col.max() == 0:
        return 0, fg.shape[1]
    on = col > 0.05 * col.max()
    xs = np.nonzero(on)[0]
    # самый длинный непрерывный кусок
    cuts = np.nonzero(np.diff(xs) > 25)[0] + 1
    segs = np.split(xs, cuts)
    best = max(segs, key=lambda s: col[s].sum())
    return max(0, int(best[0]) - 15), min(fg.shape[1], int(best[-1]) + 15)


def clean_fg(rgb, TY, BY, XL, XR):
    fg = (im.dark_mask(rgb, p) & ~im.structure_mask(rgb, p)).astype(np.uint8)
    sub = np.zeros_like(fg); sub[TY:BY, XL:XR] = fg[TY:BY, XL:XR]
    bridged = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bridged, 8)
    keep = np.zeros_like(sub)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_HEIGHT] >= 150:
            keep[lab == i] = 1
    return (keep & sub).astype(np.uint8)


def runs_at(fg, DT, y, XL, XR):
    out = []
    for a, b, c in im.row_runs(fg[y, XL:XR], gap=2):
        a += XL; b += XL; dt = float(DT[y, a:b + 1].max())
        out.append({"a": a, "b": b, "c": (a + b) / 2.0, "dt": dt, "spike": (b - a + 1) > 4 * dt + 6})
    return out


def vtx(r, base):
    return (r["a"] if abs(r["a"] - base) > abs(r["b"] - base) else r["b"]) if r["spike"] else r["c"]


def trace_one(fg, DT, TY, BY, XL, XR, seed_y, seed_x, prefer_thick, slmax=26., maxjump=70., band=85.):
    rr = {y: runs_at(fg, DT, y, XL, XR) for y in range(TY, BY)}
    st = {"x": seed_x, "sl": 0., "anc": seed_x}; tr, trun = {}, {}
    def march(ys):
        st["x"] = seed_x; st["sl"] = 0.; st["anc"] = seed_x
        for y in ys:
            runs = rr.get(y) or []
            if not runs:
                st["x"] += float(np.clip(st["sl"], -slmax, slmax)); st["sl"] *= 0.7; continue
            pred = st["x"] + float(np.clip(st["sl"], -slmax, slmax))
            cands = [r for r in runs if r["a"] - 3 <= pred <= r["b"] + 3] or \
                    [r for r in runs if abs(r["c"] - pred) <= maxjump or abs(r["c"] - st["anc"]) <= band]
            if not cands:
                st["x"] += float(np.clip(st["sl"], -slmax, slmax)); st["sl"] *= 0.7; continue
            r = min(cands, key=lambda r: abs(r["c"] - pred) - (5. * r["dt"] if prefer_thick else 0.))
            nx = vtx(r, st["anc"])
            st["sl"] = float(np.clip(0.7 * st["sl"] + 0.3 * (nx - st["x"]), -slmax, slmax))
            st["x"] = nx; tr[y] = nx; trun[y] = (r["a"], r["b"]); st["anc"] = 0.995 * st["anc"] + 0.005 * r["c"]
    march(range(seed_y, BY)); march(range(seed_y - 1, TY - 1, -1))
    return tr, trun


def main():
    a = sys.argv[1:]
    scan, nlgx = a[0], a[1]
    out = Path(a[a.index("--out") + 1] if "--out" in a else r"F:\nds\output\mk_data")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    rgb = np.asarray(Image.open(scan).convert("RGB")); H, W = rgb.shape[:2]
    da = extract(nlgx)["depth_axis"]; TY, BY = int(da["top_y"]), int(min(H, da["bottom_y"]))
    fg0 = (im.dark_mask(rgb, p) & ~im.structure_mask(rgb, p)).astype(np.uint8)
    XL, XR = bundle_band(fg0, TY, BY)
    fg = clean_fg(rgb, TY, BY, XL, XR); DT = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    ys, xs = np.where(DT[TY:BY] == DT[TY:BY].max())
    sy = TY + int(ys[len(ys) // 2]); sx = int(xs[len(xs) // 2])
    mgz, mgz_run = trace_one(fg, DT, TY, BY, XL, XR, sy, sx, True)
    fg2 = fg.copy()
    for y, (aa, bb) in mgz_run.items():
        fg2[y, max(XL, aa - 2):min(XR, bb + 3)] = 0
    DT2 = cv2.distanceTransform(fg2, cv2.DIST_L2, 5)
    nn, lab2, st2, cen2 = cv2.connectedComponentsWithStats(fg2, 8)
    if nn > 1:
        big = 1 + int(np.argmax([st2[i, cv2.CC_STAT_HEIGHT] for i in range(1, nn)]))
        sy2 = int(cen2[big][1]); xx = np.where(lab2[sy2] == big)[0]; sx2 = int(xx[len(xx)//2]) if len(xx) else sx
    else:
        sy2, sx2 = sy, sx
    mpz, _ = trace_one(fg2, DT2, TY, BY, XL, XR, sy2, sx2, False)
    known = sorted(mpz)
    if len(known) >= 2:
        kx = [mpz[y] for y in known]
        for y in sorted(mgz):
            if y not in mpz and known[0] <= y <= known[-1]:
                mpz[y] = float(np.interp(y, known, kx))
    stem = Path(scan).stem[:40]
    np.savez(out / f"{stem}_cvtraces.npz", mgz_y=np.array(list(mgz)), mgz_x=np.array(list(mgz.values())),
             mpz_y=np.array(list(mpz)), mpz_x=np.array(list(mpz.values())))
    print(f"{stem}: band x[{XL}..{XR}] MGZ={len(mgz)} MPZ={len(mpz)} -> {stem}_cvtraces.npz")


if __name__ == "__main__":
    main()
