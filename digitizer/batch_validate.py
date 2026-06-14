r"""
Батч-проверка расшифровки nlgx по датасету Archive:
для каждого nlgx (не _auto) найти соседний img\<norm>.jpg, извлечь трассу,
посчитать align-frac (доля точек трассы на чернилах). Доказывает универсальность.

python batch_validate.py [N]   # N = ограничить число скважин (по умолчанию все)
"""
import sys, re
from pathlib import Path
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL

ARCHIVE = Path(r"F:\nds\projects\Archive")
Image.MAX_IMAGE_PIXELS = None


def norm(s):
    return re.sub(r"[._]+", "_", s.lower())


def find_image(nlgx_path):
    stem = nlgx_path.stem
    img_dir = nlgx_path.parent.parent / "img"
    if not img_dir.is_dir():
        return None
    want = norm(stem)
    for ext in ("*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png"):
        for p in img_dir.glob(ext):
            if "_auto" in p.stem:
                continue
            if norm(p.stem) == want:
                return p
    return None


def align_frac(img_gray, curve, thr=120, rx=3, ry=1):
    """Векторизовано: доля точек трассы, у которых в окне ±rx,±ry есть чернила."""
    H, W = img_gray.shape
    ink = img_gray < thr
    xs = np.asarray(curve["xs"], dtype=np.int64)
    idx = np.where(xs != NULL)[0]
    if len(idx) == 0:
        return 0, 0
    x = xs[idx]; y = curve["top_y"] + idx
    ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return 0, 0
    hit = np.zeros(len(x), dtype=bool)
    for dy in range(-ry, ry+1):
        for dx in range(-rx, rx+1):
            hit |= ink[np.clip(y+dy, 0, H-1), np.clip(x+dx, 0, W-1)]
    return int(hit.sum()), len(x)


def main():
    sys.stdout.reconfigure(line_buffering=True)  # видеть прогресс сразу, без буфера
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    wells = sorted([d for d in ARCHIVE.iterdir() if d.is_dir()])[:limit]
    print(f"{'well':<22}{'nlgx':<46}{'curve':<8}{'align':>7}{'pts':>7}")
    print("-"*92)
    tot_hit = tot_pts = 0
    n_curves = n_files = 0
    for well in wells:
        wlg = well / "wlg"
        if not wlg.is_dir():
            continue
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem:
                continue
            img_path = find_image(nlgx)
            try:
                m = extract(str(nlgx))
            except Exception as e:
                print(f"{well.name:<22}{nlgx.name:<46} EXTRACT-ERR {e}")
                continue
            curves = [c for c in m["curves"]
                      if len([x for x in c["xs"] if x != NULL]) >= 10]
            if img_path is None:
                print(f"{well.name:<22}{nlgx.name:<46} NO-IMG  curves={len(curves)}")
                continue
            g = np.asarray(Image.open(img_path).convert("L"))
            n_files += 1
            for c in curves:
                hit, tot = align_frac(g, c)
                if tot == 0:
                    continue
                frac = hit/tot
                tot_hit += hit; tot_pts += tot; n_curves += 1
                flag = "" if frac > 0.9 else "  <-- LOW"
                cn = c["name"].split()[0]
                print(f"{well.name:<22}{nlgx.name[:44]:<46}{cn:<8}{frac:>7.3f}{tot:>7}{flag}")
    print("-"*92)
    if tot_pts:
        print(f"TOTAL: {n_files} files, {n_curves} curves, overall align-frac="
              f"{tot_hit/tot_pts:.4f}  ({tot_hit}/{tot_pts})")


if __name__ == "__main__":
    main()
