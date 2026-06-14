"""
dataset_build.py — собрать МАНИФЕСТ датасета из троек Archive (nlgx+img+las).
Одна строка = одна реальная кривая, с QC-полями для фильтрации перед обучением.
nlgx остаётся источником меток (трасса/калибровка берутся на лету через dataset.py).

python dataset_build.py [out_csv] [--well NAME] [--limit N]
"""
import sys, re, csv
from pathlib import Path
import numpy as np
from PIL import Image
from extract_nlgx import extract, NULL
import dataset as ds

Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")


def norm(s):
    return re.sub(r"[._]+", "_", s.lower())


def find_image(nlgx_path):
    img_dir = nlgx_path.parent.parent / "img"
    if not img_dir.is_dir():
        return None
    want = norm(nlgx_path.stem)
    for ext in ("*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png"):
        for p in img_dir.glob(ext):
            if "_auto" in p.stem:
                continue
            if norm(p.stem) == want:
                return p
    return None


def find_las(nlgx_path):
    las_dir = nlgx_path.parent.parent / "las"
    if not las_dir.is_dir():
        return None
    want = norm(nlgx_path.stem)
    for p in las_dir.glob("*.las"):
        if norm(p.stem) == want:
            return p
    return None


def align_frac(gray, curve, thr=120, rx=3, ry=1):
    H, W = gray.shape
    ink = gray < thr
    xs = np.asarray(curve["xs"], dtype=np.int64)
    idx = np.where(xs != NULL)[0]
    if len(idx) == 0:
        return 0.0, 0
    x = xs[idx]; y = curve["top_y"] + idx
    ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return 0.0, 0
    hit = np.zeros(len(x), dtype=bool)
    for dy in range(-ry, ry+1):
        for dx in range(-rx, rx+1):
            hit |= ink[np.clip(y+dy, 0, H-1), np.clip(x+dx, 0, W-1)]
    return float(hit.sum())/len(x), len(x)


FIELDS = ["well", "nlgx", "image", "las", "curve", "mnemonic", "units", "family",
          "n_pts", "align_frac", "las_col", "log_corr", "top_depth", "bottom_depth",
          "img_w", "img_h", "n_levels", "usable"]


def stub_check(model, nlgx_name):
    """Недоделанный планшет: интервал имени файла >> покрытие Depth Axis."""
    m = re.search(r"_(\d{3,5})[-_](\d{3,5})_", nlgx_name)
    if not m:
        return False
    fi = abs(float(m.group(2)) - float(m.group(1)))
    da = model.get("depth_axis", {})
    cov = abs(da.get("bottom_depth", 0) - da.get("top_depth", 0))
    return fi > 20 and cov < 0.3 * fi


def _process(m, nlgx, well, img_path, las_path, rows, st):
    """Обработать одну тройку: QC по кривым -> строки манифеста (в rows)."""
    curves = ds.real_curves(m)
    if not curves:
        return
    if stub_check(m, nlgx.stem):
        st["stub"] += 1
        print(f"  STUB {well.name}/{nlgx.name} (axis cov tiny)"); return
    gray = None; iw = ih = 0
    if img_path:
        gray = np.asarray(Image.open(img_path).convert("L")); ih, iw = gray.shape
    else:
        st["noimg"] += 1
    las_cols, las_arr = (ds.load_las(las_path) if las_path else (None, np.zeros((0, 0))))
    matches = ds.match_las(m, las_cols, las_arr)
    da = m.get("depth_axis", {})
    st["files"] += 1
    for c in curves:
        af, npts = (align_frac(gray, c) if gray is not None
                    else (None, len([x for x in c["xs"] if x != NULL])))
        mm = matches.get(c["name"], {})
        usable = (af is None or af >= 0.9) and npts >= 50
        rows.append({
            "well": well.name, "nlgx": nlgx.name,
            "image": img_path.name if img_path else "",
            "las": las_path.name if las_path else "",
            "curve": c["name"].split()[0], "mnemonic": ds.mnemonic(c["name"]),
            "units": "", "family": c["name"].strip().split()[-1], "n_pts": npts,
            "align_frac": round(af, 4) if af is not None else "",
            "las_col": mm.get("las_col") or "",
            "log_corr": round(mm["log_corr"], 4) if mm.get("log_corr") is not None else "",
            "top_depth": da.get("top_depth", ""), "bottom_depth": da.get("bottom_depth", ""),
            "img_w": iw, "img_h": ih, "n_levels": len({l for _, _, l in c["segments"]}),
            "usable": int(usable),
        })
    print(f"  {well.name}/{nlgx.name[:40]:<40} curves={len(curves)} "
          f"img={'Y' if img_path else 'N'} las={'Y' if las_path else 'N'}")


def main():
    args = sys.argv[1:]
    out_csv = "manifest.csv"
    well_filter = None; limit = 9999
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--well":
            well_filter = args[i+1]; i += 2
        elif args[i] == "--limit":
            limit = int(args[i+1]); i += 2
        else:
            rest.append(args[i]); i += 1
    if rest:
        out_csv = rest[0]
    out_csv = str(Path(__file__).parent / out_csv) if not Path(out_csv).is_absolute() else out_csv

    wells = sorted([d for d in ARCHIVE.iterdir() if d.is_dir()])
    if well_filter:
        wells = [w for w in wells if w.name == well_filter]
    wells = wells[:limit]

    rows = []
    st = {"files": 0, "stub": 0, "noimg": 0, "err": 0}
    sys.stdout.reconfigure(line_buffering=True)
    for well in wells:
        wlg = well / "wlg"
        if not wlg.is_dir():
            continue
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem:
                continue
            try:
                m = extract(str(nlgx))
            except Exception as e:
                st["err"] += 1; print(f"  ERR extract {nlgx.name}: {e}"); continue
            try:
                _process(m, nlgx, well, find_image(nlgx), find_las(nlgx), rows, st)
            except Exception as e:
                st["err"] += 1
                print(f"  ERR process {nlgx.name}: {type(e).__name__}: {e}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow(r)

    usable = [r for r in rows if r["usable"]]
    af_vals = [r["align_frac"] for r in rows if isinstance(r["align_frac"], float)]
    lc_vals = [r["log_corr"] for r in rows if isinstance(r["log_corr"], float)]
    print(f"\n=== MANIFEST {out_csv} ===")
    print(f"files={st['files']} stubs={st['stub']} no-img={st['noimg']} errors={st['err']}")
    print(f"curves total={len(rows)} usable(align>=0.9,pts>=50)={len(usable)}")
    if af_vals:
        print(f"align_frac: mean={np.mean(af_vals):.3f} median={np.median(af_vals):.3f} "
              f">=0.95: {sum(a>=0.95 for a in af_vals)}/{len(af_vals)}")
    if lc_vals:
        print(f"log_corr vs LAS (matched): mean={np.mean(lc_vals):.3f} median={np.median(lc_vals):.3f} "
              f">=0.9: {sum(a>=0.9 for a in lc_vals)}/{len(lc_vals)}")
    from collections import Counter
    mn = Counter(r["mnemonic"] for r in usable)
    print("usable curves by mnemonic:", dict(mn.most_common(20)))


if __name__ == "__main__":
    main()
