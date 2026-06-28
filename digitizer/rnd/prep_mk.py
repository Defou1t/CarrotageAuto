r"""Трек 2 / шаг 1 — нарезка 2-КАНАЛЬНЫХ тайлов для разделителя MK (MGZ vs MPZ).
Канал0 = маска MGZ (толстая), канал1 = MPZ (тонкая) из РАЗДЕЛЬНЫХ трасс nlgx-GT. Модель учит
«толстая→к0, тонкая→к1» + непрерывность → на инференсе кладёт толстую в к0, разрешая пересечения.
Тайлы вдоль объединения кривых + негативы. Сплит train/val по скважине. Чистый numpy/PIL (без torch).

  python prep_mk.py [--tile 256 --stride 140 --negfrac 0.15 --valfrac 0.18 --out F:\nds\output\mk_data]
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # digitizer на путь
import dataset as ds
from extract_nlgx import extract
from dataset_build import find_image, align_frac
Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")


def is_val(well, valfrac):
    return int(hashlib.md5(well.encode()).hexdigest(), 16) % 1000 < valfrac * 1000


def curves_mn(m, mn):
    return [c for c in ds.real_curves(m) if ds.mnemonic(c["name"]).upper() == mn]


def cut_tiles(img, m2, T, stride, negfrac, rng, maxtiles):
    """Тайлы вдоль ОБЪЕДИНЕНИЯ каналов + негативы. img HxWx3, m2 HxWx2 (MGZ,MPZ)."""
    H, W = m2.shape[:2]; half = T // 2
    union = m2.any(2)
    fg = np.argwhere(union)
    if len(fg) == 0:
        return [], []
    y0, y1 = fg[:, 0].min(), fg[:, 0].max()
    centers = []
    for cy in range(y0 + half, y1 - half + 1, stride):
        band = fg[(fg[:, 0] >= cy - stride // 2) & (fg[:, 0] < cy + stride // 2)]
        if len(band):
            centers.append((cy, int(np.median(band[:, 1])) + int(rng.integers(-30, 31))))
    if len(centers) > maxtiles:
        idx = np.linspace(0, len(centers) - 1, maxtiles).round().astype(int)
        centers = [centers[k] for k in idx]
    for _ in range(int(len(centers) * negfrac)):
        centers.append((int(rng.integers(y0 + half, max(y0 + half + 1, y1 - half))),
                        int(rng.integers(half, max(half + 1, W - half)))))
    imgs, masks = [], []
    for cy, cx in centers:
        cy = int(np.clip(cy, half, H - half)); cx = int(np.clip(cx, half, W - half))
        it = img[cy - half:cy + half, cx - half:cx + half]
        mt = m2[cy - half:cy + half, cx - half:cx + half]
        if it.shape[:2] == (T, T):
            imgs.append(it); masks.append(mt.astype(np.uint8))
    return imgs, masks


def main():
    args = sys.argv[1:]
    T = 256; stride = 110; negfrac = 0.15; valfrac = 0.18; maxtiles = 180
    out = Path(r"F:\nds\output\mk_data")
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tile": T = int(args[i+1]); i += 2
        elif a == "--stride": stride = int(args[i+1]); i += 2
        elif a == "--negfrac": negfrac = float(args[i+1]); i += 2
        elif a == "--valfrac": valfrac = float(args[i+1]); i += 2
        elif a == "--out": out = Path(args[i+1]); i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    index = []; n_tiles = {"train": 0, "val": 0}
    for wl in sorted([d for d in ARCHIVE.iterdir() if d.is_dir()]):
        wlg = wl / "wlg"
        if not wlg.is_dir():
            continue
        split = "val" if is_val(wl.name, valfrac) else "train"
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem:
                continue
            try:
                m = extract(str(nlgx))
            except Exception:
                continue
            mgz, mpz = curves_mn(m, "MGZ"), curves_mn(m, "MPZ")
            if not mgz or not mpz:
                continue
            img_path = find_image(nlgx)
            if not img_path:
                continue
            im = Image.open(img_path).convert("RGB"); rgb = np.asarray(im)
            gray = np.asarray(im.convert("L")); H, W = gray.shape
            mgz = [c for c in mgz if align_frac(gray, c)[0] >= 0.85]
            mpz = [c for c in mpz if align_frac(gray, c)[0] >= 0.85]
            if not mgz or not mpz:
                continue
            ch0 = np.zeros((H, W), bool); ch1 = np.zeros((H, W), bool)
            for c in mgz: ch0 |= ds.curve_mask(c, H, W, stroke=6)
            for c in mpz: ch1 |= ds.curve_mask(c, H, W, stroke=6)
            m2 = np.stack([ch0, ch1], -1)
            imgs, masks = cut_tiles(rgb, m2, T, stride, negfrac, rng, maxtiles)
            if not imgs:
                continue
            fn = out / split / f"{wl.name}__{nlgx.stem[:46]}.npz"
            np.savez_compressed(fn, imgs=np.stack(imgs).astype(np.uint8),
                                masks=np.stack(masks).astype(np.uint8))   # N×T×T×2
            index.append({"npz": str(fn), "well": wl.name, "split": split, "n_tiles": len(imgs)})
            n_tiles[split] += len(imgs)
            print(f"  [{split}] {wl.name}/{nlgx.stem[:30]:<30} tiles={len(imgs)}")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    wtr = len({r['well'] for r in index if r['split'] == 'train'})
    wva = len({r['well'] for r in index if r['split'] == 'val'})
    print(f"\n=== MK PREP -> {out} ===\nfiles={len(index)} tiles train={n_tiles['train']} val={n_tiles['val']} "
          f"wells train={wtr} val={wva}")


if __name__ == "__main__":
    main()
