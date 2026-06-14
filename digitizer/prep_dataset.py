"""
prep_dataset.py — нарезка обучающих тайлов для U-Net (бинарная сегментация штриха).
Источник меток — nlgx (трассы кривых). Маска = объединение трасс ПРИГОДНЫХ кривых
(align-frac>=0.9, pts>=50), растеризованных полилинией (dataset.curve_mask).
Тайлы режутся ВДОЛЬ кривой (центры по фону маски) + доля негативов (сетка/текст/фон).
Сплит train/val — по СКВАЖИНЕ (без утечки между тайлами одной скважины).

Запуск (любым python с numpy/PIL):
  python prep_dataset.py [--limit N] [--tile 256] [--stride 160] [--negfrac 0.2]
                         [--valfrac 0.15] [--out DIR]
Каждый исходник -> один .npz (imgs uint8 N×T×T×3, masks uint8 N×T×T) в train/ или val/.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import dataset as ds
from extract_nlgx import extract, NULL
from dataset_build import find_image, align_frac

Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")


def is_val(well, valfrac):
    h = int(hashlib.md5(well.encode()).hexdigest(), 16) % 1000
    return h < valfrac * 1000


def cut_tiles(img, mask, T, stride, negfrac, rng, maxtiles=80):
    """Тайлы вдоль кривой (центры по фону маски) + негативы. img RGB HxWx3, mask HxW bool."""
    H, W = mask.shape
    half = T // 2
    fg = np.argwhere(mask)
    if len(fg) == 0:
        return [], []
    y0, y1 = fg[:, 0].min(), fg[:, 0].max()
    centers = []
    for cy in range(y0 + half, y1 - half + 1, stride):
        band = fg[(fg[:, 0] >= cy - stride // 2) & (fg[:, 0] < cy + stride // 2)]
        if len(band) == 0:
            continue
        cx = int(np.median(band[:, 1]))
        centers.append((cy, cx + int(rng.integers(-30, 31))))  # x-джиттер
    if len(centers) > maxtiles:  # кап на файл (равномерно прорежаем по глубине)
        idx = np.linspace(0, len(centers) - 1, maxtiles).round().astype(int)
        centers = [centers[k] for k in idx]
    n_neg = int(len(centers) * negfrac)
    for _ in range(n_neg):
        centers.append((int(rng.integers(y0 + half, max(y0 + half + 1, y1 - half))),
                        int(rng.integers(half, max(half + 1, W - half)))))
    imgs, masks = [], []
    for cy, cx in centers:
        cy = int(np.clip(cy, half, H - half)); cx = int(np.clip(cx, half, W - half))
        it = img[cy - half:cy + half, cx - half:cx + half]
        mt = mask[cy - half:cy + half, cx - half:cx + half]
        if it.shape[:2] == (T, T):
            imgs.append(it); masks.append(mt.astype(np.uint8))
    return imgs, masks


def main():
    args = sys.argv[1:]
    limit = 9999; T = 256; stride = 160; negfrac = 0.2; valfrac = 0.15; maxtiles = 80
    out = Path(r"F:\nds\output\unet_data")
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--limit": limit = int(args[i+1]); i += 2
        elif a == "--tile": T = int(args[i+1]); i += 2
        elif a == "--stride": stride = int(args[i+1]); i += 2
        elif a == "--negfrac": negfrac = float(args[i+1]); i += 2
        elif a == "--valfrac": valfrac = float(args[i+1]); i += 2
        elif a == "--maxtiles": maxtiles = int(args[i+1]); i += 2
        elif a == "--out": out = Path(args[i+1]); i += 2
        else: i += 1
    sys.stdout.reconfigure(line_buffering=True)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    wells = sorted([d for d in ARCHIVE.iterdir() if d.is_dir()])[:limit]
    index = []
    n_tiles = {"train": 0, "val": 0}
    for wl in wells:
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
            curves = ds.real_curves(m)
            if not curves:
                continue
            img_path = find_image(nlgx)
            if not img_path:
                continue
            im = Image.open(img_path).convert("RGB")
            rgb = np.asarray(im)
            gray = np.asarray(im.convert("L"))
            H, W = gray.shape
            usable = [c for c in curves if align_frac(gray, c)[0] >= 0.9]
            if not usable:
                continue
            mask = np.zeros((H, W), bool)
            for c in usable:
                mask |= ds.curve_mask(c, H, W, stroke=3)
            imgs, masks = cut_tiles(rgb, mask, T, stride, negfrac, rng, maxtiles)
            if not imgs:
                continue
            arr_i = np.stack(imgs).astype(np.uint8)
            arr_m = np.stack(masks).astype(np.uint8)
            fn = out / split / f"{wl.name}__{nlgx.stem[:50]}.npz"
            np.savez_compressed(fn, imgs=arr_i, masks=arr_m)
            index.append({"npz": str(fn), "well": wl.name, "split": split,
                          "n_tiles": len(imgs), "n_usable_curves": len(usable),
                          "image": str(img_path)})
            n_tiles[split] += len(imgs)
            print(f"  [{split}] {wl.name}/{nlgx.stem[:36]:<36} usable={len(usable)} tiles={len(imgs)}")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    wells_tr = len({r['well'] for r in index if r['split']=='train'})
    wells_va = len({r['well'] for r in index if r['split']=='val'})
    print(f"\n=== PREP DONE -> {out} ===")
    print(f"files={len(index)}  tiles train={n_tiles['train']} val={n_tiles['val']}")
    print(f"wells train={wells_tr} val={wells_va}  (T={T} stride={stride} negfrac={negfrac})")


if __name__ == "__main__":
    main()
