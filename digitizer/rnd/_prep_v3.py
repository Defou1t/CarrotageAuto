r"""
_prep_v3.py — нарезка тайлов для доучивания U-Net v3.
Метки: точная ширина (маска) + вес-карта акцента на обороты (_relabel.build_targets).
Источники: Archive (на месте, ЧТЕНИЕ) + копии активных скважин в unet_v3\src.
Тайлы: вдоль кривой + ОВЕРСЭМПЛИНГ оборотных строк + негативы (грид/текст).
Каждый .npz: imgs uint8 N×T×T×3, masks uint8 N×T×T, weights float16 N×T×T.

python _prep_v3.py [--limit N] [--tile 256] [--stride 160]
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\output"); sys.path.insert(0, r"F:\nds\Auto\digitizer")
from _relabel import build_targets
import dataset as ds
from extract_nlgx import extract
from dataset_build import find_image, align_frac

ARCHIVE = Path(r"F:\nds\projects\Archive")
COPIES = Path(r"F:\nds\output\unet_v3\src")
OUT = Path(r"F:\nds\output\unet_v3\data")


def is_val(well, valfrac=0.15):
    return int(hashlib.md5(well.encode()).hexdigest(), 16) % 1000 < valfrac * 1000


def cut(rgb, mask, weight, T, stride, rng, maxtiles=80, negfrac=0.2):
    H, W = mask.shape; half = T // 2
    fg = np.argwhere(mask)
    if len(fg) == 0:
        return [], [], []
    y0, y1 = fg[:, 0].min(), fg[:, 0].max()
    centers = []
    for cy in range(y0 + half, y1 - half + 1, stride):
        band = fg[(fg[:, 0] >= cy - stride // 2) & (fg[:, 0] < cy + stride // 2)]
        if len(band) == 0:
            continue
        cx = int(np.median(band[:, 1]))
        centers.append((cy, cx + int(rng.integers(-30, 31))))
    if len(centers) > maxtiles:
        idx = np.linspace(0, len(centers) - 1, maxtiles).round().astype(int)
        centers = [centers[k] for k in idx]
    # ОВЕРСЭМПЛИНГ оборотов: центры на строках с весом>1 (доп. плотность там)
    wrap_rows = np.unique(np.argwhere(weight > 1.0)[:, 0])
    wrap_rows = wrap_rows[(wrap_rows >= y0 + half) & (wrap_rows < y1 - half)]
    for cy in wrap_rows[::max(1, stride // 2)]:
        b = fg[(fg[:, 0] >= cy - 40) & (fg[:, 0] < cy + 40)]
        if len(b):
            centers.append((int(cy), int(np.median(b[:, 1])) + int(rng.integers(-25, 26))))
    for _ in range(int(len(centers) * negfrac)):
        centers.append((int(rng.integers(y0 + half, max(y0 + half + 1, y1 - half))),
                        int(rng.integers(half, max(half + 1, W - half)))))
    imgs, masks, wts = [], [], []
    for cy, cx in centers:
        cy = int(np.clip(cy, half, H - half)); cx = int(np.clip(cx, half, W - half))
        it = rgb[cy - half:cy + half, cx - half:cx + half]
        if it.shape[:2] != (T, T):
            continue
        imgs.append(it)
        masks.append(mask[cy - half:cy + half, cx - half:cx + half].astype(np.uint8))
        wts.append(weight[cy - half:cy + half, cx - half:cx + half].astype(np.float16))
    return imgs, masks, wts


def sources():
    out = []
    for wl in sorted(ARCHIVE.iterdir()):
        if (wl / "wlg").is_dir():
            out.append((wl.name, wl / "wlg"))
    for wl in sorted(COPIES.iterdir()) if COPIES.is_dir() else []:
        if (wl / "wlg").is_dir():
            out.append((wl.name, wl / "wlg"))
    return out


def main():
    a = sys.argv[1:]
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else 9999
    T = int(a[a.index("--tile") + 1]) if "--tile" in a else 256
    stride = int(a[a.index("--stride") + 1]) if "--stride" in a else 160
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    (OUT / "train").mkdir(parents=True, exist_ok=True); (OUT / "val").mkdir(parents=True, exist_ok=True)
    import zipfile
    from numpy.lib import format as npf

    def npz_count(path, member="imgs"):
        try:                                              # быстро: только заголовок .npy
            with zipfile.ZipFile(path) as zf, zf.open(member + ".npy") as f:
                ver = npf.read_magic(f)
                rd = npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0
                return rd(f)[0][0]
        except Exception:                                 # фолбэк: полная загрузка
            with np.load(path) as z:
                return z[member].shape[0]

    rng = np.random.default_rng(42)
    srcs = sources()[:limit]
    skipped = 0; failed = []
    for wi, (well, wlg) in enumerate(srcs):
        split = "val" if is_val(well) else "train"
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem or "framed" in nlgx.stem:
                continue
            key = f"{well}__{nlgx.stem}"[:120]
            dst = OUT / split / f"{key}.npz"
            if dst.exists():                                  # РЕЗЮМ: уже нарезано
                skipped += 1; continue
            try:
                m = extract(str(nlgx))
                if not ds.real_curves(m):
                    continue
                img = find_image(nlgx)
                if not img:
                    continue
                im = Image.open(img).convert("RGB"); rgb = np.asarray(im); gray = np.asarray(im.convert("L"))
                mask, weight, stats = build_targets(m, gray)
                if mask.sum() == 0:
                    continue
                imgs, masks, wts = cut(rgb, mask, weight, T, stride, rng)
                if not imgs:
                    continue
                np.savez_compressed(dst, imgs=np.stack(imgs), masks=np.stack(masks), weights=np.stack(wts))
            except Exception as e:                            # битый файл — пропуск+флаг, не валим прогон
                failed.append(key); print(f"  ! ПРОПУСК {key}: {str(e)[:90]}")
                continue
        ntr = len(list((OUT / "train").glob("*.npz"))); nva = len(list((OUT / "val").glob("*.npz")))
        print(f"[{wi+1}/{len(srcs)}] {well} ({split}): npz train={ntr} val={nva} (резюм-пропуск={skipped})")
    # индекс из заголовков npz на диске (быстро, без декомпрессии данных)
    index = []
    for sp in ("train", "val"):
        for p in sorted((OUT / sp).glob("*.npz")):
            try:
                index.append({"key": p.stem, "well": p.stem.split("__")[0], "split": sp, "n": npz_count(p)})
            except Exception as e:
                print(f"  ! битый npz {p.name}: {e}")
    json.dump(index, open(OUT / "index.json", "w"), ensure_ascii=False)
    tr = sum(r["n"] for r in index if r["split"] == "train"); va = sum(r["n"] for r in index if r["split"] == "val")
    print(f"\nГОТОВО: файлов={len(index)} тайлов train={tr} val={va} | битых-пропущено={len(failed)} -> {OUT}")


if __name__ == "__main__":
    main()
