r"""
prep_bkz.py — A1 / шаг 1: нарезка 2-канальных тайлов для трекера пары градиент-зондов BKZ.
Канал0/канал1 = ink-faithful маски пары кривых (D1: GZ21/GZ31; D2: GZ41/OGZ1) из nlgx-GT.
Переиспользует MK-тулинг (ink_faithful_masks / snap / hard_rows / cut_tiles), НО:

  ★ НЕ strip_levels — в BKZ перевынос (5×/25×) = ЧАСТЬ кривой (перо продолжает off-scale),
    трекер обязан вести ВЕСЬ путь пера. Маска рисуется по трассе всех уровней; на level-скачке
    |Δx| велик → curve_mask(max_dx) РВЁТ линию (без ложного горизонт-моста), нити остаются
    вертикальными — это верно (перо физически поднимается и переносится).

  ★ Цель трекера — ЧИСТО РАЗДЕЛИТЬ 2 нити (имя GZ21/GZ31 тривиально, эксперт переименует).
    D1-пара разделима (90–176px) → каналы = чистые раздельные полосы; D2-пара часто слиплась
    (20–70px, <15px до 40%) → там MK-подобная стена (спорная тушь → оба канала, как MK).

Диагностика (bkz_diag.py) показала: трекинг = первичный провал (auto/ ~0%, 60–915px);
идентичность НЕ провал на D1. После трекинга — decode_levels на x-трассе → значение.

  python prep_bkz.py [--ch0 GZ21 --ch1 GZ31] [--side D1] [--tile 256 --stride 110]
                     [--out F:\nds\output\bkz_data] [--exclude Yatskivska_001,Pn_Zavoda_001]
  python prep_bkz.py --viz <plate.nlgx>   # рендер 2-канального таргета (санити, без нарезки)
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # digitizer на путь
import dataset as ds
from extract_nlgx import extract, NULL
from dataset_build import find_image, align_frac
# переиспользуем проверенный MK-тулинг
from prep_mk import ink_faithful_masks, snap_to_ink, hard_rows, cut_tiles, is_val
Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")


def curves_tok(m, tok):
    """Кривые по ПОЛНОМУ первому токену имени (GZ21 ≠ GZ31 — ds.mnemonic обрезает цифры)."""
    return [c for c in ds.real_curves(m) if c["name"].split()[0].upper() == tok.upper()]


def build_masks(nlgx, ch0_mn, ch1_mn, snap=False):
    """→ (rgb, ch0, ch1, ign, hard) или None. Маски всех уровней, ink-faithful."""
    nlgx = Path(nlgx)
    try:
        m = extract(str(nlgx))
    except Exception:
        return None
    c0 = curves_tok(m, ch0_mn)                   # БЕЗ strip_levels — весь путь пера
    c1 = curves_tok(m, ch1_mn)
    if not c0 or not c1:
        return None
    img_path = find_image(nlgx)
    if not img_path:
        return None
    im = Image.open(img_path).convert("RGB"); rgb = np.asarray(im)
    gray = np.asarray(im.convert("L")); H, W = gray.shape
    if snap:
        c0 = [snap_to_ink(gray, c) for c in c0]
        c1 = [snap_to_ink(gray, c) for c in c1]
    c0 = [c for c in c0 if align_frac(gray, c)[0] >= 0.72]
    c1 = [c for c in c1 if align_frac(gray, c)[0] >= 0.72]
    if not c0 or not c1:
        return None
    ch0, ch1, ign = ink_faithful_masks(gray, c0, c1, H, W)
    hard = hard_rows(ch0, ch1)
    return rgb, ch0, ch1, ign, hard


def viz(nlgx, ch0_mn, ch1_mn, out_png):
    r = build_masks(nlgx, ch0_mn, ch1_mn)
    if r is None:
        print("нет пары", ch0_mn, ch1_mn, "в", nlgx); return
    rgb, ch0, ch1, ign, hard = r
    ys, xs = np.nonzero(ch0 | ch1)
    y0, y1 = ys.min(), ys.max(); x0, x1 = max(0, xs.min() - 40), min(rgb.shape[1], xs.max() + 40)
    # окно ~1200px по центру полосы кривых
    cy = (y0 + y1) // 2; win = 1200
    yy0, yy1 = max(y0, cy - win // 2), min(y1, cy + win // 2)
    sub = rgb[yy0:yy1, x0:x1].copy()
    over = sub.copy()
    m0 = ch0[yy0:yy1, x0:x1]; m1 = ch1[yy0:yy1, x0:x1]; mi = ign[yy0:yy1, x0:x1]
    over[m0] = [255, 0, 0]      # ch0 красный
    over[m1] = [0, 200, 0]      # ch1 зелёный
    over[m0 & m1] = [255, 255, 0]  # спорная (оба) жёлтый
    over[mi & ~m0 & ~m1] = [0, 150, 255]  # ignore голубой
    blend = (0.45 * sub + 0.55 * over).astype(np.uint8)
    Image.fromarray(blend).save(out_png)
    both = int((m0 & m1).sum())
    print(f"viz -> {out_png}  окно y{yy0}-{yy1} x{x0}-{x1}; ch0px={int(m0.sum())} ch1px={int(m1.sum())} "
          f"спорных(оба)={both} ign={int(mi.sum())}")


def main():
    args = sys.argv[1:]
    if "--viz" in args:
        nlgx = args[args.index("--viz") + 1]
        ch0 = args[args.index("--ch0") + 1] if "--ch0" in args else "GZ21"
        ch1 = args[args.index("--ch1") + 1] if "--ch1" in args else "GZ31"
        sp = Path(r"C:\Users\Defou1t\AppData\Local\Temp\claude\F--nds-Auto\181f3156-af1d-4233-8c94-d4e2f7dd770f\scratchpad")
        viz(nlgx, ch0, ch1, str(sp / "bkz_target_viz.png"))
        return
    T = 256; stride = 110; negfrac = 0.15; valfrac = 0.18; maxtiles = 180; snap = False
    ch0_mn = "GZ21"; ch1_mn = "GZ31"; side = None
    out = Path(r"F:\nds\output\bkz_data")
    exclude = {"Yatskivska_001", "Pn_Zavoda_001"}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tile": T = int(args[i+1]); i += 2
        elif a == "--stride": stride = int(args[i+1]); i += 2
        elif a == "--negfrac": negfrac = float(args[i+1]); i += 2
        elif a == "--valfrac": valfrac = float(args[i+1]); i += 2
        elif a == "--ch0": ch0_mn = args[i+1]; i += 2
        elif a == "--ch1": ch1_mn = args[i+1]; i += 2
        elif a == "--side": side = args[i+1]; i += 2      # D1|D2 фильтр планшетов по имени
        elif a == "--out": out = Path(args[i+1]); i += 2
        elif a == "--snap": snap = True; i += 1
        elif a == "--exclude": exclude = set(args[i+1].split(",")); i += 2
        else: i += 1
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    index = []; n_tiles = {"train": 0, "val": 0}
    for wl in sorted([d for d in ARCHIVE.iterdir() if d.is_dir()]):
        wlg = wl / "wlg"
        if not wlg.is_dir() or wl.name in exclude:
            if wl.name in exclude: print(f"  [excl] {wl.name}")
            continue
        split = "val" if is_val(wl.name, valfrac) else "train"
        for nlgx in sorted(wlg.glob("*_BKZ_*.nlgx")):
            if "_auto" in nlgx.stem:
                continue
            if side and f"_{side}." not in nlgx.name:
                continue
            r = build_masks(nlgx, ch0_mn, ch1_mn, snap=snap)
            if r is None:
                continue
            rgb, ch0, ch1, ign, hard = r
            m2 = np.stack([ch0, ch1, ign], -1)
            imgs, masks = cut_tiles(rgb, m2, T, stride, negfrac, rng, maxtiles, hard=hard)
            if not imgs:
                continue
            fn = out / split / f"{wl.name}__{nlgx.stem[:46]}.npz"
            np.savez_compressed(fn, imgs=np.stack(imgs).astype(np.uint8),
                                masks=np.stack(masks).astype(np.uint8))
            index.append({"npz": str(fn), "well": wl.name, "split": split, "n_tiles": len(imgs)})
            n_tiles[split] += len(imgs)
            print(f"  [{split}] {wl.name}/{nlgx.stem[:30]:<30} tiles={len(imgs)}")
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    wtr = len({r['well'] for r in index if r['split'] == 'train'})
    wva = len({r['well'] for r in index if r['split'] == 'val'})
    print(f"\n=== BKZ PREP ({ch0_mn}/{ch1_mn}) -> {out} ===\nfiles={len(index)} "
          f"tiles train={n_tiles['train']} val={n_tiles['val']} wells train={wtr} val={wva}")


if __name__ == "__main__":
    main()
