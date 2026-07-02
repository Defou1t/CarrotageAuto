r"""Трек 2 / шаг 1 — нарезка тайлов для разделителя MK (MGZ vs MPZ).
Канал0 = маска MGZ (тонкая), канал1 = MPZ (толстая) из РАЗДЕЛЬНЫХ трасс nlgx-GT; канал2 = IGNORE
(тушь вне обеих кривых — не штрафуем). Модель учит толщину+непрерывность, разрешая пересечения.
Тайлы вдоль объединения кривых + негативы + ПЕРЕСЭМПЛИНГ сближений (свопы живут там). Сплит
train/val по скважине. Чистый numpy/PIL (без torch).

v7-фиксы таргетов (анализ 02.07 — таргеты сами учили схлопывание):
  1) спорная тушь (обе центр-линии в радиусе) → В ОБА канала (раньше только к0 → эрозия MPZ
     на каждом сближении = учебный сигнал «рядом со второй кривой мой канал гаснет»);
  2) тёмное-но-далёкое (> radius от обеих) → ignore-канал, НЕ жёсткий негатив (при align 0.72
     до 28% GT-точек мимо туши — модель училась давить собственную кривую);
  3) центр-линии без горизонтальных «мостов» 5×-перевыносов (curve_mask max_dx).

  python prep_mk.py [--tile 256 --stride 110 --negfrac 0.15 --valfrac 0.18 --out F:\nds\output\mk_data]
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


def ink_faithful_masks(gray, mgz, mpz, H, W, dark_thr=120, radius=6, max_dx=60, ring=14):
    """ch0/ch1 = тёмные пиксели у центральной линии MGZ/MPZ (радиус `radius`); толщина штриха
    сохраняется, маска липнет к реальной туши. СПОРНАЯ тушь (обе линии в радиусе) → в ОБА канала
    (per-channel сигмоиды это допускают). ign = тёмное в КОЛЬЦЕ radius<d<=ring вокруг своих
    кривых (зона рассинхрона GT) — в лоссе не штрафуется. Центр-линии рвутся при |Δx|>max_dx.

    УРОК v7 (гейт 02.07): ignore на ВСЮ чужую тушь снимает у модели навык подавлять соседние
    треки → на сканах MBK+MK полоса prob уезжает на чужой трек (BEZLUD px 1656). Поэтому кольцо:
    рассинхрон прощаем, далёкая чужая тушь остаётся жёстким негативом."""
    import cv2
    cen0 = np.zeros((H, W), bool); cen1 = np.zeros((H, W), bool)
    for c in mgz: cen0 |= ds.curve_mask(c, H, W, stroke=2, max_dx=max_dx)
    for c in mpz: cen1 |= ds.curve_mask(c, H, W, stroke=2, max_dx=max_dx)
    d0 = cv2.distanceTransform((~cen0).astype(np.uint8), cv2.DIST_L2, 5)
    d1 = cv2.distanceTransform((~cen1).astype(np.uint8), cv2.DIST_L2, 5)
    dark = gray < dark_thr
    near0 = dark & (d0 <= radius)
    near1 = dark & (d1 <= radius)
    ch0 = near0 | cen0                       # гарантируем центральную линию даже на слабой туши
    ch1 = near1 | cen1                       # спорные пиксели легально в обоих каналах
    ign = dark & ~ch0 & ~ch1 & (np.minimum(d0, d1) <= ring)   # только кольцо рассинхрона
    return ch0, ch1, ign


def snap_to_ink(gray, curve, win=6, dark_thr=140):
    """Снап полилинии GT к ближайшему гребню тёмного — рассинхрон скана лечится У ИСТОЧНИКА
    (анализ 02.07: при align 0.72 до 28% GT-точек мимо туши → таргеты-галлюцинации + давление
    собственной кривой). Сдвиг сглаживается медианой по 9 соседним точкам трассы: рассинхрон
    глобально-плавный (варп скана), point-wise прыжок на соседнюю кривую исключается."""
    from extract_nlgx import NULL as _N
    H, W = gray.shape
    ty = curve["top_y"]; xs = list(curve["xs"])
    pts = [(i, x) for i, x in enumerate(xs) if x != _N]
    if len(pts) < 9:
        return curve
    offs = {}
    for i, x in pts:
        y = ty + i
        if not (0 <= y < H):
            continue
        lo, hi = max(0, int(x) - win), min(W, int(x) + win + 1)
        if hi <= lo:
            continue
        seg = gray[y, lo:hi].astype(int)
        j = int(np.argmin(seg))
        if seg[j] < dark_thr:                                    # рядом реально есть тушь
            offs[i] = lo + j - x
    keys = sorted(offs)
    if not keys:
        return curve
    arr = np.array([offs[k] for k in keys], float)
    out = dict(curve); nxs = list(xs)
    for k_i, k in enumerate(keys):
        sm = float(np.median(arr[max(0, k_i - 4):k_i + 5]))
        nxs[k] = int(round(xs[k] + sm))
    out["xs"] = nxs
    return out


def hard_rows(cen0, cen1, sep_px=25):
    """Строки сближения кривых (per-row |x0−x1| < sep_px) — там живут свопы; пересэмплим."""
    H = cen0.shape[0]
    sx0 = np.zeros(H); c0 = np.zeros(H); sx1 = np.zeros(H); c1 = np.zeros(H)
    ys, xs = np.nonzero(cen0); np.add.at(sx0, ys, xs); np.add.at(c0, ys, 1)
    ys, xs = np.nonzero(cen1); np.add.at(sx1, ys, xs); np.add.at(c1, ys, 1)
    both = (c0 > 0) & (c1 > 0)
    d = np.abs(sx0 / np.maximum(c0, 1) - sx1 / np.maximum(c1, 1))
    return both & (d < sep_px)


def strip_levels(curve):
    """Копия кривой БЕЗ перевыносов: строки level>0-сегментов → NULL. Домен: 1×/5× = отдельные
    линии; 5×-ветвь в таргете канала учила модель бить по ней на инференсе (BOGAT MGZ ~129px
    мимо в правильной полосе, гейт v7 02.07). После отсечения её тушь (за ring) = негатив —
    модель учится подавлять 5×-ветвь, что и нужно 1×-оцифровке (5× — отдельный scope)."""
    from extract_nlgx import NULL as _N
    segs = [(s, e) for s, e, lvl in (curve.get("segments") or []) if lvl and lvl > 0]
    if not segs:
        return curve
    ty = curve["top_y"]
    xs = list(curve["xs"])
    for s, e in segs:
        for i in range(max(0, s - ty), min(len(xs), e - ty + 1)):
            xs[i] = _N
    out = dict(curve); out["xs"] = xs
    return out


def is_val(well, valfrac):
    return int(hashlib.md5(well.encode()).hexdigest(), 16) % 1000 < valfrac * 1000


def curves_mn(m, mn):
    return [c for c in ds.real_curves(m) if ds.mnemonic(c["name"]).upper() == mn]


def cut_tiles(img, m2, T, stride, negfrac, rng, maxtiles, hard=None, hard_boost=2):
    """Тайлы вдоль ОБЪЕДИНЕНИЯ кривых + негативы. img HxWx3, m2 HxWxC (MGZ,MPZ[,ign]).
    hard: bool[H] строки сближения — тайлы с ними дублируются hard_boost× с джиттером
    (при равномерном шаге сближения тонут среди тривиальных разделённых — модель их почти не видит)."""
    H, W = m2.shape[:2]; half = T // 2
    union = m2[..., :2].any(2)
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
    if hard is not None:
        extra = []
        for cy, cx in centers:
            if hard[max(0, cy - half):cy + half].any():
                for _ in range(hard_boost):
                    extra.append((cy + int(rng.integers(-stride // 2, stride // 2 + 1)),
                                  cx + int(rng.integers(-40, 41))))
        centers += extra
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
    T = 256; stride = 110; negfrac = 0.15; valfrac = 0.18; maxtiles = 180; scale = 1; snap = False
    out = Path(r"F:\nds\output\mk_data")
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tile": T = int(args[i+1]); i += 2
        elif a == "--stride": stride = int(args[i+1]); i += 2
        elif a == "--negfrac": negfrac = float(args[i+1]); i += 2
        elif a == "--valfrac": valfrac = float(args[i+1]); i += 2
        elif a == "--scale": scale = int(args[i+1]); i += 2   # апскейл планшета ×S перед нарезкой
        elif a == "--out": out = Path(args[i+1]); i += 2
        elif a == "--snap": snap = True; i += 1               # снап GT к туши (возвращает align-fail скважины)
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
            mgz = [strip_levels(c) for c in curves_mn(m, "MGZ")]   # без 5×-перевыносов (level>0)
            mpz = [strip_levels(c) for c in curves_mn(m, "MPZ")]
            if not mgz or not mpz:
                continue
            img_path = find_image(nlgx)
            if not img_path:
                continue
            im = Image.open(img_path).convert("RGB"); rgb = np.asarray(im)
            gray = np.asarray(im.convert("L")); H, W = gray.shape
            if snap:                                          # рассинхрон лечится до фильтра →
                mgz = [snap_to_ink(gray, c) for c in mgz]     # align-fail скважины возвращаются
                mpz = [snap_to_ink(gray, c) for c in mpz]
            mgz = [c for c in mgz if align_frac(gray, c)[0] >= 0.72]   # ниже порог → все планшеты
            mpz = [c for c in mpz if align_frac(gray, c)[0] >= 0.72]
            if not mgz or not mpz:
                continue
            # ink-faithful маски v7: тёмный штрих у центр-линии своего канала; спорная тушь в ОБА
            # канала; чужая тушь → ignore-канал (не негатив); центр-линии без 5×-мостов.
            ch0, ch1, ign = ink_faithful_masks(gray, mgz, mpz, H, W)
            hard = hard_rows(ch0, ch1)
            if scale != 1:                                   # ×S апскейл: штрих 2× шире → толщина/локализация резче
                import cv2
                rgb = cv2.resize(rgb, (W * scale, H * scale), interpolation=cv2.INTER_LINEAR)
                ch0 = cv2.resize(ch0.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST).astype(bool)
                ch1 = cv2.resize(ch1.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST).astype(bool)
                ign = cv2.resize(ign.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST).astype(bool)
                hard = np.repeat(hard, scale)
            m2 = np.stack([ch0, ch1, ign], -1)
            imgs, masks = cut_tiles(rgb, m2, T, stride * scale, negfrac, rng, maxtiles, hard=hard)
            if not imgs:
                continue
            fn = out / split / f"{wl.name}__{nlgx.stem[:46]}.npz"
            np.savez_compressed(fn, imgs=np.stack(imgs).astype(np.uint8),
                                masks=np.stack(masks).astype(np.uint8))   # N×T×T×3 (MGZ,MPZ,ign)
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
