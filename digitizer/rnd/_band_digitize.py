r"""
_band_digitize.py — BAND-ДЕТЕКТ (расширение одиночного авто-бакета на РАЗНЕСЁННЫЕ кривые).
Обобщает _rk_digitize (жёсткий 2-split) до АВТО-N полос + классификации КАЖДОЙ полосы single/multi.
Single → трасса центроидом (~1px); multi → ФЛАГ (наложение → R&D).

Сетку/клеточки/тики игнорируем через U-Net: плотность полос и ГЕЙТ берём по prob (U-Net на сетке не горит),
а ширину/прогоны (single vs multi) — на СЫРОМ чернеле, ГЕЙТИРОВАННОМ prob (вердикт A: толщину ловить сырым ink).
Сопоставляет полосы кривым рамки по x → точность vs GT.

  python _band_digitize.py <nlgx> --prob <prob.npy> [thr=45]      # быстрый путь (кэш prob, без GPU)
  ComfyUI-python _band_digitize.py <nlgx> [--ckpt M] [thr=45]     # сам считает prob
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from dataset_build import find_image

OUT = Path(r"F:\nds\output")
COLS = [(220, 40, 40), (40, 130, 230), (40, 180, 70), (210, 150, 0), (170, 60, 200), (0, 180, 180)]


def smooth(a, k):
    return np.convolve(a, np.ones(k) / k, mode="same")


def peaks_prominence(d, min_prom, min_dist):
    """локальные максимумы с PROMINENCE (высота над высшей из двух «седловин» — устойчиво к амплитуде).
    Возвращает [(pos, prominence)] выше min_prom, прорежено по min_dist (оставляя более выраженный),
    ОТСОРТИРОВАНО по prominence убыв. → можно брать топ-N."""
    n = len(d); cand = [i for i in range(1, n - 1) if d[i] >= d[i - 1] and d[i] > d[i + 1]]
    pr = []
    for i in cand:
        j = i - 1; lb = d[i]
        while j >= 0 and d[j] <= d[i]:
            lb = min(lb, d[j]); j -= 1
        j = i + 1; rb = d[i]
        while j < n and d[j] <= d[i]:
            rb = min(rb, d[j]); j += 1
        p = d[i] - max(lb, rb)
        if p >= min_prom:
            pr.append((i, p))
    pr.sort(key=lambda t: -t[1]); kept = []                # жадно по prominence + min_dist
    for i, p in pr:
        if all(abs(i - k) >= min_dist for k, _ in kept):
            kept.append((i, p))
    return kept                                            # уже по prominence убыв.


def band_detect(dens, xL, prom_frac=0.12, min_dist=25, trim_frac=0.20, minw=12):
    """КАЖДАЯ кривая = горб плотности (prominence-пик ≥ prom_frac×макс); полосы = долины между пиками,
    ОБРЕЗАНы к своему горбу (d>trim_frac×пик) → анти-leak от соседа. prob-плотность (сетку U-Net игнорит)."""
    d = smooth(dens.astype(float), 31); dmax = d.max()
    if dmax <= 0:
        return []
    pk = peaks_prominence(d, prom_frac * dmax, min_dist)
    if not pk:
        on = np.where(d > 0.05 * dmax)[0]
        return [(xL + int(on[0]), xL + int(on[-1]) + 1)] if len(on) else []
    peaks = sorted(p for p, _ in pk)
    valleys = [p + int(np.argmin(d[p:q])) for p, q in zip(peaks, peaks[1:])]
    starts = [0] + valleys; ends = valleys + [len(d)]; res = []
    for s, e, pkpos in zip(starts, ends, peaks):
        idx = np.where(d[s:e] > trim_frac * d[pkpos])[0]   # обрезка к горбу этой кривой
        if len(idx) and idx[-1] - idx[0] >= minw:
            res.append((xL + s + int(idx[0]), xL + s + int(idx[-1]) + 1))
    return res


def refine_bands(bands, dens, xL, ink_g, ty, by, minw=18, peak_frac=0.30, win=40):
    """C (под-пик приор): multi-полосу делим на число РЯДОМ СТОЯЩИХ кривых = число прод-пиков плотности
    (каждая боковая кривая = свой горб). Режем в долинах между соседними пиками. Ограничено числом пиков
    (без взрыва). Реально наложенные (AK/T merge в 1 горб) → ≤1 пик → не делим → остаются multi-флагом."""
    out = []
    for xa, xb in bands:
        kind, _, _ = classify(ink_g, xa, xb, ty, by)
        if kind == "single":
            out.append((xa, xb)); continue
        seg = smooth(dens[xa - xL:xb - xL].astype(float), 31); n = len(seg)
        if n < 2 * minw or seg.max() <= 0:
            out.append((xa, xb)); continue
        thr = peak_frac * seg.max(); pk = []
        for i in range(1, n - 1):
            if seg[i] > thr and seg[i] == seg[max(0, i - win):i + win + 1].max():
                if not pk or i - pk[-1] > minw:
                    pk.append(i)
                elif seg[i] > seg[pk[-1]]:
                    pk[-1] = i
        if len(pk) <= 1:
            out.append((xa, xb)); continue
        bounds = [0] + [p + int(np.argmin(seg[p:q])) for p, q in zip(pk, pk[1:])] + [n]
        for a, b in zip(bounds, bounds[1:]):
            if b - a >= minw:
                out.append((xa + a, xa + b))
    return out


def row_runs(rowmask, gap=3):
    xs = np.nonzero(rowmask)[0]
    if not len(xs):
        return []
    return [(int(s[0]), int(s[-1])) for s in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1)]


def classify(ink_g, xa, xb, ty, by):
    """single vs multi по доле строк с ≥2 раздельными прогонами (на prob-гейтир. чернеле)."""
    multi = inked = 0; widths = []
    for y in range(ty, by):
        rr = row_runs(ink_g[y, xa:xb])
        if rr:
            inked += 1
            if len(rr) >= 2:
                multi += 1
            widths += [b - a + 1 for a, b in rr]
    if not inked:
        return "ПУСТО", 0.0, 0
    return ("multi" if multi / inked > 0.40 else "single"), multi / max(1, inked), (int(np.median(widths)) if widths else 0)


def trace_single(weight, prob, xa, xb, ty, by, SL=22.0, GATE=18.0):
    """НЕПРЕРЫВНОСТЬ (min-jerk): на строке с ≥2 прогонами брать прогон, ближайший к предсказанию (не
    усреднять); коаст в разрывах. Чинит усреднение на оборотах/пересечениях внутри полосы."""
    cols = np.arange(xa, xb); out = {}; x = None; v = 0.0
    for y in range(ty, by):
        seg = weight[y, xa:xb]; pm = prob[y, xa:xb] > 0.4
        xs = np.nonzero(pm)[0]
        if not len(xs):
            if x is not None:
                x = x + float(np.clip(v, -SL, SL)); v *= 0.5
            continue
        rc = []                                           # центроид каждого прогона (вес = темнота)
        for r in np.split(xs, np.nonzero(np.diff(xs) > 3)[0] + 1):
            w = seg[r]; rc.append(float((cols[r] @ w / w.sum()) if w.sum() > 0 else cols[r].mean()))
        if x is None:
            x = min(rc, key=lambda c: abs(c - (xa + xb) / 2)); v = 0.0
        else:
            pred = x + float(np.clip(v, -SL, SL))
            nx = min(rc, key=lambda c: abs(c - pred))      # ближайший прогон к предсказанию
            v = 0.6 * v + 0.4 * (nx - x); x = nx
        out[y] = x
    return out


def get_prob(nlgx, m, img, ty, by, args):
    if "--prob" in args:
        return np.load(args[args.index("--prob") + 1])
    cache = OUT / f"_bandprob_{Path(nlgx).stem[:24]}.npy"
    if cache.is_file():
        return np.load(cache)
    import torch
    from unet import UNet
    sys.path.insert(0, r"F:\nds\Auto\digitizer")
    ckpt = args[args.index("--ckpt") + 1] if "--ckpt" in args else r"F:\nds\output\unet_data\unet_best.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=dev)
    net = UNet(in_ch=3, n_classes=1, base=ck.get("base", 32)); net.load_state_dict(ck["model"]); net.to(dev).eval()
    rgb = np.asarray(Image.open(img).convert("RGB")); H, W, _ = rgb.shape
    tile, ov, bs = 256, 96, 24; step = tile - ov
    ys = list(range(0, max(1, H - tile + 1), step)); xs = list(range(0, max(1, W - tile + 1), step))
    if H > tile and ys[-1] != H - tile: ys.append(H - tile)
    if W > tile and xs[-1] != W - tile: xs.append(W - tile)
    prob = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    w1 = np.hanning(tile); win = np.outer(w1, w1).astype(np.float32) + 1e-3
    batch = []; pos = []
    def flush():
        if not batch: return
        t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div(255).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            p = torch.sigmoid(net(t)).float().cpu().numpy()[:, 0]
        for (yy, xx), pm in zip(pos, p):
            prob[yy:yy+tile, xx:xx+tile] += pm * win; wsum[yy:yy+tile, xx:xx+tile] += win
        batch.clear(); pos.clear()
    for yy in [y for y in ys if y + tile > ty and y < by]:
        for xx in xs:
            batch.append(rgb[yy:yy+tile, xx:xx+tile]); pos.append((yy, xx))
            if len(batch) >= bs: flush()
    flush()
    prob = prob / np.maximum(wsum, 1e-6)
    np.save(OUT / f"_bandprob_{Path(nlgx).stem[:24]}.npy", prob)
    return prob


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]; nlgx = args[0]
    thr = float([a for a in args if a.replace(".", "").isdigit()][0]) if any(a.replace(".", "").isdigit() for a in args[1:]) else 45.0
    m = extract(nlgx); da = m["depth_axis"]
    img = find_image(Path(nlgx)) or m.get("img_path")
    gray = np.asarray(Image.open(img).convert("L")); H, W = gray.shape
    ty, by = da["top_y"], min(H, da["bottom_y"])
    bg = int(np.median(gray[::7, ::7]))
    prob = get_prob(nlgx, m, img, ty, by, args)[:H, :W]
    gate = prob > 0.30
    ink_g = (gray < bg - thr) & gate                      # сырой ink БЕЗ сетки (гейт U-Net)
    weight = ((bg - gray.astype(np.int16)).clip(0).astype(np.float32)) * gate * (prob > 0.4)
    curves = [c for c in m["curves"] if not c["name"].split()[0].startswith("DA")]
    xL = min(s["x_left"] for s in m["scale_axes"]); xR = min(W, max(s["x_right"] for s in m["scale_axes"]))
    gt = {}
    for c in curves:
        d = {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
        if d:
            gt[c["name"].split()[0]] = d
    gtmed = {k: float(np.median(list(v.values()))) for k, v in gt.items()}
    print(f"{Path(nlgx).stem[:46]} | трек x[{xL}..{xR}] y[{ty}..{by}] кривых={len(curves)} bg={bg} thr={thr}")
    print("  рамка: " + ", ".join(f"{k}@{round(v)}" for k, v in sorted(gtmed.items(), key=lambda kv: kv[1])))

    dens = prob[ty:by, xL:xR].sum(0)
    bands = band_detect(dens, xL)
    bands = refine_bands(bands, dens, xL, ink_g, ty, by)   # C: добить тесные кластеры
    print(f"\n  BAND-ДЕТЕКТ: полос={len(bands)}")
    traces = []
    for k, (xa, xb) in enumerate(bands):
        kind, mf, mw = classify(ink_g, xa, xb, ty, by)
        owner = min(gtmed, key=lambda nm: abs(gtmed[nm] - (xa + xb) / 2)) if gtmed else None
        if owner and abs(gtmed[owner] - (xa + xb) / 2) > (xb - xa):
            owner = None
        tr = trace_single(weight, prob, xa, xb, ty, by) if kind == "single" else None
        acc = ""
        if tr and owner and len(gt[owner]) >= 5:
            common = [y for y in gt[owner] if y in tr]
            if common:
                e = np.array([abs(tr[y] - gt[owner][y]) for y in common])
                acc = f" | vs {owner}({len(common)}общ) |Δx|мед={np.median(e):.1f}px ≤5px={np.mean(e<=5)*100:.0f}%"
        traces.append((xa, xb, kind, tr))
        print(f"   полоса {k+1}: x[{xa}..{xb}] шир={xb-xa} | {kind} (multi_frac={mf:.2f} мед.шир={mw}px)"
              f"{' → '+owner if owner else ''}{acc}")

    drw = Image.open(img).convert("RGB"); dr = ImageDraw.Draw(drw)
    for k, (xa, xb, kind, tr) in enumerate(traces):
        for xx in (xa, xb):
            dr.line([(xx, ty), (xx, by)], fill=(255, 160, 0), width=1)
        if tr:
            pts = sorted(tr.items())
            for (ya, xa2), (yb, xb2) in zip(pts, pts[1:]):
                if abs(yb - ya) <= 3:
                    dr.line([(xa2, ya), (xb2, yb)], fill=COLS[k % len(COLS)], width=2)
    for tag, frac in [("a", 0.2), ("b", 0.5), ("c", 0.8)]:
        yc = int(ty + frac * (by - ty)); y0, y1 = max(0, yc - 300), min(H, yc + 300)
        drw.crop((max(0, xL - 30), y0, xR + 30, y1)).save(OUT / f"band_{Path(nlgx).stem[:24]}_{tag}.png")
    print(f"  оверлей -> F:\\nds\\output\\band_{Path(nlgx).stem[:24]}_[a,b,c].png (полосы оранж, трасса цветом)")


if __name__ == "__main__":
    main()
