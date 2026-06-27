r"""
_stkds_digitize.py — ИНТЕГРАЦИЯ: автотрасса Yatskivska STK+DS (2 рамки) по ЦВЕТУ.
DA1 (лев x[51..519]): PZ чёрный, GZ зелёный (5X-оборот), SP красный. DA2 (прав x[694..1163]): DS чёрный.
Каждая (рамка,цвет)=1 кривая → цвет разделяет. Чёрные: PZ — сырое чернило+непрерывность от сида (U-Net его не
видит); DS — чернило+gate prob (сетку убирает). Цветные — маска цвета. Где есть трасса эксперта — берём её (GT-якорь),
ниже — по цвету. Инъекция в КОПИЮ → F:\nds\output\stkds_out (оригинал НЕ трогаем).

python _stkds_digitize.py
"""
import sys, struct
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from extract_nlgx import extract, NULL
from dataset_build import find_image
from write_nlgx import read_full, write_full, write_bck, set_tag

F = r"F:\nds\projects\Yatskivska_001\wlg\Yatskivska_1_STK+DS_2673_3008_500_D1.nlgx"
PROB = r"F:\nds\output\stkds_prob.npy"
OUT = Path(r"F:\nds\output\stkds_out")
SPEC = {                                                  # short: (xL, xR, цвет, prob-gate)
    "PZ1": (55, 240, "black", False),
    "GZ1": (51, 519, "green", False),
    "SP1": (382, 519, "red", False),
    "DS1": (694, 1163, "black", True),
}
COL = {"PZ1": (20, 20, 20), "GZ1": (0, 160, 0), "SP1": (220, 0, 0), "DS1": (90, 60, 200)}
SL, GATE = 26.0, 26.0


def structure_exclude(gray, bg, m, H, W):
    """вырезать СТРУКТУРУ бланка (на неё прыгают трассы): поля выше top_y/ниже bottom_y +
    ВЕРТИКАЛЬНЫЕ линии рамок/осей (известные x краёв треков ±3 — безопасно, кривая там подолгу
    не стоит) + детект сверх-устойчивых вертикалей (dark-frac>0.8). Горизонтали — в trace (grid_skip)."""
    ink = gray < bg - 45
    excl = np.zeros((H, W), bool)
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    excl[:ty] = True; excl[by:] = True
    xb = {51}                                             # лев. край DA1
    for s in m["scale_axes"]:
        xb.add(int(s["x_left"])); xb.add(int(s["x_right"]))
    for k in ("x_top", "x_bot"):
        if k in da:
            xb.add(int(da[k]))
    for x in xb:
        excl[:, max(0, x - 3):min(W, x + 4)] = True
    colf = ink[ty:by].mean(0)                             # сверх-устойчивые вертикали = рамка/ось
    for x in np.where(colf > 0.80)[0]:
        excl[:, max(0, x - 2):min(W, x + 3)] = True
    return excl


def masks(arr, gray, bg):
    import cv2
    R, G, Bl = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    mx = np.maximum(np.maximum(R, G), Bl); mn = np.minimum(np.minimum(R, G), Bl)
    # АДАПТИВНЫЙ порог: темнее ЛОКАЛЬНОГО фона на C → ловит БЛЕДНЫЙ карандаш, гасит пятна/неравномерность
    faint = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 41, 11) > 0
    red = (R > G + 18) & (R > Bl + 18) & (R > 70)
    green = (G > R + 4) & (G > Bl - 3) & (mx - mn > 5) & (~red) & (faint | (mx < bg - 25))
    black = faint & (mx - mn < 40) & (~red) & (~green)   # бледный тёмный десатур. (PZ/DS)
    return {"red": red, "green": green, "black": black}


def runs(rowmask, weight, xa, xb, cols, gap=3, grid_skip=False):
    seg = rowmask[xa:xb]; xs = np.nonzero(seg)[0]
    if not len(xs):
        return []
    out = []
    for r in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1):
        if grid_skip and len(r) > 0.5 * (xb - xa):       # горизонтальная сетка во весь трек → пропуск
            continue
        w = weight[xa:xb][r]
        out.append(float((cols[r] @ w / w.sum()) if w.sum() > 0 else cols[r].mean()))
    return out


def trace(mask, weight, gt, xa, xb, ty0, n, grid_skip):
    cols = np.arange(xb - xa); out = {}; x = None; v = 0.0
    seedx = float(np.median(list(gt.values()))) if gt else (xa + xb) / 2
    for j in range(n):
        y = ty0 + j
        if y in gt:                                       # якорь на трассу эксперта (истина)
            nx = float(gt[y]); v = 0.6 * v + 0.4 * (nx - x) if x is not None else 0.0
            x = nx; out[y] = x; continue
        if y >= mask.shape[0]:
            break
        rc = [xa + c for c in runs(mask[y], weight[y], xa, xb, cols, grid_skip=grid_skip)]
        if not rc:
            if x is not None:
                x = x + float(np.clip(v, -SL, SL)); v *= 0.6   # внутр. коаст БЕЗ эмита
            continue
        ref = (x + float(np.clip(v, -SL, SL))) if x is not None else seedx
        nx = min(rc, key=lambda c: abs(c - ref))
        if x is None or abs(nx - ref) <= GATE:
            v = 0.6 * v + 0.4 * (nx - x) if x is not None else 0.0; x = nx; out[y] = x  # эмит ТОЛЬКО при уверенном совпадении
        else:
            x = ref                                       # держим предсказание внутри, но НЕ эмитим (чисто, без сетки)
    return out


def inject(ifds, short, tr, W):
    for i, ifd in enumerate(ifds):
        tags = {t: raw for t, typ, c, raw in ifd["entries"]}
        if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
            n = struct.unpack("<I", tags[35488][:4])[0]; ty0 = struct.unpack("<I", tags[35474][:4])[0]
            xs = [int(round(tr[ty0 + j])) if (ty0 + j) in tr and 0 <= round(tr[ty0 + j]) < W else NULL for j in range(n)]
            set_tag(ifds, i, 35490, 4, xs)
            for t, v in [(35492, 1), (35494, ty0), (35496, ty0 + n - 1), (35498, 0)]:
                set_tag(ifds, i, t, 4, [v])
            vx = [(j, x) for j, x in enumerate(xs) if x != NULL]
            if vx:
                rw = [ty0 + j for j, _ in vx]; xv = [x for _, x in vx]
                for t, v in [(35478, min(xv)), (35480, min(rw)), (35482, max(xv)), (35484, max(rw))]:
                    set_tag(ifds, i, t, 4, [v])
            return n, ty0, sum(1 for x in xs if x != NULL)
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    m = extract(F); img = find_image(Path(F)) or m["img_path"]
    arr = np.asarray(Image.open(img).convert("RGB")); H, W, _ = arr.shape
    gray = np.asarray(Image.open(img).convert("L")); bg = int(np.median(gray[::7, ::7]))
    prob = np.load(PROB)[:H, :W]
    mk = masks(arr, gray, bg)
    excl = structure_exclude(gray, bg, m, H, W)           # вырезать рамки/оси/поля
    for k in mk:
        mk[k] = mk[k] & ~excl
    darkw = (bg - gray.astype(np.int16)).clip(0).astype(np.float32)
    gt_all = {c["name"].split()[0]: {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL} for c in m["curves"]}
    ifds = read_full(open(F, "rb").read())
    traces = {}
    print(f"{Path(F).name} | img {W}x{H} bg={bg}")
    for short, (xa, xb, color, pg) in SPEC.items():
        base = mk[color] & (prob > 0.3) if pg else mk[color]
        gt = gt_all.get(short, {})
        # n/ty0 из IFD
        info = None
        for ifd in ifds:
            tags = {t: raw for t, typ, c, raw in ifd["entries"]}
            if 35470 in tags and tags[35470].split(b"\x00")[0].decode("latin1").startswith(short + " "):
                info = (struct.unpack("<I", tags[35474][:4])[0], struct.unpack("<I", tags[35488][:4])[0]); break
        if not info:
            print(f"  {short}: не найден в IFD"); continue
        ty0, n = info
        tr = trace(base, darkw, gt, xa, xb, ty0, n, grid_skip=(color == "black" and not pg))
        traces[short] = tr
        nn, ty0i, cnt = inject(ifds, short, tr, W)
        # точность vs GT (где эксперт уже вёл)
        acc = ""
        if gt:
            common = [y for y in gt if y in tr]
            if common:
                e = np.array([abs(tr[y] - gt[y]) for y in common])
                acc = f" | vs GT({len(common)}) |Δx|мед={np.median(e):.1f}px ≤5px={np.mean(e<=5)*100:.0f}%"
        print(f"  {short} ({color}{'+prob' if pg else ''}) трек[{xa}..{xb}]: трасса {cnt}/{n} строк ({100*cnt/n:.0f}%){acc}")

    data = write_full(ifds)
    out_nlgx = OUT / f"{Path(F).stem}_auto.nlgx"
    out_nlgx.write_bytes(data); (OUT / f"{Path(F).stem}_auto.bck").write_bytes(write_bck(data))
    print(f"\n-> {out_nlgx}\n-> {OUT / (Path(F).stem + '_auto.bck')}")

    # оверлей QC (3 кропа)
    da = m["depth_axis"]; ty, by = 1507, min(H, 5600)
    drw = Image.fromarray(arr.copy()); dr = ImageDraw.Draw(drw)
    for short, tr in traces.items():
        c = COL[short]
        pts = sorted(tr.items())
        for (ya, xa2), (yb, xb2) in zip(pts, pts[1:]):
            if abs(yb - ya) <= 3:
                dr.line([(xa2, ya), (xb2, yb)], fill=c, width=2)
    for tag, frac in [("a", 0.12), ("b", 0.4), ("c", 0.7)]:
        yc = int(ty + frac * (by - ty)); y0, y1 = max(0, yc - 320), min(H, yc + 320)
        drw.crop((30, y0, 1180, y1)).save(OUT / f"stkds_qc_{tag}.png")
    print(f"оверлей QC -> {OUT}\\stkds_qc_[a,b,c].png (PZ чёрн/GZ зел/SP красн/DS фиол)")


if __name__ == "__main__":
    main()
