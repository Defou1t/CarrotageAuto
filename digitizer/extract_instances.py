r"""
extract_instances.py — B1 (Фаза B роадмапа §6.6.7): инстанс-извлечение линий из КАРТИНКИ
(не бинарная маска). Для авто-пайплайна без GT: выделить ЭКЗЕМПЛЯРЫ линий с их свойствами —
центр-трасса x(y), глубинный start/end, толщина, ЦВЕТ, поведение (rough_n из B2).

Идея: ЦВЕТ — сильнейший признак идентичности (правило 7 §6.6.7). Эмпирика: SP часто красная
(уникальна → фикс QC «SP↔CALI»), резистивы делятся на чёрные/синие пары. Поэтому:
  1. чернила тела трека МИНУС exclude-маска A3 (сетка/текст/полоса) — только штрих кривых;
  2. классификация по цвету (red/blue/green/black) — каналы инстансов;
  3. в каждом канале connected-components → линия-подобные → инстанс + свойства.
Внутри цвета пара линий (GZ11+GZ31 чёрные) разделяется x-непрерывностью отдельно (легче, чем все
сразу). Перевынос 1×/5× = РАЗНЫЕ штрихи → разные CC (согласуется с «отдельные линии», §6.6.7).

python extract_instances.py --nlgx <f.nlgx> [--save-overlay]
"""
import sys, re
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
from dataset_build import find_image
import dataset as ds
import detect_masks as dm
import behavior_priors as bp

COLORS = {"black": (40, 40, 40), "blue": (40, 90, 200),
          "red": (210, 50, 50), "green": (40, 160, 60)}


def classify_ink(rgb, body, dark_thr=150, sat_thr=28, pale=210):
    """Бинарные маски штриха по цвету. Цветной = sat>=sat_thr (argmax канала → red/blue/green);
    чёрный = тёмный (<dark_thr) низкосатурированный. Бледную сетку (>pale) отсекаем."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    R, G, B = [rgb[..., i].astype(np.int16) for i in range(3)]
    mx = rgb.max(2); mn = rgb.min(2); sat = (mx.astype(np.int16) - mn)
    colored = (sat >= sat_thr) & (g < pale) & (body > 0)
    red = colored & (R >= G) & (R >= B) & (R > B + 8)
    blue = colored & (B > R + 8) & (B >= G - 4)
    green = colored & (G > R + 8) & (G > B + 8)
    dark = (g < dark_thr) & (body > 0)
    black = dark & ~red & ~blue & ~green
    return {"black": black.astype(np.uint8), "blue": blue.astype(np.uint8),
            "red": red.astype(np.uint8), "green": green.astype(np.uint8)}


def instances_from_channel(mask, color, min_h, min_px=200):
    """CC канала → линия-подобные инстансы. Close мостит мелкие разрывы штриха; CC; фильтр по
    высоте/пикселям. Центр-трасса = per-row медиана x пикселей CC; толщина = медиана ширины."""
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h or area < min_px:
            continue
        sub = lab[y:y + h, x:x + w] == i
        xs_row = np.full(h, np.nan)
        th = []
        ys, xsx = np.where(sub)
        for r in range(h):
            cols = xsx[ys == r]
            if len(cols):
                xs_row[r] = x + np.median(cols)
                th.append(len(cols))
        valid = ~np.isnan(xs_row)
        if valid.sum() < min_h * 0.3:
            continue
        out.append({"color": color, "y0": int(y), "y1": int(y + h),
                    "xs_row": xs_row, "row0": int(y),
                    "thickness": float(np.median(th)) if th else 0.0,
                    "area": int(area)})
    return out


def _behavior_from_xs(m, xs_row, row0):
    """rough_n/rev из центр-трассы инстанса (формат как curve_behavior, но трасса уже x(row))."""
    valid = ~np.isnan(xs_row)
    if valid.sum() < 30:
        return None
    ys = np.where(valid)[0] + row0
    xs = xs_row[valid]
    da = m["depth_axis"]
    pxm = (da["bottom_y"] - da["top_y"]) / (da["bottom_depth"] - da["top_depth"]) or 1.0
    grid = np.arange(ys.min(), ys.max() + 1)
    xi = np.interp(grid, ys, xs)
    W = max(5, int(round(1.5 * pxm)) | 1)
    sm = np.convolve(xi, np.ones(W) / W, mode="same")
    hf = (xi - sm)[W:-W] if len(xi) > 2 * W else (xi - sm)
    span = float(np.percentile(xs, 97) - np.percentile(xs, 3)) or 1.0
    return round(float(np.std(hf)) / span, 4)


def extract_instances(rgb, m, min_h_frac=0.04):
    H, W = rgb.shape[:2]
    masks = dm.compute_masks(rgb, m)
    body = np.zeros((H, W), np.uint8)
    g = masks["geom"]; ty, by = g["top_y"], g["bottom_y"]
    xl = max(0, g["x_left"] - 5); xr = min(W, (g["x_right"] or W) + 30)
    body[ty:by, xl:xr] = 1
    body = (body > 0) & (masks["exclude"] == 0)         # тело минус не-линейные зоны A3
    chans = classify_ink(rgb, body.astype(np.uint8))
    min_h = int(min_h_frac * (by - ty))
    inst = []
    for color, msk in chans.items():
        for ins in instances_from_channel(msk, color, min_h):
            ins["rough_n"] = _behavior_from_xs(m, ins["xs_row"], ins["row0"])
            inst.append(ins)
    inst.sort(key=lambda d: -d["area"])
    return inst, masks


def _gt_xy(curve, H):
    ty = curve["top_y"]
    d = {}
    for i, x in enumerate(curve["xs"]):
        if x != NULL:
            y = ty + i
            if 0 <= y < H:
                d[y] = x
    return d


def _gt_color(rgb, gt):
    """Доминирующий цвет GT-кривой (сэмпл изображения вдоль трассы) — для сверки идентичности."""
    H, W = rgb.shape[:2]
    px = [rgb[y, x] for y, x in gt.items() if 0 <= y < H and 0 <= x < W]
    if len(px) < 20:
        return "?"
    med = np.median(np.array(px, float), 0); R, G, B = med
    if med.max() < 90 and (med.max() - med.min()) < 25:
        return "black"
    if R > G + 8 and R > B + 8:
        return "red"
    if B > R + 8:
        return "blue"
    if G > R + 8 and G > B + 8:
        return "green"
    return "black"


def validate(inst, m, rgb):
    """ПОКРЫТИЕ: резистив с перевыносом = несколько штрихов-инстансов (правило «отдельные линии»).
    Для каждой GT-кривой собираем ВСЕ штрихи с медианой |Δx|<tol на перекрытии → покрытие глубины,
    цвета штрихов, медиана ошибки. Сверяем цвет/поведение."""
    H = rgb.shape[0]
    print(f"\nИНСТАНСЫ извлечено: {len(inst)} "
          f"(цвета: {', '.join(sorted({i['color'] for i in inst}))})")
    print(f"\nПОКРЫТИЕ GT-кривых штрихами-инстансами (tol=12px):")
    TOL = 12
    for c in ds.real_curves(m):
        gt = _gt_xy(c, H)
        if len(gt) < 30:
            continue
        gtc = _gt_color(rgb, gt)
        covered, errs, cols, ds_used = set(), [], {}, 0
        for ins in inst:
            xr = ins["xs_row"]
            ov = [(y, gt[y], xr[y - ins["row0"]]) for y in gt
                  if ins["row0"] <= y < ins["row0"] + len(xr) and not np.isnan(xr[y - ins["row0"]])]
            if len(ov) < 20:
                continue
            md = float(np.median([abs(a - b) for _, a, b in ov]))
            if md < TOL:
                for y, _, _ in ov:
                    covered.add(y)
                errs += [abs(a - b) for _, a, b in ov]
                cols[ins["color"]] = cols.get(ins["color"], 0) + len(ov)
        cov = len(covered) / len(gt) * 100
        merr = float(np.median(errs)) if errs else float("nan")
        domcol = max(cols, key=cols.get) if cols else "—"
        cmatch = "✓" if domcol == gtc else f"✗(GT {gtc})"
        rn = bp.curve_behavior(m, c)
        print(f"  {bp.mnem(c['name']):<5} {c['name']:<13} покрытие={cov:3.0f}% "
              f"|Δx|={merr:4.0f}px штрихов_цвет={domcol:<5}{cmatch:<9} "
              f"rough_n GT={rn['rough_n'] if rn else '-'}")


def save_overlay(rgb, inst, m, stem):
    H, W = rgb.shape[:2]
    ov = rgb.copy()
    for ins in inst:
        col = COLORS.get(ins["color"], (255, 0, 255))
        xr = ins["xs_row"]
        for r in range(len(xr)):
            if not np.isnan(xr[r]):
                y = ins["row0"] + r; x = int(xr[r])
                if 0 <= y < H and 0 <= x < W:
                    ov[y, max(0, x - 1):x + 2] = col
    ym = (m['depth_axis']['top_y'] + m['depth_axis']['bottom_y']) // 2
    Image.fromarray(ov[ym - 300:ym + 300]).save(rf"F:\nds\output\b1_inst_{stem}_mid.png")
    Image.fromarray(ov).resize((W // 6, H // 6), Image.LANCZOS).save(
        rf"F:\nds\output\b1_inst_{stem}.png")
    print(f"\noverlay -> b1_inst_{stem}[_mid].png")


def main():
    a = sys.argv[1:]
    nlgx = a[a.index("--nlgx") + 1]
    sys.stdout.reconfigure(encoding="utf-8")
    m = extract(nlgx)
    rgb = np.asarray(Image.open(find_image(Path(nlgx))).convert("RGB"))
    inst, masks = extract_instances(rgb, m)
    g = masks["geom"]
    print(f"image {rgb.shape[1]}x{rgb.shape[0]}; трек x[{g['x_left']}..{g['x_right']}] "
          f"y[{g['top_y']}..{g['bottom_y']}]")
    validate(inst, m, rgb)
    if "--save-overlay" in a:
        save_overlay(rgb, inst, m, Path(nlgx).stem[:30])


if __name__ == "__main__":
    main()
