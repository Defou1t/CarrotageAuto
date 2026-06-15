r"""
detect_masks.py — A3 (Фаза A роадмапа §6.6.7): маска НЕ-линейных зон планшета, чтобы
трекер их НЕ цифровал. Зоны:
  • полоса линейки/шапка над Depth Axis top_y (числа масштабов, единицы, маркер ПС, тики);
  • поля ниже bottom_y и вне краёв трека (рамка);
  • ЛИНИИ СЕТКИ — горизонтальные «клеточки» (и жирные 4 м) + вертикальная рамка/сетка
    (морфологическое выделение длинных непрерывных штрихов);
  • ТЕКСТ внутри трека — печатные метки глубины (3000/3270…), числа, тики
    (компактные connected-components; кривая остаётся «высокой» компонентой и НЕ маскируется).

Источник геометрии — nlgx (top_y/bottom_y, x_left/x_right трека). Валидация: пиксели
ЭКСПЕРТНОЙ трассы (nlgx) НЕ должны попадать в текст/полосу (precision: не убиваем штрих);
пересечения с сеткой допустимы (тонкие, трекер мостит). cv2 в том же py3.14-окружении.

python detect_masks.py --nlgx <f.nlgx> [--thr 150] [--save-overlay] [--save-mask]
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract
from dataset_build import find_image
import dataset as ds


def track_geom(m):
    """Границы трека из nlgx: вертикаль — Depth Axis; горизонталь — крайние Scale Axis."""
    da = m["depth_axis"]
    xs_l = [s["x_left"] for s in m["scale_axes"] if s.get("x_left") is not None]
    xs_r = [s["x_right"] for s in m["scale_axes"] if s.get("x_right") is not None]
    return {
        "top_y": int(da["top_y"]), "bottom_y": int(da["bottom_y"]),
        "x_left": int(min(xs_l)) if xs_l else 0,
        "x_right": int(max(xs_r)) if xs_r else None,
    }


def ink_mask(gray, thr=150):
    return (gray < thr).astype(np.uint8)


def detect_hlines(light, min_len):
    """Длинные НЕПРЕРЫВНЫЕ горизонтальные штрихи = линии сетки (тонкие клеточки + жирные).
    Сетка СВЕТЛАЯ (gray≈170-190) — берём по светлому порогу. Close мостит JPEG-разрывы,
    затем open ловит протяжённые горизонтали: текст (разрывы между цифрами) не выживает."""
    closed = cv2.morphologyEx(light, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    # БЕЗ вертикальной дилатации: маска в толщину самой линии (1-2px) — пересечения с
    # кривой минимальны. min_len высокий → ловим сплошные линии (в промежутках, где кривой
    # нет — там клеточки и опасны); под кривой линия разорвана и не маскируется (там кривая).
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (int(min_len), 1)))


def detect_vlines(light, min_len):
    """Длинные непрерывные вертикали = рамка трека / вертикальная сетка. Длинное ядро,
    чтобы редкие отвесные участки кривой не ловились (кривая не стоит на одном x так долго)."""
    closed = cv2.morphologyEx(light, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(min_len))))
    return cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))


def detect_text(residual, full_ink, hmin=8, hmax=70, wmax=140, fill_min=0.18,
                pad=3, tall=200):
    """Компактные компоненты = печатный текст/числа/тики. Кривая = очень высокая
    компонента → отбрасывается. КЛЮЧ робастности на любом масштабе: кандидат отвергается,
    если он входит в ВЫСОКУЮ (>tall px) компоненту в ПОЛНЫХ чернилах — значит это кусок
    кривой (residual дробит кривую жирными линиями на короткие фрагменты, особенно 1:500).
    Маска = ПИКСЕЛИ компонент (не залитый bbox), дилатированные на pad."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    # карта высоты «родной» компоненты в полных чернилах (для каждого пикселя)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(full_ink, connectivity=8)
    full_h_map = stats2[:, cv2.CC_STAT_HEIGHT].astype(np.int32)[lab2]
    boxes, keep = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (hmin <= h <= hmax and 2 <= w <= wmax):
            continue
        if area < fill_min * w * h:
            continue
        # макс. высота родной компоненты по пикселям кандидата: высокая → кусок кривой
        cm = lab[y:y+h, x:x+w] == i
        if full_h_map[y:y+h, x:x+w][cm].max() > tall:
            continue
        boxes.append((x, y, w, h)); keep.append(i)
    sel = np.isin(lab, keep).astype(np.uint8) if keep else np.zeros(residual.shape, np.uint8)
    mask = cv2.dilate(sel, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1)))
    return mask, boxes


def compute_masks(rgb, m, thr=150, grid_thr=200, frame_pad=30):
    """dict масок (uint8 H×W): ruler, frame, grid, text + combined exclude."""
    H, W = rgb.shape[:2]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    geom = track_geom(m)
    ty, by = geom["top_y"], geom["bottom_y"]
    xl0, xr0 = geom["x_left"], geom["x_right"] or W
    # паддинг краёв: кривая может ВЫХОДИТЬ за x_left/x_right калибровки (off-scale) —
    # не маскируем эту зону как рамку
    xl = max(0, xl0 - 5); xr = min(W, xr0 + frame_pad)
    tw = max(1, xr - xl)
    ink = ink_mask(g, thr)

    # 1) полоса линейки/шапка (выше top_y) и поля ниже bottom_y + вне краёв трека
    ruler = np.zeros((H, W), np.uint8)
    ruler[:ty] = 1
    ruler[by:] = 1
    frame = np.zeros((H, W), np.uint8)
    if xl > 0:
        frame[:, :xl] = 1
    if xr < W:
        frame[:, xr:] = 1

    # ограничим детект линий/текста телом трека (между top_y..bottom_y, внутри краёв)
    body = np.zeros((H, W), np.uint8)
    body[ty:by, xl:xr] = 1
    ink_body = ink * body                                   # тёмные чернила (текст/кривые)
    # СВЕТЛО-СЕРАЯ полоса [thr, grid_thr): сама сетка, но БЕЗ тёмной кривой (<thr) — иначе
    # плоские участки кривой ловятся как горизонтали (баг Pn-BKZ). Крест кривой через линию
    # даёт тёмный разрыв → мостится close в detect_hlines.
    grid_band = (((g >= thr) & (g < grid_thr)).astype(np.uint8)) * body

    # 2) сетка-«клеточки»: ТОЛЬКО горизонтали. Вертикали НЕ маскируем — вертикальный
    #    грид-детект ловит отвесные участки кривой (особенно гладкой SP) → съедал бы штрих.
    #    Горизонталь пересекает кривую тонко (трекер мостит).
    mlh = max(60, int(0.45 * tw))
    grid = detect_hlines(grid_band, mlh) * body

    # 3) текст: тёмные чернила МИНУС тёмные ЖИРНЫЕ горизонтали (редкие 4 м-линии — чтобы
    #    отлепить цифры, сидящие на линии). Тонкую светлую сетку НЕ вычитаем (раскрошила бы
    #    кривую на 14px-куски → ложный «текст»). Кривая = высокая компонента (h≫hmax) → не текст.
    dark_hlines = detect_hlines(ink_body, mlh)
    residual = ((ink_body > 0) & (dark_hlines == 0)).astype(np.uint8)
    text, boxes = detect_text(residual, ink_body)
    text = text * body

    exclude = ((ruler | frame | grid | text) > 0).astype(np.uint8)
    return {"ruler": ruler, "frame": frame, "grid": grid, "text": text,
            "exclude": exclude, "boxes": boxes, "geom": geom, "ink": ink}


def gt_trace_mask(m, H, W):
    """Объединённая маска экспертных трасс (реальные кривые)."""
    u = np.zeros((H, W), bool)
    for c in ds.real_curves(m):
        u |= ds.curve_mask(c, H, W, stroke=3, max_gap_rows=30)
    return u


def validate(masks, m, H, W):
    gt = gt_trace_mask(m, H, W)
    tot = int(gt.sum())
    if not tot:
        print("нет GT-трасс для валидации"); return
    print(f"\nВАЛИДАЦИЯ vs экспертная трасса ({tot} px):")
    for nm in ("ruler", "frame", "grid", "text", "exclude"):
        hit = int((gt & (masks[nm] > 0)).sum())
        print(f"  трасса ∩ {nm:8}: {hit:6d} px = {hit/tot*100:5.2f}%  "
              f"(маска кроет {masks[nm].mean()*100:4.1f}% листа)")
    print("  ← text/ruler/frame должны быть ~0% (не убиваем штрих); "
          "grid малый % (пересечения, мостимы).")


def save_overlay(rgb, masks, m, H, W, stem):
    """Полный (даунскейл) + два native-кропа с подсветкой зон и GT-трассой (зелёная)."""
    gt = gt_trace_mask(m, H, W)
    def paint(reg):
        y0, y1, x0, x1 = reg
        sub = rgb[y0:y1, x0:x1].copy()
        for nm, col in [("ruler", (255, 230, 0)), ("frame", (255, 230, 0)),
                        ("grid", (0, 200, 255)), ("text", (255, 0, 0))]:
            msk = masks[nm][y0:y1, x0:x1] > 0
            sub[msk] = (0.45 * sub[msk] + 0.55 * np.array(col)).astype(np.uint8)
        g = gt[y0:y1, x0:x1]
        sub[g] = np.array([0, 220, 0], np.uint8)
        return sub
    ty = masks["geom"]["top_y"]
    # 1) полный даунскейл
    full = paint((0, H, 0, W))
    Image.fromarray(full).resize((W // 6, H // 6), Image.LANCZOS).save(
        rf"F:\nds\output\a3_overlay_{stem}.png")
    # 2) native-кроп шапки (вокруг top_y)
    Image.fromarray(paint((max(0, ty - 560), ty + 220, 0, W))).save(
        rf"F:\nds\output\a3_overlay_{stem}_header.png")
    # 3) native-кроп середины (метка глубины + клеточки)
    ym = (ty + masks["geom"]["bottom_y"]) // 2
    Image.fromarray(paint((ym - 250, ym + 250, 0, W))).save(
        rf"F:\nds\output\a3_overlay_{stem}_mid.png")
    print(f"\noverlay -> a3_overlay_{stem}[_header/_mid].png")


def main():
    a = sys.argv[1:]
    nlgx = a[a.index("--nlgx") + 1]
    thr = int(a[a.index("--thr") + 1]) if "--thr" in a else 150
    sys.stdout.reconfigure(encoding="utf-8")
    m = extract(nlgx)
    img = find_image(Path(nlgx))
    rgb = np.asarray(Image.open(img).convert("RGB"))
    H, W = rgb.shape[:2]
    masks = compute_masks(rgb, m, thr)
    g = masks["geom"]
    print(f"image {W}x{H}; трек x[{g['x_left']}..{g['x_right']}] y[{g['top_y']}..{g['bottom_y']}]")
    print(f"масок: ruler {masks['ruler'].mean()*100:.1f}%  frame {masks['frame'].mean()*100:.1f}%  "
          f"grid {masks['grid'].mean()*100:.1f}%  text {masks['text'].mean()*100:.1f}% "
          f"({len(masks['boxes'])} тексто-боксов)  exclude {masks['exclude'].mean()*100:.1f}%")
    validate(masks, m, H, W)
    stem = Path(nlgx).stem[:30]
    if "--save-overlay" in a:
        save_overlay(rgb, masks, m, H, W, stem)
    if "--save-mask" in a:
        p = rf"F:\nds\output\a3_mask_{stem}.png"
        Image.fromarray((masks["exclude"] * 255).astype(np.uint8)).save(p)
        print(f"mask -> {p}")


if __name__ == "__main__":
    main()
