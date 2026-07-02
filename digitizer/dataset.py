"""
dataset.py — API датасета для оцифровки каротажа поверх расшифрованного nlgx.

nlgx уже компактно хранит метки (пиксель-трасса + калибровка), поэтому НЕ дублируем
их на диск. Здесь:
  • курация: какие кривые реальны (не псевдо-ось DAx), матч с колонками LAS;
  • растеризация маски кривой из пиксель-трассы (полилиния, без мостов через большие
    разрывы) — для обучения U-Net и для визуальной проверки;
  • тайлинг больших планшетов на окна для сети.

Источник правды — тройка (nlgx, image, las). dataset_build.py делает из них манифест.
"""
import re, math
import numpy as np
from extract_nlgx import extract, depth_of, depth_axis_ok, NULL

DA_RE = re.compile(r"^DA\d+$", re.I)   # псевдо-кривая = линия Depth Axis


def mnemonic(curve_name):
    """'BK1 DA1 SA1' -> 'BK'  (первый токен без хвостовых цифр)."""
    tok = curve_name.strip().split()[0]
    return tok.rstrip("0123456789")


def is_real_curve(curve, min_pts=50):
    tok = curve["name"].strip().split()[0]
    if DA_RE.match(tok):
        return False
    return len([x for x in curve["xs"] if x != NULL]) >= min_pts


def real_curves(model, min_pts=50):
    return [c for c in model["curves"] if is_real_curve(c, min_pts)]


def trace_points(curve):
    """list[(row_index, x)] валидных точек по возрастанию строки."""
    return [(i, x) for i, x in enumerate(curve["xs"]) if x != NULL]


def curve_polyline(model, curve):
    """list[(x, y_image)] валидных точек трассы в пикселях изображения."""
    ty = curve["top_y"]
    return [(x, ty + i) for i, x in trace_points(curve)]


def curve_mask(curve, H, W, stroke=3, max_gap_rows=30, max_dx=None):
    """
    Бинарная маска кривой (H×W bool): полилиния по валидным точкам трассы,
    соседние точки соединяются, если разрыв по строкам <= max_gap_rows
    (иначе оставляем обрыв — не мостим реальные разрывы).
    max_dx: не соединять и при скачке |Δx| больше порога — переключение level
    (1×/5×-перевынос) идёт между соседними строками и иначе рисует горизонтальный
    «мост» через планшет, который присваивает себе чужую тушь (анализ 02.07).
    """
    from PIL import Image, ImageDraw
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    ty = curve["top_y"]
    pts = trace_points(curve)
    r = max(1, stroke // 2)
    prev = None
    for i, x in pts:
        y = ty + i
        if not (0 <= y < H and 0 <= x < W):
            prev = None
            continue
        if prev is not None and (i - prev[0]) <= max_gap_rows and \
                (max_dx is None or abs(x - prev[1]) <= max_dx):
            dr.line([prev[1], prev[2], x, y], fill=1, width=stroke)
        else:
            dr.ellipse([x-r, y-r, x+r, y+r], fill=1)
        prev = (i, x, y)
    return np.asarray(im, dtype=bool)


# ----- сопоставление кривых nlgx с колонками эталонного LAS -----

def load_las(path):
    cols, rows = None, []
    with open(path, "r", encoding="latin1") as f:
        in_data = False
        for line in f:
            s = line.strip()
            if s.startswith("~A"):
                in_data = True; cols = s[2:].split(); continue
            if in_data and s and not s.startswith("#"):
                p = s.split()
                if len(p) >= 2:
                    try: rows.append([float(v) for v in p])
                    except ValueError: pass
    return cols, np.array(rows) if rows else np.zeros((0, 0))


def _recon_curve(model, curve):
    """value(depth) реконструкция кривой (для сверки с LAS). См. validate_nlgx."""
    if not depth_axis_ok(model):
        return []
    key = curve["name"].strip().split()[-1]
    fam = sorted([s for s in model["scale_axes"]
                  if s["name"].strip().split()[-1] == key], key=lambda s: s["idx"])
    if not fam:
        return []
    ty = curve["top_y"]; segs = curve["segments"]
    def level_at(y):
        last = None
        for s, e, l in segs:
            if s <= y <= e: return l
            if e < y: last = l
        return last
    out = []
    for i, x in enumerate(curve["xs"]):
        if x == NULL: continue
        y = ty + i; L = level_at(y)
        if L is None or L >= len(fam): continue
        sa = fam[L]
        v = sa["v_left"] + (x - sa["x_left"]) / (sa["x_right"] - sa["x_left"]) * (sa["v_right"] - sa["v_left"])
        out.append((depth_of(model, y), v))
    return out


def log_corr_vs_las(model, curve, las_depths, las_vals, null=-999.25):
    recon = sorted(_recon_curve(model, curve))
    if len(recon) < 5 or len(las_depths) < 5:
        return None
    ds = np.array([d for d, _ in recon]); vs = np.array([v for _, v in recon])
    m = (las_depths >= ds[0]) & (las_depths <= ds[-1]) & (las_vals > null + 1)
    if m.sum() < 5:
        return None
    ri = np.interp(las_depths[m], ds, vs)
    lv = las_vals[m]
    ok = (ri > 0) & (lv > 0)
    if ok.sum() < 5:
        # линейная корреляция, если есть неположительные значения
        a, b = ri, lv
    else:
        a, b = np.log10(ri[ok]), np.log10(lv[ok])
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def match_las(model, las_cols, las_arr):
    """
    Сопоставить реальные кривые nlgx с колонками LAS.
    По мнемонике; при нескольких на одну мнемонику — паруем по порядку.
    Возвращает {curve_name: {"las_col": str|None, "col_idx": int|None, "log_corr": float|None}}.
    """
    res = {}
    curves = real_curves(model)
    if las_arr.size == 0 or not las_cols:
        return {c["name"]: {"las_col": None, "col_idx": None, "log_corr": None} for c in curves}
    depths = las_arr[:, 0]
    # группы по мнемонике
    from collections import defaultdict
    cur_by = defaultdict(list); col_by = defaultdict(list)
    for c in curves:
        cur_by[mnemonic(c["name"]).upper()].append(c)
    for j, name in enumerate(las_cols):
        if j == 0:  # DEPTH
            continue
        col_by[name.rstrip("0123456789").upper()].append((j, name))
    for mn, cs in cur_by.items():
        cols = col_by.get(mn, [])
        for k, c in enumerate(cs):
            if k < len(cols):
                j, cname = cols[k]
                lc = log_corr_vs_las(model, c, depths, las_arr[:, j])
                res[c["name"]] = {"las_col": cname, "col_idx": j, "log_corr": lc}
            else:
                res[c["name"]] = {"las_col": None, "col_idx": None, "log_corr": None}
    return res


# ----- тайлинг для сети -----

def tiles(H, W, tile=512, overlap=64, y0=None, y1=None):
    """Генератор боксов (x0,y0,x1,y1) для нарезки планшета на окна."""
    ys0 = 0 if y0 is None else y0
    ys1 = H if y1 is None else y1
    step = tile - overlap
    for ty in range(ys0, ys1, step):
        for tx in range(0, W, step):
            yield (tx, ty, min(tx+tile, W), min(ty+tile, ys1))
