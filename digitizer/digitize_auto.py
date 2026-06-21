r"""
digitize_auto.py — АВТОНОМНАЯ оцифровка (БЕЗ трассы-эталона): шаблон даёт только РАМКУ
(Depth/Scale Axis + определения кривых), трасса строится из ИЗОБРАЖЕНИЯ. Для новых скважин,
где эксперт задал калибровку, но НЕ трассировал (экономит ручную трассировку).

Конвейер: track_identity (B3, цвет→штрихи) → на КАЖДУЮ кривую-слот назначаем B3-линии по ЦВЕТУ
→ per-row сборка трассы (предсказание по скорости + ближайшее чернило) → set_tag(35490) → _auto.nlgx.

ОГРАНИЧЕНИЕ: per-row не берёт ГОРИЗОНТАЛЬНЫЕ спайки (микрозонды MBK/MGZ) — x(y) одозначен, спайк
многозначен; нужен 2D-обход штриха. Хорошо для пиково-ВЕРТИКАЛЬНЫХ (резистив BKZ/IK/SP).

python digitize_auto.py <frame.nlgx> [--out DIR]
"""
import sys, struct
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
from write_nlgx import read_full, write_full, write_bck, set_tag, find_ifd
from dataset_build import find_image
import track_identity as ti
import extract_instances as ei
from digitize_b3 import curve_pred


def _runs(rowmask, gap=6):
    xs = np.nonzero(rowmask)[0]
    if not len(xs):
        return []
    return [c.mean() for c in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1)]


def trace_color(chan, ty, by, slmax=25):
    """Per-row трасса одного цвета: ведём по непрерывности (предсказание x+v, ближайший прогон)."""
    tr = {}; x = v = None
    for y in range(ty, by):
        e = _runs(chan[y] > 0)
        if not e:
            if x is not None:
                x = x + float(np.clip(v, -slmax, slmax))
            continue
        if x is None:
            x = e[int(np.argmin(e))]; v = 0.0; tr[y] = x; continue
        pred = x + float(np.clip(v, -slmax, slmax))
        nx = e[int(np.argmin([abs(c - pred) for c in e]))]
        v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = x
    return tr


def digitize_one(nlgx, out=None, verbose=True):
    nlgx = str(nlgx)
    out = Path(out) if out else Path(r"F:\nds\output")
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(nlgx).stem
    m = extract(nlgx)
    img = m.get("img_path")
    if not img or not Path(img).is_file():
        img = find_image(Path(nlgx))                  # fallback: мёртвый img_path в шаблоне
    if not img or not Path(img).is_file():
        if verbose: print(f"  {stem}: нет картинки"); return None
    rgb = np.asarray(Image.open(img).convert("RGB")); H, W = rgb.shape[:2]
    lines, masks = ti.track_identity(rgb, m)
    g = masks["geom"]; ty, by = g["top_y"], min(H, g["bottom_y"])
    xl = max(0, g["x_left"] - 5); xr = min(W, (g["x_right"] or W) + 30)
    body = np.zeros((H, W), np.uint8); body[ty:by, xl:xr] = 1
    chans = ei.classify_ink(rgb, body)
    chans["black"] = (chans["black"] & (masks["exclude"] == 0)).astype(np.uint8)

    # цвет каждой кривошки определяем по доминирующему цвету B3-линий в её x-полосе;
    # для одно-кривного трека — просто доминирующий цвет всех линий.
    curves = [c for c in m["curves"] if not c["name"].split()[0].startswith("DA")]
    ifds = read_full(open(nlgx, "rb").read())
    written = []
    for c in curves:
        short = c["name"].split()[0]
        # доминирующий цвет линий (по сумме длин)
        bycol = {}
        for L in lines:
            bycol[L["color"]] = bycol.get(L["color"], 0) + len(L["tr"])
        color = max(bycol, key=bycol.get) if bycol else "black"
        tr = trace_color(chans[color], ty, by)
        if len(tr) < 30:
            continue
        n = c["n_rows"]; top_y = c["top_y"]
        new_xs = [int(round(tr[top_y + i])) if (top_y + i) in tr else NULL for i in range(n)]
        idxs = find_ifd(ifds, curve_pred(short))
        if not idxs:
            continue
        set_tag(ifds, idxs[0], 35490, 4, new_xs)
        # сегмент на весь диапазон, level 0 — без 35492/94/96/98 NeuraLOG НЕ рисует кривую
        # (рамка-заглушка имела 35492=0). NB: одношкальный выход; перевыносы (стадия 3) впишут уровни.
        set_tag(ifds, idxs[0], 35492, 4, [1])
        set_tag(ifds, idxs[0], 35494, 4, [int(top_y)])
        set_tag(ifds, idxs[0], 35496, 4, [int(top_y + n - 1)])
        set_tag(ifds, idxs[0], 35498, 4, [0])
        written.append((short, color, sum(1 for x in new_xs if x != NULL)))
    data = write_full(ifds)
    dst = out / f"{stem}_auto.nlgx"
    open(dst, "wb").write(data)
    open(out / f"{stem}_auto.bck", "wb").write(write_bck(data))
    if verbose:
        for s, col, npts in written:
            print(f"  {stem[:42]:<42} {s} ({col}) {npts} точек")
        print(f"-> {dst}")
    return str(dst)


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return
    out = a[a.index("--out") + 1] if "--out" in a else None
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    digitize_one(a[0], out)


if __name__ == "__main__":
    main()
