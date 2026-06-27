r"""
stages.py — ПОЭТАПНАЯ визуализация «как скрипт видит» лист (прямой запрос заказчика: видеть
результаты анализа НА изображении по шагам). Монтаж стадий восприятия на окне глубины, слева-направо:

  1 ОРИГИНАЛ → 2 ТЁМНОЕ (V<dark_v) → 3 ЦВЕТ (R/G/B каналы) → 4 СТРУКТУРА (вырезаемые рамка/сетка)
  → 5 ПЛАН правила (ink_foreground) → [6 МОДЕЛЬ prob → 7 ПЛАН+RECALL] → 8 ЛИНИИ+ТРАССА (AUTO/FLAG)

Панели 6–7 только если дана prob-карта recall-модели (config.prob_provider / attach_npy). Видно, ЧТО
именно берёт каждый шаг ДО трассы — где правила теряют бледное и поднимает ли модель. Чистый numpy/cv2/PIL.
"""
import numpy as np
from pathlib import Path
from . import imaging as im

CONF_RGB = {"AUTO": (255, 0, 255), "FLAG": (255, 180, 0)}


def _font(sz=20):
    from PIL import ImageFont
    for pth in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(pth, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _mask_rgb(mask, color=(0, 0, 0)):
    out = np.full((*mask.shape, 3), 255, np.uint8)
    out[mask.astype(bool)] = color
    return out


def _auto_window(rgb, sheet, p, h=1200):
    """Показательное окно = densest h-полоса переднего плана в треке (там есть форма для трассы)."""
    fr = sheet.frame
    y0, y1 = max(0, fr.top_y), min(rgb.shape[0], fr.bottom_y)
    x0 = fr.tracks[0].x_left if fr.tracks else 0
    x1 = fr.tracks[-1].x_right if fr.tracks else rgb.shape[1]
    if y1 - y0 <= h:
        return y0, y1
    rd = im.ink_foreground(rgb, p)[y0:y1, x0:x1].sum(1).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(rd)])
    band = csum[h:] - csum[:-h]
    yc = y0 + int(np.argmax(band))
    return yc, min(y1, yc + h)


def render_stages(rgb, sheet, traces, p, out, stem, window=None, prob=None, panel_w=200):
    """Собрать и сохранить <stem>_stages.png. window=(Y0,Y1) или None=авто. Возвращает (путь, meta)."""
    from PIL import Image, ImageDraw
    fr = sheet.frame
    Y0, Y1 = window if window else _auto_window(rgb, sheet, p)
    X0 = max(0, (fr.tracks[0].x_left if fr.tracks else 0) - 40)
    X1 = min(rgb.shape[1], (fr.tracks[-1].x_right if fr.tracks else rgb.shape[1]) + 12)
    HH, WW = Y1 - Y0, X1 - X0

    def w(a):
        return a[Y0:Y1, X0:X1]

    panels = [("1 ОРИГИНАЛ", w(rgb).copy()),
              ("2 ТЁМНОЕ", _mask_rgb(w(im.dark_mask(rgb, p))))]
    ch = im.color_channels(rgb, p)
    color = np.full((HH, WW, 3), 255, np.uint8)
    color[w(ch["red"])] = (220, 0, 0); color[w(ch["green"])] = (0, 160, 0); color[w(ch["blue"])] = (0, 0, 220)
    panels.append(("3 ЦВЕТ R/G/B", color))
    panels.append(("4 СТРУКТУРА", _mask_rgb(w(im.structure_mask(rgb, p)), (190, 130, 0))))
    panels.append(("5 ПЛАН правила", _mask_rgb(w(im.ink_foreground(rgb, p)))))
    if prob is not None:
        hot = (np.clip(w(prob), 0, 1) * 255).astype(np.uint8)
        heat = np.full((HH, WW, 3), 255, np.uint8)
        heat[..., 1] = 255 - hot; heat[..., 2] = 255 - hot          # белый→красный = prob
        panels.append(("6 МОДЕЛЬ prob", heat))
        panels.append(("7 ПЛАН+RECALL", _mask_rgb(w(im.ink_foreground(rgb, p, prob=prob)))))
    lt = (w(rgb).astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)   # выцвеченный фон
    for L, tr in traces:
        col = CONF_RGB.get(L.confidence, (255, 0, 255))
        for y, x in tr.items():
            if Y0 <= y < Y1 and X0 <= int(x) < X1:
                lt[y - Y0, max(0, int(x) - X0 - 1):int(x) - X0 + 2] = col
    panels.append(("8 ЛИНИИ+ТРАССА", lt))

    font = _font(20)
    ph = max(1, HH * panel_w // WW)
    tiles = []
    for name, img in panels:
        pil = Image.fromarray(img).resize((panel_w, ph), Image.LANCZOS)
        canvas = Image.new("RGB", (panel_w, ph + 28), (255, 255, 255))
        canvas.paste(pil, (0, 28))
        ImageDraw.Draw(canvas).text((3, 5), name, fill=(0, 0, 0), font=font)
        tiles.append(canvas)
    gap = 8
    mont = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), tiles[0].height),
                     (240, 240, 240))
    x = 0
    for t in tiles:
        mont.paste(t, (x, 0)); x += t.width + gap
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pth = out / f"{stem}_stages.png"
    mont.save(pth)
    return str(pth), {"window_y": [Y0, Y1],
                      "depth": [fr.depth_of(Y0), fr.depth_of(Y1)],
                      "prob": prob is not None, "n_panels": len(panels)}
