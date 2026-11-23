"""§6.386: обмен имён по цвету шапки (цвет имени — чтение шапки облачной моделью, кэш JSON {лист nlgx: {имя кривой: цвет}}).

Цвет туши кривой выдачи — та же мера, что стенд `digitizer/rnd/_color_clash.py`: по каждой 4-й строке трассы самый тёмный пиксель
в ±1 px (если темнее 200), медиана RGB при ≥ 30 таких строк; семейство оттенка при разбросе каналов ≥ 20, иначе «чёрный» (−1).
Правило: в треке (токен DA имени) пара кривых A, B, у каждой цвет туши ≠ цвету своего имени по шапке, а крест-накрест совпадает ⇒
обмен трасс между именами. Без ответа шапки для листа / имени — ничего не меняется.
"""
import json
from pathlib import Path
import numpy as np

FAM = [(345, 15), (15, 45), (45, 70), (70, 165), (165, 200), (200, 255), (255, 300), (300, 345)]
WORD = {"red": 0, "orange": 1, "brown": 1, "yellow": 2, "green": 3, "teal": 4, "blue": 5, "purple": 6, "pink": 7, "black": -1, "grey": -1}
_CACHE = {}


def _hue(rgb):
    r, g, b = (np.asarray(rgb, np.float64) / 255.0)
    mx, mn = max(r, g, b), min(r, g, b); d = (mx - mn) or 1e-9
    h = ((g - b) / d) % 6 if mx == r else ((b - r) / d + 2 if mx == g else (r - g) / d + 4)
    return (h * 60.0) % 360.0


def family(rgb):
    """медиана RGB → семейство 0…7 или −1 (чёрный / серый)"""
    if rgb is None or (max(rgb) - min(rgb)) < 20:
        return -1
    h = _hue(rgb)
    for i, (a, b) in enumerate(FAM):
        if (a <= h < b) if a < b else (h >= a or h < b):
            return i
    return 0


def ink_rgb(rgb, gray, tr):
    H, W = gray.shape
    px = []
    for y in sorted(tr)[::4]:
        x = int(round(tr[y]))
        if 0 <= y < H and 1 <= x < W - 1:
            g = gray[y, x - 1:x + 2]; j = int(np.argmin(g))
            if g[j] < 200:
                px.append(rgb[y, x - 1 + j])
    return np.median(np.array(px, float), 0).tolist() if len(px) >= 30 else None


def colors_for(spec, sheet_name):
    if spec not in _CACHE:
        p = Path(spec)
        if not p.is_file():
            p = Path(__file__).resolve().parent / "models" / spec
        _CACHE[spec] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return _CACHE[spec].get(sheet_name) or {}


def swaps(header, ours):
    """header {имя: цвет-слово}, ours {имя: семейство} → [(A, B)] обменов (каждое имя — не больше одного раза)"""
    H = {n: WORD[c] for n, c in header.items() if c in WORD}
    names = [n for n in ours if n in H]
    by = {}
    for n in names:
        by.setdefault((n.split() + [""])[1], []).append(n)
    out, used = [], set()
    for tk, L in by.items():
        mis = [n for n in L if ours[n] != H[n]]
        for i, a in enumerate(mis):
            for b in mis[i + 1:]:
                if a in used or b in used:
                    continue
                if ours[a] == H[b] and ours[b] == H[a]:
                    used |= {a, b}; out.append((a, b))
    return out
