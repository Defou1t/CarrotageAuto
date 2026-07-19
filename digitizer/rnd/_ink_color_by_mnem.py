"""КАКОГО ЦВЕТА КРИВАЯ НА САМОМ ДЕЛЕ — замер по чернилам под ЭКСПЕРТНОЙ трассой.

Повод (Эдуард, 19.07): цветов в словаре мнемоник НЕТ — их задаёт эксперт при подготовке рамки.
Но в `mnemonics.json` цвет проставлен у 6 мнемоник (GZ green, PZ/DS/DN black, SP red, SP2 orange),
и `emit._map_lines_to_slots` использует его как СТРОГИЙ фильтр:
    if s["color"] is not None and s["color"] != L.color: continue
⇒ если на листе GZ нарисована чёрным, слот GZ не может быть заполнен ВООБЩЕ.
Проверяем, насколько цвет вообще предсказуем по мнемонике.

Метод: берём точки экспертной полилинии (вершины, без интерполяции — они гарантированно на
штрихе), смотрим RGB в окне ±2px, классифицируем цвет теми же правилами, что и пайплайн
(imaging.color_channels / dark_mask), и считаем распределение по мнемоникам.

  python _ink_color_by_mnem.py [N листов]
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from extract_nlgx import extract, NULL
from dataset_build import find_image
from auto import imaging as im, meta as M
from auto.config import Config

MN = r"F:\nds\Auto\mnemonics.json"
ARCHIVE = Path(r"F:\nds\projects\Archive")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
p = Config().cv

by_mnem = defaultdict(Counter)
n_files = 0
for wlg in sorted(ARCHIVE.glob("*/wlg")):
    if n_files >= N:
        break
    for f in sorted(wlg.glob("*.nlgx")):
        if "_auto" in f.stem or n_files >= N:
            continue
        img = find_image(f)
        if not img:
            continue
        try:
            mo = extract(str(f))
            rgb = im.load_rgb(str(img))
        except Exception:
            continue
        H, W = rgb.shape[:2]
        n_files += 1
        # ⚠ НЕ строим полноразмерные маски: color_channels/dark_mask на скане 20-70 Мпикс делают
        # замер неподъёмным (45 листов не досчитались за 50 мин). Классифицируем ТОЧЕЧНО, теми же
        # правилами, что imaging.color_channels: sat=max-min, red R-G>rg_thr & R>=B (оранж по G-B),
        # green G-R>rg_thr & G>B, blue B-R>8; иначе тёмное = чернило без цвета.
        for c in mo["curves"]:
            root = M.curve_info(c["name"], MN)["root"]
            if M.mnem_root(c["name"]) == "DA":
                continue
            pts = [(c["top_y"] + i, x) for i, x in enumerate(c["xs"]) if x != NULL]
            if len(pts) < 50:
                continue
            step = max(1, len(pts) // 200)
            cnt = Counter()
            for y, x in pts[::step]:
                if not (0 <= y < H and 0 <= x < W):
                    continue
                win = rgb[max(0, y-2):min(H, y+3), max(0, x-2):min(W, x+3)].reshape(-1, 3).astype(int)
                if not len(win):
                    continue
                R, G, B = win[:, 0], win[:, 1], win[:, 2]
                mx, mn_ = win.max(1), win.min(1)
                colored = (mx - mn_ >= p.sat_thr) & (mx < 245)
                red_base = colored & (R - G > p.rg_thr) & (R >= B)
                orange = red_base & (G - B > p.orange_gb)
                red = red_base & ~orange
                green = colored & (G - R > p.rg_thr) & (G > B)
                blue = colored & (B - R > 8) & (B >= G - 4) & ~green
                dark = (mx < p.dark_v) & ~(red | orange | green | blue)
                for nm, msk in (("red", red), ("orange", orange), ("green", green),
                                ("blue", blue), ("black", dark)):
                    if msk.any():
                        cnt[nm] += 1
                        break
                else:
                    cnt["нет чернил"] += 1
            if cnt:
                top, k = cnt.most_common(1)[0]
                by_mnem[root][top] += 1

print(f"листов: {n_files}")
print(f"\n{'мнемоника':<10} {'кривых':>7}  преобладающий цвет чернил (доля листов)")
d = M.load_mnemonics(MN)["curves"]
for root, c in sorted(by_mnem.items(), key=lambda kv: -sum(kv[1].values()))[:18]:
    tot = sum(c.values())
    dist = ", ".join(f"{k}×{v}" for k, v in c.most_common(4))
    claim = (d.get(root) or {}).get("color")
    top = c.most_common(1)[0][0]
    flag = ""
    if claim:
        flag = f"   ⚠ словарь: {claim}" + ("" if claim == top else "  ← РАСХОДИТСЯ")
    print(f"{root:<10} {tot:>7}  {dist}{flag}")
