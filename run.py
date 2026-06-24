# -*- coding: utf-8 -*-
r"""
run.py — простой локальный запуск пайплайна оцифровки на ОДНОЙ паре скан+nlgx.

Обёртка вокруг digitizer/digitize_b3.py, которая снимает грабли локального прогона:
  • картинку ищет РЯДОМ с nlgx (по тому же имени), не доверяя пути D:\..\img внутри nlgx;
  • вывод кладёт в указанную папку (по умолчанию ./output), а не в захардкоженный F:\nds\output;
  • опционально рисует оверлей трассы поверх скана (цвет по имени кривой).

ПРИМЕРЫ:
  python run.py Yatskivska_1_STK+DS_150_2640_500_D1.nlgx
  python run.py файл.nlgx --image другой_скан.jpg --out output --overlay
  python run.py файл.nlgx --las           # достроить .las (значения через шкалы+уровни)
  python run.py файл.nlgx --check         # только дамп шаблона, без прогона

Сначала печатает СВОДКУ шаблона (есть ли размеченные кривые) — если кривых нет,
вытягивать нечего (нужна экспертная guide-трасса), прогон выдаст «кривых=0».
"""
import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIG = ROOT / "digitizer"
sys.path.insert(0, str(DIG))

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
import digitize_b3 as d3

IMG_EXT = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
# цвета оверлея по короткому имени кривой (BGR-нейтральные RGB)
OVCOL = {"PZ": (255, 0, 255), "GZ": (0, 200, 255), "SP": (255, 0, 0),
         "DS": (0, 200, 0), "MBK": (255, 140, 0), "BK": (160, 80, 255)}


def find_image_near(nlgx: Path):
    """Картинка с тем же stem рядом с nlgx (та же папка), затем ../img/."""
    stem = nlgx.stem
    for d in (nlgx.parent, nlgx.parent.parent / "img"):
        if not d.is_dir():
            continue
        for ext in IMG_EXT:
            p = d / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def summarize(m):
    da = m.get("depth_axis", {})
    real = [c for c in m["curves"]
            if c["name"].split()[0] not in ("DA1", "DA2") and
            sum(1 for x in c["xs"] if x != NULL) >= 30]
    print(f"  скважина: {m.get('well')} / {m.get('field')}")
    if da:
        print(f"  глубина:  {da.get('top_depth')}..{da.get('bottom_depth')} м "
              f"(y {da.get('top_y')}..{da.get('bottom_y')})")
    print(f"  шкал (Scale Axes): {len(m.get('scale_axes', []))}  "
          f"кривых-осей: {len(m['curves']) - len(real)}")
    if real:
        print(f"  РАЗМЕЧЕННЫХ КРИВЫХ: {len(real)}")
        for c in real:
            vp = sum(1 for x in c["xs"] if x != NULL)
            lv = sorted({l for _, _, l in c["segments"]})
            print(f"    • {c['name']:<16} guide_px={vp:<5} сегментов={len(c['segments'])} уровни={lv}")
    else:
        print("  РАЗМЕЧЕННЫХ КРИВЫХ: НЕТ — пустой шаблон, вытягивать нечего "
              "(нужна экспертная разметка кривых в NeuraLOG).")
    return real


def draw_overlay(auto_nlgx: Path, img: Path, out: Path):
    m = extract(str(auto_nlgx))
    rgb = np.asarray(Image.open(img).convert("RGB")); H, W = rgb.shape[:2]
    ov = rgb.copy(); drew = 0
    for c in m["curves"]:
        short = c["name"].split()[0]
        if short in ("DA1", "DA2"):
            continue
        col = OVCOL.get(short, (255, 140, 0)); ty = c["top_y"]
        for i, x in enumerate(c["xs"]):
            if x == NULL:
                continue
            y = ty + i
            if 0 <= y < H and 0 <= x < W:
                ov[y, max(0, x - 1):x + 2] = col; drew += 1
    if not drew:
        print("  (оверлей пропущен: трасса пуста)"); return
    da = m["depth_axis"]; ym = (da["top_y"] + da["bottom_y"]) // 2
    mid = out / f"{auto_nlgx.stem}_overlay_mid.png"
    full = out / f"{auto_nlgx.stem}_overlay_full.png"
    Image.fromarray(ov[max(0, ym - 400):ym + 400]).save(mid)
    Image.fromarray(ov).resize((max(1, W // 6), max(1, H // 6)), Image.LANCZOS).save(full)
    print(f"  оверлей -> {mid.name} / {full.name}")


def main():
    ap = argparse.ArgumentParser(description="Запуск оцифровки на одной паре скан+nlgx")
    ap.add_argument("nlgx", help="путь к .nlgx (шаблону с разметкой)")
    ap.add_argument("--image", help="скан явно (иначе ищется рядом по имени)")
    ap.add_argument("--out", default=str(ROOT / "output"), help="папка вывода (по умолч. ./output)")
    ap.add_argument("--las", action="store_true", help="достроить .las")
    ap.add_argument("--regrid", action="store_true", help="пересобрать Depth Grid (нативный шаг)")
    ap.add_argument("--overlay", action="store_true", help="нарисовать оверлей трассы на скане")
    ap.add_argument("--check", action="store_true", help="только проверить шаблон, без прогона")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    nlgx = Path(a.nlgx)
    if not nlgx.is_file():
        print(f"нет файла: {nlgx}"); return 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== ШАБЛОН: {nlgx.name} ===")
    m = extract(str(nlgx))
    real = summarize(m)

    if a.check:
        return 0
    if not real:
        print("\nПрогон не нужен — в шаблоне нет размеченных кривых.")
        return 0

    img = Path(a.image) if a.image else find_image_near(nlgx)
    if not img or not Path(img).is_file():
        print(f"\nНЕ НАЙДЕН СКАН рядом с {nlgx.name}. Положи картинку с тем же именем "
              f"или передай --image путь.jpg")
        return 1
    print(f"\n=== ПРОГОН (скан: {Path(img).name}) ===")
    r = d3.digitize_one(str(nlgx), str(img), str(out), a.regrid, False, a.las)
    if r.get("error"):
        print(f"ОШИБКА: {r['error']}"); return 1
    print(f"  -> {r['dst']}")
    print(f"  кривых={r['curves']} вписано={r['written']} покрытие≈{r['cover_pct']}% "
          f"own≈{r['own_px']}px verified={r['verified']}")
    if r.get("dst") and a.overlay:
        draw_overlay(Path(r["dst"]), Path(img), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
