"""
b4_calibrate.py - B4 human-in-the-loop установка Depth Axis в NeuraLOG.

Код берёт калибровку Блока 2 (пиксели граней рамки), наводит курсор в точку,
эксперт кликает ЛКМ по канве, код ловит диалог и вписывает глубину сам.

ПРЕДУСЛОВИЕ: в NeuraLOG ОТКРЫТО изображение этой скважины (File>Open).

Использование:
    python b4_calibrate.py "F:\\nds\\projects\\<well>\\img\\<file>.jpg"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"F:\nds\Auto")
from neuralog_win32 import NeuraLog


def _snap4(depth: float) -> float:
    """Снап глубины к жирной грани сетки (4 м при масштабе 1:200)."""
    return round(depth / 4.0) * 4.0


def plan_from_analysis(image_path: str) -> dict:
    ana = Path(image_path).with_name(Path(image_path).stem + "_analysis.json")
    r = json.loads(ana.read_text(encoding="utf-8"))
    cal = r["pixel_calibration"]
    if cal.get("depth_at_y0") is None:
        raise RuntimeError("Нет абсолютной калибровки глубины (depth_at_y0=None). "
                           "Прогоните analyze_log_image с LM Studio.")
    mpp, b = cal["meters_per_pixel"], cal["depth_at_y0"]
    # Грани рамки: интервал из имени/калибровки, снапнутый к сетке 4 м.
    top_d = _snap4(min(r["depth_from"], r["depth_to"]))
    bot_d = _snap4(max(r["depth_from"], r["depth_to"]))
    top_y = round((top_d - b) / mpp)
    bot_y = round((bot_d - b) / mpp)
    bands = (r.get("analysis_layers", {}).get("traces", {}) or {}).get("bands_px") or [[30, 100]]
    track_x = int(bands[0][0]) + 5
    return {
        "top_depth": top_d, "top_img_y": top_y,
        "bottom_depth": bot_d, "bottom_img_y": bot_y,
        "track_img_x": track_x, "units": "Meters",
        "image_h": r["image"]["height"],
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    image_path = sys.argv[1]
    plan = plan_from_analysis(image_path)
    print("План рамки:")
    print(f"  Top    = {plan['top_depth']} м  (img y {plan['top_img_y']})")
    print(f"  Bottom = {plan['bottom_depth']} м  (img y {plan['bottom_img_y']})")
    print(f"  track_x = {plan['track_img_x']}  | image_h = {plan['image_h']}")
    print()

    nl = NeuraLog()
    if not nl.cm.main_window:
        print("NeuraLOG не найден.")
        return 1
    if not nl._canvas_view():
        print("В NeuraLOG НЕ открыто изображение (пустой MDI). "
              "Откройте File>Open>Log Image и повторите.")
        return 1

    ok = nl.calibrate_depth_assisted(
        image_path,
        top_depth=plan["top_depth"], top_img_y=plan["top_img_y"],
        bottom_depth=plan["bottom_depth"], bottom_img_y=plan["bottom_img_y"],
        track_img_x=plan["track_img_x"], units=plan["units"],
    )
    print("РЕЗУЛЬТАТ:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
