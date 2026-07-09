"""
b4_digitize.py — B4: оцифровка изображения в NeuraLOG после рамки.

Цепочка (предусловие: открыт nlgx с установленным Depth Axis):
  1. Scale Axis base — 2 клика эксперта (лево/право), значения вписывает код
  2. Add Backup ×5 (N раз) — код-онли; правка значений 3-го+ масштаба
  3. Add New кривая (Other... → аббревиатура) — код-онли
  4. Трейсинг (BTN_FORWARD → монитор глубины → stop)
  5. Сохранение nlgx + экспорт LAS (CMD 32837)

Использование:
  python b4_digitize.py <image.jpg> [опции]
    --curve MBK            аббревиатура кривой (дефолт: из имени файла)
    --scale-left 0         значение шкалы на ЛЕВОМ краю трека
    --scale-right 5        значение шкалы на ПРАВОМ краю трека
    --backups 2            сколько раз Add Backup ×5 (0 = без перевыносов)
    --backup-type 5x       тип перевыноса (5x/2x/10x/100x/Backup Right...)
    --skip-scale           Scale Axis уже создан — начать с Add Backup
    --skip-curve           кривая уже создана — начать с трейсинга
    --no-trace             не трейсить (только калибровка)
    --no-export            не экспортировать LAS

Пример (Yatskivska MBK):
  python b4_digitize.py "F:\\nds\\projects\\Yatskivska_001\\img\\Yatskivska_1_MBK_3180_3580_200_D1.jpg" ^
      --curve MBK --scale-left 0 --scale-right 5 --backups 2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\nds\Auto")
from neuralog_win32 import NeuraLog

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("b4_digitize")


def load_analysis(image_path: str) -> dict:
    ana = Path(image_path).with_name(Path(image_path).stem + "_analysis.json")
    return json.loads(ana.read_text(encoding="utf-8"))


def plan(image_path: str, args) -> dict:
    r = load_analysis(image_path)
    cal = r["pixel_calibration"]
    mpp, b = cal["meters_per_pixel"], cal["depth_at_y0"]
    cur = r["curves"][0]
    band = cur.get("track_band_px") or [16, 745]
    top_d = min(r["depth_from"], r["depth_to"])
    bot_d = max(r["depth_from"], r["depth_to"])
    # y для кликов шкалы: чуть НИЖЕ верхней грани рамки (внутри сетки,
    # «above or below depth axis»), на чистой клетке
    scale_y = round((top_d + 6 - b) / mpp)
    curve = args.curve
    if not curve:
        m = re.match(r"^[^_]+_\d+_([A-Za-z0-9+]+)_", Path(image_path).name)
        curve = (m.group(1).split("+")[0] if m else "CURVE")
    return {
        "curve": curve,
        "left_x": int(band[0]), "right_x": int(band[1]),
        "scale_y": int(scale_y),
        "left_val": args.scale_left if args.scale_left is not None
                    else cur.get("scale_min", 0.0),
        "right_val": args.scale_right if args.scale_right is not None
                     else cur.get("scale_max", 5.0),
        "top_depth": top_d, "bottom_depth": bot_d,
        "image_h": r["image"]["height"],
    }


def fix_backup_values(nl: NeuraLog, base_left: float, base_right: float,
                      factor: float, n_backups: int):
    """
    Канон §4.4: при повторном ×5 NeuraLOG НЕ перемножает значения сам
    (scale3 остаётся = scale2). Проходим по шкалам и вписываем
    base · factor^i для i-го перевыноса.
    """
    st = nl.panel_state()
    axes = st["scale_axes"]
    log.info(f"Scale Axes в панели: {axes}")
    # base = индекс 0, перевыносы 1..n
    for i in range(1, min(n_backups, len(axes) - 1) + 1):
        want_l, want_r = base_left * factor**i, base_right * factor**i
        nl.select_scale_axis(i)
        cur = nl.panel_state()
        cl, cr = cur["scale_left"], cur["scale_right"]
        log.info(f"  SA[{i}] '{axes[i]}': панель L/R = '{cl}'/'{cr}', "
                 f"нужно {want_l:g}/{want_r:g}")
        try:
            ok = (abs(float(cl) - want_l) < 1e-6 and
                  abs(float(cr) - want_r) < 1e-6)
        except ValueError:
            ok = False
        if not ok:
            nl.set_scale_values(want_l, want_r)
            after = nl.panel_state()
            log.info(f"  SA[{i}] исправлено → '{after['scale_left']}'/"
                     f"'{after['scale_right']}'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--scale-left", type=float, default=None)
    ap.add_argument("--scale-right", type=float, default=None)
    ap.add_argument("--backups", type=int, default=0)
    ap.add_argument("--backup-type", default="5x")
    ap.add_argument("--skip-scale", action="store_true")
    ap.add_argument("--skip-curve", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--las-out", default=None)
    args = ap.parse_args()

    p = plan(args.image, args)
    print("ПЛАН ОЦИФРОВКИ:")
    for k, v in p.items():
        print(f"  {k:<12s} = {v}")

    nl = NeuraLog()
    if not nl.cm.main_window:
        print("NeuraLOG не найден.")
        return 1
    if not nl._canvas_view():
        print("Нет открытого изображения/проекта в NeuraLOG.")
        return 1

    st = nl.panel_state()
    print(f"Панель: DA={st['depth_axes']} SA={st['scale_axes']} "
          f"curves={st['curves']}")
    if not st["depth_axes"]:
        print("Depth Axis НЕ установлен — сначала b4_calibrate.py")
        return 1

    # ── 1. Scale Axis base (2 клика эксперта) ────────────────────
    if not args.skip_scale and not st["scale_axes"]:
        ok = nl.calibrate_scale_assisted(
            args.image,
            left_value=p["left_val"], right_value=p["right_val"],
            left_img_x=p["left_x"], right_img_x=p["right_x"],
            img_y=p["scale_y"],
        )
        if not ok:
            print("Scale Axis: FAILED")
            return 1

    # выбрать DA и base SA (нужно для Add Backup / Add New)
    nl.select_depth_axis(0)
    nl.select_scale_axis(0)

    # ── 2. Перевыносы Add Backup ×N ──────────────────────────────
    if args.backups > 0:
        factor = float(re.sub(r"[^\d.]", "", args.backup_type) or 5)
        for i in range(args.backups):
            n_have = len(nl.panel_state()["scale_axes"]) - 1
            if n_have >= args.backups:
                break
            # Add Backup работает от ВЫБРАННОГО scale axis — выбираем последний
            nl.select_scale_axis(len(nl.panel_state()["scale_axes"]) - 1)
            if not nl.add_backup(args.backup_type):
                print(f"Add Backup #{i+1}: FAILED")
                return 1
            print(f"Add Backup #{i+1}: OK")
        fix_backup_values(nl, p["left_val"], p["right_val"],
                          factor, args.backups)

    # ── 3. Кривая ────────────────────────────────────────────────
    st = nl.panel_state()
    if not args.skip_curve and not any(
            p["curve"].lower() in c.lower() for c in st["curves"]):
        nl.select_depth_axis(0)
        nl.select_scale_axis(0)
        if not nl.add_new_curve(p["curve"]):
            print("Add New curve: FAILED")
            return 1
        print(f"Кривая {p['curve']}: OK")

    # ── 4. Трейсинг ──────────────────────────────────────────────
    if not args.no_trace:
        print(f"Подсказка NeuraLOG: «{nl.get_prompt_text()}»")
        print(f"Трейсинг до {p['bottom_depth']} м...")
        if not nl.trace_to_depth(p["bottom_depth"], timeout=600.0):
            print("Трейсинг не дошёл до низа (см. глубину выше) — "
                  "возможно нужен стартовый клик эксперта по линии.")

    # ── 5. Сохранение + экспорт ──────────────────────────────────
    nl.save()
    if not args.no_export:
        out = args.las_out or str(
            Path(r"F:\nds\output") / (Path(args.image).stem + "_auto.las"))
        if nl.export_las(out, cmd_id=32837):
            print(f"LAS: {out}")
        else:
            print("Экспорт LAS: FAILED")
            return 1

    print("ГОТОВО.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
