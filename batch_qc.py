"""
batch_qc.py - QC-прогон Блока 2 по файлам с эталонными LAS.

Для каждого <stem>.las в <project>\\las находит <stem>.jpg|.tif в <project>\\img,
запускает analyze_log_image (свежий расчет) и compare_trace_las.
Сводка пишется в F:\\nds\\logs\\analysis_debug\\batch_qc_summary.json и stdout.

Использование:
    python batch_qc.py F:\\nds\\projects\\Semeguniv_020
    python batch_qc.py F:\\nds\\projects\\Semeguniv_020 --limit 3 --no-lm-studio
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from analyze_log_image import analyze_log_image, save_debug_overlay
from compare_trace_las import compare_curve, parse_las, plot_comparison, DEBUG_DIR

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp")


def find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = img_dir / (stem + ext)
        if candidate.is_file():
            return candidate
    return None


def qc_one(image_path: Path, las_path: Path, use_lm: bool) -> dict:
    t0 = time.time()
    entry = {"image": image_path.name, "las": las_path.name}
    try:
        result = analyze_log_image(str(image_path), use_lm_studio=use_lm)
    except Exception as e:
        entry.update({"status": "analyze_failed", "error": str(e)})
        return entry

    # Кеш analysis.json рядом с изображением — как у analyze_log_image_cached.
    cache_path = image_path.with_name(image_path.stem + "_analysis.json")
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        overlay = save_debug_overlay(str(image_path), result)
    except Exception as e:
        overlay = f"failed: {e}"

    cal = result.get("pixel_calibration", {})
    entry.update(
        {
            "status": "ok",
            "elapsed_s": round(time.time() - t0, 1),
            "cal_status": cal.get("status"),
            "cal_labels_used": cal.get("num_labels_used"),
            "cal_rmse_m": cal.get("fit_rmse_m"),
            "meters_per_pixel": cal.get("meters_per_pixel"),
            "label_step_m": cal.get("label_step_m"),
            "depth_axis_suggestion": result.get("depth_axis_suggestion"),
            "num_bands": len(result.get("analysis_layers", {}).get("traces", {}).get("bands_px", [])),
            "curves": [
                {
                    "name": c.get("name"),
                    "scale": [c.get("scale_min"), c.get("scale_max")],
                    "unit": c.get("unit"),
                    "scale_source": c.get("scale_source"),
                    "depth_start": c.get("depth_start"),
                    "depth_end": c.get("depth_end"),
                }
                for c in result.get("curves", [])
            ],
            "overlay": overlay,
        }
    )

    try:
        las = parse_las(str(las_path))
        metrics = [compare_curve(curve, las) for curve in result.get("curves", [])]
        plot_path = str(Path(DEBUG_DIR) / (image_path.stem + "_compare.png"))
        plot_comparison(metrics, plot_path)
        for m in metrics:
            m.pop("_plot_data", None)
        entry["compare"] = metrics
    except Exception as e:
        entry["compare"] = [{"status": "compare_failed", "error": str(e)}]
    return entry


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="QC-прогон Блока 2 по эталонным LAS")
    parser.add_argument("project", help="Папка проекта со скважиной (img/ las/ wlg/)")
    parser.add_argument("--limit", type=int, default=0, help="Максимум файлов")
    parser.add_argument("--no-lm-studio", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[], help="Stem-ы, которые пропустить")
    args = parser.parse_args()

    project = Path(args.project)
    las_files = sorted((project / "las").glob("*.las"))
    if args.limit:
        las_files = las_files[: args.limit]

    summary = []
    for i, las_path in enumerate(las_files, 1):
        stem = las_path.stem
        if stem in args.skip:
            continue
        image_path = find_image(project / "img", stem)
        if image_path is None:
            # Варианты вида *_D2_1.las — пробуем без суффикса _1.
            if stem.endswith("_1"):
                image_path = find_image(project / "img", stem[:-2])
            if image_path is None:
                summary.append({"las": las_path.name, "status": "image_not_found"})
                continue
        print(f"[{i}/{len(las_files)}] {stem}", flush=True)
        entry = qc_one(image_path, las_path, use_lm=not args.no_lm_studio)
        summary.append(entry)
        ok_curves = [m for m in entry.get("compare", []) if m.get("status") == "ok"]
        for m in ok_curves:
            print(
                f"    {m['curve']}: shift={m.get('best_shift_m')}m "
                f"corr={m.get('corr_shifted')} rmse={m.get('rmse')}",
                flush=True,
            )

    out_path = Path(DEBUG_DIR) / "batch_qc_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
