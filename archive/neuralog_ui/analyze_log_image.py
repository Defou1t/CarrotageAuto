"""
analyze_log_image.py - локальный анализ изображений каротажа.

MVP Блока 2:
  1. deterministic fallback из имени файла;
  2. базовая геометрия через OpenCV, если установлен;
  3. локальный OCR, если установлен;
  4. LM Studio через OpenAI-compatible endpoint, если запущен;
  5. pixel_calibration: метки глубины (y_px -> depth) для calibrate_depth() Блока 3;
  6. шкалы кривых из OCR-блоков (scale_min/scale_max + пиксельные тики x -> value);
  7. собственная трассировка кривых (полилиния + сэмплы depth/value);
  8. save_debug_overlay(): визуальная проверка найденного.

Публичные функции сохранены:
    analyze_log_image(path)
    analyze_log_image_cached(path)
    analyze_batch(paths)
    save_debug_overlay(path, result)

Командная строка:
    python analyze_log_image.py F:\\nds\\projects\\Well\\img\\file.tif
    python analyze_log_image.py file.tif --no-lm-studio
    python analyze_log_image.py file.tif --provider lm_studio --model "model-id"
    python analyze_log_image.py file.tif --debug-overlay
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}

LM_STUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"

DEBUG_OVERLAY_DIR = r"F:\nds\logs\analysis_debug"

# Типовые шаги меток глубины на советских каротажках, метры.
DEPTH_LABEL_STEPS_M = [1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0]

# Порог "тёмных чернил" (кривые и рукописный текст темнее сетки миллиметровки).
INK_GRAY_THRESHOLD = 110

UNIT_TOKENS = {
    "OHMM": "Ohmm", "OMM": "Ohmm", "ОММ": "Ohmm", "ОМ": "Ohmm",
    "MV": "mV", "МВ": "mV",
    "API": "API", "GAPI": "API",
    "%": "%",
    "МКР/Ч": "uR/h", "MKR/H": "uR/h",
    "УС/М": "us/m", "US/M": "us/m", "МКС/М": "us/m",
    "MM": "mm", "ММ": "mm",
}

CURVE_DEFAULTS = {
    "BK": {"unit": "Ohmm", "scale_min": 0.0, "scale_max": 20.0, "scale_type": "linear"},
    "MBK": {"unit": "Ohmm", "scale_min": 0.0, "scale_max": 5.0, "scale_type": "linear"},
    "RT": {"unit": "Ohmm", "scale_min": 0.2, "scale_max": 200.0, "scale_type": "log"},
    "R": {"unit": "Ohmm", "scale_min": 0.2, "scale_max": 200.0, "scale_type": "log"},
    "GK": {"unit": "API", "scale_min": 0.0, "scale_max": 150.0, "scale_type": "linear"},
    "GR": {"unit": "API", "scale_min": 0.0, "scale_max": 150.0, "scale_type": "linear"},
    "SP": {"unit": "mV", "scale_min": -50.0, "scale_max": 50.0, "scale_type": "linear"},
    "PS": {"unit": "mV", "scale_min": -50.0, "scale_max": 50.0, "scale_type": "linear"},
    "AK": {"unit": "us/m", "scale_min": 0.0, "scale_max": 0.0, "scale_type": "linear"},
    "CALI": {"unit": "mm", "scale_min": 0.0, "scale_max": 500.0, "scale_type": "linear"},
}

LOCAL_VLM_PROMPT = """\
You are analyzing a scanned well log / borehole log image.

Return ONLY valid JSON. No markdown fences, no comments, no prose.

Schema:
{
  "curves": [
    {
      "name": "BK",
      "unit": "Ohmm",
      "color": "black",
      "scale_min": 0.0,
      "scale_max": 20.0,
      "scale_type": "linear",
      "confidence": 0.0
    }
  ],
  "depth_from": 3080.0,
  "depth_to": 3520.0,
  "depth_unit": "m",
  "num_tracks": 1
}

Rules:
- Include only information visible in the image or very strongly implied by headers.
- depth_from must be less than depth_to.
- Normalize units: Ohm, ohm.m, Ohm*m -> Ohmm; millivolt -> mV.
- If unsure, use confidence below 0.5.
"""

ANTHROPIC_LEGACY_PROMPT = LOCAL_VLM_PROMPT


def analyze_log_image(
    image_path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: str = "local",
    fallback_from_filename: bool = True,
    max_tokens: int = 1200,
    use_lm_studio: bool = True,
    lm_studio_url: Optional[str] = None,
    lm_studio_timeout: float = 20.0,
    ocr_engine: str = "auto",
    extract_traces: bool = True,
    debug_overlay: bool = False,
    debug_dir: Optional[str] = None,
) -> dict:
    """
    Анализировать изображение каротажа.

    provider:
      - local      : filename + OpenCV + OCR + optional LM Studio (default)
      - lm_studio  : local pipeline with mandatory LM Studio attempt
      - anthropic  : legacy cloud path, only if explicitly selected
    """
    image_path = str(Path(image_path))
    _validate_image_path(image_path)

    provider = (provider or "local").lower()
    if provider == "anthropic":
        return _analyze_anthropic_legacy(
            image_path=image_path,
            api_key=api_key,
            model=model or "claude-opus-4-5",
            max_tokens=max_tokens,
            fallback_from_filename=fallback_from_filename,
        )

    if provider not in {"local", "lm_studio"}:
        raise ValueError("provider должен быть local, lm_studio или anthropic")

    result = _empty_result(image_path)
    result["source"] = "local"
    result["analysis_layers"] = {}

    filename_data = _parse_filename_metadata(image_path)
    result["analysis_layers"]["filename"] = filename_data
    _merge_filename_layer(result, filename_data)

    image_info = _read_image_info(image_path)
    result["image"].update(image_info)

    cv_image = None
    try:
        cv_image = _cv2_read(image_path)
    except Exception as e:
        log.debug("OpenCV unavailable or image unreadable: %s", e)

    layout = _detect_layout_opencv(image_path, image=cv_image)
    result["layout"].update(layout)
    if layout.get("num_tracks_estimate") and not result.get("num_tracks"):
        result["num_tracks"] = int(layout["num_tracks_estimate"])

    ocr_data = _run_ocr(image_path, engine=ocr_engine)
    result["ocr"] = ocr_data
    result["analysis_layers"]["ocr"] = {
        "source": ocr_data.get("source"),
        "status": ocr_data.get("status"),
        "error": ocr_data.get("error"),
    }
    _merge_ocr_layer(result, ocr_data)

    should_use_lm = use_lm_studio or provider == "lm_studio"
    lm_base_url = lm_studio_url or os.environ.get("LM_STUDIO_BASE_URL") or LM_STUDIO_DEFAULT_BASE_URL
    lm_model = model or os.environ.get("LM_STUDIO_MODEL")

    calibration = _calibrate_depth_axis(
        result,
        cv_image=cv_image,
        use_lm_studio=should_use_lm,
        lm_base_url=lm_base_url,
        lm_model=lm_model,
        lm_timeout=lm_studio_timeout,
    )
    result["pixel_calibration"] = calibration
    result["analysis_layers"]["pixel_calibration"] = {
        "status": calibration.get("status"),
        "method": calibration.get("method"),
        "num_labels": len(calibration.get("depth_labels", [])),
        "num_labels_used": calibration.get("num_labels_used", 0),
        "error": calibration.get("error", ""),
    }
    _merge_calibration_layer(result, calibration)

    scales = _extract_scales_from_ocr(result)
    result["analysis_layers"]["ocr_scales"] = {
        "num_scale_lines": len(scales),
        "scales": scales,
    }

    if should_use_lm:
        lm_data = _analyze_with_lm_studio(
            image_path=image_path,
            base_url=lm_base_url,
            model=lm_model,
            max_tokens=max_tokens,
            timeout=lm_studio_timeout,
        )
        result["analysis_layers"]["lm_studio"] = _compact_lm_layer(lm_data)
        if lm_data.get("status") == "ok":
            _merge_model_layer(result, lm_data.get("result", {}))

    if extract_traces:
        traces_layer = _extract_curve_traces(result, cv_image=cv_image)
        result["analysis_layers"]["traces"] = traces_layer

        # Рукописные шкалы OCR не берет — дочитываем шапки треков через VLM
        # и пересчитываем значения сэмплов с уточненными шкалами.
        if should_use_lm and cv_image is not None:
            scale_layer = _read_scales_with_lm_studio(
                cv_image,
                result,
                base_url=lm_base_url,
                model=lm_model,
                timeout=lm_studio_timeout,
            )
            result["analysis_layers"]["lm_scales"] = scale_layer
            if scale_layer.get("status") == "ok":
                _recompute_trace_samples(result)

    result["confidence"] = _estimate_result_confidence(result)

    if debug_overlay:
        try:
            overlay_path = save_debug_overlay(image_path, result, out_dir=debug_dir)
            result["analysis_layers"]["debug_overlay"] = {"path": overlay_path}
        except Exception as e:
            log.warning("debug overlay failed: %s", e)
            result["analysis_layers"]["debug_overlay"] = {"path": "", "error": str(e)}

    return result


def analyze_log_image_cached(
    image_path: str,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
    **kwargs,
) -> dict:
    """Анализ с кешированием JSON рядом с изображением или в cache_dir."""
    path = Path(image_path)
    cache_dir = cache_dir or str(path.parent)
    cache_path = Path(cache_dir) / (path.stem + "_analysis.json")

    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            log.info("analyze_log_image: cache %s", cache_path.name)
            return data
        except Exception:
            log.warning("Кеш поврежден, пересчитываю: %s", cache_path)

    result = analyze_log_image(str(path), **kwargs)

    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.debug("Кеш сохранен: %s", cache_path)
    except Exception as e:
        log.warning("Не удалось сохранить кеш: %s", e)

    return result


def analyze_batch(image_paths: list, cache: bool = True, **kwargs) -> list:
    """Анализ списка изображений."""
    fn = analyze_log_image_cached if cache else analyze_log_image
    results = []
    total = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        log.info("[%d/%d] %s", i, total, Path(path).name)
        try:
            results.append(fn(str(path), **kwargs))
        except Exception as e:
            log.error("Ошибка анализа %s: %s", path, e)
            results.append(_empty_result(str(path)))
    return results


def _validate_image_path(image_path: str) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    ext = Path(image_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат '{ext}'. "
            f"Допустимые: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _empty_result(image_path: str) -> dict:
    return {
        "curves": [],
        "depth_from": 0.0,
        "depth_to": 0.0,
        "depth_unit": "m",
        "num_tracks": 0,
        "image": {
            "path": str(image_path),
            "width": 0,
            "height": 0,
            "dpi": None,
        },
        "layout": {
            "track_bounds_px": [],
            "horizontal_grid_lines_px": [],
            "vertical_grid_lines_px": [],
            "num_tracks_estimate": 0,
            "status": "not_run",
        },
        "ocr": {
            "source": "none",
            "status": "not_run",
            "text": "",
            "blocks": [],
        },
        "pixel_calibration": {
            "status": "not_run",
            "method": "",
            "depth_labels": [],
            "pixel_top": None,
            "depth_top": None,
            "pixel_bottom": None,
            "depth_bottom": None,
            "meters_per_pixel": None,
            "depth_at_y0": None,
            "expected_meters_per_pixel": None,
            "label_step_m": None,
            "fit_rmse_m": None,
            "num_labels_used": 0,
            "confidence": 0.0,
            "error": "",
        },
        "depth_scale_hint": None,
        "depth_axis_suggestion": None,
        "source": "local",
        "confidence": 0.0,
        "analysis_layers": {},
    }


def _parse_filename_metadata(image_path: str) -> dict:
    """Извлечь подсказки из имени: Well_CURVE_3080_3520_200_D1.tif.

    Четвертое число (200/500/...) — вертикальный масштаб глубины 1:N,
    а не DPI: проверено на Semeguniv_020 (JPEG dpi=150, шаг меток 10 м
    занимает ~295 px, что соответствует 1:200 при 150 dpi).
    """
    stem = Path(image_path).stem
    data = {
        "source": "filename",
        "status": "unknown",
        "stem": stem,
        "well_name": "",
        "curve_hints": [],
        "depth_from": 0.0,
        "depth_to": 0.0,
        "dpi": None,
        "depth_scale": None,
        "part": "",
        "confidence": 0.0,
    }

    pattern = re.compile(
        r"^(?P<well>.+?)_"
        r"(?P<curves>[^_]+)_"
        r"(?P<from>\d{2,5}(?:[.,]\d+)?)_"
        r"(?P<to>\d{2,5}(?:[.,]\d+)?)"
        r"(?:_(?P<dpi>\d{2,4}))?"
        r"(?:_(?P<part>D\d+|P\d+|Part\d+))?$",
        re.IGNORECASE,
    )
    match = pattern.match(stem)
    if not match:
        return data

    depth_from = _to_float(match.group("from"))
    depth_to = _to_float(match.group("to"))
    if depth_from and depth_to and depth_from > depth_to:
        depth_from, depth_to = depth_to, depth_from

    curves = [
        c.strip().upper()
        for c in re.split(r"[+;,]", match.group("curves") or "")
        if c.strip()
    ]

    scale_or_dpi = int(match.group("dpi")) if match.group("dpi") else None
    data.update(
        {
            "status": "ok",
            "well_name": match.group("well"),
            "curve_hints": curves,
            "depth_from": depth_from or 0.0,
            "depth_to": depth_to or 0.0,
            "dpi": scale_or_dpi,
            "depth_scale": scale_or_dpi,
            "part": match.group("part") or "",
            "confidence": 0.65 if curves and depth_from and depth_to else 0.45,
        }
    )
    return data


def _merge_filename_layer(result: dict, data: dict) -> None:
    if data.get("status") != "ok":
        return

    if data.get("depth_from") and data.get("depth_to"):
        result["depth_from"] = float(data["depth_from"])
        result["depth_to"] = float(data["depth_to"])

    if data.get("dpi") and not result["image"].get("dpi"):
        result["image"]["dpi"] = int(data["dpi"])

    if data.get("depth_scale"):
        result["depth_scale_hint"] = int(data["depth_scale"])

    for curve_name in data.get("curve_hints", []):
        result["curves"].append(_curve_from_name(curve_name, source="filename", confidence=0.55))

    if result["curves"]:
        result["num_tracks"] = max(result.get("num_tracks", 0), 1)


def _curve_from_name(name: str, source: str, confidence: float) -> dict:
    norm_name = _normalize_curve_name(name)
    defaults = CURVE_DEFAULTS.get(norm_name, {})
    return {
        "name": norm_name,
        "unit": defaults.get("unit", ""),
        "color": "",
        "scale_min": defaults.get("scale_min", 0.0),
        "scale_max": defaults.get("scale_max", 0.0),
        "scale_type": defaults.get("scale_type", "linear"),
        "source": source,
        "confidence": confidence,
    }


def _read_image_info(image_path: str) -> dict:
    info = {"width": 0, "height": 0, "dpi": None}
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            info["width"], info["height"] = img.size
            dpi = img.info.get("dpi")
            if isinstance(dpi, tuple) and dpi:
                info["dpi"] = int(round(float(dpi[0])))
            elif isinstance(dpi, (int, float)):
                info["dpi"] = int(round(float(dpi)))
    except Exception as e:
        log.debug("PIL image info failed: %s", e)

    if info["width"] and info["height"]:
        return info

    try:
        image = _cv2_read(image_path)
        if image is not None:
            h, w = image.shape[:2]
            info["width"], info["height"] = int(w), int(h)
    except Exception as e:
        log.debug("OpenCV image info failed: %s", e)

    return info


def _detect_layout_opencv(image_path: str, image=None) -> dict:
    layout = {
        "track_bounds_px": [],
        "horizontal_grid_lines_px": [],
        "vertical_grid_lines_px": [],
        "num_tracks_estimate": 0,
        "status": "not_available",
        "error": "",
    }

    try:
        import cv2
    except ImportError:
        layout["error"] = "opencv-python is not installed"
        return layout

    if image is None:
        image = _cv2_read(image_path)
    if image is None:
        layout["error"] = "OpenCV could not read image"
        return layout

    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_len = max(80, min(w, h) // 8)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=3.14159265 / 180,
            threshold=120,
            minLineLength=min_len,
            maxLineGap=12,
        )
        if lines is None:
            layout["status"] = "ok"
            return layout

        vertical = []
        horizontal = []
        for item in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in item]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy > h * 0.35 and dx <= max(4, w * 0.015):
                vertical.append((x1 + x2) // 2)
            elif dx > w * 0.25 and dy <= max(4, h * 0.015):
                horizontal.append((y1 + y2) // 2)

        v_clusters = _cluster_positions(vertical, tolerance=max(6, w // 250))
        h_clusters = _cluster_positions(horizontal, tolerance=max(6, h // 250))
        track_bounds = _filter_track_bounds(v_clusters, width=w)

        layout.update(
            {
                "track_bounds_px": track_bounds,
                "horizontal_grid_lines_px": h_clusters,
                "vertical_grid_lines_px": v_clusters,
                "num_tracks_estimate": max(0, len(track_bounds) - 1),
                "status": "ok",
            }
        )
        return layout
    except Exception as e:
        layout["error"] = str(e)
        return layout


def _cv2_read(image_path: str):
    import cv2
    import numpy as np

    raw = np.fromfile(image_path, dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _cluster_positions(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    values = sorted(int(v) for v in values)
    clusters = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [int(round(sum(c) / len(c))) for c in clusters if len(c) >= 1]


def _filter_track_bounds(xs: list[int], width: int) -> list[int]:
    if not xs:
        return []
    # Убираем очень близкие к краям рамки страницы, оставляем возможные границы треков.
    left_margin = int(width * 0.02)
    right_margin = int(width * 0.98)
    filtered = [x for x in xs if left_margin <= x <= right_margin]
    if len(filtered) < 2:
        return filtered
    return filtered


def _run_ocr(image_path: str, engine: str = "auto") -> dict:
    engine = (engine or "auto").lower()
    data = {"source": "none", "status": "not_run", "text": "", "blocks": [], "error": ""}
    if engine in {"none", "off", "false"}:
        return data

    engines = ["pytesseract", "tesseract_cli"] if engine == "auto" else [engine]
    errors = []
    for name in engines:
        try:
            if name == "pytesseract":
                return _run_pytesseract(image_path)
            if name in {"tesseract", "tesseract_cli"}:
                return _run_tesseract_cli(image_path)
            if name == "paddleocr":
                return _run_paddleocr(image_path)
            if name == "easyocr":
                return _run_easyocr(image_path)
        except Exception as e:
            errors.append(f"{name}: {e}")

    data["status"] = "unavailable"
    data["error"] = "; ".join(errors) if errors else "no OCR engine selected"
    return data


def _run_pytesseract(image_path: str) -> dict:
    from PIL import Image
    import pytesseract

    with Image.open(image_path) as img:
        data = pytesseract.image_to_data(
            img, lang="eng+rus", output_type=pytesseract.Output.DICT
        )

    blocks = []
    words = []
    for i, raw_text in enumerate(data.get("text", [])):
        text = (raw_text or "").strip()
        conf = _to_float(data["conf"][i], -1.0)
        if not text or conf is None or conf < 0:
            continue
        words.append(text)
        blocks.append(
            {
                "bbox_px": [
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                ],
                "text": text,
                "confidence": round(conf / 100.0, 3),
            }
        )
    return {
        "source": "pytesseract",
        "status": "ok",
        "text": " ".join(words),
        "blocks": blocks,
        "error": "",
    }


def _run_tesseract_cli(image_path: str) -> dict:
    cmd = ["tesseract", image_path, "stdout", "-l", "eng+rus", "tsv"]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"exit code {completed.returncode}")

    blocks = []
    words = []
    lines = completed.stdout.splitlines()
    for line in lines[1:]:  # первая строка TSV — заголовок
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        text = cols[11].strip()
        conf = _to_float(cols[10], -1.0)
        if not text or conf is None or conf < 0:
            continue
        try:
            x, y, w, h = int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9])
        except ValueError:
            continue
        words.append(text)
        blocks.append(
            {
                "bbox_px": [x, y, w, h],
                "text": text,
                "confidence": round(conf / 100.0, 3),
            }
        )
    return {
        "source": "tesseract_cli",
        "status": "ok",
        "text": " ".join(words),
        "blocks": blocks,
        "error": "",
    }


def _run_paddleocr(image_path: str) -> dict:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    raw = ocr.ocr(image_path, cls=True)
    blocks = []
    texts = []
    for page in raw or []:
        for item in page or []:
            if len(item) < 2:
                continue
            box, rec = item[0], item[1]
            text, conf = rec[0], float(rec[1])
            texts.append(text)
            blocks.append({"bbox": box, "text": text, "confidence": conf})
    return {
        "source": "paddleocr",
        "status": "ok",
        "text": "\n".join(texts),
        "blocks": blocks,
        "error": "",
    }


def _run_easyocr(image_path: str) -> dict:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=True)
    raw = reader.readtext(image_path)
    blocks = []
    texts = []
    for box, text, conf in raw:
        texts.append(text)
        blocks.append({"bbox": box, "text": text, "confidence": float(conf)})
    return {
        "source": "easyocr",
        "status": "ok",
        "text": "\n".join(texts),
        "blocks": blocks,
        "error": "",
    }


def _merge_ocr_layer(result: dict, ocr_data: dict) -> None:
    text = ocr_data.get("text") or ""
    if not text:
        return

    depth_pair = _extract_depth_pair_from_text(text)
    if depth_pair and not (result.get("depth_from") and result.get("depth_to")):
        result["depth_from"], result["depth_to"] = depth_pair

    found_curves = _extract_curve_names_from_text(text)
    for name in found_curves:
        _upsert_curve(result, _curve_from_name(name, source="ocr", confidence=0.62))


def _extract_depth_pair_from_text(text: str) -> Optional[tuple[float, float]]:
    numbers = [_to_float(x) for x in re.findall(r"\b\d{3,5}(?:[.,]\d+)?\b", text)]
    numbers = [n for n in numbers if n is not None and 10 <= n <= 10000]
    if len(numbers) < 2:
        return None
    # Для каротажа берем самый малый и самый большой похожие на глубину.
    a, b = min(numbers), max(numbers)
    if a == b:
        return None
    return float(a), float(b)


def _extract_curve_names_from_text(text: str) -> list[str]:
    found = []
    upper = text.upper()
    for name in CURVE_DEFAULTS:
        if re.search(rf"\b{re.escape(name)}\b", upper):
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# OCR blocks: нормализация bbox
# ---------------------------------------------------------------------------


def _bbox_to_xywh(block: dict) -> Optional[tuple[int, int, int, int]]:
    """Привести bbox любого OCR-движка к (x, y, w, h) в пикселях."""
    bbox = block.get("bbox_px") or block.get("bbox")
    if not bbox:
        return None
    try:
        if isinstance(bbox[0], (list, tuple)):
            # Полигон из 4 точек (PaddleOCR / EasyOCR).
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            x, y = min(xs), min(ys)
            return int(x), int(y), int(max(xs) - x), int(max(ys) - y)
        x, y, w, h = [float(v) for v in bbox[:4]]
        return int(x), int(y), int(w), int(h)
    except (TypeError, ValueError, IndexError):
        return None


def _normalized_ocr_blocks(result: dict) -> list[dict]:
    """OCR-блоки с гарантированным bbox_px=[x,y,w,h] и центрами cx/cy."""
    normalized = []
    for block in result.get("ocr", {}).get("blocks", []) or []:
        xywh = _bbox_to_xywh(block)
        if xywh is None:
            continue
        x, y, w, h = xywh
        item = dict(block)
        item["bbox_px"] = [x, y, w, h]
        item["cx"] = x + w / 2.0
        item["cy"] = y + h / 2.0
        normalized.append(item)
    return normalized


# ---------------------------------------------------------------------------
# Pixel calibration: метки глубины (y_px -> depth) для Блока 3
# ---------------------------------------------------------------------------


def _calibrate_depth_axis(
    result: dict,
    cv_image=None,
    use_lm_studio: bool = False,
    lm_base_url: str = LM_STUDIO_DEFAULT_BASE_URL,
    lm_model: Optional[str] = None,
    lm_timeout: float = 20.0,
) -> dict:
    """
    Найти метки глубины на изображении и построить depth = a*y + b.

    Источники значений меток (по убыванию приоритета):
      1. OCR-блоки с bbox, текст которых похож на глубину;
      2. LM Studio VLM, читающий кропы рукописных меток;
      3. только геометрия: позиции блобов + ожидаемый масштаб (1:N из имени файла).
    """
    cal = dict(_empty_result("")["pixel_calibration"])
    cal["status"] = "failed"

    # Ожидаемый масштаб: метров на пиксель из DPI и масштаба 1:N.
    dpi = result.get("image", {}).get("dpi")
    scale_hint = result.get("depth_scale_hint")
    if dpi and scale_hint:
        cal["expected_meters_per_pixel"] = round(scale_hint * 25.4 / (dpi * 1000.0), 6)

    depth_lo, depth_hi = _plausible_depth_window(result)

    # --- кандидаты из OCR-блоков ---
    labels: list[dict] = []
    for block in _normalized_ocr_blocks(result):
        value = _parse_depth_label_text(block.get("text", ""), depth_lo, depth_hi)
        if value is None:
            continue
        x, y, w, h = block["bbox_px"]
        labels.append(
            {
                "x_px": x, "y_px": y, "w_px": w, "h_px": h,
                "cx": block["cx"], "cy": block["cy"],
                "depth": value, "text": block.get("text", ""),
                "source": "ocr",
                "confidence": float(block.get("confidence") or 0.5),
                "used": False,
            }
        )

    # --- кандидаты из CV-блобов (рукописные метки OCR обычно не берет) ---
    blob_error = ""
    if cv_image is not None:
        try:
            blobs = _detect_depth_label_blobs(cv_image)
        except Exception as e:
            blobs, blob_error = [], str(e)
        for bx, by, bw, bh in blobs:
            cy = by + bh / 2.0
            # Пропускаем блобы, уже накрытые OCR-метками.
            if any(abs(lab["cy"] - cy) < max(20, bh) for lab in labels):
                continue
            labels.append(
                {
                    "x_px": int(bx), "y_px": int(by), "w_px": int(bw), "h_px": int(bh),
                    "cx": bx + bw / 2.0, "cy": cy,
                    "depth": None, "text": "",
                    "source": "cv",
                    "confidence": 0.4,
                    "used": False,
                }
            )

    labels.sort(key=lambda lab: lab["cy"])
    cal["depth_labels"] = labels
    if not labels:
        cal["error"] = blob_error or "no depth label candidates (OCR blocks empty, CV blobs not found)"
        return cal

    # --- дочитать нечитанные блобы через LM Studio ---
    unread = [lab for lab in labels if lab["depth"] is None]
    if unread and use_lm_studio and cv_image is not None:
        picked = _sample_evenly(unread, 12)
        # Чанками по 4 кропа: один большой запрос с reasoning не укладывается
        # в таймаут, и тогда теряются ВСЕ значения. Неудачный чанк — 1 ретрай.
        for chunk_start in range(0, len(picked), 4):
            chunk = picked[chunk_start: chunk_start + 4]
            for _attempt in range(2):
                values = _read_label_crops_lm_studio(
                    cv_image, chunk, base_url=lm_base_url, model=lm_model, timeout=lm_timeout
                )
                if any(v is not None for v in values):
                    break
            for lab, value in zip(chunk, values):
                checked = _parse_depth_label_text(value, depth_lo, depth_hi)
                if checked is not None:
                    lab["depth"] = checked
                    lab["text"] = str(value)
                    lab["source"] = "lm_studio"
                    lab["confidence"] = 0.6

    read = [lab for lab in labels if lab["depth"] is not None]

    # --- шаг между соседними блобами по геометрии ---
    spacing_px = _median_label_spacing([lab["cy"] for lab in labels])

    if len(read) >= 2:
        pairs = [(lab["cy"], lab["depth"]) for lab in read]
        fit = _robust_linear_fit(pairs)
        if fit is not None:
            slope, intercept, inlier_idx, rmse = fit
            for i, lab in enumerate(read):
                lab["used"] = i in inlier_idx
            used = [read[i] for i in inlier_idx]
            cal.update(
                {
                    "status": "ok",
                    "method": "+".join(sorted({lab["source"] for lab in used})),
                    "meters_per_pixel": round(slope, 6),
                    "depth_at_y0": round(intercept, 3),
                    "fit_rmse_m": round(rmse, 3),
                    "num_labels_used": len(used),
                }
            )
            y_top = min(lab["cy"] for lab in used)
            y_bot = max(lab["cy"] for lab in used)
            cal["pixel_top"] = int(round(y_top))
            cal["depth_top"] = round(slope * y_top + intercept, 2)
            cal["pixel_bottom"] = int(round(y_bot))
            cal["depth_bottom"] = round(slope * y_bot + intercept, 2)

            # Шаг меток — по геометрии всех блобов, а не по выборке прочитанных
            # (на чтение отправляется лишь каждая n-я метка).
            if spacing_px and slope > 0:
                cal["label_step_m"] = _snap_label_step(spacing_px * slope)
            if not cal.get("label_step_m") and len(used) >= 2:
                cal["label_step_m"] = round(
                    min(abs(used[i + 1]["depth"] - used[i]["depth"])
                        for i in range(len(used) - 1)), 2
                )

            cal["confidence"] = min(0.95, 0.5 + 0.05 * len(used))
            expected = cal.get("expected_meters_per_pixel")
            if expected and slope > 0:
                ratio = slope / expected
                if 0.8 <= ratio <= 1.25:
                    cal["confidence"] = min(0.97, cal["confidence"] + 0.1)
                else:
                    cal["error"] = (
                        f"slope {slope:.5f} m/px disagrees with expected "
                        f"{expected:.5f} m/px (scale 1:{scale_hint}, dpi {dpi})"
                    )
                    cal["confidence"] = max(0.2, cal["confidence"] - 0.3)
            return cal

    # --- 0-1 прочитанных меток: масштаб из геометрии блобов ---
    if spacing_px and cal["expected_meters_per_pixel"]:
        step_m = _snap_label_step(spacing_px * cal["expected_meters_per_pixel"])
        if step_m:
            slope = step_m / spacing_px
            cal["label_step_m"] = step_m
            cal["meters_per_pixel"] = round(slope, 6)
            if len(read) == 1:
                anchor = read[0]
                anchor["used"] = True
                intercept = anchor["depth"] - slope * anchor["cy"]
                cal.update(
                    {
                        "status": "approx",
                        "method": f"cv_spacing+{anchor['source']}_anchor",
                        "depth_at_y0": round(intercept, 3),
                        "num_labels_used": 1,
                        "confidence": 0.4,
                    }
                )
                y_top, y_bot = labels[0]["cy"], labels[-1]["cy"]
                cal["pixel_top"] = int(round(y_top))
                cal["depth_top"] = round(slope * y_top + intercept, 2)
                cal["pixel_bottom"] = int(round(y_bot))
                cal["depth_bottom"] = round(slope * y_bot + intercept, 2)
            else:
                cal.update(
                    {
                        "status": "labels_only",
                        "method": "cv_spacing",
                        "pixel_top": int(round(labels[0]["cy"])),
                        "pixel_bottom": int(round(labels[-1]["cy"])),
                        "confidence": 0.25,
                        "error": "label values unread: need OCR for handwriting or LM Studio",
                    }
                )
            return cal

    cal["error"] = (
        blob_error
        or "labels found but neither values nor scale hint available"
    )
    cal["status"] = "labels_only" if labels else "failed"
    if labels:
        cal["pixel_top"] = int(round(labels[0]["cy"]))
        cal["pixel_bottom"] = int(round(labels[-1]["cy"]))
    return cal


def _plausible_depth_window(result: dict) -> tuple[float, float]:
    depth_from = float(result.get("depth_from") or 0.0)
    depth_to = float(result.get("depth_to") or 0.0)
    if depth_from and depth_to and depth_to > depth_from:
        margin = max(50.0, 0.3 * (depth_to - depth_from))
        return depth_from - margin, depth_to + margin
    return 10.0, 10000.0


def _parse_depth_label_text(text: Any, lo: float, hi: float) -> Optional[float]:
    if text is None:
        return None
    cleaned = str(text).strip().replace(" ", "")
    if not re.fullmatch(r"\d{2,5}(?:[.,]\d{1,2})?", cleaned):
        return None
    value = _to_float(cleaned)
    if value is None or not (lo <= value <= hi):
        return None
    return float(value)


def _median_label_spacing(ys: list[float]) -> Optional[float]:
    if len(ys) < 3:
        return None
    diffs = sorted(b - a for a, b in zip(ys, ys[1:]) if b - a > 30)
    if not diffs:
        return None
    med = diffs[len(diffs) // 2]
    # Усредняем только "нормальные" интервалы (пропуски меток дают 2x-выбросы).
    good = [d for d in diffs if 0.7 * med <= d <= 1.35 * med]
    if len(good) < 2:
        return None
    return sum(good) / len(good)


def _snap_label_step(step_estimate_m: float) -> Optional[float]:
    best = min(DEPTH_LABEL_STEPS_M, key=lambda s: abs(s - step_estimate_m))
    if abs(best - step_estimate_m) <= 0.3 * best:
        return best
    return None


def _robust_linear_fit(pairs: list[tuple[float, float]]):
    """
    Робастная прямая v = m*y + b по парам (y, v).
    Возвращает (m, b, индексы инлайеров, rmse) или None.
    """
    pairs = sorted(pairs)
    n = len(pairs)
    if n < 2:
        return None
    if n == 2:
        (y1, v1), (y2, v2) = pairs
        if abs(y2 - y1) < 10:
            return None
        m = (v2 - v1) / (y2 - y1)
        return m, v1 - m * y1, {0, 1}, 0.0

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dy = pairs[j][0] - pairs[i][0]
            if abs(dy) > 40:
                slopes.append((pairs[j][1] - pairs[i][1]) / dy)
    if not slopes:
        return None
    slopes.sort()
    m = slopes[len(slopes) // 2]
    intercepts = sorted(v - m * y for y, v in pairs)
    b = intercepts[len(intercepts) // 2]

    values = [v for _, v in pairs]
    tol = max(2.0, 0.05 * (max(values) - min(values)))
    inliers = {i for i, (y, v) in enumerate(pairs) if abs(v - (m * y + b)) <= tol}
    if len(inliers) < 2:
        return None

    # Least squares по инлайерам.
    xs = [pairs[i][0] for i in inliers]
    vs = [pairs[i][1] for i in inliers]
    k = len(xs)
    mean_x, mean_v = sum(xs) / k, sum(vs) / k
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom < 1e-9:
        return None
    m = sum((x - mean_x) * (v - mean_v) for x, v in zip(xs, vs)) / denom
    b = mean_v - m * mean_x
    rmse = (sum((v - (m * x + b)) ** 2 for x, v in zip(xs, vs)) / k) ** 0.5
    return m, b, inliers, rmse


def _text_blob_boxes(image) -> list[tuple[int, int, int, int]]:
    """
    Найти прямоугольники компактных чернильных блобов (рукописный текст).

    Длинные линии (сетка, рамки) и вертикальные участки кривых вычитаются,
    цифры одного числа склеиваются в один компонент.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ink = ((gray < INK_GRAY_THRESHOLD) * 255).astype("uint8")

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 40), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    long_lines = cv2.bitwise_or(
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, horiz_kernel),
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, vert_kernel),
    )
    text_ink = cv2.subtract(ink, long_lines)

    glue = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    grouped = cv2.dilate(text_ink, glue)

    count, _, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if not (12 <= bh <= 90 and 12 <= bw <= 300):
            continue
        if area < 0.15 * bw * bh:
            continue
        if x < w * 0.01 or x + bw > w * 0.99:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    return boxes


def _detect_depth_label_blobs(image) -> list[tuple[int, int, int, int]]:
    """
    Найти прямоугольники рукописных меток глубины.

    Метки — это 3-4-значные ЧИСЛА в вертикальной колонке с почти постоянным
    шагом (обычно колонка глубин слева/между треками). Фильтр на «числоподобную»
    форму (ширина >= 40 px,w/h >= 1.6) критичен: иначе детектор берёт колонку
    узких тик-меток шкалы (w~30, шаг ~2 м) вместо настоящих чисел глубины.
    """
    w = image.shape[1]
    boxes = [
        b for b in _text_blob_boxes(image)
        if b[2] >= 40 and b[2] >= b[3] * 1.6
    ]

    if len(boxes) < 4:
        return []

    # Скользящим окном по x ищем колонку с самой регулярной вертикальной
    # последовательностью блобов (цепочечная кластеризация здесь не годится:
    # обрывки кривых дают почти сплошное покрытие по x).
    tol = max(25, int(w * 0.03))
    best, best_score = None, 0.0
    centers = sorted({int(b[0] + b[2] / 2.0) for b in boxes})
    for center in centers:
        cluster = [b for b in boxes if abs(b[0] + b[2] / 2.0 - center) <= tol]
        if len(cluster) < 4:
            continue
        ys = sorted(b[1] + b[3] / 2.0 for b in cluster)
        diffs = [b - a for a, b in zip(ys, ys[1:])]
        med = sorted(diffs)[len(diffs) // 2]
        if med < 60:  # слишком плотно — скорее обрывки кривой, чем метки
            continue
        regular = sum(1 for d in diffs if 0.75 * med <= d <= 1.3 * med) / len(diffs)
        score = regular * regular * len(cluster)
        if score > best_score:
            best, best_score = cluster, score

    if best is None or best_score < 3.0:
        return []

    # Внутри выбранной колонки оставляем блобы, попадающие в регулярную сетку.
    ys = sorted(b[1] + b[3] / 2.0 for b in best)
    diffs = sorted(b - a for a, b in zip(ys, ys[1:]))
    med = diffs[len(diffs) // 2]
    best.sort(key=lambda b: b[1])
    filtered = [best[0]]
    for box in best[1:]:
        gap = (box[1] + box[3] / 2.0) - (filtered[-1][1] + filtered[-1][3] / 2.0)
        if gap < 0.4 * med:
            continue  # дубликат/клякса рядом с предыдущей меткой
        filtered.append(box)
    return filtered


def _sample_evenly(items: list, limit: int) -> list:
    if len(items) <= limit:
        return list(items)
    step = (len(items) - 1) / (limit - 1)
    return [items[int(round(i * step))] for i in range(limit)]


def _read_label_crops_lm_studio(
    image,
    labels: list[dict],
    base_url: str,
    model: Optional[str],
    timeout: float,
) -> list[Optional[float]]:
    """Прочитать рукописные числа на кропах меток через LM Studio VLM."""
    import cv2

    base_url = base_url.rstrip("/")
    empty = [None] * len(labels)
    try:
        model_id = model or _first_lm_studio_model(base_url=base_url, timeout=min(timeout, 2.0))
        if not model_id:
            return empty

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"You see {len(labels)} small images. Each contains one handwritten "
                    "number: a depth mark in meters from an old well log. "
                    f"Return ONLY a JSON array of {len(labels)} numbers in the same "
                    "order. Use null for unreadable images. No other text."
                ),
            }
        ]
        h, w = image.shape[:2]
        for lab in labels:
            pad = 6
            x0 = max(0, lab["x_px"] - pad)
            y0 = max(0, lab["y_px"] - pad)
            x1 = min(w, lab["x_px"] + lab["w_px"] + pad)
            y1 = min(h, lab["y_px"] + lab["h_px"] + pad)
            crop = image[y0:y1, x0:x1]
            crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            ok, encoded = cv2.imencode(".png", crop)
            if not ok:
                return empty
            data_url = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        response = _http_json(
            url=f"{base_url}/chat/completions",
            method="POST",
            payload={
                "model": model_id,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                # Reasoning-модели тратят сотни токенов на размышления до ответа.
                "max_tokens": 3000,
                "reasoning_effort": "low",
            },
            timeout=max(timeout, 120.0),
        )
        message = response.get("choices", [{}])[0].get("message", {})
        match = None
        for raw in (message.get("content", ""), message.get("reasoning_content", "")):
            raw = re.sub(r"```(?:json)?", "", raw or "").replace("```", "").strip()
            found = list(re.finditer(r"\[[^\[\]]*\]", raw, re.DOTALL))
            if found:
                match = found[-1]
                break
        if not match:
            return empty
        values = json.loads(match.group())
        if not isinstance(values, list):
            return empty
        values = values[: len(labels)] + [None] * max(0, len(labels) - len(values))
        return [_to_float(v) for v in values]
    except Exception as e:
        log.debug("LM Studio label crops failed: %s", e)
        return empty


def _merge_calibration_layer(result: dict, cal: dict) -> None:
    if cal.get("status") not in {"ok", "approx"}:
        return
    # Имя файла может отличаться от фактического диапазона на изображении —
    # заполняем глубины из калибровки только если их еще нет.
    if not (result.get("depth_from") and result.get("depth_to")):
        if cal.get("depth_top") is not None and cal.get("depth_bottom") is not None:
            result["depth_from"] = min(cal["depth_top"], cal["depth_bottom"])
            result["depth_to"] = max(cal["depth_top"], cal["depth_bottom"])


# ---------------------------------------------------------------------------
# Шкалы кривых из OCR-блоков (scale_min/scale_max + тики x -> value)
# ---------------------------------------------------------------------------


def _extract_scales_from_ocr(result: dict) -> list[dict]:
    """
    Найти в OCR-блоках строки-шкалы вида "0 2 4 6 8 ОММ".

    Возвращает список шкал и обновляет result["curves"]:
    scale_min/scale_max/scale_type/unit + value_ticks_px для пересчета x -> value.
    """
    blocks = _normalized_ocr_blocks(result)
    if not blocks:
        return []

    heights = sorted(b["bbox_px"][3] for b in blocks)
    line_tol = max(12, heights[len(heights) // 2])

    # Группируем блоки в горизонтальные строки по cy.
    blocks.sort(key=lambda b: (b["cy"], b["cx"]))
    lines: list[list[dict]] = []
    for block in blocks:
        if lines and abs(block["cy"] - lines[-1][0]["cy"]) <= line_tol:
            lines[-1].append(block)
        else:
            lines.append([block])

    scales = []
    for line in lines:
        line.sort(key=lambda b: b["cx"])
        ticks = []
        unit = ""
        for block in line:
            text = (block.get("text") or "").strip()
            value = _to_float(text.replace(",", "."))
            if value is not None and re.fullmatch(r"-?\d{1,5}(?:[.,]\d{1,2})?", text.replace(" ", "")):
                ticks.append({"x_px": block["cx"], "value": float(value)})
                continue
            normalized_unit = UNIT_TOKENS.get(text.upper().strip(".,"))
            if normalized_unit:
                unit = normalized_unit
        if len(ticks) < 3:
            continue

        scale_type = _classify_scale_ticks([t["value"] for t in ticks])
        if scale_type is None:
            continue
        scales.append(
            {
                "y_px": int(round(line[0]["cy"])),
                "x_start_px": int(round(ticks[0]["x_px"])),
                "x_end_px": int(round(ticks[-1]["x_px"])),
                "scale_min": ticks[0]["value"],
                "scale_max": ticks[-1]["value"],
                "scale_type": scale_type,
                "unit": unit,
                "ticks": ticks,
                "source": "ocr",
            }
        )

    if scales:
        _assign_scales_to_curves(result, scales)
    return scales


def _classify_scale_ticks(values: list[float]) -> Optional[str]:
    """linear для арифметической прогрессии, log для геометрической, иначе None."""
    if len(values) < 3:
        return None
    if any(b <= a for a, b in zip(values, values[1:])):
        # Шкалы могут убывать (SP), допускаем строго монотонные в обе стороны.
        if not all(b < a for a, b in zip(values, values[1:])):
            return None
    diffs = [abs(b - a) for a, b in zip(values, values[1:])]
    mean_diff = sum(diffs) / len(diffs)
    if mean_diff > 0 and all(abs(d - mean_diff) <= 0.25 * mean_diff for d in diffs):
        return "linear"
    if all(v > 0 for v in values):
        ratios = [b / a for a, b in zip(values, values[1:])]
        mean_ratio = sum(ratios) / len(ratios)
        if mean_ratio > 1.5 and all(abs(r - mean_ratio) <= 0.35 * mean_ratio for r in ratios):
            return "log"
    return None


def _assign_scales_to_curves(result: dict, scales: list[dict]) -> None:
    """Привязать найденные шкалы к кривым: по перекрытию с треками или по порядку."""
    curves = result.get("curves", [])
    if not curves:
        return
    scales_sorted = sorted(scales, key=lambda s: s["x_start_px"])
    curves_sorted = sorted(
        range(len(curves)),
        key=lambda i: curves[i].get("track_band_px", [10 ** 9])[0]
        if curves[i].get("track_band_px")
        else 10 ** 9,
    )
    for scale, curve_idx in zip(scales_sorted, curves_sorted[: len(scales_sorted)]):
        curve = curves[curve_idx]
        update = {
            "scale_min": scale["scale_min"],
            "scale_max": scale["scale_max"],
            "scale_type": scale["scale_type"],
        }
        if scale.get("unit"):
            update["unit"] = scale["unit"]
        for key, value in update.items():
            curve[key] = value
        curve["value_ticks_px"] = [
            [round(t["x_px"], 1), t["value"]] for t in scale["ticks"]
        ]
        curve["scale_source"] = "ocr"
        if float(curve.get("confidence") or 0.0) < 0.7:
            curve["confidence"] = 0.7


def _read_scales_with_lm_studio(
    cv_image,
    result: dict,
    base_url: str,
    model: Optional[str],
    timeout: float,
) -> dict:
    """
    Прочитать рукописные строки-шкалы ("0 2 4 6 8 ОММ") в шапках треков
    через LM Studio. Для кривых без OCR-шкалы обновляет scale_min/scale_max,
    unit и value_ticks_px (x-позиции блобов + значения от VLM).
    """
    layer = {"status": "not_run", "rows": [], "error": ""}
    curves = [
        c for c in result.get("curves", [])
        if c.get("track_band_px") and c.get("scale_source") != "ocr"
    ]
    if not curves:
        layer["status"] = "skipped"
        return layer

    cal = result.get("pixel_calibration", {})
    pixel_top = cal.get("pixel_top")
    pixel_bottom = cal.get("pixel_bottom")
    if pixel_top is None or pixel_bottom is None:
        layer["status"] = "skipped"
        layer["error"] = "no pixel calibration to locate header zone"
        return layer

    try:
        boxes = _text_blob_boxes(cv_image)
    except Exception as e:
        layer["status"] = "failed"
        layer["error"] = str(e)
        return layer

    # Зона поиска строки шкалы: внутри сетки, верхняя четверть рабочей области
    # (выше pixel_top идет шапка с произвольными надписями).
    y_lo = max(0, int(pixel_top) - 50)
    y_hi = int(pixel_top + 0.25 * (pixel_bottom - pixel_top))

    for curve in curves:
        bx0, bx1 = curve["track_band_px"]
        # Фильтр по центру блоба: подпись "8 ОММ" может выступать за границу
        # полосы трека, обрезать ее нельзя.
        in_band = [
            b for b in boxes
            if y_lo <= b[1] <= y_hi and bx0 - 40 <= b[0] + b[2] / 2.0 <= bx1 + 40
        ]
        row = _topmost_text_row(in_band)
        if not row:
            continue
        numbers, unit = _read_scale_row_lm_studio(
            cv_image, row, base_url=base_url, model=model, timeout=timeout
        )
        if len(numbers) < 2:
            # Длина reasoning у VLM нестабильна: одна повторная попытка.
            numbers, unit = _read_scale_row_lm_studio(
                cv_image, row, base_url=base_url, model=model, timeout=timeout
            )
        row_info = {
            "curve": curve.get("name"),
            "y_px": int(sum(b[1] + b[3] / 2.0 for b in row) / len(row)),
            "boxes": [list(b) for b in row],
            "numbers": numbers,
            "unit": unit,
        }
        layer["rows"].append(row_info)
        if len(numbers) < 2 or numbers != sorted(numbers):
            continue
        scale_type = _classify_scale_ticks(numbers) or "linear"
        curve["scale_min"] = numbers[0]
        curve["scale_max"] = numbers[-1]
        curve["scale_type"] = scale_type
        if unit:
            curve["unit"] = unit
        # Тики x->value возможны только при однозначном соответствии чисел и
        # блобов. Юнит либо отдельный последний блоб, либо склеен с последним
        # числом ("8 ОММ" = один блоб) — оба случая дают валидный zip по числам.
        if len(row) in {len(numbers), len(numbers) + (1 if unit else 0)}:
            ticks = [
                [round(b[0] + b[2] / 2.0, 1), v]
                for b, v in zip(row, numbers)
            ]
            if _ticks_fit_ok(ticks, scale_type):
                curve["value_ticks_px"] = ticks
        curve["scale_source"] = "lm_studio"
        if float(curve.get("confidence") or 0.0) < 0.72:
            curve["confidence"] = 0.72

    layer["status"] = "ok" if layer["rows"] else "failed"
    if not layer["rows"]:
        layer["error"] = "no scale text rows found in track headers"
    return layer


def _ticks_fit_ok(ticks: list, scale_type: str) -> bool:
    """Проверить, что тики ложатся на прямую value(x): защита от блобов-мусора."""
    import math

    if len(ticks) < 2:
        return False
    xs = [t[0] for t in ticks]
    if any(b - a < 20 for a, b in zip(xs, xs[1:])):
        return False
    vs = [t[1] for t in ticks]
    if scale_type == "log":
        if any(v <= 0 for v in vs):
            return False
        vs = [math.log10(v) for v in vs]
    n = len(xs)
    mean_x, mean_v = sum(xs) / n, sum(vs) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom < 1e-9:
        return False
    a = sum((x - mean_x) * (v - mean_v) for x, v in zip(xs, vs)) / denom
    b = mean_v - a * mean_x
    span = max(vs) - min(vs)
    if span <= 0:
        return False
    worst = max(abs(v - (a * x + b)) for x, v in zip(xs, vs))
    return worst <= 0.1 * span


def _topmost_text_row(boxes: list) -> list:
    """Самая верхняя горизонтальная строка из >=3 блобов с близкими cy."""
    if len(boxes) < 3:
        return []
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2.0)
    tol = 30
    rows: list[list] = [[boxes[0]]]
    for box in boxes[1:]:
        cy = box[1] + box[3] / 2.0
        row_cy = sum(b[1] + b[3] / 2.0 for b in rows[-1]) / len(rows[-1])
        if abs(cy - row_cy) <= tol:
            rows[-1].append(box)
        else:
            rows.append([box])
    for row in rows:
        if len(row) < 3:
            continue
        # Дедупликация по x: вертикальные мини-колонки ("0/10/50/250") дают
        # несколько блобов с одинаковым cx — для строки шкалы нужен один.
        row.sort(key=lambda b: b[0] + b[2] / 2.0)
        row_cy = sum(b[1] + b[3] / 2.0 for b in row) / len(row)
        deduped: list = []
        for box in row:
            cx = box[0] + box[2] / 2.0
            if deduped and abs(cx - (deduped[-1][0] + deduped[-1][2] / 2.0)) < 25:
                # Из дубликатов оставляем ближайший к средней линии строки.
                if abs(box[1] + box[3] / 2.0 - row_cy) < abs(
                    deduped[-1][1] + deduped[-1][3] / 2.0 - row_cy
                ):
                    deduped[-1] = box
                continue
            deduped.append(box)
        if len(deduped) >= 3:
            return deduped
    return []


def _read_scale_row_lm_studio(
    image,
    row_boxes: list,
    base_url: str,
    model: Optional[str],
    timeout: float,
) -> tuple[list[float], str]:
    """Прочитать одну строку шкалы. Возвращает (числа слева направо, unit)."""
    import cv2

    base_url = base_url.rstrip("/")
    empty: tuple[list[float], str] = ([], "")
    try:
        model_id = model or _first_lm_studio_model(base_url=base_url, timeout=min(timeout, 2.0))
        if not model_id:
            return empty

        h, w = image.shape[:2]
        pad = 15
        x0 = max(0, min(b[0] for b in row_boxes) - pad)
        x1 = min(w, max(b[0] + b[2] for b in row_boxes) + pad)
        y0 = max(0, min(b[1] for b in row_boxes) - pad)
        y1 = min(h, max(b[1] + b[3] for b in row_boxes) + pad)
        crop = image[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            return empty
        data_url = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

        prompt = (
            "Image: handwritten scale numbers from a well log track header. "
            "Read the MAIN horizontal row of evenly spaced numbers (like 0 2 4 6 8) "
            "and the measurement unit after it, if any (ОММ means Ohmm, МВ means mV). "
            "Ignore small auxiliary numbers or vertical columns. Do not explain. "
            'Answer with ONLY this JSON: {"numbers": [..], "unit": ".."}'
        )
        response = _http_json(
            url=f"{base_url}/chat/completions",
            method="POST",
            payload={
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 6000,
                # Поддерживается не всеми моделями; лишний ключ LM Studio игнорирует.
                "reasoning_effort": "low",
            },
            # Reasoning по зашумленному рукописному кропу может идти несколько минут.
            timeout=max(timeout, 240.0),
        )
        message = response.get("choices", [{}])[0].get("message", {})
        parsed = None
        for raw in (message.get("content", ""), message.get("reasoning_content", "")):
            raw = re.sub(r"```(?:json)?", "", raw or "").replace("```", "").strip()
            for match in reversed(list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))):
                try:
                    candidate = json.loads(match.group())
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and isinstance(candidate.get("numbers"), list):
                    parsed = candidate
                    break
            if parsed:
                break
        if not parsed:
            return empty

        numbers = [v for v in (_to_float(n) for n in parsed["numbers"]) if v is not None]
        unit = UNIT_TOKENS.get(str(parsed.get("unit") or "").upper().strip(" .,"), "")
        if not unit:
            unit = _normalize_unit(parsed.get("unit") or "")
            if unit and unit not in set(UNIT_TOKENS.values()):
                unit = ""
        return numbers, unit
    except Exception as e:
        log.debug("LM Studio scale row failed: %s", e)
        return empty


def _recompute_trace_samples(result: dict) -> None:
    """Пересчитать сэмплы depth/value после обновления шкал кривых."""
    cal = result.get("pixel_calibration", {})
    slope = cal.get("meters_per_pixel")
    intercept = cal.get("depth_at_y0")
    if cal.get("status") not in {"ok", "approx"} or not slope or intercept is None:
        return
    for curve in result.get("curves", []):
        trace = curve.get("trace")
        if not trace or not trace.get("points_px"):
            continue
        band = trace.get("band_px") or curve.get("track_band_px")
        if not band:
            continue
        value_fn = _make_value_mapper(curve, band[0], band[1])
        if value_fn is None:
            continue
        trace["samples"] = [
            [round(slope * y + intercept, 2), value_fn(x)]
            for y, x in _clip_points_to_extent(trace["points_px"], trace)
        ]


# ---------------------------------------------------------------------------
# Собственная трассировка кривых (полилиния + сэмплы depth/value)
# ---------------------------------------------------------------------------


def _row_runs(row_mask, min_gap: int = 4, max_width_frac: float = 0.5):
    """Центры контрастных «прогонов» чернил в строке (x_local, width)."""
    import numpy as np

    idx = np.nonzero(row_mask)[0]
    if idx.size == 0:
        return []
    runs = []
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - prev > min_gap:
            runs.append(((start + prev) / 2.0, prev - start + 1))
            start = v
        prev = v
    runs.append(((start + prev) / 2.0, prev - start + 1))
    limit = row_mask.shape[0] * max_width_frac if row_mask.ndim else len(row_mask) * max_width_frac
    return [rc for rc in runs if rc[1] < limit]


def _track_band_line(sub, max_jump_frac: float = 0.30, reseed_gap: int = 12):
    """
    Континуити-трекер линии в полосе трека: на каждой строке выбирает
    «прогон» чернил, ближайший к предыдущей позиции (следует по линии,
    а не усредняет несколько линий, как центроид). Возвращает x в координатах
    ПОЛОСЫ (NaN там, где линия не прослеживается).

    Зачем: центроид строки усредняет перевыносные/пересекающиеся линии и
    тянет значение к середине (срезает пики). Трекер идёт по одной линии.
    """
    import numpy as np

    h, band_w = sub.shape
    max_jump = max(15.0, band_w * max_jump_frac)
    rows_runs = []
    for r in range(h):
        idx = np.nonzero(sub[r])[0]
        if idx.size == 0:
            rows_runs.append([])
            continue
        runs = []
        start = prev = int(idx[0])
        for v in idx[1:]:
            v = int(v)
            if v - prev > 4:
                runs.append(((start + prev) / 2.0, prev - start + 1))
                start = v
            prev = v
        runs.append(((start + prev) / 2.0, prev - start + 1))
        runs = [rc for rc in runs if rc[1] < band_w * 0.5]
        rows_runs.append(runs)

    track = np.full(h, np.nan, dtype=np.float64)
    prev_x = None
    gap = 0
    for r in range(h):
        runs = rows_runs[r]
        if not runs:
            gap += 1
            if gap >= reseed_gap:
                prev_x = None
            continue
        if prev_x is None:
            # сид: самый узкий прогон (чистая линия), при равенстве — ближе к центру
            prev_x = min(runs, key=lambda rc: (rc[1], abs(rc[0] - band_w / 2.0)))[0]
            track[r] = prev_x
            gap = 0
            continue
        cand = min(runs, key=lambda rc: abs(rc[0] - prev_x))
        if abs(cand[0] - prev_x) <= max_jump:
            prev_x = cand[0]
            track[r] = prev_x
            gap = 0
        else:
            gap += 1
            if gap >= reseed_gap:
                prev_x = None
    return track


def _extract_curve_traces(result: dict, cv_image=None, max_samples: int = 1500) -> dict:
    """
    Извлечь кривые как полилинии из чернильной маски.

    Для каждой полосы трека: по каждой строке y берется центр тяжести темных
    пикселей. Если есть pixel_calibration и шкала кривой — добавляются сэмплы
    [depth_m, value].
    """
    layer = {"status": "not_run", "bands_px": [], "num_traces": 0, "error": ""}
    try:
        import numpy as np
    except ImportError:
        layer["status"] = "unavailable"
        layer["error"] = "numpy is not installed"
        return layer
    if cv_image is None:
        layer["status"] = "unavailable"
        layer["error"] = "opencv image not loaded"
        return layer

    import cv2

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cal = result.get("pixel_calibration", {})

    # Рабочая зона по вертикали: весь диапазон найденных меток глубины
    # (pixel_top/bottom могут покрывать только ПРОЧИТАННЫЕ метки и резать
    # хвост кривой), плюс запас в один шаг меток.
    pad = int(2 * (cal.get("label_step_m") or 10.0) / (cal.get("meters_per_pixel") or 0.03))
    if cal.get("depth_labels"):
        ys = [lab["cy"] for lab in cal["depth_labels"]]
        y0 = max(0, int(min(ys)) - pad)
        y1 = min(h, int(max(ys)) + pad)
    elif cal.get("pixel_top") is not None and cal.get("pixel_bottom") is not None:
        y0 = max(0, int(cal["pixel_top"]) - pad)
        y1 = min(h, int(cal["pixel_bottom"]) + pad)
    else:
        y0, y1 = int(h * 0.05), int(h * 0.99)
    if y1 - y0 < 100:
        layer["status"] = "failed"
        layer["error"] = "work region too small"
        return layer

    ink = gray[y0:y1] < INK_GRAY_THRESHOLD

    # Полосы треков: столбцы с заметной плотностью чернил.
    # Черные края скана исключаем и из порога, и из активной зоны.
    density = ink.mean(axis=0)
    kernel = np.ones(21) / 21.0
    smooth = np.convolve(density, kernel, mode="same")
    margin = max(4, int(w * 0.02))
    interior_max = float(smooth[margin: w - margin].max()) if w > 2 * margin else float(smooth.max())
    # Относительный порог с абсолютным потолком: одна аномально темная
    # полоса (тень от края скана) не должна отсекать бледные треки.
    threshold = max(0.018, min(0.12 * interior_max, 0.04))
    active = smooth > threshold
    active[:margin] = False
    active[w - margin:] = False

    bands = []
    start = None
    for x in range(w):
        if active[x] and start is None:
            start = x
        elif not active[x] and start is not None:
            bands.append([start, x])
            start = None
    if start is not None:
        bands.append([start, w])

    # Склеить близкие, выкинуть узкие и приклеенные к краям скана.
    merged = []
    for band in bands:
        if merged and band[0] - merged[-1][1] < 25:
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    label_cx = None
    label_zone = None
    if cal.get("depth_labels"):
        xs = sorted(lab["cx"] for lab in cal["depth_labels"])
        label_cx = xs[len(xs) // 2]
        label_zone = (
            int(min(lab["x_px"] for lab in cal["depth_labels"])) - 8,
            int(max(lab["x_px"] + lab["w_px"] for lab in cal["depth_labels"])) + 8,
        )
    bands = []
    for x0, x1 in merged:
        if x1 - x0 < 40:
            continue
        if x0 < w * 0.015 or x1 > w * 0.985:
            continue
        if label_cx is not None and x0 <= label_cx <= x1 and (x1 - x0) < w * 0.12:
            continue  # это колонка меток глубины, не трек
        bands.append([int(x0), int(x1)])

    # Полосы выше порога — только "ядра" треков: на пиках кривая уходит в
    # столбцы с низкой плотностью. Расширяем ядро, пока есть хоть какие-то
    # чернила и пока не уперлись в соседний трек / колонку меток / края.
    blocked = np.zeros(w, dtype=bool)
    blocked[:margin] = True
    blocked[w - margin:] = True
    if label_zone is not None:
        blocked[max(0, label_zone[0]): min(w, label_zone[1])] = True
    for x0, x1 in bands:
        blocked[x0:x1] = True
    eps = 0.004
    for band in bands:
        x = band[0]
        while x - 1 >= 0 and not blocked[x - 1] and smooth[x - 1] > eps:
            x -= 1
        band[0] = x
        x = band[1]
        while x < w and not blocked[x] and smooth[x] > eps:
            x += 1
        band[1] = x
        blocked[band[0]: band[1]] = True

    layer["bands_px"] = bands
    if not bands:
        layer["status"] = "failed"
        layer["error"] = "no ink bands detected"
        return layer

    # Маппинг глубины.
    slope = cal.get("meters_per_pixel")
    intercept = cal.get("depth_at_y0")
    depth_ok = cal.get("status") in {"ok", "approx"} and slope and intercept is not None

    curves = result.get("curves", [])
    xs_idx = np.arange(w, dtype=np.float64)
    traces_built = 0
    for band_i, (bx0, bx1) in enumerate(bands):
        sub = ink[:, bx0:bx1]
        counts = sub.sum(axis=1)
        sums = (sub * xs_idx[bx0:bx1]).sum(axis=1)
        band_w = bx1 - bx0
        # Центроид строки (надёжный baseline для одношкальных кривых).
        # Континуити-трекер `_track_band_line` пока НЕ используется по умолчанию:
        # на перевыносных данных он залипает на не-той линии; его ценность —
        # как вход для разворачивания уровней (A1), которое ещё не реализовано.
        valid = (counts > 0) & (counts < band_w * 0.6)  # широкие ряды = текст/кляксы
        rows = np.nonzero(valid)[0]
        if rows.size == 0:
            continue
        coverage = float(rows.size) / float(y1 - y0)
        if coverage < 0.15:
            continue
        mean_x = np.where(valid, sums / np.maximum(counts, 1), np.nan)

        sample_rows = rows[np.linspace(0, rows.size - 1, min(max_samples, rows.size)).astype(int)]
        points = [[int(r + y0), round(float(mean_x[r]), 1)] for r in sample_rows]

        curve = curves[band_i] if band_i < len(curves) else None
        trace = {
            "status": "ok",
            "band_px": [bx0, bx1],
            "coverage": round(coverage, 3),
            "n_rows": int(rows.size),
            "data_top_px": None,
            "data_bottom_px": None,
            "points_px": points,  # [y_px, x_px]
            "samples": [],        # [depth_m, value]
        }

        # Фактические границы данных кривой: непрерывная заливка чернилами,
        # а не редкие всплески от текста шкал/подписей.
        extent = _detect_data_extent(valid, mean_x)
        if extent is not None:
            trace["data_top_px"] = int(extent[0] + y0)
            trace["data_bottom_px"] = int(extent[1] + y0)

        if curve is not None:
            if depth_ok and extent is not None:
                curve["depth_start"] = round(slope * trace["data_top_px"] + intercept, 2)
                curve["depth_end"] = round(slope * trace["data_bottom_px"] + intercept, 2)
            value_fn = _make_value_mapper(curve, bx0, bx1)
            if depth_ok and value_fn is not None:
                sample_points = _clip_points_to_extent(points, trace)
                trace["samples"] = [
                    [round(slope * y + intercept, 2), value_fn(x)]
                    for y, x in sample_points
                ]
            curve["track_band_px"] = [bx0, bx1]
            curve["trace"] = trace
        traces_built += 1

    layer["num_traces"] = traces_built
    layer["status"] = "ok" if traces_built else "failed"
    if not traces_built:
        layer["error"] = "ink bands found but no usable traces"

    # Рекомендация для рамки Depth axis Блока 3: границы данных всех кривых,
    # округленные наружу до целого метра (линии должны попасть внутрь рамки).
    if depth_ok:
        import math

        starts = [c["depth_start"] for c in curves if c.get("depth_start") is not None]
        ends = [c["depth_end"] for c in curves if c.get("depth_end") is not None]
        if starts and ends:
            result["depth_axis_suggestion"] = {
                "depth_from": float(math.floor(min(starts))),
                "depth_to": float(math.ceil(max(ends))),
                "source": "traces",
            }
    return layer


def _detect_data_extent(
    valid,
    mean_x,
    window_px: int = 150,
    min_density: float = 0.7,
    min_std_px: float = 1.5,
):
    """
    Найти (start_row, end_row) непрерывных данных кривой.

    Данные = плотная заливка чернилами И заметная дисперсия x в окне:
    кривая извивается, а прочерченная граница трека / холостой ход пера —
    прямая вертикальная линия с дисперсией меньше пикселя.
    """
    import numpy as np

    v = valid.astype(np.float64)
    if v.size < window_px:
        return None
    x = np.nan_to_num(mean_x, nan=0.0) * v
    kernel = np.ones(window_px)
    cnt = np.convolve(v, kernel, mode="valid")
    sx = np.convolve(x, kernel, mode="valid")
    sxx = np.convolve(x * x, kernel, mode="valid")
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sx / cnt
        var = np.maximum(sxx / cnt - mean * mean, 0.0)
        std = np.sqrt(var)
    qualified = (cnt >= min_density * window_px) & (std >= min_std_px)
    idx = np.nonzero(qualified)[0]
    if idx.size == 0:
        return None

    # Рукописные шапка и подписи внизу тоже плотные и "извилистые" —
    # данные кривой это самый ДЛИННЫЙ непрерывный участок, а не first..last.
    # Разрывы до max_gap_px (складки бумаги, обрывы линии) склеиваем.
    max_gap_px = 300
    runs: list[list[int]] = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if int(i) - runs[-1][1] <= max_gap_px:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    best_run = max(runs, key=lambda r: r[1] - r[0])
    start_win, end_win = best_run[0], best_run[1] + window_px - 1
    rows = np.nonzero(valid)[0]
    start_rows = rows[rows >= start_win]
    end_rows = rows[rows <= end_win]
    if start_rows.size == 0 or end_rows.size == 0:
        return None
    return int(start_rows[0]), int(end_rows[-1])


def _clip_points_to_extent(points: list, trace: dict) -> list:
    top = trace.get("data_top_px")
    bottom = trace.get("data_bottom_px")
    if top is None or bottom is None:
        return points
    return [p for p in points if top <= p[0] <= bottom]


def _make_value_mapper(curve: dict, band_x0: int, band_x1: int):
    """Функция x_px -> значение кривой; None, если шкала неизвестна."""
    import math

    ticks = curve.get("value_ticks_px") or []
    scale_type = (curve.get("scale_type") or "linear").lower()
    if len(ticks) >= 2:
        xs = [t[0] for t in ticks]
        vs = [t[1] for t in ticks]
        if scale_type == "log":
            if any(v <= 0 for v in vs):
                return None
            vs = [math.log10(v) for v in vs]
        n = len(xs)
        mean_x, mean_v = sum(xs) / n, sum(vs) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom < 1e-9:
            return None
        a = sum((x - mean_x) * (v - mean_v) for x, v in zip(xs, vs)) / denom
        b = mean_v - a * mean_x
        if scale_type == "log":
            return lambda x: round(10 ** (a * x + b), 4)
        return lambda x: round(a * x + b, 4)

    scale_min = _to_float(curve.get("scale_min"), 0.0)
    scale_max = _to_float(curve.get("scale_max"), 0.0)
    if scale_max is None or scale_min is None or scale_max <= scale_min:
        return None
    span = band_x1 - band_x0
    if span <= 0:
        return None
    if scale_type == "log":
        if scale_min <= 0:
            return None
        log_min, log_max = math.log10(scale_min), math.log10(scale_max)
        return lambda x: round(
            10 ** (log_min + (x - band_x0) / span * (log_max - log_min)), 4
        )
    return lambda x: round(scale_min + (x - band_x0) / span * (scale_max - scale_min), 4)


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------


def save_debug_overlay(image_path: str, result: dict, out_dir: Optional[str] = None) -> str:
    """
    Нарисовать поверх изображения все, что нашел анализ, и сохранить в
    F:\\nds\\logs\\analysis_debug\\<stem>_debug.jpg. Возвращает путь к файлу.
    """
    import cv2
    import numpy as np  # noqa: F401  (cv2 требует numpy)

    image = _cv2_read(image_path)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")
    h, w = image.shape[:2]

    layout = result.get("layout", {})
    for x in layout.get("vertical_grid_lines_px", []) or []:
        cv2.line(image, (int(x), 0), (int(x), h), (0, 200, 0), 1)
    for y in layout.get("horizontal_grid_lines_px", []) or []:
        cv2.line(image, (0, int(y)), (w, int(y)), (200, 200, 0), 1)

    traces_layer = result.get("analysis_layers", {}).get("traces", {})
    for bx0, bx1 in traces_layer.get("bands_px", []) or []:
        cv2.rectangle(image, (int(bx0), 0), (int(bx1), h - 1), (0, 140, 255), 2)

    for block in _normalized_ocr_blocks(result):
        x, y, bw, bh = block["bbox_px"]
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 220, 220), 1)

    trace_colors = [(255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 128, 255)]
    for i, curve in enumerate(result.get("curves", [])):
        trace = curve.get("trace") or {}
        points = trace.get("points_px") or []
        if len(points) >= 2:
            color = trace_colors[i % len(trace_colors)]
            poly = np.array([[int(p[1]), int(p[0])] for p in points], dtype=np.int32)
            cv2.polylines(image, [poly], isClosed=False, color=color, thickness=2)
            label_pos = (int(points[0][1]), max(30, int(points[0][0]) - 10))
            cv2.putText(image, curve.get("name", f"C{i}"), label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            # Границы фактических данных кривой поперек ее полосы.
            band = trace.get("band_px") or [0, w]
            for key_px, key_depth in (("data_top_px", "depth_start"),
                                      ("data_bottom_px", "depth_end")):
                y_px = trace.get(key_px)
                if y_px is None:
                    continue
                cv2.line(image, (int(band[0]), int(y_px)), (int(band[1]), int(y_px)), color, 3)
                depth = curve.get(key_depth)
                if depth is not None:
                    cv2.putText(image, f"{depth}", (int(band[0]) + 4, int(y_px) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cal = result.get("pixel_calibration", {})
    for lab in cal.get("depth_labels", []) or []:
        x, y = int(lab["x_px"]), int(lab["y_px"])
        bw, bh = int(lab["w_px"]), int(lab["h_px"])
        color = (0, 0, 255) if lab.get("used") else (180, 0, 255)
        cv2.rectangle(image, (x, y), (x + bw, y + bh), color, 2)
        caption = lab.get("text") or ("?" if lab.get("depth") is None else str(lab["depth"]))
        cv2.putText(image, caption, (x + bw + 6, y + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for key_px, key_depth in (("pixel_top", "depth_top"), ("pixel_bottom", "depth_bottom")):
        y_px = cal.get(key_px)
        if y_px is None:
            continue
        cv2.line(image, (0, int(y_px)), (w, int(y_px)), (255, 0, 255), 2)
        depth = cal.get(key_depth)
        caption = f"{key_px}={y_px}" + (f" depth={depth}" if depth is not None else "")
        cv2.putText(image, caption, (10, max(25, int(y_px) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

    out_dir = out_dir or DEBUG_OVERLAY_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = str(Path(out_dir) / (Path(image_path).stem + "_debug.jpg"))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    encoded.tofile(out_path)
    log.info("debug overlay: %s", out_path)
    return out_path


def _analyze_with_lm_studio(
    image_path: str,
    base_url: str,
    model: Optional[str],
    max_tokens: int,
    timeout: float,
) -> dict:
    base_url = base_url.rstrip("/")
    try:
        model_id = model or _first_lm_studio_model(base_url=base_url, timeout=min(timeout, 2.0))
        if not model_id:
            return {"source": "lm_studio", "status": "unavailable", "error": "no loaded models"}

        data_url = _image_to_data_url_for_vlm(image_path)
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": LOCAL_VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        response = _http_json(
            url=f"{base_url}/chat/completions",
            method="POST",
            payload=payload,
            timeout=timeout,
        )
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = _parse_json_response(content)
        return {
            "source": "lm_studio",
            "status": "ok",
            "model": model_id,
            "raw_text": content,
            "result": parsed,
        }
    except Exception as e:
        return {"source": "lm_studio", "status": "unavailable", "error": str(e)}


def _first_lm_studio_model(base_url: str, timeout: float) -> Optional[str]:
    try:
        response = _http_json(url=f"{base_url}/models", method="GET", payload=None, timeout=timeout)
        models = response.get("data") or []
        if models:
            return models[0].get("id")
    except Exception as e:
        log.debug("LM Studio /models failed: %s", e)
    return None


def _http_json(url: str, method: str, payload: Optional[dict], timeout: float) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw)
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url}: {e}") from e


def _image_to_data_url_for_vlm(image_path: str, max_side: int = 1800) -> str:
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _compact_lm_layer(lm_data: dict) -> dict:
    return {
        "source": lm_data.get("source", "lm_studio"),
        "status": lm_data.get("status"),
        "model": lm_data.get("model", ""),
        "error": lm_data.get("error", ""),
    }


def _merge_model_layer(result: dict, model_result: dict) -> None:
    depth_from = _to_float(model_result.get("depth_from"))
    depth_to = _to_float(model_result.get("depth_to"))
    if depth_from and depth_to and not (result.get("depth_from") and result.get("depth_to")):
        if depth_from > depth_to:
            depth_from, depth_to = depth_to, depth_from
        result["depth_from"] = depth_from
        result["depth_to"] = depth_to

    if model_result.get("depth_unit"):
        result["depth_unit"] = str(model_result["depth_unit"])

    if model_result.get("num_tracks") and not result.get("num_tracks"):
        result["num_tracks"] = int(model_result["num_tracks"])

    for raw_curve in model_result.get("curves", []) or []:
        curve = _normalize_curve(raw_curve, source="lm_studio")
        if curve:
            _upsert_curve(result, curve)


def _normalize_curve(raw: dict, source: str) -> Optional[dict]:
    name = _normalize_curve_name(raw.get("name", ""))
    if not name:
        return None
    defaults = CURVE_DEFAULTS.get(name, {})
    return {
        "name": name,
        "unit": _normalize_unit(raw.get("unit") or defaults.get("unit", "")),
        "color": str(raw.get("color") or ""),
        "scale_min": _to_float(raw.get("scale_min"), defaults.get("scale_min", 0.0)),
        "scale_max": _to_float(raw.get("scale_max"), defaults.get("scale_max", 0.0)),
        "scale_type": str(raw.get("scale_type") or defaults.get("scale_type", "linear")).lower(),
        "source": source,
        "confidence": _to_float(raw.get("confidence"), 0.65),
    }


def _upsert_curve(result: dict, curve: dict) -> None:
    existing = None
    for item in result.get("curves", []):
        if item.get("name") == curve.get("name"):
            existing = item
            break
    if existing is None:
        result.setdefault("curves", []).append(curve)
        return

    incoming_conf = float(curve.get("confidence") or 0.0)
    existing_conf = float(existing.get("confidence") or 0.0)
    for key in ("unit", "color", "scale_min", "scale_max", "scale_type"):
        value = curve.get(key)
        if value not in ("", None, 0.0) and (incoming_conf >= existing_conf or existing.get(key) in ("", None, 0.0)):
            existing[key] = value
    if incoming_conf > existing_conf:
        existing["source"] = curve.get("source", existing.get("source"))
        existing["confidence"] = incoming_conf


def _estimate_result_confidence(result: dict) -> float:
    score = 0.0
    if result.get("depth_from") and result.get("depth_to"):
        score += 0.20
    if result.get("curves"):
        score += 0.20
        score += min(0.20, max(float(c.get("confidence") or 0.0) for c in result["curves"]) * 0.20)
    if result.get("layout", {}).get("status") == "ok":
        score += 0.10
    if result.get("ocr", {}).get("status") == "ok":
        score += 0.05
    if result.get("analysis_layers", {}).get("lm_studio", {}).get("status") == "ok":
        score += 0.10
    cal_status = result.get("pixel_calibration", {}).get("status")
    if cal_status == "ok":
        score += 0.15
    elif cal_status == "approx":
        score += 0.08
    if result.get("analysis_layers", {}).get("traces", {}).get("status") == "ok":
        score += 0.05
    return round(min(score, 1.0), 3)


def _analyze_anthropic_legacy(
    image_path: str,
    api_key: Optional[str],
    model: str,
    max_tokens: int,
    fallback_from_filename: bool,
) -> dict:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Для provider=anthropic установи anthropic: pip install anthropic") from e

    ext = Path(image_path).suffix.lower()
    media_type = SUPPORTED_EXTENSIONS[ext]
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": ANTHROPIC_LEGACY_PROMPT},
                    ],
                }
            ],
        )
        raw_text = response.content[0].text.strip()
        result = _parse_json_response(raw_text)
        result["source"] = "anthropic"
        result["image_path"] = image_path
        return result
    except Exception as e:
        log.warning("Anthropic legacy analysis failed: %s", e)
        if fallback_from_filename:
            result = _empty_result(image_path)
            _merge_filename_layer(result, _parse_filename_metadata(image_path))
            result["source"] = "filename_fallback"
            return result
        raise


def _parse_json_response(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Не удалось извлечь JSON из ответа: {text[:200]}")
    return json.loads(match.group())


def _normalize_curve_name(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-ZА-Я0-9]+", "", text)
    aliases = {
        "ГК": "GK",
        "ПС": "SP",
        "КС": "BK",
        "БК": "BK",
        "МБК": "MBK",
    }
    return aliases.get(text, text)


def _normalize_unit(value: Any) -> str:
    text = str(value or "").strip()
    low = text.lower().replace(" ", "")
    if low in {"ohm", "ohmm", "ohm.m", "ohm-m", "ohm*m", "омм", "ом.м"}:
        return "Ohmm"
    if low in {"mv", "milliv", "millivolt", "мв"}:
        return "mV"
    if low in {"gapi", "api"}:
        return "API"
    return text


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Локальный анализ изображения каротажа")
    parser.add_argument("image", help="Путь к изображению (.tif/.jpg/.png)")
    parser.add_argument("--no-cache", action="store_true", help="Игнорировать кеш")
    parser.add_argument("--provider", choices=["local", "lm_studio", "anthropic"], default="local")
    parser.add_argument("--model", default=None, help="ID модели LM Studio или Anthropic")
    parser.add_argument("--api-key", default=None, help="Только для provider=anthropic")
    parser.add_argument("--no-lm-studio", action="store_true", help="Не обращаться к LM Studio")
    parser.add_argument("--lm-studio-url", default=None, help="Например http://localhost:1234/v1")
    parser.add_argument(
        "--ocr-engine",
        default="auto",
        choices=["auto", "none", "pytesseract", "tesseract", "tesseract_cli", "paddleocr", "easyocr"],
    )
    parser.add_argument("--no-traces", action="store_true", help="Не извлекать полилинии кривых")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help=f"Сохранить визуализацию найденного в {DEBUG_OVERLAY_DIR}",
    )
    parser.add_argument("--debug-dir", default=None, help="Каталог для debug overlay")
    args = parser.parse_args()

    result = analyze_log_image_cached(
        args.image,
        force_refresh=args.no_cache,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        use_lm_studio=not args.no_lm_studio,
        lm_studio_url=args.lm_studio_url,
        ocr_engine=args.ocr_engine,
        extract_traces=not args.no_traces,
    )

    if args.debug_overlay:
        # Отдельно от analyze: overlay рисуется и для результата из кеша.
        try:
            path = save_debug_overlay(args.image, result, out_dir=args.debug_dir)
            print(f"debug overlay: {path}", file=sys.stderr)
        except Exception as e:
            print(f"debug overlay failed: {e}", file=sys.stderr)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
