"""
analyze_log_image.py — Анализ изображений каротажных кривых через Claude Vision API
====================================================================================
Передаёт сканированное изображение в Claude и извлекает метаданные:
кривые, единицы, шкалы, диапазон глубин.

Использование:
    from analyze_log_image import analyze_log_image, analyze_log_image_cached

    # Один раз на изображение (без кеша):
    result = analyze_log_image(r"F:\\nds\\projects\\Well_A\\img\\Well_BK_3080_3520.tif")

    # С кешем (не повторять API-запрос при следующем запуске):
    result = analyze_log_image_cached(r"F:\\nds\\...\\Well_BK_3080_3520.tif")

    # Структура ответа:
    {
      "curves": [
        {"name":"BK",  "unit":"Ohmm", "color":"black", "scale_min":0, "scale_max":20},
        {"name":"MBK", "unit":"Ohmm", "color":"red",   "scale_min":0, "scale_max":5}
      ],
      "depth_from": 3080.0,
      "depth_to":   3520.0,
      "depth_unit": "m",
      "num_tracks": 2,
      "source":     "vision"   # "vision" | "filename_fallback" | "failed"
    }

Командная строка (диагностика):
    python analyze_log_image.py path\\to\\image.tif
    python analyze_log_image.py path\\to\\image.tif --no-cache
"""

import os
import sys
import json
import re
import base64
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff", ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}

VISION_PROMPT = """\
You are analyzing a scanned well log (borehole log) image — a scientific chart showing \
physical measurements vs depth in a borehole.

Extract all information and return ONLY a valid JSON object with NO markdown fences, \
NO explanatory text — just the raw JSON:

{
  "curves": [
    {
      "name":       "BK",      // mnemonic/abbreviation from track header (e.g. BK, MBK, GK, SP, RT)
      "unit":       "Ohmm",    // unit shown in header (Ohmm, mV, GAPI, API, %, ohm.m, MV, etc.)
      "color":      "black",   // approximate line color (black, red, blue, green, brown, dashed)
      "scale_min":  0.0,       // numeric value at left/min edge of this track
      "scale_max":  20.0,      // numeric value at right/max edge of this track
      "scale_type": "linear"   // "linear" or "log" (logarithmic)
    }
  ],
  "depth_from": 3080.0,   // shallowest (top) depth from the depth scale
  "depth_to":   3520.0,   // deepest (bottom) depth from the depth scale
  "depth_unit": "m",      // "m" (metres) or "ft" (feet)
  "num_tracks": 2          // count of vertical track panels in the image
}

Rules:
- Include ALL visible curves even if header text is unclear — make a best guess
- depth_from is always LESS than depth_to (top to bottom)
- If depth scale is not visible, use 0 for both values
- Normalize units: Ohm → Ohmm, MILLIV → mV, ohm-m → Ohmm
- If a track has linear scale 0→20 and logarithmic ticks, mark scale_type "log"
"""


def analyze_log_image(
    image_path: str,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-5",
    fallback_from_filename: bool = True,
    max_tokens: int = 1200,
) -> dict:
    """
    Анализировать изображение каротажного графика через Claude Vision.

    Параметры
    ---------
    image_path            : путь к изображению (.tif/.jpg/.png/.bmp)
    api_key               : ключ Anthropic API; None → переменная окружения
    model                 : модель (рекомендуется claude-opus-4-5 или claude-sonnet-*)
    fallback_from_filename: при ошибке извлечь глубины из имени файла
    max_tokens            : лимит токенов ответа

    Возвращает dict (см. docstring модуля).
    """
    image_path = str(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    ext = Path(image_path).suffix.lower()
    media_type = SUPPORTED_EXTENSIONS.get(ext)
    if not media_type:
        raise ValueError(
            f"Неподдерживаемый формат '{ext}'. "
            f"Допустимые: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Загрузить изображение
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    file_size_kb = os.path.getsize(image_path) / 1024
    log.info(f"analyze_log_image: {Path(image_path).name}  ({file_size_kb:.0f} KB)")

    try:
        import anthropic
    except ImportError:
        raise ImportError("Установи anthropic: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
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
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
        )

        raw_text = response.content[0].text.strip()
        result = _parse_json_response(raw_text)
        result["source"]     = "vision"
        result["image_path"] = image_path

        n_curves = len(result.get("curves", []))
        log.info(
            f"  ✓ Кривых: {n_curves}  "
            f"глубины: {result.get('depth_from')}→{result.get('depth_to')} "
            f"{result.get('depth_unit', 'm')}"
        )
        return result

    except Exception as e:
        log.warning(f"analyze_log_image: ошибка API — {e}")

    # ── Fallback: имя файла ───────────────────────────────────────
    if fallback_from_filename:
        return _fallback_from_filename(image_path)

    return _empty_result(image_path)


def _parse_json_response(text: str) -> dict:
    """Распарсить JSON из ответа модели, снять markdown-обёртку если есть."""
    # Убрать code-fence ```json ... ```
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # Попробовать прямой разбор
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Попробовать найти JSON-объект внутри текста
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())

    raise ValueError(f"Не удалось извлечь JSON из ответа: {text[:200]}")


def _fallback_from_filename(image_path: str) -> dict:
    """Извлечь диапазон глубин из паттерна имени файла: _3080_3520_200_."""
    name = Path(image_path).stem
    m = re.search(r"_(\d{3,5})_(\d{3,5})_(\d{2,4})", name)
    if m:
        result = _empty_result(image_path)
        result.update({
            "depth_from": float(m.group(1)),
            "depth_to":   float(m.group(2)),
            "source":     "filename_fallback",
        })
        log.info(
            f"  Fallback из имени файла: "
            f"{result['depth_from']}→{result['depth_to']} m"
        )
        return result
    return _empty_result(image_path)


def _empty_result(image_path: str) -> dict:
    return {
        "curves":     [],
        "depth_from": 0.0,
        "depth_to":   0.0,
        "depth_unit": "m",
        "num_tracks": 0,
        "source":     "failed",
        "image_path": image_path,
    }


# ─────────────────────────────────────────────────────────────────
# КЕШИРОВАННАЯ ВЕРСИЯ
# ─────────────────────────────────────────────────────────────────

def analyze_log_image_cached(
    image_path: str,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
    **kwargs,
) -> dict:
    """
    Анализ с кешированием: JSON сохраняется рядом с изображением.
    При повторном вызове возвращает кеш без обращения к API.

    Параметры
    ---------
    cache_dir     : директория кеша (по умолчанию — папка изображения)
    force_refresh : проигнорировать кеш и перезапросить API
    **kwargs      : передаются в analyze_log_image()
    """
    path      = Path(image_path)
    cache_dir = cache_dir or str(path.parent)
    cache_path = Path(cache_dir) / (path.stem + "_analysis.json")

    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            log.info(f"analyze_log_image: кеш: {cache_path.name}")
            return data
        except Exception:
            log.warning(f"  Кеш повреждён, перезапрос...")

    result = analyze_log_image(image_path, **kwargs)

    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.debug(f"  Кеш сохранён: {cache_path}")
    except Exception as e:
        log.warning(f"  Не удалось сохранить кеш: {e}")

    return result


# ─────────────────────────────────────────────────────────────────
# ПАКЕТНЫЙ АНАЛИЗ
# ─────────────────────────────────────────────────────────────────

def analyze_batch(
    image_paths: list,
    cache: bool = True,
    **kwargs,
) -> list:
    """
    Анализ списка изображений. Возвращает список dict.
    По умолчанию кеширует результаты.
    """
    fn = analyze_log_image_cached if cache else analyze_log_image
    results = []
    total = len(image_paths)
    for i, p in enumerate(image_paths, 1):
        log.info(f"[{i}/{total}] {Path(p).name}")
        try:
            r = fn(str(p), **kwargs)
        except Exception as e:
            log.error(f"  Ошибка: {e}")
            r = _empty_result(str(p))
        results.append(r)
    ok  = sum(1 for r in results if r.get("source") == "vision")
    fb  = sum(1 for r in results if r.get("source") == "filename_fallback")
    err = sum(1 for r in results if r.get("source") == "failed")
    log.info(f"Анализ завершён: vision={ok}  fallback={fb}  failed={err}")
    return results


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Анализ изображения каротажного графика через Claude Vision API"
    )
    parser.add_argument("image", help="Путь к изображению (.tif/.jpg/.png)")
    parser.add_argument("--no-cache",  action="store_true", help="Игнорировать кеш")
    parser.add_argument("--model",     default="claude-opus-4-5")
    parser.add_argument("--api-key",   default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Файл не найден: {args.image}")
        sys.exit(1)

    result = analyze_log_image_cached(
        args.image,
        force_refresh=args.no_cache,
        model=args.model,
        api_key=args.api_key,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("curves"):
        print(f"\nКривых: {len(result['curves'])}")
        for c in result["curves"]:
            print(
                f"  {c['name']:6s}  {c['unit']:8s}  "
                f"{c.get('scale_min', '?')}..{c.get('scale_max', '?')}  "
                f"{c.get('color', '')}"
            )
