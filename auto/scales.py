r"""
scales.py — счёт ПЕРЕХОДОВ МАСШТАБА (durable §6.6.9: считать ЧИСЛО, не читать величины).
Заказчик: важно ПОПАСТЬ В КОЛИЧЕСТВО переходов, величины эксперт правит при QC.

A1 (digitizer/detect_scales): кроп масштабной линейки над top_y → VLM (LM Studio) → строки
чисел. Число строк ОМ·М = число уровней резистив-цепочки (база + ×5 + ×25…). Здесь — тонкая
автономная обёртка: считает уровни и проставляет n_levels_est на РЕЗИСТИВНЫЕ линии листа.

LM Studio ОПЦИОНАЛЕН: при недоступности возвращаем None и n_levels_est не ставим (U2 тогда не
флагает multiwrap; счёт уточнит эксперт). НЕ роняем пайплайн из-за сети.
"""


def read_ruler_levels(rgb, frame, cfg=None, model=None):
    """Прочитать линейку через A1. Возвращает inferred-структуру (resistivity_chains/caliper/sp)
    или None, если VLM недоступен. cfg.lmstudio_base переопределяет хардкод BASE в detect_scales."""
    try:
        import detect_scales as A1
    except Exception:
        return None
    if cfg is not None:
        A1.BASE = cfg.lmstudio_base
        model = model or "google/gemma-4-26b-a4b"
    try:
        _, rows, _ = A1.read_ruler(rgb, frame.top_y, model or "google/gemma-4-26b-a4b")
        if not rows:
            return None
        return A1.infer_scales(rows)
    except Exception:
        return None


def annotate_levels(sheet, ruler):
    """Проставить n_levels_est по строкам линейки: резистив-цепочка → число уровней на РЕЗИСТИВНЫЕ
    линии; SP/CALI = 1. Без ruler — ничего не трогаем (оценка останется None)."""
    if not ruler:
        return sheet
    chains = ruler.get("resistivity_chains") or []
    res_levels = max((ch.get("levels", 1) for ch in chains), default=None)
    for L in sheet.lines:
        # класс линии оцениваем по поведению (RES/CALI пиковые, SP гладкая) — мнемоники тут нет
        if L.behavior == "smooth":
            L.n_levels_est = 1
        elif res_levels is not None:
            L.n_levels_est = res_levels
    sheet.diag["ruler"] = {"resistivity_levels": res_levels,
                           "caliper": ruler.get("caliper"), "sp": ruler.get("sp")}
    return sheet


def count_levels(rgb, sheet, cfg=None, model=None):
    """Удобная связка: прочитать линейку (A1) и проставить уровни на линии листа."""
    ruler = read_ruler_levels(rgb, sheet.frame, cfg, model)
    return annotate_levels(sheet, ruler)
