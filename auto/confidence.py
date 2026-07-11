r"""
confidence.py — U2: карта AUTO/FLAG по линиям U1. Реализует durable-решение проекта
(PLAN §6.6.10/§6.6.11): НЕ множить эвристики мультилинии — целиться в РАЗРЕШИМОЕ (там ~1px),
а информационный ПОЛ честно отдавать эксперту (он его и так делает руками).

AUTO (разрешимо автоматом → 2D-обход штриха):
  • одиночная / разнесённая (своя x-полоса с зазором);
  • цвето-уникальная (красная SP, зелёная GZ — отделяется каналом);
  • гладкая (SP) без перекрытия.
FLAG (эксперту, точечно — клик на узел, НЕ обводка):
  • СБИТЫЙ ПУЧОК пиковых, пересекающихся в полосе (BKZ GZ, микрозонды): на пересечении нет ни
    цвета, ни толщины, но ИНК-РАНОВ на строке ≥2 (мультипликативность, §6.6.11). ОДИНОЧНАЯ широкая
    (DT акустики — band велик ОТ РАЗМАХА, но 1 штрих/строку) даёт ~1 ран → AUTO, НЕ пучок;
  • очень ВЫЦВЕТШАЯ (плотность ниже пола, recall-модель не подняла);
  • МУЛЬТИ-ОБОРОТ ×25 (n_levels_est большой) — трасса-only DP не берёт.

Заполняет L.confidence ∈ {'AUTO','FLAG'} и L.flag_reason.
"""
from dataclasses import dataclass


@dataclass
class ConfParams:
    bunch_overlap_px: float = 10.0     # перекрытие x-полос РАЗДЕЛЬНЫХ одноцветных пиковых = пучок
    bunch_multiplicity: float = 1.5    # медиана инк-ранов ≥ этого + одноцветный сосед = пучок
    bunch_hard_multiplicity: float = 3.0  # ≥3 ранов/строку = многожильный пучок ДАЖЕ цвето-уникальный
    faint_rel_density: float = 0.18    # плотность < доля медианы трека = выцветшая
    multiwrap_levels: int = 3          # ≥ столько уровней = мульти-оборот → FLAG


def _overlap(a, b, margin):
    """Перекрываются ли x-полосы линий a,b (с допуском margin)."""
    return min(a.x_hi, b.x_hi) - max(a.x_lo, b.x_lo) > -margin


def classify(sheet, cp: ConfParams = None):
    cp = cp or ConfParams()
    lines = sheet.lines
    # медиана плотности по треку — для порога «выцветшая»
    by_track = {}
    for L in lines:
        by_track.setdefault(L.track_index, []).append(L)
    track_med_density = {t: (sorted(g, key=lambda x: x.density)[len(g) // 2].density or 1.0)
                         for t, g in by_track.items()}

    n_auto = n_flag = 0
    for t, group in by_track.items():
        med = track_med_density[t] or 1.0
        color_count = {}                                  # сколько линий каждого цвета в треке
        for L in group:
            color_count[L.color] = color_count.get(L.color, 0) + 1
        # сбитый пучок: одноцветные пиковые с перекрытием x-полос
        bunched = set()
        for color in {L.color for L in group}:
            same = [L for L in group if L.color == color and L.behavior == "peaky"]
            for i in range(len(same)):
                for j in range(i + 1, len(same)):
                    if _overlap(same[i], same[j], cp.bunch_overlap_px):
                        bunched.add(id(same[i])); bunched.add(id(same[j]))
        for L in group:
            reason = None
            # ЦВЕТО-УНИКАЛЬНАЯ линия: цвет ОДНОЗНАЧНО задаёт идентичность (durable: цвето-уникальная
            # → AUTO). n_runs~2 у неё = резкий ЗИГЗАГ одиночной кривой (2 рана на развороте), а НЕ
            # слипание (STK: чёрная PZ / красная SP разделены цветом, трасса 100% on-ink). Реальный
            # СХЛОПНУТЫЙ пучок цвето-уникальным не бывает и/или даёт n_runs≥3.
            color_unique = color_count.get(L.color, 0) == 1
            if L.density < cp.faint_rel_density * med and not color_unique:
                # выцветшая ниже пола плотности — FLAG, НО не для цвето-уникальной: цвет задаёт
                # идентичность, бледная зелёная GZ / оранжевая SP2 всё равно трассируются (STK_4020).
                reason = "faint"
            elif id(L) in bunched:
                reason = "bunched_crossing"               # сбитый одноцветный пучок
            elif ((getattr(L, "n_runs_med", 0) or 0) >= cp.bunch_hard_multiplicity
                  and not color_unique):
                reason = "bunched_crossing"               # ≥3 рана/строку — плотный многожильный пучок
            elif ((getattr(L, "n_runs_med", 0) or 0) >= cp.bunch_multiplicity
                  and not color_unique):
                # СХЛОПНУТЫЙ пучок: ≥~2 инк-рана на строку в полосе И есть одноцветный сосед =
                # физически несколько штрихов одного цвета (идентичность неоднозначна). Одиночная
                # цвето-уникальная с n_runs~2 (резкий зигзаг) → AUTO: color pins identity.
                reason = "bunched_crossing"
            elif (L.n_levels_est or 0) >= cp.multiwrap_levels:
                reason = "multiwrap"
            if reason:
                L.confidence = "FLAG"; L.flag_reason = reason; n_flag += 1
            else:
                L.confidence = "AUTO"; L.flag_reason = None; n_auto += 1

    sheet.diag["confidence"] = {"auto": n_auto, "flag": n_flag,
                                "auto_pct": round(100 * n_auto / max(1, n_auto + n_flag))}
    return sheet
