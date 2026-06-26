r"""
confidence.py — U2: карта AUTO/FLAG по линиям U1. Реализует durable-решение проекта
(PLAN §6.6.10/§6.6.11): НЕ множить эвристики мультилинии — целиться в РАЗРЕШИМОЕ (там ~1px),
а информационный ПОЛ честно отдавать эксперту (он его и так делает руками).

AUTO (разрешимо автоматом → 2D-обход штриха):
  • одиночная / разнесённая (своя x-полоса с зазором);
  • цвето-уникальная (красная SP, зелёная GZ — отделяется каналом);
  • гладкая (SP) без перекрытия.
FLAG (эксперту, точечно — клик на узел, НЕ обводка):
  • СБИТЫЙ одноцветный ПУЧОК пиковых, пересекающихся в узкой x-полосе (BKZ GZ, микрозонды) —
    на пересечении нет ни цвета, ни толщины; кривизна = случайность (замер §6.6.11);
  • очень ВЫЦВЕТШАЯ (плотность ниже пола, recall-модель не подняла);
  • МУЛЬТИ-ОБОРОТ ×25 (n_levels_est большой) — трасса-only DP не берёт.

Заполняет L.confidence ∈ {'AUTO','FLAG'} и L.flag_reason.
"""
from dataclasses import dataclass


@dataclass
class ConfParams:
    bunch_overlap_px: float = 10.0     # перекрытие x-полос одноцветных пиковых = сбитый пучок
    wide_band_factor: float = 6.0      # x_band > factor×толщина при пиковости = схлопнутый пучок
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
            if L.density < cp.faint_rel_density * med:
                reason = "faint"                          # выцветшая ниже пола плотности
            elif id(L) in bunched:
                reason = "bunched_crossing"               # сбитый одноцветный пучок
            elif L.behavior == "peaky" and L.thickness and L.x_band > cp.wide_band_factor * L.thickness \
                    and L.n_strokes >= 3:
                reason = "bunched_crossing"               # схлопнутый пучок (1 широкая «multi»)
            elif (L.n_levels_est or 0) >= cp.multiwrap_levels:
                reason = "multiwrap"
            if reason:
                L.confidence = "FLAG"; L.flag_reason = reason; n_flag += 1
            else:
                L.confidence = "AUTO"; L.flag_reason = None; n_auto += 1

    sheet.diag["confidence"] = {"auto": n_auto, "flag": n_flag,
                                "auto_pct": round(100 * n_auto / max(1, n_auto + n_flag))}
    return sheet
