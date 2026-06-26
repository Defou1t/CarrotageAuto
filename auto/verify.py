r"""
verify.py — НЕЗАВИСИМАЯ проверка результата ПРОТИВ ИЗОБРАЖЕНИЯ. Backbone принципа «эксперт
ТОЛЬКО проверяет»: оцифровщик сам себя оценивает, эксперт смотрит флаги, а не ищет ошибки с нуля.

Обёртка над проверенным digitizer/qc_trace.py (off_ink/spike/gap/oof/overlap + score 0..100).
Работает на сдаваемом _auto.nlgx (нужна ось глубины из рамки). Возвращает per-curve score +
список подозрительных глубин (для клика эксперта) + сводку. НЕ требует эталонной трассы.
"""
from pathlib import Path


def verify_auto(auto_nlgx, image=None):
    """qc_trace на _auto.nlgx → list[dict] (per-curve: score, off_ink, spikes, gap_m, oof, overlap)."""
    import qc_trace
    return qc_trace.qc_file(str(auto_nlgx), image=image, overlay=False, verbose=False)


def summarize(qc_rows):
    """Сводка QC: медиана/мин score, сколько кривых ниже порога «сдаваемо» (90)."""
    if not qc_rows:
        return {"ok": False, "reason": "нет кривых для QC (ось глубины/картинка?)"}
    scores = sorted(q["score"] for q in qc_rows)
    below = [q["name"] for q in qc_rows if q["score"] < 90]
    return {
        "ok": True, "n_curves": len(qc_rows),
        "score_median": scores[len(scores) // 2], "score_min": scores[0],
        "deliverable": [q["name"] for q in qc_rows if q["score"] >= 90],
        "review": below,
        "per_curve": [{"name": q["name"], "color": q.get("color"), "score": q["score"],
                       "off_ink_pct": q["off_ink_pct"], "spikes": q["spikes"],
                       "gap_m": q["gap_m"], "overlap_m": q.get("overlap_m", 0),
                       "suspect_depths": q.get("spike_d", [])} for q in
                      sorted(qc_rows, key=lambda q: q["score"])],
    }


def verify(auto_nlgx, image=None):
    """Удобная связка: проверить + сводка (для UI/CLI)."""
    return summarize(verify_auto(auto_nlgx, image))
