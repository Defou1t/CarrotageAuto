r"""
app.py — диспетчер операций UI: «операция → вызов пайплайна», JSON-безопасный результат.
Тяжёлые импорты (numpy/cv2/torch/pipeline) — ЛЕНИВЫЕ внутри операций, чтобы сервер стартовал
и в окружении без них (тогда операция вернёт понятную ошибку, а не уронит UI).

Операции: test / analyze / vectorize / refine / verify / feedback.
Любая ловит исключение и возвращает {ok:false, error, trace} — UI это покажет.
"""
import traceback
from dataclasses import asdict
from pathlib import Path


def _build_cfg(payload):
    """Config из payload: пути (out/data) + переопределения CV-параметров (для 'уточнения')."""
    from ..config import Config
    cfg = Config()
    if payload.get("out"):
        cfg.out = Path(payload["out"])
    if payload.get("data"):
        cfg.data = Path(payload["data"])
    for k, v in (payload.get("params") or {}).items():
        if hasattr(cfg.cv, k) and v is not None:
            cur = getattr(cfg.cv, k)
            try:
                setattr(cfg.cv, k, type(cur)(v))
            except Exception:
                pass
    if payload.get("ckpt"):
        try:
            from .. import prob
            prob.attach(cfg, payload["ckpt"], device=payload.get("device"))
        except Exception as e:
            cfg._prob_error = repr(e)
    return cfg


def op_test(payload, cfg):
    """Тест отдельной функции (быстрая проверка кусков без полного прогона)."""
    name = payload.get("test", "filename")
    from .. import meta as M
    if name == "filename":
        m = M.parse_filename(payload["image"], cfg.mnemonics)
        return {"result": asdict(m)}
    if name == "curve_info":
        return {"result": M.curve_info(payload.get("name", "SP"), cfg.mnemonics)}
    if name == "frame":
        from .. import imaging, frame as F
        rgb = imaging.load_rgb(payload["image"])
        m = M.parse_filename(payload["image"], cfg.mnemonics)
        fr = F.detect_frame(rgb, m, cfg.cv)
        return {"result": {"tracks": [[t.x_left, t.x_right] for t in fr.tracks],
                           "top_y": fr.top_y, "bottom_y": fr.bottom_y,
                           "px_per_m": fr.px_per_m, "grid_period_px": fr.grid_period_px,
                           "diag": fr.diag}}
    if name == "prob":
        from .. import imaging, prob as P
        prov = P.make_prob_provider(payload["ckpt"], device=payload.get("device"))
        rgb = imaging.load_rgb(payload["image"])
        pm = prov(rgb)
        ov = P.save_prob_overlay(rgb, pm, cfg.ensure_out(), Path(payload["image"]).stem[:40])
        return {"result": {"model": prov.meta,
                           "coverage_pct": round(float((pm > 0.4).mean()) * 100, 2)},
                "overlay": ov}
    return {"result": f"неизвестный тест: {name}"}


def _run_pipeline(payload, cfg, frame_nlgx=None, las=False):
    from .. import pipeline
    sheet, traces, res = pipeline.run(payload["image"], frame_nlgx=frame_nlgx, cfg=cfg,
                                      read_ruler=bool(payload.get("ruler")), las=las)
    return {"understanding": sheet.to_dict(), "artifacts": res,
            "overlay": res.get("overlay"),
            "n_auto": sheet.diag.get("confidence", {}).get("auto"),
            "n_flag": sheet.diag.get("confidence", {}).get("flag")}


def op_analyze(payload, cfg):
    """Анализ изображения: U0→U1→[A1]→U2 → understanding.json + overlay (без nlgx)."""
    return _run_pipeline(payload, cfg, frame_nlgx=None, las=False)


def op_vectorize(payload, cfg):
    """Векторизация: полный конвейер + инъекция в рамку → _auto.nlgx(+bck)[+las]."""
    fr = payload.get("frame") or None
    return _run_pipeline(payload, cfg, frame_nlgx=fr, las=bool(payload.get("las")))


def op_refine(payload, cfg):
    """Уточнение/улучшение: повторный прогон с переопределёнными параметрами (cfg уже их применил)."""
    fr = payload.get("frame") or None
    out = _run_pipeline(payload, cfg, frame_nlgx=fr, las=bool(payload.get("las")))
    out["applied_params"] = payload.get("params") or {}
    return out


def op_verify(payload, cfg):
    """Проверка (QC) сдаваемого _auto.nlgx против изображения (score, флаги для эксперта)."""
    from .. import verify
    return {"result": verify.verify(payload["nlgx"], payload.get("image"))}


def op_feedback(payload, cfg):
    """Обратная связь: исправленный экспертом nlgx → сравнение с авто + QC + опц. интейк в корпус."""
    from .. import feedback
    return {"result": feedback.feedback(
        payload["corrected"], auto_nlgx=payload.get("nlgx"), image=payload.get("image"),
        corpus=str(cfg.corpus), do_ingest=bool(payload.get("ingest")))}


_OPS = {"test": op_test, "analyze": op_analyze, "vectorize": op_vectorize,
        "refine": op_refine, "verify": op_verify, "feedback": op_feedback}


def dispatch(op, payload):
    """Единая точка: выбрать операцию, собрать cfg, выполнить, вернуть JSON-безопасный dict."""
    if op not in _OPS:
        return {"ok": False, "error": f"неизвестная операция: {op}"}
    try:
        cfg = _build_cfg(payload)
        res = _OPS[op](payload, cfg)
        return {"ok": True, "op": op, **res}
    except Exception as e:
        return {"ok": False, "op": op, "error": repr(e), "trace": traceback.format_exc()[-1800:]}
