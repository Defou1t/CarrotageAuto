r"""
feedback.py — петля «эксперт ИСПРАВИЛ → ПОВТОРНЫЙ анализ» (запрос заказчика: исправленный вариант
отправить обратно для повторного анализа).

Эксперт правит трассу в NeuraLOG → отдаёт исправленный .nlgx. Здесь:
  • compare  — где наш _auto разошёлся с исправленным (медиана |Δx| на общих строках + ТОП-зоны
               расхождения по глубине) → точные места, где автомат ошибся (вход для улучшения);
  • qc       — qc_trace на ИСПРАВЛЕННОМ (контроль, что правка чистая);
  • ingest   — положить исправленный в корпус (priors / будущее дообучение recall-модели).

Это замыкает цикл: проверка → правка → обратно → мы видим расхождение → подстраиваем параметры/модель.
"""
from pathlib import Path


def _trace_xy(curve):
    NULL = 0xFFFFFFFF
    ty = curve["top_y"]
    return {ty + i: x for i, x in enumerate(curve["xs"]) if x != NULL}


def compare(auto_nlgx, corrected_nlgx, top_n=8):
    """Расхождение наш _auto vs исправленный эксперта по кривым (общие строки). Возвращает per-curve
    {median_dx, n_common, worst_depths} — где автомат разошёлся сильнее всего."""
    from extract_nlgx import extract, depth_of, depth_axis_ok
    import dataset as ds
    a = extract(str(auto_nlgx)); b = extract(str(corrected_nlgx))
    da_ok = depth_axis_ok(b)
    bcur = {c["name"].split()[0]: c for c in ds.real_curves(b)}
    out = []
    for c in ds.real_curves(a):
        nm = c["name"].split()[0]
        if nm not in bcur:
            out.append({"curve": nm, "status": "нет в исправленном"}); continue
        ta = _trace_xy(c); tb = _trace_xy(bcur[nm])
        common = sorted(set(ta) & set(tb))
        if len(common) < 20:
            out.append({"curve": nm, "status": "мало общих строк", "n_common": len(common)}); continue
        diffs = [(y, abs(ta[y] - tb[y])) for y in common]
        meddx = sorted(d for _, d in diffs)[len(diffs) // 2]
        worst = sorted(diffs, key=lambda t: -t[1])[:top_n]
        wd = [(round(depth_of(b, y), 1) if da_ok else y, round(dx, 1)) for y, dx in worst]
        out.append({"curve": nm, "median_dx_px": round(meddx, 1), "n_common": len(common),
                    "worst": wd})
    return out


def ingest(corrected_nlgx, corpus, image=None):
    """Положить исправленный эталон (+bck, +las рядом, +img) в корпус для приоров/дообучения."""
    import shutil
    corpus = Path(corpus); (corpus / "wlg").mkdir(parents=True, exist_ok=True)
    (corpus / "las").mkdir(parents=True, exist_ok=True)
    f = Path(corrected_nlgx)
    dst = corpus / "wlg" / f.name
    shutil.copy2(f, dst)
    got = {"nlgx": str(dst)}
    bck = f.with_suffix(".bck")
    if bck.is_file():
        shutil.copy2(bck, corpus / "wlg" / bck.name); got["bck"] = bck.name
    las = f.with_suffix(".las")
    if las.is_file():
        shutil.copy2(las, corpus / "las" / las.name); got["las"] = las.name
    return got


def feedback(corrected_nlgx, auto_nlgx=None, image=None, corpus=None, do_ingest=False):
    """Оркестратор петли: сравнить с авто (если дан) + QC исправленного + опц. интейк в корпус."""
    res = {"corrected": str(corrected_nlgx)}
    if auto_nlgx:
        res["compare"] = compare(auto_nlgx, corrected_nlgx)
    from . import verify as verify_mod
    res["qc_corrected"] = verify_mod.verify(corrected_nlgx, image)
    if do_ingest and corpus:
        res["ingested"] = ingest(corrected_nlgx, corpus, image)
    return res
