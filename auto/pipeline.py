r"""
pipeline.py — ЕДИНЫЙ вход автономного image-first векторизатора (курс v2, ../ROADMAP_v2.md).

  python -m auto.pipeline <скан.jpg> [--frame рамка.nlgx] [--ruler] [--las] [--out DIR]

Конвейер (всё ОТ ИЗОБРАЖЕНИЯ, без экспертной трассы):
  load → meta(имя файла) → U0 frame → U1 understand → [A1 ruler] → U2 confidence → 2D-обход → emit.

Артефакты в --out: <stem>_understanding.json (СКОЛЬКО линий/переходов, рамка, AUTO/FLAG),
<stem>_overlay.png (AUTO-трассы цветом, FLAG-зоны жёлтым), и при --frame — <stem>_auto.nlgx(+bck)
[+las]. Рамка (--frame) = ЛЁГКИЙ каркас NeuraLOG (Depth/Scale Axis + слоты), БЕЗ трассы;
фабрикацию рамки с нуля НЕ делаем (крашит NeuraLOG, §6.6.10). Без --frame — только понимание+overlay.
"""
import sys
import argparse
from pathlib import Path

from . import imaging
from . import meta as meta_mod
from . import frame as frame_mod
from . import understand as understand_mod
from . import confidence as confidence_mod
from . import trace2d
from . import scales as scales_mod
from . import emit as emit_mod
from .config import Config


def run(image_path, frame_nlgx=None, cfg=None, read_ruler=False, las=False,
        prob_npy=None, stages=False):
    cfg = cfg or Config()
    if prob_npy:                                    # подключить recall-модель (сохранённая prob-карта)
        from . import prob as prob_mod
        prob_mod.attach_npy(cfg, prob_npy)
    rgb = imaging.load_rgb(image_path)
    m = meta_mod.parse_filename(image_path, cfg.mnemonics)
    # Рамка из ШАБЛОНА, если дан --frame (калибровка точнее авто-детекта, см. frame_from_nlgx),
    # иначе автономный U0 из картинки.
    if frame_nlgx:
        fr = frame_mod.frame_from_nlgx(frame_nlgx, m, cfg.cv, rgb=rgb)
    else:
        fr = frame_mod.detect_frame(rgb, m, cfg.cv)
    if getattr(fr, "row_shift", None) is not None:  # дрейфующая лента (косой скан) — выпрямляем,
        rgb = frame_mod.apply_row_shift(rgb, fr.row_shift)  # дальше ВСЁ в выпрямленных координатах
    prob = cfg.prob_provider(rgb) if cfg.prob_provider else None   # recall-модель, если задана
    sheet = understand_mod.understand(rgb, fr, m, cfg.cv, prob=prob)
    if read_ruler:
        scales_mod.count_levels(rgb, sheet, cfg)
    confidence_mod.classify(sheet)
    # ★ §6.204: число слотов раскладки ПО ТРЕКУ — до ведения, чтобы декодер не был ограничен
    # числом линий U1. Считается тем же `emit._slot_track`, что и сама раскладка, по тем же
    # кривым шаблона (без DA — как в пуловых дампах). Без шаблона считать не по чему — ручка
    # молча не действует (`sheet.k_slots` остаётся пустым, `rowdec` ведёт по линиям U1).
    sheet.k_slots = {}
    if getattr(cfg.cv, "rowdec_k_slots", False) and frame_nlgx:
        from extract_nlgx import extract as _extract
        _model = _extract(str(frame_nlgx))
        for c in _model.get("curves", []):
            if meta_mod.mnem_root(c.get("name", "")) == "DA":
                continue
            ti = emit_mod._slot_track(_model, c, sheet.frame)
            sheet.k_slots[ti] = sheet.k_slots.get(ti, 0) + 1
    traces = trace2d.trace_auto(rgb, sheet, cfg.cv)
    stem = Path(image_path).stem
    res = emit_mod.emit(sheet, traces, cfg.ensure_out(), stem, rgb=rgb,
                        frame_nlgx=frame_nlgx, mnemonics_path=cfg.mnemonics,
                        image=image_path, las=las, cv=cfg.cv)
    if stages is not False:                         # поэтапный монтаж «как скрипт видит»
        from . import stages as stages_mod
        window = None if stages in (True, "auto") else stages
        spath, _ = stages_mod.render_stages(rgb, sheet, traces, cfg.cv,
                                            cfg.ensure_out(), stem, window=window, prob=prob)
        res["stages"] = spath
    return sheet, traces, res


def _print_summary(sheet, res):
    m, f = sheet.meta, sheet.frame
    print(f"\n=== ПОНИМАНИЕ: {getattr(m, 'raw_stem', '')} ===")
    print(f"  скважина {getattr(m,'well',None)} | кривые {getattr(m,'curves_token',None)} "
          f"| глубина {getattr(m,'top_depth',None)}..{getattr(m,'bottom_depth',None)} "
          f"| масштаб 1:{getattr(m,'scale',None)}")
    print(f"  ожидаемые кривые (из имени): {getattr(m,'expected_curves',[])}")
    ppm = f.px_per_m
    print(f"  РАМКА: треков={len(f.tracks)} верх/низ y={f.top_y}..{f.bottom_y} "
          f"px/м={ppm:.2f}" if ppm else f"  РАМКА: треков={len(f.tracks)} y={f.top_y}..{f.bottom_y}")
    for t in f.tracks:
        print(f"     трек {t.index}: x[{t.x_left}..{t.x_right}] линий={sheet.per_track.get(t.index,0)}")
    conf = sheet.diag.get("confidence", {})
    print(f"  ЛИНИЙ всего={len(sheet.lines)}  AUTO={conf.get('auto',0)} "
          f"FLAG={conf.get('flag',0)} ({conf.get('auto_pct',0)}% авто)")
    for L in sheet.lines:
        tag = L.confidence or "?"
        extra = f" levels~{L.n_levels_est}" if L.n_levels_est else ""
        fl = f" [{L.flag_reason}]" if L.flag_reason else ""
        print(f"     t{L.track_index} {L.color:<5} x≈{L.x_center:6.0f} полоса={L.x_band:4.0f} "
              f"{L.behavior:<6} тол={L.thickness:.1f} штр={L.n_strokes} {tag}{fl}{extra}")
    print(f"  АРТЕФАКТЫ: {', '.join(f'{k}={v}' for k,v in res.items() if not isinstance(v,list))}")


def main():
    ap = argparse.ArgumentParser(description="Автономная image-first векторизация каротажа (v2)")
    ap.add_argument("image", help="скан планшета (.jpg/.tif/.png)")
    ap.add_argument("--frame", help="лёгкая рамка NeuraLOG (.nlgx) для nlgx-выдачи (опц.)")
    ap.add_argument("--ruler", action="store_true", help="прочитать линейку через A1/VLM (LM Studio)")
    ap.add_argument("--las", action="store_true", help="достроить .las (нужна --frame)")
    ap.add_argument("--prob", help="prob-карта recall-модели (.npy) — анализ ИСПОЛЬЗУЕТ модель (gate бледных)")
    ap.add_argument("--stages", nargs="?", const="auto", default=False,
                    help="дамп поэтапного монтажа «как скрипт видит» (опц. Y0:Y1, иначе авто-окно)")
    ap.add_argument("--out", help="папка вывода (иначе CARROTAGE_OUT/ ./output)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    if not Path(a.image).is_file():
        print(f"нет файла: {a.image}"); return 1
    cfg = Config()
    if a.out:
        cfg.out = Path(a.out)
    stages = a.stages
    if isinstance(stages, str) and ":" in stages:
        y0, y1 = stages.split(":"); stages = (int(y0), int(y1))
    sheet, traces, res = run(a.image, frame_nlgx=a.frame, cfg=cfg,
                             read_ruler=a.ruler, las=a.las, prob_npy=a.prob, stages=stages)
    _print_summary(sheet, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
