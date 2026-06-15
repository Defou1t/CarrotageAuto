r"""
pipeline.py — ЕДИНЫЙ вход нового пайплайна оцифровки. Ничего лишнего: 1 скрипт, 2 режима.

  python pipeline.py process <файл.nlgx | папка> [--grid4m] [--patch-scan] [--out DIR]
      Обработать скан(ы) → сдаваемый _auto.nlgx(+bck). Папка = батч по всем *.nlgx + CSV.

  python pipeline.py learn <папка с проверенными nlgx(+bck)+las>
      Принять проверенные эталоны в КОРПУС (F:\nds\corpus) и обновить priors.json —
      поведенческие приоры (rough_n по классам, SP-порог) + статистику перевыносов.
      Чем больше проверенных эталонов — тем точнее приоры (петля «дообучения» эвристик;
      корпус также готов для переобучения U-Net через prep_dataset).

process не требует U-Net/torch (cv2/py3.14). Маппинг линий к кривым — по трассе шаблона.
"""
import sys, json, shutil, re
from pathlib import Path
from collections import defaultdict
import numpy as np
from extract_nlgx import extract
import dataset as ds
import behavior_priors as bp
import digitize_b3 as d3

CORPUS = Path(r"F:\nds\corpus")
PRIORS = CORPUS / "priors.json"


def cmd_process(args):
    target = args[0]
    out = args[args.index("--out") + 1] if "--out" in args else None
    grid_step = 4.0 if "--grid4m" in args else None
    patch_scan = "--patch-scan" in args
    p = Path(target)
    if p.is_dir():
        files = sorted(f for f in p.glob("*.nlgx") if "_auto" not in f.stem)
        print(f"БАТЧ process: {len(files)} файлов")
        rows = [d3.digitize_one(f, None, out, grid_step, patch_scan) for f in files]
        okn = sum(1 for r in rows if r["dst"])
        cov = [r["cover_pct"] for r in rows if r["cover_pct"] is not None]
        print(f"\nГОТОВО: {okn}/{len(files)}; медиана покрытия {np.median(cov) if cov else '-'}%")
    else:
        image = args[args.index("--image") + 1] if "--image" in args else None
        r = d3.digitize_one(target, image, out, grid_step, patch_scan)
        print(f"-> {r.get('dst')}")


def cmd_learn(args):
    src = Path(args[0])
    if not src.is_dir():
        print(f"не папка: {src}"); return
    (CORPUS / "wlg").mkdir(parents=True, exist_ok=True)
    (CORPUS / "las").mkdir(parents=True, exist_ok=True)
    # 1) интейк: копируем проверенные эталоны в корпус (nlgx+bck+las)
    ing = {"nlgx": 0, "bck": 0, "las": 0}
    for f in src.rglob("*"):
        if f.suffix.lower() == ".nlgx" and "_auto" not in f.stem:
            shutil.copy2(f, CORPUS / "wlg" / f.name); ing["nlgx"] += 1
            bck = f.with_suffix(".bck")
            if bck.is_file():
                shutil.copy2(bck, CORPUS / "wlg" / bck.name); ing["bck"] += 1
        elif f.suffix.lower() == ".las":
            shutil.copy2(f, CORPUS / "las" / f.name); ing["las"] += 1
    print(f"интейк в корпус: nlgx={ing['nlgx']} bck={ing['bck']} las={ing['las']} -> {CORPUS}")

    # 2) рефреш приоров по ВСЕМУ корпусу
    files = [f for f in (CORPUS / "wlg").glob("*.nlgx") if "_auto" not in f.stem]
    by_cls = defaultdict(lambda: {"rough_n": [], "rev": []})
    lvl_hist = defaultdict(int); mult_hist = defaultdict(int); ncurves = 0
    for f in files:
        try:
            m = extract(str(f))
        except Exception:
            continue
        for c in ds.real_curves(m):
            fb = bp.curve_behavior(m, c)
            if fb:
                cl = bp.curve_class(c["name"])
                by_cls[cl]["rough_n"].append(fb["rough_n"]); by_cls[cl]["rev"].append(fb["rev"])
                ncurves += 1
            levels = [l for _, _, l in c["segments"]]
            lvl_hist[max(levels) if levels else 0] += 1
    # SP-порог разделимости (между SP и пиковыми)
    sp = np.array(by_cls["SP"]["rough_n"]) if by_cls["SP"]["rough_n"] else np.array([])
    peaky = np.array(by_cls["RES"]["rough_n"] + by_cls["CALI"]["rough_n"])
    sp_thr = float((np.percentile(sp, 90) + np.percentile(peaky, 10)) / 2) if len(sp) and len(peaky) else None
    priors = {
        "corpus_files": len(files), "corpus_curves": ncurves,
        "behavior": {cl: {"rough_n_med": round(float(np.median(d["rough_n"])), 4),
                          "rev_med": round(float(np.median(d["rev"])), 3), "n": len(d["rough_n"])}
                     for cl, d in by_cls.items() if d["rough_n"]},
        "sp_rough_n_threshold": round(sp_thr, 4) if sp_thr else None,
        "maxlevel_hist": {str(k): v for k, v in sorted(lvl_hist.items())},
    }
    PRIORS.write_text(json.dumps(priors, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"priors обновлён по {len(files)} эталонам ({ncurves} кривых) -> {PRIORS}")
    print(f"  SP rough_n порог={priors['sp_rough_n_threshold']}; "
          f"классы={ {k: v['rough_n_med'] for k, v in priors['behavior'].items()} }")


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    a = sys.argv[1:]
    if not a or a[0] not in ("process", "learn"):
        print(__doc__); return
    (cmd_process if a[0] == "process" else cmd_learn)(a[1:])


if __name__ == "__main__":
    main()
