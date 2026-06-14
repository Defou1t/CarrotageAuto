r"""
batch_eval.py — итоговая оценка U-Net трассы по VAL-файлам (модель их не видела).
Для каждой кривой: пиксельная ошибка трассы vs эксперт + log_corr vs эталон LAS.
Агрегация по типу трека (single-curve / multi-curve) и по мнемонике — чтобы
видеть, где U-Net силён (одно-кривные) и где нужен v2 (наложенные).

ЗАПУСК venv ComfyUI:
  python batch_eval.py <ckpt> [--limit 30] [--device cuda]
"""
import sys, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from infer import load_model, evaluate
from extract_nlgx import extract
import dataset as ds
from dataset_build import find_image
import json

ARCHIVE = Path(r"F:\nds\projects\Archive")


def main():
    ckpt = sys.argv[1]; args = sys.argv[2:]
    limit = 30; device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(r"F:\nds\output\unet_data")
    i = 0
    while i < len(args):
        if args[i] == "--limit": limit = int(args[i+1]); i += 2
        elif args[i] == "--device": device = args[i+1]; i += 2
        elif args[i] == "--data": data = Path(args[i+1]); i += 2
        else: i += 1
    sys.stdout.reconfigure(line_buffering=True)

    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    val_wells = sorted({r["well"] for r in index if r["split"] == "val"})
    print(f"VAL wells ({len(val_wells)}): {val_wells}")

    files = []
    for w in val_wells:
        wlg = ARCHIVE / w / "wlg"
        if not wlg.is_dir():
            continue
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem:
                continue
            if find_image(nlgx):
                files.append(nlgx)
    # равномерная выборка до limit (разброс по скважинам)
    if len(files) > limit:
        idx = np.linspace(0, len(files)-1, limit).round().astype(int)
        files = [files[k] for k in sorted(set(idx))]
    print(f"eval {len(files)} val files\n")

    net, ck = load_model(ckpt, device)
    print(f"model epoch={ck.get('epoch')} val_dice={ck.get('val_dice'):.4f} device={device}\n")

    agg = defaultdict(lambda: {"px": [], "lc": []})   # key -> metrics
    for nlgx in files:
        try:
            m = extract(str(nlgx))
            ncur = len(ds.real_curves(m))
            ttype = "single" if ncur == 1 else "multi"
            rows = evaluate(net, device, str(nlgx), str(find_image(nlgx)))
        except Exception as e:
            print(f"  ERR {nlgx.stem[:40]}: {type(e).__name__}: {e}"); continue
        for name, px, lc in rows:
            mn = re.sub(r"\d+$", "", name)
            if px and px[0] is not None:
                agg[(ttype, "·all")]["px"].append(px[0])
                agg[(ttype, mn)]["px"].append(px[0])
            if lc is not None:
                agg[(ttype, "·all")]["lc"].append(lc)
                agg[(ttype, mn)]["lc"].append(lc)

    print("\n==================== ИТОГ: U-Net трасса на VAL ====================")
    print(f"{'track':<8}{'mnem':<8}{'n':>4}{'px_med':>9}{'px_p25':>8}{'logcorr_med':>13}")
    for ttype in ("single", "multi"):
        keys = sorted([k for k in agg if k[0] == ttype], key=lambda k: (k[1] != "·all", k[1]))
        for k in keys:
            px = agg[k]["px"]; lc = agg[k]["lc"]
            if not px:
                continue
            print(f"{ttype:<8}{k[1]:<8}{len(px):>4}{np.median(px):>9.1f}"
                  f"{np.percentile(px,25):>8.1f}{(np.median(lc) if lc else float('nan')):>13.3f}")
    print("\nbaseline (variant A, эвристика): single — бимодально (1px либо мимо),"
          " медиана-медиан ~76px, log_corr ~0.72; multi — усреднение ~76px.")


if __name__ == "__main__":
    main()
