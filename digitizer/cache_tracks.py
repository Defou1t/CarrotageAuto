r"""
cache_tracks.py — закэшировать выходы v1-трекера на наложенных файлах, чтобы
разрабатывать пост-хок развязку свопов на CPU (без GPU, параллельно с переобучением).
Сохраняет на файл: v1-трассы линий (x по строкам) + GT-трассы кривых + run-центры по
строкам (для возможной пере-сборки на пересечениях).

python cache_tracks.py <ckpt> [--limit 8] [--out DIR]
"""
import sys, pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from infer import load_model, predict_prob
from track_multi import row_runs, track
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
import json

Image.MAX_IMAGE_PIXELS = None
ARCHIVE = Path(r"F:\nds\projects\Archive")


def main():
    ckpt = sys.argv[1]; args = sys.argv[2:]
    limit = 8; out = Path(r"F:\nds\output\unet_data\trackcache")
    data = Path(r"F:\nds\output\unet_data")
    i = 0
    while i < len(args):
        if args[i] == "--limit": limit = int(args[i+1]); i += 2
        elif args[i] == "--out": out = Path(args[i+1]); i += 2
        else: i += 1
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = load_model(ckpt, device)

    # наложенные val-файлы (>=3 реальных кривых) + Semeguniv BKZ/MK для референса
    idx = json.loads((data / "index.json").read_text(encoding="utf-8"))
    val_wells = sorted({r["well"] for r in idx if r["split"] == "val"})
    files = []
    for w in val_wells + ["Semeguniv_020"]:
        wlg = ARCHIVE / w / "wlg"
        if not wlg.is_dir(): continue
        for nlgx in sorted(wlg.glob("*.nlgx")):
            if "_auto" in nlgx.stem: continue
            try:
                m = extract(str(nlgx))
            except Exception:
                continue
            if len(ds.real_curves(m)) >= 3 and find_image(nlgx):
                files.append(nlgx)
    if len(files) > limit:
        sel = np.linspace(0, len(files)-1, limit).round().astype(int)
        files = [files[k] for k in sorted(set(sel))]

    for nlgx in files:
        m = extract(str(nlgx))
        rgb = np.asarray(Image.open(find_image(nlgx)).convert("RGB"))
        H, W, _ = rgb.shape
        da = m["depth_axis"]; y0, y1 = da["top_y"], min(H, da["bottom_y"])
        prob = predict_prob(net, rgb, device, y0=y0, y1=y1)
        lines, N = track(prob, rgb, y0, y1)
        rows = list(range(y0, y1))
        runs = {y: [r[0] for r in row_runs(prob[y])] for y in rows}
        gts = []
        for c in ds.real_curves(m):
            ty = c["top_y"]
            d = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL and y0 <= ty+i < y1}
            if len(d) >= 20:
                gts.append((c["name"].split()[0], d))
        rec = {"stem": nlgx.stem, "y0": y0, "y1": y1, "N": N,
               "lines": [dict(L["tr"]) for L in lines], "gts": gts, "runs": runs}
        pickle.dump(rec, open(out / f"{nlgx.stem[:60]}.pkl", "wb"))
        print(f"cached {nlgx.stem[:46]:<46} N={N} lines={len(lines)} gts={len(gts)}")
    print(f"\n{len(files)} файлов в {out}")


if __name__ == "__main__":
    main()
