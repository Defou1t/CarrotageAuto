r"""
decode_e2e.py — СКВОЗНОЙ тест автономных ЗНАЧЕНИЙ: U-Net трасса (НАША, не эксперта) →
декодер уровней перевыносов (DP) → реконструкция value vs эталонный LAS.
Демонстрирует, что корректные значения получаются БЕЗ ручной разметки уровней экспертом
(нужна лишь калибровка/семейство шкал из шаблона; уровни выводит decode_levels).

Сравнение corr-vs-LAS (log): наша+декод | наша+oracle(GT-уровни) | наша+base(L0) | эксперт+декод.

ЗАПУСК (venv ComfyUI python):
  python decode_e2e.py <ckpt> --nlgx <f.nlgx> [--lam 0.7]
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from infer import load_model, predict_prob, extract_trace
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image, find_las
from decode_levels import build_family, gt_levels, decode, scale_map, is_resistive

Image.MAX_IMAGE_PIXELS = None


def corr_log(xs_by_row, levels, family, m, las_depths, las_vals):
    maps = [scale_map(s) for s in family]; K = len(family)
    da = m["depth_axis"]; ty, by = da["top_y"], da["bottom_y"]
    td, bd = da["top_depth"], da["bottom_depth"]
    def dof(y): return td + (y - ty) * (bd - td) / (by - ty)
    R = []; L = []
    for y, x in xs_by_row.items():
        v = maps[min(levels.get(y, 0), K-1)](x)
        d = dof(y); li = int(np.clip(np.searchsorted(las_depths, d), 0, len(las_depths)-1))
        lv = las_vals[li]
        if np.isfinite(lv) and lv > 0 and v > 0:
            R.append(math.log(v)); L.append(math.log(lv))
    if len(R) < 30:
        return None
    return float(np.corrcoef(R, L)[0, 1])


def main():
    ckpt = sys.argv[1]; args = sys.argv[2:]
    nlgx = args[args.index("--nlgx")+1]
    lam = float(args[args.index("--lam")+1]) if "--lam" in args else 0.7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    m = extract(nlgx)
    img = find_image(Path(nlgx)); las = find_las(Path(nlgx))
    rgb = np.asarray(Image.open(img).convert("RGB"))
    H, W, _ = rgb.shape
    da = m["depth_axis"]; y0, y1 = da["top_y"], min(H, da["bottom_y"])
    cols, arr = ds.load_las(las); depths = arr[:, 0]
    matches = ds.match_las(m, cols, arr)

    net, ck = load_model(ckpt, device)
    print(f"{Path(nlgx).stem[:46]}  model val_dice {ck.get('val_dice'):.3f}")
    prob = predict_prob(net, rgb, device, y0=y0, y1=y1)

    for c in ds.real_curves(m):
        if not is_resistive(c["name"]):
            continue
        fam = build_family(m, c); gl = gt_levels(c)
        ci = matches.get(c["name"], {}).get("col_idx")
        if len(fam) < 2 or ci is None:
            continue
        ty = c["top_y"]; n = len(c["xs"])
        exp = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL}
        ex = np.array(list(exp.values()))
        wx0 = max(0, int(ex.min() - 30)); wx1 = min(W, int(ex.max() + 30))
        tr = extract_trace(prob, ty, n, 0.4, wx0, wx1)         # НАША трасса (центроид по prob)
        ours = {ty+i: float(tr[i]) for i in range(n) if not np.isnan(tr[i])}
        if len(ours) < 100:
            print(f"  {c['name'].split()[0]:<8} мало точек U-Net ({len(ours)})"); continue
        # пиксельная ошибка нашей трассы vs эксперт (форма)
        comm = [y for y in ours if y in exp]
        px = float(np.median([abs(ours[y]-exp[y]) for y in comm])) if comm else float('nan')

        dec_ours = decode(ours, fam, lam)
        las_v = arr[:, ci]
        c_ours_dec = corr_log(ours, dec_ours, fam, m, depths, las_v)
        c_ours_orac = corr_log(ours, gl, fam, m, depths, las_v)
        c_ours_base = corr_log(ours, {y: 0 for y in ours}, fam, m, depths, las_v)
        c_exp_dec = corr_log(exp, decode(exp, fam, lam), fam, m, depths, las_v)
        def f(v): return f"{v:.3f}" if v is not None else "  -  "
        print(f"  {c['name'].split()[0]:<8} fam={len(fam)} maxL={max(gl.values()) if gl else 0} "
              f"наша_форма={px:.1f}px | corr: наша+декод={f(c_ours_dec)} наша+oracle={f(c_ours_orac)} "
              f"наша+base={f(c_ours_base)} эксперт+декод={f(c_exp_dec)}")


if __name__ == "__main__":
    main()
