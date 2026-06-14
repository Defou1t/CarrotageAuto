r"""
infer.py — инференс U-Net на полном планшете и извлечение трассы.
Конвейер: image -> тайловый прогон сети (стичинг с перекрытием) -> вероятностная
карта штриха -> per-row трасса (взвешенный центроид по вероятности в окне) ->
оценка vs трасса эксперта (пиксели) и vs эталон LAS (через оракул-калибровку nlgx).

ЗАПУСК интерпретатором venv ComfyUI (torch):
  python infer.py <ckpt> --nlgx <f.nlgx> [--image <f.jpg>] [--device cuda|cpu]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from unet import UNet
from extract_nlgx import extract, NULL, depth_axis_ok
import dataset as ds
from baseline_trace import family_of, level_fn, values_from_trace, corr_stats
from dataset_build import find_image, find_las

Image.MAX_IMAGE_PIXELS = None


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    net = UNet(in_ch=3, n_classes=1, base=ck.get("base", 32))
    net.load_state_dict(ck["model"])
    net.to(device).eval()
    return net, ck


def _tile_coords(H, W, tile, step):
    ys = list(range(0, max(1, H - tile + 1), step))
    xs = list(range(0, max(1, W - tile + 1), step))
    if ys[-1] != H - tile and H > tile:
        ys.append(H - tile)
    if xs[-1] != W - tile and W > tile:
        xs.append(W - tile)
    return ys, xs


@torch.no_grad()
def predict_prob(net, rgb, device, tile=256, ov=96, bs=24, y0=0, y1=None):
    """Вероятностная карта штриха (HxW float32) тайловым прогоном с перекрытием.
       y0..y1 ограничивает вертикальный диапазон (экономит время на полях)."""
    H, W, _ = rgb.shape
    y1 = H if y1 is None else min(H, y1)
    step = tile - ov
    ys, xs = _tile_coords(H, W, tile, step)
    ys = [y for y in ys if y + tile > y0 and y < y1]
    prob = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    # окно-вес (косинус) для мягкого стичинга
    win1 = np.hanning(tile); win = np.outer(win1, win1).astype(np.float32) + 1e-3
    batch = []; pos = []
    use_amp = (device == "cuda")
    def flush():
        if not batch:
            return
        t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div(255).to(device)
        if use_amp:
            with torch.amp.autocast("cuda"):
                p = torch.sigmoid(net(t)).float().cpu().numpy()[:, 0]
        else:
            p = torch.sigmoid(net(t)).cpu().numpy()[:, 0]
        for (yy, xx), pm in zip(pos, p):
            prob[yy:yy+tile, xx:xx+tile] += pm * win
            wsum[yy:yy+tile, xx:xx+tile] += win
        batch.clear(); pos.clear()
    for yy in ys:
        for xx in xs:
            batch.append(rgb[yy:yy+tile, xx:xx+tile]); pos.append((yy, xx))
            if len(batch) >= bs:
                flush()
    flush()
    return prob / np.maximum(wsum, 1e-6)


def extract_trace(prob, top_y, n_rows, thr=0.4, x0=0, x1=None):
    """per-row x = взвешенный по вероятности центроид среди prob>thr в окне [x0,x1)."""
    H, W = prob.shape
    x1 = W if x1 is None else x1
    out = np.full(n_rows, np.nan)
    cols = np.arange(x0, x1)
    for i in range(n_rows):
        y = top_y + i
        if not (0 <= y < H):
            continue
        seg = prob[y, x0:x1]
        m = seg > thr
        if m.any():
            w = seg[m]
            out[i] = float((cols[m] * w).sum() / w.sum())
    return out


def evaluate(net, device, nlgx, image, thr=0.4):
    m = extract(nlgx)
    rgb = np.asarray(Image.open(image).convert("RGB"))
    H, W, _ = rgb.shape
    curves = ds.real_curves(m)
    da = m["depth_axis"]
    y0, y1 = da["top_y"], min(H, da["bottom_y"] + 1)
    prob = predict_prob(net, rgb, device, y0=y0, y1=y1)
    # LAS для value-оценки
    las = find_las(Path(nlgx))
    las_cols, las_arr = (ds.load_las(las) if las else (None, np.zeros((0, 0))))
    matches = ds.match_las(m, las_cols, las_arr) if las_arr.size else {}
    depths = las_arr[:, 0] if las_arr.size else None
    print(f"{Path(nlgx).stem[:46]}  curves={len(curves)}")
    rows = []
    for c in curves:
        top_y = c["top_y"]
        pts = [(i, x) for i, x in enumerate(c["xs"]) if x != NULL]
        if len(pts) < 50:
            continue
        exp_rows = np.array([i for i, _ in pts]); exp_x = np.array([x for _, x in pts])
        # окно колонки по протяжённости кривой (как в baseline) — для мультикривных
        xpad = int(0.4 * (exp_x.max() - exp_x.min()) + 20)
        wx0 = max(0, exp_x.min() - xpad); wx1 = min(W, exp_x.max() + xpad)
        tr = extract_trace(prob, top_y, len(c["xs"]), thr, wx0, wx1)
        valid = ~np.isnan(tr[exp_rows])
        px = None
        if valid.sum() >= 20:
            dxe = np.abs(tr[exp_rows][valid] - exp_x[valid])
            px = (float(np.median(dxe)), float(np.mean(dxe <= 3)), float(valid.mean()))
        mm = matches.get(c["name"], {})
        lc = None
        if mm.get("col_idx") is not None and depth_axis_ok(m):
            fam = family_of(m, c); lvl = level_fn(c)
            recon = values_from_trace(m, c, tr, top_y, fam, lvl)
            st = corr_stats(recon, depths, las_arr[:, mm["col_idx"]])
            lc = st["log_corr"] if st else None
        rows.append((c["name"].split()[0], px, lc))
        pxs = f"px_med={px[0]:.1f} w3={px[1]:.2f}" if px else "px=-"
        print(f"   {c['name'].split()[0]:<8} {pxs}  log_corr={lc if lc is None else round(lc,3)}")
    return rows


def main():
    ckpt = sys.argv[1]
    args = sys.argv[2:]
    nlgx = image = None; device = "cuda" if torch.cuda.is_available() else "cpu"
    i = 0
    while i < len(args):
        if args[i] == "--nlgx": nlgx = args[i+1]; i += 2
        elif args[i] == "--image": image = args[i+1]; i += 2
        elif args[i] == "--device": device = args[i+1]; i += 2
        else: i += 1
    if image is None and nlgx:
        image = str(find_image(Path(nlgx)))
    sys.stdout.reconfigure(line_buffering=True)
    net, ck = load_model(ckpt, device)
    print(f"model {ckpt} (epoch {ck.get('epoch')}, val_dice {ck.get('val_dice')}) device={device}")
    evaluate(net, device, nlgx, image)


if __name__ == "__main__":
    main()
