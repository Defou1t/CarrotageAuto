r"""
_recall_validate.py — валидация ЦЕЛИ эксперта: видит ли prob бледные PZ/GZ?
Гоняет ckpt на STK+DS, мерит RECALL per-кривую = доля GT-точек трассы, где prob>порог.
Сравнивать: unet_best (старый) vs unet_recall (новый). Цель: ↑recall на PZ(чёрн.бледн.)/GZ(зел.выцв.).

  D:\ComfyUI\...\python_embeded\python.exe _recall_validate.py <ckpt> [thr=0.3]
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from unet import UNet
from extract_nlgx import extract, NULL
from dataset_build import find_image

F = r"F:\nds\projects\Yatskivska_001\wlg\Yatskivska_1_STK+DS_2673_3008_500_D1.nlgx"


@torch.no_grad()
def predict(net, rgb, dev, tile=256, ov=96, bs=24, y0=0, y1=None):
    H, W, _ = rgb.shape; y1 = H if y1 is None else min(H, y1); step = tile - ov
    ys = list(range(0, max(1, H - tile + 1), step)); xs = list(range(0, max(1, W - tile + 1), step))
    if H > tile and ys[-1] != H - tile: ys.append(H - tile)
    if W > tile and xs[-1] != W - tile: xs.append(W - tile)
    ys = [y for y in ys if y + tile > y0 and y < y1]
    prob = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    w1 = np.hanning(tile); win = np.outer(w1, w1).astype(np.float32) + 1e-3
    batch = []; pos = []
    def flush():
        if not batch: return
        t = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div(255).to(dev)
        p = torch.sigmoid(net(t)).float().cpu().numpy()[:, 0]
        for (yy, xx), pm in zip(pos, p):
            prob[yy:yy+tile, xx:xx+tile] += pm * win; wsum[yy:yy+tile, xx:xx+tile] += win
        batch.clear(); pos.clear()
    for yy in ys:
        for xx in xs:
            batch.append(rgb[yy:yy+tile, xx:xx+tile]); pos.append((yy, xx))
            if len(batch) >= bs: flush()
    flush()
    return prob / np.maximum(wsum, 1e-6)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ckpt = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    fpath = sys.argv[3] if len(sys.argv) > 3 else F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = extract(fpath); img = find_image(Path(fpath)) or m["img_path"]
    if not (img and Path(img).is_file()):
        img = r"F:\nds\projects\Yatskivska_001\img\Yatskivska_1_STK+DS_2673_3008_500_D1.jpg"
    rgb = np.asarray(Image.open(img).convert("RGB")); H, W = rgb.shape[:2]
    da = m["depth_axis"]; ty, by = da["top_y"], min(H, da["bottom_y"])
    ck = torch.load(ckpt, map_location=dev)
    net = UNet(in_ch=3, n_classes=1, base=ck.get("base", 32)); net.load_state_dict(ck["model"]); net.to(dev).eval()
    prob = predict(net, rgb, dev, y0=ty, y1=by)
    print(f"ckpt={Path(ckpt).name} (recall={ck.get('val_recall')} dice={ck.get('val_dice')}) thr={thr}")
    gt = {c["name"].split()[0]: {c["top_y"] + i: x for i, x in enumerate(c["xs"]) if x != NULL}
          for c in m["curves"] if c["name"].split()[0] in ("PZ1", "GZ1", "SP1", "DS1")}
    for nm, d in gt.items():
        hit = tot = 0
        for y, x in d.items():
            xi = int(round(x))
            if 0 <= y < H and 0 <= xi < W:
                tot += 1; hit += prob[y, max(0, xi - 3):xi + 4].max() > thr
        print(f"  {nm}: recall={100*hit/max(1,tot):.0f}% (n={tot})")
    np.save(Path(ckpt).with_suffix(".stkdsprob.npy"), prob)


if __name__ == "__main__":
    main()
