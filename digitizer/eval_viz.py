r"""
eval_viz.py — диагностика «где датасет/модель мешают» (запускать после train.py,
интерпретатором venv ComfyUI с torch).
  • прогон модели по val-тайлам -> per-tile Dice;
  • разбивка Dice ПО СКВАЖИНАМ (видно, какие скважины/типы проваливаются);
  • ТРИПТИХИ худших тайлов: [изображение | разметка(GT) | предсказание] — по ним
    видно: плохая РАЗМЕТКА (зелёная маска мимо чернил = шум меток, чинить датасет)
    или ТРУДНОЕ изображение (бледно/наложено = задача для модели).
  • при наличии manifest_all.csv — корреляция качества с align_frac (метка-шум).

python eval_viz.py <ckpt> [--data DIR] [--worst 40] [--device cuda]
Выходы: <data>/eval_worst.png, <data>/eval_per_well.csv
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from unet import UNet


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    net = UNet(in_ch=3, n_classes=1, base=ck.get("base", 32))
    net.load_state_dict(ck["model"]); net.to(device).eval()
    return net, ck


@torch.no_grad()
def tile_dice(net, imgs, device, bs=64, thr=0.5, eps=1e-6):
    """per-tile Dice + предсказанные вероятности (для триптихов худших)."""
    dices = []; probs = []
    for k in range(0, len(imgs), bs):
        t = torch.from_numpy(imgs[k:k+bs]).permute(0, 3, 1, 2).float().div(255).to(device)
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                p = torch.sigmoid(net(t)).float()
        else:
            p = torch.sigmoid(net(t))
        probs.append(p.cpu().numpy()[:, 0])
    return np.concatenate(probs) if probs else np.zeros((0,))


def dice_of(pred_bin, gt):
    inter = (pred_bin & gt).sum()
    s = pred_bin.sum() + gt.sum()
    if s == 0:
        return 1.0  # обе пусты (негатив-тайл) — совпали
    return 2 * inter / s


def triptych(img, gt, prob, thr=0.5):
    """[img | img+GT(зел) | img+pred(красн)] горизонтально, T×(3T)×3."""
    T = img.shape[0]
    base = img.copy()
    g = base.copy(); g[gt > 0] = (0.3 * g[gt > 0] + 0.7 * np.array([0, 220, 0])).astype(np.uint8)
    pr = base.copy(); pm = prob > thr
    pr[pm] = (0.3 * pr[pm] + 0.7 * np.array([230, 0, 0])).astype(np.uint8)
    sep = np.full((T, 3, 3), 255, np.uint8)
    return np.concatenate([base, sep, g, sep, pr], axis=1)


def main():
    ckpt = sys.argv[1]; args = sys.argv[2:]
    data = Path(r"F:\nds\output\unet_data"); worst = 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    i = 0
    while i < len(args):
        if args[i] == "--data": data = Path(args[i+1]); i += 2
        elif args[i] == "--worst": worst = int(args[i+1]); i += 2
        elif args[i] == "--device": device = args[i+1]; i += 2
        else: i += 1
    sys.stdout.reconfigure(line_buffering=True)

    net, ck = load_model(ckpt, device)
    print(f"model epoch={ck.get('epoch')} val_dice(train-time)={ck.get('val_dice')} device={device}")
    index = [r for r in json.loads((data / "index.json").read_text(encoding="utf-8"))
             if r["split"] == "val"]

    per_well = {}            # well -> [dices]
    worst_heap = []          # (dice, npz, idx)
    all_d = []
    for rec in index:
        z = np.load(rec["npz"])
        imgs, masks = z["imgs"], z["masks"]
        probs = tile_dice(net, imgs, device)
        for j in range(len(imgs)):
            d = dice_of(probs[j] > 0.5, masks[j] > 0)
            all_d.append(d)
            per_well.setdefault(rec["well"], []).append(d)
            worst_heap.append((d, rec["npz"], j))
        del z

    all_d = np.array(all_d)
    print(f"\nVAL tiles={len(all_d)}  mean Dice={all_d.mean():.4f}  median={np.median(all_d):.4f}")
    print(f"  Dice<0.3: {np.mean(all_d<0.3)*100:.1f}%   Dice>0.8: {np.mean(all_d>0.8)*100:.1f}%")

    # per-well (худшие сверху)
    rows = sorted(((w, np.mean(d), len(d)) for w, d in per_well.items()), key=lambda r: r[1])
    print("\nPER-WELL Dice (worst 12):")
    for w, md, n in rows[:12]:
        print(f"  {w:<22} dice={md:.3f}  tiles={n}")
    with open(data / "eval_per_well.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["well", "mean_dice", "n_tiles"])
        for w, md, n in rows:
            wr.writerow([w, round(float(md), 4), n])

    # триптихи худших (исключая пустые негатив-тайлы — там Dice неинформативен)
    worst_heap.sort(key=lambda t: t[0])
    picks = [t for t in worst_heap if t[0] < 0.95][:worst]
    if picks:
        cells = []
        cache = {}
        for d, npz, j in picks:
            if npz not in cache:
                cache[npz] = np.load(npz)
            z = cache[npz]
            img = z["imgs"][j]; gt = z["masks"][j]
            p = tile_dice(net, img[None], device)[0]
            cells.append(triptych(img, gt, p))
        T = cells[0].shape[0]
        grid = np.full(((T + 6) * len(cells) - 6, cells[0].shape[1], 3), 255, np.uint8)
        for k, c in enumerate(cells):
            grid[k*(T+6):k*(T+6)+T] = c
        out = data / "eval_worst.png"
        Image.fromarray(grid).save(out)
        print(f"\nworst {len(picks)} tiles -> {out}  (столбцы: image | GT зел | pred красн)")
        print(f"  Dice худших: {[round(t[0],2) for t in picks[:10]]}")


if __name__ == "__main__":
    main()
