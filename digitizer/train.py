r"""
train.py — обучение U-Net (бинарная сегментация штриха кривой) на тайлах prep_dataset.
ЗАПУСКАТЬ интерпретатором venv ComfyUI (torch 2.11+cu130, RTX 5080):
  D:\ComfyUI\StabilityMatrix\Data\Packages\ComfyUI\venv\Scripts\python.exe train.py [args]
Аргументы: --data DIR --epochs N --bs N --lr F --out FILE
Метрики: Dice/IoU на val. Сохраняет лучший чекпойнт по val Dice.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from unet import UNet, dice_bce_loss


class TileDS(Dataset):
    def __init__(self, index, split, augment=False):
        self.aug = augment
        recs = [r for r in index if r["split"] == split]
        # преаллокация (без удвоения памяти на concatenate)
        T = 256
        if recs:
            with np.load(recs[0]["npz"]) as z0:
                T = z0["imgs"].shape[1]
        N = sum(int(r.get("n_tiles", 0)) for r in recs)
        if N == 0 and recs:  # индекс без n_tiles — посчитать по shape
            N = 0
            for r in recs:
                with np.load(r["npz"], mmap_mode="r") as z:
                    N += z["imgs"].shape[0]
        self.imgs = np.empty((N, T, T, 3), np.uint8)
        self.masks = np.empty((N, T, T), np.uint8)
        pos = 0
        for r in recs:
            with np.load(r["npz"]) as z:
                n = z["imgs"].shape[0]
                self.imgs[pos:pos+n] = z["imgs"]
                self.masks[pos:pos+n] = z["masks"]
                pos += n
        self.imgs = self.imgs[:pos]; self.masks = self.masks[:pos]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32) / 255.0   # T×T×3
        msk = self.masks[i].astype(np.float32)          # T×T
        if self.aug:
            if np.random.rand() < 0.5:                  # hflip
                img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
            if np.random.rand() < 0.5:                  # vflip
                img = img[::-1].copy(); msk = msk[::-1].copy()
            if np.random.rand() < 0.5:                  # brightness
                img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
        img = torch.from_numpy(img).permute(2, 0, 1)    # 3×T×T
        msk = torch.from_numpy(msk).unsqueeze(0)        # 1×T×T
        return img, msk


def dice_iou(logits, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    iou = ((inter + eps) / (union - inter + eps)).mean().item()
    return dice, iou


def main():
    args = sys.argv[1:]
    data = Path(r"F:\nds\output\unet_data"); epochs = 30; bs = 32; lr = 1e-3
    out = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--data": data = Path(args[i+1]); i += 2
        elif a == "--epochs": epochs = int(args[i+1]); i += 2
        elif a == "--bs": bs = int(args[i+1]); i += 2
        elif a == "--lr": lr = float(args[i+1]); i += 2
        elif a == "--out": out = args[i+1]; i += 2
        else: i += 1
    out = out or str(data / "unet_best.pt")
    sys.stdout.reconfigure(line_buffering=True)

    index = json.loads((data / "index.json").read_text(encoding="utf-8"))
    tr = TileDS(index, "train", augment=True)
    va = TileDS(index, "val", augment=False)
    print(f"train tiles={len(tr)} val tiles={len(va)}")
    if len(tr) == 0:
        print("NO TRAIN TILES — run prep_dataset.py first"); return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} {torch.cuda.get_device_name(0) if dev=='cuda' else ''}")
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0)

    net = UNet(in_ch=3, n_classes=1, base=32).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None
    best = -1.0
    for ep in range(1, epochs + 1):
        net.train(); t0 = time.time(); losses = []
        for img, msk in dl_tr:
            img = img.to(dev); msk = msk.to(dev)
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast(dev):
                    logit = net(img); loss = dice_bce_loss(logit, msk)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logit = net(img); loss = dice_bce_loss(logit, msk)
                loss.backward(); opt.step()
            losses.append(loss.item())
        # val
        net.eval(); ds_, io_ = [], []
        with torch.no_grad():
            for img, msk in dl_va:
                img = img.to(dev); msk = msk.to(dev)
                if scaler:
                    with torch.amp.autocast(dev):
                        logit = net(img)
                else:
                    logit = net(img)
                d, io = dice_iou(logit.float(), msk)
                ds_.append(d); io_.append(io)
        vdice = float(np.mean(ds_)) if ds_ else 0.0
        viou = float(np.mean(io_)) if io_ else 0.0
        print(f"ep {ep:>3}/{epochs}  loss={np.mean(losses):.4f}  val_dice={vdice:.4f} "
              f"val_iou={viou:.4f}  {time.time()-t0:.1f}s")
        if vdice > best:
            best = vdice
            torch.save({"model": net.state_dict(), "epoch": ep, "val_dice": vdice,
                        "base": 32}, out)
    print(f"\nBEST val_dice={best:.4f} -> {out}")


if __name__ == "__main__":
    main()
