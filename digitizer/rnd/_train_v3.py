r"""
_train_v3.py — ДООБУЧЕНИЕ U-Net v3: 1 канал, метки точной ширины + ВЗВЕШЕННЫЙ лосс
(BCE × вес-карта оборотов + Dice). Инициализация от unet_best.pt (старое не трогаем).
Запуск ИНТЕРПРЕТАТОРОМ ComfyUI (torch+CUDA):
  D:\ComfyUI\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe _train_v3.py [--epochs N --bs N --lr F]
-> F:\nds\output\unet_v3\unet_v3.pt
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from unet import UNet, dice_bce_loss

DATA = Path(r"F:\nds\output\unet_v3\data")
INIT = r"F:\nds\output\unet_data\unet_best.pt"
OUT = r"F:\nds\output\unet_v3\unet_v3.pt"


class TileDSv3(Dataset):
    def __init__(self, split, augment=False, cap=None, seed=0):
        self.aug = augment
        index = json.load(open(DATA / "index.json", encoding="utf-8"))
        recs = [r for r in index if r["split"] == split]
        if cap:                                            # подвыборка ПО ФАЙЛАМ до лимита (память!)
            rng = np.random.default_rng(seed); rng.shuffle(recs)
            keep, acc = [], 0
            for r in recs:
                keep.append(r); acc += r["n"]
                if acc >= cap:
                    break
            recs = keep
        T = 256; N = sum(r["n"] for r in recs)
        self.imgs = np.empty((N, T, T, 3), np.uint8)
        self.masks = np.empty((N, T, T), np.uint8)
        self.wts = np.empty((N, T, T), np.float16)
        pos = 0
        for r in recs:
            with np.load(DATA / split / f"{r['key']}.npz") as z:
                n = z["imgs"].shape[0]
                self.imgs[pos:pos + n] = z["imgs"]; self.masks[pos:pos + n] = z["masks"]
                self.wts[pos:pos + n] = z["weights"]; pos += n
        self.imgs = self.imgs[:pos]; self.masks = self.masks[:pos]; self.wts = self.wts[:pos]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32) / 255.0
        msk = self.masks[i].astype(np.float32); wt = self.wts[i].astype(np.float32)
        if self.aug:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy(); wt = wt[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1].copy(); msk = msk[::-1].copy(); wt = wt[::-1].copy()
            if np.random.rand() < 0.5:
                img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
            if np.random.rand() < 0.5:
                img = np.clip((img - 0.5) * np.random.uniform(0.7, 1.4) + 0.5, 0, 1)
            if np.random.rand() < 0.5:
                img = np.clip(img ** np.random.uniform(0.7, 1.5), 0, 1)
            if np.random.rand() < 0.3:
                img = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.from_numpy(msk).unsqueeze(0), torch.from_numpy(wt).unsqueeze(0))


def wloss(logits, target, weight, eps=1.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weight).sum() / weight.sum().clamp(min=1.0)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return bce + (1 - (num / den).mean())


def dice_iou(logits, target, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum((1, 2, 3)); union = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item(), ((inter + eps) / (union - inter + eps)).mean().item()


def main():
    a = sys.argv[1:]
    epochs = int(a[a.index("--epochs") + 1]) if "--epochs" in a else 30
    bs = int(a[a.index("--bs") + 1]) if "--bs" in a else 24
    lr = float(a[a.index("--lr") + 1]) if "--lr" in a else 5e-4
    maxtr = int(a[a.index("--maxtrain") + 1]) if "--maxtrain" in a else 80000
    maxva = int(a[a.index("--maxval") + 1]) if "--maxval" in a else 12000
    out = a[a.index("--out") + 1] if "--out" in a else OUT
    scratch = "--scratch" in a; plain = "--plainloss" in a; noamp = "--noamp" in a
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = TileDSv3("train", augment=True, cap=maxtr); va = TileDSv3("val", cap=maxva)
    print(f"device={dev} train={len(tr)} val={len(va)} epochs={epochs} bs={bs} lr={lr} "
          f"scratch={scratch} plainloss={plain} -> {out}")
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=bs)
    net = UNet(in_ch=3, n_classes=1, base=32).to(dev)
    if not scratch:
        ck = torch.load(INIT, map_location=dev); net.load_state_dict(ck["model"])
        print(f"инициализация от unet_best (было val_dice={ck.get('val_dice')})")
    else:
        print("инициализация С НУЛЯ (random)")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(dev) if (dev == "cuda" and not noamp) else None    # noamp=fp32, без NaN
    if noamp:
        print("AMP ОТКЛЮЧЁН (fp32) — стабильно к NaN")
    best = -1.0
    for ep in range(1, epochs + 1):
        net.train(); t0 = time.time(); losses = []
        for img, msk, wt in dl_tr:
            img, msk, wt = img.to(dev), msk.to(dev), wt.to(dev); opt.zero_grad()
            with torch.amp.autocast(dev, enabled=bool(scaler)):
                logit = net(img)
                loss = dice_bce_loss(logit, msk) if plain else wloss(logit, msk, wt)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)     # страховка от NaN
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        net.eval(); ds_, io_ = [], []
        with torch.no_grad():
            for img, msk, wt in dl_va:
                img, msk = img.to(dev), msk.to(dev)
                with torch.amp.autocast(dev, enabled=bool(scaler)):
                    logit = net(img)
                d, io = dice_iou(logit.float(), msk); ds_.append(d); io_.append(io)
        vd = float(np.mean(ds_)) if ds_ else 0.0
        print(f"ep {ep:>2}/{epochs} loss={np.mean(losses):.4f} val_dice={vd:.4f} {time.time()-t0:.0f}s")
        if vd > best:
            best = vd
            torch.save({"model": net.state_dict(), "epoch": ep, "val_dice": vd, "base": 32}, out)
            print(f"   * сохранён best -> {out}")
    print(f"\nBEST val_dice={best:.4f} -> {out}")


if __name__ == "__main__":
    main()
