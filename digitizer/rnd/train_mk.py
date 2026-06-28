r"""Трек 2 / шаг 2 — обучение РАЗДЕЛИТЕЛЯ MK: UNet(2 канала) MGZ(к0,толстая) / MPZ(к1,тонкая).
PER-CHANNEL dice форсит каждый канал отдельно (против схлопывания в union). fp32 (стабильно).
Запуск интерпретатором venv ComfyUI (torch/RTX5080):
  <venv>\python.exe train_mk.py [--epochs 28 --bs 16 --lr 4e-4]
-> F:\nds\output\mk_data\mk_sep.pt (лучшая по val per-channel dice)
"""
import sys, glob, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unet import UNet
DATA = Path(r"F:\nds\output\mk_data")
OUT = DATA / "mk_sep.pt"


class MK(Dataset):
    def __init__(self, split, augment=False):
        self.aug = augment
        imgs, masks = [], []
        for f in sorted(glob.glob(str(DATA / split / "*.npz"))):
            d = np.load(f); imgs.append(d["imgs"]); masks.append(d["masks"])
        self.imgs = np.concatenate(imgs); self.masks = np.concatenate(masks)   # N×T×T×3, N×T×T×2

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i].astype(np.float32) / 255.0       # T×T×3
        msk = self.masks[i].astype(np.float32)              # T×T×2
        if self.aug:
            if np.random.rand() < 0.5:                       # гориз. флип (идентичность по каналу, не позиции)
                img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
            if np.random.rand() < 0.5:                       # верт. флип
                img = img[::-1].copy(); msk = msk[::-1].copy()
            img = np.clip(img * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        img = torch.from_numpy(img.transpose(2, 0, 1))      # 3×T×T
        msk = torch.from_numpy(msk.transpose(2, 0, 1))      # 2×T×T
        return img, msk


def loss_fn(logits, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits); dice = 0.0
    for c in range(target.shape[1]):                         # PER-CHANNEL dice
        num = 2 * (p[:, c] * target[:, c]).sum((1, 2)) + 1.0
        den = p[:, c].sum((1, 2)) + target[:, c].sum((1, 2)) + 1.0
        dice = dice + (1 - num / den).mean()
    return bce + dice / target.shape[1]


@torch.no_grad()
def val_dice(model, dl, dev):
    model.eval(); ds = [0.0, 0.0]; n = 0
    for img, msk in dl:
        img, msk = img.to(dev), msk.to(dev)
        p = (torch.sigmoid(model(img)) > 0.5).float()
        for c in range(2):
            num = 2 * (p[:, c] * msk[:, c]).sum((1, 2))
            den = p[:, c].sum((1, 2)) + msk[:, c].sum((1, 2)) + 1e-6
            ds[c] += (num / den).sum().item()
        n += len(img)
    return ds[0] / n, ds[1] / n


def main():
    a = sys.argv[1:]
    ep = 28; bs = 16; lr = 4e-4; base = 48          # base32→48: больше ёмкости для разделения
    if "--epochs" in a: ep = int(a[a.index("--epochs") + 1])
    if "--bs" in a: bs = int(a[a.index("--bs") + 1])
    if "--lr" in a: lr = float(a[a.index("--lr") + 1])
    if "--base" in a: base = int(a[a.index("--base") + 1])
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(MK("train", augment=True), batch_size=bs, shuffle=True, num_workers=0)
    va = DataLoader(MK("val"), batch_size=bs, num_workers=0)
    print(f"device={dev} train={len(tr.dataset)} val={len(va.dataset)} ep={ep} bs={bs}")
    model = UNet(in_ch=3, n_classes=2, base=base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    best = -1
    for e in range(ep):
        model.train(); t0 = time.time(); tot = 0.0
        for img, msk in tr:
            img, msk = img.to(dev), msk.to(dev)
            opt.zero_grad(); out = model(img); l = loss_fn(out, msk)
            l.backward(); opt.step(); tot += l.item()
        sched.step()
        dm, dp = val_dice(model, va, dev)
        score = (dm + dp) / 2
        print(f"ep{e+1:2d} loss={tot/len(tr):.3f} val_dice MGZ={dm:.3f} MPZ={dp:.3f} avg={score:.3f} ({time.time()-t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": e + 1, "val_dice": score,
                        "n_classes": 2, "base": base}, OUT)
            print(f"   ✓ best -> {OUT}")
    print(f"DONE best avg val_dice={best:.3f}")


if __name__ == "__main__":
    main()
