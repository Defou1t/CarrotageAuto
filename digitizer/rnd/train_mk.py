r"""Трек 2 / шаг 2 — обучение РАЗДЕЛИТЕЛЯ MK: UNet(2 канала) MGZ(к0,тонкая) / MPZ(к1,толстая).
PER-CHANNEL dice форсит каждый канал отдельно (против схлопывания в union). fp32 (стабильно).
v7: маски могут нести 3-й ignore-канал (чужая тушь — не штрафуем, анализ 02.07); чекпойнт
отбирается по dice − 0.3·collapse (dice не чувствует схлопывание двух узких каналов на одной
кривой — а это главный failure mode production-метрики).
Запуск интерпретатором venv ComfyUI (torch/RTX5080):
  <venv>\python.exe train_mk.py [--epochs 28 --bs 16 --lr 4e-4 --data F:\nds\output\mk_data]
-> <data>\mk_sep.pt (лучшая по val score)
"""
import sys, glob, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unet import UNet
DATA = Path(r"F:\nds\output\mk_data")


class MK(Dataset):
    CAP = 0                                                 # 0 = все тайлы; >0 = потолок (RAM: 512² тайлы тяжёлые)

    def __init__(self, split, augment=False):
        self.aug = augment
        files = sorted(glob.glob(str(DATA / split / "*.npz")))
        per = 10**9 if not MK.CAP or split != "train" else max(1, MK.CAP // max(1, len(files)))
        rng = np.random.default_rng(0)
        imgs, masks = [], []
        for f in files:
            d = np.load(f); im = d["imgs"]; mk = d["masks"]
            if len(im) > per:                               # субсэмпл per-файл — пик RAM ограничен
                sel = rng.choice(len(im), per, replace=False)
                im, mk = im[sel], mk[sel]
            imgs.append(im); masks.append(mk)
        self.imgs = np.concatenate(imgs); self.masks = np.concatenate(masks)   # N×T×T×3, N×T×T×{2,3}
        if self.masks.shape[-1] == 2:                       # старый преп без ignore-канала
            self.masks = np.concatenate([self.masks, np.zeros_like(self.masks[..., :1])], -1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        import cv2
        img = self.imgs[i].astype(np.float32) / 255.0       # T×T×3
        msk = self.masks[i].astype(np.float32)              # T×T×3 (MGZ,MPZ,ign)
        if self.aug:
            if np.random.rand() < 0.5:                       # гориз. флип
                img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
            if np.random.rand() < 0.5:                       # верт. флип
                img = img[::-1].copy(); msk = msk[::-1].copy()
        # DT-КАНАЛ (толщина штриха перпендикулярно) = явный различитель MGZ(тонкий)/MPZ(толстый);
        # из тёмного, ДО intensity-джиттера. RGB модель толщину учит, DT даёт её прямо.
        dark = (img.max(2) < 110 / 255.0).astype(np.uint8)
        dt = np.clip(cv2.distanceTransform(dark, cv2.DIST_L2, 5) / 8.0, 0, 1).astype(np.float32)
        if self.aug:
            img = np.clip(img * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        img4 = np.concatenate([img, dt[..., None]], -1)     # T×T×4 (RGB+DT)
        img = torch.from_numpy(img4.transpose(2, 0, 1))     # 4×T×T
        msk = torch.from_numpy(msk.transpose(2, 0, 1))      # 3×T×T (MGZ,MPZ,ign)
        return img, msk


def loss_fn(logits, target3):
    """BCE+dice по 2 каналам; ignore-канал (target3[:,2]) обнуляет вес пикселя — чужая тушь
    не учится как фон (раньше модель училась давить рассинхронизированную собственную кривую)."""
    target = target3[:, :2]
    w = 1.0 - target3[:, 2:3]                                # B×1×T×T → broadcast на оба канала
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight=w)
    p = torch.sigmoid(logits) * w; tw = target * w
    dice = 0.0
    for c in range(2):                                       # PER-CHANNEL dice (вне ignore)
        num = 2 * (p[:, c] * tw[:, c]).sum((1, 2)) + 1.0
        den = p[:, c].sum((1, 2)) + tw[:, c].sum((1, 2)) + 1.0
        dice = dice + (1 - num / den).mean()
    return bce + dice / 2


def to_label(msk):
    """Маска (B×3×T×T) → метки {0=фон,1=MGZ,2=MPZ,-100=ignore} (B×T×T long)."""
    lab = torch.zeros(msk.shape[0], msk.shape[2], msk.shape[3], dtype=torch.long, device=msk.device)
    lab[msk[:, 0] > 0.5] = 1
    lab[msk[:, 1] > 0.5] = 2                                  # спорные (оба канала) → MPZ; для CE некритично
    lab[msk[:, 2] > 0.5] = -100                               # ignore_index
    return lab


def loss_fn_softmax(logits, target3):
    """softmax-КОНКУРЕНЦИЯ {фон,MGZ,MPZ}: пиксель принадлежит ровно одному классу (бьёт «оба горят»).
    weighted CE (фон дёшев, ignore не штрафуется) + per-fg soft-dice."""
    lab = to_label(target3)
    w = torch.tensor([0.3, 1.0, 1.0], device=logits.device)
    ce = torch.nn.functional.cross_entropy(logits, lab, weight=w, ignore_index=-100)
    p = torch.softmax(logits, 1); dice = 0.0
    for c in (1, 2):                                         # foreground dice (MGZ/MPZ)
        tc = (lab == c).float()
        num = 2 * (p[:, c] * tc).sum((1, 2)) + 1.0
        den = p[:, c].sum((1, 2)) + tc.sum((1, 2)) + 1.0
        dice = dice + (1 - num / den).mean()
    return ce + dice / 2


@torch.no_grad()
def val_dice(model, dl, dev, softmax=False):
    """(dice_MGZ, dice_MPZ, collapse_frac). Collapse-прокси = per-row argmax обоих каналов
    схлопнут (<3px) при GT-разделении >10px — прямой прокси production-провала, dice его не видит."""
    model.eval(); ds = [0.0, 0.0]; n = 0
    col_bad = 0; col_all = 0
    T = None
    for img, msk in dl:
        img, msk = img.to(dev), msk.to(dev)
        logits = model(img)
        if softmax:
            am = torch.argmax(logits, 1)                     # B×T×T в {0,1,2}
            p = torch.stack([(am == 1).float(), (am == 2).float()], 1)
            pr = torch.softmax(logits, 1)[:, 1:]
        else:
            pr = torch.sigmoid(logits)
            p = (pr > 0.5).float()
        for c in range(2):
            num = 2 * (p[:, c] * msk[:, c]).sum((1, 2))
            den = p[:, c].sum((1, 2)) + msk[:, c].sum((1, 2)) + 1e-6
            ds[c] += (num / den).sum().item()
        n += len(img)
        # collapse-прокси по строкам тайла
        if T is None:
            T = msk.shape[-1]
        xs = torch.arange(T, device=dev, dtype=torch.float32)
        gsum = msk[:, :2].sum(-1)                            # B×2×T
        gx = (msk[:, :2] * xs).sum(-1) / gsum.clamp(min=1)   # B×2×T (центр GT per-row)
        pk = pr.argmax(-1).float()                           # B×2×T (пик prob per-row)
        pv = pr.max(-1).values                               # B×2×T
        valid = (gsum[:, 0] > 0) & (gsum[:, 1] > 0) & ((gx[:, 0] - gx[:, 1]).abs() > 10) \
                & (pv[:, 0] > 0.4) & (pv[:, 1] > 0.4)
        col_all += int(valid.sum())
        col_bad += int((valid & ((pk[:, 0] - pk[:, 1]).abs() < 3)).sum())
    return ds[0] / n, ds[1] / n, col_bad / max(1, col_all)


def main():
    global DATA
    a = sys.argv[1:]
    ep = 28; bs = 16; lr = 4e-4; base = 48          # base32→48: больше ёмкости для разделения
    if "--epochs" in a: ep = int(a[a.index("--epochs") + 1])
    if "--bs" in a: bs = int(a[a.index("--bs") + 1])
    if "--lr" in a: lr = float(a[a.index("--lr") + 1])
    if "--base" in a: base = int(a[a.index("--base") + 1])
    if "--data" in a: DATA = Path(a[a.index("--data") + 1])
    softmax = "--softmax" in a                       # 3-класс softmax-конкуренция vs незав. сигмоиды
    scale = int(a[a.index("--scale") + 1]) if "--scale" in a else 1   # метка масштаба препа → в чекпойнт для infer
    if "--maxtrain" in a: MK.CAP = int(a[a.index("--maxtrain") + 1])  # потолок train-тайлов (RAM для 512²)
    OUT = DATA / "mk_sep.pt"
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(MK("train", augment=True), batch_size=bs, shuffle=True, num_workers=0)
    va = DataLoader(MK("val"), batch_size=bs, num_workers=0)
    tile = int(tr.dataset.imgs.shape[1])                 # размер тайла из данных (512 у честного 2×)
    print(f"device={dev} train={len(tr.dataset)} val={len(va.dataset)} ep={ep} bs={bs} tile={tile} scale={scale} softmax={softmax}")
    ncls = 3 if softmax else 2
    crit = loss_fn_softmax if softmax else loss_fn
    model = UNet(in_ch=4, n_classes=ncls, base=base).to(dev)   # RGB+DT
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    best = -1
    for e in range(ep):
        model.train(); t0 = time.time(); tot = 0.0
        for img, msk in tr:
            img, msk = img.to(dev), msk.to(dev)
            opt.zero_grad(); out = model(img); l = crit(out, msk)
            l.backward(); opt.step(); tot += l.item()
        sched.step()
        dm, dp, col = val_dice(model, va, dev, softmax)
        score = (dm + dp) / 2 - 0.3 * col                    # чекпойнт по dice − collapse (не голому dice)
        print(f"ep{e+1:2d} loss={tot/len(tr):.3f} val_dice MGZ={dm:.3f} MPZ={dp:.3f} avg={(dm+dp)/2:.3f} "
              f"collapse={col:.3f} score={score:.3f} ({time.time()-t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": e + 1, "val_dice": (dm + dp) / 2,
                        "val_collapse": col, "score": score,
                        "n_classes": ncls, "base": base, "in_ch": 4, "softmax": softmax,
                        "scale": scale, "tile": tile}, OUT)
            print(f"   ✓ best -> {OUT}")
    print(f"DONE best score={best:.3f}")


if __name__ == "__main__":
    main()
