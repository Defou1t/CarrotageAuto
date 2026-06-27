r"""
_train_recall.py — дотюн U-Net на ВЫСОКИЙ RECALL бледных кривых (цель эксперта: prob видит бледный
PZ-чёрный и выцветший GZ). От unet_best + FADE-аугментация (сдвиг чернил к фону = выцветание) +
TVERSKY-лосс (β>α штрафует FN = recall) + взвеш.BCE. Негативы (грид/текст) уже в тайлах → не галлюцинирует.
Цвет берётся ПОЗЖЕ per-инстанс (не в модели). fp32 (стабильно к NaN).

  D:\ComfyUI\...\python_embeded\python.exe _train_recall.py [--epochs 16 --bs 24 --lr 4e-4 --maxtrain 70000]
-> F:\nds\output\unet_v3\unet_recall.pt
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, r"F:\nds\Auto\digitizer")
from unet import UNet

DATA = Path(r"F:\nds\output\unet_v3\data")
INIT = r"F:\nds\output\unet_data\unet_best.pt"
OUT = r"F:\nds\output\unet_v3\unet_recall.pt"


class DS(Dataset):
    def __init__(self, split, augment=False, cap=None, seed=0):
        self.aug = augment
        index = json.load(open(DATA / "index.json", encoding="utf-8"))
        recs = [r for r in index if r["split"] == split]
        if cap:
            rng = np.random.default_rng(seed); rng.shuffle(recs)
            keep, acc = [], 0
            for r in recs:
                keep.append(r); acc += r["n"]
                if acc >= cap:
                    break
            recs = keep
        T = 256; N = sum(r["n"] for r in recs)
        self.imgs = np.empty((N, T, T, 3), np.uint8); self.masks = np.empty((N, T, T), np.uint8)
        self.wts = np.empty((N, T, T), np.float16); pos = 0
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
            # FADE: тянем к локальному светлому фону (выцветание бледной кривой) — КЛЮЧ для recall
            if np.random.rand() < 0.65:
                bgv = float(np.median(img)) if np.median(img) > 0.5 else 0.85
                f = np.random.uniform(0.15, 0.50)
                img = np.clip(img + (bgv - img) * f, 0, 1)
            if np.random.rand() < 0.5:
                img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
            if np.random.rand() < 0.5:
                img = np.clip((img - 0.5) * np.random.uniform(0.6, 1.4) + 0.5, 0, 1)
            if np.random.rand() < 0.5:
                img = np.clip(img ** np.random.uniform(0.7, 1.5), 0, 1)
            if np.random.rand() < 0.4:
                img = np.clip(img + np.random.normal(0, 0.025, img.shape).astype(np.float32), 0, 1)
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.from_numpy(msk).unsqueeze(0), torch.from_numpy(wt).unsqueeze(0))


def tversky_bce(logits, target, weight, alpha=0.3, beta=0.6, eps=1.0):
    """BCE(взвеш.) + Tversky(β>α → штраф FN = recall на бледном)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weight).sum() / weight.sum().clamp(min=1.0)
    p = torch.sigmoid(logits)
    tp = (p * target).sum((1, 2, 3)); fp = (p * (1 - target)).sum((1, 2, 3)); fn = ((1 - p) * target).sum((1, 2, 3))
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return bce + (1 - tv.mean())


def recall_prec(logits, target, thr=0.4, eps=1e-6):
    p = (torch.sigmoid(logits) > thr).float()
    tp = (p * target).sum((1, 2, 3)); fn = ((1 - p) * target).sum((1, 2, 3)); fp = (p * (1 - target)).sum((1, 2, 3))
    rec = ((tp + eps) / (tp + fn + eps)).mean().item()
    pre = ((tp + eps) / (tp + fp + eps)).mean().item()
    return rec, pre


def main():
    a = sys.argv[1:]
    epochs = int(a[a.index("--epochs") + 1]) if "--epochs" in a else 16
    bs = int(a[a.index("--bs") + 1]) if "--bs" in a else 24
    lr = float(a[a.index("--lr") + 1]) if "--lr" in a else 4e-4
    maxtr = int(a[a.index("--maxtrain") + 1]) if "--maxtrain" in a else 70000
    maxva = int(a[a.index("--maxval") + 1]) if "--maxval" in a else 12000
    out = a[a.index("--out") + 1] if "--out" in a else OUT
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DS("train", augment=True, cap=maxtr); va = DS("val", cap=maxva)
    print(f"device={dev} train={len(tr)} val={len(va)} epochs={epochs} bs={bs} lr={lr} fade+tversky -> {out}")
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    dl_va = DataLoader(va, batch_size=bs)
    net = UNet(in_ch=3, n_classes=1, base=32).to(dev)
    ck = torch.load(INIT, map_location=dev); net.load_state_dict(ck["model"])
    print(f"инициализация от unet_best (val_dice={ck.get('val_dice')})")
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = -1.0
    for ep in range(1, epochs + 1):
        net.train(); t0 = time.time(); losses = []
        for img, msk, wt in dl_tr:
            img, msk, wt = img.to(dev), msk.to(dev), wt.to(dev); opt.zero_grad()
            loss = tversky_bce(net(img), msk, wt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            losses.append(loss.item())
        net.eval(); recs, pres = [], []
        with torch.no_grad():
            for img, msk, wt in dl_va:
                img, msk = img.to(dev), msk.to(dev)
                r, p = recall_prec(net(img).float(), msk); recs.append(r); pres.append(p)
        rec, pre = float(np.mean(recs)), float(np.mean(pres))
        score = 2 * rec * pre / max(1e-6, rec + pre)
        print(f"ep {ep:>2}/{epochs} loss={np.mean(losses):.4f} val_recall={rec:.3f} val_prec={pre:.3f} F1={score:.3f} {time.time()-t0:.0f}s")
        if rec > best:                                      # СОХРАНЯЕМ ПО RECALL (цель: видеть бледное)
            best = rec
            torch.save({"model": net.state_dict(), "epoch": ep, "val_recall": rec, "val_prec": pre, "base": 32}, out)
            print(f"   * сохранён best-recall -> {out}")
    print(f"\nBEST val_recall={best:.3f} -> {out}")


if __name__ == "__main__":
    main()
