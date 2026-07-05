r"""Трек 2 / вариант 4 (05.07) — обучение UNetEmb (prob + эмбеддинг идентичности).
Атакует идентичность в НАЛОЖЕНИИ обученной ассоциацией (не величиной prob). prob-
голова = как в train_mk (BCE+dice, ignore-weighted); emb-голова = discriminative
loss на ЧИСТЫХ (не-наложенных) пикселях MGZ/MPZ. Датасет по умолчанию mk_data_v8
(эталоны исключены — гейт честный held-out). Запуск venv ComfyUI (RTX5080):
  <venv>\python.exe train_mk_emb.py [--data F:\nds\output\mk_data_v8 --epochs 28
     --bs 8 --base 48 --emb 8 --lam 0.5 --maxtrain 0]  -> <data>\mk_emb.pt
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_mk as T                       # MK dataset, loss_fn (prob)
from unet_emb import UNetEmb, discriminative_loss


@torch.no_grad()
def validate(model, dl, dev):
    """prob dice (MGZ,MPZ) + emb-сепарация (inter/intra центров кластеров на чистых пикс)."""
    model.eval(); ds = [0.0, 0.0]; n = 0; seps = []
    for img, msk in dl:
        img, msk = img.to(dev), msk.to(dev)
        logits, emb = model(img)
        p = (torch.sigmoid(logits) > 0.5).float()
        for c in range(2):
            num = 2 * (p[:, c] * msk[:, c]).sum((1, 2))
            den = p[:, c].sum((1, 2)) + msk[:, c].sum((1, 2)) + 1e-6
            ds[c] += (num / den).sum().item()
        n += len(img)
        ch0 = (msk[:, 0] > 0.5) & ~(msk[:, 1] > 0.5)
        ch1 = (msk[:, 1] > 0.5) & ~(msk[:, 0] > 0.5)
        for bi in range(len(img)):
            if int(ch0[bi].sum()) < 5 or int(ch1[bi].sum()) < 5:
                continue
            e0 = emb[bi, :, ch0[bi]]; e1 = emb[bi, :, ch1[bi]]
            mu0, mu1 = e0.mean(1), e1.mean(1)
            intra = ((e0 - mu0[:, None]).norm(dim=0).mean() + (e1 - mu1[:, None]).norm(dim=0).mean()) / 2
            inter = (mu0 - mu1).norm()
            seps.append(float(inter / (intra + 1e-6)))
    return ds[0] / n, ds[1] / n, (float(np.mean(seps)) if seps else 0.0)


def main():
    a = sys.argv[1:]
    ep = 28; bs = 8; lr = 4e-4; base = 48; emb = 8; lam = 0.5
    data = r"F:\nds\output\mk_data_v8"
    if "--data" in a: data = a[a.index("--data") + 1]
    if "--epochs" in a: ep = int(a[a.index("--epochs") + 1])
    if "--bs" in a: bs = int(a[a.index("--bs") + 1])
    if "--lr" in a: lr = float(a[a.index("--lr") + 1])
    if "--base" in a: base = int(a[a.index("--base") + 1])
    if "--emb" in a: emb = int(a[a.index("--emb") + 1])
    if "--lam" in a: lam = float(a[a.index("--lam") + 1])
    if "--maxtrain" in a: T.MK.CAP = int(a[a.index("--maxtrain") + 1])
    T.DATA = Path(data)
    scale = int(a[a.index("--scale") + 1]) if "--scale" in a else 1
    OUT = Path(data) / "mk_emb.pt"
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(T.MK("train", augment=True), batch_size=bs, shuffle=True, num_workers=0)
    va = DataLoader(T.MK("val"), batch_size=bs, num_workers=0)
    tile = int(tr.dataset.imgs.shape[1])
    print(f"device={dev} train={len(tr.dataset)} val={len(va.dataset)} ep={ep} bs={bs} "
          f"base={base} emb={emb} lam={lam} tile={tile}")
    model = UNetEmb(in_ch=4, n_prob=2, emb_dim=emb, base=base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    best = -1
    for e in range(ep):
        model.train(); t0 = time.time(); tp = te = 0.0
        for img, msk in tr:
            img, msk = img.to(dev), msk.to(dev)
            logits, embv = model(img)
            lp = T.loss_fn(logits, msk)
            ch0 = (msk[:, 0] > 0.5) & ~(msk[:, 1] > 0.5)
            ch1 = (msk[:, 1] > 0.5) & ~(msk[:, 0] > 0.5)
            le = discriminative_loss(embv, [ch0, ch1])
            loss = lp + lam * le
            opt.zero_grad(); loss.backward(); opt.step()
            tp += float(lp); te += float(le)
        sched.step()
        dm, dp, sep = validate(model, va, dev)
        score = (dm + dp) / 2 + 0.02 * min(sep, 10)        # dice + лёгкий бонус за emb-сепарацию
        print(f"ep{e+1:2d} Lprob={tp/len(tr):.3f} Lemb={te/len(tr):.3f} dice MGZ={dm:.3f} "
              f"MPZ={dp:.3f} emb_sep={sep:.2f} score={score:.3f} ({time.time()-t0:.0f}s)")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": e + 1, "val_dice": (dm + dp) / 2,
                        "emb_sep": sep, "score": score, "base": base, "emb_dim": emb,
                        "in_ch": 4, "n_prob": 2, "scale": scale, "tile": tile, "emb": True}, OUT)
            print(f"   ✓ best -> {OUT}")
    print(f"DONE best score={best:.3f}")


if __name__ == "__main__":
    main()
