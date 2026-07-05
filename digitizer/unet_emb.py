"""
unet_emb.py — U-Net с ДВУМЯ головами для трека 2 (вариант 4, 05.07): к обычным
2 prob-каналам (MGZ/MPZ детекция) добавлена ЭМБЕДДИНГ-голова (E-мерная на пиксель).
Идея (associative embedding, De Brabandere 2017): пиксели одной кривой стягиваются
в эмбеддинге, разных — расталкиваются ⇒ идентичность из ОБУЧЕННОЙ АССОЦИАЦИИ, а не
из величины prob (которая размывается в наложении).

⚠ ОГРАНИЧЕНИЕ: пиксель на ИСТИННОМ наложении принадлежит обеим кривым — один
эмбеддинг двух не выразит. Поэтому emb-лосс учим на ЧИСТЫХ (не-наложенных) пикселях
(ch0-only / ch1-only); они якорят 2 кластера, наложение решается близостью. prob-
голова остаётся (для полосы/детекции). Запуск venv ComfyUI (torch/RTX5080).
"""
import torch
import torch.nn as nn
from unet import double_conv


class UNetEmb(nn.Module):
    def __init__(self, in_ch=4, n_prob=2, emb_dim=8, base=48):
        super().__init__()
        b = base
        self.d1 = double_conv(in_ch, b)
        self.d2 = double_conv(b, b * 2)
        self.d3 = double_conv(b * 2, b * 4)
        self.d4 = double_conv(b * 4, b * 8)
        self.bott = double_conv(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.u4 = double_conv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.u3 = double_conv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.u2 = double_conv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.u1 = double_conv(b * 2, b)
        self.out = nn.Conv2d(b, n_prob, 1)                  # prob (MGZ/MPZ)
        self.emb = nn.Conv2d(b, emb_dim, 1)                 # эмбеддинг идентичности
        self.emb_dim = emb_dim

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(bn), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x), self.emb(x)


def discriminative_loss(emb, inst_masks, dv=0.5, dd=1.5, a=1.0, bpush=1.0, greg=1e-3):
    """emb: B×E×H×W. inst_masks: список из 2 масок (B×H×W bool) — ЧИСТЫЕ пиксели
    MGZ / MPZ (без наложения). Discriminative loss: var (стягивание к центру
    инстанса за полем dv) + dist (расталкивание центров за 2·dd) + reg."""
    B = emb.shape[0]
    Lv = Ld = Lr = 0.0
    nb = 0
    for bi in range(B):
        means = []
        var = 0.0
        cnt = 0
        for m in inst_masks:
            sel = m[bi]
            n = int(sel.sum())
            if n < 5:
                continue
            e = emb[bi, :, sel]                              # E×n
            mu = e.mean(1, keepdim=True)                     # E×1
            means.append(mu.squeeze(1))
            d = (e - mu).norm(dim=0)                         # n
            var = var + torch.clamp(d - dv, min=0).pow(2).mean()
            cnt += 1
        if cnt == 0:
            continue
        Lv = Lv + var / cnt
        if cnt >= 2:
            mu = torch.stack(means)                          # C×E
            dif = (mu.unsqueeze(0) - mu.unsqueeze(1)).norm(dim=2)  # C×C
            C = mu.shape[0]
            eye = torch.eye(C, device=emb.device).bool()
            push = torch.clamp(2 * dd - dif[~eye], min=0).pow(2).mean()
            Ld = Ld + push
        Lr = Lr + torch.stack(means).norm(dim=1).mean()
        nb += 1
    if nb == 0:
        return emb.sum() * 0.0
    return (a * Lv + bpush * Ld + greg * Lr) / nb


if __name__ == "__main__":
    m = UNetEmb(in_ch=4, emb_dim=8, base=32)
    x = torch.randn(2, 4, 128, 128)
    pr, em = m(x)
    print(f"UNetEmb OK: prob {tuple(pr.shape)} emb {tuple(em.shape)}")
    m0 = torch.zeros(2, 128, 128, dtype=torch.bool); m0[:, 40:60, 40:60] = True
    m1 = torch.zeros(2, 128, 128, dtype=torch.bool); m1[:, 70:90, 70:90] = True
    L = discriminative_loss(em, [m0, m1])
    print("disc_loss:", float(L))
