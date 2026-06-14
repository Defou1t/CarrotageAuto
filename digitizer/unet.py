"""
unet.py — компактный U-Net на чистом PyTorch (без segmentation_models_pytorch,
чтобы не ставить пакеты в venv ComfyUI). Бинарная сегментация штриха кривой.
Запускать интерпретатором venv ComfyUI (torch 2.11+cu130, RTX 5080).
"""
import torch
import torch.nn as nn


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=1, base=32):
        super().__init__()
        b = base
        self.d1 = double_conv(in_ch, b)
        self.d2 = double_conv(b, b*2)
        self.d3 = double_conv(b*2, b*4)
        self.d4 = double_conv(b*4, b*8)
        self.bott = double_conv(b*8, b*16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.u4 = double_conv(b*16, b*8)
        self.up3 = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.u3 = double_conv(b*8, b*4)
        self.up2 = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.u2 = double_conv(b*4, b*2)
        self.up1 = nn.ConvTranspose2d(b*2, b, 2, stride=2)
        self.u1 = double_conv(b*2, b)
        self.out = nn.Conv2d(b, n_classes, 1)

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
        return self.out(x)


def dice_bce_loss(logits, target, eps=1.0):
    """BCE + soft Dice (для разреженного штриха при сильном дисбалансе классов)."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = 1 - (num / den).mean()
    return bce + dice


if __name__ == "__main__":
    m = UNet()
    x = torch.randn(2, 3, 256, 256)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"UNet OK: in {tuple(x.shape)} -> out {tuple(y.shape)}, params {n/1e6:.2f}M")
