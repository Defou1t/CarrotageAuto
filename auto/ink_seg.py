r"""ink_seg.py — СЕТЬ «ТУШЬ ОЦИФРОВАННЫХ КРИВЫХ» ПО ЦВЕТНОМУ СКАНУ (24.10, §6.345).

Декодер (`rowdec`) видит одно серое «темнее бумаги» без сетки и прыгает по рукописным отметкам глубины, подписям и линиям сетки.
Оракул §6.345: карта декодера, погашенная везде, кроме туши кривых эталона, даёт честный путь 31% слотов, где его не было. Эта сеть —
небольшая U-Net по RGB (`digitizer/rnd/_seg_net.py`, цель — ±2 px от ломаной эксперта); её вероятность гасит карту декодера:
prob × (m + (1 − m)·seg). Веса — {sd, base}; torch грузится лениво.
"""
import re
import numpy as np

_NET = {}
# §6.350: семейства сети туши по семействам (канал 1 + k — FAMS[k]; последний — прочие)
FAMS = ["GZ", "SP", "GK", "OGZ", "DS", "BK", "ZAT", "CALI", "NGK", "MBK", "IK", "T", "TM", "MGZ", "MPZ", "PZ", "OTHER"]


def fam_index(root):
    r = re.sub(r"^BKZ_", "", str(root))
    return FAMS.index(r) if r in FAMS[:-1] else len(FAMS) - 1


def build_net(base=32, out=1):
    import torch.nn as nn

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(s):
            super().__init__()
            c = [base, base * 2, base * 4, base * 8]
            s.e = nn.ModuleList([blk(3, c[0]), blk(c[0], c[1]), blk(c[1], c[2]), blk(c[2], c[3])])
            s.mid = blk(c[3], c[3])
            s.up = nn.ModuleList([nn.ConvTranspose2d(c[3], c[3], 2, 2), nn.ConvTranspose2d(c[3], c[2], 2, 2),
                                  nn.ConvTranspose2d(c[2], c[1], 2, 2), nn.ConvTranspose2d(c[1], c[0], 2, 2)])
            s.d = nn.ModuleList([blk(2 * c[3], c[3]), blk(2 * c[2], c[2]), blk(2 * c[1], c[1]), blk(2 * c[0], c[0])])
            s.out = nn.Conv2d(c[0], out, 1)           # §6.350: out > 1 — канал 0 тушь, далее семейства
            s.pool = nn.MaxPool2d(2)

        def forward(s, x):
            import torch
            sk = []
            for e in s.e:
                x = e(x); sk.append(x); x = s.pool(x)
            x = s.mid(x)
            for u, d, k in zip(s.up, s.d, reversed(sk)):
                x = d(torch.cat([u(x), k], 1))
            return s.out(x)[:, 0] if out == 1 else s.out(x)

    return UNet()


def load(path):
    import torch
    key = str(path)
    if key not in _NET:
        ck = torch.load(key, map_location="cpu", weights_only=False)
        net = build_net(int(ck.get("base", 32)), int(ck.get("out", 1)))
        net.load_state_dict(ck["sd"]); net.eval()
        dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
        _NET[key] = (net.to(dev), dev)
    return _NET[key]


def prob_map(rgb, y0, y1, x0, x1, path, tile=1024, ov=64):
    """→ float32 [y1 − y0, x1 − x0] — вероятность туши оцифрованной кривой (плитки tile² с полями ov); у сети по семействам
    (§6.350, out > 1) — [out, y1 − y0, x1 − x0]."""
    import torch
    net, dev = load(path)
    H, W = rgb.shape[:2]
    y0, y1, x0, x1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
    out = None
    step = tile - 2 * ov
    for ty in range(y0, y1, step):
        for tx in range(x0, x1, step):
            a0, a1 = max(0, ty - ov), min(H, ty + step + ov)
            b0, b1 = max(0, tx - ov), min(W, tx + step + ov)
            ph, pw = (-(a1 - a0)) % 16, (-(b1 - b0)) % 16
            sub = rgb[a0:a1, b0:b1]
            if ph or pw:
                sub = np.pad(sub, ((0, ph), (0, pw), (0, 0)), mode="edge")
            with torch.no_grad():
                x = torch.from_numpy(np.ascontiguousarray(sub.transpose(2, 0, 1)))[None].to(dev).float() / 255.0
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):   # без CUDA autocast("cuda") падает
                    pr = torch.sigmoid(net(x).float())[0].cpu().numpy()
            r0, r1 = ty, min(y1, ty + step); c0, c1 = tx, min(x1, tx + step)
            if out is None:
                out = np.zeros(pr.shape[:-2] + (y1 - y0, x1 - x0), np.float32)
            out[..., r0 - y0:r1 - y0, c0 - x0:c1 - x0] = pr[..., r0 - a0:r1 - a0, c0 - b0:c1 - b0]
    return out if out is not None else np.zeros((max(0, y1 - y0), max(0, x1 - x0)), np.float32)
