"""§6.325: «настоящая ли трасса» по кусочкам скана вдоль кандидата выбора трассы.

Кусочки — ТО ЖЕ, что `digitizer/rnd/_trace_patch.py`: `NP` точек равномерно по строкам трассы (без крайних), кусок `S`² вокруг
точки; канал 0 — темнота (бумага строки − серый), канал 1 — маска трассы (±1 px по x). Сеть `Net` — та же, что учит
`digitizer/rnd/_trace_cnn.py`. Веса — `trace_cnn_*.pt` рядом с моделью выбора трассы (`line_choice_*.pkl` → `trace_cnn_*.pt`)."""
from pathlib import Path
import numpy as np

NP, S = 8, 40
_NET = {}


def patches(gray, paper, d, np_=NP, s_=S):
    """d — плотная трасса {строка: x}; paper — бумага по строкам (`level_ink.paper_of`). → uint8 [np_, 2, s_, s_] или None"""
    ys = np.array(sorted(d))
    if len(ys) < 50:
        return None
    H, W = gray.shape; h = s_ // 2
    pts = ys[np.linspace(0, len(ys) - 1, np_ + 2).round().astype(int)[1:-1]]
    P = np.zeros((np_, 2, s_, s_), np.uint8)
    for j, y in enumerate(pts):
        x = int(round(d[int(y)])); y = int(y)
        y0, x0 = y - h, x - h
        ya, yb, xa, xb = max(0, y0), min(H, y0 + s_), max(0, x0), min(W, x0 + s_)
        if ya >= yb or xa >= xb:
            continue
        g = gray[ya:yb, xa:xb].astype(np.int16)
        P[j, 0, ya - y0:yb - y0, xa - x0:xb - x0] = np.clip(paper[ya:yb, None].astype(np.int16) - g, 0, 255).astype(np.uint8)
        for yy in range(ya, yb):
            if yy in d:
                xx = int(round(d[yy])) - x0
                if 0 <= xx < s_:
                    P[j, 1, yy - y0, max(0, xx - 1):xx + 2] = 255
    return P


def build_net(k_out=1):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
            s.f = nn.Sequential(c(2, 32), c(32, 32), nn.MaxPool2d(2), c(32, 64), c(64, 64), nn.MaxPool2d(2), c(64, 96),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            s.h = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, k_out))

        def forward(s, x):                              # x [B, NP, 2, S, S]
            import torch
            B, n = x.shape[:2]
            e = s.f(x.flatten(0, 1)).view(B, n, -1)
            o = s.h(torch.cat([e.mean(1), e.max(1).values], 1))
            return o[:, 0] if k_out == 1 else o
    return Net()


def cnn_path(lc_path):
    """line_choice_*.pkl → trace_cnn_*.pt рядом"""
    p = Path(lc_path)
    return p.with_name(p.name.replace("line_choice", "trace_cnn").replace(".pkl", ".pt"))


def _load(path):
    key = str(path)
    if key not in _NET:
        import torch
        dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
        net = build_net(); net.load_state_dict(torch.load(key, map_location="cpu")); net.eval()
        _NET[key] = (net.to(dev), dev)
    return _NET[key]


def scores(gray, paper, dense_list, path, bs=256):
    """→ np.array оценок (nan — короткая трасса)"""
    import torch
    net, dev = _load(path)
    Ps = [patches(gray, paper, d) for d in dense_list]
    out = np.full(len(Ps), np.nan, np.float32)
    idx = [i for i, p in enumerate(Ps) if p is not None]
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.from_numpy(np.stack([Ps[j] for j in b])).to(dev).float() / 255.0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out[b] = torch.sigmoid(net(xb).float()).cpu().numpy()
    return out
