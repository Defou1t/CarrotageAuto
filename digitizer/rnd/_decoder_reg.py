r"""_decoder_reg.py — ВТОРАЯ ВЕТКА §6.24: ПОЛНАЯ РЕГРЕССИЯ x(строка) окном (фоновая ставка).

Постановка §6.8 в чистом виде: раны как сущность исчезают, модель по растру полосы предсказывает
координату кривой на каждой строке. Вход чанка — окно CH строк × ±RC px, центрированное на
ТОЧКЕ ВХОДА (где кривая была на первой строке чанка); выход — softmax по колонкам на каждой
строке. Идентичность задана входом (точка входа + непрерывность), а не вычитается из маски —
именно этим постановка отличается от закрытого U-Net-разделителя (§6.3/§6.8).

★ ЧЕСТНОСТЬ ИНФЕРЕНСА: чанки СЦЕПЛЯЮТСЯ по собственному предсказанию (точка входа следующего
чанка — то, что модель сама выдала), teacher-forcing на инференсе нет. Это прямой ответ на
урок §6.24 итер.1 (exposure bias).

★ ГЕЙТ ТОТ ЖЕ: `_relatch_bench`, 5 держанных скважин, 24 кривые, честные = med<=3px И cov>=0.9.
Пока честных не больше 3/24 — ветка ни на что не претендует (durable-правило трека 2).

  <ComfyUI>\python_embeded\python.exe _decoder_reg.py --epochs 6
"""
import sys, argparse, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import _relatch_bench as BE
from _decoder_seq import csr_band

REG = Path(r"F:\nds\output\taskS\decoder\reg")
CKPT = Path(r"F:\nds\output\taskS\decoder\reg_model.pt")

CH = 192          # строк в чанке
RC = 96           # ±RC px вокруг точки входа
NCOL = 2 * RC + 1
ADV = 96          # на сколько строк сдвигается окно на инференсе (перецентровка)


class RowDecoder(nn.Module):
    """Растр окна → логиты по колонкам на каждой строке. Дилатации ТОЛЬКО по строкам: контекст
    нужен вдоль глубины (десятки строк), колонки остаются в исходном разрешении — это координата."""

    def __init__(self, ch=24, dil=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, (5, 3), padding=(2 * d, 1), dilation=(d, 1)), nn.ReLU(),
                          nn.Conv2d(ch, ch, 1), nn.ReLU())
            for d in dil])
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        z = self.inp(x)
        for b in self.blocks:
            z = z + b(z)
        return self.out(z).squeeze(1)              # (B, CH, NCOL) — логиты по колонкам


class Packed:
    """Построчно упакованная маска полосы: индексируется как bool-массив на НАБОРЕ строк.
    Нужна, чтобы 25 листов помещались в RAM (развёрнутыми это ~10 ГБ)."""

    def __init__(self, bits, shape):
        self.bits = bits; self.shape = shape

    def rows(self, r):
        return np.unpackbits(self.bits[r], axis=1)[:, :self.shape[1]].astype(bool)


def load_curves():
    """Все кривые всех листов + скважина (сплит по скважинам, не по чанкам)."""
    out = []
    for f in sorted(REG.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        well = str(d["well"])
        for i in range(int(d["n"])):
            sh = tuple(d[f"shape_{i}"])
            # маска остаётся УПАКОВАННОЙ (построчно), разворачивается только окно в crop()
            out.append({"band": Packed(d[f"band_{i}"], sh), "gt": d[f"gt_{i}"], "lo": int(d["meta"][i][0]),
                        "y0": int(d["meta"][i][1]), "well": well, "name": str(d["names"][i])})
    return out


def crop(band, lo, r, cx):
    """Окно CH строк с локальной строки r, колонки cx±RC (cx — АБСОЛЮТНЫЙ x). (2,CH,NCOL).
    `band` — Packed (обучение) или обычный bool-массив (инференс, растр из CSR-кэша bench)."""
    Hb, Wb = band.shape
    rows = r + np.arange(CH)
    cols = int(round(cx)) - lo + np.arange(-RC, RC + 1)
    vy = (rows >= 0) & (rows < Hb); vx = (cols >= 0) & (cols < Wb)
    cr = np.clip(rows, 0, Hb - 1); cc = np.clip(cols, 0, Wb - 1)
    ink = band.rows(cr)[:, cc] if isinstance(band, Packed) else band[np.ix_(cr, cc)]
    val = vy[:, None] & vx[None, :]
    return np.stack([ink & val, val]).astype(np.float32)


def sample_batch(curves, rng, bs):
    X = np.zeros((bs, 2, CH, NCOL), np.float32)
    T = np.zeros((bs, CH), np.int64)
    M = np.zeros((bs, CH), np.float32)
    for i in range(bs):
        c = curves[rng.integers(len(curves))]
        n = len(c["gt"])
        r = int(rng.integers(0, max(1, n - CH)))
        cx = float(c["gt"][r])                       # точка входа = истина на первой строке чанка
        X[i] = crop(c["band"], c["lo"], r, cx)
        seg = c["gt"][r:r + CH]
        t = np.round(seg - (round(cx) - RC)).astype(np.int64)
        ok = (t >= 0) & (t < NCOL)
        T[i, :len(t)] = np.clip(t, 0, NCOL - 1); M[i, :len(t)] = ok
    return X, T, M


def train(epochs, steps, bs, lr, dev):
    curves = load_curves()
    wells = sorted({c["well"] for c in curves})
    val_w = set(wells[::5])
    tr = [c for c in curves if c["well"] not in val_w]
    va = [c for c in curves if c["well"] in val_w]
    print(f"кривых {len(curves)}: train {len(tr)} / val {len(va)} (валидационные {sorted(val_w)})")
    net = RowDecoder().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    lossf = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.02)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        t0 = time.time(); run = 0.0
        for _ in range(steps):
            X, T, M = sample_batch(tr, rng, bs)
            x = torch.from_numpy(X).to(dev); t = torch.from_numpy(T).to(dev)
            m = torch.from_numpy(M).to(dev)
            lg = net(x)
            l = (lossf(lg.reshape(-1, NCOL), t.reshape(-1)) * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sched.step()
            run += l.item()
        # ВАЛ: медианная |px| ошибка на чанках держанных скважин (teacher-forced вход, но
        # предсказание собственное — это ВЕРХНЯЯ оценка, честный замер даёт только гейт bench)
        net.eval(); errs = []
        with torch.no_grad():
            for _ in range(20):
                X, T, M = sample_batch(va, rng, bs)
                p = net(torch.from_numpy(X).to(dev)).argmax(-1).cpu().numpy()
                d = np.abs(p - T)[M > 0]
                errs.append(np.median(d) if len(d) else np.nan)
        net.train()
        print(f"эпоха {ep+1}/{epochs}  loss {run/steps:.4f}  ВАЛ med |px| {np.nanmedian(errs):.1f}  [{time.time()-t0:.0f}с]")
    torch.save({"sd": net.state_dict(), "geom": [CH, RC, ADV]}, CKPT)
    print(f"-> {CKPT}")
    return net


def make_tracer(net, dev):
    """Трассировщик для bench: цепочка чанков, точка входа следующего — СОБСТВЕННОЕ предсказание."""
    cache = {}

    def tracer(rec, csr, H):
        key = (id(csr), rec["lo"], rec["hi"])
        if key not in cache:
            cache.clear(); cache[key] = csr_band(csr, rec["lo"], rec["hi"], H)
        band = cache[key]
        lo = rec["lo"]; base = rec["base"]
        y0, y1 = max(0, rec["y0"]), min(H - 1, rec["y1"])
        A, B, C = BE._runs_at(csr, y0)                # старт — как в базе: ближайший к base ран
        cx = float(C[int(np.argmin(np.abs(C - base)))]) if len(C) else base
        tr = {}; y = y0
        with torch.no_grad():
            while y <= y1:
                x = torch.from_numpy(crop(band, lo, y, cx)[None]).to(dev)
                p = net(x)[0].argmax(-1).cpu().numpy() + (round(cx) - RC)
                n = min(ADV, y1 - y + 1)
                for k in range(min(CH, y1 - y + 1)):
                    tr[y + k] = float(p[k])
                cx = float(p[min(n, CH - 1)])
                y += n
        return tr
    return tracer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"устройство: {dev}")
    if a.gate_only:
        net = RowDecoder().to(dev); net.load_state_dict(torch.load(CKPT, map_location=dev)["sd"])
    else:
        net = train(a.epochs, a.steps, a.bs, a.lr, dev)
    net.eval()
    if not a.no_gate:
        print(f"\n{'='*66}\n=== ГЕЙТ на держанных скважинах (bench, 24 кривые) ===")
        BE.report("БАЗА (прод)", BE.run_strategy())
        BE.report("РЕГРЕССИЯ x(строка), окно", BE.run_strategy(tracer=make_tracer(net, dev)))
