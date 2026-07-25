r"""trace_seq.py — ОБУЧЕННЫЙ ОКОННЫЙ СЕЛЕКТОР РАНОВ как замена жадного выбора (§6.66).

Жадный `trace2d.trace_line` решает по ОДНОЙ строке: берёт ран, ближайший к предсказанию, и назад
не смотрит. Замеры ветки показали, чем это кончается: отказ — не «прыжок на соседа», а
ПОВТОРЯЮЩИЕСЯ ЭКСКУРСИИ (§6.58), и вырезать их нельзя в принципе (§6.60). Гладкости одной тоже не
хватает: глобальный ДП уходит на соседнюю кривую целиком (§6.61), а признак идентичности,
измеряемый по одной строке, ломается на пересечениях (§6.64 — ширина штриха).

Работает ровно то, что видит КОНТЕКСТ: CNN сворачивает патч маски чернил ±64 строки вокруг
предсказания и выбирает ран по «идёт ли через эту колонку согласованный ход».

ЗАМЕР (`_pick_gate.py --seq`, три независимых набора, 66 листов / 260 кривых):
    жадный (прод)  66 честных   |  обученный  102   (+55%)
    p90 ошибки     349/233/287px|              174/208/202px
    точных строк   0.28/0.15/0.33|             0.53/0.40/0.64
    улучшено 24 листа, ДЕГРАДИРОВАЛО 2.

⚠ ВЫКЛЮЧЕН ПО УМОЛЧАНИЮ и включается только `CVParams.seq_model`. Torch НЕ становится жёсткой
зависимостью прода: без него (или без чекпойнта) пайплайн работает как раньше — см. `available()`.

⚠⚠ ОПРЕДЕЛЕНИЕ СЕТИ ЗДЕСЬ ПРОДУБЛИРОВАНО из `digitizer/rnd/_decoder_seq.py` СОЗНАТЕЛЬНО: прод не
должен импортировать исследовательские модули. Источник истины для ОБУЧЕНИЯ остаётся там.
★ Расхождение ловится тестом `selftest()`: он грузит чекпойнт и сверяет выход с эталонными
значениями; при любом расхождении архитектур веса либо не загрузятся, либо дадут другой ответ.
"""
import numpy as np

R_ROWS, ROW_STEP = 64, 4                 # ±64 строки, каждая 4-я  → 33 строки
R_COLS, COL_STEP = 96, 2                 # ±96 px, каждый 2-й      → 97 колонок
NROW = 2 * R_ROWS // ROW_STEP + 1
NCOL = 2 * R_COLS // COL_STEP + 1
MAXC = 6                                 # кандидатов, скорим только ближайшие к предсказанию
NF = 10                                  # признаков на кандидата


def available(model_path):
    """Есть ли torch И чекпойнт. Прод обязан работать без обоих."""
    if not model_path:
        return False
    try:
        import torch                                            # noqa: F401
    except Exception:
        return False
    from pathlib import Path
    return Path(model_path).is_file()


def _features(A, B, C, pred, x, v, base, n_pick=MAXC):
    """Признаки кандидатов строки. ИДЕНТИЧНЫ обучению (`_decoder_core.features`)."""
    if len(A) == 0:
        return np.array([], int), np.zeros((0, NF))
    order = np.argsort(np.abs(C - pred))[:n_pick]
    a = A[order].astype(np.float64); b = B[order].astype(np.float64); c = C[order]
    gap = np.maximum(np.maximum(a - pred, pred - b), 0.0)
    width = b - a
    overlap = ((a - 2 <= pred) & (pred <= b + 2)).astype(np.float64)
    nv = 0.6 * v + 0.4 * (c - x)
    widest = np.zeros(len(order)); widest[np.argmax(width)] = 1.0
    nearest = np.zeros(len(order)); nearest[np.argmin(np.abs(c - pred))] = 1.0
    X = np.stack([gap / 50.0, (c - pred) / 50.0, width / 20.0, np.abs(c - x) / 50.0,
                  np.abs(c - base) / 200.0, overlap, widest, nearest,
                  np.full(len(order), len(A) / 5.0), np.abs(nv - v) / 30.0], axis=1)
    return order, X


def _patch(band, lo, y, pred):
    """Окно вокруг (y, pred) из бандового растра. valid=0 за краем — модель обязана отличать
    «чисто» от «не знаем». ИДЕНТИЧНО обучению (`_decoder_seq_data.patch`)."""
    H, Wb = band.shape
    ys = y + np.arange(-R_ROWS, R_ROWS + 1, ROW_STEP)
    xs = int(round(pred)) - lo + np.arange(-R_COLS, R_COLS + 1, COL_STEP)
    vy = (ys >= 0) & (ys < H)
    vx = (xs >= 0) & (xs < Wb)
    ink = band[np.ix_(np.clip(ys, 0, H - 1), np.clip(xs, 0, Wb - 1))].astype(bool)
    val = vy[:, None] & vx[None, :]
    return ink & val, val


def _build_net(torch, nn):
    class WindowSelector(nn.Module):
        def __init__(self, ch=32, emb=48, ctx=3):
            super().__init__()
            self.ctx = ctx
            self.cnn = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, ch, 3, stride=(2, 1), padding=1), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.ReLU(),
            )
            self.col = nn.Sequential(nn.Conv1d(ch, emb, 3, padding=1), nn.ReLU(),
                                     nn.Conv1d(emb, emb, 3, padding=1), nn.ReLU())
            self.trunk = nn.Sequential(nn.Linear(emb * (2 * ctx + 1) + NF, 128), nn.ReLU(),
                                       nn.Linear(128, 64), nn.ReLU())
            self.head = nn.Linear(64, 1)
            self.head_off = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                          nn.Linear(32, 1), nn.Tanh())

        def forward(self, P, F, M):
            B = P.shape[0]
            z = self.cnn(P).mean(dim=2)
            e = self.col(z)
            cidx = torch.round(F[:, :, 1] * 50.0 / COL_STEP).long() + (NCOL // 2)
            off = torch.arange(-self.ctx, self.ctx + 1, device=P.device)
            g = (cidx.unsqueeze(-1) + off).clamp(0, NCOL - 1)
            idx = g.reshape(B, 1, -1).expand(-1, e.shape[1], -1)
            gathered = torch.gather(e, 2, idx).reshape(B, e.shape[1], MAXC, -1)
            gathered = gathered.permute(0, 2, 1, 3).reshape(B, MAXC, -1)
            h = self.trunk(torch.cat([gathered, F], dim=-1))
            return self.head(h).squeeze(-1).masked_fill(M == 0, -1e9)
    return WindowSelector


def make_tracer(model_path, device=None):
    """Вернуть функцию с сигнатурой `trace2d.trace_line`, но с обученным выбором рана.
    Дорогая часть (загрузка весов, сборка CUDA-графа) делается ОДИН раз на процесс."""
    import torch
    import torch.nn as nn
    from . import imaging as im
    from . import trace2d as T

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = _build_net(torch, nn)().to(dev)
    net.load_state_dict(torch.load(model_path, map_location=dev)["sd"])
    net.eval()

    # ★ CUDA-ГРАФ (§6.67): сеть крошечная, 89% времени — запуск ядер. Граф даёт ×9.6 при
    # ПОБИТОВО том же результате (проверено allclose). Без cuda — обычный вызов.
    G = {}
    if dev == "cuda":
        sP = torch.zeros(1, 2, NROW, NCOL, device=dev)
        sF = torch.zeros(1, MAXC, NF, device=dev)
        sM = torch.zeros(1, MAXC, device=dev)
        with torch.no_grad():
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    net(sP, sF, sM)
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                sOut = net(sP, sF, sM)
        G.update(g=gr, P=sP, F=sF, M=sM, out=sOut)

    def score(ink, val, X, n):
        pt = np.stack([ink, val]).astype(np.float32)[None]
        f = np.zeros((1, MAXC, NF), np.float32); f[0, :n] = X
        mm = np.zeros((1, MAXC), np.float32); mm[0, :n] = 1
        if not G:
            return net(torch.from_numpy(pt).to(dev), torch.from_numpy(f).to(dev),
                       torch.from_numpy(mm).to(dev))
        G["P"].copy_(torch.from_numpy(pt)); G["F"].copy_(torch.from_numpy(f))
        G["M"].copy_(torch.from_numpy(mm))
        G["g"].replay()
        return G["out"]

    def trace_line(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14, x_range=None,
                   jump_limit=None):
        H, W = fg.shape
        lo = max(0, int(x_range[0])) if x_range is not None else max(0, int(line.x_lo) - band_pad)
        hi = (min(W, int(x_range[1]) + 1) if x_range is not None
              else min(W, int(line.x_hi) + band_pad + 1))
        base = line.x_center
        band = np.ascontiguousarray(fg[:, lo:hi] > 0)
        x = None; v = 0.0; tr = {}
        with torch.no_grad():
            for y in range(max(0, line.y0), min(H, line.y1 + 1)):
                runs = im.row_runs(fg[y, lo:hi])
                if not runs:
                    if x is not None:                       # коаст через короткий разрыв
                        x = x + float(np.clip(v, -slmax, slmax))
                    continue
                A = np.array([r[0] + lo for r in runs])
                Bb = np.array([r[1] + lo for r in runs])
                C = np.array([r[2] + lo for r in runs], float)
                if x is None:
                    k = int(np.argmin(np.abs(C - base)))
                    x = float(C[k]); v = 0.0; tr[y] = x
                    continue
                pred = x + float(np.clip(v, -slmax, slmax))
                idx, X = _features(A, Bb, C, pred, x, v, base)
                if len(idx) == 1:                           # выбора нет — сеть не нужна
                    k = int(idx[0])
                else:
                    ink, val = _patch(band, lo, y, pred)
                    sc = score(ink, val, X, len(idx))
                    k = int(idx[int(sc[0].argmax().item())])
                nx = float(C[k])
                v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
        T._extend_ends(tr, fg, lo, hi, slmax)
        return tr

    return trace_line
