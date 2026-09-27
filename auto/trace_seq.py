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
★ Расхождение ловится `selftest()` — он сверяет признаки, патч и выход сети с эталонами, которые
СГЕНЕРИРОВАНЫ обучающей стороной и лежат в `models/<модель>.golden.json`. Утверждение «веса либо
не загрузятся, либо дадут другой ответ» неверно как гарантия: геометрия окна (NROW/NCOL/MAXC/
COL_STEP) не входит ни в одну форму весов, поэтому её дрейф `load_state_dict` не видит вовсе —
для этого в чекпойнте есть ключ `geom`, и его сверяет `_check_geom` при каждой загрузке.
"""
from pathlib import Path

import numpy as np

R_ROWS, ROW_STEP = 64, 4                 # ±64 строки, каждая 4-я  → 33 строки
R_COLS, COL_STEP = 96, 2                 # ±96 px, каждый 2-й      → 97 колонок
NROW = 2 * R_ROWS // ROW_STEP + 1
NCOL = 2 * R_COLS // COL_STEP + 1
MAXC = 6                                 # кандидатов, скорим только ближайшие к предсказанию
NF = 10                                  # признаков на кандидата

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = "seq_model_d45p.pt"      # лежит В РЕПОЗИТОРИИ (§6.68), 359 КБ


def resolve(model_path):
    """Путь к чекпойнту → Path либо None (пусто = селектор выключен).

    ⚠ ПОЧЕМУ НЕ ПРОСТО Path(): до §6.68 путь в конфиге указывал в `output/taskS/decoder`, который
    есть ровно на одной машине, — у любого другого человека селектор молча не включался бы.
    Поэтому голое ИМЯ файла (без каталога) ищется в `auto/models/`, где вес лежит под контролем
    версий; путь с каталогом берётся как есть — это ход для чужого/экспериментального чекпойнта."""
    if not model_path:
        return None
    p = Path(model_path)
    return MODELS_DIR / p.name if p.name == str(model_path) else p


def _check_geom(ckpt):
    """Сверить геометрию окна с записанной в чекпойнт. Вернуть текст расхождения или "".

    ⚠ ЗАЧЕМ ОТДЕЛЬНО: NROW/NCOL/MAXC/COL_STEP не входят НИ В ОДНУ форму весов, поэтому
    `load_state_dict` их дрейф не видит — веса загрузятся, а ответ будет другой. Это единственное
    расхождение, которое иначе проходит совершенно молча.
    Чекпойнт без ключа `geom` (старее §6.66) не является ошибкой — сверять просто нечем."""
    want = [NROW, NCOL, MAXC, COL_STEP]
    got = list(ckpt.get("geom") or want)
    return ("" if got == want else
            f"геометрия чекпойнта {got} != геометрии кода {want} [NROW, NCOL, MAXC, COL_STEP]")


def available(model_path):
    """Есть ли torch И чекпойнт. Прод обязан работать без обоих."""
    p = resolve(model_path)
    if p is None:
        return False
    try:
        import torch                                            # noqa: F401
    except Exception:
        return False
    return p.is_file()


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


def _hist_chan(torch, P, H):
    """§6.226: канал истории своей трассы из номеров колонок H (B, NROW), −1 — нет. ИДЕНТИЧНО `_decoder_seq`."""
    hc = torch.zeros(P.shape[0], 1, P.shape[2], P.shape[3], device=P.device)
    ok = (H >= 0).float().unsqueeze(1).unsqueeze(-1)
    hc.scatter_(3, H.clamp(min=0).long().unsqueeze(1).unsqueeze(-1), ok)
    return torch.cat([P, hc], dim=1)


def hist_cols(hist, y, pred):
    """§6.226: история своей трассы в окне — колонка, где трасса была, для строк окна ВЫШЕ текущей (−1 — нет).
    ИДЕНТИЧНО `_decoder_seq_data.hist_cols` (обучение)."""
    h = np.full(NROW, -1, np.int64)
    for i in range(NROW):
        r = y - R_ROWS + i * ROW_STEP
        if r >= y:
            break
        x = hist.get(r)
        if x is not None:
            c = int(round((x - pred) / COL_STEP)) + NCOL // 2
            if 0 <= c < NCOL:
                h[i] = c
    return h


def _build_net_xl(torch, nn):
    """§6.233: СЕЛЕКТОР XL — ИДЕНТИЧНО `_decoder_seq.WindowSelectorXL` (сверка — `_seqxl_parity.py`)."""
    class WindowSelectorXL(nn.Module):
        def __init__(self, ch=48, ctx=3, hist=True, nblk=4):
            super().__init__()
            self.ctx, self.hist = ctx, hist
            self.stem = nn.Sequential(nn.Conv2d(3 if hist else 2, ch, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(ch, ch, 3, stride=(2, 1), padding=1), nn.ReLU())
            self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                                                       nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(nblk)])
            self.red = nn.Conv2d(ch, 16, 1)
            R = (NROW + 1) // 2
            self.trunk = nn.Sequential(nn.Linear(16 * R * (2 * ctx + 1) + NF, 256), nn.ReLU(),
                                       nn.Linear(256, 128), nn.ReLU())
            self.head = nn.Linear(128, 1)
            self.head_off = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

        def forward(self, P, F, M, H=None):
            B = P.shape[0]
            if self.hist:
                P = _hist_chan(torch, P, H)
            z = self.stem(P)
            for b in self.blocks:
                z = torch.relu(z + b(z))
            e = self.red(z)
            C_, R_, W_ = e.shape[1], e.shape[2], e.shape[3]
            cidx = torch.round(F[:, :, 1] * 50.0 / COL_STEP).long() + (NCOL // 2)
            off = torch.arange(-self.ctx, self.ctx + 1, device=P.device)
            g = (cidx.unsqueeze(-1) + off).clamp(0, NCOL - 1).reshape(B, 1, -1)
            flat = e.reshape(B, C_ * R_, W_)
            got = torch.gather(flat, 2, g.expand(-1, C_ * R_, -1))
            got = got.reshape(B, C_ * R_, MAXC, -1).permute(0, 2, 1, 3).reshape(B, MAXC, -1)
            h = self.trunk(torch.cat([got, F], dim=-1))
            return self.head(h).squeeze(-1).masked_fill(M == 0, -1e9)
    return WindowSelectorXL


def _build_net(torch, nn, hist=False):
    class WindowSelector(nn.Module):
        def __init__(self, ch=32, emb=48, ctx=3):
            super().__init__()
            self.ctx = ctx
            self.hist = hist                                # §6.226: третий канал — история (только у весов с `hist`)
            self.cnn = nn.Sequential(
                nn.Conv2d(3 if hist else 2, 16, 3, padding=1), nn.ReLU(),
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

        def forward(self, P, F, M, H=None):
            B = P.shape[0]
            if self.hist:
                P = _hist_chan(torch, P, H)
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

    # ⚠ УСТРОЙСТВО ВЫБИРАЕТСЯ ПО `device_count()`, А НЕ ПО `is_available()` (§6.106). При
    # `CUDA_VISIBLE_DEVICES=""` этот сборкой torch (2.10+cu130) `is_available()` возвращает True,
    # а `device_count()` — 0; тогда `map_location="cuda"` роняет ЗАГРУЗКУ ЧЕКПОЙНТА, то есть
    # пайплайн падает вместо того, чтобы честно считать на CPU. Поймано живьём при попытке увести
    # замер с занятой видеокарты.
    dev = device or ("cuda" if torch.cuda.device_count() > 0 else "cpu")
    ckpt = torch.load(resolve(model_path), map_location=dev, weights_only=False)
    bad = _check_geom(ckpt)
    if bad:
        # ⚠ ПАДАЕМ, А НЕ ОТКАТЫВАЕМСЯ НА ЖАДНЫЙ: селектор запросили явно, и тихая подмена его
        # чужой геометрией дала бы выдачу ХУЖЕ прода под видом улучшения. Молчаливый откат уместен
        # только когда модель не запрашивали (пустой seq_model) или нет torch — см. available().
        raise ValueError(f"{resolve(model_path)}: {bad}")
    arch = ckpt.get("arch", "base"); use_hist = bool(ckpt.get("hist", False))     # §6.233: по полям чекпойнта
    net = (_build_net_xl(torch, nn)(hist=use_hist) if arch == "xl" else _build_net(torch, nn, hist=use_hist)()).to(dev)
    net.load_state_dict(ckpt["sd"])
    net.eval()

    # ★ CUDA-ГРАФ (§6.67): сеть крошечная, 89% времени — запуск ядер. Граф даёт ×9.6 при
    # ПОБИТОВО том же результате (проверено allclose). Без cuda — обычный вызов.
    G = {}
    if dev == "cuda":
        # ★ §6.219 (25.09): вход сети — ОДИН закреплённый буфер на хосте и ОДИН буфер на GPU, виды которого захвачены
        #   графом; на строку — одно асинхронное копирование вместо трёх (было 62% времени листа). Выдача побайтно та же
        #   (сверка повтора с кэша трасс против A/B на всём поле, `tcache_parity.txt`); ≈ ×1.25 на лист.
        _nP, _nF, _nM = 2 * NROW * NCOL, MAXC * NF, MAXC
        dbuf = torch.zeros(_nP + _nF + _nM, device=dev)
        sP = dbuf[:_nP].view(1, 2, NROW, NCOL)
        sF = dbuf[_nP:_nP + _nF].view(1, MAXC, NF)
        sM = dbuf[_nP + _nF:].view(1, MAXC)
        hbuf = torch.zeros(_nP + _nF + _nM, pin_memory=True)
        hnp = hbuf.numpy()
        if use_hist:
            sH = torch.full((1, NROW), -1, dtype=torch.long, device=dev)
            hH = torch.full((NROW,), -1, dtype=torch.long).pin_memory()
        with torch.no_grad():
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    net(sP, sF, sM, sH) if use_hist else net(sP, sF, sM)
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                sOut = net(sP, sF, sM, sH) if use_hist else net(sP, sF, sM)
        G.update(g=gr, P=sP, F=sF, M=sM, out=sOut, dbuf=dbuf, hbuf=hbuf, hnp=hnp, n=(_nP, _nF, _nM))
        if use_hist:
            G.update(sH=sH, hH=hH, hHn=hH.numpy())

    def score(ink, val, X, n, hrow=None):
        pt = np.stack([ink, val]).astype(np.float32)[None]
        f = np.zeros((1, MAXC, NF), np.float32); f[0, :n] = X
        mm = np.zeros((1, MAXC), np.float32); mm[0, :n] = 1
        if not G:
            if use_hist:
                return net(torch.from_numpy(pt).to(dev), torch.from_numpy(f).to(dev), torch.from_numpy(mm).to(dev),
                           torch.from_numpy(hrow)[None].to(dev))
            return net(torch.from_numpy(pt).to(dev), torch.from_numpy(f).to(dev),
                       torch.from_numpy(mm).to(dev))
        _nP, _nF, _nM = G["n"]; h = G["hnp"]
        h[:_nP] = pt.reshape(-1); h[_nP:_nP + _nF] = f.reshape(-1); h[_nP + _nF:] = mm.reshape(-1)
        G["dbuf"].copy_(G["hbuf"], non_blocking=True)
        if use_hist:
            G["hHn"][:] = hrow
            G["sH"][0].copy_(G["hH"], non_blocking=True)
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
        x = None; v = 0.0; tr = {}; hist = {}
        with torch.no_grad():
            for y in range(max(0, line.y0), min(H, line.y1 + 1)):
                runs = im.row_runs(fg[y, lo:hi])
                if not runs:
                    if x is not None:                       # коаст через короткий разрыв
                        x = x + float(np.clip(v, -slmax, slmax))
                        if use_hist:
                            hist[y] = x
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
                    sc = score(ink, val, X, len(idx), hist_cols(hist, y, pred) if use_hist else None)
                    k = int(idx[int(sc[0].argmax().item())])
                nx = float(C[k])
                v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
                if use_hist:
                    hist[y] = x
        T._extend_ends(tr, fg, lo, hi, slmax)
        return tr

    return trace_line


def selftest(model_path=DEFAULT_MODEL, golden_path=None):
    """Поймать РАСХОЖДЕНИЕ прод-копии с обучающим источником истины. Список проблем; пусто = ок.

    ⚠ ЗАЧЕМ ОН ВООБЩЕ НУЖЕН. Сеть, патч и признаки продублированы здесь из `digitizer/rnd/*`
    сознательно (прод не должен импортировать исследовательский код) — но у дублирования есть
    цена: копия может тихо разъехаться с оригиналом, и тогда ТЕ ЖЕ веса дадут ДРУГОЙ ответ, ничего
    не сломав и не бросив исключения. Ни одна из таких ошибок не меняет форм весов:
    порядок каналов [ink, val], формула колонки кандидата, порядок конкатенации [gathered, F],
    перепутанные is_widest/is_nearest, потеря обнуления ink за краем, дрейф R_ROWS/COL_STEP.

    ★ Эталоны СГЕНЕРИРОВАНЫ ОБУЧАЮЩЕЙ СТОРОНОЙ (`digitizer/rnd/_decoder_seq.py`,
    `_decoder_core.py`) и лежат рядом с весами в `<модель>.golden.json`. Считай их эта же копия —
    тест сверял бы копию сама с собой и не ловил бы ничего. Пересобирать эталоны ТОЛЬКО вместе со
    сменой чекпойнта, скриптом из заголовка json.

    Требует torch; без него возвращает единственную строку-предупреждение, а не падает."""
    import json
    import hashlib

    bad = []
    mp = resolve(model_path)
    if mp is None or not mp.is_file():
        return [f"нет чекпойнта: {mp}"]
    gp = Path(golden_path) if golden_path else mp.with_suffix(".golden.json")
    if not gp.is_file():
        return [f"нет эталонов: {gp}"]
    try:
        import torch
    except Exception as e:
        return [f"torch недоступен, сверка невозможна: {e}"]

    g = json.loads(gp.read_text(encoding="utf-8"))
    ckpt = torch.load(mp, map_location="cpu", weights_only=False)

    # A. ГЕОМЕТРИЯ И КОНСТАНТЫ. Единственные величины, которые меняют ответ и не входят в формы
    #    весов, — их обязан сверять кто-то явно, иначе дрейф проходит молча.
    e = _check_geom(ckpt)
    if e:
        bad.append("A/геометрия: " + e)
    if list(g["geom"]) != [NROW, NCOL, MAXC, COL_STEP]:
        bad.append(f"A/эталон: geom эталона {g['geom']} != кода {[NROW, NCOL, MAXC, COL_STEP]}")
    sha = hashlib.sha256(mp.read_bytes()).hexdigest()
    if sha != g["ckpt_sha256"]:
        bad.append(f"A/чекпойнт: sha256 {sha[:12]} != эталонного {g['ckpt_sha256'][:12]} — "
                   f"веса заменены, эталоны надо пересобрать")

    # B. ИНВЕНТАРЬ ВЕСОВ. Понятное сообщение вместо простыни от strict-загрузки.
    if sorted(ckpt["sd"]) != g["sd_keys"]:
        bad.append("B/веса: набор ключей state_dict разошёлся с эталонным")
    n = int(sum(t.numel() for t in ckpt["sd"].values()))
    if n != g["n_params"]:
        bad.append(f"B/веса: параметров {n} != эталонных {g['n_params']}")

    # C. ПРИЗНАКИ. Вход подобран: есть ран поверх pred, есть ран шире 20px, самый широкий НЕ
    #    совпадает с ближайшим, ранов >1 и <MAXC, v ненулевая — иначе колонки неразличимы.
    f = g["features"]
    order, X = _features(np.array(f["A"]), np.array(f["B"]), np.array(f["C"], float),
                         f["pred"], f["x"], f["v"], f["base"])
    if order.tolist() != f["order"]:
        bad.append(f"C/признаки: порядок кандидатов {order.tolist()} != эталонного {f['order']}")
    elif not np.allclose(X, np.array(f["X"]), atol=1e-12, rtol=0):
        d = np.abs(X - np.array(f["X"])).max(axis=0)
        bad.append(f"C/признаки: X разошлась, макс. отклонение по колонкам {d.round(6).tolist()}")

    # D. ПАТЧ. Точка у края ПО ОБЕИМ осям сразу: проверяется и valid, и обнуление ink за краем.
    #    Полоса задана АНАЛИТИЧЕСКИ (без rng), чтобы эталон не зависел от версии генератора.
    yy, xx = np.mgrid[0:g["patch"]["H"], 0:g["patch"]["Wb"]]
    band = np.abs(xx - (12 + 6 * np.sin(yy / 7.0))) < 2.0
    for key, tag in (("patch", "D/патч"), ("patch_round", "D/патч(округление)")):
        q = g[key]
        ink, val = _patch(band, 0, q["y"], q["pred"])
        if list(ink.shape) != g["patch"]["shape"]:
            bad.append(f"{tag}: форма {list(ink.shape)} != эталонной {g['patch']['shape']}")
            continue
        if int(ink.sum()) != q["ink_sum"] or int(val.sum()) != q["val_sum"]:
            bad.append(f"{tag}: ink {int(ink.sum())}/{q['ink_sum']}, "
                       f"val {int(val.sum())}/{q['val_sum']}")
        if hashlib.md5(np.packbits(ink).tobytes()).hexdigest() != q["ink_md5"]:
            bad.append(f"{tag}: картинка окна разошлась при совпавших суммах")
        if key == "patch" and int((ink & ~val).sum()) != g["patch"]["ink_outside_val"]:
            bad.append(f"{tag}: ink НЕ обнулён за краем — модель обучена отличать «чисто» "
                       f"от «не знаем», это разные входы")

    # E. ВЫХОД СЕТИ. Ловит порядок каналов, формулу колонки, ctx-окно, порядок конкатенации и
    #    masked_fill — ничего из этого не меняет форм весов и не бросает исключений.
    import torch.nn as nn

    net = _build_net(torch, nn)().eval()
    net.load_state_dict(ckpt["sd"])
    ink0, val0 = _patch(band, 0, g["patch"]["y"], g["patch"]["pred"])
    P = torch.from_numpy(np.stack([ink0, val0]).astype(np.float32)[None])
    F = torch.zeros(1, MAXC, NF); F[0, :len(f["order"])] = torch.tensor(f["X"], dtype=torch.float32)
    M = torch.zeros(1, MAXC); M[0, :len(f["order"])] = 1
    with torch.no_grad():
        sc = net(P, F, M)[0].numpy()
    if not np.allclose(sc, np.array(g["scores"], np.float32), atol=1e-5, rtol=0):
        bad.append(f"E/сеть: скоры {np.round(sc, 4).tolist()} != эталонных "
                   f"{[round(v, 4) for v in g['scores']]}")

    # F. ГОДНОСТЬ САМОГО ЭТАЛОНА. `_features` возвращает ИНДЕКСЫ ранов (order), и выбирать надо
    #    idx[argmax(sc)], а не argmax(sc) — перепутать их можно ровно один раз, но тогда трасса
    #    садится на произвольный ран. Эталон различает эти два варианта только если order НЕ
    #    тождественная перестановка; если кто-то переподберёт вход и это свойство потеряется,
    #    тест начнёт молча пропускать целый класс ошибок. ⚠ Сам вызов трассировщика здесь не
    #    проверяется: для этого нужен прогон trace_line по листу, он в selftest не входит.
    if f["order"] == sorted(f["order"]):
        bad.append("F/эталон: order тождественный — вход перестал различать idx[argmax] и "
                   "argmax, эталонный случай надо переподобрать")
    return bad
