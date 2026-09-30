r"""rowdec.py — ПОСТРОЧНЫЙ ДЕКОДЕР В ОТГРУЖАЕМОМ ПУТИ (Задача 9; §6.131 → §6.140).

ЧТО ЭТО. Альтернатива `trace2d.trace_auto`: вместо того чтобы вести каждую линию отдельно и
терять её личность на пересечении (§6.116 — у половины кривых лучшая трасса идёт по ДРУГОЙ
кривой), сеть на каждой строке трека предсказывает положение ВСЕХ кривых сразу. Ошибка такого
решения локальна, и медиана её переживает (§6.133: V0 = 100%).

ЗАМЕРЕНО ДО ВНЕСЕНИЯ (пуловый путь, 5 фолдов по скважинам, каждая кривая вне своего фолда, §6.140):
    правило (порядок по x + мостик)  20.3%
    ★ прототип                       44.0%   медиана по фолдам 45.9%, худший фолд +21.4 пункта
    потолок с мостиком (§6.136)      86.4%
⚠⚠ ЭТО ПУЛОВОЕ ЧИСЛО. §6.123: выигрыш раскладки, намеренный на пулах, УЖЕ ОДИН РАЗ не дожил до
выданного файла (+3.9% → −1.8%). Поэтому модуль включается ФЛАГОМ, а решение о внесении принималось только
после A/B на отгружаемом пути. ★ ВКЛЮЧЁН В ПРОД 05.09 (§6.184, `CVParams.row_decoder = "rowdec_of5_f0_s0.pt"`),
дальше §6.209 / §6.213 / §6.215 — см. `auto/config.py`. (Уточнено при аудите 26.09: здесь стояло «по умолчанию ВЫКЛЮЧЕН».)

ВКЛЮЧЕНИЕ: `cfg.cv.row_decoder = "rowdec_of5_f0_s0.pt"`. Без torch или без файла — молча НЕ
включается, но говорит об этом вслух (та же дисциплина, что у `seq_model`, §6.68: тихая подмена
режима недопустима, иначе «у меня другие цифры» вскроется через полгода).
"""
import numpy as np

_NET = {}
_SAID = set()


def _announce(msg):
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"  {msg}")


def available(ckpt):
    """Можно ли включить: есть torch и файл чекпойнта."""
    if not ckpt:
        return False
    try:
        import torch                                        # noqa: F401
    except Exception:
        return False
    return resolve(ckpt) is not None


_FOLDS = {}


def fold_of_well(well):
    """→ номер фолда, ДЕРЖАВШЕГО эту скважину, или None.

    ⚠⚠ ЗАЧЕМ. Каждый чекпойнт обучен на 160 скважинах из 200 и держал 40. Если гнать A/B одним
    чекпойнтом, у большинства листов их скважина ОКАЖЕТСЯ В ОБУЧЕНИИ, и замер померит запоминание,
    а не обобщение — ровно ловушка §6.70 («набор собран отдельно» ≠ «модель его не видела»;
    там пересечение завысилось вдвое и перевернуло вывод). Поэтому лист декодируется моделью
    ТОГО фолда, который его скважину держал."""
    import json
    from pathlib import Path
    if not _FOLDS:
        wells = set()
        for f in sorted(Path(r"F:/nds/output/taskS/rowdec_crops").glob("man_*of*.json")):
            for t in json.loads(f.read_text(encoding="utf-8"))["tracks"]:
                wells.add(t["well"])
        for i, w in enumerate(sorted(wells)):
            _FOLDS[w.upper()] = i % 5
    return _FOLDS.get(str(well).upper())


_DIR = ""          # каталог варианта обучения; пусто = замороженный набор (см. `resolve`)
_WMAP = None       # карта «лист → скважина манифеста» для честной держанности (§6.153)


def resolve(ckpt):
    """⚠⚠ ЗАМОРОЖЕННЫЙ НАБОР ИЩЕТСЯ ПЕРВЫМ. 23.08 вскрылось, что обучение и замер писали в ОДНИ
    имена файлов: пока шли A/B, чекпойнты фолдов 0 и 1 были перезаписаны другим вариантом лосса,
    и редакции 1-3 сравнивались на СМЕШАННОМ наборе (2 фолда одним кодом, 3 другим). Это ровно
    §6.71 — сравнивать можно только кэши, собранные одним кодом. Лечится не аккуратностью, а
    каталогом, в который обучение не пишет.
    ★ `cv.rowdec_dir` перебивает замороженный каталог — так вариант обучения (например с фоновым
    членом лосса, §6.134) меряется СВОИМ каталогом, а не переименованием и не записью поверх.
    ⚠ Имена внутри каталога те же (`rowdec_of5_f<k>_s0.pt`), поэтому выбор чекпойнта по скважине
    (`auto5`) работает в любом варианте без правок."""
    from pathlib import Path
    # ⚠⚠ ЗАДАННЫЙ КАТАЛОГ — НЕ «ПОДСКАЗКА», А ТРЕБОВАНИЕ. Если в нём чекпойнта нет, откат к
    # замороженному набору означал бы, что режим B прогона молча померил режим A: числа выйдут
    # правдоподобные, а сравнение окажется пустым. Ровно то, чем ветка горела в §6.68 и §6.117 ⇒
    # падаем громко, а не подставляем соседний вес.
    if _DIR:
        c = Path(_DIR) / ckpt
        if c.is_file():
            return c
        raise FileNotFoundError(
            f"rowdec_dir={_DIR!r} задан, но {ckpt} там нет. Откат к замороженному набору запрещён: "
            f"он превратил бы вариант обучения в повтор базового режима")
    for c in (Path(r"F:/nds/output/taskS/rowdec_model/frozen_nobg") / ckpt,
              Path(__file__).resolve().parent / "models" / ckpt,
              Path(r"F:/nds/output/taskS/rowdec_model") / ckpt):
        if c.is_file():
            return c
    return None


def _load(ckpt):
    # ⛔ 26.09 (аудит): кэш сетей был по ГОЛОМУ имени файла, а `resolve` зависит от `_DIR` (rowdec_dir) — второй вариант
    #   каталога в том же процессе молча получил бы веса первого (ровно смешение, от которого защищает `resolve`). Ключ —
    #   полный разрешённый путь. В проде каталог один — выдача та же.
    return _load_path(resolve(ckpt))


def _load_path(path):
    """сеть по ЯВНОМУ пути чекпойнта (кэш по полному пути) — для `_load` и для ансамбля `rowdec_ens`"""
    key = str(path)
    if key in _NET:
        return _NET[key]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        """⚠ Архитектура ПОВТОРЯЕТ `digitizer/rnd/_rowdec_net.py` в точности. Расхождение своей
        копии с обучающей библиотекой не всплывает никогда — обе стороны «работают» (§6.90, где
        float64-обход деревьев расходился с float32 sklearn). Поэтому: любая правка сети — здесь
        и там одновременно, а проверка — сверкой выходов на одном кропе."""

        def __init__(self, emb=8, ch=32):
            super().__init__()
            self.stem = nn.Conv2d(1, ch, 3, padding=1)
            self.blocks = nn.ModuleList([nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=(d, d), dilation=(d, d)),
                nn.GroupNorm(4, ch), nn.ReLU(inplace=True)) for d in (1, 2, 4, 8, 16, 32)])
            self.head_p = nn.Conv2d(ch, 1, 1)
            self.head_e = nn.Conv2d(ch, emb, 1)

        def forward(self, x):
            h = F.relu(self.stem(x))
            for b in self.blocks:
                h = h + b(h)
            return self.head_p(h), self.head_e(h)

    ck = torch.load(key, map_location="cpu")
    net = Net(ck["emb"], ck.get("ch", 32))
    net.load_state_dict(ck["sd"]); net.eval()
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    _NET[key] = (net.to(dev), dev)
    return _NET[key]


def _band(rgb, p, x0, x1):
    """Вход декодера: «насколько темнее бумаги СВОЕЙ строки», минус структура — ровно то, на чём
    сеть училась (§6.133/§6.140). ⚠ Не прод-бинарь `dark_v`: он теряет бледную тушь, и потолок
    падает с 58% до 50% ещё до всякой модели."""
    from . import imaging as im
    V = im.value_channel(rgb)
    H = V.shape[0]
    paper = np.empty(H, np.float32)
    for y0 in range(0, H, 4096):
        y1 = min(H, y0 + 4096)
        paper[y0:y1] = np.percentile(V[y0:y1, ::4], 90, axis=1)
    dark = np.clip(paper[:, None] - V.astype(np.float32), 0, 255).astype(np.uint8)
    dark[im.structure_mask(rgb, p)] = 0
    return dark[:, x0:x1]


_COLS = ("black", "red", "orange", "green", "blue")


def _colmap(rgb, p):
    """Карта цвета туши на весь лист: 0 = нет туши, дальше индекс по `_COLS`+1.

    ⚠⚠ ЗАЧЕМ ОНА ВООБЩЕ НУЖНА ДЕКОДЕРУ. Прод ведёт каждую линию по маске ЕЁ ЦВЕТА
    (`trace2d._color_fg`), то есть цвет у него в построении и подпись почти не путается. Декодер
    же работает по одному серому «насколько темнее бумаги» (`_band`) и цвета не видит вовсе —
    §6.141 намерил цену этого: на отгрузке +61 кривая БЕЗ имени и −41 С именем, знак устойчив в
    трёх прогонах подряд. Ни привязка 1:1, ни передача решения раскладке подпись не чинят ⇒
    траектория обязана принести цвет с собой.
    ★ Маски — ТЕ ЖЕ, что у прода (`imaging.color_channels` + `structure_mask` + `dark_mask`), и
    считаются один раз на скан (мемо `imaging._mres`), а не на трек."""
    from . import imaging as im
    res = im._mres(rgb)
    key = ("rowdec_colmap", id(p))
    if key in res:
        return res[key]
    cm = im.color_channels(rgb, p)
    st = im.structure_mask(rgb, p)
    out = np.zeros(rgb.shape[:2], np.uint8)
    blk = im.dark_mask(rgb, p) & ~st
    for i, c in enumerate(_COLS):
        if c == "black":
            continue
        m = cm[c] & ~st
        out[m] = i + 1
        blk = blk & ~m
    out[blk & (out == 0)] = 1                       # чёрный — последним: он не должен затирать цвет
    res[key] = out
    return out


def ink_color(rgb, p, tr, step=8, halfw=3, min_rows=8):
    """→ (цвет туши ПОД траекторией, доля голосовавших строк) или (None, 0.0).

    ⚠ ГОЛОС — СТРОКА, А НЕ ПИКСЕЛЬ. По сумме пикселей одно пересечение с соседкой другого цвета
    вносит целый сгусток и перебивает всю кривую (замер `_rowdec_color.py`: под экспертной `AK1`
    865 чёрных пикселей против 300 красных, и все 300 принесены одним пересечением). По голосам
    строк пересечение стоит ровно столько, сколько занимает строк."""
    cmap = _colmap(rgb, p)
    H, W = cmap.shape
    vote = np.zeros(len(_COLS) + 1, np.int32)
    for y in sorted(tr)[::step]:
        y = int(y)
        if not (0 <= y < H):
            continue
        x = int(round(tr[y]))
        x0, x1 = max(0, x - halfw), min(W, x + halfw + 1)
        if x1 <= x0:
            continue
        b = np.bincount(cmap[y, x0:x1], minlength=len(_COLS) + 1)
        b[0] = 0
        if b.sum():
            vote[int(b.argmax())] += 1
    n = int(vote.sum())
    if n < min_rows:
        return None, 0.0
    j = int(vote.argmax())
    return _COLS[j - 1], vote[j] / n


def trace_track(rgb, track, k, p, ckpt, y0, y1, emb_min_k=3, wjump=0.15, wemb=1.0, conf_out=None):
    """→ list[{row: x}] длиной k: траектории кривых трека в координатах ЛИСТА."""
    import torch
    net, dev = _load(ckpt)
    # ★ 30.09 (§6.252): АНСАМБЛЬ КАРТ. `cv.rowdec_ens` — доп. каталоги моделей (тот же чекпойнт фолда в каждом, через
    #   запятую): карты вероятностей усредняются; эмбеддинги личности — одной модели, `cv.rowdec_ens_emb` (0 = основная,
    #   1.. — по порядку `rowdec_ens`): склейка эмбеддингов не влезает в память на длинных треках. `cv.rowdec_tta` = 1 —
    #   ещё проход по отражённой по x полосе (карта отражается обратно; эмбеддинги не трогаются). Оба выкл = прежний путь
    #   бит-в-бит (ветка `len(views) == 1` — прежний код без изменений).
    views = [(net, False)]
    for d_ in [s.strip() for s in str(getattr(p, "rowdec_ens", "") or "").replace(";", ",").split(",") if s.strip()]:
        from pathlib import Path as _P
        q_ = _P(d_) / ckpt
        if not q_.is_file():
            raise FileNotFoundError(f"rowdec_ens: {q_} нет — ансамбль без модели фолда был бы другим замером")
        views.append((_load_path(q_)[0], False))
    if int(getattr(p, "rowdec_tta", 0) or 0):
        views += [(nt, True) for nt, _ in list(views)]
    emb_i = int(getattr(p, "rowdec_ens_emb", 0) or 0)
    if not 0 <= emb_i < len(views) or views[emb_i][1]:
        raise ValueError(f"rowdec_ens_emb={emb_i}: моделей {sum(1 for _, f in views if not f)}")
    x0, x1 = int(track.x_left), int(track.x_right) + 1
    band = _band(rgb, p, x0, x1)
    H, Wb = band.shape
    y0 = max(0, int(y0)); y1 = min(H, int(y1))
    if y1 - y0 < 64 or Wb < 16:
        return [{} for _ in range(k)]

    prob = np.zeros((y1 - y0, Wb), np.float32)
    embs = None
    STEP, OV, WIN = 512, 64, 512
    xcuts = [(c, min(Wb, c + WIN)) for c in range(0, max(1, Wb - 1), WIN - 64)]
    for (cx0, cx1) in xcuts:
        for gy in range(y0, y1, STEP - 2 * OV):
            gy2 = min(y1, gy + STEP)
            if gy2 - gy < 32:
                continue
            sub = band[gy:gy2, cx0:cx1].astype(np.float32) / 255.0
            with torch.no_grad():
                if len(views) == 1:
                    pp, ee = net(torch.from_numpy(sub)[None, None].to(dev))
                    pp = torch.sigmoid(pp)[0, 0].float().cpu().numpy()
                    ee = ee[0].float().cpu().numpy()
                else:
                    xt = torch.from_numpy(sub)[None, None].to(dev)
                    acc, ee = None, None
                    for vi, (nt, flip) in enumerate(views):
                        pv, ev = nt(torch.flip(xt, dims=[3]) if flip else xt)
                        pv = torch.sigmoid(pv)[0, 0].float()
                        if flip:
                            pv = torch.flip(pv, dims=[1])
                        acc = pv if acc is None else acc + pv
                        if vi == emb_i:
                            ee = ev[0].float().cpu().numpy()
                    pp = (acc / len(views)).cpu().numpy()
            if embs is None:
                embs = np.zeros((ee.shape[0], y1 - y0, Wb), np.float32)
            v0 = gy + (OV if gy > y0 else 0); v1 = gy2 - (OV if gy2 < y1 else 0)
            wx0 = cx0 + (32 if cx0 > 0 else 0); wx1 = cx1 - (32 if cx1 < Wb else 0)
            if wx1 <= wx0:
                wx0, wx1 = cx0, cx1
            prob[v0 - y0:v1 - y0, wx0:wx1] = pp[v0 - gy:v1 - gy, wx0 - cx0:wx1 - cx0]
            embs[:, v0 - y0:v1 - y0, wx0:wx1] = ee[:, v0 - gy:v1 - gy, wx0 - cx0:wx1 - cx0]
    if embs is None:
        return [{} for _ in range(k)]

    # ★ §6.207: порог пиков — ручка `cv.rowdec_peak_thr` (умолчание 0.6 = прежний прод бит-в-бит).
    #   На пуловом пути 0.3 даёт +49 из 1441 (+3.4 п.п., 4 фолда из 5 вверх): бледная кривая рядом
    #   с тёмной при 0.6 выпадает из кандидатов строки целиком. Критерий приёмки задан в §6.207.
    _v = getattr(p, "rowdec_peak_thr", None)
    pthr = 0.6 if _v is None else float(_v)      # ⚠ не `or 0.6`: 0.0 («все пики») превращался бы в 0.6
    peaks = []
    for i in range(prob.shape[0]):
        row = prob[i]
        thr = pthr * float(row.max())
        idx = np.where((row >= thr) & (row >= np.roll(row, 1)) & (row >= np.roll(row, -1)))[0]
        if len(idx) > 8 * k:
            idx = np.sort(idx[np.argsort(row[idx])[-8 * k:]])
        peaks.append(idx)

    # прототипы личности; ⚠ при k ≤ 2 эмбеддинг ВЫКЛЮЧЕН: кластер вырожден и работает шумом
    # (§6.140: K=1 69.7 → 72.7, K=2 54.3 → 56.5 после выключения).
    use_emb = k >= emb_min_k
    mu = None
    # ★ §6.246: старт прототипов личности (`cv.rowdec_emb_init`: "rand" = прежний случайный, прод бит-в-бит; "x" — пики
    #   опорной строки, где их ровно k, ближайшей к середине окна, по порядку x) и вес эмбеддинга (`cv.rowdec_emb_w`)
    wemb = float(getattr(p, "rowdec_emb_w", wemb) if getattr(p, "rowdec_emb_w", None) is not None else wemb)
    if use_emb:
        pts = [(i, x) for i in range(0, prob.shape[0], 7) for x in peaks[i]]
        if len(pts) >= k * 8:
            M = np.stack([embs[:, i, x] for i, x in pts])
            if (getattr(p, "rowdec_emb_init", "rand") or "rand") == "x":
                cand = [i for i in range(prob.shape[0]) if len(peaks[i]) == k]
                if cand:
                    i0 = min(cand, key=lambda i: abs(i - prob.shape[0] // 2))
                    mu = np.stack([embs[:, i0, x] for x in sorted(peaks[i0])]).astype(M.dtype)
            if mu is None:
                rng = np.random.default_rng(0)
                mu = M[rng.choice(len(M), k, replace=False)]
            for _ in range(12):
                lab = ((M[:, None, :] - mu[None]) ** 2).sum(-1).argmin(1)
                for j in range(k):
                    if (lab == j).any():
                        mu[j] = M[lab == j].mean(0)
        else:
            use_emb = False

    out = []
    taken = [set() for _ in range(prob.shape[0])]
    # ★ §6.237 УДЕРЖАНИЕ (`cv.rowdec_hold` — штраф за строку; 0 = выкл = прежний путь бит-в-бит). Путь обязан брать пик в
    #   каждой строке, где пики есть; на пересечении, занятом прежним путём, своей кривой в строке нет, и путь прыгал на
    #   чужую. С ручкой путь может «удержаться» — не брать пик, сохранив x, — но ТОЛЬКО если доступных пиков в 3·dy + 3 px
    #   от него нет (слабый живой пик своей кривой обязан быть взят), не дольше `rowdec_hold_gmax` строк подряд. Строки
    #   удержания в трассу не пишутся и не занимают пиков.
    hold = float(getattr(p, "rowdec_hold", 0.0) or 0.0)
    gmax = int(getattr(p, "rowdec_hold_gmax", 29) or 29)
    for j in range(k):
        rows, cands, locs = [], [], []
        for i in range(prob.shape[0]):
            idx = np.array([x for x in peaks[i]
                            if not any(abs(x - t) <= 3 for t in taken[i])], int)
            if not len(idx):
                continue
            loc = -np.log(np.clip(prob[i, idx], 1e-6, 1.0))
            if use_emb and mu is not None:
                E = np.stack([embs[:, i, x] for x in idx])
                loc = loc + wemb * np.sqrt(((E - mu[j]) ** 2).sum(-1))
            rows.append(i); cands.append(idx.astype(float)); locs.append(loc)
        if len(rows) < 30:
            out.append({}); continue
        if hold > 0:
            out.append(_viterbi_hold(rows, cands, locs, taken, x0, y0, k, wjump, hold, gmax)); continue
        dp = [locs[0]]; bp = [np.full(len(cands[0]), -1, int)]
        for t in range(1, len(rows)):
            dy = max(1, rows[t] - rows[t - 1])
            jump = np.abs(cands[t][:, None] - cands[t - 1][None, :]) / dy
            tot = dp[t - 1][None, :] + wjump * jump
            arg = tot.argmin(1)
            dp.append(locs[t] + tot[np.arange(len(cands[t])), arg]); bp.append(arg)
        tr, s = {}, int(np.argmin(dp[-1]))
        for t in range(len(rows) - 1, -1, -1):
            x = int(cands[t][s])
            tr[rows[t] + y0] = float(x + x0); taken[rows[t]].add(x)
            s = int(bp[t][s])
            if s < 0:
                break
        out.append(tr)
    # ★ §6.240: уверенность траектории — медиана вероятности карты вдоль неё (для вето в `emit`); ключ — id словаря.
    #   ⛔ 27.09 (разбор): не модульный словарь — UI многопоточный, и параллельный прогон стирал чужие записи (вето молча
    #   выключалось). Словарь даёт вызывающий (`trace_auto`), живёт один вызов.
    if conf_out is not None:
        for tr in out:
            if tr:
                ys = np.fromiter(tr.keys(), np.int64, len(tr)) - y0
                xs = np.fromiter(tr.values(), np.float64, len(tr)).astype(np.int64) - x0
                conf_out[id(tr)] = float(np.median(prob[ys, xs]))
    return out


class _Trs(list):
    """Список (Line, трасса) декодера с уверенностью трасс `.conf` (§6.240) по тому же порядку."""
    conf = None


def _viterbi_hold(rows, cands, locs, taken, x0, y0, k, wjump, wskip, gmax):
    """§6.237: Витерби одного пути с состоянием удержания. Состояния строки = пики + удержания (копии состояний прошлой
    строки с той же x; не больше max(2k, 6) лучших). Удержание: +wskip за строку, счётчик ≤ gmax, и только если рядом с x
    нет доступного пика. Путь кончается в состоянии пика. Копия проверена против `digitizer/rnd/_dec_joint.py`."""
    Hmax = max(2 * k, 6)
    X = cands[0].copy(); D = locs[0].copy(); G = np.zeros(len(X), int); PK = np.ones(len(X), bool)
    hist = [(X, np.full(len(X), -1, int), PK)]
    for t in range(1, len(rows)):
        dy = max(1, rows[t] - rows[t - 1])
        jump = np.abs(cands[t][:, None] - X[None, :]) / dy
        tot = D[None, :] + wjump * jump
        arg = tot.argmin(1)
        Dp = locs[t] + tot[np.arange(len(cands[t])), arg]
        near = np.abs(cands[t][None, :] - X[:, None]).min(1) if len(cands[t]) else np.full(len(X), np.inf)
        hold_ok = np.flatnonzero((G + dy <= gmax) & (near > 3 * dy + 3))
        if len(hold_ok):
            hc = D[hold_ok] + wskip * dy
            keep = hold_ok[np.argsort(hc)[:Hmax]]
            Xh = X[keep]; Dh = D[keep] + wskip * dy; Gh = G[keep] + dy
        else:
            keep = np.zeros(0, int); Xh = np.zeros(0); Dh = np.zeros(0); Gh = np.zeros(0, int)
        X = np.concatenate([cands[t], Xh]); D = np.concatenate([Dp, Dh])
        G = np.concatenate([np.zeros(len(cands[t]), int), Gh])
        PK = np.concatenate([np.ones(len(cands[t]), bool), np.zeros(len(Xh), bool)])
        hist.append((X, np.concatenate([arg, keep]), PK))
    endc = np.flatnonzero(hist[-1][2])
    s = int(endc[np.argmin(D[endc])]) if len(endc) else int(np.argmin(D))
    tr = {}
    for t in range(len(rows) - 1, -1, -1):
        Xt, bpt, pkt = hist[t]
        if pkt[s]:
            x = int(Xt[s]); tr[rows[t] + y0] = float(x + x0); taken[rows[t]].add(x)
        s = int(bpt[s])
        if s < 0:
            break
    return tr


def trace_auto(rgb, sheet, p):
    """Замена `trace2d.trace_auto` целиком: по треку — k линий, k траекторий.
    ⚠ Линиям траектории раздаются по x_center — тем же ключом, каким прод раскладывает линии по
    слотам (§6.105). Никакой новой логики раскладки здесь НЕТ намеренно: замер обязан показать
    вклад ДЕКОДЕРА, а не смеси декодера с новой раскладкой (§6.74 — без абляции вклад неизмерим)."""
    global _DIR
    _DIR = getattr(p, "rowdec_dir", "") or ""
    ck = getattr(p, "row_decoder", "")
    # ★ "auto5" = выбрать чекпойнт по скважине листа (см. fold_of_well). Скважина, которой нет в
    # обучающем списке, декодируется фолдом 0 — она не видена НИ ОДНОЙ моделью, это честно.
    if ck == "auto5":
        w = getattr(getattr(sheet, "meta", None), "well", "")
        # ★★ §6.153: скважина из ИМЕНИ ФАЙЛА промахивается мимо манифеста на 52.8% листов
        # отгрузки, и тогда молча берётся фолд 0 — а скважины нет в обучении лишь у 0.6%.
        # ⇒ 51% листов декодирует модель, ВИДЕВШАЯ скважину. `rowdec_wellmap` — карта
        # «лист → скважина манифеста» для ЗАМЕРА: она делает держанность настоящей.
        # ⚠⚠ ЗАДАНА КАРТА ⇒ ПРОМАХ ПО НЕЙ — ПАДЕНИЕ, А НЕ ФОЛД 0. Тихий откат и есть та самая
        # ошибка: он выглядит как «скважина незнакомая», а на деле она в обучении.
        wm = getattr(p, "rowdec_wellmap", "") or ""
        if wm:
            global _WMAP
            if _WMAP is None:
                # ⚠ `Path` в этом модуле импортируется ВНУТРИ функций, в `trace_auto` его нет —
                # без своего импорта здесь был бы NameError на первом же листе с картой.
                import json as _j
                from pathlib import Path as _P
                _WMAP = _j.loads(_P(wm).read_text(encoding="utf-8"))
            stem = getattr(getattr(sheet, "meta", None), "raw_stem", "") or ""
            if stem not in _WMAP:
                raise KeyError(f"rowdec_wellmap={wm!r}: листа {stem!r} в карте НЕТ — "
                               f"держанность недоказуема, замер остановлен (§6.153)")
            w = _WMAP[stem]
        f = fold_of_well(w)
        if wm and f is None:
            raise KeyError(f"rowdec_wellmap: скважина {w!r} листа {stem!r} не в манифесте "
                           f"обучения — фолд 0 здесь был бы утечкой (§6.153)")
        ck = f"rowdec_of5_f{0 if f is None else f}_s0.pt"
        _announce(f"построчный декодер: скважина {w} → фолд {f if f is not None else '— (не в обучении)'}"
                  f"{'  [карта скважин]' if wm else ''}")
    if not available(ck):
        try:
            import torch                                    # noqa: F401
            why = "нет файла чекпойнта"
        except Exception:
            why = "torch не установлен"
        _announce(f"⚠ row_decoder={ck!r} ЗАПРОШЕН, НО НЕ ВКЛЮЧЁН ({why}) — идёт обычная трассировка")
        return None
    _announce(f"трассировка: ПОСТРОЧНЫЙ ДЕКОДЕР {resolve(ck).name}")

    by_track = {}
    for L in sheet.lines:
        if L.confidence != "AUTO" and not getattr(p, "trace_flagged", False):
            continue
        by_track.setdefault(L.track_index, []).append(L)

    # ★ §6.204: K НЕ НИЖЕ ЧИСЛА СЛОТОВ ТРЕКА (`cv.rowdec_k_slots`; с 11.09 умолчание True, ОТКАТ = False —
    # прежний путь бит-в-бит; уточнено при аудите 26.09). Декодер выдаёт ровно K траекторий, и при K_U1 < K_слотов недостающие кривые
    # не выдаются физически: на 124 треках честного поля так теряются 106 проводимых кривых, а
    # недосчёт на одну стоит 9.5 п.п. против 1.6 за пересчёт на три (`_rowdec_eval.py --k-offset`).
    # Треки, где U1 не нашёл ни одной линии, при включённой ручке тоже ведутся — по всей рамке.
    kslots = (getattr(sheet, "k_slots", None) or {}) if getattr(p, "rowdec_k_slots", False) else {}
    for ti, n in kslots.items():
        if n > 0 and ti not in by_track and 0 <= ti < len(sheet.frame.tracks):
            by_track[ti] = []

    out = []
    conf = {}                       # §6.240: id(трасса) → медиана p, только для этого вызова
    for ti, lines in by_track.items():
        track = sheet.frame.tracks[ti]
        if lines:
            y0 = min(L.y0 for L in lines); y1 = max(L.y1 for L in lines)
        else:
            y0, y1 = int(sheet.frame.top_y), int(sheet.frame.bottom_y)
        K = max(len(lines), int(kslots.get(ti, 0)))
        if K <= 0:
            continue
        # ★ §6.239: K + `rowdec_k_plus` путей (0 = прод бит-в-бит). §6.237: из невзятых удержанием 71 из 196 ведутся по
        #   ДРУГОЙ нарисованной линии (> 50 px) — лишняя линия трека, которой нет в эталоне, забирает путь.
        K += int(getattr(p, "rowdec_k_plus", 0) or 0)
        trs = trace_track(rgb, track, K, p, ck, y0, y1, conf_out=conf)
        # ── ПРИВЯЗКА ТРАЕКТОРИИ К ЛИНИИ: 1:1 ПО СТОИМОСТИ, А НЕ СОРТИРОВКОЙ ────────────────
        # ⚠⚠ ЗАЧЕМ. Подпись решает не декодер, а то, КАКОЙ ЛИНИИ досталась траектория: линия несёт
        # цвет, класс и полосу, по которым `emit._map_lines_to_slots` раскладывает по слотам.
        # Первая редакция сортировала обе стороны по x и сшивала по порядку — и §6.141 намерил
        # цену: на отгрузке −7 кривых С ИМЕНЕМ при +71 БЕЗ ИМЕНИ. То есть геометрия улучшалась, а
        # подпись портилась ровно здесь. Теперь: стоимость = |медиана трассы − x_center линии| плюс
        # штраф за выход за полосу линии, назначение взаимно однозначное.
        trs = [t for t in trs if len(t) >= 30]
        if not trs:
            continue
        med = [float(np.median(list(t.values()))) for t in trs]
        # ── ★ ЦВЕТ ПОД ТРАЕКТОРИЕЙ (`cv.rowdec_color`, 0 = ВЫКЛ = поведение §6.141) ──────────
        # ⚠⚠ ЗАЧЕМ И ПОЧЕМУ НЕ ВЕТО. Цвет — единственный признак линии, которого траектория
        # декодера не несёт, а раскладка по нему узнаёт кривую. Но §6.13: цвет в `mnemonics.json`
        # НЕ доменный факт (SP нарисована чёрной на 17 листах из 20), да и сама маска ошибается на
        # выцветших бланках ⇒ штраф, а не запрет: где цвет померился и разошёлся, пара дорожает
        # ровно на `rowdec_color` пикселей эквивалента, но остаётся возможной.
        # ⚠ Мерится ТОЛЬКО при включённой ручке: `_colmap` — лишний проход по скану.
        wcol = float(getattr(p, "rowdec_color", 0.0) or 0.0)
        tcol = [ink_color(rgb, p, t)[0] for t in trs] if wcol else [None] * len(trs)
        cost = np.zeros((len(trs), len(lines)), np.float64)
        for i, mx in enumerate(med):
            for j, L in enumerate(lines):
                c = abs(mx - L.x_center)
                if mx < L.x_lo - 8 or mx > L.x_hi + 8:      # вне полосы линии — дорого, но не запрет
                    c += 500.0
                if tcol[i] is not None and tcol[i] != L.color:
                    c += wcol
                cost[i, j] = c
        # ── РЕДАКЦИЯ 3: ПУЛ КАНДИДАТОВ ВМЕСТО ГОТОВОЙ ПАРЫ ────────────────────────────────
        # ⚠⚠ ЧТО ПОКАЗАЛИ ДВЕ ПРЕДЫДУЩИЕ РЕДАКЦИИ (§6.141): и сортировка по x_center, и назначение
        # 1:1 по стоимости дают ОДИН И ТОТ ЖЕ безымянный выигрыш (+71 и +69) и ОДИНАКОВО теряют на
        # подписи (−7 и −31, оба внутри шума SD 26). Значит подпись портит не привязка, а то, что
        # решение принимается ЗДЕСЬ, вслепую, вместо обученной раскладки: прод-трасса держится
        # полосы своей линии по построению, а траектория декодера полосы не знает.
        # ⇒ отдаём раскладке ВСЕ правдоподобные пары «линия × траектория» (траектория в полосе
        # линии ±8px), и пусть `emit._map_lines_to_slots` со `slot_model` (§6.130) выбирает сам.
        # Раскладка делает 1:1 внутри трека, поэтому дубли контракт не ломают.
        # ⚠ УМОЛЧАНИЕ `False` — ПАРА 1:1. Здесь стояло `True`, то есть код по умолчанию выбирал
        # вариант, ОТВЕРГНУТЫЙ замером (§6.141: пул −120 против пары +61). Верный вариант получался
        # только там, где стенд задавал ручку явно.
        if getattr(p, "rowdec_pool", False):
            added = 0
            for i, t in enumerate(trs):
                for j, L in enumerate(lines):
                    if L.x_lo - 8 <= med[i] <= L.x_hi + 8:
                        out.append((L, t)); added += 1
            if not added and lines:   # ни одна не попала в полосу — лучшая пара, чтобы лист не опустел
                i = int(np.argmin(cost.min(axis=1))); j = int(np.argmin(cost[i]))
                out.append((lines[j], trs[i]))
        else:
            used_i, used_j = set(), set()
            for i, j in sorted(((i, j) for i in range(len(trs)) for j in range(len(lines))),
                               key=lambda z: cost[z[0], z[1]]):
                if i in used_i or j in used_j:
                    continue
                used_i.add(i); used_j.add(j)
                out.append((lines[j], trs[i]))
            # ★ §6.204: траекториям сверх числа линий U1 — СИНТЕТИЧЕСКАЯ линия, иначе раскладке
            #   нечего сажать в слот (она узнаёт кривую по цвету, классу и x_center линии).
            #   Только при включённой ручке: без неё len(trs) ≤ len(lines) и ветка мертва.
            if kslots:
                for i, t in enumerate(trs):
                    if i not in used_i:
                        out.append((_synth_line(lines, ti, t, med[i], rgb, p), t))
    res = _Trs(out)
    res.conf = [(conf.get(id(t)),) for _, t in out]
    return res


SYNTH = "kslots-synth"      # маркер синтетической линии в `flag_reason` (§6.206): читает `emit`


def _synth_line(lines, ti, tr, mx, rgb, p):
    """Линия для траектории, у которой нет линии U1 (§6.204). Клон БЛИЖАЙШЕЙ по x линии трека —
    цвет и класс наследуются (по ним раскладка узнаёт кривую), полоса ставится вокруг медианы
    трассы. Если линий на треке нет вовсе — цвет по туши под траекторией, класс неизвестен."""
    from dataclasses import replace
    ys = sorted(tr)
    y0, y1 = int(ys[0]), int(ys[-1])
    if lines:
        L = min(lines, key=lambda q: abs(q.x_center - mx))
        half = max(8.0, L.x_band / 2)
        return replace(L, x_center=float(mx), x_lo=float(mx - half), x_hi=float(mx + half),
                       y0=y0, y1=y1, confidence="AUTO", flag_reason=SYNTH,
                       x_hard_lo=None, x_hard_hi=None)
    from .understand import Line
    col = ink_color(rgb, p, tr)[0] or "black"
    return Line(track_index=ti, color=col, x_center=float(mx), x_lo=float(mx - 20),
                x_hi=float(mx + 20), y0=y0, y1=y1, thickness=2.0, rough_n=None, behavior="?",
                n_strokes=1, density=0.0, confidence="AUTO", flag_reason=SYNTH)
