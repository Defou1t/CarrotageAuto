r"""_rowdec_eval.py — БОЕВОЙ ПУТЬ ОЦЕНКИ ПОСТРОЧНОГО ДЕКОДЕРА + ПАРИТЕТ С ПОТОЛКОМ (гейт G0).

⚠⚠ ЗАЧЕМ ПАРИТЕТ ДО ОБУЧЕНИЯ. Стенд оценки — это код, и он ошибается так же, как всякий другой:
§6.133 пришлось пересчитывать целиком, потому что `cv2.imread` молча терял 10.6% листов, а числа
выглядели полными. Поэтому здесь сначала гоняется ОРАКУЛЬНЫЙ режим: головы модели подменяются
эталоном, а весь остальной путь (пики → матчинг → мостик → 1:1 → med/cov) идёт боевой. Если он не
воспроизводит уже опубликованные числа §6.136, обучать нечего — сломан стенд.

ПАРИТЕТНЫЕ ЦЕЛИ (подвыборка `_rowdec_data`, 528 треков / 1455 кривых, посчитаны из дампов
`rowceil_*of6.pkl` тем же фильтром):
    без мостика (V3ц)            60.0%
    с мостиком   (V3м)           86.4%
    кривые с разрывами ≤50 строк 94.4%
Допуск ±4 п.п. — путь не побитово тот же (здесь матчинг траектория→эталон 1:1, там кривая→свой
эталон), поэтому требовать совпадения до десятых нельзя, а расхождение в разы обязано быть поймано.

РЕЖИМЫ ИДЕНТИЧНОСТИ (`--ident`):
  oracle  — кандидат достаётся той кривой, чей эталон он накрывает (потолок, для G0);
  order   — кандидаты сортируются по x и раздаются по порядку (это контроль A/B `_rowdec_base`);
  model   — эмбеддинги обученной сети (когда появится; сюда же ляжет абляция G3).

  <ComfyUI>\python_embeded\python.exe _rowdec_eval.py --ident oracle --shard 0/4
  <ComfyUI>\python_embeded\python.exe _rowdec_eval.py --sum --tag evaloracle
"""
import sys, argparse, json, pickle
from itertools import permutations
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
from collections import defaultdict
import numpy as np
from auto import imaging as im

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"F:/nds/output/taskS/rowdec")
ap.add_argument("--out", default=r"F:/nds/output/taskS")
ap.add_argument("--tag", default="")
ap.add_argument("--ident", default="oracle", choices=["oracle", "order", "model"])
ap.add_argument("--shard", default="0/4")
ap.add_argument("--thr", type=int, default=90)
ap.add_argument("--tol", type=float, default=3.0, help="допуск накрытия рана эталоном (§6.26)")
ap.add_argument("--max-rows", type=int, default=20000)
ap.add_argument("--no-bridge", action="store_true", help="не мостить разрывы (для сверки с V3ц)")
ap.add_argument("--ckpt", default="", help="чекпойнт для --ident model")
ap.add_argument("--ablate-emb", action="store_true",
                help="ГЕЙТ G3: выключить эмбеддинг, матчить только по положению. Если потеря "
                     "меньше 8 пунктов — выигрыш дал мостик и передний план, а не личность")
ap.add_argument("--win", type=int, default=512, help="ширина окна инференса, колонок")
ap.add_argument("--wjump", type=float, default=0.15, help="штраф прыжка в Витерби (на строку)")
ap.add_argument("--wemb", type=float, default=1.0, help="вес расстояния эмбеддинга")
# ★ §6.140: эмбеддинг помогает при K≥3 (+12.7 и +7.1 пункта) и ВРЕДИТ при K≤2 (−3.0 и −2.2):
# при одной-двух кривых прототип личности вырожден (k-means кладёт всё в один кластер), и его
# расстояние становится шумом в стоимости. Порог — правило ДЕКОДИРОВАНИЯ, переобучения не требует.
ap.add_argument("--emb-min-k", type=int, default=3, help="включать эмбеддинг только при K ≥ этого")
ap.add_argument("--two-pass", action="store_true",
                help="второй проход: СГЛАЖЕННАЯ траектория первого прохода как приор + совместное "
                     "назначение всех K на строке. ⚠ приор ФИКСИРОВАН (не бегущее состояние), "
                     "поэтому ошибка остаётся локальной — иначе вернётся снос всей кривой (§6.116)")
ap.add_argument("--wpred", type=float, default=0.08, help="вес приора сглаженной траектории")
ap.add_argument("--fold", type=int, default=-1, help="оценивать только держанный фолд (-1 = все треки)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--sum", action="store_true")
a = ap.parse_args()
OUT = Path(a.out)
TAG = a.tag or f"eval{a.ident}"
HON = lambda m, c: m is not None and m <= 3.0 and c >= 0.9
PARITY = {"без мостика (V3ц)": 60.0, "с мостиком (V3м)": 86.4, "разрывы ≤50 строк": 94.4}


def match_1to1(err):
    K = err.shape[0]
    if K <= 6:
        best, bp = None, None
        for p in permutations(range(K)):
            s = sum(err[i, p[i]] for i in range(K))
            if best is None or s < best:
                best, bp = s, p
        return list(bp)
    order = np.dstack(np.unravel_index(np.argsort(err, axis=None), err.shape))[0]
    ur, uc, out = set(), set(), [None] * K
    for r, c in order:
        if r in ur or c in uc:
            continue
        ur.add(int(r)); uc.add(int(c)); out[int(r)] = int(c)
    return [c if c is not None else 0 for c in out]


_NET = {}


def net_maps(band, y0, y1, x0, x1):
    """Прогнать сеть по куску полосы → (prob[R×W], emb[E×R×W]). Куски режутся по строкам с
    перекрытием 32: рецептивное поле сети ±63 строки, и без перекрытия шов испортил бы решение
    ровно там, где оно нужнее всего."""
    import torch
    net = _NET["net"]
    sub = band[y0:y1, x0:x1].astype(np.float32) / 255.0
    with torch.no_grad():
        t = torch.from_numpy(sub)[None, None].to(_NET["dev"])
        p, e = net(t)
        return torch.sigmoid(p)[0, 0].float().cpu().numpy(), e[0].float().cpu().numpy()


def decode_model(band, ys, xs):
    """Построчные положения из голов сети. Личность — эмбеддингом: прототипы кривых набираются
    по треку, дальше на каждой строке пики раздаются прототипам 1:1 по стоимости эмбеддинга.
    ⚠ Порядок по x НЕ используется как признак — он и есть контроль 18.2% (§6.134)."""
    import torch
    H, Wb = band.shape
    K = xs.shape[1]
    ours = np.full((len(ys), K), np.nan, np.float32)
    STEP, OV = 512, 64
    # окно по колонкам: полоса шире окна режется тоже, с перекрытием
    xcuts = [(c, min(Wb, c + a.win)) for c in range(0, max(1, Wb - 1), a.win - 64)]
    prob = np.zeros((len(ys), Wb), np.float32)
    embs = None
    # ⚠⚠ ПРОГОН ИДЁТ ПО СТРОКАМ ИЗОБРАЖЕНИЯ, А НЕ ПО ИНДЕКСАМ ТАРГЕТА. Первая редакция резала
    # куски по индексам `ys` и требовала, чтобы они шли подряд, — а эталон рвётся (у первого же
    # держанного трека `ys` НЕ сплошные), поэтому пропускались почти все куски, карта оставалась
    # нулевой и честных выходило 0 при том, что сеть локализует с медианой 0.85px. Дефект был в
    # обвязке, а не в модели, и виден только сквозным разбором одного трека.
    ymin, ymax = int(ys[0]), int(ys[-1]) + 1
    pos = {int(y): i for i, y in enumerate(ys)}
    for (cx0, cx1) in xcuts:
        for gy0 in range(ymin, ymax, STEP - 2 * OV):
            gy1 = min(H, gy0 + STEP)
            if gy1 - gy0 < 32:
                continue
            p, e = net_maps(band, gy0, gy1, cx0, cx1)
            if embs is None:
                embs = np.zeros((e.shape[0], len(ys), Wb), np.float32)
            v0 = gy0 + (OV if gy0 > ymin else 0)
            v1 = gy1 - (OV if gy1 < ymax else 0)
            # ⚠ ПИШЕМ ТОЛЬКО СЕРЕДИНУ ОКНА ПО КОЛОНКАМ. Последнее окно бывает шириной 2-128
            # колонок (у 33% треков ≤128), и, записывая его целиком, оно ЗАТИРАЛО уже посчитанные
            # значения соседнего окна своими краевыми — а краю свёртки не хватает контекста.
            wx0 = cx0 + (32 if cx0 > 0 else 0)
            wx1 = cx1 - (32 if cx1 < Wb else 0)
            if wx1 <= wx0:
                wx0, wx1 = cx0, cx1
            for gy in range(v0, v1):
                i = pos.get(gy)
                if i is None:
                    continue
                prob[i, wx0:wx1] = p[gy - gy0, wx0 - cx0:wx1 - cx0]
                embs[:, i, wx0:wx1] = e[:, gy - gy0, wx0 - cx0:wx1 - cx0]
    if embs is None:
        return ours

    # ── ПИКИ СТРОКИ: порог ОТНОСИТЕЛЬНЫЙ, а не абсолютный ──────────────────────────────────
    # ⚠ Первая редакция брала `prob > 0.3` и получала ~109 «пиков» на строку: сеть обучена с
    # `pos_weight=50`, поэтому её карта смещена вверх (медиана 0.387, 88% пикселей выше 0.3).
    # Локализация при этом ТОЧНАЯ — медиана до ближайшего пика 0.85px, — то есть дефект был в
    # отборе, а не в модели, и обнаружился только прямым разбором одного трека.
    # Берём локальные максимумы выше половины максимума СВОЕЙ строки и оставляем 3K лучших:
    # декодеру нужны кандидаты, а не вся карта.
    peaks = []
    for i in range(len(ys)):
        row = prob[i]
        # ⚠ БЫЛО `max(0.5, 0.5*row.max())` — и это ТОЖДЕСТВЕННО 0.5: prob идёт из sigmoid, значит
        # row.max() < 1, значит 0.5*row.max() < 0.5 ВСЕГДА. «Относительный порог» не срабатывал ни
        # разу, а выглядел рабочим, потому что 0.5 сам по себе даёт правдоподобные 12-17 пиков.
        thr = 0.6 * float(row.max())
        idx = np.where((row >= thr) & (row >= np.roll(row, 1)) & (row >= np.roll(row, -1)))[0]
        # ⚠ Отсечка «оставить 3K лучших ПО ВЕРОЯТНОСТИ» опиралась на то самое ранжирование, которое
        # замер признал негодным (верный пик top-1 лишь у 23-32% строк). Теперь предел мягче и
        # печатается recall кандидатов — иначе потеря верного пика невидима.
        if len(idx) > 8 * K:
            idx = np.sort(idx[np.argsort(row[idx])[-8 * K:]])
        peaks.append(idx)
    # прототипы личности: k-means по эмбеддингам пиков (K кластеров на трек)
    pts = [(i, x) for i in range(0, len(ys), 7) for x in peaks[i]]
    if len(pts) < K * 8:
        return ours
    M = np.stack([embs[:, i, x] for i, x in pts])
    rng = np.random.default_rng(0)
    mu = M[rng.choice(len(M), K, replace=False)]
    for _ in range(12):
        d = ((M[:, None, :] - mu[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(K):
            if (lab == k).any():
                mu[k] = M[lab == k].mean(0)
    # ── ВЫБОР ПИКА: ВИТЕРБИ ПО СТРОКАМ, А НЕ ПОСТРОЧНЫЙ АРГМАКС ────────────────────────────
    # ⚠⚠ ПОЧЕМУ. Прямой разбор четырёх держанных треков: `argmax` головы вероятности даёт медиану
    # 5.9-297px и лишь 18-34% строк в пределах 3px, а БЛИЖАЙШИЙ ИЗ ПИКОВ — медиану 2.0-2.8px и
    # 58-77%. То есть сеть КАНДИДАТОВ НАХОДИТ, но не ранжирует: одной строки для этого мало.
    # Ранжирование даёт непрерывность — та самая, которой у построчного аргмакса нет по построению.
    # Стоимость перехода: −log p (уверенность) + вес·|Δx| (гладкость) + вес·расстояние эмбеддинга
    # до прототипа кривой (личность, §6.135). Пути ищутся по одному, занятые пики исключаются —
    # это и есть ограничение 1:1 (§6.115), только вдоль всей кривой, а не на одной строке.
    W_JUMP, W_EMB = a.wjump, a.wemb
    use_emb = (not a.ablate_emb) and K >= a.emb_min_k
    taken = [set() for _ in range(len(ys))]
    for k in range(K):
        rows, cands, locs = [], [], []
        for i in range(len(ys)):
            # ⚠ ЗАНИМАЕТСЯ ОКРЕСТНОСТЬ, А НЕ СТОЛБЕЦ. Раньше `taken` держал точный x, и две
            # траектории спокойно ехали по одной кривой в 1-2px друг от друга — каждая гладкая,
            # каждая с полным покрытием, а match_1to1 потом раздавал их разным эталонам.
            idx = np.array([x for x in peaks[i]
                            if not any(abs(x - t_) <= 3 for t_ in taken[i])], int)
            if not len(idx):
                continue
            loc = -np.log(np.clip(prob[i, idx], 1e-6, 1.0))
            if use_emb:
                E = np.stack([embs[:, i, x] for x in idx])
                loc = loc + W_EMB * np.sqrt(((E - mu[k]) ** 2).sum(-1))
            rows.append(i); cands.append(idx.astype(float)); locs.append(loc)
        if len(rows) < 30:
            continue
        # ── ПРЯМОЙ ПРОХОД ──────────────────────────────────────────────────────────────────
        dp = [locs[0]]
        bp = [np.full(len(cands[0]), -1, int)]
        for t in range(1, len(rows)):
            # ⚠ штраф прыжка НОРМИРУЕТСЯ на расстояние по строкам: между соседними строками
            # кривая сдвигается на пиксели, а через разрыв в 200 строк — законно на десятки.
            # ⚠ dy — по НОМЕРУ СТРОКИ ИЗОБРАЖЕНИЯ: индексы массива ys не учитывают разрывы эталона
            # (на треках встречаются разрывы до 990 строк), и штраф прыжка через разрыв был бы
            # посчитан как для соседних строк.
            dy = max(1, int(ys[rows[t]]) - int(ys[rows[t - 1]]))
            jump = np.abs(cands[t][:, None] - cands[t - 1][None, :]) / dy
            tot = dp[t - 1][None, :] + W_JUMP * jump
            arg = tot.argmin(1)
            dp.append(locs[t] + tot[np.arange(len(cands[t])), arg])
            bp.append(arg)
        # ── ОБРАТНЫЙ ПРОХОД ────────────────────────────────────────────────────────────────
        j = int(np.argmin(dp[-1]))
        for t in range(len(rows) - 1, -1, -1):
            i = rows[t]; x = int(cands[t][j])
            ours[i, k] = float(x); taken[i].add(x)
            j = int(bp[t][j])
            if j < 0:
                break

    if not a.two_pass:
        return ours

    # ── ВТОРОЙ ПРОХОД: СОВМЕСТНОЕ НАЗНАЧЕНИЕ ПО СТРОКЕ ────────────────────────────────────
    # Первый проход ведёт кривые ПО ОДНОЙ и жадно: кто раньше занял пик, тот его и держит, а
    # порядок задаёт номер кластера k-means, то есть случайный сид. Здесь все K решаются вместе.
    # ⚠ Приор — СГЛАЖЕННАЯ траектория первого прохода, ФИКСИРОВАННАЯ функция строки, а не бегущее
    # состояние: именно поэтому ошибка второго прохода остаётся локальной (§6.131), а не уводит
    # кривую целиком, как это делает трассировщик со состоянием (§6.116).
    W = 64
    pred = np.full_like(ours, np.nan)
    for k in range(K):
        v = ours[:, k]
        ok = np.isfinite(v)
        if ok.sum() < 30:
            continue
        idx = np.where(ok)[0]
        iv = np.interp(np.arange(len(ys)), idx, v[idx])
        ker = np.ones(min(W, len(iv))) / min(W, len(iv))
        pred[:, k] = np.convolve(iv, ker, mode="same")

    out2 = np.full_like(ours, np.nan)
    for i in range(len(ys)):
        idx = peaks[i]
        if not len(idx):
            continue
        cost = np.zeros((len(idx), K), np.float32)
        base = -np.log(np.clip(prob[i, idx], 1e-6, 1.0))
        E = np.stack([embs[:, i, x] for x in idx]) if use_emb else None
        for k in range(K):
            c = base.copy()
            if use_emb:
                c = c + W_EMB * np.sqrt(((E - mu[k]) ** 2).sum(-1))
            if np.isfinite(pred[i, k]):
                c = c + a.wpred * np.abs(idx.astype(float) - pred[i, k])
            cost[:, k] = c
        used_p, used_k = set(), set()
        for pi, ki in sorted(((p_, k_) for p_ in range(len(idx)) for k_ in range(K)),
                             key=lambda z: cost[z[0], z[1]]):
            if pi in used_p or ki in used_k:
                continue
            out2[i, ki] = float(idx[pi]); used_p.add(pi); used_k.add(ki)
    return out2


def decode(band, ys, xs, ident):
    if ident == "model":
        return decode_model(band, ys, xs)
    """→ ours[len(ys) × K]: положение каждой траектории на каждой строке (NaN = нет ответа)."""
    H, Wb = band.shape
    K = xs.shape[1]
    ours = np.full((len(ys), K), np.nan, np.float32)
    for i, y in enumerate(ys):
        if not (0 <= y < H):
            continue
        runs = im.row_runs(band[y] >= a.thr)
        if not runs:
            continue
        if ident == "oracle":
            # ⚠ 1:1 НА СТРОКЕ: ран достаётся ОДНОЙ кривой (§6.115). Порядок разбора — по
            # близости эталона к центру рана, чтобы «свой» слот забирал ран раньше чужого.
            taken = set()
            order = sorted(range(K), key=lambda k: min(
                (abs(float(xs[i, k]) - c) for _, _, c in runs), default=1e9))
            for k in order:
                x = float(xs[i, k])
                for ri, (x0, x1, c) in enumerate(runs):
                    if ri in taken:
                        continue
                    if x0 - a.tol <= x <= x1 + a.tol:
                        ours[i, k] = c; taken.add(ri); break
        else:                                   # order — контроль без личности
            rr = runs
            if len(rr) > K:
                rr = sorted(sorted(rr, key=lambda r: r[1] - r[0], reverse=True)[:K],
                            key=lambda r: r[2])
            for j in range(min(K, len(rr))):
                ours[i, j] = rr[j][2]
    return ours


def run_track(p):
    d = np.load(p)
    band, ys, xs = d["band"], d["ys"], d["xs"]
    K = xs.shape[1]
    if a.max_rows and len(ys) > a.max_rows:
        idx = np.linspace(0, len(ys) - 1, a.max_rows).astype(int)
        ys, xs = ys[idx], xs[idx]
    ours = decode(band, ys, xs, a.ident)

    err = np.full((K, K), 1e9, np.float32)
    for j in range(K):
        ok = ~np.isnan(ours[:, j])
        if ok.sum() < 30:
            continue
        for k in range(K):
            err[j, k] = float(np.median(np.abs(ours[ok, j] - xs[ok, k])))
    perm = match_1to1(err)

    out = []
    for j in range(K):
        k = perm[j]
        ok = ~np.isnan(ours[:, j]); idx = np.where(ok)[0]
        cov = float(ok.sum()) / max(1, len(ys))
        med = float(np.median(np.abs(ours[ok, j] - xs[ok, k]))) if ok.sum() >= 30 else None
        if a.no_bridge or len(idx) < 2:
            out.append((K, med, cov, 0)); continue
        allr = np.arange(len(ys))
        iv = np.interp(allr, idx, ours[idx, j])
        inside = (allr >= idx[0]) & (allr <= idx[-1])   # за краями interp держит константу
        e = np.abs(iv[inside] - xs[inside, k])
        out.append((K, float(np.median(e)) if len(e) >= 30 else None,
                    float(inside.sum()) / max(1, len(ys)), int(np.max(np.diff(idx)))))
    return out


def collect(i, n):
    man = []
    for f in sorted(Path(a.data).glob("manifest_*of*.json")):
        man += json.loads(f.read_text(encoding="utf-8"))
    # ⚠ ДЕРЖАННЫЙ ФОЛД — ПО СКВАЖИНАМ И ТЕМ ЖЕ ПРАВИЛОМ, ЧТО В ОБУЧЕНИИ (§6.87, §6.70): иначе
    # модель померится на том, чему училась, и узнается это только прямой сверкой.
    if a.fold >= 0:
        uw = sorted({m["well"] for m in man})
        hold = {w for k, w in enumerate(uw) if k % a.folds == a.fold}
        man = [m for m in man if m["well"] in hold]
        print(f"★ ДЕРЖАННЫЙ ФОЛД {a.fold}/{a.folds}: скважин {len(hold)} из {len(uw)}, "
              f"треков {len(man)}")
    files = sorted({m["file"] for m in man})
    mine = files[len(files) * i // n:len(files) * (i + 1) // n]
    print(f"★ ШАРД {i}/{n}: треков {len(mine)} из {len(files)}, идентичность={a.ident}, "
          f"мостик={'выкл' if a.no_bridge else 'вкл'}")
    rows, bad = [], defaultdict(int)
    for c, fn in enumerate(mine, 1):
        try:
            rows += run_track(Path(a.data) / fn)
        except Exception as e:
            bad[f"{type(e).__name__}: {e}"[:60]] += 1
        if c % 20 == 0 or c == len(mine):
            print(f"  {c}/{len(mine)}  кривых {len(rows)}")
    p = OUT / f"{TAG}_{i}of{n}.pkl"
    pickle.dump(dict(rows=rows, tracks=len(mine), bad=dict(bad), ident=a.ident,
                     bridge=not a.no_bridge), open(p, "wb"))
    print(f"★ готово: кривых {len(rows)} → {p}")
    for s, c in bad.items():
        print(f"    ⚠ {s}: {c}")


def summarise():
    fs = sorted(OUT.glob(f"{TAG}_*of*.pkl"))
    if not fs:
        sys.exit("нет дампов")
    den = fs[0].stem.split("of")[1]
    fs = [f for f in fs if f.stem.endswith("of" + den)]
    rows, tracks, ident, bridge = [], 0, "?", None
    for f in fs:
        d = pickle.load(open(f, "rb"))
        rows += d["rows"]; tracks += d["tracks"]; ident = d["ident"]; bridge = d["bridge"]
    N = max(1, len(rows))
    hon = sum(1 for K, m, c, g in rows if HON(m, c))
    print(f"шардов {len(fs)} из {den}" + ("  ⚠⚠ НЕПОЛНЫЙ" if len(fs) < int(den) else ""))
    print(f"треков {tracks}, кривых {len(rows)}, идентичность={ident}, "
          f"мостик={'ВКЛ' if bridge else 'выкл'}")
    print(f"\n★ ЧЕСТНЫХ: {hon} из {len(rows)} = {100*hon/N:.1f}%")
    g = [r for r in rows if r[3] and r[3] <= 50]
    hg = sum(1 for K, m, c, gg in g if HON(m, c))
    if g:
        print(f"   на кривых с разрывами ≤50 строк: {hg} из {len(g)} = {100*hg/len(g):.1f}%")

    # ── ГЕЙТ G0: сверка с опубликованным потолком ─────────────────────────────────────────
    if ident == "oracle":
        tgt = PARITY["без мостика (V3ц)"] if not bridge else PARITY["с мостиком (V3м)"]
        got = 100 * hon / N
        ok = abs(got - tgt) <= 4.0
        print(f"\n{'★ ПАРИТЕТ G0 ПРОЙДЕН' if ok else '⛔ ПАРИТЕТ G0 НЕ ПРОЙДЕН'}: "
              f"{got:.1f}% против цели {tgt:.1f}% (допуск ±4 п.п.)")
        if bridge and g:
            gt = PARITY["разрывы ≤50 строк"]; gg = 100 * hg / len(g)
            print(f"{'★' if abs(gg-gt) <= 4 else '⛔'} разрыв ≤50: {gg:.1f}% против {gt:.1f}%")
        if not ok:
            print("⇒ стенд НЕ воспроизводит §6.136 — обучать нечего, чинить стенд (§6.133).")

    print(f"\n{'K':<6}{'кривых':>9}{'честных':>10}{'доля':>8}")
    agg = defaultdict(lambda: [0, 0])
    for K, m, c, gg in rows:
        b = agg[min(K, 5)]; b[0] += 1; b[1] += int(HON(m, c))
    for k in sorted(agg):
        v = agg[k]
        print(f"{('%d' % k) if k < 5 else '5+':<6}{v[0]:>9}{v[1]:>10}{100*v[1]/v[0]:>7.1f}%")
    print(f"\nконтроль без личности: A 18.2% (без мостика), B 19.1% (с мостиком) — "
          f"гейт прототипа: ≥ +10 пунктов над B.")


def load_net():
    """⚠ Сеть строится ТЕМ ЖЕ классом, что учил `_rowdec_net.py` — импортом, а не копией:
    расхождение своей копии с обучающей библиотекой ловится только побитовой сверкой (§6.90),
    а дешевле его просто не создавать."""
    import torch, importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rowdec_net_mod", str(Path(__file__).with_name("_rowdec_net.py")))
    sys.argv = [sys.argv[0], "--overfit", "0", "--epochs", "0"]      # модуль парсит argparse
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    ck = torch.load(a.ckpt, map_location="cpu")
    net = mod.Net(ck["emb"], ck.get("ch", 32))
    net.load_state_dict(ck["sd"]); net.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _NET.update(net=net.to(dev), dev=dev)
    print(f"★ чекпойнт {Path(a.ckpt).name}: emb={ck['emb']}, фолд {ck['fold']}/{ck['folds']}, "
          f"seed {ck['seed']}, устройство {dev}"
          + ("   ⚠ АБЛЯЦИЯ ЭМБЕДДИНГА (G3)" if a.ablate_emb else ""))


if a.sum:
    summarise()
else:
    if a.ident == "model":
        if not a.ckpt:
            sys.exit("--ident model требует --ckpt")
        load_net()
    i, n = (int(v) for v in a.shard.split("/"))
    collect(i, n)
