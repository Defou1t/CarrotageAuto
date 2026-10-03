r"""level_ink.py — УРОВЕНЬ МАСШТАБА (1×/5×/25×…) ПО СКАНУ ВОКРУГ ЛИНИИ (§6.266, 03.10).

ЗАЧЕМ. Значение в LAS зависит от уровня шкалы участка кривой: на 5×-участке, записанном как 1×, ошибка впятеро. Прежний
декодер оборотов (`decode_levels`) смотрит только на трассу и только у резистивных кривых; в значениях он не лучше
«всё на 1×» (§6.262–6.265). Заказчик (03.10): «большие значения всегда в масштабе, который левее»; штрих 5× такой же;
«иногда пунктир от завершения масштаба до начала (справа налево)»; «каждую линию отдельно смотрю». Отсюда модель
смотрит на скан вокруг самой линии.

ЧТО ДЕЛАЕТ. Для кривой с цепочкой масштабов (≥ 2 уровней в каркасе) — каждая 8-я строка трассы выдачи:
- профиль туши поперёк полосы шкал (128 корзин, максимум затемнения по 8 строкам);
- занятость полосы другими трассами выдачи;
- тушь в точках «близнеца» (то же значение на соседнем уровне);
- уровень прежнего декодера и константы цепочки.

1D-свёртка по позициям (поле ±2000 строк) → уровни 0…3 (маска по длине цепочки). Ансамбль трёх моделей (среднее
вероятностей) → argmax → на все строки по ближайшей позиции → `enforce_min_run` 25 → сегменты.

ЗАМЕРЕНО ДО ВНЕСЕНИЯ (§6.266 v2, именные честные В ЗНАЧЕНИЯХ, `_level_ink.py`):
- поле, вне фолда по скважинам: 683 → 712 (+29, листов ↑51/↓27, p = 0.0054);
- держанный сорт A: 172 → 181 (+9).

⚠ АРХИТЕКТУРА И ПРИЗНАКИ ПОВТОРЯЮТ `digitizer/rnd/_level_ink.py` (scan_curve + feats + Net) В ТОЧНОСТИ. Расхождение своей
копии с обучающей не всплывает никогда (§6.90) ⇒ любая правка — в обоих местах, проверка — сверкой признаков и уровней
на кэше (`digitizer/rnd/_level_ink_parity.py`).
Без torch или без файла весов — прежний декодер, и об этом говорится вслух (дисциплина `seq_model`, §6.68).
"""
import json
import re
from pathlib import Path

import numpy as np

KMAX = 4
RES = {"GZ", "OGZ", "BK", "MBK", "BKZ", "BMK", "IK", "IKA", "IKR", "PZ", "MGZ", "MPZ", "BKP", "LL", "LLD", "LLS"}
_MODELS = Path(__file__).resolve().parent / "models"
_NET = {}
_SAID = set()


def _announce(msg):
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"  {msg}")


def group(root):
    r = re.sub(r"^BKZ_", "", root)
    if r in RES:
        return 0
    if r == "GK":
        return 1
    if r in ("NGK", "NNK", "NNKB", "NNKM", "NKTB", "NKTM"):
        return 2
    if r.startswith("TM"):
        return 3
    return 4


def _val(s, x):
    return s["v_left"] + (x - s["x_left"]) * (s["v_right"] - s["v_left"]) / ((s["x_right"] - s["x_left"]) or 1)


def _pix(s, v):
    return s["x_left"] + (v - s["v_left"]) * (s["x_right"] - s["x_left"]) / ((s["v_right"] - s["v_left"]) or 1)


def _seg_levels(segs, rows):
    out = np.zeros(len(rows), np.int16)
    for y0, y1, lv in segs or []:
        if lv:
            out[(rows >= y0) & (rows <= y1)] = lv
    return out


def dense_rows(top_y, xs, null, max_gap=200):
    """плотная трасса (строка → x) как `_multi_replica_probe.dense`: вершины + линейная вставка в разрывах ≤ max_gap"""
    pts = [(top_y + i, x) for i, x in enumerate(xs) if x != null]
    out = {}
    for (y0v, x0v), (y1v, x1v) in zip(pts, pts[1:]):
        out[y0v] = float(x0v)
        if 0 < y1v - y0v <= max_gap:
            for yy in range(y0v + 1, y1v):
                out[yy] = x0v + (x1v - x0v) * (yy - y0v) / (y1v - y0v)
    if pts:
        out[pts[-1][0]] = float(pts[-1][1])
    ty = np.array(sorted(out), np.int32)
    return ty, np.array([out[y] for y in ty], np.float32)


def paper_of(gray):
    """уровень бумаги по строке — 90-й перцентиль каждого 4-го пикселя (как скан обучения)"""
    H = gray.shape[0]
    paper = np.empty(H, np.int16)
    for r in range(0, H, 2048):
        paper[r:r + 2048] = np.percentile(gray[r:r + 2048, ::4], 90, axis=1)
    return paper


def features(gray, paper, chain, name, root, ty, tx, lw_segs, others, stride=8, nb=128):
    """признаки одной кривой — ТО ЖЕ, что `_level_ink.scan_curve` без меток. others — [(имя, строки, x)] трасс выдачи
    листа (всех, кроме DA пустых; своя исключается по имени). → dict или None (короткая кривая / узкая полоса)."""
    H, W = gray.shape
    st, half = stride, stride // 2
    ch = chain; K = len(ch)
    xl = min(min(s["x_left"], s["x_right"]) for s in ch); xr = max(max(s["x_left"], s["x_right"]) for s in ch)
    w = max(8.0, xr - xl); m = 0.05 * w
    bx0 = int(max(0, np.floor(xl - m))); bx1 = int(min(W, np.ceil(xr + m)))
    ncol = bx1 - bx0
    if len(ty) < 50 or ncol < nb:
        return None
    y0 = int(max(half, ty[0])); y1 = int(min(H - st + half, ty[-1]))
    if y1 - y0 < st * 10:
        return None
    Pp = np.arange(y0, y1 + 1, st, dtype=np.int64)
    n = len(Pp)
    j = np.clip(np.searchsorted(ty, Pp), 0, len(ty) - 1); has = ty[j] == Pp
    xo = np.where(has, tx[j], np.nan).astype(np.float64)
    lw = _seg_levels(lw_segs, Pp)
    TWX = []
    for k in range(min(K, KMAX) - 1):
        TWX.append(_pix(ch[k + 1], _val(ch[k], xo)))
        TWX.append(_pix(ch[k], _val(ch[k + 1], xo)))
    NT = 2 * (KMAX - 1)
    tw = np.zeros((n, NT), np.uint8); twin = np.zeros((n, NT), np.uint8)
    ink = np.zeros((n, nb), np.uint8)
    edges = np.linspace(0, ncol, nb + 1).astype(np.int64)[:-1]
    CH = 512
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH); nbk = c1 - c0
        r0 = int(Pp[c0]) - half; r1 = r0 + nbk * st
        G = gray[r0:r1, bx0:bx1].astype(np.int16)
        D = np.clip(paper[r0:r1, None] - G, 0, 255).astype(np.uint8)
        Dk = D.reshape(nbk, st, ncol).max(1)
        ink[c0:c1] = np.maximum.reduceat(Dk, edges, axis=1)
        for t, xc in enumerate(TWX):
            xs_ = xc[c0:c1]
            ok = np.isfinite(xs_) & (xs_ >= bx0 + 3) & (xs_ <= bx1 - 4)
            ci = np.round(np.where(ok, xs_, bx0 + 3) - bx0).astype(np.int64)
            idx = np.clip(ci[:, None] + np.arange(-3, 4)[None, :], 0, ncol - 1)
            v = Dk[np.arange(nbk)[:, None], idx].max(1)
            tw[c0:c1, t] = np.where(ok, v, 0); twin[c0:c1, t] = ok
    occ = np.zeros((n, nb), np.uint8)
    for nm, oy, ox in others:
        if nm == name or oy[-1] < Pp[0] - half or oy[0] > Pp[-1] + half:
            continue
        k = np.clip(np.searchsorted(oy, Pp), 1, len(oy) - 1)
        kn = np.where(np.abs(oy[k] - Pp) <= np.abs(oy[k - 1] - Pp), k, k - 1)
        near = np.abs(oy[kn] - Pp) <= half
        b = np.floor((ox[kn] - bx0) / ncol * nb).astype(np.int64)
        sel = near & (b >= 0) & (b < nb)
        occ[np.flatnonzero(sel), b[sel]] = 1
    s0, s1 = ch[0], ch[1]
    r0_ = (s0["v_right"] - s0["v_left"]) or 1.0
    f = abs((s1["v_right"] - s1["v_left"]) / r0_)
    shift = float(abs(s1["v_left"] - s0["v_right"]) < 1e-6 * max(1.0, abs(s0["v_right"])))
    u = np.where(has, (xo - bx0) / ncol, np.nan).astype(np.float32)
    const = np.array([float(K == 2), float(K == 3), float(K >= 4), np.log(max(1e-3, f)), shift]
                     + [float(group(root) == g_) for g_ in range(5)], np.float32)
    return dict(K=K, P=Pp, ink=ink, occ=occ, u=u, has=has, tw=tw, twin=twin, lw=lw.astype(np.int8), const=const)


def _build_net(meta):
    import torch.nn as nn

    class CNorm(nn.Module):
        def __init__(s, ch):
            super().__init__(); s.ln = nn.LayerNorm(ch)

        def forward(s, x):
            return s.ln(x.transpose(1, 2)).transpose(1, 2)

    class Net(nn.Module):
        """⚠ ПОВТОРЯЕТ `digitizer/rnd/_level_ink.py` Net в точности"""

        def __init__(s, cin, ch=64, dil=(1, 2, 4, 8, 16, 32) * 2, k=5):
            super().__init__()
            s.inp = nn.Conv1d(cin, ch, 1)
            s.blocks = nn.ModuleList([nn.Sequential(CNorm(ch), nn.GELU(), nn.Conv1d(ch, ch, k, dilation=d, padding=d * (k // 2)))
                                      for d in dil])
            s.out = nn.Sequential(CNorm(ch), nn.GELU(), nn.Conv1d(ch, KMAX, 1))

        def forward(s, x):
            h = s.inp(x)
            for b in s.blocks:
                h = h + b(h)
            return s.out(h)

    return Net(meta["cin"], meta["ch"], tuple(meta["dil"]), meta["k"])


def resolve(spec, sheet=None):
    """spec — имя файла весов в `auto/models` (прод) или `oof:<каталог>` (замер: лист поля — модель ТОГО фолда, что держал
    его скважину, по карте `level_ink_folds.json`; прочие листы — модель на всём поле). → путь или None."""
    if not spec:
        return None
    if spec.startswith("oof:"):
        d = Path(spec[4:])
        mp = d / "level_ink_folds.json"
        key = str(mp)
        if key not in _NET:
            _NET[key] = json.loads(mp.read_text(encoding="utf-8"))
        k = _NET[key].get(sheet)
        return d / (f"level_ink_f{k}.pt" if k is not None else "level_ink_all.pt")
    p = Path(spec)
    if not p.is_absolute():
        p = _MODELS / spec
    return p if p.is_file() else None


def available(spec):
    try:
        import torch                                        # noqa: F401
    except Exception:
        return False
    if not spec:
        return False
    if spec.startswith("oof:"):
        return (Path(spec[4:]) / "level_ink_all.pt").is_file()
    return resolve(spec) is not None


def _load(path):
    key = str(path)
    if key in _NET:
        return _NET[key]
    import torch
    ck = torch.load(key, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu"
    nets = []
    for sd in ck["sds"]:
        net = _build_net(ck)
        net.load_state_dict(sd); net.eval()
        nets.append(net.to(dev))
    _NET[key] = (nets, ck, dev)
    return _NET[key]


def _input(f, Lb, nb, dev):
    """вход сети [1, CIN, Lb] — ТО ЖЕ, что `_level_ink.feats` (края: позиции за концом кривой берут признаки последней,
    маскированные valid; тушь близнецов — без маски, как при обучении)"""
    import torch
    import torch.nn.functional as F
    n = len(f["P"])
    gi = np.minimum(np.arange(Lb), n - 1)
    valid = torch.from_numpy((np.arange(Lb) < n).astype(np.float32)).to(dev)[None]
    T = lambda a_: torch.from_numpy(np.ascontiguousarray(a_[gi])).to(dev)[None]
    ink = T(f["ink"]).float() / 255.0; occ = T(f["occ"]).float()
    u = T(np.nan_to_num(f["u"], nan=0.0).astype(np.float32)); has = T(f["has"].astype(np.float32))
    hv = has * valid
    b = torch.clamp((u * nb).long(), 0, nb - 1)
    oh = F.one_hot(b, nb).float() * hv[..., None]
    rel = b[..., None] + torch.arange(-nb // 2, nb // 2, device=dev)
    rv = ((rel >= 0) & (rel < nb)).float() * hv[..., None]
    rel = rel.clamp(0, nb - 1)
    rink = torch.gather(ink, 2, rel) * rv; rocc = torch.gather(occ, 2, rel) * rv
    lw = T(np.minimum(f["lw"], KMAX - 1).astype(np.int64))
    cst = torch.from_numpy(f["const"]).to(dev)[None, None, :].expand(1, Lb, -1)
    sc = [u[..., None] * hv[..., None], hv[..., None], T(f["tw"]).float() / 255.0, T(f["twin"]).float(),
          F.one_hot(lw, KMAX).float() * valid[..., None], cst * valid[..., None]]
    x = torch.cat([ink * valid[..., None], occ * valid[..., None], oh, rink, rocc] + sc, dim=2)
    return x.transpose(1, 2).contiguous()


def predict(f, path):
    """→ уровни по позициям (argmax среднего softmax ансамбля)"""
    import torch
    nets, meta, dev = _load(path)
    n = len(f["P"])
    Lb = -(-n // 1024) * 1024
    with torch.no_grad():
        x = _input(f, Lb, meta["nb"], dev)
        kc = min(f["K"], KMAX)
        pr = 0
        for net in nets:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lo = net(x)
            lo = lo.float()
            lo[:, kc:, :] = -1e4
            pr = pr + torch.softmax(lo, dim=1)
        pr = (pr / len(nets))[0, :, :n].cpu().numpy()
    return pr.argmax(0).astype(np.int16)


def min_run_runs(starts, levels, ends, min_run):
    """`refine.enforce_min_run` на пробегах (строки сплошные) — ТО ЖЕ, что `_level_ink.min_run_runs`"""
    R = [[int(s), int(e), int(l)] for s, e, l in zip(starts, ends, levels)]

    def merge(R):
        out = []
        for r in R:
            if out and out[-1][2] == r[2]:
                out[-1][1] = r[1]
            else:
                out.append(r)
        return out
    R = merge(R)
    changed = True
    while changed:
        changed = False
        for k2, (s, e, lv) in enumerate(R):
            if e - s + 1 >= min_run:
                continue
            pv = R[k2 - 1] if k2 > 0 else None; nx = R[k2 + 1] if k2 + 1 < len(R) else None
            if pv is None and nx is None:
                continue
            if pv is None:
                to = nx[2]
            elif nx is None:
                to = pv[2]
            else:
                to = pv[2] if (pv[1] - pv[0] + 1) >= (nx[1] - nx[0] + 1) else nx[2]
            if to != lv:
                R[k2][2] = to; R = merge(R); changed = True
                break
    return R


def segments(f, levels, top_y, n_rows, stride=8, min_run=25):
    """уровни по позициям → сегменты тегов 35494/35496/35498 на [top_y, top_y + n_rows − 1]: строки выше первой позиции —
    уровень 0, ниже последней — уровень последней (как мера: точки смены наследуются)"""
    half = stride // 2
    Pp = f["P"].astype(np.int64); p = levels
    ch_ = np.flatnonzero(np.diff(np.concatenate([[-1], p])) != 0)
    starts = Pp[ch_] - half; starts[0] = Pp[0] - half
    ends = np.concatenate([Pp[ch_[1:]] - half - 1, [Pp[-1] - half + stride - 1]])
    R = min_run_runs(starts, p[ch_], ends, min_run)
    y_end = top_y + n_rows - 1
    segs = []
    if R[0][0] > top_y:
        segs.append([top_y, R[0][0] - 1, 0])
    for s, e, lv in R:
        s = max(s, top_y)
        if s > y_end:
            break
        if segs and segs[-1][2] == lv:
            segs[-1][1] = e
        else:
            segs.append([s, e, lv])
    segs[-1][1] = y_end
    for i in range(len(segs) - 1):                  # сплошное покрытие: конец = начало следующего − 1
        segs[i][1] = segs[i + 1][0] - 1
    return [s[0] for s in segs], [s[1] for s in segs], [int(s[2]) for s in segs]


# ───────────── §6.267: сдвиговые цепочки (термометрия) — уровень по гладким участкам трассы ─────────────

def is_shift(chain):
    """сдвиговая цепочка: у каждой пары соседних уровней v_left растёт, ширина диапазона та же (±20%)"""
    if len(chain) < 2:
        return False
    for p_, q_ in zip(chain, chain[1:]):
        wp = p_["v_right"] - p_["v_left"]; wq = q_["v_right"] - q_["v_left"]
        if not (q_["v_left"] > p_["v_left"]) or wp == 0 or abs(wq - wp) > 0.2 * abs(wp):
            return False
    return True


def decode_shift(ty, tx, chain, step=0.03, min_len=150, edge=20):
    """§6.267 — ТО ЖЕ, что `digitizer/rnd/_level_tm.py`. Гладкие участки плотной трассы (|Δx| ≤ step·ширины на строку,
    ≥ min_len строк); первый — уровень 0; следующий — уровень предыдущего или +1, что даёт меньший разрыв значения между
    концом предыдущего и началом этого (медианы edge крайних строк). → {строка начала участка: уровень} или None."""
    K = len(chain)
    w = abs(chain[0]["x_right"] - chain[0]["x_left"]) or 1.0
    tx = tx.astype(np.float64)
    ok = (np.diff(ty) == 1) & (np.abs(np.diff(tx)) <= step * w)
    R, i, n = [], 0, len(ty)
    while i < n - 1:
        if not ok[i]:
            i += 1; continue
        j = i
        while j < n - 1 and ok[j]:
            j += 1
        if ty[j] - ty[i] + 1 >= min_len:
            R.append((i, j))
        i = j + 1
    if not R:
        return None
    lv, k, pv = {}, 0, None
    for i, j in R:
        xs = float(np.median(tx[i:i + edge])); xe = float(np.median(tx[max(i, j - edge + 1):j + 1]))
        if pv is not None and k + 1 < K:
            if abs(_val(chain[k + 1], xs) - pv) < abs(_val(chain[k], xs) - pv):
                k += 1
        lv[int(ty[i])] = k
        pv = _val(chain[k], xe)
    return lv


def segments_from_changes(lv, top_y, n_rows):
    """точки смены {строка: уровень} → сегменты на [top_y, top_y + n_rows − 1] (до первой точки — 0, после — наследуется)"""
    y_end = top_y + n_rows - 1
    pts = sorted((max(int(y), top_y), int(l)) for y, l in lv.items() if y <= y_end)
    segs = [[top_y, y_end, 0]]
    for y, l in pts:
        if y == segs[-1][0]:
            segs[-1][2] = l
        elif l != segs[-1][2]:
            segs[-1][1] = y - 1; segs.append([y, y_end, l])
    out = []
    for s_ in segs:
        if out and out[-1][2] == s_[2]:
            out[-1][1] = s_[1]
        else:
            out.append(s_)
    return [s_[0] for s_ in out], [s_[1] for s_ in out], [s_[2] for s_ in out]

