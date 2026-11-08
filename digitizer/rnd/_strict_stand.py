r"""_strict_stand.py — СТРОГАЯ МЕРА ВЫБРОСОВ ВЫДАЧИ И СТЕНД ИХ ВЫЧИСТКИ (03.11, §6.353).

Мера приёмки (медиана на плоскости ≤ 3 px, покрытие ≥ 0.9) не видит, что трасса на десятках–сотнях строк стоит на вертикали сетки
или рамке, прыгает на цифры глубины и т. п. — лишь бы больше половины строк были на месте (замечание заказчика 03.11 по выборке:
БВп69 ПМ — 32% строк выдачи дальше 10 px от линии эксперта, BEZLUD_051 BK — 14%, а вердикт «верно»).

Строгая мера по паре «наша кривая — эталон того же имени», в пикселях (уровни масштаба — отдельно, мерой приёмки):
  лишнее = строки выдачи дальше 10 px на плоскости от ломаной эталона (в пределах его глубин ± 5 строк);
  пропуск = строки эталона дальше 10 px от ломаной выдачи;
  серии лишнего по классам: «вертикаль» (x почти постоянен: размах ≤ 3 px, ≥ 15 строк — линия сетки / рамка / пунктир),
  «спайк» (≤ 3 строк), «прочее»;
  ЧИСТАЯ = верна мерой приёмки И лишнее ≤ 1% строк И пропуск ≤ 1% строк.
Варианты вычистки (`--filters`) применяются к выдаче в памяти, мера приёмки и строгая — до и после.

  _strict_stand.py --workers 2 [--filters spike3 spike5 flat ...] [--limit N]  → taskS/_strict_<tag>.tsv, сводка в stdout
"""
import sys, argparse, json, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
for _p in (r"F:\nds\Auto", r"F:\nds\Auto\digitizer", r"F:\nds\Auto\digitizer\rnd"):
    sys.path.insert(0, _p)
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

OFF, CLEAN = 10.0, 0.01


def runs_of(mask, ys):
    """серии подряд идущих True (по соседним строкам, разрыв ≤ 2) → [(i0, i1)]"""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1] and ys[j + 1] - ys[j] <= 2:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def strict(S, w, gt, mis=frozenset()):
    """→ строгая мера: w — выдача, уже пересчитанная в шкалу уровня эталона (`relevel`), gt — эталон ({y: x}).
    off_o — строки выдачи, где у эталона есть точка, дальше OFF px от его ломаной; gap — строки выдачи внутри глубин эталона, где
    у эталона точки нет (эксперт линию не вёл); out — строки выдачи вне глубин эталона; off_t — строки эталона дальше OFF от выдачи.
    mis — строки, где уровень масштаба выдачи ≠ уровню эталона: лишнее на них — «уровень», остальное — по виду серии в пикселях."""
    y0, y1 = min(gt), max(gt)
    com = [y for y in sorted(w) if y in gt]
    if len(com) < 30:
        return None
    S._TREES.clear()
    do = S._plane(gt, com, w); dt = S._plane(w, sorted(gt), gt)
    ys = np.array(com); xs = np.array([w[y] for y in com])
    r = {"n_o": len(com), "n_t": len(gt), "off_o": int((do > OFF).sum()), "off_t": int((dt > OFF).sum()),
         "gap": sum(1 for y in w if y0 <= y <= y1 and y not in gt), "out": sum(1 for y in w if y < y0 or y > y1),
         "lvl": 0, "vert": 0, "spike": 0, "other": 0, "r_lvl": 0, "r_vert": 0, "r_spike": 0, "r_other": 0}
    lm = np.array([y in mis for y in com], bool); off = do > OFF
    for i0, i1 in runs_of(off & lm, ys):
        r["lvl"] += i1 - i0 + 1; r["r_lvl"] += 1
    for i0, i1 in runs_of(off & ~lm, ys):
        n = i1 - i0 + 1; seg = xs[i0:i1 + 1]
        k = "vert" if n >= 15 and np.ptp(seg) <= 3 else "spike" if n <= 3 else "other"
        r[k] += n; r["r_" + k] += 1
    return r


# ── вычистка ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def f_spike(tr, L, D=10.0, E=8.0):
    """отлёт ≤ L строк от хорды соседей дальше D px при соседях, близких друг к другу (≤ E px) → строки по хорде"""
    ys = sorted(tr); xs = np.array([tr[y] for y in ys], float); out = dict(tr); n = len(ys); i = 1
    while i < n - 1:
        hit = None
        for j in range(i, min(n - 1, i + L)):
            if ys[j + 1] - ys[i - 1] != j - i + 2:
                break
            a, b = xs[i - 1], xs[j + 1]
            if abs(a - b) > E:
                continue
            t = (np.array(ys[i:j + 1]) - ys[i - 1]) / (ys[j + 1] - ys[i - 1])
            if np.all(np.abs(xs[i:j + 1] - (a + (b - a) * t)) > D):
                hit = j
        if hit is None:
            i += 1; continue
        a, b = xs[i - 1], xs[hit + 1]
        for k in range(i, hit + 1):
            out[ys[k]] = a + (b - a) * (ys[k] - ys[i - 1]) / (ys[hit + 1] - ys[i - 1])
        i = hit + 1
    return out


FILTERS = {"spike3": lambda t: f_spike(t, 3), "spike5": lambda t: f_spike(t, 5), "spike10": lambda t: f_spike(t, 10),
           "spike10e15": lambda t: f_spike(t, 10, E=15.0)}


def l_quiet(segs, tr, ch, R=15, frac=0.15):
    """уровни выдачи без «тихих» смен: смена уровня принимается, только если в ±R строк от неё трасса прыгает ≥ frac ширины
    шкалы (перо ушло за край / переход на перо другого масштаба); иначе уровень прежний → {y: уровень}"""
    segs = sorted(segs or [])
    if len(ch) < 2 or len(segs) < 2:
        return {y: int(l) for a, b, l in segs for y in range(int(a), int(b) + 1)}
    W = abs(ch[0]["x_right"] - ch[0]["x_left"]) or 1
    out, cur = {}, int(segs[0][2])
    for k, (a, b, l) in enumerate(segs):
        if k and int(l) != cur:
            xs = [tr[y] for y in range(int(a) - R, int(a) + R + 1) if y in tr]
            if len(xs) >= 3 and max(xs) - min(xs) >= frac * W:
                cur = int(l)
        for y in range(int(a), int(b) + 1):
            out[y] = cur
    return out


def _runs_rows(ys, Lmin, gap=3):
    """подряд идущие строки (разрыв ≤ gap) длиной ≥ Lmin → [(y0, y1)]"""
    out, st = [], None
    for i, y in enumerate(ys):
        if st is None:
            st = y
        elif y - ys[i - 1] > gap:
            if ys[i - 1] - st + 1 >= Lmin:
                out.append((st, ys[i - 1]))
            st = y
    if st is not None and ys and ys[-1] - st + 1 >= Lmin:
        out.append((st, ys[-1]))
    return out


def _jump_in_out(tr, y0, y1, k=12):
    """насколько трасса сдвинулась, входя в участок [y0, y1] и выходя из него (по строкам y0 − k → y0, y1 → y1 + k)"""
    a = [tr[y] for y in range(y0 - k, y0) if y in tr]; b = [tr[y] for y in range(y1 + 1, y1 + k + 1) if y in tr]
    j_in = abs(tr[y0] - np.median(a)) if a else 0.0
    j_out = abs(tr[y1] - np.median(b)) if b else 0.0
    return j_in + j_out


def s_dedup(wr, Lmin=20, near=3.0, mode="gap", ratio=3.0, min_jump=30.0):
    """две наши кривые на одной туши ≥ Lmin строк: «пришелец» — та, что впрыгнула и выпрыгнула (сдвиг на входе+выходе
    ≥ min_jump и ≥ ratio × сдвига другой); его строки участка → пропуск (gap) или мост по хорде (int)"""
    out = {k: dict(v) for k, v in wr.items()}
    names = list(wr)
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            a, b = wr[A], wr[B]
            ys = [y for y in sorted(a) if y in b and abs(a[y] - b[y]) <= near]
            for y0, y1 in _runs_rows(ys, Lmin):
                ja, jb = _jump_in_out(a, y0, y1), _jump_in_out(b, y0, y1)
                if ja >= min_jump and ja >= ratio * jb:
                    loser = A
                elif jb >= min_jump and jb >= ratio * ja:
                    loser = B
                else:
                    continue
                tr = out[loser]; src = wr[loser]
                if mode == "gap":
                    for y in range(y0, y1 + 1):
                        tr.pop(y, None)
                else:
                    pa = [y for y in range(y0 - 12, y0) if y in src]; pb = [y for y in range(y1 + 1, y1 + 13) if y in src]
                    if not pa or not pb:
                        for y in range(y0, y1 + 1):
                            tr.pop(y, None)
                        continue
                    ya, yb = pa[-1], pb[0]; xa, xb = src[ya], src[yb]
                    for y in range(ya + 1, yb):
                        tr[y] = xa + (xb - xa) * (y - ya) / (yb - ya)
    return out


def _slope(tr, y, sgn, k=8):
    """наклон трассы (px на строку) по k строкам перед y (sgn = −1) или после y (sgn = +1); None — мало точек"""
    ys = [y + sgn * i for i in range(1, k + 1) if y + sgn * i in tr]
    if len(ys) < k // 2:
        return None
    a = np.polyfit(ys, [tr[q] for q in ys], 1)[0]
    return float(a)


def s_cross(wr, ratio=0.5, touch=2.0, k=8, min_turn=0.5):
    """пересечения двух наших кривых: в точке касания / смены знака xA − xB сравнить «каждая продолжает свою» и «обмен хвостами»
    по сохранению наклона (до → после); обмен, если его цена < ratio × цены без обмена и без обмена обе ветки разворачиваются
    (|Δнаклона| ≥ min_turn px/строку) — отражение от точки касания вместо пересечения"""
    out = {k_: dict(v) for k_, v in wr.items()}
    names = list(wr)
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            a, b = out[A], out[B]
            ys = [y for y in sorted(a) if y in b]
            if len(ys) < 50:
                continue
            d = np.array([a[y] - b[y] for y in ys])
            ev = []                                          # точки событий: смена знака или касание (|d| ≤ touch), по одной на серию
            j = 1
            while j < len(ys):
                if (abs(d[j]) <= touch or np.sign(d[j]) != np.sign(d[j - 1])) and ys[j] - ys[j - 1] <= 2:
                    j0 = j
                    while j + 1 < len(ys) and abs(d[j + 1]) <= touch and ys[j + 1] - ys[j] <= 2:
                        j += 1
                    ev.append((ys[j0], ys[j]))
                j += 1
            flips = []
            for y0, y1 in ev:
                sab, sbb = _slope(a, y0, -1, k), _slope(b, y0, -1, k)
                saa, sba = _slope(a, y1, +1, k), _slope(b, y1, +1, k)
                if None in (sab, sbb, saa, sba):
                    continue
                keep = abs(sab - saa) + abs(sbb - sba); swap = abs(sab - sba) + abs(sbb - saa)
                if swap < ratio * keep and abs(sab - saa) >= min_turn and abs(sbb - sba) >= min_turn:
                    flips.append(y1)
            if not flips:
                continue
            flips.append(10 ** 9)
            sw = False; fi = 0
            for y in ys:
                while y > flips[fi]:
                    sw = not sw; fi += 1
                if sw:
                    a[y], b[y] = b[y], a[y]
    return out


def ink_frac(gray, paper, y, xa, xb, dark=45):
    """доля тёмных пикселей (темнее бумаги на dark) вдоль строк y, y+1 между xa и xb без концов ±2 px — перемычка прыжка"""
    lo, hi = int(min(xa, xb)) + 2, int(max(xa, xb)) - 2
    if hi <= lo or not (0 <= y < gray.shape[0] - 1):
        return 1.0
    lo, hi = max(0, lo), min(gray.shape[1], hi)
    seg = np.minimum(gray[y, lo:hi], gray[y + 1, lo:hi]).astype(float)
    return float(np.mean(seg < paper - dark)) if len(seg) else 1.0


def s_inkx(wr, LW, gray, paper, tau=0.5, J=8, E=15.0, Lmax=400, drift=0.5):
    """уход с линии по прыжкам без туши: трасса ушла прыжком > J px, вдоль которого туши < tau (не перевынос — уровень тот же),
    и вернулась таким же прыжком почти туда, откуда ушла (≤ max(E, drift × длины)) не дальше Lmax строк → участок — мост по хорде
    (замер 03.11: AUC доли туши 0.83 на 13789 прыжках 42 листов; < 0.5 — 47% ложных при 0.7% настоящих)"""
    out = {}
    for g, w in wr.items():
        ys = sorted(w); tr = dict(w); lw = LW.get(g, {})
        jumps = [y0 for y0, y1 in zip(ys, ys[1:]) if y1 == y0 + 1 and abs(w[y1] - w[y0]) > J
                 and lw.get(y0, 0) == lw.get(y1, 0) and ink_frac(gray, paper, y0, w[y0], w[y1]) < tau]
        i = 0
        while i < len(jumps):
            a = jumps[i]; xa = w[a]; nxt = i + 1
            for k in range(i + 1, len(jumps)):
                b = jumps[k]
                if b - a > Lmax:
                    break
                if b + 1 in w and abs(w[b + 1] - xa) <= max(E, drift * (b - a)):
                    xb = w[b + 1]
                    for y in range(a + 1, b + 1):
                        tr[y] = xa + (xb - xa) * (y - a) / (b + 1 - a)
                    nxt = k + 1
                    break
            i = nxt
        out[g] = tr
    return out


INKFILTERS = {"inkx": lambda wr, LW, gr, pa: s_inkx(wr, LW, gr, pa), "inkx3": lambda wr, LW, gr, pa: s_inkx(wr, LW, gr, pa, tau=0.3),
              "inkx5L": lambda wr, LW, gr, pa: s_inkx(wr, LW, gr, pa, Lmax=2000), "inkx5d": lambda wr, LW, gr, pa: s_inkx(wr, LW, gr, pa, drift=1.0)}


def s_spikeseg(wr, rgb, net, NF, ts=0.1, tc=0.5, long_=0):
    """§6.356: уколы без туши сети под собой при туши на хорде → хорда (`auto.spike.fix_all`; long_ — и отлёты 4…long_ строк);
    NF[кривая] += замен"""
    from auto import spike as SP
    out = {}
    for g, w in wr.items():
        out[g], log = SP.fix_all(w, rgb, net, ts, tc, long_)
        NF[g] += len(log)
    return out


SEGFILTERS = {"spikeseg": lambda wr, rgb, net, NF: s_spikeseg(wr, rgb, net, NF),
              "spikeseg2": lambda wr, rgb, net, NF: s_spikeseg(wr, rgb, net, NF, ts=0.2, tc=0.5),
              "spikeseg15": lambda wr, rgb, net, NF: s_spikeseg(wr, rgb, net, NF, long_=15)}


SFILTERS = {"dedupgap": lambda wr: s_dedup(wr, mode="gap"), "dedupint": lambda wr: s_dedup(wr, mode="int"),
            "dedupgap50": lambda wr: s_dedup(wr, Lmin=50, mode="gap"),
            "xswap": lambda wr: s_cross(wr, 0.5), "xswap33": lambda wr: s_cross(wr, 0.33), "xswap5t1": lambda wr: s_cross(wr, 0.5, min_turn=1.0)}


def l_jump(segs, tr, ch, frac=0.5, R=2):
    """прыжок трассы ≥ frac ширины шкалы (за ≤ R строк) без смены уровня — перевынос, уровень обязан смениться: прыжок влево → +1
    («большие значения всегда в масштабе, который левее»), вправо → −1; действует до следующей смены уровня выдачи"""
    lv = {y: int(l) for a, b, l in sorted(segs or []) for y in range(int(a), int(b) + 1)}
    if len(ch) < 2:
        return lv
    xl, xr = sorted((ch[0]["x_left"], ch[0]["x_right"])); W = (xr - xl) or 1
    ys = sorted(tr); out = dict(lv); shift = 0
    for i in range(1, len(ys)):
        y0, y1 = ys[i - 1], ys[i]
        if lv.get(y1, 0) != lv.get(y0, 0):
            shift = 0                                    # выдача сама сменила уровень — её решение
        elif y1 - y0 <= R and abs(tr[y1] - tr[y0]) >= frac * W:
            if tr[y1] < tr[y0] and tr[y0] >= xr - 0.15 * W:          # перо ушло за правый край → масштаб левее
                shift += 1
            elif tr[y1] > tr[y0] and tr[y0] <= xl + 0.25 * W and lv.get(y1, 0) + shift > 0:   # вернулось в свой масштаб
                shift -= 1
        if shift:
            out[y1] = min(len(ch) - 1, max(0, lv.get(y1, 0) + shift))
    return out


def c_band(w, ch, m=0.0):
    """§6.355: строки выдачи вне полосы шкал трека (объединение x всех уровней цепочки ± m px) — долой (цифры глубины, рамка
    соседнего трека); у эталона такие строки — редкость"""
    if not ch:
        return w
    xl = min(min(s_["x_left"], s_["x_right"]) for s_ in ch) - m; xr = max(max(s_["x_left"], s_["x_right"]) for s_ in ch) + m
    return {y: x for y, x in w.items() if xl <= x <= xr}


CFILTERS = {"band0": lambda w, ch: c_band(w, ch, 0), "band5": lambda w, ch: c_band(w, ch, 5), "band20": lambda w, ch: c_band(w, ch, 20)}


LFILTERS = {"lvljump": lambda s, t, c: l_jump(s, t, c), "lvljump4": lambda s, t, c: l_jump(s, t, c, 0.4),
            "lvljump6": lambda s, t, c: l_jump(s, t, c, 0.6),
            "lvlq15": lambda s, t, c: l_quiet(s, t, c, 15), "lvlq40": lambda s, t, c: l_quiet(s, t, c, 40),
            "lvlq15f10": lambda s, t, c: l_quiet(s, t, c, 15, 0.10)}


def one(args):
    nm, part, src, got, tr_map, filt = args[:6]
    segdir = args[6] if len(args) > 6 else ""
    import _sample_review as S
    try:
        mt, mw, gts, wr = S.curves(src, got)
    except Exception as e:
        return nm, part, [], f"{type(e).__name__}: {e}"
    import decode_levels as DL
    LG = {c["name"]: S.lv_rows(c) for c in mt["curves"] if c["name"] in gts}
    CH = {c["name"]: DL.build_family(mt, c) for c in mt["curves"] if c["name"] in gts}
    LW = {c["name"]: S.lv_rows(c) for c in mw["curves"] if c["name"] in wr}
    SW = {c["name"]: c.get("segments") for c in mw["curves"] if c["name"] in wr}
    WS = {}                                              # вычистки уровня листа (по всем кривым сразу) — до покривных
    gray = paper = None
    if any(p_ in INKFILTERS for fn in filt for p_ in fn.split("+")):
        import _final_exif
        from dataset_build import find_image
        img = find_image(Path(src))
        if img is None:
            return nm, part, [], "нет скана"
        rgb = _final_exif.load_rgb(str(img))
        gray = rgb.min(axis=2) if rgb.ndim == 3 else rgb
        del rgb
        paper = float(np.percentile(gray[::7, ::7], 80))
    rgb = net = None; NF = {fn: defaultdict(int) for fn in filt}
    if segdir and any(p_ in SEGFILTERS for fn in filt for p_ in fn.split("+")):
        import _final_exif
        from dataset_build import find_image
        img = find_image(Path(src))
        if img is None:
            return nm, part, [], "нет скана"
        rgb = _final_exif.load_rgb(str(img))
        import torch
        torch.set_num_threads(4)                         # сеть туши на ЦП окошками: 4 потока на процесс, не все ядра
        FO = json.loads((Path(segdir) / "folds.json").read_text(encoding="utf-8")); k_ = FO.get(Path(src).parent.parent.name)
        net = str(Path(segdir) / (f"f{k_}.pt" if k_ is not None else "all.pt"))   # сеть туши вне фолда скважины листа
    for fn in filt:
        sp = [p_ for p_ in fn.split("+") if p_ in SFILTERS or p_ in INKFILTERS or p_ in SEGFILTERS]
        if sp:
            ww = wr
            for p_ in sp:
                ww = SFILTERS[p_](ww) if p_ in SFILTERS else INKFILTERS[p_](ww, LW, gray, paper) if p_ in INKFILTERS \
                    else SEGFILTERS[p_](ww, rgb, net, NF[fn])
            WS[fn] = ww
    del rgb
    rows = []
    for g, gt in gts.items():
        if g not in wr:
            continue
        rec = {"curve": g, "n_gt": len(gt), "gt_out": len(gt) - len(c_band(gt, CH.get(g, []))), "n_w": len(wr[g]),
               "w_out": len(wr[g]) - len(c_band(wr[g], CH.get(g, [])))}
        for fn in ["base"] + filt:
            w, lw, lg = WS.get(fn, wr)[g], LW.get(g, {}), LG.get(g, {})
            if not w:
                rec[fn] = {"hon": False}; continue
            for part_ in fn.split("+"):                  # «spike3+lvlq15» — по очереди
                if part_ in FILTERS:
                    w = FILTERS[part_](w)
                elif part_ in CFILTERS:
                    w = CFILTERS[part_](w, CH.get(g, []))
                elif part_ in LFILTERS:
                    lw = LFILTERS[part_](SW.get(g), w, CH.get(g, []))
            S._TREES.clear()
            wl = S.relevel(w, lw, lg, CH.get(g, []))
            hon = S.HON(*S.err(wl, gt))
            st = strict(S, wl, gt, frozenset(y for y in w if lw.get(y, 0) != lg.get(y, 0)))
            rec[fn] = {"hon": bool(hon), **(st or {})}
            if fn != "base" and NF.get(fn):
                rec[fn]["n_fix"] = NF[fn][g]
        rows.append(rec)
    return nm, part, rows, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--filters", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--sheets", default="", help="файл со списком листов (иначе все поле + вне + сорт A)")
    ap.add_argument("--mode", default="NSE19", help="режим выдачи в taskS/rp_lvl (по умолчанию повтор прода NSE19)")
    ap.add_argument("--seg", default=r"F:/nds/output/taskS/ink_seg_f", help="сети туши вне фолда (folds.json, f{k}.pt) — для spikeseg")
    a = ap.parse_args()
    import _sample_review as S
    S.OOF = S.TS / "rp_lvl" / a.mode
    TRACK = S.load_tracks(); SETS = S.eval_sets(); SRC = {q.name: q for q in S.ARCH.glob("*/wlg/*.nlgx")}
    only = set(Path(a.sheets).read_text(encoding="utf-8").split("\n")) if a.sheets else None
    jobs = []
    for part, SS in SETS.items():
        for nm in sorted(SS):
            if only is not None and nm not in only:
                continue
            got = next(iter(sorted((S.OOF / S.sheet_dir(nm)).glob("*_auto.nlgx"))), None)
            if got is not None and nm in SRC:
                jobs.append((nm, part, SRC[nm], got, TRACK.get(nm, {}), a.filters, a.seg))
    if a.limit:
        jobs = jobs[::max(1, len(jobs) // a.limit)][:a.limit]
    print(f"★ листов {len(jobs)}, вычистка {a.filters or '—'}, процессов {a.workers}")
    # 08.11 (отключения питания): итог каждого листа сразу дописывается в `_strict_<tag>.part.jsonl` (первая строка — подпись запуска);
    #   перезапуск с той же подписью досчитывает только недостающие листы
    import os, hashlib
    part = Path(r"F:/nds/output/taskS") / f"_strict_{a.tag}.part.jsonl"
    sig = json.dumps({"mode": a.mode, "filters": a.filters, "limit": a.limit, "sheets": a.sheets, "seg": a.seg,
                      "jobs": hashlib.md5("\n".join(j[0] for j in jobs).encode("utf-8")).hexdigest()}, ensure_ascii=False)
    res = []
    if part.exists():
        ln = part.read_text(encoding="utf-8").splitlines()
        if ln and ln[0] == sig:
            for l in ln[1:]:
                try:
                    res.append(json.loads(l))
                except ValueError:
                    pass                                  # строка, оборванная отключением питания
            print(f"★ досчёт: готовых листов с прошлого раза {len(res)}")
        else:
            part.replace(part.with_name(part.name + f".stale_{int(time.time())}"))
    have = {r[0] for r in res}
    todo = [j for j in jobs if j[0] not in have]
    fresh = not part.exists()
    tail_ok = fresh or part.read_bytes()[-1:] == b"\n"
    fh = open(part, "a", encoding="utf-8")
    if fresh:
        fh.write(sig + "\n"); fh.flush()
    elif not tail_ok:
        fh.write("\n"); fh.flush()                        # оборванная последняя строка — новая с начала строки
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(one, todo, chunksize=4), 1):
            res.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush(); os.fsync(fh.fileno())
            if k % 200 == 0:
                print(f"  {k}/{len(todo)} листов, {time.time() - t0:.0f} с")
    fh.close()
    out = Path(r"F:/nds/output/taskS") / f"_strict_{a.tag}.json"
    tmp = out.with_name(out.name + ".tmp")                # через подмену: оборванный итог не выглядит готовым для драйвера
    tmp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8"); tmp.replace(out)
    part.replace(part.with_name(part.name + ".done"))
    report(res, ["base"] + a.filters)
    print(f"★ {out}")


def report(res, fns):
    for part in ["поле+вне", "сорт A"]:
        R = [(nm, rec) for nm, p, rows, _ in res if (p == "сорт A") == (part == "сорт A") for rec in rows]
        print(f"\n## {part}: пар «наша — эталон того же имени» {len(R)}")
        for fn in fns:
            ok = [rec[fn] for _, rec in R if rec[fn].get("n_o")]
            hon = [s for s in ok if s["hon"]]
            clean = [s for s in hon if s["off_o"] <= CLEAN * s["n_o"] and s["off_t"] <= CLEAN * s["n_t"]]
            dirty5 = [s for s in hon if s["off_o"] > 0.05 * s["n_o"]]
            oo = sum(s["off_o"] for s in ok); no = sum(s["n_o"] for s in ok)
            gp = [s for s in hon if s["gap"] > CLEAN * s["n_o"]]
            print(f"  {fn:11s} верных мерой приёмки {len(hon):5d}; из них ЧИСТЫХ {len(clean):5d} ({len(clean) / max(len(hon), 1):.0%}), "
                  f"с лишним > 5% строк {len(dirty5):4d}; лишних строк всего {oo / max(no, 1):.2%}; у {len(gp)} верных выдача "
                  f"> 1% строк там, где эксперт линию не вёл")
            if fn == "base":
                for sub, S_ in (("у верных", hon), ("у всех", ok)):
                    tot = sum(s["off_o"] for s in S_) or 1
                    print(f"     лишнее {sub}: уровень {sum(s['lvl'] for s in S_) / tot:.0%} ({sum(s['r_lvl'] for s in S_)} серий), вертикаль {sum(s['vert'] for s in S_) / tot:.0%} ({sum(s['r_vert'] for s in S_)} серий), "
                          f"спайк ≤ 3 строк {sum(s['spike'] for s in S_) / tot:.0%} ({sum(s['r_spike'] for s in S_)} серий), "
                          f"прочее {sum(s['other'] for s in S_) / tot:.0%} ({sum(s['r_other'] for s in S_)} серий)")
        if len(fns) > 1:
            for fn in fns[1:]:
                d = Counter()
                for _, rec in R:
                    b, f = rec["base"], rec[fn]
                    d["приёмка +"] += (not b["hon"]) and f["hon"]; d["приёмка −"] += b["hon"] and not f["hon"]
                print(f"  {fn}: приёмка стала верной {d['приёмка +']}, перестала {d['приёмка −']}")


if __name__ == "__main__":
    main()
