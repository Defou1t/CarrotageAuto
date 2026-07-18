r"""decode_img_mbk.py — ТРАЕКТОРИЯ-ДЕКОД уровней ×1/×5 для MBK (Трек 4, сессия 3, 17.07).

ПОЧЕМУ ОТДЕЛЬНО ОТ decode_levels: для MBK уровень (×1/×5/×25) — НЕ функция значения и НЕ
чистый wrap. MBK = одна сильно-свингующая линия; тот же x представим на неск. шкалах
(x даёт v=2 на ×1 ИЛИ v=11 на ×5 — обе в кадре), значение ~непрерывно через переход. Чистый
value-DP (decode_levels.decode) даёт MBK 0.538 (замер Archive 82 кривых): демоутит в ×1 слишком
рано (GT1→DP0 21пп), т.к. 46% подъёмов без чистого x-скачка.

РЕШЕНИЕ (подтверждено абляцией): добавить IMAGE off-scale сигнал — перо у ПРАВОГО рельса шкалы
(rightmost curve-чернило у x_right) => можно вверх; иначе подъём дорог. + sticky-down (держим
×5-зону; сам по себе НЕ помогает, весь выигрыш от image). Результат: MBK 0.538→0.595 (+5.7пп,
+23/−15 кривых), per-curve медиана, 82 кривых / 21 скв (широко). Плато 0.595 = recall rail-детекта
66% (42% подъёмов БЕЗ видимого off-scale = конвенция эксперта, не восстановима из картинки —
проверено: рост чувствительности детектора поднимает FP, net 0). Параметры-оптимум ниже.

СКОУП СТРОГО MBK. GZ/OGZ/PZ/BK декодятся обычным decode_levels.decode (у них чистый wrap; image-
гейт их РЕГРЕССИРУЕТ). BK 0.862 уже у потолка (оборачивается чисто), image-гейт → 0.826.

Использование:
    from decode_img_mbk import mbk_levels, is_mbk
    if is_mbk(curve_name):
        lv = mbk_levels(xs_by_row, family, gray, ...)   # gray = grayscale скан (np.uint8 HxW)
    else:
        lv = decode_levels.decode(...)

CLI (воспроизводит метрику Archive): python decode_img_mbk.py [--limit N]
"""
import sys, math, re
from pathlib import Path
import numpy as np
from decode_levels import scale_map

# Оптимум (rail_tune / decode_img_sweep, 17.07): rail=0.12, look=12, down_mult=2, img_w=15.
DEFAULTS = dict(rail=0.12, look=12, down_mult=2.0, img_w=15.0, ink_thr=115, lam=0.4,
                dxfrac=0.12, gate_w=4.0)


def is_mbk(curve_name):
    return re.sub(r"\d+$", "", curve_name.split()[0]).upper() == "MBK"


def _rightmost_leftmost(ink_row, xL, xR, gap=4, minw=3, maxw=60):
    """Крайние x curve-ранов (ширина minw..maxw, грид тоньше) в [xL,xR]."""
    seg = ink_row[xL:xR + 1]
    xs = np.nonzero(seg)[0]
    if not len(xs):
        return None, None
    rmost = lmost = None
    for c in np.split(xs, np.nonzero(np.diff(xs) > gap)[0] + 1):
        w = c[-1] - c[0] + 1
        if minw <= w <= maxw:
            r = int(c[-1]) + xL; l = int(c[0]) + xL
            if rmost is None or r > rmost:
                rmost = r
            if lmost is None or l < lmost:
                lmost = l
    return lmost, rmost


def rail_evidence(gray, family, rows, rail=0.12, look=12, ink_thr=115):
    """per-row bool: было ли касание ПРАВОГО (up) / ЛЕВОГО (dn) рельса в окне [-look,0].
    up => перо ушло за край ×1 (off-scale) => подъём уровня оправдан."""
    H, W = gray.shape
    ink = gray < ink_thr
    xL = min(family[0]["x_left"], family[0]["x_right"])
    xR = max(family[0]["x_left"], family[0]["x_right"])
    Wd = (xR - xL) or 1
    rr = xR - rail * Wd
    rl = xL + rail * Wd
    touchR = {}; touchL = {}
    for y in rows:
        l, r = _rightmost_leftmost(ink[y], xL, xR) if 0 <= y < H else (None, None)
        touchR[y] = (r is not None and r >= rr)
        touchL[y] = (l is not None and l <= rl)
    rs = sorted(rows)
    up = {}; dn = {}
    for i, y in enumerate(rs):
        win = rs[max(0, i - look):i + 1]
        up[y] = any(touchR[z] for z in win)
        dn[y] = any(touchL[z] for z in win)
    return up, dn


def decode_img(xs_by_row, family, up_ev, dn_ev, lam=0.4, dxfrac=0.12, gate_w=4.0,
               down_mult=2.0, img_w=15.0):
    """DP по непрерывности лог-значения + IMAGE rail-гейт (up_ev) + sticky-down (down_mult).
    ВВЕРХ дёшев только если up_ev[y] (перо было у правого рельса); ВНИЗ удорожен (держим зону),
    кроме left-rail возврата (dn_ev). up_ev/dn_ev: dict row->bool."""
    maps = [scale_map(s) for s in family]
    K = len(family)
    rows = sorted(xs_by_row)
    if K <= 1 or len(rows) < 2:
        return {y: 0 for y in rows}
    width = abs(family[0]["x_right"] - family[0]["x_left"]) or 1
    dxthr = max(1.0, dxfrac * width)

    def logv(k, x):
        return math.log(abs(maps[k](x)) + 1.0)
    dp = [0.0] * K
    back = []
    for i in range(1, len(rows)):
        y = rows[i]; x = xs_by_row[y]; xp = xs_by_row[rows[i - 1]]; dx = x - xp
        ndp = [1e18] * K; bk = [0] * K
        lvp = [logv(kp, xp) for kp in range(K)]
        for k in range(K):
            vk = logv(k, x); best = 1e18; bki = 0
            for kp in range(K):
                if k == kp:
                    tc = 0.0
                else:
                    nlev = k - kp
                    want = -1 if nlev > 0 else 1
                    aligned = (dx * want) > 0
                    mag = min(1.0, abs(dx) / dxthr)
                    consistency = mag if aligned else 0.0
                    tc = lam * abs(nlev) + gate_w * lam * (1.0 - consistency)
                    if nlev > 0:                          # ВВЕРХ: нужен image off-scale
                        if not up_ev.get(y, False):
                            tc += img_w * lam * nlev
                    else:                                 # ВНИЗ: держим ×5-зону
                        if not dn_ev.get(y, False):
                            tc *= down_mult
                c = dp[kp] + (vk - lvp[kp]) ** 2 + tc
                if c < best:
                    best = c; bki = kp
            ndp[k] = best; bk[k] = bki
        dp = ndp; back.append(bk)
    k = int(np.argmin(dp))
    lv = {rows[-1]: k}
    for i in range(len(rows) - 1, 0, -1):
        k = back[i - 1][k]; lv[rows[i - 1]] = k
    return lv


def eager_demote(lv, xs_by_row, family):
    """приоритет НИЗШЕМУ множителю. value из DP; уровень = наименьший k, где |value| <= vmax[k].
    Замер Archive (81 MBK): dp 0.418/44.6flip/x1=0.81 → dp+eager 0.435/12.3flip/x1=0.98 (чистейшая
    трасса, максимум ×1).
    ⚠ КАВЕАТ (QC-график 17.07 BOGAT_011 MBK1): eager СИСТЕМАТИЧЕСКИ НЕДО-ЧИТАЕТ высоко-R зоны —
    «prefer ×1» буквально теряет ×5/×25-экскурсии (геологически важные пики). Эксперт держит ×1
    лишь 57% (не лень — реальные высоко-R зоны). ⇒ eager ПЕРЕ-исполняет конвенцию; для value НЕ
    брать в высоко-R. mode="image" (ближе к эксперту, сохраняет пики) — безопаснее для продакшна."""
    maps = [scale_map(s) for s in family]
    vmax = [max(abs(s["v_left"]), abs(s["v_right"])) for s in family]
    out = {}
    for y, x in xs_by_row.items():
        L = min(lv.get(y, 0), len(family) - 1)
        v = abs(maps[L](x)); k = 0
        while k < len(family) - 1 and v > 1.02 * vmax[k]:
            k += 1
        out[y] = k
    return out


def value_hampel(lv, xs_by_row, family, win=2):
    """Убирает изолированные level-flip (5×-спайки значения): строку кидаем на уровень с лог-value
    ближайшим к медиане окна соседей. Замер: dp+eager+hampel corr 0.472 (ЛУЧШИЙ по value), НО
    flip/1k 80 (в 6× грязнее eager) ⇒ хуже для визуал-QC. Опция для чистой value-фиделити (LAS),
    НЕ для nlgx-QC. По умолчанию ВЫКЛ."""
    import math
    maps = [scale_map(s) for s in family]; K = len(family)
    rows = sorted(xs_by_row)
    logv = {y: math.log(abs(maps[min(lv[y], K - 1)](xs_by_row[y])) + 1.0) for y in rows}
    out = dict(lv)
    for i, y in enumerate(rows):
        lo = max(0, i - win); hi = min(len(rows), i + win + 1)
        neigh = [logv[rows[j]] for j in range(lo, hi) if j != i]
        if len(neigh) < 2:
            continue
        med = float(np.median(neigh)); best_k = lv[y]; best_d = abs(logv[y] - med)
        for k2 in range(K):
            d = abs(math.log(abs(maps[k2](xs_by_row[y])) + 1.0) - med)
            if d < best_d - 1e-9:
                best_d = d; best_k = k2
        out[y] = best_k
    return out


def mbk_levels(xs_by_row, family, gray, mode="image", **kw):
    """Полный MBK-декод. gray=None => fallback на чистый DP (безопасно без картинки).
    mode:
      "image"  — DP + image off-scale rail-гейт (level-acc 0.595 vs ленивый GT; corr_LAS 0.446).
      "eager"  — DP + eager-demote (КОНВЕНЦИЯ Эдуарда prefer ×1; ЧИЩЕ всех для nlgx-QC; corr 0.435).
      "corr"   — DP + eager-demote + value-Hampel (лучший value-corr 0.472, но визуально грязный).
    ★ РЕФРЕЙМ 17.07: level-acc — ложный прокси; истина = value-фиделити (corr vs LAS): GT-oracle
    0.844, любой автономный декод из трассы+картинки ≤0.47 (×5 vs ×25 неразличимы без внешнего
    scale-сигнала: image 42% слеп, соседняя кривая не делит ×5/×25, шапка = только лестница).
    Остаток ×5/×25 = QC эксперта. Выбор mode — за Эдуардом по QC в NeuraLOG."""
    p = {**DEFAULTS, **kw}
    from decode_levels import decode
    if gray is None or len(family) < 2 or mode == "dp":
        base = decode(xs_by_row, family, lam=p["lam"], dxfrac=p["dxfrac"], gate_w=p["gate_w"])
    else:
        rows = sorted(xs_by_row)
        up_ev, dn_ev = rail_evidence(gray, family, rows, rail=p["rail"], look=p["look"],
                                     ink_thr=p["ink_thr"])
        base = decode_img(xs_by_row, family, up_ev, dn_ev, lam=p["lam"], dxfrac=p["dxfrac"],
                          gate_w=p["gate_w"], down_mult=p["down_mult"], img_w=p["img_w"])
    if mode in ("eager", "corr"):
        base = eager_demote(base, xs_by_row, family)
    if mode == "corr":
        base = value_hampel(base, xs_by_row, family)
    return base


def _cli():
    """Воспроизводит метрику Archive: per-curve медиана MBK acc (decode_img vs DP-база)."""
    sys.stdout.reconfigure(encoding="utf-8")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from extract_nlgx import extract, NULL
    import dataset as ds
    from dataset_build import find_image
    from decode_levels import build_family, gt_levels, decode
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 9999
    ARCHIVE = Path(r"F:\nds\projects\Archive")
    files = [n for wlg in sorted(ARCHIVE.glob("*/wlg")) for n in sorted(wlg.glob("*.nlgx"))
             if "_auto" not in n.stem]
    a_img = []; a_base = []; done = 0
    for n in files:
        if done >= limit:
            break
        try:
            m = extract(str(n))
        except Exception:
            continue
        img = find_image(n)
        if not img:
            continue
        gray = None
        for c in ds.real_curves(m):
            if not is_mbk(c["name"]):
                continue
            fam = build_family(m, c); gl = gt_levels(c)
            if len(fam) < 2 or (max(gl.values()) if gl else 0) < 1:
                continue
            ty = c["top_y"]
            xs = {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL}
            common = [y for y in sorted(xs) if y in gl]
            if len(common) < 100:
                continue
            if gray is None:
                gray = np.asarray(Image.open(img).convert("L"))
            dec = mbk_levels(xs, fam, gray)
            base = decode(xs, fam, lam=0.4)
            a_img.append(np.mean([min(dec[y], 3) == min(gl[y], 3) for y in common]))
            a_base.append(np.mean([min(base[y], 3) == min(gl[y], 3) for y in common]))
            done += 1
    if a_img:
        print(f"MBK кривых={len(a_img)}  DP-база медиана={np.median(a_base):.3f}  "
              f"decode_img_mbk медиана={np.median(a_img):.3f}  Δ={np.median(a_img)-np.median(a_base):+.3f}")


if __name__ == "__main__":
    _cli()
