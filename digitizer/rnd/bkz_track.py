r"""
bkz_track.py — A2: классический мульти-strand трекер BKZ + паррование ×1↔×5 по геометрии шкал
+ выбор on-scale (→ decode_levels). Пивот Эдуарда 07.07 после того как A1 (UNet per-pixel
curve-id) оказался мисматчем: каждый градиент-зонд нарисован ДВУМЯ идентичными нитями (×1 база +
×5 перевынос), эксперт ведёт on-scale, избыточная нить не трассирована (bkz_diag / STEP-1C).

ПОСТАНОВКА:
  1. ink_mask: тёмная тушь (grid светлый) в x-полосе GZ-трека.
  2. row_runs → link в СТРАНДЫ (полилинии): ран(y)→ран(y+1) по перекрытию/близости.
  3. паррование: страйд A(x) и B(x') — одна кривая (×1/×5), если base(xA) ≈ 5x(xB) по глубине
     (геометрия SA base + "5X" SA; избыточная нить = scale-трансформ ведомой).
  4. on-scale/строку: base где значение в диапазоне, иначе ×5 (value-continuity, decode_levels).

СОСТОЯНИЕ (07.07, durable):
  M1 (link_strands): ★ детект нитей РАБОТАЕТ — страйды покрывают GT 100%, медиана 1.0px
     (GZ21 <3px 82%, GZ31 93%). Тушь-нити детектируемы. (флаг --m1)
  M2 (decode_curve, DP по ран×уровень + value/spatial continuity + wrap-gate) + reconstruct_pair:
     ГЕЙТ 4 D1-плейта (sequential, individual eff≤3): одна кривая 55-84%, вторая 39-54%.
     ★ ORACLE-ПОТОЛОК (best-of-both, любой из 2 путей →GT) = 48-96% (обычно 72-96%): МАТЕРИАЛ
     ЕСТЬ, узкое место = SWAP-FREE НАЗНАЧЕНИЕ (2 пути свопаются между кривыми). Для сравнения
     MK production был ~58% строгого eff≤3 — тут ЛУЧШАЯ кривая уже вровень БЕЗ обучения, но
     ВТОРАЯ отстаёт. Перепробовано (НЕ побили sequential, durable): joint 2-curve DP с жёстким
     взаимоисключением ранов (ХУЖЕ — форсирует 2 рана где 2-я кривая разрежена: 3160 oracle
     93→55); relabel min-jerk 2-track (нейтрально/вредит: 3474 61→52 — greedy свопит на level-
     скачках). Корень: per-row value-continuity DP смещён от высокоамплитудных нитей → 2-я
     трасса садится на низкозначную реплику; identity/level сцеплены.
  ★ СЛЕДУЮЩИЙ РЫЧАГ (не сделан): СТРУКТУРНОЕ ×1↔×5 паррование НИТЕЙ — не per-row, а на уровне
     СТРАНДОВ M1: смёрджить фрагменты → каждый странд = (кривая,шкала) → стежка ×1/×5 по
     геометрии (странд у правого края + странд на ×5-позиции с непрерывным value = одна кривая)
     → назначение 2 кривым БЕЗ per-row свопов. Обходит смещение per-row DP.

  python bkz_track.py <plate.nlgx> [--track auto|D1|D2] [--m1]   (авто-ищет парный img)
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # digitizer
from extract_nlgx import extract, NULL
from dataset_build import find_image
Image.MAX_IMAGE_PIXELS = None


def ink_mask(rgb, dark_thr=115):
    """Тёмная нейтральная тушь (чёрные кривые V<~115). Сетка (светло-синяя V~166) исключена;
    насыщенно-синие/фиолетовые грид-мажоры давятся по избытку B над R (grid холоднее туши)."""
    R, G, B = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    gray = (R + G + B) / 3.0
    blueish = (B - R > 22) & (B > 90)          # синий/фиолетовый грид
    return (gray < dark_thr) & ~blueish


def row_runs(rowbool, minw=1):
    """Раны True в строке → list[(a,b,center)] (a,b включительно)."""
    out = []
    n = len(rowbool); i = 0
    while i < n:
        if rowbool[i]:
            j = i
            while j + 1 < n and rowbool[j + 1]:
                j += 1
            if j - i + 1 >= minw:
                out.append((i, j, (i + j) / 2.0))
            i = j + 1
        else:
            i += 1
    return out


def link_strands(mask, x0, x1, y0, y1, max_gap_x=10, max_skip_y=6, min_len=40):
    """Связать раны в полилинии-странды. Жадно: активный странд продолжается ближайшим по x
    раном в пределах max_gap_x (стерпит короткий вертикальный разрыв max_skip_y строк);
    иначе новый странд. Возвращает list[dict row->x_center], отсортированных по длине."""
    active = []   # list of dict(row->x), last_row, last_x
    done = []
    for y in range(y0, y1):
        runs = row_runs(mask[y, x0:x1])
        runs = [(a + x0, b + x0, c + x0) for a, b, c in runs]
        used = [False] * len(runs)
        # продолжить активные
        for st in active:
            best = -1; bestd = 1e9
            for k, (a, b, c) in enumerate(runs):
                if used[k]:
                    continue
                # перекрытие предсказания с раном ИЛИ близость центра
                pred = st["x"]
                d = 0.0 if (a - max_gap_x <= pred <= b + max_gap_x) else abs(c - pred)
                if d < bestd:
                    bestd = d; best = k
            if best >= 0 and bestd <= max_gap_x:
                a, b, c = runs[best]; used[best] = True
                pred = st["x"]
                nx = c if (a <= pred <= b) else float(np.clip(pred + np.sign(c - pred) * min(abs(c - pred), 30), a, b))
                st["tr"][y] = nx; st["x"] = nx; st["miss"] = 0
            else:
                st["miss"] += 1
        # спавн новых из неиспользованных ранов
        for k, (a, b, c) in enumerate(runs):
            if not used[k]:
                active.append({"tr": {y: c}, "x": c, "miss": 0})
        # отставить умершие
        keep = []
        for st in active:
            if st["miss"] > max_skip_y:
                done.append(st)
            else:
                keep.append(st)
        active = keep
    done += active
    strands = [st["tr"] for st in done if len(st["tr"]) >= min_len]
    strands.sort(key=len, reverse=True)
    return strands


def curve_rows(c):
    ty = c["top_y"]
    return {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL}


def track_x_range(m, tokens):
    """x-полоса трека по кривым-целям (+поля)."""
    xs = []
    for c in m["curves"]:
        if c["name"].split()[0] in tokens:
            xs += [x for x in c["xs"] if x != NULL]
    if not xs:
        return None
    return max(0, min(xs) - 60), max(xs) + 60


PAIRS = {"D1": ("GZ21", "GZ31"), "D2": ("GZ41", "OGZ1")}

# --- M2: сборка on-scale трассы кривой через DP по (ран × уровень) + геометрия шкал ---
import math
import decode_levels as DL


def runs_by_row(mask, x0, x1, y0, y1, minw=1):
    """dict row -> list[run_center] тёмных ранов в полосе трека."""
    out = {}
    for y in range(y0, y1):
        rr = row_runs(mask[y, x0:x1], minw=minw)
        if rr:
            out[y] = [c + x0 for _, _, c in rr]
    return out


def decode_curve(rbr, family, lam=1.2, dxfrac=0.12, gate_w=4.0, sp_w=0.5, sp_scale=40.0,
                 exclude=None, excl_w=18):
    """DP: выбрать на каждой строке (x из ранов, level из семейства), минимизируя разрыв
    log|value| + ПРОСТРАНСТВЕННУЮ непрерывность (перо не телепортируется в пределах шкалы:
    same-level штраф sp_w·(Δx/sp_scale)² — держит трассу на СВОЕЙ кривой, гасит перескок на
    соседнюю ~130px) + штраф смены уровня + направленный wrap-гейт (уровень↑ ⟺ x резко падает).
    Обобщение decode_levels на НЕизвестный x (несколько ранов/строку). exclude: dict row->x
    другой кривой (раны в ±excl_w выбрасываются, анти-коллапс двух трасс). → dict row->(x, level)."""
    maps = [DL.scale_map(s) for s in family]
    K = len(family)
    width = abs(family[0]["x_right"] - family[0]["x_left"]) or 1
    dxthr = max(1.0, dxfrac * width)
    rows = sorted(rbr)

    def cands(y):
        xs = rbr[y]
        if exclude and y in exclude:
            ex = exclude[y]
            xs = [x for x in xs if abs(x - ex) > excl_w]
        return xs or rbr[y]                       # если всё исключили — верни исходные (не рвём)

    def logv(k, x):
        return math.log(abs(maps[k](x)) + 1.0)

    prev_states = []      # list of (x,k)
    prev_cost = []
    back = []             # per row: list of prev-index
    row_states = []
    for i, y in enumerate(rows):
        xs = cands(y)
        states = [(x, k) for x in xs for k in range(K)]
        row_states.append(states)
        cost = [0.0] * len(states)
        bk = [-1] * len(states)
        if i == 0:
            prev_states, prev_cost = states, cost; back.append(bk); continue
        dy = 1  # шаг строки
        for si, (x, k) in enumerate(states):
            vk = logv(k, x)
            best = 1e18; bi = -1
            for pi, (xp, kp) in enumerate(prev_states):
                dxp = x - xp
                if k == kp:
                    tc = sp_w * (dxp / sp_scale)**2          # same-level: пространств. непрерывность
                else:
                    nlev = k - kp
                    want = -1 if nlev > 0 else 1
                    aligned = (dxp * want) > 0
                    mag = min(1.0, abs(dxp) / dxthr)
                    tc = lam * abs(nlev) + gate_w * lam * (1.0 - (mag if aligned else 0.0))
                c = prev_cost[pi] + (vk - logv(kp, xp))**2 + tc
                if c < best:
                    best = c; bi = pi
            cost[si] = best; bk[si] = bi
        prev_states, prev_cost = states, cost; back.append(bk)
    if not row_states or not prev_cost:
        return {}
    si = int(np.argmin(prev_cost))
    out = {}
    for i in range(len(rows) - 1, -1, -1):
        x, k = row_states[i][si]
        out[rows[i]] = (x, k)
        si = back[i][si]
        if si < 0:
            break
    return out


def remove_path_runs(mask, path, x0, x1, dilate=3):
    """Копия маски без РАНОВ, на которых сидит path (dict row->(x,level)): в каждой строке
    гасим связный ран, содержащий x пути (±dilate). Так вторая трасса ищется на ДРУГОЙ туши
    (x-исключение по центру не гнало её на жирную нить справа)."""
    mm = mask.copy()
    H, W = mask.shape
    for y, xk in path.items():
        if not (0 <= y < H):
            continue
        x = int(round(xk[0]))
        a = x
        while a - 1 >= x0 and mm[y, a - 1]:
            a -= 1
        b = x
        while b + 1 < x1 and mm[y, b + 1]:
            b += 1
        mm[y, max(x0, a - dilate):min(x1, b + dilate + 1)] = False
    return mm


def joint_decode_pair(rbr, famA, famB, lam=1.2, dxfrac=0.12, gate_w=4.0,
                      sp_w=0.5, sp_scale=40.0, max_runs=6):
    """JOINT DP по ДВУМ кривым: state=(ранA,levelA,ранB,levelB), ранA≠ранB (взаимоисключение —
    анти-коллапс + форсирует вторую кривую на СВОЮ нить, а не низкозначную реплику). Стоимость
    = value+spatial continuity на КАЖДУЮ кривую (identity держится непрерывностью). → (trA,trB)
    dict row->(x,level). Векторно: переход раскладывается tA(a',a)+tB(b',b)."""
    mapsA = [DL.scale_map(s) for s in famA]; mapsB = [DL.scale_map(s) for s in famB]
    KA, KB = len(famA), len(famB)
    dxtA = max(1.0, dxfrac * abs(famA[0]["x_right"] - famA[0]["x_left"]))
    dxtB = max(1.0, dxfrac * abs(famB[0]["x_right"] - famB[0]["x_left"]))
    rows = sorted(rbr)

    def lv(maps, k, x):
        return math.log(abs(maps[k](x)) + 1.0)

    def tc(maps, dxt, xp, kp, x, k):
        d = x - xp
        if k == kp:
            t = sp_w * (d / sp_scale)**2
        else:
            nl = k - kp; want = -1 if nl > 0 else 1
            mag = min(1.0, abs(d) / dxt)
            t = lam * abs(nl) + gate_w * lam * (1.0 - (mag if (d * want) > 0 else 0.0))
        return (lv(maps, k, x) - lv(maps, kp, xp))**2 + t

    hist = []  # (runs, As, Bs, JA, JB, back)
    pcost = None; pinfo = None
    for r, y in enumerate(rows):
        runs = rbr[y][:max_runs]; n = len(runs)
        As = [(i, k) for i in range(n) for k in range(KA)]
        Bs = [(i, k) for i in range(n) for k in range(KB)]
        JA = []; JB = []
        for ai, (ia, _) in enumerate(As):
            for bi, (ib, _) in enumerate(Bs):
                if n >= 2 and ia == ib:
                    continue
                JA.append(ai); JB.append(bi)
        JA = np.array(JA, int); JB = np.array(JB, int); S = len(JA)
        if r == 0:
            pcost = np.zeros(S); hist.append((runs, As, Bs, JA, JB, np.full(S, -1, int)))
            pinfo = (runs, As, Bs, JA, JB); continue
        pruns, pAs, pBs, pJA, pJB = pinfo
        tA = np.array([[tc(mapsA, dxtA, pruns[pi], pk, runs[ia], ka) for (ia, ka) in As]
                       for (pi, pk) in pAs])                    # (nPA, nA)
        tB = np.array([[tc(mapsB, dxtB, pruns[pi], pk, runs[ib], kb) for (ib, kb) in Bs]
                       for (pi, pk) in pBs])                    # (nPB, nB)
        total = pcost[:, None] + tA[pJA[:, None], JA[None, :]] + tB[pJB[:, None], JB[None, :]]
        bp = np.argmin(total, axis=0)
        pcost = total[bp, np.arange(S)]
        hist.append((runs, As, Bs, JA, JB, bp)); pinfo = (runs, As, Bs, JA, JB)
    if not hist:
        return {}, {}
    s = int(np.argmin(pcost)); trA = {}; trB = {}
    for r in range(len(rows) - 1, -1, -1):
        runs, As, Bs, JA, JB, bp = hist[r]; y = rows[r]
        ia, ka = As[JA[s]]; ib, kb = Bs[JB[s]]
        trA[y] = (runs[ia], ka); trB[y] = (runs[ib], kb)
        s = bp[s]
        if s < 0:
            break
    return trA, trB


def relabel_2track(a, b, slmax=45.0):
    """Пересобрать 2 swap-free кривые из 2 путей a,b (каждый row->(x,level)). Пожадному 2-track
    min-jerk: на строке 2 точки {a,b} назначаются к C1/C2 по МИНИМУМУ прыжка от предыдущей
    позиции каждой (identity = пространств. непрерывность; кривые ~130px раздельны на D1).
    Смена уровня (x прыгает у ОДНОЙ кривой) терпима: другая не двигается → ориентация та же."""
    rows = sorted(set(a) | set(b))
    c1 = {}; c2 = {}
    c1x = c2x = None
    for y in rows:
        pts = []
        if y in a:
            pts.append(a[y])
        if y in b:
            pts.append(b[y])
        if not pts:
            continue
        if c1x is None:                                  # инициализация: левый→C1, правый→C2
            pts.sort(key=lambda p: p[0])
            c1[y] = pts[0]; c1x = pts[0][0]
            if len(pts) > 1:
                c2[y] = pts[1]; c2x = pts[1][0]
            continue
        if len(pts) == 1:
            p = pts[0]                                   # к ближайшей из C1/C2
            if c2x is None or abs(p[0] - c1x) <= abs(p[0] - c2x):
                c1[y] = p; c1x = p[0]
            else:
                c2[y] = p; c2x = p[0]
            continue
        p, q = pts
        if c2x is None:
            c2x = q[0]
        straight = abs(p[0] - c1x) + abs(q[0] - c2x)
        swap = abs(q[0] - c1x) + abs(p[0] - c2x)
        if swap < straight:
            p, q = q, p
        c1[y] = p; c1x = p[0]; c2[y] = q; c2x = q[0]
    return c1, c2


def reconstruct_pair(m, tokens, mask, x0, x1, y0, y1, joint=False, relabel=False):
    """Две on-scale трассы. По умолч. sequential (декод A → вычесть ран A → декод B) + relabel
    (min-jerk swap-free). joint=True → joint 2-curve DP. → dict token->{row:(x,level)}."""
    fams = {}
    for t in tokens:
        c = next((c for c in m["curves"] if c["name"].split()[0] == t), None)
        if c is None:
            continue
        fams[t] = DL.build_family(m, c)
    toks = [t for t in tokens if t in fams and len(fams[t]) >= 1]
    if len(toks) < 2:
        return {}
    rbr = runs_by_row(mask, x0, x1, y0, y1)
    if joint:
        a, b = joint_decode_pair(rbr, fams[toks[0]], fams[toks[1]])
    else:
        a = decode_curve(rbr, fams[toks[0]])
        rbrB = runs_by_row(remove_path_runs(mask, a, x0, x1), x0, x1, y0, y1)
        b = decode_curve(rbrB, fams[toks[1]])
    if relabel:
        a, b = relabel_2track(a, b)
    return {toks[0]: a, toks[1]: b}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    nlgx = Path(args[0])
    side = args[args.index("--track") + 1] if "--track" in args else "auto"
    if side == "auto":
        side = "D2" if "_D2" in nlgx.name else "D1"
    tokens = PAIRS[side]
    m = extract(str(nlgx))
    img = find_image(nlgx)
    rgb = np.asarray(Image.open(img).convert("RGB"))
    H, W = rgb.shape[:2]
    xr = track_x_range(m, tokens)
    if xr is None:
        print("нет целевых кривых", tokens); return
    x0, x1 = int(xr[0]), int(min(W, xr[1]))
    # y-диапазон по данным кривых
    gts = {t: curve_rows(next(c for c in m["curves"] if c["name"].split()[0] == t)) for t in tokens
           if any(c["name"].split()[0] == t for c in m["curves"])}
    ally = [y for g in gts.values() for y in g]
    y0, y1 = min(ally), max(ally) + 1
    mask = ink_mask(rgb)
    if "--m1" in args:
        strands = link_strands(mask, x0, x1, y0, y1)
        print(f"{nlgx.name} side={side} track x[{x0}..{x1}] y[{y0}..{y1}] strands={len(strands)} "
              f"(lens: {[len(s) for s in strands[:8]]})")
        for t, g in gts.items():
            errs = [min(abs(s[y] - gx) for s in strands if y in s)
                    for y, gx in g.items() if any(y in s for s in strands)]
            if errs:
                errs = np.array(errs)
                print(f"  {t}: GT n={len(g)} matched={len(errs)/len(g)*100:.0f}% nearest med={np.median(errs):.1f}px "
                      f"<3px={np.mean(errs<3)*100:.0f}% <8px={np.mean(errs<8)*100:.0f}%")
        return
    # M2: on-scale реконструкция пары + гейт vs GT (eff<=3px)
    rec = reconstruct_pair(m, tokens, mask, x0, x1, y0, y1)
    print(f"{nlgx.name} side={side} track x[{x0}..{x1}] y[{y0}..{y1}] recon curves={list(rec.keys())}")
    # назначить каждую реконструкцию к лучше совпадающей GT-кривой
    def match_err(tr, g):
        e = [abs(tr[y][0] - g[y]) for y in g if y in tr]
        return np.array(e) if e else np.array([1e9])
    used = set()
    for rt, tr in rec.items():
        best = None; beste = 1e9
        for gt, g in gts.items():
            if gt in used:
                continue
            me = np.median(match_err(tr, g))
            if me < beste:
                beste = me; best = gt
        if best is None:
            continue
        used.add(best); g = gts[best]
        e = match_err(tr, g); cov = np.mean([y in tr for y in g]) * 100
        print(f"  recon[{rt}]→GT[{best}]: n={len(g)} cov={cov:.0f}% |err|med={np.median(e):.1f}px "
              f"eff≤3={np.mean(e<=3)*100:.0f}% ≤8={np.mean(e<=8)*100:.0f}% >15={np.mean(e>15)*100:.0f}%")
    # best-of-both: покрывает ли ЛЮБОЙ из 2 путей GT-кривую (отделяет НАЗНАЧЕНИЕ от пропуска)
    for gt, g in gts.items():
        be = []
        for y, gx in g.items():
            cand = [abs(tr[y][0] - gx) for tr in rec.values() if y in tr]
            if cand:
                be.append(min(cand))
        if be:
            be = np.array(be)
            print(f"  best-of-both →GT[{gt}]: eff≤3={np.mean(be<=3)*100:.0f}% ≤8={np.mean(be<=8)*100:.0f}% "
                  f">15={np.mean(be>15)*100:.0f}%")


if __name__ == "__main__":
    main()
