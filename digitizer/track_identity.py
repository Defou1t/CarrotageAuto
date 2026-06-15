r"""
track_identity.py — B3 (Фаза B роадмапа §6.6.7): трекер С ИДЕНТИЧНОСТЬЮ. Связывает чернила в
линии ПОЦВЕТНО (B1) с непрерывностью + поведением (B2) так, чтобы линия НЕ прыгала на чужую
(баг QC «GZ5 прыгает на GZ4», «SP переходит на CALI»).

Опора: A3 exclude-маска (не-линейные зоны) + классификация по цвету (B1 `classify_ink`) →
в КАЖДОМ цветовом канале мало линий (1-2, часто в разных x-полосах) → ведём их сверху-вниз и
снизу-вверх по непрерывности (предсказание x по наклону, гейт прыжка + полоса-якорь против
перескока, коаст на разрывах). Жадное назначение (без scipy — работаем в py3.14/cv2).
Цвет фиксирует идентичность жёстко (разные каналы не смешиваются); якорь+наклон — внутри цвета.

python track_identity.py --nlgx <f.nlgx> [--save-overlay]
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from extract_nlgx import extract, NULL
from dataset_build import find_image
import dataset as ds
import detect_masks as dm
import extract_instances as ei
import behavior_priors as bp

COLORS = {"black": (30, 30, 30), "blue": (40, 90, 220),
          "red": (220, 40, 40), "green": (40, 170, 60)}


def _endpoints(s):
    """Концы штриха + наклон у нижнего конца (для траекторной сшивки)."""
    xr = s["xs_row"]; r0 = s["row0"]
    val = np.where(~np.isnan(xr))[0]
    ytop, ybot = r0 + int(val[0]), r0 + int(val[-1])
    xtop, xbot = float(xr[val[0]]), float(xr[val[-1]])
    tail = val[val >= val[-1] - 40]                    # ~хвост у низа
    if len(tail) >= 5:
        sl = float(np.polyfit(tail + r0, xr[tail], 1)[0])
    else:
        sl = 0.0
    return ytop, ybot, xtop, xbot, max(-12.0, min(12.0, sl))


def link_strokes(strokes, x_tol=22.0, gap_max=80, overlap_max=40):
    """Сшивает штрихи B1 в ЛИНИИ по траекторной непрерывности ВНУТРИ цвета. A→B (B ниже):
    тот же цвет, разрыв глубины в [−overlap_max, gap_max], и x верха B ≈ экстраполяция низа A
    по наклону (|Δ|<x_tol). Перевынос (скачок x≫x_tol) НЕ линкуется → остаётся отдельной линией
    (правило §6.6.7). Жадная цепочка. Возвращает линии {color, tr:{y:x}}."""
    for s in strokes:
        s["ep"] = _endpoints(s)
    lines = []
    by_color = {}
    for s in strokes:
        by_color.setdefault(s["color"], []).append(s)
    for color, ss in by_color.items():
        ss.sort(key=lambda s: s["ep"][0])              # по ytop
        used = [False] * len(ss)
        for i in range(len(ss)):
            if used[i]:
                continue
            chain = [i]; used[i] = True
            cur = i
            while True:
                _, ybot, _, xbot, sl = ss[cur]["ep"]
                best, bd = None, x_tol
                for j in range(len(ss)):
                    if used[j]:
                        continue
                    ytop_j, _, xtop_j, _, _ = ss[j]["ep"]
                    gap = ytop_j - ybot
                    if gap < -overlap_max or gap > gap_max:
                        continue
                    pred = xbot + sl * max(0, gap)
                    d = abs(xtop_j - pred)
                    if d < bd:
                        bd, best = d, j
                if best is None:
                    break
                used[best] = True; chain.append(best); cur = best
            tr = {}
            for k in chain:
                xr = ss[k]["xs_row"]; r0 = ss[k]["row0"]
                for r in range(len(xr)):
                    if not np.isnan(xr[r]):
                        tr[r0 + r] = float(xr[r])
            if len(tr) >= 30:
                lines.append({"color": color, "tr": tr, "n_strokes": len(chain)})
    return lines


def track_identity(rgb, m, min_h_frac=0.006):
    """B3: B1-штрихи (низкий min_h, чтобы не терять короткие) → траекторная сшивка в линии."""
    strokes, masks = ei.extract_instances(rgb, m, min_h_frac=min_h_frac)
    lines = link_strokes(strokes)
    lines.sort(key=lambda L: -len(L["tr"]))
    return lines, masks


# ---------- валидация: покрытие + own_px + свопы (в памяти vs GT) ----------

def _gt(m, H):
    out = []
    for c in ds.real_curves(m):
        ty = c["top_y"]
        d = {ty + i: x for i, x in enumerate(c["xs"]) if x != NULL and 0 <= ty + i < H}
        if len(d) >= 30:
            out.append((bp.mnem(c["name"]), c["name"].split()[0], d, c))
    return out


def validate(lines, m, rgb):
    H = rgb.shape[0]
    gts = _gt(m, H)
    print(f"\nЛИНИЙ извлечено: {len(lines)} "
          f"(цвета: {', '.join(sorted({l['color'] for l in lines}))})")
    # медианы x кривых → треки (зазор>250px), для свопов
    med = {nm2: np.median(list(d.values())) for _, nm2, d, _ in gts}
    order = sorted(med, key=med.get); trk = {}; t = 0
    for k, nm2 in enumerate(order):
        if k and med[nm2] - med[order[k - 1]] > 250:
            t += 1
        trk[nm2] = t
    print(f"\n{'curve':<6}{'цвет':<6}{'покрыт':>7}{'own_px':>8}{'swap%':>7}{'лин':>5}  партнёр")
    TOL = 12; tot_rows = tot_sw = 0
    for mn, nm2, gd, c in gts:
        gtc = ei._gt_color(rgb, gd)
        # АГРЕГАТ: все одноцветные линии, трассирующие ЭТУ кривую (медиана |Δx|<TOL на перекрытии).
        # Резистив-перевынос = несколько линий-уровней → собираем покрытие по всем.
        agg = {}; nlin = 0
        for L in lines:
            if L["color"] != gtc:
                continue
            ov = [(y, gd[y], L["tr"][y]) for y in gd if y in L["tr"]]
            if len(ov) < 20:
                continue
            if np.median([abs(a - b) for _, a, b in ov]) < TOL:
                nlin += 1
                for y, _, bx in ov:
                    if y not in agg or abs(bx - gd[y]) < abs(agg[y] - gd[y]):
                        agg[y] = bx
        if not agg:
            print(f"{mn:<6}{gtc:<6}{'—':>7}"); continue
        peers = [p for p in gts if p[1] != nm2 and trk[p[1]] == trk[nm2]]
        own = []; sw = 0; pc = {}
        for y, ax in agg.items():
            d_own = abs(gd[y] - ax); own.append(d_own)
            bd = d_own - 3.0; part = None
            for pmn, pnm, pd, _ in peers:
                if y in pd and abs(pd[y] - ax) < bd:
                    bd = abs(pd[y] - ax); part = pnm
            if part:
                sw += 1; pc[part] = pc.get(part, 0) + 1
        nr = len(own); cov = nr / len(gd) * 100
        sf = sw / nr * 100 if nr else 0
        pp = max(pc, key=pc.get) if pc else "-"
        print(f"{mn:<6}{gtc:<6}{cov:>6.0f}%{np.median(own):>8.1f}{sf:>6.0f}%{nlin:>5}  {pp}")
        tot_rows += nr; tot_sw += sw
    print(f"\nИТОГО свопнутых строк: {tot_sw}/{tot_rows} ({100*tot_sw/max(1,tot_rows):.1f}%)")


def save_overlay(rgb, lines, m, stem):
    H, W = rgb.shape[:2]
    ov = rgb.copy()
    for L in lines:
        col = COLORS.get(L["color"], (255, 0, 255))
        for y, x in L["tr"].items():
            x = int(x)
            if 0 <= y < H and 0 <= x < W:
                ov[y, max(0, x - 1):x + 2] = col
    ym = (m["depth_axis"]["top_y"] + m["depth_axis"]["bottom_y"]) // 2
    Image.fromarray(ov[ym - 300:ym + 300]).save(rf"F:\nds\output\b3_track_{stem}_mid.png")
    Image.fromarray(ov).resize((W // 6, H // 6), Image.LANCZOS).save(
        rf"F:\nds\output\b3_track_{stem}.png")
    print(f"\noverlay -> b3_track_{stem}[_mid].png")


def main():
    a = sys.argv[1:]
    nlgx = a[a.index("--nlgx") + 1]
    sys.stdout.reconfigure(encoding="utf-8")
    m = extract(nlgx)
    rgb = np.asarray(Image.open(find_image(Path(nlgx))).convert("RGB"))
    lines, masks = track_identity(rgb, m)
    g = masks["geom"]
    print(f"image {rgb.shape[1]}x{rgb.shape[0]}; трек x[{g['x_left']}..{g['x_right']}] "
          f"y[{g['top_y']}..{g['bottom_y']}]")
    validate(lines, m, rgb)
    if "--save-overlay" in a:
        save_overlay(rgb, lines, m, Path(nlgx).stem[:30])


if __name__ == "__main__":
    main()
