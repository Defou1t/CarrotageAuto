r"""
decode_levels.py — ДЕКОДЕР ПЕРЕВЫНОСОВ (уровней масштаба) глобальным DP/Viterbi.
Перо «оборачивается»: уходя за правый край шкалы, значение продолжается в следующем
масштабе (×5/…), а x-трасса прыгает к левому краю. Истинное значение НЕПРЕРЫВНО, поэтому
ищем последовательность level(y), минимизирующую разрывы log-значения + штраф переходов.

DP: dp[y][k] = min_k' dp[y-1][k'] + (logV_k(x_y) - logV_k'(x_{y-1}))^2 + lam*[k≠k'].
Семейство масштабов = base (имя кривой = "MNEM DAi SAj") + цепочка next (тег 35273).

Валидация (на ЭКСПЕРТНОЙ трассе — изолирует качество декодера): декодированные уровни vs
GT (тег 35498) + реконструкция value vs эталонный LAS (corr), сравнение с base-only и oracle-GT.

python decode_levels.py [--limit N] [--lam 4.0]
"""
import sys, math, re
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_las

ARCHIVE = Path(r"F:\nds\projects\Archive")

# Резистивные мнемоники — для них применима ×5/лог-модель перевыносов.
# SP/TP/DT/GK/NGK/NNK/CALI/DS/ZAT и пр. — другая физика (знак/линейность) → декодер не применяем.
RESIST = {"GZ", "OGZ", "BK", "MBK", "BKZ", "BMK", "IK", "IKA", "IKR",
          "PZ", "MGZ", "MPZ", "BKP", "LL", "LLD", "LLS"}


def mnem(name):
    return re.sub(r"\d+$", "", name.split()[0])


def is_resistive(name):
    return mnem(name) in RESIST


def scale_map(sa):
    xL, xR = sa["x_left"], sa["x_right"]
    vL, vR = sa["v_left"], sa["v_right"]
    span = (xR - xL) or 1
    return lambda x: vL + (x - xL) * (vR - vL) / span


def demote_representable(lv, xs_by_row, family, margin=1.02):
    """ПОСТ-ПРОХОД (Эдуард 12.07: «приоритет 1х, 5х только когда значение за границей»): сегмент
    уровня k, чьи значения ЦЕЛИКОМ влезают в уровень k-1 (|v_k(x)| ≤ v_{k-1}^max), понижается —
    перо не упиралось в рельс, значит не оборачивалось. Убирает «застревание» DP на 5х в зоне
    перекрытия (STK PZ: value~20 ложно на 5х), НЕ трогая реальные 5х (там v>1х-макс, замер Archive).
    Итеративно сверху вниз до фикспоинта."""
    maps = [scale_map(s) for s in family]
    K = len(family)
    if K < 2:
        return lv
    vmax = [max(abs(s["v_left"]), abs(s["v_right"])) for s in family]   # потолок значения на уровне
    lv = dict(lv)
    changed = True
    while changed:
        changed = False
        # сегменты по уровню
        rows = sorted(lv)
        s = rows[0]; cur = lv[rows[0]]; prev = rows[0]
        segs = []
        for y in rows[1:]:
            if lv[y] != cur:
                segs.append((s, prev, cur)); s = y; cur = lv[y]
            prev = y
        segs.append((s, prev, cur))
        for a, b, k in segs:
            if k <= 0:
                continue
            vals = [abs(maps[k](xs_by_row[y])) for y in range(a, b + 1) if y in xs_by_row]
            if vals and max(vals) <= margin * vmax[k - 1]:   # весь сегмент влезает в уровень k-1
                for y in range(a, b + 1):
                    if y in lv:
                        lv[y] = k - 1
                changed = True
    return lv


def build_family(m, curve):
    """Цепочка масштабов кривой: base (имя-суффикс) → next → … (level 0..K)."""
    key = " ".join(curve["name"].split()[1:])
    by_idx = {s["idx"]: s for s in m["scale_axes"] if s.get("idx") is not None}
    by_name = {s["name"]: s for s in m["scale_axes"]}
    base = by_name.get(key)
    if base is None:
        return []
    chain = [base]; seen = {base.get("idx")}
    cur = base.get("next")
    while cur is not None and cur != NULL and cur in by_idx and cur not in seen:
        seen.add(cur); chain.append(by_idx[cur]); cur = by_idx[cur].get("next")
    return chain


def logv(maps, k, x):
    return math.log(abs(maps[k](x)) + 1.0)


def decode(xs_by_row, family, lam=0.7, dxfrac=0.12, gate_w=4.0, level_bias=0.0, rail_gate=0.0,
           down_mult=1.0):
    """xs_by_row: dict row->x; family: список scale-словарей. -> dict row->level.
    НАПРАВЛЕННЫЙ wrap-гейт: шкалы перекрываются, поэтому уровень меняется НЕ у потолка,
    а на ОБОРОТЕ пера, видимом как РЕЗКИЙ СДВИГ x в определённую сторону:
      level вверх (k>kp) ⟺ x резко ПАДАЕТ (перо ушло вправо → перенесли влево);
      level вниз  (k<kp) ⟺ x резко РАСТЁТ.
    Переход дёшев лишь если |Δx| велик И знак согласован с направлением уровня; иначе дорог.
    dxthr = dxfrac·ширина_шкалы; цена перехода = lam·|Δlevel| + gate_w·lam·(1-consistency).

    level_bias — ПРИОР К НИЗКОМУ УРОВНЮ (Эдуард 12.07: «приоритет чаще 1х»): штраф bias·k на
    строку. В зоне перекрытия значений (низкое значение представимо и на 1х, и на 5х) DP без
    приора «плавает» на 5х (STK PZ: 73% ложно-5х при value~20); bias удерживает на 1х, пока
    непрерывность значения строго не потребует выше."""
    maps = [scale_map(s) for s in family]
    K = len(family)
    rows = sorted(xs_by_row)
    if K <= 1 or len(rows) < 2:
        return {y: 0 for y in rows}
    width = abs(family[0]["x_right"] - family[0]["x_left"])
    dxthr = max(1.0, dxfrac * width)
    xl = min(family[0]["x_left"], family[0]["x_right"])
    rail_x = xl + rail_gate * width          # «упор»: переход ВВЕРХ дёшев только если xp ≥ этого
    dp = [level_bias * k for k in range(K)]
    back = []
    for i in range(1, len(rows)):
        x = xs_by_row[rows[i]]; xp = xs_by_row[rows[i-1]]; dx = x - xp
        ndp = [1e18] * K; bk = [0] * K
        lvp = [logv(maps, kp, xp) for kp in range(K)]
        for k in range(K):
            vk = logv(maps, k, x)
            best = 1e18; bki = 0
            for kp in range(K):
                if k == kp:
                    tc = 0.0
                else:
                    nlev = k - kp
                    want = -1 if nlev > 0 else 1          # вверх → ждём Δx<0
                    aligned = (dx * want) > 0
                    mag = min(1.0, abs(dx) / dxthr)
                    consistency = mag if aligned else 0.0
                    tc = lam * abs(nlev) + gate_w * lam * (1.0 - consistency)
                    # РЕЛЬС-ГЕЙТ (Эдуард 12.07): оборот ВВЕРХ физически только когда перо упёрлось
                    # в правый край (xp у рельса). Переход вверх вдали от рельса = ложный (шум
                    # трассы) → дорог. Не трогает переход вниз (возврат плавный).
                    if nlev > 0 and rail_gate > 0 and xp < rail_x:
                        tc += gate_w * lam * nlev
                    # АСИММЕТРИЯ ВНИЗ (калибровка GT Эдуарда 12.07): смена масштаба на 5х часто по
                    # СОГЛАШЕНИЮ записи (глубже пишут 5х), а не по упору — вернуться на 1х перо не
                    # должно, пока не настоящий оборот вправо. DP слишком легко падал вниз (BK: ушёл
                    # с 5х на 2107, а GT держит до 2795). down_mult удорожает переход ВНИЗ.
                    if nlev < 0:
                        tc *= down_mult
                c = dp[kp] + (vk - lvp[kp])**2 + tc + level_bias * k
                if c < best:
                    best = c; bki = kp
            ndp[k] = best; bk[k] = bki
        dp = ndp; back.append(bk)
    k = int(np.argmin(dp))
    lv = {rows[-1]: k}
    for i in range(len(rows)-1, 0, -1):
        k = back[i-1][k]; lv[rows[i-1]] = k
    return lv


def gt_levels(curve):
    """level по строкам из сегментов (тег 35498)."""
    out = {}
    for s, e, l in curve["segments"]:
        for y in range(s, e+1):
            out[y] = l
    return out


def reconstruct(xs_by_row, levels, family):
    maps = [scale_map(s) for s in family]
    K = len(family)
    return {y: maps[min(levels.get(y, 0), K-1)](x) for y, x in xs_by_row.items()}


def corr_log(rec_by_row, m, las_depths, las_vals):
    """corr реконструкции с LAS в log-пространстве, сопоставление по глубине."""
    da = m["depth_axis"]; ty, by = da["top_y"], da["bottom_y"]
    td, bd = da["top_depth"], da["bottom_depth"]
    def dof(y): return td + (y - ty) * (bd - td) / (by - ty)
    R = []; L = []
    for y, v in rec_by_row.items():
        d = dof(y); li = int(np.clip(np.searchsorted(las_depths, d), 0, len(las_depths)-1))
        lv = las_vals[li]
        if np.isfinite(lv) and lv > 0 and v > 0:
            R.append(math.log(v)); L.append(math.log(lv))
    if len(R) < 30:
        return None
    return float(np.corrcoef(R, L)[0, 1])


def main():
    limit = int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else 25
    lam = float(sys.argv[sys.argv.index("--lam")+1]) if "--lam" in sys.argv else 0.7
    allcurves = "--all" in sys.argv  # не фильтровать по резистивности (диагностика)
    sys.stdout.reconfigure(encoding="utf-8")
    files = []
    for wlg in sorted(ARCHIVE.glob("*/wlg")):
        for n in sorted(wlg.glob("*.nlgx")):
            if "_auto" not in n.stem:
                files.append(n)
    done = 0; nonres = 0
    print(f"{'curve':<16}{'fam':>4}{'GTmax':>6}{'lvl_acc':>8}{'corr_dec':>9}{'corr_base':>10}{'corr_orac':>10}")
    accs = []; cds = []; cbs = []
    for n in files:
        if done >= limit:
            break
        try:
            m = extract(str(n))
        except Exception:
            continue
        las = find_las(n)
        if not las:
            continue
        cols, arr = ds.load_las(las)
        if arr.size == 0:
            continue
        matches = ds.match_las(m, cols, arr)
        depths = arr[:, 0]
        for c in ds.real_curves(m):
            fam = build_family(m, c)
            gl = gt_levels(c)
            gmax = max(gl.values()) if gl else 0
            if len(fam) < 2 or gmax == 0:
                continue  # только кривые С перевыносами и цепочкой
            if not allcurves and not is_resistive(c["name"]):
                nonres += 1; continue  # лог/×5-модель только для резистивных
            mm = matches.get(c["name"], {}); ci = mm.get("col_idx")
            if ci is None:
                continue
            ty = c["top_y"]
            xs = {ty+i: x for i, x in enumerate(c["xs"]) if x != NULL}
            if len(xs) < 100:
                continue
            dec = decode(xs, fam, lam)
            common = [y for y in xs if y in gl]
            acc = np.mean([dec[y] == gl[y] for y in common]) if common else float('nan')
            rec_dec = reconstruct(xs, dec, fam)
            rec_base = reconstruct(xs, {y: 0 for y in xs}, fam)
            rec_orac = reconstruct(xs, gl, fam)
            cd = corr_log(rec_dec, m, depths, arr[:, ci])
            cb = corr_log(rec_base, m, depths, arr[:, ci])
            co = corr_log(rec_orac, m, depths, arr[:, ci])
            print(f"{c['name'].split()[0][:15]:<16}{len(fam):>4}{gmax:>6}{acc:>8.2f}"
                  f"{(cd if cd is not None else float('nan')):>9.3f}"
                  f"{(cb if cb is not None else float('nan')):>10.3f}"
                  f"{(co if co is not None else float('nan')):>10.3f}")
            accs.append(acc)
            if cd is not None: cds.append(cd)
            if cb is not None: cbs.append(cb)
            done += 1
            if done >= limit:
                break
    if accs:
        cds = np.array(cds); cbs = np.array(cbs)
        gain = int(np.sum(cds >= cbs + 0.05)); worse = int(np.sum(cds < cbs - 0.05))
        strong = int(np.sum(cds >= 0.85))
        print(f"\nlam={lam}  кривых={len(accs)}  level-acc мед={np.nanmedian(accs):.2f} "
              f"среднее={np.nanmean(accs):.2f}")
        print(f"  corr_dec мед={np.median(cds):.3f} среднее={np.mean(cds):.3f}  corr_base мед={np.median(cbs):.3f}")
        print(f"  corr_dec≥0.85: {strong}/{len(cds)} ; лучше base(+0.05): {gain} ; ХУЖЕ base(-0.05): {worse}")
        print(f"  пропущено не-резистивных кривых: {nonres}")


if __name__ == "__main__":
    main()
