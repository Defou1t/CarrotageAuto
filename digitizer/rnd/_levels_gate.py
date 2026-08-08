r"""_levels_gate.py — ГЕЙТ ВТОРОЙ ПОЛОВИНЫ ЗАДАЧИ: ПЕРЕХОДЫ МАСШТАБОВ (×2/×5, backup left-right).

Постановка заказчика (22.07): имена и маппинг слот↔кривая НЕ нужны — на входе всегда рамка,
горизонтали, созданные кривые с верными именами и ЗАГОТОВЛЕННЫЕ цепочки scale axis. Нужно
(1) различать линии между собой и (2) ПОНИМАТЬ ПЕРЕХОДЫ МАСШТАБОВ. Вторая половина до сих пор
не мерилась отдельно — только косвенно, через value-corr (§1: 0.47 против 0.89 у эксперта).

Здесь она отделена от трассировки НАЧИСТО: уровни декодируются по ЭКСПЕРТНОЙ трассе (геометрия
идеальна), значит всё, что видно, — качество самого декодера переходов, а не следствие латча.

  ЧТО МЕРИТСЯ (покривой, только кривые с цепочкой ≥2 шкал):
    acc      — доля строк с верным уровнем (эталон: сегменты кривой, тег 35498);
    база     — та же доля, если ВСЕГДА брать level 0 (сколько даёт «ничего не делать»);
    переходы — сколько смен уровня у эксперта и сколько нашёл декодер (пропуск/ложные);
    ошибка на ГРАНИЦЕ — медиана |строка нашего перехода − строка эксперта|, строк.

⚠ acc сама по себе обманчива: если кривая почти вся на level 0, «всегда 0» даёт 90%+. Поэтому
главные числа — ПЕРЕХОДЫ, а не acc.

  python _levels_gate.py [--sheets 40] [--lam 0.7]
"""
import sys, argparse, io, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
import decode_levels as DL
from _multi_replica_probe import dense
from _decoder_data import train_sheets
from _levels_decode2 import decode2

ap = argparse.ArgumentParser()
ap.add_argument("--sheets", type=int, default=40)
ap.add_argument("--lam", type=float, default=0.7)
ap.add_argument("--dxfrac", type=float, default=0.12)
ap.add_argument("--gate-w", type=float, default=4.0)
ap.add_argument("--sweep", action="store_true",
                help="перебор (lam, dxfrac, gate_w) с ЧЕСТНЫМ разделением листов: подбор на "
                     "половине скважин, проверка на другой (иначе это подгонка)")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--flat", action="store_true", help="цена перехода БЕЗ множителя |Δlevel| (§6.35)")
ap.add_argument("--linear", action="store_true", help="непрерывность значения ЛИНЕЙНАЯ, не log (§6.35)")
ap.add_argument("--ablate", action="store_true", help="абляция 4 вариантов (flat × linear) на проверочных")
ap.add_argument("--kind", default=None, choices=["аддитивная","мультипликат."], help="абляция только на этом классе цепочек")
ap.add_argument("--values", action="store_true",
                help="мерить ЗНАЧЕНИЯ против LAS, а не уровни. Уточнение заказчика (22.07): backup "
                     "right — АДДИТИВНЫЙ сдвиг диапазона (0-20 → 20-40 → 40-60), поэтому пропуск "
                     "перехода даёт ошибку в ШИРИНУ ШКАЛЫ, а не в 5×; линейная метрика обязательна")
a = ap.parse_args()


def transitions(levels_by_row):
    """Строки, где уровень МЕНЯЕТСЯ (переход пера на другую шкалу)."""
    ys = sorted(levels_by_row)
    return [ys[i] for i in range(1, len(ys)) if levels_by_row[ys[i]] != levels_by_row[ys[i - 1]]]


def value_probe(sheets, lam, dxfrac, gate_w):
    """ЗНАЧЕНИЯ против LAS при ИДЕАЛЬНОЙ геометрии (экспертная трасса).

    Три варианта уровней сравниваются на одних и тех же строках:
      база0   — всегда level 0 (что получится, если переходы не понимать вовсе);
      декодер — decode() с заданными параметрами;
      оракул  — уровни эксперта (потолок: сколько даст ИДЕАЛЬНОЕ понимание переходов).
    Метрики ЛИНЕЙНЫЕ (corr и медиана |отн. ошибки|), т.к. backup — аддитивный сдвиг."""
    from dataset_build import find_las
    out = []
    SKIP, done = {}, 0          # §6.106: сверка списка — обязательная печать, а не отладка
    for n in sheets:
        las = find_las(n)
        if not las:
            SKIP["нет LAS"] = SKIP.get("нет LAS", 0) + 1
            continue
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m = extract(str(n))
                cols, arr = ds.load_las(str(las))
                matches = ds.match_las(m, cols, arr)
        except Exception:
            SKIP["падение чтения"] = SKIP.get("падение чтения", 0) + 1
            continue
        da = m.get("depth_axis") or {}
        if not da.get("bottom_y"):
            SKIP["нет оси глубин"] = SKIP.get("нет оси глубин", 0) + 1
            continue
        ty, by = da["top_y"], da["bottom_y"]; td, bd = da["top_depth"], da["bottom_depth"]
        if by == ty:
            SKIP["вырожденная ось"] = SKIP.get("вырожденная ось", 0) + 1
            continue
        done += 1
        depths = arr[:, 0]
        for c in ds.real_curves(m):
            fam = DL.build_family(m, c); gl = DL.gt_levels(c)
            ci = matches.get(c["name"], {}).get("col_idx")
            if len(fam) < 2 or not gl or ci is None:
                continue
            d = dense(c)
            exp = {y: d[y] for y in sorted(d) if y in gl}
            if len(exp) < 200:
                continue
            dec = DL.decode(exp, fam, lam, dxfrac, gate_w)
            maps = [DL.scale_map(s) for s in fam]; K = len(fam)
            lasv = arr[:, ci]
            got = {"база0": ([], []), "декодер": ([], []), "оракул": ([], [])}
            for y, x in exp.items():
                dep = td + (y - ty) * (bd - td) / (by - ty)
                li = int(np.clip(np.searchsorted(depths, dep), 0, len(depths) - 1))
                lv = lasv[li]
                if not np.isfinite(lv):
                    continue
                for tag, lev in (("база0", 0), ("декодер", dec.get(y, 0)), ("оракул", gl.get(y, 0))):
                    v = maps[min(int(lev), K - 1)](x)
                    got[tag][0].append(v); got[tag][1].append(lv)
            if len(got["база0"][0]) < 50:
                continue
            row = {"well": n.parent.parent.name, "short": c["name"].split()[0], "K": K}
            for tag, (vv, ll) in got.items():
                vv = np.array(vv); ll = np.array(ll)
                row[tag + "_corr"] = float(np.corrcoef(vv, ll)[0, 1]) if vv.std() > 0 else float("nan")
                den = np.where(np.abs(ll) > 1e-9, np.abs(ll), np.nan)
                row[tag + "_rel"] = float(np.nanmedian(np.abs(vv - ll) / den))
            out.append(row)
    # ⚠⚠ СВЕРКА СПИСКА (§6.106): «кривых с LAS и цепочкой ≥2 шкал: N» само по себе не говорит,
    # СКОЛЬКО листов до этого счёта вообще дошло, — а отсев здесь четырёхступенчатый.
    _sk = sum(SKIP.values())
    print(f"  СВЕРКА СПИСКА: обработано {done} + пропущено {_sk} = {done + _sk} против длины "
          f"списка {len(sheets)}   {'★ СОШЛОСЬ' if done + _sk == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
    for k_, v_ in sorted(SKIP.items(), key=lambda q: -q[1]):
        print(f"    пропущено «{k_}»: {v_}")
    return out


if a.values:
    sheets = train_sheets(a.sheets)
    rows = value_probe(sheets, a.lam, a.dxfrac, a.gate_w)
    print(f"кривых с LAS и цепочкой ≥2 шкал: {len(rows)}\n")
    print(f"{'скважина':<12}{'кривая':<8}{'K':>3}{'corr 0':>8}{'corr дек':>9}{'corr орк':>9}"
          f"{'отн0':>8}{'отн дек':>9}{'отн орк':>9}")
    for r in rows:
        print(f"{r['well']:<12}{r['short']:<8}{r['K']:>3}{r['база0_corr']:>8.2f}"
              f"{r['декодер_corr']:>9.2f}{r['оракул_corr']:>9.2f}"
              f"{r['база0_rel']:>8.2f}{r['декодер_rel']:>9.2f}{r['оракул_rel']:>9.2f}")
    if rows:
        for tag in ("база0", "декодер", "оракул"):
            cc = np.array([r[tag + "_corr"] for r in rows]); rr = np.array([r[tag + "_rel"] for r in rows])
            print(f"{tag:<9} corr med {np.nanmedian(cc):.3f}   отн.ошибка med {np.nanmedian(rr):.2f}")
        print("\nЧитать: «оракул» — потолок при ИДЕАЛЬНОМ понимании переходов. Если он немногим")
        print("лучше «база0» — переходы на этих кривых не решают, и вкладываться надо не в них.")
    sys.exit(0)


def collect(sheets):
    """Дорогая часть (чтение nlgx + плотная GT) — ОДИН раз; перебор параметров потом дешёвый."""
    out = []
    _skipped = 0
    for n in sheets:
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                m = extract(str(n))
        except Exception:
            _skipped += 1               # §6.106: молчаливый пропуск листа обязан быть виден
            continue
        well = n.parent.parent.name
        for c in ds.real_curves(m):
            # ⚠ ФИЛЬТР is_resistive СНЯТ (22.07): он отбирал ТОЛЬКО резистивные ×5-цепочки, т.е.
            # заведомо мультипликативные. Абляция «линейной непрерывности» на таком наборе была
            # предрешена — аддитивные backup-цепочки (TMSS/MNDS, K=26-29) в неё не попадали вовсе.
            fam = DL.build_family(m, c)
            gl = DL.gt_levels(c)
            if len(fam) < 2 or not gl:
                continue
            d = dense(c)
            if len(d) < 200:
                continue
            exp = {y: d[y] for y in sorted(d) if y in gl}
            if len(exp) >= 200:
                out.append({"well": well, "short": c["name"].split()[0], "fam": fam,
                            "gl": gl, "exp": exp})
    print(f"  СВЕРКА СПИСКА: обработано {len(sheets) - _skipped} + пропущено {_skipped} = "
          f"{len(sheets)} против длины списка {len(sheets)}   ★ СОШЛОСЬ")
    return out


def score(items, lam, dxfrac, gate_w, flat=False, linear=False, auto=False):
    """ПОЛНОТА и ТОЧНОСТЬ переходов (±50 строк).

    ⚠ Точность обязательна: удешевление перехода (маленький lam) поднимает полноту ТЕМ, ЧТО
    декодер сыплет переходами. Без точности перебор выродится в спам, и §6.20.3-урок повторится
    на другом объекте."""
    hit = tot = ok_d = tot_d = 0
    for it in items:
        dec = (decode2(it["exp"], it["fam"], lam, dxfrac, gate_w, flat=flat, linear=linear, auto=auto)
               if (flat or linear or auto) else DL.decode(it["exp"], it["fam"], lam, dxfrac, gate_w))
        ys = [y for y in it["exp"] if y in dec]
        tg = transitions({y: it["gl"][y] for y in ys})
        td = transitions({y: dec[y] for y in ys})
        for g in tg:
            if td and min(abs(t - g) for t in td) <= 50:
                hit += 1
        for t in td:
            if tg and min(abs(t - g) for g in tg) <= 50:
                ok_d += 1
        tot += len(tg); tot_d += len(td)
    rec = hit / max(1, tot); prec = ok_d / max(1, tot_d)
    return rec, prec


def chain_kind(fam):
    """АДДИТИВНАЯ цепочка (backup right: 0-20 → 20-40, размах ПОСТОЯНЕН) или МУЛЬТИПЛИКАТИВНАЯ
    (перевынос ×5: размах растёт кратно). Классификация по медианному отношению размахов соседних
    звеньев — от неё зависит, в какой метрике мерить непрерывность значения."""
    sp = [abs(s["v_right"] - s["v_left"]) or 1e-9 for s in fam]
    r = float(np.median([sp[i + 1] / sp[i] for i in range(len(sp) - 1)]))
    return ("аддитивная" if r < 1.5 else "мультипликат."), r


if a.ablate:
    items = collect(train_sheets(a.sheets))
    for it in items:
        it["kind"], it["ratio"] = chain_kind(it["fam"])
    from collections import Counter
    cnt = Counter(it["kind"] for it in items)
    print(f"типы цепочек: {dict(cnt)}")
    for kind in ("аддитивная", "мультипликат."):
        sub = [it for it in items if it["kind"] == kind]
        if sub:
            print(f"  {kind}: кривых {len(sub)}, звеньев med "
                  f"{np.median([len(it['fam']) for it in sub]):.0f}, "
                  f"скважин {len({it['well'] for it in sub})}")
    if a.kind:
        items = [it for it in items if it["kind"] == a.kind]
        print(f"\n>>> ТОЛЬКО {a.kind}: {len(items)} кривых")
    wells = sorted({it["well"] for it in items})
    TR = [it for it in items if it["well"] in set(wells[::2])]
    VA = [it for it in items if it["well"] in set(wells[1::2])]
    print(f"кривых {len(items)}: подбор {len(TR)} / проверка {len(VA)}\n")
    print(f"{'вариант':<22}{'подбор пол/точн':>18}{'ПРОВЕРКА пол/точн':>20}  параметры")
    GRID = [(l, d, g) for l in (0.02, 0.05, 0.15, 0.7) for d in (0.01, 0.03, 0.12)
            for g in (2.0, 4.0, 8.0)]
    for flat, linear, auto, tag in ((False, False, False, "как есть"),
                                    (True, False, False, "+ плоская цена"),
                                    (False, True, False, "+ линейное значение"),
                                    (True, True, False, "обе правки"),
                                    (False, False, True, "★ ПО ТИПУ ЦЕПОЧКИ")):
        best = None
        for l, d, g in GRID:                      # параметры подбираются ДЛЯ КАЖДОГО варианта
            r, p = score(TR, l, d, g, flat, linear, auto)
            f = 0.0 if r + p == 0 else 2 * r * p / (r + p)
            if best is None or f > best[0]:
                best = (f, l, d, g, r, p)
        _, l, d, g, r, p = best
        vr, vp = score(VA, l, d, g, flat, linear, auto)
        print(f"{tag:<22}{100*r:>9.0f}%/{100*p:<7.0f}%{100*vr:>11.0f}%/{100*vp:<8.0f}%  "
              f"lam={l} dxfrac={d} gate_w={g}")
    print("\nЧитать: правка засчитывается ТОЛЬКО если растут проверочные при своих лучших")
    print("параметрах — иначе выигрыш куплен подбором, а не моделью домена.")
    sys.exit(0)

if a.sweep:
    items = collect(train_sheets(a.sheets))
    wells = sorted({it["well"] for it in items})
    tr_w = set(wells[::2]); va_w = set(wells[1::2])       # разделение ПО СКВАЖИНАМ
    TR = [it for it in items if it["well"] in tr_w]
    VA = [it for it in items if it["well"] in va_w]
    print(f"кривых {len(items)}: подбор {len(TR)} (скважин {len(tr_w)}) / "
          f"проверка {len(VA)} (скважин {len(va_w)})")
    def f1(r, p):
        return 0.0 if r + p == 0 else 2 * r * p / (r + p)

    br, bp = score(TR, 0.7, 0.12, 4.0)
    vr, vp = score(VA, 0.7, 0.12, 4.0)
    print(f"текущие (0.7, 0.12, 4.0): подбор полнота {100*br:.0f}% точность {100*bp:.0f}%   "
          f"проверка полнота {100*vr:.0f}% точность {100*vp:.0f}%")
    best = (f1(br, bp), 0.7, 0.12, 4.0, br, bp)
    grid = []
    for lam in (0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 1.0):
        for dxf in (0.005, 0.01, 0.02, 0.03, 0.06, 0.12, 0.2):
            for gw in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
                r, p = score(TR, lam, dxf, gw)
                grid.append((f1(r, p), lam, dxf, gw, r, p))
                if f1(r, p) > best[0]:
                    best = (f1(r, p), lam, dxf, gw, r, p)
    _, lam, dxf, gw, r, p = best
    vr2, vp2 = score(VA, lam, dxf, gw)
    print(f"ЛУЧШИЕ по F1 на подборе: lam={lam} dxfrac={dxf} gate_w={gw} → "
          f"полнота {100*r:.0f}% точность {100*p:.0f}%")
    print(f"★ НА ПРОВЕРОЧНЫХ СКВАЖИНАХ: полнота {100*vr2:.0f}% (было {100*vr:.0f}%)   "
          f"точность {100*vp2:.0f}% (было {100*vp:.0f}%)")
    grid.sort(reverse=True)
    print("\nтоп-8 по F1 (подбор) — проверка ПЛАТО, а не одиночного пика:")
    for f, l_, d_, g_, r_, p_ in grid[:8]:
        print(f"   lam={l_:<5} dxfrac={d_:<6} gate_w={g_:<5} полнота {100*r_:>3.0f}% точность {100*p_:>3.0f}%")
    print("\nЧитать: если на проверочных прирост НЕ повторяется — подгонка, параметры не менять.")
    sys.exit(0)

print(f"{'скважина':<12}{'кривая':<8}{'шкал':>5}{'строк':>7}{'acc':>7}{'база0':>7}"
      f"{'перех.GT':>9}{'нашли':>7}{'|Δстрок|':>9}  вердикт")
rows = []
_sheets = train_sheets(a.sheets)
_skipped_main = 0
for n in _sheets:
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m = extract(str(n))
    except Exception as e:
        _skipped_main += 1          # §6.106: молчаливый пропуск листа обязан быть виден
        continue
    well = n.parent.parent.name
    for c in ds.real_curves(m):
        if not DL.is_resistive(c["name"]):
            continue
        fam = DL.build_family(m, c)
        gl = DL.gt_levels(c)
        if len(fam) < 2 or not gl:
            continue
        d = dense(c)
        if len(d) < 200:
            continue
        exp = {y: d[y] for y in sorted(d) if y in gl}
        if len(exp) < 200:
            continue
        dec = DL.decode(exp, fam, a.lam)
        ys = [y for y in exp if y in dec]
        acc = float(np.mean([dec[y] == gl[y] for y in ys]))
        base0 = float(np.mean([gl[y] == 0 for y in ys]))
        tg = transitions({y: gl[y] for y in ys})
        td = transitions({y: dec[y] for y in ys})
        if tg:
            dmin = [min(abs(t - g) for t in td) if td else 10 ** 9 for g in tg]
            dm = float(np.median(dmin))
        else:
            dm = float("nan")
        v = ("нет переходов" if not tg else
             ("★ переходы найдены" if td and dm <= 50 else
              ("промах по месту" if td else "✗ ПРОПУЩЕНЫ ВСЕ")))
        rows.append((acc, base0, len(tg), len(td), dm))
        print(f"{well:<12}{c['name'].split()[0]:<8}{len(fam):>5}{len(ys):>7}{acc:>7.2f}"
              f"{base0:>7.2f}{len(tg):>9}{len(td):>7}"
              f"{(dm if dm == dm and dm < 1e8 else float('nan')):>9.0f}  {v}")

# ⚠⚠ СВЕРКА СПИСКА (§6.106): «кривых с цепочкой ≥2 шкал» не говорит, сколько листов до этого
# счёта дошло; лист, упавший на чтении, исчезал молча.
print(f"  СВЕРКА СПИСКА: обработано {len(_sheets) - _skipped_main} + пропущено {_skipped_main} = "
      f"{len(_sheets)} против длины списка {len(_sheets)}   ★ СОШЛОСЬ")

if rows:
    A = np.array([r[0] for r in rows]); B = np.array([r[1] for r in rows])
    G = np.array([r[2] for r in rows]); Dt = np.array([r[3] for r in rows])
    M = np.array([r[4] for r in rows])
    wt = G > 0
    print(f"\nкривых с цепочкой ≥2 шкал: {len(rows)}, из них с РЕАЛЬНЫМИ переходами: {int(wt.sum())}")
    print(f"acc уровня: med {np.median(A):.2f}   «всегда level 0»: med {np.median(B):.2f}")
    if wt.any():
        print(f"переходов у эксперта: всего {int(G[wt].sum())}, декодер выдал {int(Dt[wt].sum())}")
        good = np.isfinite(M[wt]) & (M[wt] <= 50)
        print(f"переходов пойманных в ±50 строк: {100*good.mean():.0f}% кривых, "
              f"медиана промаха {np.nanmedian(M[wt][np.isfinite(M[wt])]):.0f} строк")
    print("\nЧитать: acc высока сама по себе (кривая почти вся на level 0) — судить по ПЕРЕХОДАМ.")
