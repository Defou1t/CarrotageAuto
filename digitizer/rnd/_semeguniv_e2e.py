r"""_semeguniv_e2e.py — СКВОЗНОЙ ПРОГОН НА КОНТРОЛЬНОМ ЛИСТЕ (Эдуард, 22.07).

Лист: Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx (7 кривых, 2 трека, есть LAS).
Из nlgx берутся ТОЛЬКО рамка и созданные кривые (имена + цепочки масштабов). Трасса и уровни —
наши. Экспертные трассы используются ИСКЛЮЧИТЕЛЬНО для сравнения.

★★ ЗАЩИТА ОТ КОПИРОВАНИЯ ЭКСПЕРТА (главное требование). Три независимые проверки:
  L1  наша трасса не равна экспертной побитово (ловушка §4: emit пишет в КОПИЮ nlgx, у
      незаписанного слота остаётся экспертная кривая и метрика даёт «идеальные 0.0px»);
  L2  наши строки не совпадают с сеткой строк эксперта (экспертная трасса — ВЕРШИНЫ полилинии,
      4-14% строк, §6.4; наша — сплошная. Совпадение сетки = мы взяли его точки);
  L3  доля точных совпадений x по общим строкам < 50% (частичное копирование).
Любая сработавшая проверка помечает кривую как НЕГОДНУЮ и её числа не идут в сводку.

Уровни декодируются НАШЕЙ трассой, параметрами из §6.35 (lam 0.05, dxfrac 0.01, gate_w 8.0) и с
выбором метрики непрерывности ПО ТИПУ ЦЕПОЧКИ. Значения сверяются с LAS.

  <ComfyUI>\python_embeded\python.exe _semeguniv_e2e.py [--seq] [--force-auto]
"""
import sys, io, json, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
from extract_nlgx import extract, NULL
import dataset as ds
import decode_levels as DL
from dataset_build import find_image, find_las
from _multi_replica_probe import dense
from _levels_decode2 import decode2, is_additive
from auto.pipeline import run as pipe_run
from auto.config import Config
from auto import confidence as confidence_mod
from auto import emit as emit_mod
from auto import trace2d as T

SHEET = Path(r"F:\nds\projects\Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx")
OUT = Path(r"F:\nds\output\taskS\semeguniv")
LAM, DXF, GW = 0.05, 0.01, 8.0                 # §6.35, подобрано на держанных скважинах

ap = argparse.ArgumentParser()
ap.add_argument("--seq", action="store_true", help="трассировать оконным селектором (§6.26)")
ap.add_argument("--ckpt", default="seq_model_d45p.pt")
ap.add_argument("--force-auto", action="store_true", help="снять FLAG-гейт (иначе трассируются не все)")
ap.add_argument("--no-escalate", action="store_true",
                help="заглушить эскалацию полосы до ТРЕКА (refine.py:292-299). Абляция 22.07 на "
                     "этом листе: она одна уводит 7/7 честных в 4/7 (OGZ1 1.2px→1267px)")
a = ap.parse_args()

if a.no_escalate:
    from auto import refine as refine_mod
    _despike, _drop = refine_mod.despike, refine_mod.drop_transits

    def refine_no_escalate(fg, line, frame, p, trace_line_fn, track, n_same_color=1, siblings=None):
        hlo = getattr(line, "x_hard_lo", None); hhi = getattr(line, "x_hard_hi", None)
        hard = hlo is not None and hhi is not None
        kw = {"wide_run": 10 ** 6} if getattr(line, "prefer_body", False) else {}
        tr = (trace_line_fn(fg, line, frame, p, x_range=(int(hlo), int(hhi)), **kw) if hard
              else trace_line_fn(fg, line, frame, p, **kw))
        if len(tr) < 30:
            return tr
        tr, _ = _despike(tr, win=p.despike_win, k=p.despike_k, min_jump=p.despike_min_jump)
        if getattr(line, "prefer_body", False):
            tr, _ = _drop(tr, min_span=p.transit_min_span)
        return tr
    refine_mod.refine_trace = refine_no_escalate

if a.force_auto:
    _classify = confidence_mod.classify

    def classify_all_auto(sheet, *args, **kw):
        r = _classify(sheet, *args, **kw)
        for L in sheet.lines:
            L.confidence = "AUTO"
        return r
    confidence_mod.classify = classify_all_auto

TRACES = {}
_map = emit_mod._map_lines_to_slots


def map_capture(traces, model, frame, mnemonics_path, *rest):   # *rest: §6.105, пятый арг `cv`
    TRACES["all"] = list(traces)
    return _map(traces, model, frame, mnemonics_path, *rest)


emit_mod._map_lines_to_slots = map_capture

if a.seq:
    import torch
    from _decoder_core import features
    from _decoder_seq import WindowSelector, OUT as MODELS
    from _decoder_seq_data import MAXC, patch
    from auto import imaging as im
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    NET = WindowSelector().to(DEV)
    NET.load_state_dict(torch.load(MODELS / a.ckpt, map_location=DEV)["sd"]); NET.eval()

    def patched(fg, line, frame, p, band_pad=8, slmax=30.0, wide_run=14, x_range=None,
                jump_limit=None):
        H, W = fg.shape
        if x_range is not None:
            lo = max(0, int(x_range[0])); hi = min(W, int(x_range[1]) + 1)
        else:
            lo = max(0, int(line.x_lo) - band_pad); hi = min(W, int(line.x_hi) + band_pad + 1)
        base = line.x_center
        band = np.ascontiguousarray(fg[:, lo:hi] > 0)
        x = None; v = 0.0; tr = {}
        with torch.no_grad():
            for y in range(max(0, line.y0), min(H, line.y1 + 1)):
                runs = im.row_runs(fg[y, lo:hi])
                if not runs:
                    if x is not None:
                        x = x + float(np.clip(v, -slmax, slmax))
                    continue
                A = np.array([r[0] + lo for r in runs]); B = np.array([r[1] + lo for r in runs])
                C = np.array([r[2] + lo for r in runs], float)
                if x is None:
                    k = int(np.argmin(np.abs(C - base)))
                    x = float(C[k]); v = 0.0; tr[y] = x; continue
                pred = x + float(np.clip(v, -slmax, slmax))
                idx, X = features(A, B, C, pred, x, v, base, MAXC)
                if len(idx) == 1:
                    k = int(idx[0])
                else:
                    ink, val = patch(band, lo, y, pred)
                    pt = torch.from_numpy(np.stack([ink, val]).astype(np.float32))[None].to(DEV)
                    f = np.zeros((1, MAXC, 10), np.float32); f[0, :len(idx)] = X
                    m = np.zeros((1, MAXC), np.float32); m[0, :len(idx)] = 1
                    sc = NET(pt, torch.from_numpy(f).to(DEV), torch.from_numpy(m).to(DEV))
                    k = int(idx[int(sc[0].argmax().item())])
                nx = float(C[k])                            # точка = ЦЕНТР рана (§6.26)
                v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
        T._extend_ends(tr, fg, lo, hi, slmax)
        return tr
    T.trace_line = patched


def leak_checks(ours, exp_rows, exp_x):
    """(L1,L2,L3) — см. докстринг. Возвращает список сработавших меток."""
    bad = []
    oy = np.array(sorted(ours)); ox = np.array([ours[y] for y in oy], float)
    if len(oy) == len(exp_rows) and np.array_equal(oy, exp_rows) and np.allclose(ox, exp_x):
        bad.append("L1:побитово=эксперт")
    if len(oy) == len(exp_rows) and np.array_equal(oy, exp_rows):
        bad.append("L2:сетка строк=эксперт")
    common = np.intersect1d(oy, exp_rows)
    if len(common) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(exp_rows.tolist(), exp_x.tolist()))
        same = np.mean([abs(om[y] - em[y]) < 1e-9 for y in common])
        if same > 0.5:
            bad.append(f"L3:совпало x у {100*same:.0f}% строк")
    return bad


img = find_image(SHEET); las = find_las(SHEET)
print(f"лист: {SHEET.name}\nкартинка: {img}\nLAS: {las}")
print(f"режим: {'ОКОННЫЙ СЕЛЕКТОР' if a.seq else 'база'}"
      f"{', FLAG снят' if a.force_auto else ''}, уровни lam={LAM} dxfrac={DXF} gate_w={GW}\n")

cfg = Config(); cfg.out = OUT / ("seq" if a.seq else "base")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sheet, traces, res = pipe_run(str(img), frame_nlgx=str(SHEET), cfg=cfg, stages=False)
print(f"U1 нашёл линий: {len(sheet.lines)}   записано слотов: {res.get('written')}")

m = extract(str(SHEET))
gts = {c["name"]: c for c in ds.real_curves(m) if any(x != NULL for x in c["xs"])}
alltr = TRACES.get("all", [])
cols, arr = ds.load_las(str(las)); matches = ds.match_las(m, cols, arr)
da = m["depth_axis"]; ty, by = da["top_y"], da["bottom_y"]; td, bd = da["top_depth"], da["bottom_depth"]
depths = arr[:, 0]

# ★★ НАЗНАЧЕНИЕ ОДИН-К-ОДНОМУ (постановка заказчика: «отличать линии между собой»).
# Раньше каждая кривая независимо брала лучшую для себя трассу — и три кривые правого пучка
# выбрали ОДНУ И ТУ ЖЕ (трасса #9). Так «честные» можно набрать одной хорошей трассой на все
# слоты. Теперь пары (кривая, трасса) разбираются жадно по возрастанию med, и трасса, уже
# отданная кривой, ДРУГОЙ не достаётся: K кривых ← K РАЗНЫХ трасс.
def assign_one_to_one(gts_map, alltr_):
    pairs = []
    for nm, gm in gts_map.items():
        for ti, (L, tr) in enumerate(alltr_):
            common = [y for y in tr if y in gm]
            if len(common) < 30:
                continue
            dd = np.array([abs(tr[y] - gm[y]) for y in common])
            pairs.append((float(np.median(dd)), len(common) / len(gm), nm, ti))
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))     # сперва полноразмерные (cov>=0.9), затем по med
    got, used = {}, set()
    for med, cov, nm, ti in pairs:
        if nm in got or ti in used:
            continue
        got[nm] = (med, cov, ti); used.add(ti)
    return got


GMAPS = {}
for name, g in gts.items():
    d = dense(g)
    if len(d) >= 50:
        GMAPS[name] = {y: d[y] for y in sorted(d)}
ASSIGN = assign_one_to_one(GMAPS, alltr)

print(f"\n{'кривая':<8}{'цепь':>5}{'тип':>7}{'наша med':>10}{'cov':>6}{'p3%':>6}"
      f"{'перех.GT':>9}{'наши':>6}{'corr LAS':>9}{'corr баз0':>10}  контроль")
rows = []
for name, g in gts.items():
    d = dense(g)
    if len(d) < 50:
        continue
    ey = np.array(sorted(d)); ex = np.array([d[y] for y in ey], float)
    exp_rows = np.array([g["top_y"] + i for i, x in enumerate(g["xs"]) if x != NULL])
    exp_x = np.array([x for x in g["xs"] if x != NULL], float)
    gmap = dict(zip(ey.tolist(), ex.tolist()))
    # ★ ОТБОР ЛУЧШЕЙ ТРАССЫ — ЧЕСТНЫЙ (§6.9: med без cov не цитировать). Ранний вариант брал
    # argmin(med) и на GZ11 выбрал ФРАГМЕНТ (cov 0.23, med 1.9), обойдя полноразмерную трассу:
    # это тот же метрический самообман, только в правиле отбора. Теперь: минимальная med СРЕДИ
    # трасс с cov>=0.9; если таких нет — трасса с максимальным cov (и честно видно, что её нет).
    cands = []
    for ti, (L, tr) in enumerate(alltr):     # имена не назначаем (§6.33)
        common = [y for y in tr if y in gmap]
        if len(common) < 30:
            continue
        dd = np.array([abs(tr[y] - gmap[y]) for y in common])
        cands.append((float(np.median(dd)), len(common) / len(ey), tr, ti))
    if not cands or name not in ASSIGN:
        print(f"{name.split()[0]:<8}   нашей трассы нет"); continue
    bmed, bcov, bti = ASSIGN[name]                   # трасса, ЗАКРЕПЛЁННАЯ за этой кривой
    best = alltr[bti][1]
    if best is None:  # noqa
        print(f"{name.split()[0]:<8}   нашей трассы нет"); continue
    leak = leak_checks(best, exp_rows, exp_x)
    common = [y for y in best if y in gmap]
    dd = np.array([abs(best[y] - gmap[y]) for y in common])
    cov = len(common) / len(ey); p3 = float((dd <= 3).mean())
    fam = DL.build_family(m, g); gl = DL.gt_levels(g)
    kind = ("аддит" if fam and is_additive(fam) else "мульт") if len(fam) > 1 else "-"
    # rail_from_trace: упор берётся из САМОЙ трассы, а не из x_right объявленной оси (§6.41 —
    # рамка часто сдвоенная, тушь занимает медианно 50% ширины оси)
    dec = (decode2(best, fam, LAM, DXF, GW, auto=True, rail_from_trace=True)
           if len(fam) > 1 else {y: 0 for y in best})

    def trans(lv):
        ys = sorted(lv)
        return [ys[i] for i in range(1, len(ys)) if lv[ys[i]] != lv[ys[i - 1]]]
    tg = trans({y: gl[y] for y in sorted(gl)}) if gl else []
    tdd = trans(dec)
    ci = matches.get(name, {}).get("col_idx")
    c_dec = c_b0 = float("nan")
    if ci is not None and len(fam) >= 1:
        maps = [DL.scale_map(s) for s in fam]; K = len(fam)
        lasv = arr[:, ci]; V1 = []; V0 = []; LL = []
        for y, x in best.items():
            dep = td + (y - ty) * (bd - td) / (by - ty)
            li = int(np.clip(np.searchsorted(depths, dep), 0, len(depths) - 1))
            lv = lasv[li]
            if not np.isfinite(lv):
                continue
            V1.append(maps[min(int(dec.get(y, 0)), K - 1)](x)); V0.append(maps[0](x)); LL.append(lv)
        if len(LL) > 50:
            V1 = np.array(V1); V0 = np.array(V0); LL = np.array(LL)
            c_dec = float(np.corrcoef(V1, LL)[0, 1]) if V1.std() > 0 else float("nan")
            c_b0 = float(np.corrcoef(V0, LL)[0, 1]) if V0.std() > 0 else float("nan")
    ctrl = "✓ наша" if not leak else " | ".join(leak)
    rows.append({"name": name.split()[0], "med": bmed, "cov": cov, "p3": p3, "trace": bti,
                 "tg": len(tg), "td": len(tdd), "corr": c_dec, "corr0": c_b0, "leak": leak})
    print(f"{name.split()[0]:<8}{len(fam):>5}{kind:>7}{bmed:>10.1f}{cov:>6.2f}{100*p3:>6.0f}"
          f"{len(tg):>9}{len(tdd):>6}{c_dec:>9.2f}{c_b0:>10.2f}  {ctrl}")

# ★★ «ОТЛИЧАТЬ ЛИНИИ МЕЖДУ СОБОЙ» (постановка заказчика §6.33): одна наша трасса не имеет права
# засчитываться ДВУМ кривым. Без этой проверки счёт честных ничего не значит — можно «выиграть»,
# подав одну хорошую трассу на все слоты.
from collections import Counter
usage = Counter(r["trace"] for r in rows)
dup = {t: c for t, c in usage.items() if c > 1}
if dup:      # после назначения один-к-одному этого быть не должно — страховка от ошибки в коде
    print(f"\n⚠⚠ ОШИБКА НАЗНАЧЕНИЯ: одна трасса у нескольких кривых: {dup}")
else:
    print(f"\n✓ РАЗЛИЧЕНИЕ ЛИНИЙ: {len(rows)} кривых ← {len(set(usage))} РАЗНЫХ трасс (один-к-одному)")

# ── ПОЧЕМУ ДЕКОДЕР ВРЕДИТ НА OGZ1: цена ошибки, а не её количество ──────────────────────────
# Шкала уровня 1 в 5 раз шире уровня 0 (OGZ1: v[-11..8] → v[-55..40]). Несколько ложных строк на
# уровне 1 дают значения ±55 там, где кривая лежит в ±11 — и линейная corr рушится. Поэтому здесь
# corr считается для РЯДА ЗНАЧЕНИЙ level_bias (приор к низкому уровню) и для ОРАКУЛЬНЫХ уровней:
# оракул отделяет «виновата трасса» от «виноват декодер».
RAILS = (0.0, 0.3, 0.5, 0.7, 0.85)
RAILROWS = []
print(f"\nПРИОР К НИЗКОМУ УРОВНЮ (level_bias) — слепой рычаг:")
print(f"{'кривая':<8}{'база0':>8}" + "".join(f"{'bias='+str(b):>10}" for b in (0.0, 0.005, 0.02, 0.05, 0.1)) + f"{'ОРАКУЛ':>9}")
for name, g in gts.items():
    fam = DL.build_family(m, g)
    if len(fam) < 2 or name not in ASSIGN:
        continue
    tr = alltr[ASSIGN[name][2]][1]
    ci = matches.get(name, {}).get("col_idx")
    if ci is None:
        continue
    maps = [DL.scale_map(s) for s in fam]; K = len(fam); lasv = arr[:, ci]
    gl = DL.gt_levels(g)

    def corr_for(levels):
        V, L = [], []
        for y, x in tr.items():
            dep = td + (y - ty) * (bd - td) / (by - ty)
            li = int(np.clip(np.searchsorted(depths, dep), 0, len(depths) - 1))
            lv = lasv[li]
            if not np.isfinite(lv):
                continue
            V.append(maps[min(int(levels.get(y, 0)), K - 1)](x)); L.append(lv)
        V = np.array(V); L = np.array(L)
        return float(np.corrcoef(V, L)[0, 1]) if len(V) > 50 and V.std() > 0 else float("nan")
    cells = [corr_for({y: 0 for y in tr})]
    for b in (0.0, 0.005, 0.02, 0.05, 0.1):
        cells.append(corr_for(decode2(tr, fam, LAM, DXF, GW, level_bias=b, auto=True)))
    cells.append(corr_for(gl))
    print(f"{name.split()[0]:<8}{cells[0]:>8.2f}" + "".join(f"{c:>10.2f}" for c in cells[1:-1])
          + f"{cells[-1]:>9.2f}")
    RAILROWS.append((name.split()[0], corr_for, fam, gl,
                     [corr_for(decode2(tr, fam, LAM, DXF, GW, rail_gate=rg, auto=True))
                      for rg in RAILS]))

# ДОМЕННЫЙ рычаг вместо слепого приора: перо уходит на backup, ТОЛЬКО упершись в правый край
# шкалы (Эдуард: значения читаются у шапки, переход происходит на обороте пера). rail_gate делает
# переход ВВЕРХ дорогим, если предыдущая точка НЕ у упора.
print(f"\nРЕЛЬС-ГЕЙТ (rail_gate) — доменное ограничение «переход только от упора»:")
print(f"{'кривая':<8}" + "".join(f"{'rail='+str(r):>10}" for r in RAILS))
for nm, _, _, _, cells in RAILROWS:
    print(f"{nm:<8}" + "".join(f"{c:>10.2f}" for c in cells))

good = [r for r in rows if not r["leak"]]
print(f"\nкривых разобрано: {len(rows)}, из них НЕ помечено утечкой: {len(good)}")
if len(good) != len(rows):
    print("⚠⚠ ЕСТЬ ПОДОЗРЕНИЕ НА КОПИРОВАНИЕ ЭКСПЕРТА — числа по этим кривым НЕГОДНЫ")
if good:
    print(f"med(med) {np.median([r['med'] for r in good]):.1f}px   "
          f"med(cov) {np.median([r['cov'] for r in good]):.2f}   "
          f"★честных {sum(1 for r in good if r['med']<=3 and r['cov']>=0.9)}/{len(good)}")
    cc = [r["corr"] for r in good if r["corr"] == r["corr"]]
    c0 = [r["corr0"] for r in good if r["corr0"] == r["corr0"]]
    if cc:
        print(f"corr с LAS: наши уровни med {np.median(cc):.3f}   «всегда 1х» med {np.median(c0):.3f}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"e2e_{'seq' if a.seq else 'base'}.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
