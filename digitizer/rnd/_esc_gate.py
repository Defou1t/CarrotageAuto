r"""_esc_gate.py — ШИРОКИЙ ГЕЙТ: ВРЕДИТ ЛИ ЭСКАЛАЦИЯ ПОЛОСЫ ДО ТРЕКА (refine.py:292-299).

§6.37: на контрольном листе Semeguniv заглушение этой ветки дало 4/7 → 7/7 честных (med 2.5 →
1.3px). Но §6.15 и §6.28.3 заглушали её же на 5 листах стенда и пользы не видели. Гипотеза:
эскалация вредна ТАМ, ГДЕ ПОЛОСА U1 ХОРОШАЯ, и безразлична там, где она плохая. Одним листом
такое не решается (durable §4/§6.20.5) — нужен широкий прогон.

Метрика — как в §6.36, со всеми контролями:
  * назначение трасс кривым ОДИН-К-ОДНОМУ (иначе одна трасса «закрывает» несколько кривых);
  * отбор среди cov>=0.9, med без cov не цитируется (§6.9);
  * контроль копирования эксперта (L1 сетка+значения, L3 доля точных совпадений x);
  * имена НЕ назначаются (постановка §6.33) — сравнение перестановочно-свободное.
Трассировщик — БАЗОВЫЙ (прод-умолчание): вопрос про эскалацию, а не про селектор.

  python _esc_gate.py [--n 20]
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
from dataset_build import find_image
from _multi_replica_probe import dense
from auto.pipeline import run as pipe_run
from auto.config import Config
from auto import confidence as confidence_mod
from auto import emit as emit_mod
from auto import refine as refine_mod
from _decoder_data import train_sheets, HELD
from _relatch_bench import SH, ARCH

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()

_classify = confidence_mod.classify


def classify_all_auto(sheet, *args, **kw):
    r = _classify(sheet, *args, **kw)
    for L in sheet.lines:
        L.confidence = "AUTO"                    # иначе трассируется 1-2 линии и сравнивать нечего
    return r


confidence_mod.classify = classify_all_auto

TR = {}
_map = emit_mod._map_lines_to_slots


def map_capture(traces, model, frame, mnemonics_path):
    TR["all"] = list(traces)
    return _map(traces, model, frame, mnemonics_path)


emit_mod._map_lines_to_slots = map_capture
ORIG_REFINE = refine_mod.refine_trace
_despike, _drop = refine_mod.despike, refine_mod.drop_transits


def refine_no_escalate(fg, line, frame, p, trace_line_fn, track):
    """refine_trace БЕЗ ветки «недотяг → полоса во весь трек». Всё остальное сохранено."""
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


def leaked(tr, g):
    er = np.array([g["top_y"] + i for i, x in enumerate(g["xs"]) if x != NULL])
    ex = np.array([x for x in g["xs"] if x != NULL], float)
    oy = np.array(sorted(tr)); ox = np.array([tr[y] for y in oy], float)
    if len(oy) == len(er) and np.array_equal(oy, er) and np.allclose(ox, ex):
        return True
    common = np.intersect1d(oy, er)
    if len(common) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(er.tolist(), ex.tolist()))
        if np.mean([abs(om[y] - em[y]) < 1e-9 for y in common]) > 0.5:
            return True
    return False


def measure(n):
    """Один лист: (честных, всего кривых, med(med), утечек). Трассы берутся ДО emit."""
    img = find_image(n)
    if not img:
        return None
    TR.clear()
    cfg = Config(); cfg.out = Path(r"F:\nds\output\taskS\esc_gate") / n.stem[:40]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
    except Exception as e:
        return ("ERR", type(e).__name__)
    alltr = TR.get("all", [])
    m = extract(str(n))
    gts = {c["name"]: c for c in ds.real_curves(m) if any(x != NULL for x in c["xs"])}
    GM = {}
    for nm, g in gts.items():
        d = dense(g)
        if len(d) >= 50:
            GM[nm] = {y: d[y] for y in sorted(d)}
    if not GM or not alltr:
        return (0, len(GM), float("nan"), 0)
    pairs = []
    for nm, gm in GM.items():
        for ti, (L, tr) in enumerate(alltr):
            common = [y for y in tr if y in gm]
            if len(common) < 30:
                continue
            dd = np.array([abs(tr[y] - gm[y]) for y in common])
            pairs.append((float(np.median(dd)), len(common) / len(gm), nm, ti))
    pairs.sort(key=lambda q: (q[1] < 0.9, q[0]))
    got, used = {}, set()
    for med, cov, nm, ti in pairs:                # ОДИН-К-ОДНОМУ
        if nm in got or ti in used:
            continue
        got[nm] = (med, cov, ti); used.add(ti)
    honest = leaks = 0; meds = []
    for nm, (med, cov, ti) in got.items():
        if leaked(alltr[ti][1], gts[nm]):
            leaks += 1; continue
        meds.append(med)
        if med <= 3 and cov >= 0.9:
            honest += 1
    return (honest, len(GM), float(np.median(meds)) if meds else float("nan"), leaks)


sheets = [Path(r"F:\nds\projects\Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx")]
sheets += [ARCH / s for s in SH]                 # 5 листов стенда (трудный класс)
sheets += [s for s in train_sheets(a.n) if s not in sheets]
print(f"листов: {len(sheets)}\n")
print(f"{'лист':<46}{'кривых':>7}{'ЧЕСТНЫХ есk':>12}{'ЧЕСТНЫХ без':>12}{'med есk':>9}{'med без':>9}")
tot = [0, 0, 0]
rows = []
for n in sheets:
    refine_mod.refine_trace = ORIG_REFINE
    r1 = measure(n)
    refine_mod.refine_trace = refine_no_escalate
    r2 = measure(n)
    if not r1 or not r2 or r1[0] == "ERR" or r2[0] == "ERR":
        print(f"{n.name[:44]:<46}  пропуск ({r1 if r1 else '-'} / {r2 if r2 else '-'})"); continue
    h1, k1, m1, lk1 = r1; h2, k2, m2, lk2 = r2
    tot[0] += k1; tot[1] += h1; tot[2] += h2
    rows.append({"sheet": n.name, "k": k1, "h_esc": h1, "h_noesc": h2, "med_esc": m1, "med_noesc": m2,
                 "leaks": lk1 + lk2})
    mark = " ★" if h2 > h1 else (" ✗" if h2 < h1 else "")
    print(f"{n.name[:44]:<46}{k1:>7}{h1:>12}{h2:>12}{m1:>9.1f}{m2:>9.1f}{mark}"
          + (f"  утечек {lk1+lk2}" if lk1 + lk2 else ""))

print(f"\nИТОГО кривых {tot[0]}: ЧЕСТНЫХ с эскалацией {tot[1]}, БЕЗ эскалации {tot[2]}")
better = sum(1 for r in rows if r["h_noesc"] > r["h_esc"])
worse = sum(1 for r in rows if r["h_noesc"] < r["h_esc"])
print(f"листов лучше без эскалации: {better}, хуже: {worse}, без изменений: {len(rows)-better-worse}")
Path(r"F:\nds\output\taskS\esc_gate").mkdir(parents=True, exist_ok=True)
(Path(r"F:\nds\output\taskS\esc_gate") / "esc_gate.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nЧитать: правка идёт в auto/ ТОЛЬКО если честных больше и НИ ОДИН лист не деградировал")
print("существенно — по прецеденту §6.12 (немонотонность отката) и §6.20.5.")
