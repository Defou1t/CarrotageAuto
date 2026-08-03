r"""_esc_limit.py — ОГРАНИЧЕНИЕ ЭСКАЛАЦИИ ПОЛОСЫ (вместо её удаления). Гейт вариантов.

§6.37: ветка `refine.py:292-299` («≥3% строк с чернилами за полосой → перетрасс полосой ВО ВЕСЬ
ТРЕК») — единственный компонент доводки, который рушит идентичность: 26 листов, честных 7 → 19
при полном заглушении. Но заглушать НЕЛЬЗЯ: ветка решает реальную задачу (перо ушло к упору,
полоса это срезала; калибровка BK теряла выносы к рельсу).

ТРИ ДЕФЕКТА ТЕКУЩЕЙ РЕАЛИЗАЦИИ (каждый подтверждён замером):
 1. РАСШИРЕНИЕ НЕОГРАНИЧЕНО: полоса OGZ1 ≈1000px, трек 2237px ⇒ после эскалации в полосе ВСЕ
    7 кривых листа. Ось 1 §6.29: расширение полосы разрушает идентичность.
 2. КРИТЕРИЙ ПРИЁМКИ ПООЩРЯЕТ ОТКАЗ: «достаёт дальше» (xw.max() > xt.max()+5) тривиально
    выполняется, когда трасса ПЕРЕСКОЧИЛА НА СОСЕДА — сосед лежит дальше по x. Тест не отличает
    «дотянулись до упора» от «ушли на чужую кривую».
 3. УСЛОВИЕ СРАБАТЫВАНИЯ СЛИШКОМ СЛАБОЕ на многокривом треке: чернила за полосой есть почти
    всегда (это соседние кривые), порог 3% достигается мгновенно.

ВАРИАНТЫ (гейтятся на тех же 26 листах, метрика и контроли §6.36):
  esc       — как в проде (эталон);
  noesc     — полное заглушение (§6.37, верхняя граница выигрыша);
  margin    — расширять на ±M px вместо всего трека (лечит дефект 1);
  neighbor  — расширять до СЕРЕДИНЫ между своей линией и ближайшим соседом того же цвета
              (граница Вороного) — лечит дефект 1 доменно, без магической константы;
  bulk      — полный трек, но принимать ТОЛЬКО если основная часть трассы не сдвинулась
              (медиана |Δx| ≤ tol): удлинение на концах — да, другая кривая — нет (лечит дефект 2);
  nb+bulk   — neighbor + bulk.

  python _esc_limit.py [--n 20] [--margin 150] [--bulk-tol 20]
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
from auto import trace2d as T
from _decoder_data import train_sheets
from _relatch_bench import SH, ARCH

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--margin", type=float, default=150.0)
ap.add_argument("--bulk-tol", type=float, default=20.0)
ap.add_argument("--modes", default="esc,noesc,margin,neighbor,bulk,nb+bulk")
a = ap.parse_args()

_classify = confidence_mod.classify


def classify_all_auto(sheet, *args, **kw):
    r = _classify(sheet, *args, **kw)
    for L in sheet.lines:
        L.confidence = "AUTO"
    return r


confidence_mod.classify = classify_all_auto

TR = {}
_map = emit_mod._map_lines_to_slots


def map_capture(traces, model, frame, mnemonics_path, *rest):   # *rest: §6.105, пятый арг `cv`
    TR["all"] = list(traces)
    return _map(traces, model, frame, mnemonics_path, *rest)


emit_mod._map_lines_to_slots = map_capture

# refine_trace не получает список линий листа, а для границы Вороного он нужен ⇒ перехватываем
# trace_auto и запоминаем линии текущего листа (поведение самого trace_auto не меняется).
LINES = {"all": []}
_trace_auto = T.trace_auto


def trace_auto_cap(rgb, sheet, p=None):
    LINES["all"] = list(sheet.lines)
    return _trace_auto(rgb, sheet, p)


T.trace_auto = trace_auto_cap
_despike, _drop = refine_mod.despike, refine_mod.drop_transits


def neighbor_bounds(line):
    """Границы Вороного между линией и ближайшими соседями ТОГО ЖЕ ЦВЕТА на том же треке
    (именно они делят маску `fg` и потому реально конкурируют за тушь)."""
    lo = hi = None
    for L in LINES["all"]:
        if L is line or L.color != line.color or L.track_index != line.track_index:
            continue
        if L.x_center < line.x_center:
            b = (L.x_center + line.x_center) / 2
            lo = b if lo is None else max(lo, b)
        elif L.x_center > line.x_center:
            b = (L.x_center + line.x_center) / 2
            hi = b if hi is None else min(hi, b)
    return lo, hi


def make_refine(mode, margin, bulk_tol):
    def refine(fg, line, frame, p, trace_line, track, n_same_color=1):
        hlo = getattr(line, "x_hard_lo", None); hhi = getattr(line, "x_hard_hi", None)
        hard = hlo is not None and hhi is not None
        kw = {"wide_run": 10 ** 6} if getattr(line, "prefer_body", False) else {}
        tr = (trace_line(fg, line, frame, p, x_range=(int(hlo), int(hhi)), **kw) if hard
              else trace_line(fg, line, frame, p, **kw))
        if len(tr) < 30:
            return tr
        if mode != "noesc":
            band_pad = 8
            lo = max(0, int(line.x_lo) - band_pad); hi = min(fg.shape[1], int(line.x_hi) + band_pad + 1)
            tl, tr_ = max(0, track.x_left), min(fg.shape[1], track.x_right)
            if hard:
                lo = max(lo, int(hlo)); hi = min(hi, int(hhi))
                tl = max(tl, int(hlo)); tr_ = min(tr_, int(hhi))
            # ★ ОГРАНИЧЕНИЕ ЗОНЫ РАСШИРЕНИЯ (дефект 1)
            if mode == "margin":
                tl = max(tl, lo - margin); tr_ = min(tr_, hi + margin)
            elif mode in ("neighbor", "nb+bulk"):
                nlo, nhi = neighbor_bounds(line)
                if nlo is not None:
                    tl = max(tl, nlo)
                if nhi is not None:
                    tr_ = min(tr_, nhi)
            rows = sorted(tr)
            beyond = 0
            for y in rows:
                if (hi < tr_ and fg[y, int(hi):int(tr_)].any()) or \
                   (tl < lo and fg[y, int(tl):int(lo)].any()):
                    beyond += 1
            if beyond / max(1, len(rows)) >= 0.03 and tr_ > tl:
                tr_wide = trace_line(fg, line, frame, p, x_range=(int(tl), int(tr_)), **kw)
                if len(tr_wide) >= 0.8 * len(tr):
                    xw = np.array([tr_wide[y] for y in sorted(tr_wide)])
                    xt = np.array([tr[y] for y in rows])
                    reaches = xw.max() > xt.max() + 5 or xw.min() < xt.min() - 5
                    ok = reaches
                    if mode in ("bulk", "nb+bulk"):
                        # ★ ПРИЁМКА ПО ТЕЛУ ТРАССЫ (дефект 2): удлинение на концах допустимо,
                        # смещение ОСНОВНОЙ массы точек = ушли на другую кривую.
                        common = [y for y in rows if y in tr_wide]
                        if len(common) < 30:
                            ok = False
                        else:
                            shift = float(np.median([abs(tr_wide[y] - tr[y]) for y in common]))
                            ok = reaches and shift <= bulk_tol
                    if ok:
                        tr = tr_wide
        tr, _ = _despike(tr, win=p.despike_win, k=p.despike_k, min_jump=p.despike_min_jump)
        if getattr(line, "prefer_body", False):
            tr, _ = _drop(tr, min_span=p.transit_min_span)
        return tr
    return refine


ORIG = refine_mod.refine_trace


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
    img = find_image(n)
    if not img:
        return None
    TR.clear()
    cfg = Config(); cfg.out = Path(r"F:\nds\output\taskS\esc_limit") / n.stem[:40]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
    except Exception:
        return None
    alltr = TR.get("all", [])
    m = extract(str(n))
    gts = {c["name"]: c for c in ds.real_curves(m) if any(x != NULL for x in c["xs"])}
    GM = {}
    for nm, g in gts.items():
        d = dense(g)
        if len(d) >= 50:
            GM[nm] = {y: d[y] for y in sorted(d)}
    if not GM or not alltr:
        return (0, len(GM), float("nan"))
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
    for med, cov, nm, ti in pairs:
        if nm in got or ti in used:
            continue
        got[nm] = (med, cov, ti); used.add(ti)
    honest = 0; meds = []
    for nm, (med, cov, ti) in got.items():
        if leaked(alltr[ti][1], gts[nm]):
            continue
        meds.append(med)
        if med <= 3 and cov >= 0.9:
            honest += 1
    return (honest, len(GM), float(np.median(meds)) if meds else float("nan"))


sheets = [Path(r"F:\nds\projects\Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx")]
sheets += [ARCH / s for s in SH]
sheets += [s for s in train_sheets(a.n) if s not in sheets]
modes = a.modes.split(",")
print(f"листов: {len(sheets)}   варианты: {modes}   margin={a.margin} bulk_tol={a.bulk_tol}\n")

res = {mo: [] for mo in modes}
for n in sheets:
    line = f"{n.name[:40]:<42}"
    for mo in modes:
        refine_mod.refine_trace = ORIG if mo == "esc" else make_refine(mo, a.margin, a.bulk_tol)
        r = measure(n)
        res[mo].append(r)
        line += f"{(r[0] if r else -1):>4}/{(r[1] if r else 0):<3}"
    print(line)
refine_mod.refine_trace = ORIG

print(f"\n{'вариант':<12}{'ЧЕСТНЫХ':>9}{'кривых':>8}{'med(med)':>10}")
base = None
for mo in modes:
    rr = [r for r in res[mo] if r]
    h = sum(r[0] for r in rr); k = sum(r[1] for r in rr)
    md = float(np.nanmedian([r[2] for r in rr]))
    if mo == "esc":
        base = h
    print(f"{mo:<12}{h:>9}{k:>8}{md:>10.1f}" + ("" if base is None or mo == "esc" else
          f"   {'★ +' if h > base else ('✗ -' if h < base else '= ')}{abs(h-base)}"))
Path(r"F:\nds\output\taskS\esc_limit").mkdir(parents=True, exist_ok=True)
(Path(r"F:\nds\output\taskS\esc_limit") / "res.json").write_text(
    json.dumps({mo: res[mo] for mo in modes}, ensure_ascii=False), encoding="utf-8")
print("\nЦель: вариант, который держит честных на уровне noesc, НО сохраняет ветку дотяга.")
