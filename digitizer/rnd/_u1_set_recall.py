r"""_u1_set_recall.py — СКОЛЬКО КРИВЫХ МЫ ВЕДЁМ ПРАВИЛЬНО, ЕСЛИ ИМЯ НАЗНАЧИТ ЭКСПЕРТ.

Почему вопрос поставлен так. §6.31: U1 даёт линию, попадающую в кривую, у 96% слотов, а
`emit._map_lines_to_slots` сажает её лишь в 12%. Чинить маппинг геометрией НЕЧЕМ — замер 22.07:
x-диапазоны scale-осей слотов на листе СОВПАДАЮТ (1-2 разных на 4-8 слотов; это §6.22 с другой
стороны), а порядок кривых в файле не равен порядку слева-направо НИ НА ОДНОМ из 5 листов.
Т.е. рамка не несёт идентичности, и «какая кривая чья» неоткуда взять.

Тогда меняется вопрос: если ИМЯ назначает эксперт (а не пайплайн), сколько кривых пайплайн уже
ведёт честно? Здесь это меряется прямо: трассируются ВСЕ линии (FLAG снят), и для каждой
экспертной кривой ищется ЛУЧШАЯ из наших трасс.

  НАБОРНАЯ ЧЕСТНОСТЬ = доля кривых листа, у которых ЕСТЬ наша трасса с med<=3px И cov>=0.9.

Это верхняя граница пайплайна без решения задачи идентичности — и одновременно нижняя граница
пользы для эксперта: столько кривых ему останется только ПОДПИСАТЬ.
⚠ Трассы НЕ добиваются и не интерполируются (§6.20.3): cov считается по строкам, которые реально
пройдены.

  <ComfyUI>\python_embeded\python.exe _u1_set_recall.py [--seq]
"""
import sys, io, argparse, contextlib
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
from auto import trace2d as T
from _relatch_bench import SH, ARCH

ap = argparse.ArgumentParser()
ap.add_argument("--seq", action="store_true", help="трассировать оконным селектором (§6.26)")
ap.add_argument("--ckpt", default="seq_model_d45p.pt")
ap.add_argument("--stitch", action="store_true",
                help="сшивать трассы РАЗНЫХ линий по вертикали (замер 22.07: точность 2-6px есть, "
                     "cov 0.31-0.78 — линия U1 накрывает лишь часть глубины кривой)")
ap.add_argument("--stitch-tol", type=float, default=20.0, help="допуск стыковки по x, px")
a = ap.parse_args()

_classify = confidence_mod.classify


def classify_all_auto(sheet, *args, **kw):
    r = _classify(sheet, *args, **kw)
    for L in sheet.lines:
        L.confidence = "AUTO"                       # трассировать ВСЕ линии, а не только AUTO
    return r


confidence_mod.classify = classify_all_auto

TRACES = {}
_map = emit_mod._map_lines_to_slots


def map_capture(traces, model, frame, mnemonics_path):
    TRACES["all"] = list(traces)                     # ВСЕ трассы листа, до маппинга
    return _map(traces, model, frame, mnemonics_path)


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
    NF = 10

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
                    f = np.zeros((1, MAXC, NF), np.float32); f[0, :len(idx)] = X
                    m = np.zeros((1, MAXC), np.float32); m[0, :len(idx)] = 1
                    sc = NET(pt, torch.from_numpy(f).to(DEV), torch.from_numpy(m).to(DEV))
                    k = int(idx[int(sc[0].argmax().item())])
                nx = float(C[k])
                v = 0.6 * v + 0.4 * (nx - x); x = nx; tr[y] = float(nx)
        T._extend_ends(tr, fg, lo, hi, slmax)
        return tr
    T.trace_line = patched

def stitch(traces, tol):
    """Сшить трассы РАЗНЫХ линий, если они продолжают друг друга по глубине И сходятся по x на
    стыке. НИЧЕГО НЕ ДОСТРАИВАЕТСЯ: берётся ТОЛЬКО объединение реально пройденных строк, дыры
    остаются дырами (иначе это добивка §6.20.3 — метрический артефакт).
    Жадно: пока есть пара, у которой конец одной и начало другой ближе tol по x."""
    items = [dict(tr) for _, tr in traces if len(tr) >= 30]
    changed = True
    while changed and len(items) > 1:
        changed = False
        best = None
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                a_, b_ = items[i], items[j]
                ai, aj = max(a_), min(b_)
                if not (aj > ai):                       # b должна идти НИЖЕ a
                    continue
                gap = aj - ai
                if gap > 400:                           # слишком далеко — не одна кривая
                    continue
                d = abs(a_[ai] - b_[aj])
                if d <= tol and (best is None or d < best[0]):
                    best = (d, i, j)
        if best:
            _, i, j = best
            items[i].update(items[j]); items.pop(j); changed = True
    return [(None, t) for t in items]


print(f"трассировщик: {'ОКОННЫЙ СЕЛЕКТОР' if a.seq else 'база (прод)'}"
      f"{', СШИВКА фрагментов' if a.stitch else ''}\n")
print(f"{'скважина':<12}{'кривая':<8}{'линий':>6}{'лучш.med':>10}{'cov':>7}{'p3%':>7}  вердикт")
tot = honest = near = 0
for rel in SH:
    n = ARCH / rel
    well = n.parent.parent.name
    img = find_image(n)
    TRACES.clear()
    cfg = Config(); cfg.out = Path(r"F:\nds\output\taskS\set_recall") / well
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            pipe_run(str(img), frame_nlgx=str(n), cfg=cfg, stages=False)
    except Exception as e:
        print(f"{well:<12} ERR {type(e).__name__}: {e}"); continue
    alltr = TRACES.get("all", [])
    if a.stitch:
        alltr = stitch(alltr, a.stitch_tol)
    mo = extract(str(n))
    for g in [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]:
        d = dense(g)
        if len(d) < 50:
            continue
        tot += 1
        gy = np.array(sorted(d)); gx = np.array([d[y] for y in gy], float)
        gmap = dict(zip(gy.tolist(), gx.tolist()))
        best = (1e9, 0.0, 0.0)
        for L, tr in alltr:
            common = [y for y in tr if y in gmap]
            if len(common) < 30:
                continue
            dd = np.array([abs(tr[y] - gmap[y]) for y in common])
            med = float(np.median(dd)); cov = len(common) / len(gy)
            if med < best[0]:
                best = (med, cov, float((dd <= 3).mean()))
        med, cov, p3 = (best + (0.0,))[:3] if len(best) == 2 else best
        ok = med <= 3 and cov >= 0.9
        honest += ok
        near += (med <= 10 and cov >= 0.9)
        # p3 — доля строк в пределах 3px. Нужна как ЗАЩИТА ОТ АРТЕФАКТА СШИВКИ: med устойчива к
        # 30% чужих строк (§6.20.3), p3 — нет. med<=3 при низком p3 = сшили с чужой кривой.
        print(f"{well:<12}{g['name'].split()[0]:<8}{len(alltr):>6}{med:>10.1f}{cov:>7.2f}"
              f"{100*p3:>7.0f}  {'★ ЧЕСТНАЯ' if ok else ('~ <=10px' if med <= 10 and cov >= 0.9 else '')}")
print(f"\nкривых: {tot}")
print(f"★ НАБОРНАЯ ЧЕСТНОСТЬ (есть трасса med<=3px И cov>=0.9): {honest}/{tot} = {100*honest/max(1,tot):.0f}%")
print(f"  то же с порогом 10px: {near}/{tot} = {100*near/max(1,tot):.0f}%")
print("\nЧитать: это пайплайн БЕЗ решения задачи идентичности — столько кривых эксперту")
print("останется только ПОДПИСАТЬ. Сравнивать с прод-результатом 0/24 (§6.31).")
