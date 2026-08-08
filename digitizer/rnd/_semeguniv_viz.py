r"""_semeguniv_viz.py — ВИЗУАЛЬНЫЙ КОНТРОЛЬ результата на контрольном листе.

§4 (durable): «смотреть ВЕСЬ планшет, а не кропы» — метрики росли, а глазом дефект был виден
только на обзоре всей кривой. Здесь строится обзорная картинка: НАША трасса (сплошная) поверх
скана, рядом — экспертная (пунктиром), покривой цвета. Плюс полоса-врезка с увеличением.

НАШИ трассы берутся тем же путём, что в `_semeguniv_e2e.py` (оконный селектор, FLAG снят,
эскалация полосы заглушена) — то есть картинка показывает ровно то, что дало 7/7 честных.

  <ComfyUI>\python_embeded\python.exe _semeguniv_viz.py [--scale 4]
"""
import sys, io, argparse, contextlib
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np
import cv2
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from _multi_replica_probe import dense

SHEET = Path(r"F:\nds\projects\Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx")
OUT = Path(r"F:\nds\output\taskS\semeguniv")

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=int, default=4, help="во сколько раз уменьшить обзор")
ap.add_argument("--traces", default=None, help="json с нашими трассами (иначе прогон пайплайна)")
ap.add_argument("--seq", default="", help="чекпойнт селектора; пусто = ЖАДНЫЙ trace2d (§6.106)")
a = ap.parse_args()

PAL = [(230, 60, 60), (60, 190, 60), (60, 120, 240), (240, 170, 40),
       (200, 60, 200), (40, 200, 200), (150, 90, 40)]

# ── наши трассы: тот же путь, что дал 7/7 (см. _semeguniv_e2e.py) ──
from auto import confidence as confidence_mod, emit as emit_mod, refine as refine_mod, trace2d as T
_classify = confidence_mod.classify


def classify_all_auto(sheet, *args, **kw):
    r = _classify(sheet, *args, **kw)
    for L in sheet.lines:
        L.confidence = "AUTO"
    return r


confidence_mod.classify = classify_all_auto
_despike, _drop = refine_mod.despike, refine_mod.drop_transits


def refine_no_escalate(fg, line, frame, p, trace_line_fn, track):
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
TR = {}
_map = emit_mod._map_lines_to_slots


def cap(traces, model, frame, mnem, *rest):         # *rest: §6.105, пятый аргумент `cv`
    TR["all"] = list(traces)
    return _map(traces, model, frame, mnem, *rest)


emit_mod._map_lines_to_slots = cap

from auto.pipeline import run as pipe_run
from auto.config import Config

img_path = find_image(SHEET)
cfg = Config(); cfg.out = OUT / "viz"
cfg.cv.seq_model = a.seq        # §6.106: путь трассировки задаёт стенд, а не наличие torch
with contextlib.redirect_stdout(io.StringIO()):
    pipe_run(str(img_path), frame_nlgx=str(SHEET), cfg=cfg, stages=False)
alltr = TR.get("all", [])

m = extract(str(SHEET))
gts = {c["name"]: c for c in ds.real_curves(m) if any(x != NULL for x in c["xs"])}
GM = {}
for nm, g in gts.items():
    d = dense(g)
    if len(d) >= 50:
        GM[nm] = {y: d[y] for y in sorted(d)}

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
for med, cov, nm, ti in pairs:                      # один-к-одному
    if nm in got or ti in used:
        continue
    got[nm] = (med, cov, ti); used.add(ti)

rgb = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
s = a.scale
vis = cv2.resize(rgb, (W // s, H // s), interpolation=cv2.INTER_AREA)
vis = (vis.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)   # осветлить фон

print(f"{'кривая':<8}{'med':>7}{'cov':>7}  цвет")
for i, (nm, (med, cov, ti)) in enumerate(sorted(got.items(), key=lambda q: q[0])):
    col = PAL[i % len(PAL)]
    tr = alltr[ti][1]
    ys = np.array(sorted(tr)); xs = np.array([tr[y] for y in ys])
    pts = np.stack([xs / s, ys / s], 1).astype(np.int32)
    cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)             # НАША — сплошная
    gm = GM[nm]
    gy = np.array(sorted(gm)); gx = np.array([gm[y] for y in gy])
    step = max(1, len(gy) // 400)
    for k in range(0, len(gy) - step, 2 * step):                      # эксперт — пунктир
        p1 = (int(gx[k] / s), int(gy[k] / s)); p2 = (int(gx[k + step] / s), int(gy[k + step] / s))
        cv2.line(vis, p1, p2, col, 2, cv2.LINE_AA)
    cv2.putText(vis, f"{nm.split()[0]} {med:.1f}px", (8, 22 + 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    print(f"{nm.split()[0]:<8}{med:>7.1f}{cov:>7.2f}  RGB{col}")

cv2.putText(vis, "сплошная = НАША трасса, пунктир = эксперт", (8, vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
OUT.mkdir(parents=True, exist_ok=True)
p_all = OUT / "overview.png"
cv2.imwrite(str(p_all), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"\nобзор: {p_all}  ({vis.shape[1]}×{vis.shape[0]})")

# врезка: участок 1500 строк в натуральном масштабе, где кривые гуще всего
y0 = H // 2 - 750; y1 = y0 + 1500
crop = rgb[y0:y1].copy()
crop = (crop.astype(np.float32) * 0.5 + 255 * 0.5).astype(np.uint8)
for i, (nm, (med, cov, ti)) in enumerate(sorted(got.items(), key=lambda q: q[0])):
    col = PAL[i % len(PAL)]
    tr = alltr[ti][1]
    pts = [(int(tr[y]), int(y - y0)) for y in sorted(tr) if y0 <= y < y1]
    if len(pts) > 1:
        cv2.polylines(crop, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
    gm = GM[nm]
    for y in sorted(gm):
        if y0 <= y < y1 and (y - y0) % 12 < 5:
            cv2.circle(crop, (int(gm[y]), int(y - y0)), 2, col, -1)
p_crop = OUT / "crop_mid.png"
cv2.imwrite(str(p_crop), cv2.cvtColor(cv2.resize(crop, (crop.shape[1] // 2, crop.shape[0] // 2)),
                                      cv2.COLOR_RGB2BGR))
print(f"врезка (середина, 1500 строк): {p_crop}")
