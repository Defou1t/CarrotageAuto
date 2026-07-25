r"""_joint_trace.py — СОВМЕСТНАЯ ТРАССИРОВКА СО ВЗАИМНЫМ ИСКЛЮЧЕНИЕМ («отличать линии между собой»).

Задача заказчика №1 (§6.33): различать линии. Замер §6.36 на контрольном листе показал механизм
провала: GZ41/OGZ1/GZ51 лежат в одном x-диапазоне, каждая линия трассируется НЕЗАВИСИМО — и все
три приходят на одну и ту же тушь (одна трасса засчиталась трём кривым).

Здесь линии одного цвета ведутся ОДНОВРЕМЕННО: на каждой строке раны распределяются между
активными трассами НАЗНАЧЕНИЕМ, а не «каждому ближайший».

★ ДОМЕННАЯ ОГОВОРКА: кривые РЕАЛЬНО пересекаются, и на пересечении физически лежат в одной туши.
Поэтому жёсткий запрет «две трассы в одном ране» неверен — он бы рвал трассы на каждом кроссинге.
Ёмкость рана считается по ЕГО ШИРИНЕ: узкий держит одну линию, слипшийся широкий — несколько
(cap = 1 + ширина // cap_px). Это прямое следствие §6.20.4 (в момент слияния свой ран ШИРЕ).

Порядок разбора внутри строки — жадный по возрастанию цены (Hungarian тут избыточен: ланов 2-6,
кандидатов 2-8, а жадность по отсортированным парам даёт то же назначение в подавляющем большинстве
строк и на порядок дешевле на 14 тыс. строк × 13 линий).

  <ComfyUI>\python_embeded\python.exe _joint_trace.py [--excl] [--seq] [--cap-px 20]
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
from auto import frame as F, understand as U, confidence as C, meta as M
from auto import imaging as im, trace2d as T
from auto.config import DEFAULT

SHEET = Path(r"F:\nds\projects\Semeguniv_001\wlg\Semeguniv_1_BKZ_3400_3640_200_D1.nlgx")
MN = r"F:\nds\Auto\mnemonics.json"
SLMAX = 30.0

ap = argparse.ArgumentParser()
ap.add_argument("--excl", action="store_true", help="взаимное исключение (иначе — как сейчас, независимо)")
ap.add_argument("--cap-px", type=int, default=20, help="сколько px ширины рана даёт +1 ёмкости")
ap.add_argument("--seq", action="store_true", help="цена по оконному селектору вместо расстояния")
ap.add_argument("--ckpt", default="seq_model_d45p.pt")
ap.add_argument("--wide-run", type=int, default=10 ** 6,
                help="правило прода: ран шире этого → точка НЕ центр, а дальний от базлайна край "
                     "(дефолт прода 14; здесь по умолчанию выключено = всегда центр, §6.26)")
ap.add_argument("--despike", action="store_true", help="применить refine.despike, как в проде")
ap.add_argument("--escalate", action="store_true",
                help="эскалация полосы до ТРЕКА при недотяге (refine.py:292-299), как в проде")
ap.add_argument("--extend", action="store_true", help="_extend_ends, как в проде")
a = ap.parse_args()

NET = DEV = None
if a.seq:
    import torch
    from _decoder_core import features
    from _decoder_seq import WindowSelector, OUT as MODELS
    from _decoder_seq_data import MAXC, patch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    NET = WindowSelector().to(DEV)
    NET.load_state_dict(torch.load(MODELS / a.ckpt, map_location=DEV)["sd"]); NET.eval()


def joint_trace(fg, lines, excl, cap_px, wide_run=10 ** 6, escalate=False, track=None):
    """Все линии ОДНОГО цвета одновременно. Возвращает {индекс линии: {row: x}}."""
    H, W = fg.shape
    lanes = []
    for i, L in enumerate(lines):
        lo_, hi_ = max(0, int(L.x_lo) - 8), min(W, int(L.x_hi) + 8 + 1)
        if escalate and track is not None:      # прод-ветка «недотяг → полоса во весь трек»
            lo_, hi_ = max(0, track.x_left), min(W, track.x_right)
        lanes.append({"i": i, "lo": lo_, "hi": hi_,
                      "base": L.x_center, "y0": max(0, L.y0), "y1": min(H - 1, L.y1),
                      "x": None, "v": 0.0, "tr": {}, "A": None})
    y0 = min(l["y0"] for l in lanes); y1 = max(l["y1"] for l in lanes)
    for y in range(y0, y1 + 1):
        act = [l for l in lanes if l["y0"] <= y <= l["y1"]]
        if not act:
            continue
        runs = im.row_runs(fg[y])
        if not runs:
            for l in act:
                if l["x"] is not None:
                    l["x"] += float(np.clip(l["v"], -SLMAX, SLMAX))
            continue
        A = np.array([r[0] for r in runs]); B = np.array([r[1] for r in runs])
        Cc = np.array([r[2] for r in runs], float)
        cap = [1 + (int(B[j]) - int(A[j])) // cap_px for j in range(len(runs))]
        # цена (лан, ран): расстояние от предсказания; ран вне полосы лана запрещён
        pairs = []
        for li, l in enumerate(act):
            if l["x"] is None:
                pred = l["base"]
            else:
                pred = l["x"] + float(np.clip(l["v"], -SLMAX, SLMAX))
            for j in range(len(runs)):
                if B[j] < l["lo"] or A[j] > l["hi"]:
                    continue
                gap = max(A[j] - pred, pred - B[j], 0.0)
                pairs.append((gap, li, j))
        pairs.sort()
        taken_l = set(); used = [0] * len(runs)
        assign = {}
        for gap, li, j in pairs:
            if li in taken_l:
                continue
            if excl and used[j] >= cap[j]:
                continue
            assign[li] = j; taken_l.add(li); used[j] += 1
        for li, l in enumerate(act):
            j = assign.get(li)
            if j is None:                     # ничего не досталось — коаст по инерции
                if l["x"] is not None:
                    l["x"] += float(np.clip(l["v"], -SLMAX, SLMAX))
                continue
            aj, bj, cj = float(A[j]), float(B[j]), float(Cc[j])
            # правило прода: на ШИРОКОМ ране точка = дальний от базлайна КРАЙ, а не центр.
            # §6.26 замерил, что оно стоит десятки px; здесь оно включается флагом для абляции.
            nx = (bj if abs(bj - l["base"]) >= abs(aj - l["base"]) else aj) \
                if (bj - aj) >= wide_run else cj
            if l["x"] is None:
                l["x"] = nx; l["v"] = 0.0
            else:
                l["v"] = 0.6 * l["v"] + 0.4 * (nx - l["x"]); l["x"] = nx
            l["tr"][y] = nx
    return {l["i"]: l["tr"] for l in lanes}


p = DEFAULT.cv
rgb = cv2.cvtColor(cv2.imread(str(find_image(SHEET)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m0 = M.parse_filename(SHEET.name, MN)
    fr = F.frame_from_nlgx(str(SHEET), m0, p, rgb=rgb)
    if getattr(fr, "row_shift", None) is not None:
        rgb = F.apply_row_shift(rgb, fr.row_shift)
    sheet = U.understand(rgb, fr, m0, p)
    C.classify(sheet)
print(f"линий U1: {len(sheet.lines)}   взаимное исключение: {'ДА' if a.excl else 'нет'}"
      f"   ёмкость: +1 на {a.cap_px}px ширины")

alltr = []
for col in sorted({L.color for L in sheet.lines}):
    idxs = [i for i, L in enumerate(sheet.lines) if L.color == col]
    fg = T._color_fg(rgb, col, p)
    lines_c = [sheet.lines[i] for i in idxs]
    trk = sheet.frame.tracks[lines_c[0].track_index] if lines_c else None
    trs = joint_trace(fg, lines_c, a.excl, a.cap_px, a.wide_run, a.escalate, trk)
    for k, tr in trs.items():
        if a.extend:
            L = lines_c[k]
            T._extend_ends(tr, fg, max(0, int(L.x_lo) - 8), min(fg.shape[1], int(L.x_hi) + 9), SLMAX)
        if a.despike and len(tr) >= 30:
            from auto import refine as RF
            tr, _ = RF.despike(tr, win=p.despike_win, k=p.despike_k, min_jump=p.despike_min_jump)
        if len(tr) >= 30:
            alltr.append((sheet.lines[idxs[k]], tr))
print(f"трасс получено: {len(alltr)}   "
      f"[wide_run={a.wide_run if a.wide_run < 10**6 else 'выкл'} despike={a.despike} "
      f"escalate={a.escalate} extend={a.extend}]")

mo = extract(str(SHEET))
gts = {c["name"]: c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])}
GM = {}
for name, g in gts.items():
    d = dense(g)
    if len(d) >= 50:
        GM[name] = {y: d[y] for y in sorted(d)}

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

print(f"\n{'кривая':<8}{'med':>9}{'cov':>7}{'трасса':>8}  контроль утечки")
res = []
for nm in GM:
    if nm not in got:
        print(f"{nm.split()[0]:<8}   не назначено"); continue
    med, cov, ti = got[nm]
    tr = alltr[ti][1]
    g = gts[nm]
    er = np.array([g["top_y"] + i for i, x in enumerate(g["xs"]) if x != NULL])
    ex = np.array([x for x in g["xs"] if x != NULL], float)
    oy = np.array(sorted(tr)); ox = np.array([tr[y] for y in oy], float)
    bad = []
    if len(oy) == len(er) and np.array_equal(oy, er):
        bad.append("L2:сетка=эксперт")
        if np.allclose(ox, ex):
            bad.append("L1:побитово")
    common = np.intersect1d(oy, er)
    if len(common) >= 30:
        om = dict(zip(oy.tolist(), ox.tolist())); em = dict(zip(er.tolist(), ex.tolist()))
        s = np.mean([abs(om[y] - em[y]) < 1e-9 for y in common])
        if s > 0.5:
            bad.append(f"L3:{100*s:.0f}% точных совпадений")
    res.append((med, cov))
    print(f"{nm.split()[0]:<8}{med:>9.1f}{cov:>7.2f}{ti:>8}  {'✓ наша' if not bad else ' | '.join(bad)}")
if res:
    h = sum(1 for md, cv in res if md <= 3 and cv >= 0.9)
    print(f"\nmed(med) {np.median([r[0] for r in res]):.1f}px   "
          f"med(cov) {np.median([r[1] for r in res]):.2f}   ★ЧЕСТНЫХ {h}/{len(GM)}")
