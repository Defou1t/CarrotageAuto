r"""_seq_onpolicy.py — РЕШЕНИЯ СЕЛЕКТОРА НА ЕГО СОБСТВЕННЫХ ТРАЕКТОРИЯХ (DAgger-сбор, §6.221, 25.09).

ЗАЧЕМ. §6.220: в 4.5 раза больше данных дали лишь +0.6 п.п. точности решений (93.24 → 93.84%, на трудных 77.8 → 79.5%).
Выборка прод-селектора строится ведением ПО ЭТАЛОНУ с искусственным дрейфом (`extract_sheet`): предсказание — эталон плюс
случайный сдвиг, то есть почти всегда в пустом месте. А ломается селектор (§6.219: в 72% прыжков своя линия продолжалась)
в другом состоянии: трасса уже СТОИТ на чужой нарисованной линии, со своей скоростью. Таких состояний в выборке почти нет —
классический сдвиг распределения при обучении подражанием.

ЧТО. Ведение идёт так же, как в проде (`auto/trace_seq.trace_line`: раны строки → `features` → окно `patch` → скор сети →
центр выбранного рана → v = 0.6 v + 0.4 Δx; пустая строка — коаст), но решение принимает МОДЕЛЬ `--drive`, а метка каждой
строки — ран, накрывающий эталон ±3 px. Пишется каждое `--every`-е решение, где кандидатов > 1 и верный среди них есть.
Если трасса дальше `--reset-px` от эталона `--reset-rows` строк подряд — сброс на эталон (иначе лист тратится на безнадёжное).
Полоса и цвет — как в `extract_sheet` (полоса эталона ± pad, лучший цвет). Формат — как у обучающей выборки (P, F, L, M, D,
wells) ⇒ годится и для `_seq_eval.py` (точность на СВОИХ состояниях), и для дообучения. Модель на CPU (GPU занят кэшем).

  _seq_onpolicy.py --drive seq_model_big.pt --only-list wellmap_sheets.txt --sheets 60 --tag onpol_field --shard 0/2
"""
import sys, io, argparse, contextlib, time
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"F:\nds\Auto")
sys.path.insert(0, r"F:\nds\Auto\digitizer")
sys.path.insert(0, r"F:\nds\Auto\digitizer\rnd")
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ts", default=r"F:/nds/output/taskS")
ap.add_argument("--drive", required=True, help="чекпойнт, который ВЕДЁТ траекторию (имя в decoder/ или путь)")
ap.add_argument("--only-list", default="", help="листы только из этого списка (иначе — вне скважин поля и сорта A)")
ap.add_argument("--sheets", type=int, default=60)
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--reset-px", type=float, default=150.0)
ap.add_argument("--reset-rows", type=int, default=200)
ap.add_argument("--pad", type=int, default=28)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--tag", default="onpol")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--max-hours", type=float, default=0.0)
ap.add_argument("--no-save", action="store_true", help="только счёт, без записи выборки")
a = ap.parse_args()
TS = Path(a.ts)
import torch
torch.set_num_threads(a.threads)
import cv2
from extract_nlgx import extract, NULL
import dataset as ds
from dataset_build import find_image
from _multi_replica_probe import dense
from auto import imaging as im, trace2d as T
from auto.config import DEFAULT
from _decoder_core import features
from _decoder_seq_data import patch, pack, OUT, R_ROWS, ROW_STEP, R_COLS, COL_STEP, NROW, NCOL, MAXC
from _decoder_seq import WindowSelector
SLMAX = 30.0
NFEAT = 10

ck = Path(a.drive) if Path(a.drive).parent != Path(".") else OUT / a.drive
net = WindowSelector()
net.load_state_dict(torch.load(ck, map_location="cpu")["sd"]); net.eval()

# ── листы (та же логика, что `_seq_data_big.py`) ──
_CROSS = ("BKZ", "MK", "MGZ", "STK", ", ")
src = {}
for q in Path(r"F:\nds\projects\Archive").glob("*/wlg/*.nlgx"):
    src.setdefault(q.name, q)
per = {}
if a.only_list:
    for nm in [l.strip() for l in (TS / a.only_list).read_text(encoding="utf-8").splitlines() if l.strip()]:
        if nm in src:
            per.setdefault(src[nm].parent.parent.name, []).append(src[nm])
else:
    lst = lambda f: [l.strip() for l in (TS / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    EXCL = {src[n].parent.parent.name for n in lst("wellmap_sheets.txt") + lst("holdoutA_sheets.txt") if n in src}
    EXCL |= {"BOGAT_011", "BOGAT_015", "LEVEN_023", "BEZLUD_051", "YULIIV_055"}
    for q in sorted(src.values(), key=lambda q: (0 if any(t in q.name for t in _CROSS) else 1, q.name)):
        if q.parent.parent.name not in EXCL:
            per.setdefault(q.parent.parent.name, []).append(q)
order, i = [], 0
while len(order) < a.sheets:
    added = False
    for w in sorted(per):
        if i < len(per[w]):
            order.append(per[w][i]); added = True
            if len(order) >= a.sheets:
                break
    if not added:
        break
    i += 1
SH_I, SH_N = (int(v) for v in a.shard.split("/"))
mine = order[SH_I::SH_N]
print(f"★ СБОР НА СВОИХ ТРАЕКТОРИЯХ: ведёт {ck.name}; листов {len(mine)} (шард {SH_I}/{SH_N}), каждое {a.every}-е решение, "
      f"сброс {a.reset_px:.0f} px × {a.reset_rows} строк")


def score(band, lo, y, pred, X, n):
    ink, val = patch(band, lo, y, pred)
    p = torch.from_numpy(np.stack([ink, val]).astype(np.float32))[None]
    f = np.zeros((1, MAXC, NFEAT), np.float32); f[0, :n] = X
    m = np.zeros((1, MAXC), np.float32); m[0, :n] = 1
    with torch.no_grad():
        return net(p, torch.from_numpy(f), torch.from_numpy(m))[0].numpy()


def sheet(n, rng):
    rgb = cv2.cvtColor(cv2.imread(str(find_image(n)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    if len(gts) < 2:
        return None
    p = DEFAULT.cv
    colors = ["black", "red", "orange", "green", "blue"]
    fgs = {c: T._color_fg(rgb, c, p) for c in colors}
    P, F, L, M, D = [], [], [], [], []
    st = dict(dec=0, ok=0, near_ok=0, hard=0, hard_ok=0, resets=0, rows=0, on=0, don=0, don_ok=0, doff=0, doff_ok=0,
              curves=0, hon=0, hon90=0)
    for g in gts:
        d = dense(g)
        if len(d) < 100:
            continue
        rows = np.array(sorted(d)); xsr = np.array([d[y] for y in rows])
        base = float(np.median(xsr))
        lo = max(0, int(xsr.min()) - a.pad); hi = min(W, int(xsr.max()) + a.pad + 1)
        best_c, best_hit = "black", -1
        for c in colors:
            fg = fgs[c]
            hit = sum(1 for k in range(0, len(rows), 20) if fg[rows[k], max(0, int(xsr[k]) - 3):int(xsr[k]) + 4].any())
            if hit > best_hit:
                best_hit, best_c = hit, c
        band = np.ascontiguousarray(fgs[best_c][:, lo:hi] > 0)
        allr = np.arange(rows.min(), rows.max() + 1)
        allx = np.interp(allr, rows, xsr)
        x = float(allx[0]); v = 0.0; far = 0; cnt = 0; errs = []
        for j in range(1, len(allr)):
            y = int(allr[j]); gt_x = allx[j]
            st["rows"] += 1
            runs = im.row_runs(band[y])
            if not runs:
                x = x + float(np.clip(v, -SLMAX, SLMAX))
            else:
                A = np.array([r[0] + lo for r in runs]); B = np.array([r[1] + lo for r in runs])
                C = np.array([r[2] + lo for r in runs], float)
                pred = x + float(np.clip(v, -SLMAX, SLMAX))
                idx, X = features(A, B, C, pred, x, v, base, MAXC)
                aa = A[idx].astype(float); bb = B[idx].astype(float)
                lab = ((aa - 3 <= gt_x) & (gt_x <= bb + 3)).astype(np.uint8)
                if len(idx) == 1:
                    kk = 0
                else:
                    sc = score(band, lo, y, pred, X, len(idx))
                    kk = int(np.argmax(sc[:len(idx)]))
                    if lab.sum():
                        cnt += 1
                        near = int(np.argmin(np.abs(C[idx] - pred)))
                        st["dec"] += 1; st["ok"] += int(lab[kk]); st["near_ok"] += int(lab[near])
                        if not lab[near]:
                            st["hard"] += 1; st["hard_ok"] += int(lab[kk])
                        if abs(x - gt_x) <= 3:          # УДЕРЖАНИЕ: трасса на своей линии — не бросит ли
                            st["don"] += 1; st["don_ok"] += int(lab[kk])
                        else:                           # ВОЗВРАТ: трасса не на своей — вернётся ли
                            st["doff"] += 1; st["doff_ok"] += int(lab[kk])
                        if cnt % a.every == 0:
                            ink, val = patch(band, lo, y, pred)
                            xf = np.zeros((MAXC, NFEAT), np.float32); xf[:len(idx)] = X
                            lf = np.zeros(MAXC, np.uint8); lf[:len(idx)] = lab
                            mf = np.zeros(MAXC, np.uint8); mf[:len(idx)] = 1
                            df = np.zeros(MAXC, np.float32)
                            hw = np.maximum((bb - aa) / 2, 1.0)
                            df[:len(idx)] = np.clip((gt_x - C[idx]) / hw, -1, 1)
                            P.append(pack(ink, val)); F.append(xf); L.append(lf); M.append(mf); D.append(df)
                nx = float(C[idx[kk]])
                v = 0.6 * v + 0.4 * (nx - x); x = nx
            st["on"] += int(abs(x - gt_x) <= 3)
            errs.append(abs(x - gt_x))
            far = far + 1 if abs(x - gt_x) > a.reset_px else 0
            if far >= a.reset_rows:
                x = float(gt_x); v = 0.0; far = 0; st["resets"] += 1
        # ★ ПОКРИВОЙ СЧЁТ как в метрике: медиана |Δx| ≤ 3 px (строки кривой, ведение с её начала)
        if errs:
            st["curves"] += 1; e = np.array(errs)
            st["hon"] += int(np.median(e) <= 3.0); st["hon90"] += int((e <= 3.0).mean() >= 0.9)
    if not P:
        return None, st
    P = np.array(P, np.uint8); F = np.array(F, np.float32); L = np.array(L, np.uint8)
    M = np.array(M, np.uint8); D = np.array(D, np.float32)
    if a.cap and len(P) > a.cap:
        s = rng.choice(len(P), a.cap, replace=False)
        P, F, L, M, D = P[s], F[s], L[s], M[s], D[s]
    return (P, F, L, M, D), st


rng = np.random.default_rng(777 + SH_I)
PA, FA, LA, MA, DA, WA = [], [], [], [], [], []
TOT = dict(dec=0, ok=0, near_ok=0, hard=0, hard_ok=0, resets=0, rows=0, on=0, don=0, don_ok=0, doff=0, doff_ok=0,
           curves=0, hon=0, hon90=0)
T0 = time.time(); done = 0
for n in mine:
    if find_image(n) is None:
        continue
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = sheet(n, rng)
    except Exception as e:
        print(f"  !! {n.name[:44]:<46} {type(e).__name__}: {e}"); continue
    if r is None:
        continue
    data, st = r
    for k in TOT:
        TOT[k] += st[k]
    if data is not None:
        P, F, L, M, D = data
        PA.append(P); FA.append(F); LA.append(L); MA.append(M); DA.append(D)
        WA += [n.parent.parent.name] * len(P)
    done += 1
    print(f"  {n.name[:44]:<46} решений {st['dec']:>6}: модель {100*st['ok']/max(1,st['dec']):5.1f}%, ближайший "
          f"{100*st['near_ok']/max(1,st['dec']):5.1f}%, на трудных {100*st['hard_ok']/max(1,st['hard']):5.1f}%; "
          f"на эталоне {100*st['on']/max(1,st['rows']):5.1f}% строк, сбросов {st['resets']}  [{time.time()-T0:.0f} с]")
    if a.max_hours and (time.time() - T0) / 3600 > a.max_hours:
        print("★ ПАРТИЯ ОКОНЧЕНА — сохраняю собранное"); break
dst = OUT / f"{a.tag}_{SH_I}of{SH_N}.npz"
if PA and not a.no_save:
    np.savez(dst, P=np.vstack(PA), F=np.vstack(FA), L=np.vstack(LA), M=np.vstack(MA), D=np.vstack(DA),
             wells=np.array(WA), geom=np.array([R_ROWS, ROW_STEP, R_COLS, COL_STEP, NROW, NCOL, MAXC]))
print(f"ИТОГО листов {done}: решений {TOT['dec']}, модель верна {100*TOT['ok']/max(1,TOT['dec']):.2f}%, ближайший "
      f"{100*TOT['near_ok']/max(1,TOT['dec']):.2f}%, трудных {TOT['hard']} ({100*TOT['hard']/max(1,TOT['dec']):.1f}%), "
      f"на трудных {100*TOT['hard_ok']/max(1,TOT['hard']):.2f}%; трасса на эталоне {100*TOT['on']/max(1,TOT['rows']):.1f}% строк, "
      f"сбросов {TOT['resets']} → {dst}")
print(f"★ УДЕРЖАНИЕ (трасса на своей линии, решений {TOT['don']}): верно {100*TOT['don_ok']/max(1,TOT['don']):.2f}% ⇒ бросает линию в "
      f"{100-100*TOT['don_ok']/max(1,TOT['don']):.2f}% решений; ВОЗВРАТ (не на своей, свой ран среди кандидатов, решений {TOT['doff']}): "
      f"верно {100*TOT['doff_ok']/max(1,TOT['doff']):.2f}%")
print(f"★ КРИВЫХ {TOT['curves']}: медиана |Δx| ≤ 3 px у {TOT['hon']}, ≥ 90% строк в 3 px у {TOT['hon90']}")
