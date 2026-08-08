r"""_decoder_seq_data.py — ВЫБОРКА ДЛЯ ОКОННОГО СЕЛЕКТОРА (§6.24 → контекст последовательности).

ЧТО МЕНЯЕТСЯ ПРОТИВ `_decoder_data.py`. Там решение на строке описывалось 10 числами ТОЙ ЖЕ
строки, и §6.24 упёрся ровно в это: линейно-poly модель на одной строке не держит идентичность
на широкой полосе. Здесь к каждому решению добавляется ОКНО: патч маски чернил
±R_ROWS строк × ±R_COLS px вокруг предсказания. Кандидат выбирается не «кто ближе», а «через
чью колонку идёт согласованный ход на десятках строк вперёд/назад».

★ ЧЕСТНОСТЬ (то же, что в §6.24): держанные скважины (кэш bench) в обучение НЕ входят; признаки
считает та же `_decoder_core.features`; патч строит ОДНА функция `patch()` — ею же пользуется
инференс в `_decoder_seq.py`, иначе train/test-скью.

★ ДРЕЙФ обязателен (урок итерации 1, §6.24): на on-curve состоянии ближайший ран прав в 99%,
учиться нечему. Латч случается со СНЕСЁННОГО состояния. Калибровка σ — по печатаемой доле
«база (ближайший) берёт чужой ран»: в §6.24 рабочая точка была ≈20%.

  python _decoder_seq_data.py --sheets 25 --drift 20 --out seq_train.npz
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
from auto import imaging as im, trace2d as T
from auto.config import DEFAULT
from _decoder_core import features, NF
from _decoder_data import train_sheets, HELD, SLMAX

OUT = Path(r"F:\nds\output\taskS\decoder")

# ── ГЕОМЕТРИЯ ОКНА (общая для обучения и инференса; менять только вместе с моделью) ──
R_ROWS, ROW_STEP = 64, 4          # ±64 строки, каждая 4-я  → 33 строки
R_COLS, COL_STEP = 96, 2          # ±96 px, каждый 2-й      → 97 колонок
NROW = 2 * R_ROWS // ROW_STEP + 1
NCOL = 2 * R_COLS // COL_STEP + 1
MAXC = 6                          # кандидатов на решение (= n_pick в features())
PBYTES = (2 * NROW * NCOL + 7) // 8


def patch(band, lo, y, pred):
    """Окно вокруг (y, pred) из БАНДОВОГО растра чернил `band` (H × (hi-lo), bool; колонка 0 = x=lo).

    Возвращает (ink, valid) размера NROW×NCOL. valid=0 там, где окно вышло за лист по строкам или
    за полосу по колонкам — модель обязана отличать «чисто» от «не знаем».
    ЭТА ЖЕ функция вызывается на инференсе (растр там собирается из CSR-кэша bench)."""
    H, Wb = band.shape
    ys = y + np.arange(-R_ROWS, R_ROWS + 1, ROW_STEP)
    xs = int(round(pred)) - lo + np.arange(-R_COLS, R_COLS + 1, COL_STEP)
    vy = (ys >= 0) & (ys < H)
    vx = (xs >= 0) & (xs < Wb)
    ink = band[np.ix_(np.clip(ys, 0, H - 1), np.clip(xs, 0, Wb - 1))].astype(bool)
    val = vy[:, None] & vx[None, :]
    return ink & val, val


def pack(ink, val):
    return np.packbits(np.concatenate([ink.ravel(), val.ravel()]))


def unpack(b):
    f = np.unpackbits(b, count=2 * NROW * NCOL).astype(np.float32)
    return f.reshape(2, NROW, NCOL)


def extract_sheet(n, pad, p, drift, rng, row_step, cap=0):
    """Решения одного листа. Структура прохода — как в `_decoder_data.extract_sheet`
    (teacher-forcing по экспертной трассе + инъекция дрейфа), добавлено сохранение окна."""
    rgb = cv2.cvtColor(cv2.imread(str(find_image(n)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    if len(gts) < 2:
        return None                                    # для латч-обучения нужны соседи
    colors = ["black", "red", "orange", "green", "blue"]
    fgs = {c: T._color_fg(rgb, c, p) for c in colors}
    P, F, L, M, D = [], [], [], [], []
    near_ok = near_n = 0

    for g in gts:
        d = dense(g)
        if len(d) < 100:
            continue
        rows = np.array(sorted(d)); xsr = np.array([d[y] for y in rows])
        base = float(np.median(xsr))
        lo = max(0, int(xsr.min()) - pad); hi = min(W, int(xsr.max()) + pad + 1)
        best_c, best_hit = "black", -1
        for c in colors:                               # лучший цвет: где под GT больше туши
            fg = fgs[c]
            hit = sum(1 for k in range(0, len(rows), 20)
                      if fg[rows[k], max(0, int(xsr[k]) - 3):int(xsr[k]) + 4].any())
            if hit > best_hit:
                best_hit, best_c = hit, c
        band = np.ascontiguousarray(fgs[best_c][:, lo:hi] > 0)

        allr = np.arange(rows.min(), rows.max() + 1)
        allx = np.interp(allr, rows, xsr)
        v = 0.0; prevx = None; dr = 0.0
        for j, y in enumerate(allr):
            gt_x = allx[j]
            if prevx is None:
                prevx = gt_x; v = 0.0; continue
            if drift:
                dr += rng.normal(0, drift * 0.4)
                if rng.random() < 0.02:                # перезахват: иногда дрейф сбрасывается
                    dr = 0.0
                dr = float(np.clip(dr, -1.6 * drift, 1.6 * drift))
            xhat = prevx + dr
            pred = xhat + float(np.clip(v, -SLMAX, SLMAX))
            v_next = 0.6 * v + 0.4 * (gt_x - prevx)
            if j % row_step == 0:
                row = im.row_runs(band[y])
                if row:
                    A = np.array([r[0] + lo for r in row]); B = np.array([r[1] + lo for r in row])
                    C = np.array([r[2] + lo for r in row], float)
                    idx, X = features(A, B, C, pred, xhat, v, base, MAXC)
                    if len(idx):
                        a = A[idx].astype(float); b = B[idx].astype(float)
                        lab = ((a - 3 <= gt_x) & (gt_x <= b + 3)).astype(np.uint8)
                        if lab.sum():                  # своей туши на строке нет — не учим
                            ink, val = patch(band, lo, y, pred)
                            xf = np.zeros((MAXC, NF), np.float32); xf[:len(idx)] = X
                            lf = np.zeros(MAXC, np.uint8); lf[:len(idx)] = lab
                            mf = np.zeros(MAXC, np.uint8); mf[:len(idx)] = 1
                            # ТАРГЕТ ТОЧКИ ВНУТРИ РАНА (§6.26): смещение эксперта от центра рана,
                            # нормированное полушириной. Осмыслен только там, где ран накрывает GT.
                            df = np.zeros(MAXC, np.float32)
                            hw = np.maximum((b - a) / 2, 1.0)
                            df[:len(idx)] = np.clip((gt_x - C[idx]) / hw, -1, 1)
                            P.append(pack(ink, val)); F.append(xf); L.append(lf); M.append(mf)
                            D.append(df)
                            k = int(np.argmin(np.abs(C[idx] - pred)))
                            near_ok += int(lab[k]); near_n += 1
            v = v_next; prevx = gt_x
    if not P:
        return None
    P = np.array(P, np.uint8); F = np.array(F, np.float32)
    L = np.array(L, np.uint8); M = np.array(M, np.uint8); D = np.array(D, np.float32)
    if cap and len(P) > cap:                           # лист не должен доминировать в выборке
        s = rng.choice(len(P), cap, replace=False)
        P, F, L, M, D = P[s], F[s], L[s], M[s], D[s]
    return (P, F, L, M, D, near_ok, near_n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=int, default=25)
    ap.add_argument("--pad", type=int, default=28)     # = pad20 + BAND_PAD8 стенда bench
    ap.add_argument("--drift", type=float, default=20.0)
    ap.add_argument("--row-step", type=int, default=4, help="брать каждое N-е решение (объём патчей)")
    ap.add_argument("--cap", type=int, default=15000, help="максимум решений с листа (0 = без ограничения)")
    ap.add_argument("--out", default="seq_train.npz")
    a = ap.parse_args()
    p = DEFAULT.cv
    rng = np.random.default_rng(12345)
    sheets = train_sheets(a.sheets)
    print(f"обучающих листов: {len(sheets)} (held-out {sorted(HELD)} исключены), дрейф σ={a.drift}")
    print(f"окно: {NROW}×{NCOL} (±{R_ROWS} строк /{ROW_STEP}, ±{R_COLS}px /{COL_STEP}), {PBYTES} байт/решение")
    PA, FA, LA, MA, DA, WA = [], [], [], [], [], []
    ok = nn = 0
    SKIP, done = {}, 0          # §6.106: сверка списка — обязательная печать, а не отладка
    for n in sheets:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                r = extract_sheet(n, a.pad, p, a.drift, rng, a.row_step, a.cap)
        except Exception as e:
            SKIP["падение"] = SKIP.get("падение", 0) + 1
            print(f"  !! {n.name[:44]:<46} {type(e).__name__}: {e}"); continue
        if r is None:
            SKIP["нет данных"] = SKIP.get("нет данных", 0) + 1
            print(f"  -- {n.name[:44]:<46} нет данных"); continue
        P, F, L, M, D, k, m = r
        PA.append(P); FA.append(F); LA.append(L); MA.append(M); DA.append(D)
        WA += [n.parent.parent.name] * len(P)
        ok += k; nn += m; done += 1
        print(f"  {n.name[:44]:<46} {len(P):>7} решений, ближайший прав {100*k/max(1,m):.1f}%")
    # ⚠⚠ СВЕРКА СПИСКА (§6.106, образец `_pool_oracle.py`): выборка, собранная по НЕПОЛНОМУ
    # списку листов, выглядит совершенно так же, как полная, — только модель учится не на том.
    _sk = sum(SKIP.values())
    print(f"  СВЕРКА СПИСКА: обработано {done} + пропущено {_sk} = {done + _sk} против длины "
          f"списка {len(sheets)}   {'★ СОШЛОСЬ' if done + _sk == len(sheets) else '⛔ НЕ СОШЛОСЬ'}")
    for k_, v_ in sorted(SKIP.items(), key=lambda q: -q[1]):
        print(f"    пропущено «{k_}»: {v_}")
    P = np.vstack(PA); F = np.vstack(FA); L = np.vstack(LA); M = np.vstack(MA); D = np.vstack(DA)
    wells = np.array(WA)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / a.out, P=P, F=F, L=L, M=M, D=D, wells=wells,
             geom=np.array([R_ROWS, ROW_STEP, R_COLS, COL_STEP, NROW, NCOL, MAXC]))
    print(f"\nИТОГО: {len(P)} решений, {P.nbytes/1e6:.0f} МБ патчей, скважин {len(set(WA))}")
    print(f"  БАЗА (ближайший) берёт СВОЙ ран: {100*ok/max(1,nn):.1f}%  "
          f"→ ошибается {100-100*ok/max(1,nn):.1f}% (цель ≈20%, §6.24 итер.2)")
    print(f"  -> {OUT / a.out}")
