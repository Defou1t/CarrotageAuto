r"""_decoder_data.py — ВЫБОРКА ДЛЯ ОБУЧАЕМОГО СЕЛЕКТОРА: teacher-forcing по экспертным трассам.

На каждой строке экспертной кривой известно, КУДА она пошла → правильный ран известен. Полоса
берётся ШИРОКОЙ (как в bench: весь размах кривой ± паддинг), чтобы в неё попадали РАНЫ СОСЕДЕЙ
на строках пересечения — именно на них модель учится не латчиться.

★ ДЕРЖАННЫЕ скважины (кэш bench: BOGAT_011/015, LEVEN_023, BEZLUD_051, YULIIV_055) в обучение
НЕ входят — тест честный. Обучаемся на прочем архиве.

  python _decoder_data.py [--sheets 25] [--pad 28]
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

ARCH = Path(r"F:\nds\projects\Archive")
OUT = Path(r"F:\nds\output\taskS\decoder")
HELD = {"BOGAT_011", "BOGAT_015", "LEVEN_023", "BEZLUD_051", "YULIIV_055"}  # тест = кэш bench
SLMAX = 30.0


_CROSS = ("BKZ", "MK", "MGZ", "STK", ", ")     # токены с пересекающимися кривыми (латч учится тут)


def _cross_first(sheets):
    """Кроссинг-листы вперёд: латч-обучение возможно только там, где ≥2 кривые делят полосу."""
    return sorted(sheets, key=lambda n: (0 if any(t in n.name for t in _CROSS) else 1, n.name))


def train_sheets(limit):
    """Мульти-кривые листы архива со скважин НЕ из held-out, round-robin по скважинам,
    кроссинг-классы (BKZ/MK/…) в приоритете — иначе выборку заполняют одиночные ZSK/AK."""
    per = {}
    for w in sorted(d for d in ARCH.iterdir() if d.is_dir() and d.name not in HELD):
        wl = w / "wlg"
        sheets = sorted(wl.glob("*.nlgx")) if wl.is_dir() else sorted(w.glob("*.nlgx"))
        good = [n for n in sheets if find_image(n) is not None]
        if good:
            per[w.name] = _cross_first(good)
    out, i = [], 0
    while len(out) < limit:
        added = False
        for w, sh in per.items():
            if i < len(sh):
                out.append(sh[i]); added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        i += 1
    return out


def extract_sheet(n, pad, p, drift=0.0, rng=None):
    rgb = cv2.cvtColor(cv2.imread(str(find_image(n)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    mo = extract(str(n))
    gts = [c for c in ds.real_curves(mo) if any(x != NULL for x in c["xs"])]
    if len(gts) < 2:
        return None                                   # для латч-обучения нужны соседи
    colors = ["black", "red", "orange", "green", "blue"]
    fgs = {c: T._color_fg(rgb, c, p) for c in colors}
    Xs, ys, grp = [], [], []
    gid = [0]

    def emit(A, B, C, pred, x, v, base, gt_x):
        idx, X = features(A, B, C, pred, x, v, base)
        if len(idx) == 0:
            return
        a = A[idx].astype(float); b = B[idx].astype(float)
        lab = ((a - 3 <= gt_x) & (gt_x <= b + 3)).astype(float)   # ран накрывает точку эксперта
        if lab.sum() == 0:
            return                                    # своей туши на строке нет — не учим
        Xs.append(X); ys.append(lab); grp.append(np.full(len(idx), gid[0])); gid[0] += 1

    for g in gts:
        d = dense(g)
        if len(d) < 100:
            continue
        rows = np.array(sorted(d)); xsr = np.array([d[y] for y in rows])
        base = float(np.median(xsr))
        lo = max(0, int(xsr.min()) - pad); hi = min(W, int(xsr.max()) + pad + 1)
        # лучший цвет: где под GT больше туши
        best_c, best_hit = "black", -1
        for c in colors:
            fg = fgs[c]
            hit = sum(1 for k in range(0, len(rows), 20)
                      if fg[rows[k], max(0, int(xsr[k]) - 3):int(xsr[k]) + 4].any())
            if hit > best_hit:
                best_hit, best_c = hit, c
        fg = fgs[best_c]
        v = 0.0
        gpos = {int(r): float(xv) for r, xv in zip(rows, xsr)}
        rr = range(int(rows.min()), int(rows.max()) + 1)
        prevx = None
        dd = {}
        # плотная интерполяция GT на все строки
        allr = np.arange(rows.min(), rows.max() + 1)
        allx = np.interp(allr, rows, xsr)
        gmap = dict(zip(allr.tolist(), allx.tolist()))
        dr = 0.0                                       # накопленный ДРЕЙФ состояния (латч-модель)
        for j, y in enumerate(allr):
            gt_x = allx[j]
            if prevx is None:
                prevx = gt_x; v = 0.0; continue
            # ДРИФТ (§6.20.2: латч случается со СНЕСЁННОГО состояния, не on-curve). Веримое
            # положение x_hat блуждает вокруг истины; pred считается ОТ него, но правильный ран
            # (label) — по ИСТИННОМУ gt_x. Модель учится восстанавливаться, а не брать ближайший.
            if drift and rng is not None:
                dr += rng.normal(0, drift * 0.4)
                if rng.random() < 0.02:                # перезахват: иногда дрейф сбрасывается
                    dr = 0.0
                dr = float(np.clip(dr, -1.6 * drift, 1.6 * drift))
            xhat = prevx + dr
            pred = xhat + float(np.clip(v, -SLMAX, SLMAX))
            row = im.row_runs(fg[y, lo:hi])
            if row:
                A = np.array([r[0] + lo for r in row]); B = np.array([r[1] + lo for r in row])
                C = np.array([r[2] + lo for r in row], float)
                emit(A, B, C, pred, xhat, v, base, gt_x)
            v = 0.6 * v + 0.4 * (gt_x - prevx); prevx = gt_x
    if not Xs:
        return None
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(grp), n.parent.parent.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=int, default=25)
    ap.add_argument("--pad", type=int, default=28)
    ap.add_argument("--drift", type=float, default=0.0, help="σ инъекции дрейфа состояния (px); 0=teacher-forcing")
    ap.add_argument("--out", default="train.npz")
    a = ap.parse_args()
    p = DEFAULT.cv
    rng = np.random.default_rng(12345) if a.drift else None
    sheets = train_sheets(a.sheets)
    print(f"обучающих листов: {len(sheets)} (held-out {sorted(HELD)} исключены), дрейф σ={a.drift}")
    XA, yA, gA, wells = [], [], [], []
    gbase = 0
    for n in sheets:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                r = extract_sheet(n, a.pad, p, drift=a.drift, rng=rng)
        except Exception as e:
            print(f"  !! {n.name[:44]:<46} {type(e).__name__}: {e}"); continue
        if r is None:
            print(f"  -- {n.name[:44]:<46} нет данных"); continue
        X, y, g, well = r
        g = g + gbase; gbase = g.max() + 1
        XA.append(X); yA.append(y); gA.append(g); wells += [well] * len(X)
        print(f"  {n.name[:44]:<46} {len(X):>7} канд, {int(y.sum()):>6} полож, реш {len(np.unique(g))}")
    X = np.vstack(XA); y = np.concatenate(yA); g = np.concatenate(gA)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / a.out, X=X, y=y, g=g)
    print(f"\nИТОГО: {len(X)} кандидатов, {int(y.sum())} положительных, {len(np.unique(g))} решений")
    print(f"  доля решений с ровно 1 положительным: "
          f"{np.mean([ (y[g==gi].sum()==1) for gi in np.unique(g)[:2000]]):.2f}")
    print(f"  -> {OUT / a.out}")
